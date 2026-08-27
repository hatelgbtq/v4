#!/usr/bin/env python3
"""
DEN Depth Forgetting Analysis.

Investigates catastrophic forgetting immediately after depth-growth insertion.
Analyzes activation distributions, weight drift, and tests freezing/warmup
strategies to mitigate forgetting.

Usage:
    # Run all experiments
    python experiments/depth_forgetting_analysis.py --experiment all

    # Run a specific experiment
    python experiments/depth_forgetting_analysis.py --experiment baseline
    python experiments/depth_forgetting_analysis.py --experiment freeze_inserted --freeze-iters 500
    python experiments/depth_forgetting_analysis.py --experiment freeze_neighbors
    python experiments/depth_forgetting_analysis.py --experiment warmup --warmup-iters 200
    python experiments/depth_forgetting_analysis.py --experiment activation_analysis
    python experiments/depth_forgetting_analysis.py --experiment identity_monitor
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.den import DEN
from models.grow_depth import insert_hidden_layer, should_insert_layer, get_architecture_summary
from models.layers import DynamicLinear
from models.utils import knowledge_preservation_loss, get_prev_weight_slices, accuracy as accuracy_fn
from models.prune import select_active_neurons, merge_sub_network
from datasets.permuted_mnist import get_permuted_mnist_loaders

# =========================================================================
#  Constants
# =========================================================================

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "forgetting_analysis"
N_TASKS = 4
MAX_ITER = 400
BATCH_SIZE = 256
LR = 0.001
INPUT_DIM = 784
HIDDEN_DIMS = [256, 128]
NUM_CLASSES = 10
DEPTH_GROWTH_INTERVAL = 1  # insert after every task
SEED = 1004

EX_K = 10
L1_LAMBDA = 1e-5
L2_LAMBDA = 1e-4
GL_LAMBDA = 0.001
REGULAR_LAMBDA = 0.5
LOSS_THR = 0.01
SPL_THR = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================
#  Baseline Runner
# =========================================================================

def make_default_cfg():
    return {
        "dataset": "permuted_mnist",
        "num_tasks": N_TASKS,
        "batch_size": BATCH_SIZE,
        "max_iter": MAX_ITER,
        "lr": LR,
        "input_dim": INPUT_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "num_classes": NUM_CLASSES,
        "ex_k": EX_K,
        "l1_lambda": L1_LAMBDA,
        "l2_lambda": L2_LAMBDA,
        "gl_lambda": GL_LAMBDA,
        "regular_lambda": REGULAR_LAMBDA,
        "loss_thr": LOSS_THR,
        "spl_thr": SPL_THR,
        "depth_growth_enabled": True,
        "depth_growth_interval": DEPTH_GROWTH_INTERVAL,
        "seed": SEED,
        "log_dir": str(RESULTS_DIR / "baseline"),
    }


def make_model(cfg=None):
    if cfg is None:
        cfg = make_default_cfg()
    return DEN(
        input_dim=cfg["input_dim"],
        hidden_dims=cfg["hidden_dims"],
        num_classes=cfg["num_classes"],
        ex_k=cfg.get("ex_k", EX_K),
        l1_lambda=cfg.get("l1_lambda", L1_LAMBDA),
        l2_lambda=cfg.get("l2_lambda", L2_LAMBDA),
        gl_lambda=cfg.get("gl_lambda", GL_LAMBDA),
        regular_lambda=cfg.get("regular_lambda", REGULAR_LAMBDA),
        loss_thr=cfg.get("loss_thr", LOSS_THR),
        spl_thr=cfg.get("spl_thr", SPL_THR),
        depth_growth_enabled=cfg.get("depth_growth_enabled", True),
        depth_growth_config={"interval": cfg.get("depth_growth_interval", 1)},
    )


def get_data(num_tasks=N_TASKS):
    train_loaders, val_loaders, test_loaders = get_permuted_mnist_loaders(
        num_tasks=num_tasks,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )
    return train_loaders, val_loaders, test_loaders


def evaluate_all(model, test_loaders, num_tasks):
    """Evaluate on all tasks, return list of accuracies."""
    model.eval()
    accs = []
    for tid in range(num_tasks):
        _, acc = model.evaluate_task(test_loaders[tid], tid)
        accs.append(acc)
    model.train()
    return accs


def forgetting(acc_matrix):
    """Compute forgetting for tasks BEFORE the current one."""
    num_t = len(acc_matrix)
    if num_t < 2:
        return []
    forget = []
    for j in range(num_t - 1):
        best = max(acc_matrix[k][j] for k in range(j, num_t))
        current = acc_matrix[-1][j]
        forget.append(best - current)
    return forget


def accuracy(preds, labels):
    return accuracy_fn(preds, labels)


# =========================================================================
#  Experimental DEN Subclass
# =========================================================================

class ExperimentalDEN(DEN):
    """DEN with hooks for depth-forgetting experiments.

    Supports:
        baseline:          Standard training (no modifications).
        freeze_inserted:   Freeze inserted layer for first N iterations.
        freeze_neighbors:  Freeze layer before and after insertion.
        warmup:            Linear LR ramp for inserted layer.
        activation_analysis: Capture activations pre/post insertion.
        identity_monitor:   Track weight drift of inserted layer.
    """

    def __init__(self, *args, experiment_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_cfg = experiment_config or {"type": "baseline"}
        self.inserted_layer_idx = None

        # Tracking
        self.weight_norm_log = []        # (task_id, iter, norm)
        self.neighbor_delta_log = []     # (task_id, iter, before_norm, after_norm)
        self.loss_log = []               # (task_id, iter, loss_val)
        self.activation_before = {}      # layer_idx -> list[mean, std]
        self.activation_after = {}       # layer_idx -> list[mean, std]
        self._global_iter = 0
        self._prev_params_snapshot = None
        self._neighbor_params_before = None

    def set_inserted_layer(self, idx):
        self.inserted_layer_idx = idx

    # ------------------------------------------------------------------
    #  Activation capture
    # ------------------------------------------------------------------

    def capture_activations(self, loader, task_id, label="before"):
        """Run forward pass and record mean/std per hidden layer."""
        self.eval()
        activations = defaultdict(list)
        hooks = []

        def _hook_fn(layer_idx):
            def fn(_, __, output):
                activations[layer_idx].append(output.detach().cpu())
            return fn

        for i, layer in enumerate(self.hidden_layers):
            hooks.append(layer.register_forward_hook(_hook_fn(i)))

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self._device)
                x = x.view(x.size(0), -1)
                if task_id in self.timestamps:
                    stamp = self.timestamps[task_id]
                    n_layers = len(stamp) - 1
                    h = x
                    for i in range(n_layers):
                        layer = self.hidden_layers[i]
                        h = F.relu(layer(h, out_slice=stamp[i],
                                        in_slice=stamp[i - 1] if i > 0 else h.size(-1)))
                else:
                    self(x, task_id=task_id)

        for h in hooks:
            h.remove()

        result = {}
        for idx, acts in activations.items():
            all_acts = torch.cat(acts, dim=0)
            result[idx] = (all_acts.mean().item(), all_acts.std().item())

        if label == "before":
            self.activation_before = result
        else:
            self.activation_after = result
        self.train()
        return result

    # ------------------------------------------------------------------
    #  Weight norm tracking
    # ------------------------------------------------------------------

    def _log_weight_norms(self):
        if self.inserted_layer_idx is None:
            return
        w = self.hidden_layers[self.inserted_layer_idx].weight.data
        n = min(w.size(0), w.size(1))
        I = torch.eye(n, device=w.device)
        norm = (w[:n, :n] - I).norm().item()
        self.weight_norm_log.append((self._global_iter, norm))

    def _log_neighbor_deltas(self):
        if self.inserted_layer_idx is None or self._neighbor_params_before is None:
            return
        idx = self.inserted_layer_idx
        delta = 0.0
        for ni in [idx - 1, idx + 1]:
            if 0 <= ni < len(self.hidden_layers):
                w = self.hidden_layers[ni].weight.data
                key = f"hidden_layers.{ni}.weight"
                if key in self._neighbor_params_before:
                    prev = self._neighbor_params_before[key].to(w.device)
                    delta += (w - prev).norm().item()
        self.neighbor_delta_log.append((self._global_iter, delta))

    # ------------------------------------------------------------------
    #  Experiment training loop
    # ------------------------------------------------------------------

    def _experiment_train_loop(
        self, task_id, train_loader, max_iter, lr,
        use_kp=True, use_group_lasso=False,
        record_metrics=True,
    ):
        """Generic training loop that applies experiment modifications."""
        exp_type = self.exp_cfg.get("type", "baseline")
        idx = self.inserted_layer_idx

        # Snapshot neighbor params for delta tracking
        if record_metrics:
            self._neighbor_params_before = {
                name: p.data.clone().cpu()
                for name, p in self.named_parameters()
            }

        # Build param groups
        param_groups = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            group = {"params": [param], "lr": lr, "name": name}
            if exp_type == "warmup" and idx is not None and f"hidden_layers.{idx}." in name:
                group["warmup"] = True
            else:
                group["warmup"] = False
            param_groups.append(group)

        optimizer = torch.optim.Adam(param_groups, lr=lr)
        prev_slices = self._get_prev_weight_slices() if use_kp else None

        warmup_iters = self.exp_cfg.get("warmup_iters", 200)
        freeze_iters = self.exp_cfg.get("freeze_iters", 500)
        base_lr = lr

        losses = []
        weight_norms = []

        for iteration in range(max_iter):
            # --- Apply experiment modifications ---
            if exp_type == "freeze_inserted" and idx is not None:
                frozen = iteration < freeze_iters
                for p in self.hidden_layers[idx].parameters():
                    p.requires_grad = not frozen

            elif exp_type == "freeze_neighbors" and idx is not None:
                if idx > 0:
                    for p in self.hidden_layers[idx - 1].parameters():
                        p.requires_grad = False
                if idx < len(self.hidden_layers) - 1:
                    for p in self.hidden_layers[idx + 1].parameters():
                        p.requires_grad = False

            elif exp_type == "warmup" and idx is not None:
                w_factor = min(iteration / max(warmup_iters, 1), 1.0)
                for group in optimizer.param_groups:
                    if group.get("warmup", False):
                        group["lr"] = base_lr * w_factor

            # --- Training step ---
            x, y = next(iter(train_loader))
            x, y = x.to(self._device), y.to(self._device)
            x = x.view(x.size(0), -1)

            optimizer.zero_grad()
            logits = self(x, task_id=task_id)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss += self.l2_lambda * 0.5 * sum(
                p.pow(2).sum() for p in self.parameters() if p.requires_grad
            )
            if use_kp:
                loss += knowledge_preservation_loss(self, self.regular_lambda, prev_slices)

            loss.backward()
            optimizer.step()
            self._apply_l1_soft_thresholding()
            if use_group_lasso:
                self._apply_group_lasso_step()

            loss_val = loss.item()
            losses.append(loss_val)
            if record_metrics:
                self.loss_log.append((task_id, self._global_iter, loss_val))

            # --- Log weight norms ---
            if record_metrics and idx is not None:
                self._log_weight_norms()
                self._log_neighbor_deltas()

            self._global_iter += 1

        # Restore requires_grad for neighbors if frozen
        if exp_type == "freeze_neighbors" and idx is not None:
            for p in self.parameters():
                p.requires_grad = True

        return losses

    # ------------------------------------------------------------------
    #  Override training methods for experiments
    # ------------------------------------------------------------------

    def _train_sub_network(
        self, task_id, active_indices, sub_weights,
        train_loader, val_loader, max_iter, lr,
    ):
        """Override: train sub-network with experiment modifications."""
        exp_type = self.exp_cfg.get("type", "baseline")
        idx = self.inserted_layer_idx

        shared_weights = self._get_shared_weights()
        head_w = self.output_heads[f"task_{task_id}"].weight.data.clone()
        head_b = self.output_heads[f"task_{task_id}"].bias.data.clone()

        sub_net = self._build_sub_network(sub_weights, task_id).to(self._device)

        # Freeze inserted layer in sub-network for freeze experiments
        if exp_type in ("freeze_inserted", "freeze_neighbors") and idx is not None:
            if idx < len(sub_net.hidden_layers):
                for p in sub_net.hidden_layers[idx].parameters():
                    p.requires_grad = False

        # Freeze neighbors in sub-network
        if exp_type == "freeze_neighbors" and idx is not None:
            for ni in [idx - 1, idx + 1]:
                if 0 <= ni < len(sub_net.hidden_layers):
                    for p in sub_net.hidden_layers[ni].parameters():
                        p.requires_grad = False

        sub_optimizer = torch.optim.Adam(
            [p for p in sub_net.parameters() if p.requires_grad], lr=lr
        )

        warmup_iters = self.exp_cfg.get("warmup_iters", 200)
        base_lr = lr

        for iteration in range(max_iter):
            # Warmup for inserted layer in sub-network
            if exp_type == "warmup" and idx is not None and idx < len(sub_net.hidden_layers):
                w_factor = min(iteration / max(warmup_iters, 1), 1.0)
                for pg in sub_optimizer.param_groups:
                    for pi, p in enumerate(pg["params"]):
                        if p is sub_net.hidden_layers[idx].weight or p is sub_net.hidden_layers[idx].bias:
                            pg["lr"] = base_lr * w_factor

            x, y = next(iter(train_loader))
            x, y = x.to(self._device), y.to(self._device)
            x = x.view(x.size(0), -1)

            sub_optimizer.zero_grad()
            logits = sub_net(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss += self.l2_lambda * 0.5 * sum(
                p.pow(2).sum() for p in sub_net.parameters() if p.requires_grad
            )
            kp_loss = self._sub_net_kp_loss(sub_net, sub_weights)
            loss += self.regular_lambda * kp_loss
            loss.backward()
            sub_optimizer.step()
            self._apply_l1_soft_thresholding_module(sub_net)

            self.loss_log.append((task_id, self._global_iter, loss.item()))
            self._global_iter += 1

        trained_sub = {}
        for i in range(self.n_hidden_layers):
            trained_sub[f"layer{i}/weight"] = sub_net.hidden_layers[i].weight.data.clone()
            trained_sub[f"layer{i}/bias"] = sub_net.hidden_layers[i].bias.data.clone()
        trained_sub["output/weight"] = sub_net.output_head.weight.data.clone()
        trained_sub["output/bias"] = sub_net.output_head.bias.data.clone()

        shared_weights, head_w, head_b = merge_sub_network(
            shared_weights, head_w, head_b,
            active_indices, trained_sub, self.n_hidden_layers,
        )
        for i in range(self.n_hidden_layers):
            self.hidden_layers[i].weight.data = shared_weights[i]["weight"]
            self.hidden_layers[i].bias.data = shared_weights[i]["bias"]
        self.output_heads[f"task_{task_id}"].weight.data = head_w
        self.output_heads[f"task_{task_id}"].bias.data = head_b

        val_loss, _ = self._evaluate(val_loader, task_id)
        return val_loss

    def _train_expanded_network(self, task_id, train_loader, val_loader, max_iter, lr):
        """Override full-network training with experiment modifications."""
        self._experiment_train_loop(
            task_id, train_loader, max_iter, lr,
            use_kp=True, use_group_lasso=True,
        )

    def _train_after_split(self, task_id, train_loader, val_loader, max_iter, lr):
        """Override split training with experiment modifications."""
        self._experiment_train_loop(
            task_id, train_loader, max_iter, lr,
            use_kp=True, use_group_lasso=False,
        )


# =========================================================================
#  Experiment Runner
# =========================================================================

def run_experiment(exp_cfg, result_subdir):
    """Run a single experiment and return results dict."""
    log_dir = RESULTS_DIR / result_subdir
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_default_cfg()
    model = ExperimentalDEN(**{k: cfg[k] for k in [
        "input_dim", "hidden_dims", "num_classes", "ex_k",
        "l1_lambda", "l2_lambda", "gl_lambda", "regular_lambda",
        "loss_thr", "spl_thr", "depth_growth_enabled",
    ]}, depth_growth_config={"interval": cfg["depth_growth_interval"]},
        experiment_config=exp_cfg)
    model.to(DEVICE)

    train_loaders, val_loaders, test_loaders = get_data(N_TASKS)

    all_test_accs = []  # acc_matrix: [after_task][task_evaluated]
    depth_insertion_task = None

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {exp_cfg.get('type', 'baseline')}")
    if exp_cfg.get("freeze_iters"):
        print(f"  freeze_iters={exp_cfg['freeze_iters']}")
    if exp_cfg.get("warmup_iters"):
        print(f"  warmup_iters={exp_cfg['warmup_iters']}")
    print(f"{'='*70}")

    for task_id in range(N_TASKS):
        print(f"\n  --- Task {task_id + 1}/{N_TASKS} ---")

        # --- Activation analysis: capture BEFORE insertion training ---
        if exp_cfg.get("type") == "activation_analysis" and task_id == 1:
            print("  [*] Capturing activations BEFORE depth insertion...")
            acts = model.capture_activations(test_loaders[0], 0, label="before")
            for lidx, (m, s) in sorted(acts.items()):
                print(f"      Layer {lidx}: mean={m:.4f}, std={s:.4f}")

        train_loader = train_loaders[task_id]
        val_loader = val_loaders[task_id]
        test_loader = test_loaders[task_id]

        # Check if model had a depth insertion BEFORE this task
        is_first_after_insertion = (
            depth_insertion_task is not None and
            task_id == depth_insertion_task + 1
        )

        if is_first_after_insertion and exp_cfg.get("type") not in ("baseline", "activation_analysis", "identity_monitor"):
            print(f"  [*] Applying experiment '{exp_cfg['type']}' for task {task_id}")
            use_experiment = True
        else:
            use_experiment = False

        # Standard training for all tasks except the one after insertion
        if not use_experiment:
            test_acc, sparsity, expansion_info = model.add_task(
                task_id=task_id,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                max_iter=cfg["max_iter"],
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
                device=DEVICE,
                verbose=True,
            )
        else:
            # Custom training with experiment modifications
            test_acc = _train_task_with_experiment(
                model, task_id, train_loader, val_loader,
                test_loader, cfg["max_iter"], cfg["lr"],
                cfg["batch_size"], DEVICE,
            )

        # Check if depth insertion happened (look at architecture log)
        if model.architecture_log and model.architecture_log[-1].get("depth_inserted", False):
            depth_insertion_task = task_id
            model.set_inserted_layer(model.n_hidden_layers - 1)
            print(f"  [*] Depth insertion detected. Inserted layer at index {model.inserted_layer_idx}")
            print(f"      Architecture: {get_architecture_summary(model)}")

            # Identity verification
            if exp_cfg.get("type") in ("identity_monitor", "all"):
                idx = model.inserted_layer_idx
                w = model.hidden_layers[idx].weight.data
                n = min(w.size(0), w.size(1))
                I = torch.eye(n, device=w.device)
                id_norm = (w[:n, :n] - I).norm().item()
                print(f"      (W - I).norm() after insertion: {id_norm:.6f}")

            # Activation analysis: capture AFTER insertion
            if exp_cfg.get("type") == "activation_analysis":
                print("  [*] Capturing activations AFTER depth insertion...")
                acts = model.capture_activations(test_loaders[0], 0, label="after")
                for lidx, (m, s) in sorted(acts.items()):
                    print(f"      Layer {lidx}: mean={m:.4f}, std={s:.4f}")

        # Evaluate all tasks so far
        task_accs = evaluate_all(model, test_loaders, task_id + 1)
        all_test_accs.append(task_accs)
        forget_vals = forgetting(all_test_accs)

        print(f"  Accuracies: {[f'{a:.4f}' for a in task_accs]}")
        if forget_vals:
            print(f"  Forgetting: {[f'{f:.4f}' for f in forget_vals]}")
        print(f"  Neurons/layer: {model.get_neuron_counts()}")
        print(f"  Hidden layers: {model.n_hidden_layers}")

    # --- Final evaluation ---
    final_accs = evaluate_all(model, test_loaders, N_TASKS)
    forget_vals = forgetting(all_test_accs)

    results = {
        "experiment": exp_cfg.get("type", "baseline"),
        "config": dict(exp_cfg),
        "final_accuracies": final_accs,
        "forgetting": forget_vals,
        "avg_forgetting": float(np.mean(forget_vals)) if forget_vals else 0.0,
        "acc_matrix": all_test_accs,
        "depth_insertion_task": depth_insertion_task,
        "weight_norm_log": model.weight_norm_log,
        "neighbor_delta_log": model.neighbor_delta_log,
        "loss_log": model.loss_log,
        "activation_before": model.activation_before,
        "activation_after": model.activation_after,
        "neuron_counts": model.get_neuron_counts(),
        "n_hidden_layers": model.n_hidden_layers,
        "architecture_log": model.architecture_log,
    }

    with open(log_dir / "results.json", "w") as f:
        json.dump(_to_native(results), f, indent=2, default=str)

    _plot_experiment_results(results, log_dir)
    print(f"\n  [*] Results saved to {log_dir}")
    return results


def _train_task_with_experiment(model, task_id, train_loader, val_loader,
                                 test_loader, max_iter, lr, batch_size, device):
    """Custom training for the first task after depth insertion with experiment mods."""
    model._device = device
    model.to(device)
    model.task_list.append(task_id)
    model._add_output_head(task_id)
    model._save_prev_params()

    # --- Selective retraining ---
    early_iter = max(int(max_iter / 10), 10)
    model._freeze_hidden_layers()

    head = model.output_heads[f"task_{task_id}"]
    head_optim = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(early_iter):
        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        head_optim.zero_grad()
        with torch.no_grad():
            h = x
            for layer in model.hidden_layers:
                h = F.relu(layer(h))
        logits = head(h)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        head_optim.step()

    model._unfreeze_all()

    shared_weights = model._get_shared_weights()
    head_w = model.output_heads[f"task_{task_id}"].weight.data
    active_indices, sub_weights = select_active_neurons(
        shared_weights, head_w, model.n_hidden_layers,
    )
    model._train_sub_network(
        task_id, active_indices, sub_weights,
        train_loader, val_loader, max_iter, lr,
    )

    # --- Dynamic expansion ---
    val_loss, _ = model._evaluate(val_loader, task_id)
    expansion_info = [0] * model.n_hidden_layers

    if val_loss > model.loss_thr:
        for i in range(model.n_hidden_layers):
            model.hidden_layers[i].expand_output_units(model.ex_k, task_id)
            expansion_info[i] = model.ex_k
            if i > 0:
                model.hidden_layers[i].expand_input_units(model.ex_k)
        for head in model.output_heads.values():
            head.expand_input_units(model.ex_k)
        model.to(device)
        model._train_expanded_network(
            task_id, train_loader, val_loader, max_iter, lr,
        )
        n_pruned = model._prune_dead_units()

    # --- Split & duplication ---
    model._split_and_duplicate(
        task_id, train_loader, val_loader, max_iter, lr, False, expansion_info,
    )

    # --- Save timestamps ---
    model._save_timestamps(task_id)

    test_loss, test_acc = model._evaluate(test_loader, task_id)
    model._save_prev_params()
    model._log_architecture(task_id, tuple(expansion_info))

    print(f"  [*] Task {task_id} (experiment): test_acc={test_acc:.4f}")
    return test_acc


# =========================================================================
#  Plotting
# =========================================================================

def _plot_experiment_results(results, log_dir):
    """Generate plots for experiment results."""
    exp_name = results["experiment"]
    acc_matrix = results.get("acc_matrix", [])
    loss_log = results.get("loss_log", [])
    wn_log = results.get("weight_norm_log", [])
    nd_log = results.get("neighbor_delta_log", [])

    n_tasks = len(acc_matrix)

    # --- Accuracy per task ---
    if acc_matrix:
        fig, ax = plt.subplots(figsize=(8, 5))
        for j in range(n_tasks):
            accs = [acc_matrix[k][j] for k in range(j, n_tasks)]
            ax.plot(range(j, n_tasks), accs, marker="o", label=f"Task {j+1}")
        ax.set_xlabel("After training task")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"Test Accuracy – {exp_name}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(log_dir / "accuracy.png", dpi=150)
        plt.close(fig)

    # --- Forgetting ---
    forgetting_vals = results.get("forgetting", [])
    if forgetting_vals:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(1, len(forgetting_vals) + 1), forgetting_vals, color="crimson")
        ax.set_xlabel("Task")
        ax.set_ylabel("Forgetting")
        ax.set_title(f"Forgetting per Task – {exp_name}")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(log_dir / "forgetting.png", dpi=150)
        plt.close(fig)

    # --- Training loss curve ---
    if loss_log:
        fig, ax = plt.subplots(figsize=(8, 4))
        iters = [x[1] for x in loss_log]
        losses = [x[2] for x in loss_log]
        ax.plot(iters, losses, alpha=0.7, linewidth=0.5)
        ax.set_xlabel("Global iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss – {exp_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(log_dir / "loss.png", dpi=150)
        plt.close(fig)

    # --- Weight norm evolution ---
    if wn_log:
        fig, ax = plt.subplots(figsize=(8, 4))
        iters = [x[0] for x in wn_log]
        norms = [x[1] for x in wn_log]
        ax.plot(iters, norms, color="darkorange", linewidth=1)
        ax.set_xlabel("Global iteration")
        ax.set_ylabel("(W - I).norm()")
        ax.set_title(f"Inserted Layer Weight Drift – {exp_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(log_dir / "weight_norm.png", dpi=150)
        plt.close(fig)

    # --- Neighbor delta ---
    if nd_log:
        fig, ax = plt.subplots(figsize=(8, 4))
        iters = [x[0] for x in nd_log]
        deltas = [x[1] for x in nd_log]
        ax.plot(iters, deltas, color="green", linewidth=1)
        ax.set_xlabel("Global iteration")
        ax.set_ylabel("Neighbor weight change (L2)")
        ax.set_title(f"Neighbor Layer Weight Change – {exp_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(log_dir / "neighbor_delta.png", dpi=150)
        plt.close(fig)


def _to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [_to_native(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return obj


# =========================================================================
#  Summary
# =========================================================================

def print_summary(all_results):
    """Print a comparative summary of all experiments."""
    print("\n" + "="*70)
    print("  COMPARATIVE SUMMARY")
    print("="*70)

    rows = []
    for name, res in all_results.items():
        final_accs = res.get("final_accuracies", [])
        forget = res.get("forgetting", [])
        avg_forget = res.get("avg_forgetting", 0.0)
        task1_before = res.get("task1_before", "N/A")
        task1_after = final_accs[0] if len(final_accs) > 0 else "N/A"
        wn_log = res.get("weight_norm_log", [])
        final_wn = wn_log[-1][1] if wn_log else "N/A"

        rows.append({
            "name": name,
            "task1_acc": f"{task1_after:.4f}" if isinstance(task1_after, float) else str(task1_after),
            "avg_forget": f"{avg_forget:.4f}",
            "final_wn": f"{final_wn:.4f}" if isinstance(final_wn, float) else str(final_wn),
            "forget_per_task": [f"{f:.4f}" for f in forget],
        })

    print(f"  {'Experiment':<25} {'Task1 Acc':<12} {'Avg Forget':<12} {'Final (W-I) norm':<18}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*18}")
    for r in rows:
        print(f"  {r['name']:<25} {r['task1_acc']:<12} {r['avg_forget']:<12} {r['final_wn']:<18}")

    # Find best strategy
    best = min(rows, key=lambda r: float(r['avg_forget']))
    print(f"\n  [-] Best strategy: {best['name']}")
    print(f"      Avg forgetting: {best['avg_forget']}, Task 1 accuracy: {best['task1_acc']}")


def print_activation_summary(results):
    """Print activation comparison."""
    print("\n" + "="*70)
    print("  ACTIVATION ANALYSIS SUMMARY")
    print("="*70)
    before = results.get("activation_before", {})
    after = results.get("activation_after", {})
    all_keys = sorted(set(list(before.keys()) + list(after.keys())))
    for k in all_keys:
        b = before.get(k, ("N/A", "N/A"))
        a = after.get(k, ("N/A", "N/A"))
        print(f"  Layer {k}: before (mean={b[0]:.4f}, std={b[1]:.4f})  "
              f"after (mean={a[0]:.4f}, std={a[1]:.4f})")


def print_identity_summary(results):
    """Print identity verification summary."""
    print("\n" + "="*70)
    print("  IDENTITY VERIFICATION SUMMARY")
    print("="*70)
    wn_log = results.get("weight_norm_log", [])
    if wn_log:
        print(f"  Initial (W-I).norm(): {wn_log[0][1]:.6f}" if len(wn_log) > 0 else "")
        print(f"  Final   (W-I).norm(): {wn_log[-1][1]:.6f}" if len(wn_log) > 0 else "")
        if len(wn_log) > 2:
            norms = [x[1] for x in wn_log]
            print(f"  Max drift: {max(norms):.6f}")
            print(f"  Avg drift rate: {(norms[-1] - norms[0]) / len(norms):.6f}/iter")

    nd_log = results.get("neighbor_delta_log", [])
    if nd_log:
        print(f"\n  Neighbor weight change:")
        print(f"  Initial neighbor delta: {nd_log[0][1]:.6f}" if len(nd_log) > 0 else "")
        print(f"  Final   neighbor delta: {nd_log[-1][1]:.6f}" if len(nd_log) > 0 else "")


# =========================================================================
#  Main
# =========================================================================

def main():
    global MAX_ITER, SEED, N_TASKS

    parser = argparse.ArgumentParser(
        description="DEN Depth Forgetting Analysis"
    )
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=["all", "baseline", "freeze_inserted", "freeze_neighbors",
                 "warmup", "activation_analysis", "identity_monitor"],
        help="Experiment to run"
    )
    parser.add_argument("--freeze-iters", type=int, default=200,
                        help="Number of iterations to freeze inserted layer")
    parser.add_argument("--warmup-iters", type=int, default=200,
                        help="Number of warmup iterations")
    parser.add_argument("--max-iter", type=int, default=MAX_ITER,
                        help="Training iterations per task")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed")
    args = parser.parse_args()

    MAX_ITER = args.max_iter
    SEED = args.seed

    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = {
        "baseline": {"type": "baseline"},
        "freeze_inserted": {"type": "freeze_inserted", "freeze_iters": args.freeze_iters},
        "freeze_neighbors": {"type": "freeze_neighbors"},
        "warmup": {"type": "warmup", "warmup_iters": args.warmup_iters},
        "activation_analysis": {"type": "activation_analysis"},
        "identity_monitor": {"type": "identity_monitor"},
    }

    if args.experiment == "all":
        to_run = list(experiments.keys())
    else:
        to_run = [args.experiment]

    all_results = {}
    for exp_name in to_run:
        cfg = experiments[exp_name]
        subdir = exp_name
        if exp_name == "freeze_inserted":
            subdir = f"freeze_inserted_{args.freeze_iters}"
        elif exp_name == "warmup":
            subdir = f"warmup_{args.warmup_iters}"

        results = run_experiment(cfg, subdir)
        all_results[subdir] = results

    # Print summaries
    if args.experiment == "all":
        print_summary(all_results)

    if "activation_analysis" in to_run and "activation_analysis" in all_results:
        print_activation_summary(all_results["activation_analysis"])

    if "identity_monitor" in to_run and "identity_monitor" in all_results:
        print_identity_summary(all_results["identity_monitor"])

    print(f"\n  [*] All results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
