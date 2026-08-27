"""
Dynamically Expandable Network (DEN) – main model.

Implements the full algorithm described in:

    Yoon, Yang, Lee, Hwang.  "Lifelong Learning with Dynamically
    Expandable Networks."  ICLR 2018.

Lifecycle for each task T:

  1. First task (T=0) → standard supervised training.
  2. Subsequent tasks:
       a. **Selective retraining** — freeze shared layers, train only
          the output head briefly, identify the active sub-network,
          train it with knowledge-preservation, merge back.
       b. **Dynamic expansion** — if the loss threshold is not met,
          add *ex_k* hidden units with group-lasso regularisation,
          prune dead units.
       c. **Split & duplication** — detect drifted neurons, duplicate
          them so one copy preserves old knowledge and the other
          adapts.
  3. Save timestamps for task-specific inference.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .layers import DynamicLinear, TaskOutputHead
from .prune import select_active_neurons, merge_sub_network
from .grow import (
    group_lasso_step,
    find_useless_new_units,
)
from .grow_depth import should_insert_layer, insert_hidden_layer, get_architecture_summary
from .utils import knowledge_preservation_loss, get_prev_weight_slices, accuracy as accuracy_fn


def _make_cycling_iter(loader):
    """Create an iterator that restarts when exhausted."""
    it = iter(loader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(loader)


def _tensorify_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return {"_tensor": obj.detach().cpu()}
    if isinstance(obj, dict):
        return {k: _tensorify_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tensorify_to_cpu(v) for v in obj]
    return obj


def _tensorify_from_cpu(obj):
    if isinstance(obj, dict) and set(obj.keys()) == {"_tensor"}:
        return obj["_tensor"]
    if isinstance(obj, dict):
        return {k: _tensorify_from_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tensorify_from_cpu(v) for v in obj]
    return obj


class DEN(nn.Module):
    """
    Dynamically Expandable Network.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dims : list[int]
        Sizes of the hidden layers (excluding input and output).
    num_classes : int
        Number of output classes per task.
    ex_k : int
        Number of units added during each expansion / split.
    l1_lambda : float
        L1 sparsity penalty (applied via soft-thresholding).
    l2_lambda : float
        L2 weight-decay penalty.
    gl_lambda : float
        Group-lasso penalty on newly added units.
    regular_lambda : float
        Weight for the knowledge-preservation (L2 toward previous
        weights) term.
    loss_thr : float
        If the loss after selective retraining exceeds this threshold,
        trigger dynamic expansion.
    spl_thr : float
        L2-norm change above which a neuron is considered "drifted"
        and eligible for splitting.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        ex_k: int = 10,
        l1_lambda: float = 1e-5,
        l2_lambda: float = 1e-4,
        gl_lambda: float = 0.001,
        regular_lambda: float = 0.5,
        loss_thr: float = 0.01,
        spl_thr: float = 0.05,
        depth_growth_enabled: bool = False,
        depth_growth_config: dict | None = None,
        embedder: nn.Module | None = None,
        context_growth_enabled: bool = False,
        context_growth_thr: float = 0.05,
        context_growth_step: int = 64,
        context_max: int = 1024,
        lstm_hidden: int = 256,
        current_context: int = 0,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.ex_k = ex_k
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.gl_lambda = gl_lambda
        self.regular_lambda = regular_lambda
        self.loss_thr = loss_thr
        self.spl_thr = spl_thr
        self.depth_growth_enabled = depth_growth_enabled
        self.depth_growth_config = depth_growth_config or {}

        # Optional shared input embedding (e.g. token ids -> vectors for
        # language modeling).  Must project to ``input_dim`` features.
        self.embedder = embedder
        self.tie_embeddings = tie_embeddings

        # --- Context growth (Approach A: rebuild first layer) ---
        self.context_growth_enabled = context_growth_enabled
        self.context_growth_thr = context_growth_thr
        self.context_growth_step = context_growth_step
        self.context_max = context_max
        self.current_context = current_context
        self.lstm_hidden = lstm_hidden
        self.emb_dim = embedder.embedding_dim if embedder is not None else 0

        self.n_hidden_layers = len(hidden_dims)

        # --- Build initial hidden layers ---
        prev_dim = input_dim
        self.hidden_layers = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims):
            layer = DynamicLinear(prev_dim, h_dim, task_id=0)
            self.hidden_layers.append(layer)
            prev_dim = h_dim

        # --- Task-specific output heads (built lazily) ---
        self.output_heads = nn.ModuleDict()

        # Tracking
        self.task_list: list[int] = []
        self.prev_params: dict[str, torch.Tensor] = {}
        self.timestamps: dict[int, list[int]] = {}
        self._device = torch.device("cpu")

        # Architecture evolution log (for depth-growth experiments)
        self.architecture_log: list[dict] = []

        # Cross-task tracker for data-driven depth-growth criteria
        self.depth_growth_tracker: dict = {}
        # Cooldown: number of tasks to wait after a growth event
        self.growth_cooldown_tasks: int = 3
        self._last_growth_task: int = -100
        # Fraction of each mini-batch to draw from replay buffer
        self.replay_fraction: float = 0.3
        # Consolidation (periodic pruning) settings — conservative defaults
        self.consolidation_interval: int = 3
        self.consolidation_gl: float = max(self.gl_lambda, 0.001)
        self.consolidation_iters: int = 8
        self.consolidation_row_norm_thr: float = 1e-2

    # ================================================================
    #  Forward
    # ================================================================

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed token ids and flatten to (B, context * emb_dim)."""
        if self.embedder is None:
            return x.view(x.size(0), -1)
        return self.embedder(x).view(x.size(0), -1)

    def forward(
        self,
        x: torch.Tensor,
        task_id: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        If *task_id* is given, each hidden layer is sliced to the
        number of input/output neurons that existed when that task
        was trained (recorded in ``self.timestamps``).

        Slicing matches the original TF implementation where newly
        added neurons are always appended and stamps track dimensions.
        """
        x = self._embed(x)
        h = x
        if task_id is not None and task_id in self.timestamps:
            stamp = self.timestamps[task_id]
            # stamp includes [out_0, out_1, ..., out_{N-1}, num_classes]
            # Number of hidden layers at task time = len(stamp) - 1.
            # Layers inserted *after* this task are skipped during
            # prediction (they didn't exist when the task was trained).
            n_layers_at_task = len(stamp) - 1
            for i in range(n_layers_at_task):
                layer = self.hidden_layers[i]
                h = F.relu(layer(h, out_slice=stamp[i], in_slice=stamp[i - 1] if i > 0 else h.size(-1)))
            tid = task_id if task_id is not None else self.task_list[-1]
            head = self.output_heads[f"task_{tid}"]
            h = head(h, in_slice=stamp[-2] if len(stamp) >= 2 else None)
        else:
            for layer in self.hidden_layers:
                h = F.relu(layer(h))
            tid = task_id if task_id is not None else self.task_list[-1]
            head = self.output_heads[f"task_{tid}"]
            h = head(h)
        return h

    # ================================================================
    #  Lifecycle
    # ================================================================

    def add_task(
        self,
        task_id: int,
        train_loader,
        val_loader,
        test_loader,
        max_iter: int,
        lr: float,
        batch_size: int,
        device: torch.device,
        verbose: bool = True,
        replay_buffer: list | None = None,
        teacher_model: nn.Module | None = None,
        kd_alpha: float = 0.5,
    ) -> tuple[float, float, tuple[int, ...]]:
        self._device = device
        self.to(device)
        if self.depth_growth_enabled:
            self.depth_growth_tracker["device"] = device

        if task_id == 0:
            return self._train_first_task(
                task_id, train_loader, val_loader, test_loader,
                max_iter, lr, batch_size, verbose, replay_buffer=replay_buffer,
            )
        else:
            # attach teacher snapshot for KD distillation during replay
            self._teacher = None
            self._kd_alpha = 0.0
            if teacher_model is not None:
                try:
                    self._teacher = teacher_model.to(self._device)
                    self._teacher.eval()
                    self._kd_alpha = kd_alpha
                except Exception:
                    self._teacher = None
                    self._kd_alpha = 0.0

            res = self._train_subsequent_task(
                task_id, train_loader, val_loader, test_loader,
                max_iter, lr, batch_size, verbose, replay_buffer=replay_buffer,
            )

            # clear teacher after training
            self._teacher = None
            self._kd_alpha = 0.0
            return res

    # ================================================================
    #  Checkpointing (save / resume)
    # ================================================================

    def save_checkpoint(self, path: str | Path):
        """Persist the model so training can resume without replaying
        past tasks.  Captures the *grown* architecture (expanded /
        pruned / split layers, inserted hidden layers, per-task heads)
        plus all learned weights and growth bookkeeping."""
        state = {
            "config": {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "num_classes": self.num_classes,
                "ex_k": self.ex_k,
                "l1_lambda": self.l1_lambda,
                "l2_lambda": self.l2_lambda,
                "gl_lambda": self.gl_lambda,
                "regular_lambda": self.regular_lambda,
                "loss_thr": self.loss_thr,
                "spl_thr": self.spl_thr,
                "depth_growth_enabled": self.depth_growth_enabled,
                "depth_growth_config": self.depth_growth_config,
                "context_growth_enabled": self.context_growth_enabled,
                "context_growth_thr": self.context_growth_thr,
                "context_growth_step": self.context_growth_step,
                "context_max": self.context_max,
                "current_context": self.current_context,
                "tie_embeddings": getattr(self, 'tie_embeddings', False),
            },
            "structure": [
                {"in": layer.in_features, "out": layer.out_features}
                for layer in self.hidden_layers
            ],
            "head_tasks": sorted(
                int(k.split("_")[1]) for k in self.output_heads.keys()
            ),
            "n_hidden_layers": self.n_hidden_layers,
            "task_list": list(self.task_list),
            "timestamps": {int(k): list(v) for k, v in self.timestamps.items()},
            "prev_params": {k: v.detach().cpu() for k, v in self.prev_params.items()},
            "architecture_log": self.architecture_log,
            "depth_growth_tracker": _tensorify_to_cpu(self.depth_growth_tracker),
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }
        if self.embedder is not None:
            state["embedder"] = {
                "num_embeddings": self.embedder.num_embeddings,
                "embedding_dim": self.embedder.embedding_dim,
            }
        state["tie_embeddings"] = getattr(self, "tie_embeddings", False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "DEN":
        """Rebuild a DEN from a checkpoint, preserving the grown
        architecture exactly as it was when training stopped."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = state["config"]
        model = cls(**cfg)

        # Rebuild hidden layers with their exact grown dimensions.
        layers = nn.ModuleList()
        for s in state["structure"]:
            layers.append(DynamicLinear(s["in"], s["out"]))
        model.hidden_layers = layers
        model.n_hidden_layers = state["n_hidden_layers"]

        # Rebuild task-specific heads (all share the last layer's size).
        last_out = state["structure"][-1]["out"]
        for task_id in state["head_tasks"]:
            model.output_heads[f"task_{task_id}"] = TaskOutputHead(
                last_out, cfg["num_classes"], task_id,
                embedder=(model.embedder if getattr(model, 'embedder', None) is not None else None),
                tie=bool(cfg.get("tie_embeddings", False)),
            )

        # Optional embedding layer.
        if "embedder" in state:
            emb = state["embedder"]
            model.embedder = nn.Embedding(emb["num_embeddings"], emb["embedding_dim"])

        # restore tie flag
        model.tie_embeddings = bool(state.get("tie_embeddings", False))

        # Restore all parameters, buffers and growth bookkeeping.
        model.load_state_dict(state["state_dict"], strict=True)
        model.task_list = list(state["task_list"])
        model.timestamps = {int(k): list(v) for k, v in state["timestamps"].items()}
        model.prev_params = {k: v.clone() for k, v in state["prev_params"].items()}
        model.architecture_log = list(state["architecture_log"])
        model.depth_growth_tracker = _tensorify_from_cpu(state["depth_growth_tracker"])
        return model

    # ---------------------------------------------------------------
    # Embedding / tying helpers
    # ---------------------------------------------------------------
    def expand_embedding_vocab(self, new_num_embeddings: int):
        """Expand the embedding matrix to `new_num_embeddings` rows.

        Existing weights are copied; new rows are zero-initialized.
        """
        if self.embedder is None:
            raise RuntimeError("No embedder to expand")
        cur = self.embedder.num_embeddings
        if new_num_embeddings <= cur:
            return
        old_w = self.embedder.weight.data
        new_w = torch.zeros(new_num_embeddings, old_w.size(1), device=old_w.device)
        new_w[:cur] = old_w
        self.embedder = nn.Embedding(new_num_embeddings, old_w.size(1))
        with torch.no_grad():
            self.embedder.weight.data = new_w
        self.emb_dim = self.embedder.embedding_dim

    def enable_embedding_tying(self):
        """Enable weight-tying between the embedder and all output heads.

        Replaces existing heads with tied `TaskOutputHead` and solves a
        least-squares projection so the new tied heads approximate the
        original weights.
        """
        if self.embedder is None:
            print("  [!] No embedder present; cannot enable tying")
            return
        for key in list(self.output_heads.keys()):
            # read current full weight/bias via helper if available
            old = self.output_heads[key]
            try:
                w = old.get_weight()
                b = old.get_bias()
            except Exception:
                # fallback: try direct attributes
                w = old.weight.data.clone() if hasattr(old, 'weight') else None
                b = old.bias.data.clone() if hasattr(old, 'bias') else None
            if w is None or b is None:
                continue
            # create a tied head and set its projection to match w
            task_id = int(key.split("_")[1])
            tied = TaskOutputHead(w.size(1), self.num_classes, task_id, embedder=self.embedder, tie=True)
            tied.set_weight_and_bias(w, b)
            self.output_heads[key] = tied
        self.tie_embeddings = True

    # ---------------------------------------------------------------
    #  First task
    # ---------------------------------------------------------------

    def _train_first_task(
        self,
        task_id: int,
        train_loader,
        val_loader,
        test_loader,
        max_iter: int,
        lr: float,
        batch_size: int,
        verbose: bool,
        replay_buffer: list | None = None,
    ) -> tuple[float, float, tuple[int, ...]]:
        self.task_list.append(task_id)
        self._add_output_head(task_id)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        data_iter = _make_cycling_iter(train_loader)

        for iteration in range(max_iter):
            x, y = next(data_iter)
            x, y = x.to(self._device), y.to(self._device)
            x = x.view(x.size(0), -1)
            b = x.size(0)
            # Mix replay samples to reduce forgetting during expansion training
            k = 0
            idx = None
            if replay_buffer and len(replay_buffer) > 0:
                k = max(1, min(len(replay_buffer), int(self.replay_fraction * x.size(0))))
                idx = torch.randperm(len(replay_buffer))[:k]
                # unpack replay entries (may be (x,y) or (x,y,teacher_probs))
                rx_list, ry_list = [], []
                for ii in idx:
                    entry = replay_buffer[int(ii)]
                    rx_list.append(entry[0])
                    ry_list.append(entry[1])
                rx = torch.stack(rx_list).to(self._device)
                ry = torch.stack(ry_list).to(self._device)
                rx = rx.view(rx.size(0), -1)
                x = torch.cat([x, rx], dim=0)
                y = torch.cat([y, ry], dim=0)

            optimizer.zero_grad()
            logits = self(x, task_id=task_id)
            # multiclass next-token prediction: use cross-entropy with class indices
            loss = F.cross_entropy(logits, y)
            reg = self.l2_lambda * 0.5 * sum(
                p.pow(2).sum() for p in self.parameters()
            )
            loss = loss + reg
            # KD distillation on replayed samples (MSE between probs)
            if k > 0 and hasattr(self, '_teacher') and getattr(self, '_teacher', None) is not None and getattr(self, '_kd_alpha', 0.0) > 0.0:
                # Only perform KD if replay entries include teacher probabilities
                has_teacher_probs = all(len(replay_buffer[int(ii)]) > 2 and replay_buffer[int(ii)][2] is not None for ii in idx)
                if has_teacher_probs:
                    student_replay_logits = logits[b:]
                    # teacher stored softmax probs; match with student softmax
                    student_log = F.log_softmax(student_replay_logits, dim=1)
                    teacher_probs = torch.stack([replay_buffer[int(ii)][2] for ii in idx]).to(self._device)
                    # Distill by matching teacher distribution with KL divergence
                    kd_loss = self._kd_alpha * F.kl_div(student_log, teacher_probs, reduction='batchmean')
                    loss = loss + kd_loss
            loss.backward()
            optimizer.step()
            self._apply_l1_soft_thresholding()

        test_loss, test_acc = self._evaluate(test_loader, task_id)
        self._save_prev_params()
        self._save_timestamps(task_id)
        self._log_architecture(task_id, tuple(0 for _ in range(self.n_hidden_layers)))

        exp_info = tuple(0 for _ in range(self.n_hidden_layers))
        return test_acc, self._avg_sparsity(), exp_info

    # ---------------------------------------------------------------
    #  Subsequent tasks
    # ---------------------------------------------------------------

    def _train_subsequent_task(
        self,
        task_id: int,
        train_loader,
        val_loader,
        test_loader,
        max_iter: int,
        lr: float,
        batch_size: int,
        verbose: bool,
        replay_buffer: list | None = None,
    ) -> tuple[float, float, tuple[int, ...]]:
        self.task_list.append(task_id)
        self._add_output_head(task_id)
        self._save_prev_params()

        # --- Step 1: Selective Retraining ---
        if verbose:
            print("\n  [*] Selective retraining")
        loss_after_selective = self._selective_retrain(
            task_id, train_loader, val_loader,
            max_iter, lr, verbose, replay_buffer=replay_buffer,
        )

        expansion_info = [0] * self.n_hidden_layers

        # --- Step 2: Dynamic Expansion (if loss too high) ---
        if loss_after_selective > self.loss_thr:
            if verbose:
                print("  [*] Network expansion triggered")
            self._dynamic_expansion(
                task_id, train_loader, val_loader,
                max_iter, lr, verbose,
                expansion_info, replay_buffer=replay_buffer,
            )

        # --- Step 3: Split & Duplication ---
        if verbose:
            print("  [*] Split & duplication")
        self._split_and_duplicate(
            task_id, train_loader, val_loader,
            max_iter, lr, verbose,
            expansion_info,
            replay_buffer=replay_buffer,
        )

        # --- Save timestamps BEFORE depth growth ---
        # Must save before insertion so the current task's stamp reflects
        # the architecture that was actually trained.  If we saved after
        # insertion, the stamp would include the new layer, causing the
        # forward pass to use it for this task — even though this task
        # was never trained with it.
        self._save_timestamps(task_id)

        # --- Step 4: Depth Growth (data-driven) ---
        # After training this task, decide whether to insert a new hidden
        # layer before the next task.  The decision logic is delegated to
        # the criterion selected in the config (see models/criteria.py).
        depth_inserted = False
        if self.depth_growth_enabled:
            # Update tracker for criteria
            self.depth_growth_tracker.setdefault("val_losses", []).append(
                loss_after_selective
            )
            self.depth_growth_tracker.setdefault("neurons_added", []).append(
                sum(expansion_info)
            )
            if "device" not in self.depth_growth_tracker:
                self.depth_growth_tracker["device"] = self._device
            # Save a probe batch for activation-based criteria
            if "probe_batch" not in self.depth_growth_tracker:
                try:
                    probe_x, _ = next(iter(train_loader))
                    self.depth_growth_tracker["probe_batch"] = probe_x
                except StopIteration:
                    pass

            dg_config = {
                "depth_growth_enabled": True,
                "depth_growth_criterion": self.depth_growth_config.get("criterion", ""),
                **self.depth_growth_config,
            }
            if should_insert_layer(task_id, {}, self, dg_config):
                if verbose:
                    print("  [*] Depth growth: inserting new hidden layer")
                insert_idx = insert_hidden_layer(self, task_id)
                depth_inserted = True
                # mark insertion as a growth event
                self._last_growth_task = task_id
                self._warmup_new_layer(task_id, train_loader)
                self._save_prev_params()
                if verbose:
                    arch = get_architecture_summary(self)
                    print(
                        f"      Inserted at position {insert_idx}. "
                        f"Layers now: {arch['n_hidden_layers']}, "
                        f"neurons: {arch['neurons_per_layer']}"
                    )

        # --- Evaluation ---
        test_loss, test_acc = self._evaluate(test_loader, task_id)
        if verbose:
            print(
                f"  [*] Task {task_id}: test_loss={test_loss:.4f}, "
                f"test_acc={test_acc:.4f}, "
                f"sparsity={self._avg_sparsity():.4f}"
            )

        self._save_prev_params()
        self._log_architecture(task_id, tuple(expansion_info), depth_inserted=depth_inserted)
        # Periodic consolidation pruning to cap growth and remove redundancies
        try:
            if task_id % self.consolidation_interval == 0 and task_id > 0:
                self._consolidate()
        except Exception:
            pass
        # Context growth check (caller must rebuild loaders if True)
        self._context_grew = self._maybe_grow_context(test_loss)
        return test_acc, self._avg_sparsity(), tuple(expansion_info)

    # ---------------------------------------------------------------
    #  Context growth
    # ---------------------------------------------------------------

    def _maybe_grow_context(self, val_loss: float) -> bool:
        """Check if context window should grow based on val loss.

        When growth triggers, rebuilds the first hidden layer with larger
        input_dim to accommodate the new context size. Old weights are
        copied into the matching columns; new columns are zero-initialized.

        Returns True if context was grown (caller must rebuild data loaders).
        """
        if not self.context_growth_enabled:
            return False
        if val_loss > self.context_growth_thr and self.current_context < self.context_max:
            old_ctx = self.current_context
            new_ctx = min(old_ctx + self.context_growth_step, self.context_max)
            old_input_dim = old_ctx * self.emb_dim
            new_input_dim = new_ctx * self.emb_dim

            # Rebuild first hidden layer with new input dimension
            first_layer = self.hidden_layers[0]
            old_w = first_layer.weight.data  # [out, old_in]
            old_b = first_layer.bias.data
            old_anchor = first_layer.weight_anchor if hasattr(first_layer, 'weight_anchor') else None
            old_bias_anchor = first_layer.bias_anchor if hasattr(first_layer, 'bias_anchor') else None
            old_timestamp = first_layer.timestamp if hasattr(first_layer, 'timestamp') else None

            # Create new layer with larger input
            new_layer = DynamicLinear(new_input_dim, first_layer.out_features, task_id=0)

            # Copy old weights: each word's embeddings are contiguous
            # old weight columns [0:old_input_dim] map to new columns [0:old_input_dim]
            with torch.no_grad():
                new_layer.weight.data[:, :old_input_dim] = old_w
                new_layer.bias.data = old_b.clone()
                if old_anchor is not None:
                    new_layer.weight_anchor[:, :old_input_dim] = old_anchor
                if old_bias_anchor is not None:
                    new_layer.bias_anchor = old_bias_anchor.clone()
                if old_timestamp is not None:
                    new_layer.timestamp = old_timestamp.clone()

            self.hidden_layers[0] = new_layer
            self.input_dim = new_input_dim
            self.current_context = new_ctx

            print(f"  [*] Context grown: {old_ctx} -> {new_ctx} (input_dim: {old_input_dim} -> {new_input_dim})")
            return True
        return False

    # ================================================================
    #  Selective Retraining
    # ================================================================

    def _selective_retrain(
        self,
        task_id: int,
        train_loader,
        val_loader,
        max_iter: int,
        lr: float,
        verbose: bool,
        replay_buffer: list | None = None,
    ) -> float:
        """
        Phase 1: freeze shared layers → train only the output head
        briefly → identify active sub-network → train sub-network
        with knowledge preservation → merge back.
        """
        early_iter = max(int(max_iter / 5), 50)

        # --- 1a. Train only the output head ---
        self._freeze_hidden_layers()
        self._train_head_only(
            task_id, train_loader, early_iter, lr, replay_buffer=replay_buffer,
        )

        # --- 1b. Identify active neurons ---
        shared_weights = self._get_shared_weights()
        head = self.output_heads[f"task_{task_id}"]
        head_w = head.get_weight()
        # Build a small probe batch to collect layer activations for
        # activation-driven sub-network selection.
        probe_batch = None
        if "probe_batch" in self.depth_growth_tracker:
            probe_batch = self.depth_growth_tracker.get("probe_batch")
        else:
            try:
                probe_x, _ = next(iter(train_loader))
                probe_batch = probe_x
            except Exception:
                probe_batch = None

        activations = None
        if probe_batch is not None:
            with torch.no_grad():
                xb = probe_batch.to(self._device)
                xb = xb.view(xb.size(0), -1)
                acts = []
                h = self._embed(xb)
                for layer in self.hidden_layers:
                    h = F.relu(layer(h))
                    acts.append(h.detach().cpu())
                activations = acts

        active_indices, sub_weights = select_active_neurons(
            shared_weights, head_w, self.n_hidden_layers, activations=activations,
        )

        # --- 1c. Train sub-network with knowledge preservation ---
        self._unfreeze_all()
        loss_after = self._train_sub_network(
            task_id, active_indices, sub_weights,
            train_loader, val_loader, max_iter, lr, replay_buffer=replay_buffer,
        )
        return loss_after

    def _freeze_hidden_layers(self):
        for layer in self.hidden_layers:
            for p in layer.parameters():
                p.requires_grad = False

    def _unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def _get_shared_weights(self) -> list[dict[str, torch.Tensor]]:
        return [
            {"weight": layer.weight.data.clone(), "bias": layer.bias.data.clone()}
            for layer in self.hidden_layers
        ]

    def _train_head_only(
        self,
        task_id: int,
        train_loader,
        num_iters: int,
        lr: float,
        replay_buffer: list | None = None,
    ):
        head = self.output_heads[f"task_{task_id}"]
        optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        data_iter = _make_cycling_iter(train_loader)
        for _ in range(num_iters):
            x, y = next(data_iter)
            x, y = x.to(self._device), y.to(self._device)
            b = x.size(0)
            # Mix replay samples (if any) to reduce forgetting
            k = 0
            idx = None
            if replay_buffer and len(replay_buffer) > 0:
                k = max(1, min(len(replay_buffer), int(self.replay_fraction * x.size(0))))
                idx = torch.randperm(len(replay_buffer))[:k]
                rx_list, ry_list = [], []
                for ii in idx:
                    entry = replay_buffer[int(ii)]
                    rx_list.append(entry[0])
                    ry_list.append(entry[1])
                rx = torch.stack(rx_list).to(self._device)
                ry = torch.stack(ry_list).to(self._device)
                rx = rx.view(rx.size(0), -1)
                x = torch.cat([x, rx], dim=0)
                y = torch.cat([y, ry], dim=0)
            optimizer.zero_grad()
            with torch.no_grad():
                h = self._embed(x)
                for layer in self.hidden_layers:
                    h = F.relu(layer(h))
            logits = head(h)
            loss = F.cross_entropy(logits, y)
            # KD on replay (if teacher probs attached)
            if k > 0 and hasattr(self, '_teacher') and getattr(self, '_teacher', None) is not None and getattr(self, '_kd_alpha', 0.0) > 0.0:
                has_teacher_probs = all(len(replay_buffer[int(ii)]) > 2 and replay_buffer[int(ii)][2] is not None for ii in idx)
                if has_teacher_probs:
                    student_replay_logits = logits[b:]
                    student_log = F.log_softmax(student_replay_logits, dim=1)
                    teacher_probs = torch.stack([replay_buffer[int(ii)][2] for ii in idx]).to(self._device)
                    loss = loss + self._kd_alpha * F.kl_div(student_log, teacher_probs, reduction='batchmean')
            loss.backward()
            optimizer.step()

    def _train_sub_network(
        self,
        task_id: int,
        active_indices: list[list[int]],
        sub_weights: dict[str, torch.Tensor],
        train_loader,
        val_loader,
        max_iter: int,
        lr: float,
        replay_buffer: list | None = None,
    ) -> float:
        """
        Build a temporary sub-network, train with knowledge
        preservation, merge back into the full model.
        """
        # Snapshot full weights
        shared_weights = self._get_shared_weights()
        head = self.output_heads[f"task_{task_id}"]
        head_w = head.get_weight().clone()
        head_b = head.get_bias().clone()

        # Build sub-network
        sub_net = self._build_sub_network(sub_weights, task_id).to(self._device)
        sub_optimizer = torch.optim.Adam(sub_net.parameters(), lr=lr)

        sub_iters = max(3 * max_iter, max_iter)
        data_iter = _make_cycling_iter(train_loader)
        for iteration in range(sub_iters):
            x, y = next(data_iter)
            x, y = x.to(self._device), y.to(self._device)
            b = x.size(0)
            x = x.view(x.size(0), -1)

            # Mix replay samples to the training batch to reduce forgetting
            k = 0
            idx = None
            if replay_buffer and len(replay_buffer) > 0:
                k = max(1, min(len(replay_buffer), int(self.replay_fraction * x.size(0))))
                idx = torch.randperm(len(replay_buffer))[:k]
                rx_list, ry_list = [], []
                for ii in idx:
                    entry = replay_buffer[int(ii)]
                    rx_list.append(entry[0])
                    ry_list.append(entry[1])
                rx = torch.stack(rx_list).to(self._device)
                ry = torch.stack(ry_list).to(self._device)
                rx = rx.view(rx.size(0), -1)
                x = torch.cat([x, rx], dim=0)
                y = torch.cat([y, ry], dim=0)

            sub_optimizer.zero_grad()
            logits = sub_net(x)
            loss = F.cross_entropy(logits, y)
            # L2
            loss += self.l2_lambda * 0.5 * sum(
                p.pow(2).sum() for p in sub_net.parameters()
            )
            # Knowledge preservation
            kp_loss = self._sub_net_kp_loss(sub_net, sub_weights)
            loss += self.regular_lambda * kp_loss
            # KD distillation on replayed samples
            if k > 0 and hasattr(self, '_teacher') and getattr(self, '_teacher', None) is not None and getattr(self, '_kd_alpha', 0.0) > 0.0:
                has_teacher_probs = all(len(replay_buffer[int(ii)]) > 2 and replay_buffer[int(ii)][2] is not None for ii in idx)
                if has_teacher_probs:
                    student_replay_logits = logits[b:]
                    student_log = F.log_softmax(student_replay_logits, dim=1)
                    teacher_probs = torch.stack([replay_buffer[int(ii)][2] for ii in idx]).to(self._device)
                    loss = loss + self._kd_alpha * F.kl_div(student_log, teacher_probs, reduction='batchmean')
            loss.backward()
            sub_optimizer.step()
            self._apply_l1_soft_thresholding_module(sub_net)

        # Extract trained sub-weights
        trained_sub: dict[str, torch.Tensor] = {}
        for i in range(self.n_hidden_layers):
            trained_sub[f"layer{i}/weight"] = sub_net.hidden_layers[i].weight.data.clone()
            trained_sub[f"layer{i}/bias"] = sub_net.hidden_layers[i].bias.data.clone()
        trained_sub["output/weight"] = sub_net.output_head.weight.data.clone()
        trained_sub["output/bias"] = sub_net.output_head.bias.data.clone()

        # Merge back
        shared_weights, head_w, head_b = merge_sub_network(
            shared_weights, head_w, head_b,
            active_indices, trained_sub, self.n_hidden_layers,
        )
        for i in range(self.n_hidden_layers):
            self.hidden_layers[i].weight.data = shared_weights[i]["weight"]
            self.hidden_layers[i].bias.data = shared_weights[i]["bias"]
        head = self.output_heads[f"task_{task_id}"]
        # delegate setting to head so tied/untied variants handle correctly
        try:
            head.set_weight_and_bias(head_w, head_b)
        except Exception:
            # fallback to direct assignment for compatibility
            head.weight.data = head_w
            head.bias.data = head_b

        val_loss, _ = self._evaluate(val_loader, task_id)
        return val_loss

    def _build_sub_network(
        self, sub_weights: dict[str, torch.Tensor], task_id: int,
    ) -> nn.Module:
        """Construct a simple MLP from sub-network weight dicts."""

        class SubNetwork(nn.Module):
            def __init__(self, hidden_layers, output_head, embedder=None):
                super().__init__()
                self.hidden_layers = hidden_layers
                self.output_head = output_head
                self.embedder = embedder

            def forward(self, x):
                if self.embedder is not None:
                    h = self.embedder(x).view(x.size(0), -1)
                else:
                    h = x.view(x.size(0), -1)
                for layer in self.hidden_layers:
                    h = F.relu(layer(h))
                return self.output_head(h)

        hidden = nn.ModuleList()
        for i in range(self.n_hidden_layers):
            w = sub_weights[f"layer{i}/weight"]
            lin = nn.Linear(w.size(1), w.size(0), bias=True)
            lin.weight.data = w
            lin.bias.data = sub_weights[f"layer{i}/bias"]
            hidden.append(lin)

        out_head = nn.Linear(
            sub_weights["output/weight"].size(1), self.num_classes, bias=True,
        )
        out_head.weight.data = sub_weights["output/weight"]
        out_head.bias.data = sub_weights["output/bias"]
        return SubNetwork(hidden, out_head, embedder=self.embedder)

    def _sub_net_kp_loss(
        self, sub_net: nn.Module, sub_weights: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self._device)
        for i, layer in enumerate(sub_net.hidden_layers):
            key = f"layer{i}/weight"
            if key in sub_weights:
                loss += 0.5 * (layer.weight - sub_weights[key].to(self._device)).pow(2).sum()
        return loss

    # ================================================================
    #  Dynamic Expansion
    # ================================================================

    def _dynamic_expansion(
        self,
        task_id: int,
        train_loader,
        val_loader,
        max_iter: int,
        lr: float,
        verbose: bool,
        expansion_info: list[int],
        replay_buffer: list | None = None,
    ):
        """Add *ex_k* new units to each layer, train with group-lasso,
        then prune dead units.  Matches the paper's *extend_bottom*
        and *extend_param* operations."""
        # Cooldown check: avoid repeated growth events
        if task_id - self._last_growth_task < self.growth_cooldown_tasks:
            return

        for i in range(self.n_hidden_layers):
            # Output expansion for every hidden layer
            self.hidden_layers[i].expand_output_units(self.ex_k, task_id)
            expansion_info[i] = self.ex_k
            # Input expansion for layers deeper than the first
            # (layer 0 receives from fixed-dim data).
            if i > 0:
                self.hidden_layers[i].expand_input_units(self.ex_k)

        # Expand output heads' input dimensions
        for head in self.output_heads.values():
            head.expand_input_units(self.ex_k)

        self.to(self._device)
        self._train_expanded_network(
            task_id, train_loader, val_loader, max_iter, lr, replay_buffer=replay_buffer,
        )
        n_pruned = self._prune_dead_units()
        if verbose:
            print(f"  [*] Pruned {n_pruned} dead units")
        # Mark growth event so subsequent tasks observe cooldown
        if n_pruned >= 0:
            self._last_growth_task = task_id

    def _train_expanded_network(
        self,
        task_id: int,
        train_loader,
        val_loader,
        max_iter: int,
        lr: float,
        replay_buffer: list | None = None,
    ):
        optimizer = self._make_optimizer(lr, task_id=task_id)
        prev_slices = self._get_prev_weight_slices()
        # Track gradient norm ratios for gradient_imbalance criterion
        _grad_norms: list[float] = []

        for it, (x, y) in enumerate(train_loader):
            max_iter = min(max_iter, len(train_loader))
            if it >= max_iter:
                break
            x, y = x.to(self._device), y.to(self._device)
            x = x.view(x.size(0), -1)

            optimizer.zero_grad()
            logits = self(x, task_id=task_id)
            loss = F.cross_entropy(logits, y)
            loss += self.l2_lambda * 0.5 * sum(
                p.pow(2).sum() for p in self.parameters()
            )
            loss += knowledge_preservation_loss(self, self.regular_lambda, prev_slices)
            loss.backward()
            # Gradient masking/scaling: damp old-unit gradients and amplify
            # newly-added-unit gradients so new neurons learn faster while
            # preserving prior knowledge.
            for li, layer in enumerate(self.hidden_layers):
                if layer.weight.grad is None:
                    continue
                n_new = min(self.ex_k, layer.out_features)
                if n_new <= 0:
                    continue
                # Zero grads for rows corresponding to old units
                keep_start = layer.out_features - n_new
                if keep_start > 0:
                    # allow small adaptation of old units by scaling their gradients
                    layer.weight.grad[:keep_start, :] *= 0.05
                    layer.bias.grad[:keep_start] *= 0.05
                # Amplify gradients for newly-added rows so they learn faster
                if n_new > 0:
                    new_mult = 3.0
                    layer.weight.grad[keep_start:, :] *= new_mult
                    layer.bias.grad[keep_start:] *= new_mult
                # For the next layer, only allow grads into the new input cols
                if li < len(self.hidden_layers) - 1:
                    nxt = self.hidden_layers[li + 1]
                    if nxt.weight.grad is not None and nxt.in_features >= n_new:
                        # scale older input-col grads, amplify new input-col grads
                        cut = nxt.in_features - n_new
                        if cut > 0:
                            nxt.weight.grad[:, :cut] *= 0.05
                        nxt.weight.grad[:, cut:] *= 3.0
                else:
                    # Output heads: zero old input cols (only allow new cols)
                    for head in self.output_heads.values():
                        if getattr(head, 'tie', False):
                            if getattr(head, 'proj').grad is not None and head.in_features >= n_new:
                                cut = head.in_features - n_new
                                if cut > 0:
                                    head.proj.grad[:, :cut] *= 0.05
                                head.proj.grad[:, cut:] *= 3.0
                        else:
                            if head.weight.grad is not None and head.in_features >= n_new:
                                cut = head.in_features - n_new
                                if cut > 0:
                                    head.weight.grad[:, :cut] *= 0.05
                                head.weight.grad[:, cut:] *= 3.0

            optimizer.step()

            self._apply_l1_soft_thresholding()
            self._apply_group_lasso_step()

            # Track gradient norm ratio (shallowest / deepest)
            if self.depth_growth_enabled and len(self.hidden_layers) >= 2:
                g_norms = []
                for layer in self.hidden_layers:
                    if layer.weight.grad is not None:
                        g_norms.append(layer.weight.grad.norm().item())
                if len(g_norms) >= 2:
                    ratio = max(g_norms) / (min(g_norms) + 1e-8)
                    _grad_norms.append(ratio)

        if _grad_norms:
            avg_ratio = sum(_grad_norms) / len(_grad_norms)
            self.depth_growth_tracker.setdefault("grad_norm_ratios", []).append(
                avg_ratio
            )

    def _warmup_new_layer(self, task_id: int, train_loader, n_iter: int = 150, lr: float = 5e-4):
        """Briefly train only the newly-inserted layer on the current task.

        Freezes all hidden layers except the newly-inserted one (and the
        layer after it, if one exists) and runs a few quick SGD steps.
        """
        if not self.hidden_layers:
            return
        insert_idx = self.n_hidden_layers - 1  # layers are appended at the end
        if insert_idx < 0:
            return
        new_layer = self.hidden_layers[insert_idx]
        layers_to_train = [new_layer]
        if insert_idx + 1 < self.n_hidden_layers:
            layers_to_train.append(self.hidden_layers[insert_idx + 1])

        frozen = []
        for i, layer in enumerate(self.hidden_layers):
            if layer not in layers_to_train:
                frozen.append(layer)
                for p in layer.parameters():
                    p.requires_grad_(False)

        warm_params = [
            p for layer in layers_to_train for p in layer.parameters()
            if p.requires_grad
        ]
        if not warm_params:
            for layer in frozen:
                for p in layer.parameters():
                    p.requires_grad_(True)
            return

        opt = torch.optim.Adam(warm_params, lr=lr)
        l2_lambda = self.l2_lambda
        iters = 0
        for x, y in train_loader:
            if iters >= n_iter:
                break
            x = x.to(self._device)
            y = y.to(self._device)
            opt.zero_grad()
            logits = self(x, task_id=task_id)
            loss = F.cross_entropy(logits, y)
            if l2_lambda:
                l2 = sum(p.pow(2).sum() for p in warm_params)
                loss = loss + l2_lambda * 0.5 * l2
            loss.backward()
            opt.step()
            iters += 1

        for layer in frozen:
            for p in layer.parameters():
                p.requires_grad_(True)

    def _prune_dead_units(self) -> int:
        """
        Remove newly-added units that were zeroed out by group-lasso,
        scanning from the last hidden layer back to the first so that
        the connection book-keeping is consistent.
        """
        total = 0
        for layer_idx in range(self.n_hidden_layers - 1, -1, -1):
            layer = self.hidden_layers[layer_idx]
            n_new = min(self.ex_k, layer.out_features)
            dead = find_useless_new_units(layer.weight.data, n_new)
            if not dead:
                continue
            # Clamp indices to valid range to guard against any prior
            # dimension inconsistency (e.g. from expand/prune interleaving).
            feat = layer.out_features
            dead = [d for d in dead if 0 <= d < feat]
            if not dead:
                continue
            keep = torch.ones(feat, dtype=torch.bool, device=self._device)
            keep[dead] = False
            # Prune this layer's output units
            self._prune_layer_output(layer_idx, keep)
            total += len(dead)
        return total

    def _consolidate(self):
        """Run a short consolidation pass: stronger group-lasso steps
        followed by pruning of low-norm neurons."""
        old_gl = self.gl_lambda
        try:
            self.gl_lambda = self.consolidation_gl
            for _ in range(self.consolidation_iters):
                # apply group-lasso proximal step
                self._apply_group_lasso_step()
            # prune units with small row norms across all hidden layers
            total_pruned = 0
            for li in range(self.n_hidden_layers - 1, -1, -1):
                layer = self.hidden_layers[li]
                row_norms = layer.weight.data.norm(dim=1)
                to_prune = (row_norms < self.consolidation_row_norm_thr).nonzero(as_tuple=True)[0]
                if len(to_prune) == 0:
                    continue
                keep = torch.ones(layer.out_features, dtype=torch.bool, device=self._device)
                keep[to_prune] = False
                self._prune_layer_output(li, keep)
                total_pruned += len(to_prune)
            if total_pruned > 0:
                # re-save anchors and prev params
                self._save_prev_params()
        finally:
            self.gl_lambda = old_gl

    def _prune_layer_output(self, layer_idx: int, keep: torch.Tensor):
        """Remove output neurons where keep == False."""
        layer = self.hidden_layers[layer_idx]
        layer.weight.data = layer.weight.data[keep]
        layer.bias.data = layer.bias.data[keep]
        layer.timestamp = layer.timestamp[keep]
        layer.weight_anchor = layer.weight_anchor[keep]
        layer.bias_anchor = layer.bias_anchor[keep]
        removed = (~keep).sum().item()
        layer.out_features -= removed

        # Adjust the next layer's input dimension
        if layer_idx < self.n_hidden_layers - 1:
            nxt = self.hidden_layers[layer_idx + 1]
            nxt.weight.data = nxt.weight.data[:, keep]
            nxt.weight_anchor = nxt.weight_anchor[:, keep]
            nxt.in_features -= removed
        else:
            for head in self.output_heads.values():
                head.prune_input_columns(keep)

    # ================================================================
    #  Split & Duplication
    # ================================================================

    def _split_and_duplicate(
        self,
        task_id: int,
        train_loader,
        val_loader,
        max_iter: int,
        lr: float,
        verbose: bool,
        expansion_info: list[int],
        replay_buffer: list | None = None,
    ):
        """
        Detect drifted neurons and split them.  For each layer we
        compute the per-neuron drift (L2 change in incoming weight
        vector), select the top ``ex_k`` drifted neurons, and
        duplicate each into an old (preserved) and new (trainable)
        copy.
        """
        split_counts: list[int] = []

        for layer_idx in range(self.n_hidden_layers):
            layer = self.hidden_layers[layer_idx]
            w_prev = self.prev_params.get(
                f"hidden_layers.{layer_idx}.weight", layer.weight.data
            ).to(self._device)
            b_prev = self.prev_params.get(
                f"hidden_layers.{layer_idx}.bias", layer.bias.data
            ).to(self._device)

            # Clamp to the overlapping region (previous size ≤ current size).
            min_out = min(w_prev.size(0), layer.weight.size(0))
            min_in = min(w_prev.size(1), layer.weight.size(1))
            w_prev_clamped = w_prev[:min_out, :min_in]
            b_prev_clamped = b_prev[:min_out]
            cur_clamped = layer.weight.data[:min_out, :min_in]

            # Drift per output neuron: ||prev[j,:] - cur[j,:]||_2
            drift_per_unit = (w_prev_clamped - cur_clamped).norm(p=2, dim=1)

            # If the current layer has more columns (from input expansion
            # during dynamic expansion), pad prev_weight with zeros so
            # split_output_units receives consistent dimensions.
            split_prev = w_prev_clamped
            if w_prev_clamped.size(1) < layer.weight.size(1):
                pad = torch.zeros(
                    min_out,
                    layer.weight.size(1) - w_prev_clamped.size(1),
                    device=w_prev_clamped.device,
                )
                split_prev = torch.cat([w_prev_clamped, pad], dim=1)
            above_mask = drift_per_unit > self.spl_thr
            drifted = above_mask.nonzero(as_tuple=True)[0]

            if len(drifted) == 0:
                split_counts.append(0)
                continue

            # Take top *ex_k* by drift magnitude
            drift_vals = drift_per_unit[drifted]
            top_k = min(self.ex_k, len(drifted))
            top_indices = drifted[drift_vals.argsort(descending=True)[:top_k]].tolist()

            n_split = layer.split_output_units(
                top_indices, split_prev, b_prev_clamped, task_id,
            )
            split_counts.append(n_split)
            expansion_info[layer_idx] += n_split

            # The next layer needs extra input columns
            if layer_idx < self.n_hidden_layers - 1:
                self.hidden_layers[layer_idx + 1].expand_input_units(n_split)
            else:
                for head in self.output_heads.values():
                    head.expand_input_units(n_split)

        if sum(split_counts) > 0:
            self.to(self._device)
            self._train_after_split(
                task_id, train_loader, val_loader, max_iter, lr, replay_buffer=replay_buffer,
            )
            # record split as a growth-like event
            self._last_growth_task = task_id

        if verbose:
            print(f"  [*] Split counts per layer: {split_counts}")

    def _train_after_split(
        self,
        task_id: int,
        train_loader,
        val_loader,
        max_iter: int,
        lr: float,
        replay_buffer: list | None = None,
    ):
        optimizer = self._make_optimizer(lr, task_id=task_id)
        prev_slices = self._get_prev_weight_slices()
        data_iter = _make_cycling_iter(train_loader)

        for _ in range(max_iter):
            x, y = next(data_iter)
            x, y = x.to(self._device), y.to(self._device)
            b = x.size(0)
            x = x.view(x.size(0), -1)

            # Mix replay samples to reduce forgetting during post-split training
            k = 0
            idx = None
            if replay_buffer and len(replay_buffer) > 0:
                k = max(1, min(len(replay_buffer), int(self.replay_fraction * x.size(0))))
                idx = torch.randperm(len(replay_buffer))[:k]
                rx_list, ry_list = [], []
                for ii in idx:
                    entry = replay_buffer[int(ii)]
                    rx_list.append(entry[0])
                    ry_list.append(entry[1])
                rx = torch.stack(rx_list).to(self._device)
                ry = torch.stack(ry_list).to(self._device)
                rx = rx.view(rx.size(0), -1)
                x = torch.cat([x, rx], dim=0)
                y = torch.cat([y, ry], dim=0)

            optimizer.zero_grad()
            logits = self(x, task_id=task_id)
            loss = F.cross_entropy(logits, y)
            loss += self.l2_lambda * 0.5 * sum(
                p.pow(2).sum() for p in self.parameters()
            )
            loss += knowledge_preservation_loss(self, self.regular_lambda, prev_slices)
            if k > 0 and hasattr(self, '_teacher') and getattr(self, '_teacher', None) is not None and getattr(self, '_kd_alpha', 0.0) > 0.0:
                has_teacher_probs = all(len(replay_buffer[int(ii)]) > 2 and replay_buffer[int(ii)][2] is not None for ii in idx)
                if has_teacher_probs:
                    student_replay_logits = logits[b:]
                    student_log = F.log_softmax(student_replay_logits, dim=1)
                    teacher_probs = torch.stack([replay_buffer[int(ii)][2] for ii in idx]).to(self._device)
                    loss = loss + self._kd_alpha * F.kl_div(student_log, teacher_probs, reduction='batchmean')
            loss.backward()
            optimizer.step()
            self._apply_l1_soft_thresholding()

    # ================================================================
    #  Regularisation helpers
    # ================================================================

    def _apply_l1_soft_thresholding(self):
        if self.l1_lambda == 0:
            return
        with torch.no_grad():
            for name, p in self.named_parameters():
                if name.startswith("embedder."):
                    continue
                th = self.l1_lambda
                p.data = torch.sign(p.data) * torch.clamp(p.data.abs() - th, min=0.0)

    def _apply_l1_soft_thresholding_module(self, module: nn.Module):
        if self.l1_lambda == 0:
            return
        with torch.no_grad():
            for name, p in module.named_parameters():
                if name.startswith("embedder."):
                    continue
                th = self.l1_lambda
                p.data = torch.sign(p.data) * torch.clamp(p.data.abs() - th, min=0.0)

    def _apply_group_lasso_step(self):
        if self.gl_lambda == 0:
            return
        with torch.no_grad():
            for layer in self.hidden_layers:
                layer.weight.data = group_lasso_step(
                    layer.weight.data, self.gl_lambda,
                )

    def _get_prev_weight_slices(self) -> dict[str, tuple[torch.Tensor, tuple]]:
        return get_prev_weight_slices(self, self.prev_params)

    def _make_optimizer(self, lr: float, task_id: int | None = None,
                        new_lr_mult: float = 3.0, old_lr_mult: float = 0.2):
        """Construct an optimizer with param-groups.

        Any hidden layer that contains neurons with timestamp == task_id
        (i.e. units added during this task) will have its parameters placed
        in the "new" group and receive a larger LR. The current task's
        output head is also treated as new. Remaining parameters use a
        reduced LR to better preserve prior knowledge.
        """
        fresh_params = []
        old_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            assigned = False
            # Hidden layer params: named like 'hidden_layers.<i>.weight'
            if name.startswith("hidden_layers."):
                parts = name.split('.')
                try:
                    idx = int(parts[1])
                except Exception:
                    idx = None
                if idx is not None and 0 <= idx < len(self.hidden_layers):
                    layer = self.hidden_layers[idx]
                    # If any unit in this layer has timestamp == task_id,
                    # treat the whole layer as new (coarse-grained grouping).
                    if task_id is not None and (layer.timestamp == task_id).any():
                        fresh_params.append(p)
                        assigned = True
            # Output heads: named like 'output_heads.<task_k>.weight'
            if not assigned and name.startswith("output_heads."):
                parts = name.split('.')
                if len(parts) >= 2:
                    key = parts[1]
                    # current task head considered fresh
                    if task_id is not None and key == f"task_{task_id}":
                        fresh_params.append(p)
                        assigned = True
            if not assigned:
                old_params.append(p)

        param_groups = []
        if fresh_params:
            param_groups.append({"params": fresh_params, "lr": lr * new_lr_mult})
        if old_params:
            param_groups.append({"params": old_params, "lr": lr * old_lr_mult})

        if not param_groups:
            return torch.optim.Adam(self.parameters(), lr=lr)
        return torch.optim.Adam(param_groups)

    def _save_prev_params(self):
        self.prev_params = {
            name: param.data.clone().cpu()
            for name, param in self.named_parameters()
        }
        # Also store anchors on layers/heads so future knowledge-preservation
        # can reference stable targets (L2-SP style). Update buffer copies.
        for layer in self.hidden_layers:
            try:
                layer.weight_anchor = layer.weight.data.clone()
                layer.bias_anchor = layer.bias.data.clone()
            except Exception:
                pass
        for head in self.output_heads.values():
            try:
                if getattr(head, 'tie', False):
                    head.proj_anchor = head.proj.data.clone()
                    head.bias_anchor = head.bias.data.clone()
                else:
                    head.weight_anchor = head.weight.data.clone()
                    head.bias_anchor = head.bias.data.clone()
            except Exception:
                pass

    def _save_timestamps(self, task_id: int):
        stamp: list[int] = []
        for layer in self.hidden_layers:
            stamp.append(layer.out_features)
        head = self.output_heads[f"task_{task_id}"]
        stamp.append(head.num_classes)
        self.timestamps[task_id] = stamp

    def _add_output_head(self, task_id: int):
        key = f"task_{task_id}"
        if key not in self.output_heads:
            prev_dim = self.hidden_layers[-1].out_features
            self.output_heads[key] = TaskOutputHead(
                prev_dim, self.num_classes, task_id,
                embedder=(self.embedder if getattr(self, 'embedder', None) is not None else None),
                tie=getattr(self, 'tie_embeddings', False),
            )

    def _log_architecture(
        self, task_id: int,
        expansion_info: tuple[int, ...],
        depth_inserted: bool = False,
    ):
        """Record the current network architecture for post-hoc analysis."""
        arch = get_architecture_summary(self)
        arch["task_id"] = task_id
        arch["expansion_info"] = expansion_info
        arch["depth_inserted"] = depth_inserted
        self.architecture_log.append(arch)

    # ================================================================
    #  Evaluation
    # ================================================================

    def _evaluate(self, loader, task_id: int) -> tuple[float, float]:
        self.eval()
        total_loss = 0.0
        preds_list, labels_list = [], []
        total_ce = 0.0
        topk_counts = {1: 0, 5: 0, 10: 0}
        total_samples = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self._device), y.to(self._device)
                x = x.view(x.size(0), -1)
                logits = self(x, task_id=task_id)
                # Primary training loss: cross-entropy for multiclass next-token prediction
                loss = F.cross_entropy(logits, y)
                total_loss += loss.item() * x.size(0)

                # For language-model style metrics compute cross-entropy / perplexity
                # y is class indices (LongTensor)
                true_idx = y
                # cross-entropy expects logits and class indices
                ce = F.cross_entropy(logits, true_idx, reduction='sum')
                total_ce += ce.item()
                # Top-k accuracy from logits
                for k in topk_counts.keys():
                    topk = torch.topk(logits, k=k, dim=1).indices
                    matches = (topk == true_idx.unsqueeze(1)).any(dim=1).sum().item()
                    topk_counts[k] += matches
                total_samples += x.size(0)

                # store softmax probs for downstream accuracy computation
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds_list.append(probs)
                labels_list.append(true_idx.cpu().numpy())
        self.train()
        avg_loss = total_loss / len(loader.dataset)
        all_preds = np.concatenate(preds_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
        # Cross-entropy and perplexity
        avg_ce = total_ce / max(1, len(loader.dataset))
        try:
            perplexity = math.exp(avg_ce)
        except OverflowError:
            perplexity = float('inf')

        # Top-k accuracies
        topk_acc = {k: (topk_counts[k] / max(1, total_samples)) for k in topk_counts}

        # Compute accuracy from predicted probabilities and true indices
        preds_idx = np.argmax(all_preds, axis=1)
        acc = float((preds_idx == all_labels).mean()) if all_labels.size > 0 else 0.0

        # Print additional metrics for transparency
        print(
            f"  [*] Eval metrics (task {task_id}): avg_bce={avg_loss:.6f}, acc={acc:.6f}, "
            f"avg_ce={avg_ce:.6f}, ppl={perplexity:.2f}, top1={topk_acc[1]:.4f}, top5={topk_acc[5]:.4f}, top10={topk_acc[10]:.4f}"
        )

        return avg_loss, acc

    def evaluate_task(self, loader, task_id: int) -> tuple[float, float]:
        return self._evaluate(loader, task_id)

    def _avg_sparsity(self) -> float:
        total = 0.0
        zeros = 0.0
        for p in self.parameters():
            total += p.numel()
            zeros += (p == 0).sum().item()
        return (zeros + 1) / (total + 1) if total > 0 else 0.0

    def get_neuron_counts(self) -> list[int]:
        """Return number of output neurons per hidden layer."""
        return [layer.out_features for layer in self.hidden_layers]
