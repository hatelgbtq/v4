"""
DEN training orchestrator.

Manages the multi-task continual-learning loop:
  - Iterates over tasks sequentially.
  - For each task, calls ``model.add_task(...)`` which internally
    runs selective retraining, expansion, and splitting.
  - Evaluates on *all* tasks seen so far (forward transfer / forgetting).
  - Logs metrics and optionally writes visualisation data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.den import DEN
from models.utils import accuracy


class Trainer:
    """
    High-level trainer that runs the DEN lifelong-learning loop.

    Parameters
    ----------
    model : DEN
    device : torch.device
    log_dir : str or Path
        Directory for logs, plots, and checkpoints.
    resume_from : str or Path or None
        Path to a DEN checkpoint.  If the file exists, the model is
        rebuilt from it and training continues with the next task
        instead of restarting from task 0.
    """

    def __init__(
        self,
        model: DEN,
        device: torch.device,
        log_dir: str | Path = "./results",
        resume_from: str | Path | None = None,
        dataset_cfg: dict | None = None,
    ):
        self.model = model
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_cfg = dataset_cfg or {}

        self.history: dict[str, list] = {
            "test_acc": [],
            "avg_forgetting": [],
            "sparsity": [],
            "neuron_counts": [],
            "neurons_added": [],
            "neurons_split": [],
        }

        # Simple rehearsal buffer: store a small fixed number of examples
        # per task to replay during training of future tasks.
        # Stored as tuples (task_id, x, y)
        self.replay_buffer: list[tuple] = []
        self.replay_per_task: int = 100
        # Reservoir + coreset settings
        self.replay_capacity: int = 2000
        self.coreset_size: int = 800
        self._replay_seen: int = 0

        # Resume: swap in the checkpointed model and restore metrics.
        self.checkpoint_path = Path(resume_from) if resume_from else None
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            self.model = DEN.load_checkpoint(self.checkpoint_path)
            self.model.to(device)
            self._load_history()
            print(
                f"  [*] Resumed from {self.checkpoint_path}: "
                f"{len(self.model.task_list)} task(s) already trained."
            )

    def train(
        self,
        train_loaders: list,
        val_loaders: list,
        test_loaders: list,
        max_iter: int,
        lr: float,
        batch_size: int,
        verbose: bool = True,
    ) -> dict[str, list]:
        """
        Run the full continual-learning sequence.

        Parameters
        ----------
        train_loaders, val_loaders, test_loaders : list[DataLoader]
            One loader per task.
        max_iter : int
            Training iterations per task.
        lr : float
            Learning rate.
        batch_size : int
            Batch size.
        verbose : bool
            Print progress.

        Returns
        -------
        history : dict
            Logged metrics.
        """
        num_tasks = len(train_loaders)
        start_task = len(self.model.task_list)
        all_test_accs: list[list[float]] = (
            _to_native(self.history["test_acc"]) if self.history["test_acc"] else []
        )

        for task_id in range(start_task, num_tasks):
            if verbose:
                print(f"\n{'='*60}")
                print(f"  TASK {task_id + 1} / {num_tasks}")
                print(f"{'='*60}")

            t0 = time.time()

            train_loader = train_loaders[task_id]
            val_loader = val_loaders[task_id]
            test_loader = test_loaders[task_id]

            # Pass a balanced replay pool (sampled across tasks)
            # Snapshot teacher model for KD distillation (lightweight copy)
            import copy
            teacher_model = copy.deepcopy(self.model)
            teacher_model.eval()

            replay_pool = self._get_replay_pool(teacher_model)
            test_acc, sparsity, expansion_info = self.model.add_task(
                task_id=task_id,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                max_iter=max_iter,
                lr=lr,
                batch_size=batch_size,
                device=self.device,
                verbose=verbose,
                replay_buffer=replay_pool,
                teacher_model=teacher_model,
                kd_alpha=getattr(self, 'kd_alpha', 0.5),
            )

            # After training this task, store a small rehearsal set.
            self._update_replay(train_loader, task_id, teacher_model=teacher_model)

            # --- Context growth: rebuild loaders for remaining tasks ---
            if getattr(self.model, '_context_grew', False):
                self.model._context_grew = False
                new_ctx = self.model.current_context
                if verbose:
                    print(f"\n  [*] Rebuilding data loaders for new context={new_ctx}")
                # Clear replay buffer — old samples have wrong context size
                self.replay_buffer.clear()
                self._replay_seen = 0
                remaining = num_tasks - task_id - 1
                if remaining > 0 and self.dataset_cfg:
                    from datasets.text import get_text_loaders
                    new_train, new_val, new_test, _ = get_text_loaders(
                        data_path=self.dataset_cfg["data_path"],
                        num_tasks=self.dataset_cfg.get("num_tasks", num_tasks),
                        batch_size=self.dataset_cfg.get("batch_size", 256),
                        context=new_ctx,
                        emb_dim=self.dataset_cfg.get("emb_dim", 64),
                        vocab_size=self.dataset_cfg.get("vocab_size", 1000),
                        stories_per_task=self.dataset_cfg.get("stories_per_task"),
                        seed=self.dataset_cfg.get("seed", 1004),
                    )
                    # Replace loaders for future tasks
                    for i in range(task_id + 1, num_tasks):
                        train_loaders[i] = new_train[i]
                        val_loaders[i] = new_val[i]
                        test_loaders[i] = new_test[i]

                # Clear any cached probe batch (it may have the old context size)
                try:
                    if "probe_batch" in self.model.depth_growth_tracker:
                        del self.model.depth_growth_tracker["probe_batch"]
                except Exception:
                    pass

            elapsed = time.time() - t0

            # Evaluate on *all* tasks seen so far
            task_accs: list[float] = []
            for eval_tid in range(task_id + 1):
                eval_loader = test_loaders[eval_tid]
                _, acc_val = self.model.evaluate_task(eval_loader, eval_tid)
                task_accs.append(acc_val)

            all_test_accs.append(task_accs)

            # Forgetting: per-task drop from best-so-far accuracy
            forgetting = self._compute_forgetting(all_test_accs)

            # Log
            self.history["test_acc"].append(task_accs)
            self.history["avg_forgetting"].append(float(np.mean(forgetting)) if forgetting else 0.0)
            self.history["sparsity"].append(sparsity)
            self.history["neuron_counts"].append(self.model.get_neuron_counts())
            self.history["neurons_added"].append(sum(expansion_info))
            self.history["neurons_split"].append(
                sum(self.model.get_neuron_counts()) - self._prev_total_neurons()
            )
            self._prev_total = sum(self.model.get_neuron_counts())

            # Depth-growth tracking
            n_layers = self.model.n_hidden_layers
            self.history.setdefault("n_hidden_layers", []).append(n_layers)
            self.history.setdefault("depth_insertions", []).append(
                self.model.architecture_log[-1].get("depth_inserted", False)
                if self.model.architecture_log else False
            )

            # Persist after every task so a later run can resume here
            cp = self.log_dir / "checkpoint.pt"
            self.model.save_checkpoint(cp)
            self._save_metrics()
            if verbose:
                print(f"\n  [*] Checkpoint saved to {cp} (resume with --resume)")

            if verbose:
                print(f"\n  [*] Results after Task {task_id + 1}:")
                for j, acc in enumerate(task_accs):
                    print(f"      Task {j + 1} test acc: {acc:.4f}")
                print(f"      Avg forgetting: {self.history['avg_forgetting'][-1]:.4f}")
                print(f"      Neurons per layer: {self.model.get_neuron_counts()}")
                print(f"      Hidden layers: {n_layers}")
                print(f"      Sparsity: {sparsity:.4f}")
                print(f"      Time: {elapsed:.1f}s")

        # Final summary
        if verbose:
            self._print_summary(all_test_accs)

        # Save metrics
        self._save_metrics()

        return self.history

    def _compute_forgetting(
        self, all_test_accs: list[list[float]],
    ) -> list[float]:
        """
        Forgetting for task i = max over previous checkpoints of
        (acc_on_task_i_at_its_best - acc_on_task_i_now).
        """
        num_tasks = len(all_test_accs)
        if num_tasks < 2:
            return []
        forgetting = []
        for j in range(num_tasks - 1):
            best = max(all_test_accs[k][j] for k in range(j, num_tasks))
            current = all_test_accs[-1][j]
            forgetting.append(best - current)
        return forgetting

    def _prev_total_neurons(self) -> int:
        return getattr(self, "_prev_total", 0)

    def _load_history(self):
        """Reload metrics.json from a previous run so logging continues."""
        path = self.log_dir / "metrics.json"
        if not path.exists():
            self._prev_total = sum(self.model.get_neuron_counts())
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._prev_total = sum(self.model.get_neuron_counts())
            return
        for k, v in data.items():
            if k in self.history:
                self.history[k] = _to_native(v)
        self._prev_total = sum(self.model.get_neuron_counts())

    def _update_replay(self, train_loader, task_id: int, teacher_model=None):
        """Sample up to `replay_per_task` examples from the given train
        loader and append them (on CPU) to the replay buffer."""
        # Use reservoir sampling to maintain a fixed-capacity buffer
        import random
        for x, y in train_loader:
            for i in range(x.size(0)):
                # increment seen counter for this example
                self._replay_seen += 1
                xi = x[i].cpu()
                yi = y[i].cpu()
                # compute teacher probs if teacher_model given
                tp = None
                if teacher_model is not None:
                    try:
                        with torch.no_grad():
                            # forward with teacher using this task's head
                            tlogit = teacher_model(x[i].unsqueeze(0).to(self.device), task_id=task_id)
                            tp = torch.softmax(tlogit, dim=-1).squeeze(0).cpu()
                    except Exception:
                        tp = None

                entry = (task_id, xi, yi) if tp is None else (task_id, xi, yi, tp)

                if len(self.replay_buffer) < self.replay_capacity:
                    self.replay_buffer.append(entry)
                else:
                    j = random.randint(0, self._replay_seen - 1)
                    if j < self.replay_capacity:
                        self.replay_buffer[j] = entry
        # Optionally cap overall buffer size to keep memory bounded
        cap = max(1000, self.replay_per_task * len(self.model.task_list))
        if len(self.replay_buffer) > cap:
            # keep most recent examples
            self.replay_buffer = self.replay_buffer[-cap:]

    def _get_replay_pool(self, teacher_model=None):
        """Return a balanced list of (x,y) pairs sampled across stored tasks.

        Ensures each past task contributes roughly equally to the pool.
        """
        if not self.replay_buffer:
            return []

        # If no teacher provided, return k-center coreset without teacher probs
        use_teacher = teacher_model is not None

        # If buffer is small, return everything (attach teacher probs if available)
        if len(self.replay_buffer) <= self.coreset_size:
            if not use_teacher:
                return [(x, y) for (_tid, x, y) in self.replay_buffer]
            pool = []
            for (tid, x, y) in self.replay_buffer:
                tp = None
                try:
                    with torch.no_grad():
                        # teacher provides a full-class distribution (softmax)
                        tp = torch.softmax(teacher_model(x.unsqueeze(0), task_id=tid), dim=-1).squeeze(0).cpu()
                except Exception:
                    tp = None
                pool.append((x, y, tp))
            return pool

        # Greedy k-center coreset selection on flattened inputs
        import numpy as np
        import random

        X = [x.numpy().ravel() for (_tid, x, y) in self.replay_buffer]
        Y = [y for (_tid, x, y) in self.replay_buffer]
        n = len(X)
        k = min(self.coreset_size, n)
        # initialize
        idxs = [random.randrange(n)]
        dists = np.linalg.norm(np.stack(X) - X[idxs[0]], axis=1)
        for _ in range(1, k):
            next_idx = int(dists.argmax())
            idxs.append(next_idx)
            newd = np.linalg.norm(np.stack(X) - X[next_idx], axis=1)
            dists = np.minimum(dists, newd)

        if not use_teacher:
            pool = [(self.replay_buffer[i][1], self.replay_buffer[i][2]) for i in idxs]
            return pool
        pool = []
        for i in idxs:
            tid, x, y = self.replay_buffer[i]
            try:
                with torch.no_grad():
                    tp = torch.softmax(teacher_model(x.unsqueeze(0), task_id=tid), dim=-1).squeeze(0).cpu()
            except Exception:
                tp = None
            pool.append((x, y, tp))
        return pool

    def _print_summary(self, all_test_accs: list[list[float]]):
        print(f"\n{'='*60}")
        print("  FINAL SUMMARY")
        print(f"{'='*60}")
        num_tasks = len(all_test_accs)
        for j in range(num_tasks):
            accs = [all_test_accs[k][j] for k in range(j, num_tasks)]
            print(f"  Task {j + 1}: best={max(accs):.4f}  final={accs[-1]:.4f}")

    def _save_metrics(self):
        out = self.log_dir / "metrics.json"
        # Convert numpy values to native Python
        serializable = {}
        for k, v in self.history.items():
            serializable[k] = _to_native(v)
        serializable["architecture_log"] = _to_native(self.model.architecture_log)
        serializable["n_tasks_trained"] = len(self.model.task_list)
        with open(out, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  [*] Metrics saved to {out}")


def _to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [_to_native(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
