#!/usr/bin/env python3
"""
Benchmark: Original DEN vs DEN + Depth Growth.

Runs both models on 4-task Permuted MNIST at 200 iterations/task
for multiple random seeds, then prints a comparison table.

Usage:
    python benchmark.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ORIGINAL_DIR = BASE / ".." / "DEN"
DEPTH_DIR = BASE
RESULTS_DIR = BASE / "results" / "depth_vs_width"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [1004, 1005, 1006]
TASKS = 4
MAX_ITER = 200

CONFIG = dict(
    dataset="permuted_mnist",
    num_tasks=TASKS,
    max_iter=MAX_ITER,
    lr=0.001,
    batch_size=256,
    log_dir="results/depth_vs_width",
)


def run_original(seed: int) -> dict:
    """Run original DEN, return parsed metrics."""
    log = RESULTS_DIR / f"original_seed{seed}"
    cmd = [
        sys.executable, str(ORIGINAL_DIR / "train.py"),
        "--dataset", CONFIG["dataset"],
        "--num-tasks", str(CONFIG["num_tasks"]),
        "--max-iter", str(CONFIG["max_iter"]),
        "--lr", str(CONFIG["lr"]),
        "--batch-size", str(CONFIG["batch_size"]),
        "--log-dir", str(log),
        "--seed", str(seed),
        "--no-plots",
    ]
    env = {"PYTHONPATH": str(ORIGINAL_DIR)}
    t0 = time.time()
    subprocess.run(cmd, check=True, capture_output=True, env=env, cwd=str(ORIGINAL_DIR))
    elapsed = time.time() - t0
    metrics = _load_metrics(log)
    metrics["training_time"] = elapsed
    return metrics


def run_depth(seed: int) -> dict:
    """Run DEN + Depth Growth, return parsed metrics."""
    log = RESULTS_DIR / f"depth_seed{seed}"
    cmd = [
        sys.executable, str(DEPTH_DIR / "train.py"),
        "--dataset", CONFIG["dataset"],
        "--num-tasks", str(CONFIG["num_tasks"]),
        "--max-iter", str(CONFIG["max_iter"]),
        "--lr", str(CONFIG["lr"]),
        "--batch-size", str(CONFIG["batch_size"]),
        "--log-dir", str(log),
        "--seed", str(seed),
        "--no-plots",
        "--depth-growth-enabled",
        "--depth-growth-interval", "3",
    ]
    env = {"PYTHONPATH": str(DEPTH_DIR)}
    t0 = time.time()
    subprocess.run(cmd, check=True, capture_output=True, env=env, cwd=str(DEPTH_DIR))
    elapsed = time.time() - t0
    metrics = _load_metrics(log)
    metrics["training_time"] = elapsed
    return metrics


def _load_metrics(log_dir: Path) -> dict:
    """Load metrics.json from a run directory."""
    path = Path(log_dir) / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _test_acc_last_row(metrics: dict) -> list[float]:
    """Return the final accuracy for each task (last row of test_acc)."""
    acc = metrics.get("test_acc", [])
    if acc and len(acc) > 0:
        return acc[-1]
    return []


def _calc_total_params(neuron_counts: list[list[int]], n_tasks: int, input_dim: int = 784, num_classes: int = 10) -> int:
    """Estimate total parameters (weights + biases)."""
    if not neuron_counts:
        return 0
    last = neuron_counts[-1]
    params = 0
    prev = input_dim
    for out in last:
        params += prev * out + out
        prev = out
    params += n_tasks * (prev * num_classes + num_classes)
    return params


def avg(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def compute_summary(metrics_list: list[dict]) -> dict:
    """Average metrics over multiple seeds."""
    n_tasks = max(len(_test_acc_last_row(m)) for m in metrics_list) if metrics_list else 0
    n_seeds = len(metrics_list)

    # Per-task final accuracy averaged across seeds
    per_task_avg = []
    for t in range(n_tasks):
        vals = [m for m in metrics_list if t < len(_test_acc_last_row(m))]
        per_task_avg.append(avg([_test_acc_last_row(v)[t] for v in vals]))

    # Overall average accuracy
    all_accs = [a for m in metrics_list for a in _test_acc_last_row(m)]
    overall_avg_acc = avg(all_accs)

    # Average forgetting (last entry in history = cumulative)
    avg_forget = avg([m.get("avg_forgetting", [0])[-1] if isinstance(m.get("avg_forgetting"), list) else m.get("avg_forgetting", 0.0) for m in metrics_list])

    # Number of hidden layers (last neuron count length)
    n_layers_vals = []
    for m in metrics_list:
        nc = m.get("neuron_counts", [])
        if nc:
            n_layers_vals.append(len(nc[-1]))
    n_layers_final = avg(n_layers_vals) if n_layers_vals else 0

    # Total parameters
    total_params_vals = []
    for m in metrics_list:
        nc = m.get("neuron_counts", [])
        total_params_vals.append(_calc_total_params(nc, n_tasks))
    total_params_avg = avg(total_params_vals) if total_params_vals else 0

    # Training time
    times = [m.get("training_time", 0.0) for m in metrics_list]
    training_time_avg = avg(times)

    return {
        "per_task_acc": per_task_avg,
        "avg_accuracy": overall_avg_acc,
        "avg_forgetting": avg_forget,
        "n_layers_final": n_layers_final,
        "total_params_avg": total_params_avg,
        "training_time_avg": training_time_avg,
    }


def main():
    print("=" * 70)
    print("Benchmark: Original DEN vs DEN + Depth Growth")
    print(f"  Tasks: {TASKS},  Max iter/task: {MAX_ITER}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    all_original = []
    all_depth = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        print("  Original DEN...")
        metrics_o = run_original(seed)
        all_original.append(metrics_o)

        print("  DEN + Depth...")
        metrics_d = run_depth(seed)
        all_depth.append(metrics_d)

        o_acc = _test_acc_last_row(metrics_o)
        d_acc = _test_acc_last_row(metrics_d)
        o_nc = metrics_o.get("neuron_counts", [])
        d_nc = metrics_d.get("neuron_counts", [])
        print(f"    Original acc: {[f'{a:.4f}' for a in o_acc]}")
        print(f"    Depth   acc: {[f'{a:.4f}' for a in d_acc]}")
        if o_nc:
            print(f"    Original layers: {len(o_nc[-1])}, neurons: {o_nc[-1]}")
        if d_nc:
            print(f"    Depth   layers: {len(d_nc[-1])}, neurons: {d_nc[-1]}")

    # Aggregate
    o_summary = compute_summary(all_original)
    d_summary = compute_summary(all_depth)

    # --- Print comparison table ---
    print()
    print(f"  {'Metric':<28} {'Original DEN':<18} {'DEN + Depth':<18}")
    print(f"  {'-'*28} {'-'*18} {'-'*18}")
    print(f"  {'Average Accuracy':<28} {o_summary['avg_accuracy']:<18.4f} {d_summary['avg_accuracy']:<18.4f}")
    print(f"  {'Average Forgetting':<28} {o_summary['avg_forgetting']:<18.4f} {d_summary['avg_forgetting']:<18.4f}")
    print(f"  {'Final Hidden Layers':<28} {o_summary['n_layers_final']:<18.1f} {d_summary['n_layers_final']:<18.1f}")
    print(f"  {'Total Parameters':<28} {o_summary['total_params_avg']:<18.0f} {d_summary['total_params_avg']:<18.0f}")
    print(f"  {'Training Time (s)':<28} {o_summary['training_time_avg']:<18.1f} {d_summary['training_time_avg']:<18.1f}")
    print()

    # Per-task breakdown
    print("  Per-task final accuracy (averaged across seeds):")
    print(f"  {'Task':<8} {'Original DEN':<18} {'DEN + Depth':<18}")
    print(f"  {'-'*8} {'-'*18} {'-'*18}")
    for t in range(max(len(o_summary["per_task_acc"]), len(d_summary["per_task_acc"]))):
        o_val = o_summary["per_task_acc"][t] if t < len(o_summary["per_task_acc"]) else 0
        d_val = d_summary["per_task_acc"][t] if t < len(d_summary["per_task_acc"]) else 0
        print(f"  {t+1:<8} {o_val:<18.4f} {d_val:<18.4f}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    acc_diff = d_summary["avg_accuracy"] - o_summary["avg_accuracy"]
    forget_diff = d_summary["avg_forgetting"] - o_summary["avg_forgetting"]
    params_ratio = d_summary["total_params_avg"] / o_summary["total_params_avg"] if o_summary["total_params_avg"] > 0 else 0

    print(f"\n  Did depth growth improve accuracy?")
    if abs(acc_diff) < 0.005:
        print(f"    No meaningful difference ({acc_diff:+.4f})")
    elif acc_diff > 0:
        print(f"    Yes (+{acc_diff:.4f})")
    else:
        print(f"    No ({acc_diff:.4f})")

    print(f"\n  Did depth growth reduce forgetting?")
    if abs(forget_diff) < 0.005:
        print(f"    No meaningful difference ({forget_diff:+.4f})")
    elif forget_diff < 0:
        print(f"    Yes ({forget_diff:.4f})")
    else:
        print(f"    No (+{forget_diff:.4f})")

    print(f"\n  Did depth growth increase parameter efficiency?")
    print(f"    Parameters ratio: {params_ratio:.2f}x")
    if params_ratio < 0.95:
        print(f"    Yes (fewer parameters)")
    elif params_ratio > 1.05:
        print(f"    No ({params_ratio:.0f}% more parameters)")
    else:
        print(f"    Similar parameter count")

    print(f"\n  Was the extra complexity justified?")
    if abs(acc_diff) < 0.005 and abs(forget_diff) < 0.005 and params_ratio > 1.0:
        print(f"    No. Depth growth adds complexity (more layers, more parameters)")
        print(f"    without improving accuracy or reducing forgetting.")
    elif acc_diff > 0 and params_ratio < 1.05:
        print(f"    Possibly. Minor accuracy gain without large parameter increase.")
    elif forget_diff < -0.01:
        print(f"    Possibly. Reduced forgetting by {abs(forget_diff):.4f}")
    else:
        print(f"    Not clearly. Benefits are marginal or negative.")

    print()
    print(f"  Raw results by seed saved in: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
