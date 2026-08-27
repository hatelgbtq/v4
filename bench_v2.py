#!/usr/bin/env python3
"""
Benchmark V2 criteria against V1 fixed-interval baseline.

Usage:
    python bench_v2.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

V1_DIR = Path("/home/nabeel/Desktop/dumb_shit/AImodeltraining/DEN_DepthGrowth")
V2_DIR = Path("/home/nabeel/Desktop/dumb_shit/AImodeltraining/DEN_DepthGrowth_v2")
RESULTS_DIR = V2_DIR / "results" / "bench_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [1004, 1005, 1006]
TASKS = 4
MAX_ITER = 200

EXPERIMENTS = {
    "v1_fixed_interval": {
        "dir": V1_DIR,
        "args": ["--depth-growth-enabled", "--depth-growth-interval", "3"],
    },
    "v2_val_loss_plateau": {
        "dir": V2_DIR,
        "args": ["--depth-growth-enabled", "--depth-growth-criterion", "val_loss_plateau"],
    },
    "v2_repeated_expansion": {
        "dir": V2_DIR,
        "args": ["--depth-growth-enabled", "--depth-growth-criterion", "repeated_expansion"],
    },
    "v2_neuron_saturation": {
        "dir": V2_DIR,
        "args": ["--depth-growth-enabled", "--depth-growth-criterion", "neuron_saturation"],
    },
    "v2_gradient_imbalance": {
        "dir": V2_DIR,
        "args": ["--depth-growth-enabled", "--depth-growth-criterion", "gradient_imbalance"],
    },
    "v2_representation_similarity": {
        "dir": V2_DIR,
        "args": ["--depth-growth-enabled", "--depth-growth-criterion", "representation_similarity"],
    },
}


def run_experiment(name: str, exp_dir: Path, extra_args: list[str], seed: int) -> dict:
    log = RESULTS_DIR / f"{name}_seed{seed}"
    cmd = [
        sys.executable, str(exp_dir / "train.py"),
        "--dataset", "permuted_mnist",
        "--num-tasks", str(TASKS),
        "--max-iter", str(MAX_ITER),
        "--lr", "0.001",
        "--batch-size", "256",
        "--log-dir", str(log),
        "--seed", str(seed),
        "--no-plots",
    ] + extra_args
    t0 = time.time()
    subprocess.run(cmd, check=True, capture_output=True, env={"PYTHONPATH": str(exp_dir)}, cwd=str(exp_dir))
    elapsed = time.time() - t0
    metrics = _load_metrics(log)
    metrics["training_time"] = elapsed
    return metrics


def _load_metrics(log_dir: Path) -> dict:
    path = Path(log_dir) / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


def main():
    print("=" * 80)
    print("Benchmark: V1 Fixed-Interval vs V2 Data-Driven Criteria")
    print(f"  Tasks: {TASKS}, Max iter: {MAX_ITER}, Seeds: {SEEDS}")
    print("=" * 80)

    results = {}
    for name, exp in EXPERIMENTS.items():
        print(f"\n--- {name} ---")
        runs = []
        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            m = run_experiment(name, exp["dir"], exp["args"], seed)
            runs.append(m)
            print("done")

        # Aggregate
        last_accs = [r["test_acc"][-1] for r in runs]
        forgets = [r["avg_forgetting"][-1] for r in runs]
        n_layers_list = [len(r["neuron_counts"][-1]) for r in runs]
        times = [r["training_time"] for r in runs]

        def est_params(nc, tasks=TASKS, inp=784, out=10):
            last = nc[-1]
            p = 0
            prev = inp
            for o in last:
                p += prev * o + o
                prev = o
            p += tasks * (prev * out + out)
            return p

        params_list = [est_params(r["neuron_counts"]) for r in runs]

        # Per-task final accuracy averaged across seeds
        per_task = []
        for t in range(TASKS):
            vals = [r["test_acc"][-1][t] for r in runs if t < len(r["test_acc"][-1])]
            per_task.append(avg(vals))

        all_accs = [a for r in runs for a in r["test_acc"][-1]]

        results[name] = {
            "avg_acc": avg(all_accs),
            "per_task_acc": per_task,
            "avg_forgetting": avg(forgets),
            "n_layers": avg(n_layers_list),
            "total_params": avg(params_list),
            "time": avg(times),
            "insertion_tasks": [r.get("depth_insertions", []) for r in runs],
        }
        print(f"    Acc: {results[name]['avg_acc']:.4f}, Forgetting: {results[name]['avg_forgetting']:.4f}, Layers: {results[name]['n_layers']:.1f}, Params: {results[name]['total_params']:.0f}")

    # Table
    print("\n" + "=" * 80)
    print(f"{'Experiment':<32} {'Avg Acc':<10} {'Forgetting':<12} {'Layers':<8} {'Params':<10} {'Time (s)':<10}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<32} {r['avg_acc']:<10.4f} {r['avg_forgetting']:<12.4f} {r['n_layers']:<8.1f} {r['total_params']:<10.0f} {r['time']:<10.1f}")

    print("\n" + "=" * 80)
    print("Per-task accuracy (averaged across seeds):")
    print(f"{'Experiment':<32} ", end="")
    for t in range(TASKS):
        print(f"Task {t+1}    ", end="")
    print()
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<32} ", end="")
        for t in range(TASKS):
            print(f"{r['per_task_acc'][t]:<10.4f}", end="")
        print()

    # Conclusion
    baseline = results["v1_fixed_interval"]
    print("\n" + "=" * 80)
    print("CONCLUSION (vs V1 fixed-interval baseline)")
    print("=" * 80)
    for name, r in results.items():
        if name == "v1_fixed_interval":
            continue
        acc_diff = r["avg_acc"] - baseline["avg_acc"]
        forget_diff = baseline["avg_forgetting"] - r["avg_forgetting"]
        params_ratio = r["total_params"] / baseline["total_params"]
        print(f"\n  {name}:")
        print(f"    Accuracy: {acc_diff:+.4f} vs baseline")
        print(f"    Forgetting: {forget_diff:+.4f} vs baseline (lower is better)")
        print(f"    Parameters: {params_ratio:.2f}x of baseline")
        print(f"    Layers: {r['n_layers']:.1f} vs {baseline['n_layers']:.1f}")

    print("\n" + "=" * 80)
    print("Results saved in:", RESULTS_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
