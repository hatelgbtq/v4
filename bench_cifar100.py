#!/usr/bin/env python
"""Benchmark depth-growth variants on Split CIFAR-100.

Runs each variant over a set of seeds, caching per-run metrics under
results/bench_cifar100/<variant>_seed<seed>/metrics.json.  Completed runs
are skipped on re-execution.

Variants:
  width_only                   : no depth growth (baseline)
  v1_regular_interval          : insert every N tasks
  v1_random_insert             : insert at random task boundaries
  v2_val_loss_plateau          : data-driven, val-loss plateau criterion
  v2_repeated_expansion        : data-driven, repeated-expansion criterion
  v2_neuron_saturation         : data-driven, neuron-saturation criterion
  v2_gradient_imbalance        : data-driven, gradient-imbalance criterion
  v2_representation_similarity : data-driven, CKA-similarity criterion
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from datasets.split_cifar100 import get_split_cifar100_loaders
from models.den import DEN
from trainers.trainer import Trainer

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "bench_cifar100"
DEFAULT_SEEDS = [1004, 1005, 1006]

VARIANT_NAMES = [
    "width_only",
    "v1_regular_interval",
    "v1_random_insert",
    "v2_val_loss_plateau",
    "v2_repeated_expansion",
    "v2_neuron_saturation",
    "v2_gradient_imbalance",
    "v2_representation_similarity",
]

VARIANT_CONFIGS: dict[str, dict] = {
    "width_only": {
        "depth_growth_enabled": False,
        "depth_growth_config": {},
    },
    "v1_regular_interval": {
        "depth_growth_enabled": True,
        "depth_growth_config": {"criterion": "v1_interval", "interval": 2},
    },
    "v1_random_insert": {
        "depth_growth_enabled": True,
        "depth_growth_config": {"criterion": "v1_random"},
    },
    "v2_val_loss_plateau": {
        "depth_growth_enabled": True,
        "depth_growth_config": {"criterion": "val_loss_plateau"},
    },
    "v2_repeated_expansion": {
        "depth_growth_enabled": True,
        "depth_growth_config": {"criterion": "repeated_expansion"},
    },
    "v2_neuron_saturation": {
        "depth_growth_enabled": True,
        "depth_growth_config": {"criterion": "neuron_saturation"},
    },
    "v2_gradient_imbalance": {
        "depth_growth_enabled": True,
        "depth_growth_config": {"criterion": "gradient_imbalance"},
    },
    "v2_representation_similarity": {
        "depth_growth_enabled": True,
        "depth_growth_config": {"criterion": "representation_similarity"},
    },
}

BENCH_CFG = {
    "input_dim": 3072,
    "hidden_dims": [192, 96],
    "num_classes": 20,
    "num_tasks": 5,
    "batch_size": 256,
    "max_iter": 500,
    "lr": 0.001,
    "ex_k": 10,
    "l1_lambda": 1e-5,
    "l2_lambda": 1e-4,
    "gl_lambda": 1e-3,
    "regular_lambda": 0.5,
    "loss_thr": 0.01,
    "spl_thr": 0.05,
}


def build_model(variant: str, cfg: dict) -> DEN:
    ov = VARIANT_CONFIGS[variant]
    return DEN(
        input_dim=cfg["input_dim"],
        hidden_dims=cfg["hidden_dims"],
        num_classes=cfg["num_classes"],
        ex_k=cfg["ex_k"],
        l1_lambda=cfg["l1_lambda"],
        l2_lambda=cfg["l2_lambda"],
        gl_lambda=cfg["gl_lambda"],
        regular_lambda=cfg["regular_lambda"],
        loss_thr=cfg["loss_thr"],
        spl_thr=cfg["spl_thr"],
        depth_growth_enabled=ov["depth_growth_enabled"],
        depth_growth_config=dict(ov["depth_growth_config"]),
    )


def run_variant(
    variant: str, seed: int, cfg: dict, max_iter: int, lr: float,
    verbose: bool, data_root: str,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loaders, val_loaders, test_loaders = get_split_cifar100_loaders(
        num_tasks=cfg["num_tasks"],
        batch_size=cfg["batch_size"],
        seed=seed,
        data_root=data_root,
    )

    model = build_model(variant, cfg)
    run_dir = RESULTS / f"{variant}_seed{seed}"
    trainer = Trainer(model, device="cpu", log_dir=run_dir)

    t0 = time.time()
    history = trainer.train(
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        test_loaders=test_loaders,
        max_iter=max_iter,
        lr=lr,
        batch_size=cfg["batch_size"],
        verbose=verbose,
    )
    elapsed = time.time() - t0

    acc_matrix = history["test_acc"]  # rows = after task k, cols = task j
    num_tasks = len(acc_matrix)

    # Per-task final recall and average accuracy
    per_task = [row[-1] for row in acc_matrix]
    avg_acc = float(np.mean(per_task))

    # Forgetting: average over tasks of (best-so-far recall - final recall)
    forget_per_task = []
    for j in range(num_tasks):
        best = max(acc_matrix[k][j] for k in range(j, num_tasks))
        forget_per_task.append(best - acc_matrix[-1][j])
    forget = float(np.mean(forget_per_task)) if forget_per_task else 0.0

    layers_per_task = history.get("n_hidden_layers", [])
    initial_layers = len(cfg["hidden_dims"])
    layers_added = [
        max(0, layers_per_task[i] - initial_layers)
        for i in range(len(layers_per_task))
    ] if layers_per_task else []
    total_layers = layers_per_task[-1] if layers_per_task else initial_layers
    total_params = sum(p.numel() for p in model.parameters())
    # Dense (fully-connected, no pruning) parameter estimate
    est = (cfg["input_dim"] + 1) * cfg["hidden_dims"][0]
    for a, b in zip(cfg["hidden_dims"], cfg["hidden_dims"][1:]):
        est += (a + 1) * b
    est += (cfg["hidden_dims"][-1] + 1) * cfg["num_classes"]
    est *= num_tasks
    est_params = est

    return {
        "variant": variant,
        "seed": seed,
        "data": "split_cifar100",
        "num_tasks": num_tasks,
        "max_iter": max_iter,
        "lr": lr,
        "batch_size": cfg["batch_size"],
        "test_acc": [float(a) for a in per_task],
        "avg_acc": avg_acc,
        "forget": forget,
        "layers_added": layers_added,
        "total_layers": total_layers,
        "total_params": total_params,
        "est_params": est_params,
        "training_time": elapsed,
        "timestamp": time.time(),
    }


def metrics_path(variant: str, seed: int) -> Path:
    return RESULTS / f"{variant}_seed{seed}" / "metrics.json"


def load_cached(variant: str, seed: int) -> dict | None:
    p = metrics_path(variant, seed)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            r = json.load(f)
    except Exception:
        return None
    if "avg_acc" not in r:
        return None
    return r


def save_metrics(run_dir: Path, metrics: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=BENCH_CFG["num_tasks"])
    ap.add_argument("--max-iter", type=int, default=BENCH_CFG["max_iter"])
    ap.add_argument("--lr", type=float, default=BENCH_CFG["lr"])
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--variants", type=str, nargs="+", default=None)
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if cached metrics exist")
    args = ap.parse_args()

    variants = args.variants or VARIANT_NAMES
    unknown = [v for v in variants if v not in VARIANT_CONFIGS]
    if unknown:
        print(f"[bench] unknown variant(s): {unknown}", file=sys.stderr)
        return 1

    cfg = dict(BENCH_CFG)
    cfg["num_tasks"] = args.tasks
    cfg["num_classes"] = 100 // args.tasks

    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"[bench] results -> {RESULTS}")

    runs = []
    for seed in args.seeds:
        for variant in variants:
            run_dir = RESULTS / f"{variant}_seed{seed}"
            cached = None if args.force else load_cached(variant, seed)
            if cached is not None:
                times = [cached.get("training_time", 0.0)]
                print(
                    f"[bench] cached {variant} seed {seed}: "
                    f"avg_acc={cached['avg_acc']:.4f} "
                    f"forget={cached['forget']:.4f} "
                    f"(time={times[0]:.0f}s)"
                )
                runs.append(cached)
                continue

            print(f"[bench] running {variant} seed {seed} "
                  f"({args.tasks} tasks, {args.max_iter} iters/task) ...")
            m = run_variant(
                variant, seed, cfg, args.max_iter, args.lr,
                verbose=args.verbose, data_root=args.data_root,
            )
            save_metrics(run_dir, m)
            print(
                f"[bench]   ok: avg_acc={m['avg_acc']:.4f} "
                f"forget={m['forget']:.4f} layers={m['total_layers']} "
                f"params={m['total_params']} ({m['training_time']:.0f}s)"
            )
            runs.append(m)

    write_summary(runs)
    write_report(runs)
    return 0


def write_summary(runs: list[dict]):
    by_variant: dict[str, list[dict]] = {}
    for r in runs:
        by_variant.setdefault(r["variant"], []).append(r)

    summary = {"runs": len(runs), "variants": []}
    for v in VARIANT_NAMES:
        if v not in by_variant:
            continue
        rs = by_variant[v]
        summary["variants"].append({
            "variant": v,
            "n_seeds": len(rs),
            "avg_acc": float(np.mean([r["avg_acc"] for r in rs])),
            "avg_forget": float(np.mean([r["forget"] for r in rs])),
            "avg_layers": float(np.mean([r["total_layers"] for r in rs])),
            "avg_params": float(np.mean([r["total_params"] for r in rs])),
            "seeds": [r["seed"] for r in rs],
        })
    summary["variants"].sort(key=lambda e: e["avg_acc"], reverse=True)

    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[bench] wrote summary.json")


def write_report(runs: list[dict]):
    lines = ["DEN Depth-Growth Benchmark — Split CIFAR-100", "=" * 62, ""]
    lines.append(f"runs: {len(runs)}")
    lines.append("")
    header = (f"{'variant':<32}{'avg_acc':>10}{'forget':>10}"
              f"{'layers':>8}{'params':>12}")
    lines.append(header)
    lines.append("-" * len(header))

    by_variant: dict[str, list[dict]] = {}
    for r in runs:
        by_variant.setdefault(r["variant"], []).append(r)

    for v in VARIANT_NAMES:
        if v not in by_variant:
            continue
        rs = by_variant[v]
        acc = float(np.mean([r["avg_acc"] for r in rs]))
        fgt = float(np.mean([r["forget"] for r in rs]))
        lay = float(np.mean([r["total_layers"] for r in rs]))
        par = int(np.mean([r["total_params"] for r in rs]))
        lines.append(f"{v:<32}{acc:>10.4f}{fgt:>10.4f}{lay:>8.2f}{par:>12}")

    lines.append("")
    for r in sorted(runs, key=lambda r: (r["variant"], r["seed"])):
        accs = ", ".join(f"{a:.3f}" for a in r["test_acc"])
        lines.append(
            f"{r['variant']:<32} seed {r['seed']}: "
            f"avg={r['avg_acc']:.4f} forget={r['forget']:.4f} "
            f"layers={r['total_layers']} params={r['total_params']} "
            f"[{accs}]"
        )

    with open(RESULTS / "REPORT.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[bench] wrote REPORT.txt")


if __name__ == "__main__":
    sys.exit(main())