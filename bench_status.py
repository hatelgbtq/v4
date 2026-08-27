#!/usr/bin/env python
"""Show benchmark run status (cached vs pending) for bench_cifar100.py."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "bench_cifar100"

VARIANTS = {
    "width_only",
    "v1_regular_interval",
    "v1_random_insert",
    "v2_val_loss_plateau",
    "v2_repeated_expansion",
    "v2_neuron_saturation",
    "v2_gradient_imbalance",
    "v2_representation_similarity",
}
SEEDS = [1004, 1005, 1006]

TOTAL = len(VARIANTS) * len(SEEDS)


def main() -> int:
    if not RESULTS.exists():
        print(f"no results dir yet: {RESULTS}")
        return 0

    done = []
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        mp = d / "metrics.json"
        if not mp.exists():
            continue
        try:
            with open(mp) as f:
                r = json.load(f)
            done.append((d.name, r.get("avg_acc"), r.get("training_time", 0.0)))
        except Exception:
            done.append((d.name, None, 0.0))

    n_finished = len({name.rsplit("_seed", 1)[0] for name, _, _ in done})
    n_total = 0
    if done:
        seeds_in_cache = {name.rsplit("_seed", 1)[1] for name, _, _ in done}
        n_total = len({name.rsplit("_seed", 1)[0] for name, _, _ in done}) * len(
            seeds_in_cache
        )

    variant_set = {name.rsplit("_seed", 1)[0] for name, _, _ in done}
    completed = [v for v in sorted(VARIANTS) if v in variant_set]
    pending = [v for v in sorted(VARIANTS) if v not in variant_set]

    print(f"completed runs: {len(done)}")
    if completed:
        print(f"  variants fully cached: {', '.join(completed)}")
    if pending:
        print(f"  variants pending: {', '.join(pending)}")
    print()
    for name, acc, t in sorted(done):
        acc_s = f"{acc:.4f}" if acc is not None else "?"
        print(f"  {name:<38} avg_acc={acc_s:<8} time={t:.0f}s")


if __name__ == "__main__":
    main()