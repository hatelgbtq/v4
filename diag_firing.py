#!/usr/bin/env python3
"""
Diagnostic: when does each depth-growth criterion fire?

Runs one full 5-task Split-CIFAR-100 training per criterion (seed 1004)
and prints, for every task, the criterion's decision plus the measured
values (val losses, neurons added, saturation fraction, grad ratio, CKA).

Requirement: insertion should fire at task 2-3, not 4-5.

Usage:
    python diag_firing.py [--max-iter 500] [--seed 1004]
"""

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import numpy as np
import torch
import yaml

import models.den as den_mod
from models.den import DEN
from trainers.trainer import Trainer

CRITERIA = [
    "val_loss_plateau",
    "repeated_expansion",
    "neuron_saturation",
    "gradient_imbalance",
    "representation_similarity",
]

LOG: list[dict] = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1004)
    parser.add_argument("--criterion", type=str, default=None,
                        choices=CRITERIA + [None])
    args = parser.parse_args()

    cfg = yaml.safe_load((BASE / "configs" / "split_cifar100.yaml").read_text())
    cfg["num_classes"] = 100 // cfg.get("num_tasks", 5)
    cfg["max_iter"] = args.max_iter
    cfg["seed"] = args.seed

    log_base = BASE / "results" / "diag_firing"
    log_base.mkdir(parents=True, exist_ok=True)

    orig_fn = den_mod.should_insert_layer

    def wrapped(task_id: int, history: dict, model, config: dict):
        tr = model.depth_growth_tracker
        entry = {
            "criterion": config.get("depth_growth_criterion", ""),
            "task_id": task_id,
            "decision": False,
            "val_losses": list(tr.get("val_losses", [])),
            "neurons_added": list(tr.get("neurons_added", [])),
            "last_saturation_fraction": tr.get("last_saturation_fraction"),
            "last_grad_norm_ratio": tr.get("last_grad_norm_ratio"),
            "last_max_cka": tr.get("last_max_cka"),
        }
        entry["decision"] = bool(orig_fn(task_id, history, model, config))
        LOG.append(entry)
        return entry["decision"]

    den_mod.should_insert_layer = wrapped

    from datasets.split_cifar100 import get_split_cifar100_loaders

    to_run = [args.criterion] if args.criterion else CRITERIA
    for crit in to_run:
        LOG.clear()
        seed = cfg["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loaders, val_loaders, test_loaders = get_split_cifar100_loaders(
            num_tasks=cfg["num_tasks"],
            batch_size=cfg["batch_size"],
            seed=seed,
        )
        model = DEN(
            input_dim=cfg["input_dim"],
            hidden_dims=cfg["hidden_dims"],
            num_classes=cfg["num_classes"],
            ex_k=cfg.get("ex_k", 10),
            l1_lambda=cfg.get("l1_lambda", 1e-5),
            l2_lambda=cfg.get("l2_lambda", 1e-4),
            gl_lambda=cfg.get("gl_lambda", 0.001),
            regular_lambda=cfg.get("regular_lambda", 0.5),
            loss_thr=cfg.get("loss_thr", 0.01),
            spl_thr=cfg.get("spl_thr", 0.05),
            depth_growth_enabled=True,
            depth_growth_config={
                "criterion": crit,
                "interval": cfg.get("depth_growth_interval", 3),
                "patience": cfg.get("patience", 1),
                "min_delta": cfg.get("min_delta", 0.005),
                "consecutive_expansions": cfg.get("consecutive_expansions", 2),
                "saturation_ratio": cfg.get("saturation_ratio", 0.02),
                "saturation_threshold": cfg.get("saturation_threshold", 0.3),
                "imbalance_ratio": cfg.get("imbalance_ratio", 0.7),
                "cka_threshold": cfg.get("cka_threshold", 0.7),
            },
        )
        device = torch.device("cpu")
        trainer = Trainer(model, device, log_dir=log_base / crit)
        trainer.train(
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            max_iter=cfg["max_iter"],
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            verbose=True,
        )

        print(f"\n========== FIRING SUMMARY: {crit} ==========")
        print(f"{'task':<6}{'decision':<10}{'val_losses':<18}{'neurons_added':<16}"
              f"{'sat_frac':<10}{'grad_ratio':<11}{'max_cka':<9}")
        for e in LOG:
            print(f"{e['task_id'] + 1:<6}{str(e['decision']):<10}"
                  f"{str(e['val_losses']):<18}{str(e['neurons_added']):<16}"
                  f"{str(e['last_saturation_fraction']):<10}"
                  f"{str(e['last_grad_norm_ratio']):<11}"
                  f"{str(e['last_max_cka']):<9}")
        fired = [e["task_id"] + 1 for e in LOG if e["decision"]]
        print(f"  -> inserted at tasks: {fired or 'NEVER'}")

    den_mod.should_insert_layer = orig_fn


if __name__ == "__main__":
    main()