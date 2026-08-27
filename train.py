#!/usr/bin/env python3
"""
DEN: Dynamically Expandable Networks – Training Entry Point.

Usage:
    python train.py --config configs/permuted_mnist.yaml
    python train.py --config configs/split_mnist.yaml  --gpu 0
    python train.py --dataset permuted_mnist  --num-tasks 5  --max-iter 2000
"""

from __future__ import annotations

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np

from models.den import DEN
from trainers.trainer import Trainer
from visualization.visualize import plot_all
from datasets.text import (
    get_text_loaders,
    save_vocab,
    load_vocab,
    tokenize_to_ids,
    generate as generate_text,
)

DATASET_REGISTRY = {
    "permuted_mnist": "datasets.permuted_mnist.get_permuted_mnist_loaders",
    "split_mnist": "datasets.split_mnist.get_split_mnist_loaders",
    "mnist": "datasets.mnist.get_mnist_loaders",
    "text": "datasets.text.get_text_loaders",
}


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_loaders(cfg: dict):
    """Import and call the appropriate dataset function.

    Returns ``(train_loaders, val_loaders, test_loaders, meta)`` where
    ``meta`` is a dict of dataset-specific info (or None).  For ``text``
    it holds the vocabulary, context length and embedding dims.
    """
    ds_name = cfg["dataset"]
    if ds_name == "permuted_mnist":
        from datasets.permuted_mnist import get_permuted_mnist_loaders
        return *get_permuted_mnist_loaders(
            num_tasks=cfg.get("num_tasks", 10),
            batch_size=cfg.get("batch_size", 256),
            seed=cfg.get("seed", 1004),
        ), None
    elif ds_name == "split_mnist":
        from datasets.split_mnist import get_split_mnist_loaders
        return *get_split_mnist_loaders(
            batch_size=cfg.get("batch_size", 256),
        ), None
    elif ds_name == "split_cifar100":
        from datasets.split_cifar100 import get_split_cifar100_loaders
        return *get_split_cifar100_loaders(
            num_tasks=cfg.get("num_tasks", 5),
            batch_size=cfg.get("batch_size", 256),
            seed=cfg.get("seed", 1004),
        ), None
    elif ds_name == "text":
        return *get_text_loaders(
            data_path=cfg["data_path"],
            num_tasks=cfg.get("num_tasks", 5),
            batch_size=cfg.get("batch_size", 256),
            context=cfg.get("context", 4),
            emb_dim=cfg.get("emb_dim", 32),
            vocab_size=cfg.get("vocab_size", 2048),
            stories_per_task=cfg.get("stories_per_task"),
            seed=cfg.get("seed", 1004),
        ),
    else:
        from datasets.mnist import get_mnist_loaders
        train_loader, test_loader = get_mnist_loaders(
            batch_size=cfg.get("batch_size", 256),
        )
        return [train_loader], [test_loader], [test_loader], None


def main():
    parser = argparse.ArgumentParser(description="DEN: Dynamically Expandable Networks")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID (default: cpu)")
    parser.add_argument("--dataset", type=str, default=None, choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ex-k", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the checkpoint in log_dir instead of starting over")
    parser.add_argument("--generate", type=str, default=None, metavar="PROMPT",
                        help="After training, generate text continuing this prompt (text dataset)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Text corpus file (one story per line) for the 'text' dataset")
    parser.add_argument("--context", type=int, default=None, help="Context window size (words)")
    parser.add_argument("--emb-dim", type=int, default=None, help="Embedding size per token")
    parser.add_argument("--vocab-size", type=int, default=None, help="Vocabulary size")
    parser.add_argument("--depth-growth-enabled", action="store_true", help="Enable depth growth")
    parser.add_argument("--depth-growth-interval", type=int, default=None, help="Tasks between depth insertions (V1 baseline)")
    parser.add_argument("--depth-growth-criterion", type=str, default=None,
                        help="Data-driven criterion: val_loss_plateau | repeated_expansion | neuron_saturation | gradient_imbalance | representation_similarity")
    parser.add_argument("--context-growth-enabled", action="store_true", help="Enable automatic context window growth")
    parser.add_argument("--context-growth-thr", type=float, default=None, help="Val loss threshold to trigger context growth")
    parser.add_argument("--context-growth-step", type=int, default=None, help="Tokens to add per growth event")
    parser.add_argument("--context-max", type=int, default=None, help="Maximum context window size")
    args = parser.parse_args()

    # --- Load config ---
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = {
            "dataset": args.dataset or "permuted_mnist",
            "num_tasks": 10,
            "batch_size": 256,
            "max_iter": 5000,
            "lr": 0.001,
            "input_dim": 784,
            "hidden_dims": [312, 128],
            "num_classes": 10,
            "ex_k": 5,
            "l1_lambda": 1e-5,
            "l2_lambda": 1e-4,
            "gl_lambda": 0.008,
            "regular_lambda": 0.1,
            "loss_thr": 0.01,
            "spl_thr": 0.05,
            "log_dir": "results/default",
            "seed": 1004,
        }

    # default rehearsal setting
    cfg.setdefault("replay_per_task", 800)
    cfg.setdefault("replay_fraction", 0.6)

    # Override with CLI args
    for key, arg_key in [
        ("num_tasks", "num_tasks"),
        ("max_iter", "max_iter"),
        ("lr", "lr"),
        ("batch_size", "batch_size"),
        ("ex_k", "ex_k"),
        ("log_dir", "log_dir"),
        ("seed", "seed"),
    ]:
        val = getattr(args, arg_key.replace("-", "_"), None)
        if val is not None:
            cfg[key] = val

    # Depth-growth CLI overrides
    if args.depth_growth_enabled:
        cfg["depth_growth_enabled"] = True
    if args.depth_growth_interval is not None:
        cfg["depth_growth_interval"] = args.depth_growth_interval
    if args.depth_growth_criterion is not None:
        cfg["depth_growth_criterion"] = args.depth_growth_criterion

    # Context-growth CLI overrides
    if args.context_growth_enabled:
        cfg["context_growth_enabled"] = True
    if args.context_growth_thr is not None:
        cfg["context_growth_thr"] = args.context_growth_thr
    if args.context_growth_step is not None:
        cfg["context_growth_step"] = args.context_growth_step
    if args.context_max is not None:
        cfg["context_max"] = args.context_max

    # Text-dataset CLI overrides
    for key, arg_key in [
        ("data_path", "data_path"),
        ("context", "context"),
        ("emb_dim", "emb_dim"),
        ("vocab_size", "vocab_size"),
    ]:
        val = getattr(args, arg_key.replace("-", "_"), None)
        if val is not None:
            cfg[key] = val

    # --- Device ---
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    elif cfg.get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"  [*] Device: {device}")
    print(f"  [*] Config: {cfg}")

    # --- Seed ---
    seed = cfg.get("seed", 1004)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Loaders ---
    train_loaders, val_loaders, test_loaders, meta = get_loaders(cfg)
    if cfg.get("dataset") == "split_cifar100":
        cfg["num_classes"] = 100 // cfg.get("num_tasks", 5)
    log_dir = cfg.get("log_dir", "results/default")

    if meta is not None:  # text dataset: adapt network dims to the corpus
        cfg["input_dim"] = meta["input_dim"]
        cfg["num_classes"] = meta["num_classes"]
        save_vocab(meta, f"{log_dir}/vocab.json")
        print(
            f"  [*] Text corpus: vocab={meta['vocab_size']} words, "
            f"context={meta['context']}, input_dim={cfg['input_dim']}"
        )

    # --- Model ---
    embedder = None
    if meta is not None:
        embedder = nn.Embedding(meta["vocab_size"], meta["emb_dim"])
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
        depth_growth_enabled=cfg.get("depth_growth_enabled", False),
        depth_growth_config={
            "criterion": cfg.get("depth_growth_criterion", ""),
            "interval": cfg.get("depth_growth_interval", 3),
            "patience": cfg.get("patience", 1),
            "min_delta": cfg.get("min_delta", 0.005),
            "consecutive_expansions": cfg.get("consecutive_expansions", 2),
            "saturation_ratio": cfg.get("saturation_ratio", 0.02),
            "saturation_threshold": cfg.get("saturation_threshold", 0.3),
            "imbalance_ratio": cfg.get("imbalance_ratio", 0.7),
            "cka_threshold": cfg.get("cka_threshold", 0.7),
        },
        embedder=embedder,
        context_growth_enabled=cfg.get("context_growth_enabled", False),
        context_growth_thr=cfg.get("context_growth_thr", 0.05),
        context_growth_step=cfg.get("context_growth_step", 64),
        context_max=cfg.get("context_max", 1024),
        current_context=cfg.get("context", 0) if cfg.get("context_growth_enabled", False) else 0,
        lstm_hidden=cfg.get("lstm_hidden", 256),
        tie_embeddings=False,
    )

    # Optionally enable embedding tying after construction (safe replacement)
    if model.embedder is not None:
        model.enable_embedding_tying()

    # --- Trainer ---
    resume_from = f"{log_dir}/checkpoint.pt" if args.resume else None
    # Pass dataset config so trainer can rebuild loaders on context growth
    dataset_cfg = cfg if cfg.get("dataset") == "text" else None
    trainer = Trainer(model, device, log_dir=log_dir, resume_from=resume_from, dataset_cfg=dataset_cfg)
    # configure trainer/model rehearsal settings
    trainer.replay_per_task = cfg.get("replay_per_task", trainer.replay_per_task)
    model.replay_fraction = cfg.get("replay_fraction", getattr(model, "replay_fraction", 0.3))

    # --- Train (skip if just generating from checkpoint) ---
    if not (args.generate and args.resume):
        history = trainer.train(
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            test_loaders=test_loaders,
            max_iter=cfg["max_iter"],
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            verbose=True,
        )

        # --- Visualize ---
        if not args.no_plots:
            plot_all(history, log_dir)

    # --- Generate text (text dataset only) ---
    if args.generate is not None and meta is not None:
        id_to_word, context = load_vocab(f"{log_dir}/vocab.json")
        prompt_ids = tokenize_to_ids(args.generate, meta["vocab"])
        out = generate_text(
            trainer.model,
            token_ids=prompt_ids,
            id_to_word=id_to_word,
            context=context,
            max_len=cfg.get("gen_length", 40),
            temperature=cfg.get("gen_temperature", 1.0),
            task_id=None,
            device=device,
        )
        print(f"\n  Prompt: {args.generate}\n  Generated: {out}")

    print("\n  [*] Done.")


if __name__ == "__main__":
    main()
