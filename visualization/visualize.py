"""
Visualisation utilities for DEN training runs.

Automatically generates and saves plots showing:
  - Test accuracy per task over time
  - Forgetting per task
  - Neuron counts per layer
  - Cumulative neurons added / split
  - Parameter sparsity
  - Loss curves

All plots are saved to the log directory.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_all(history: dict, log_dir: str | Path):
    """
    Generate and save all diagnostic plots.

    Parameters
    ----------
    history : dict
        Output of ``Trainer.train()`` → ``Trainer.history``.
    log_dir : str or Path
        Where to save the PNG files.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    _plot_accuracy(history, log_dir)
    _plot_forgetting(history, log_dir)
    _plot_neuron_growth(history, log_dir)
    _plot_neurons_added_split(history, log_dir)
    _plot_sparsity(history, log_dir)
    _plot_parameter_growth(history, log_dir)
    _plot_depth_growth(history, log_dir)

    print(f"  [*] Plots saved to {log_dir}")


def _plot_accuracy(history: dict, log_dir: Path):
    """Test accuracy per task after each training step."""
    test_accs = history.get("test_acc", [])
    if not test_accs:
        return

    num_tasks = len(test_accs)
    fig, ax = plt.subplots(figsize=(8, 5))
    for j in range(num_tasks):
        accs = [test_accs[k][j] for k in range(j, num_tasks)]
        ax.plot(range(j, num_tasks), accs, marker="o", label=f"Task {j+1}")
    ax.set_xlabel("After training task")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Test Accuracy per Task")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(log_dir / "accuracy.png", dpi=150)
    plt.close(fig)


def _plot_forgetting(history: dict, log_dir: Path):
    """Average forgetting after each task."""
    forgetting = history.get("avg_forgetting", [])
    if not forgetting:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(forgetting) + 1), forgetting, marker="s", color="crimson")
    ax.set_xlabel("After training task")
    ax.set_ylabel("Average forgetting")
    ax.set_title("Forgetting")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(log_dir / "forgetting.png", dpi=150)
    plt.close(fig)


def _plot_neuron_growth(history: dict, log_dir: Path):
    """Number of neurons per hidden layer over time."""
    neuron_counts = history.get("neuron_counts", [])
    if not neuron_counts:
        return
    num_layers = len(neuron_counts[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(num_layers):
        counts = [nc[i] for nc in neuron_counts]
        ax.plot(range(1, len(counts) + 1), counts, marker="o", label=f"Layer {i+1}")
    ax.set_xlabel("After training task")
    ax.set_ylabel("Neuron count")
    ax.set_title("Neuron Growth per Hidden Layer")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(log_dir / "neuron_growth.png", dpi=150)
    plt.close(fig)


def _plot_neurons_added_split(history: dict, log_dir: Path):
    """Number of neurons added vs split per task."""
    added = history.get("neurons_added", [])
    split = history.get("neurons_split", [])
    if not added and not split:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(1, max(len(added), len(split)) + 1)
    if added:
        ax.bar([i - 0.15 for i in x], added, width=0.3, label="Added (expansion)", color="steelblue")
    if split:
        ax.bar([i + 0.15 for i in x], split, width=0.3, label="Split", color="darkorange")
    ax.set_xlabel("Task")
    ax.set_ylabel("Neurons")
    ax.set_title("Neurons Added vs Split per Task")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(log_dir / "neurons_added_split.png", dpi=150)
    plt.close(fig)


def _plot_sparsity(history: dict, log_dir: Path):
    """Parameter sparsity over time."""
    sparsity = history.get("sparsity", [])
    if not sparsity:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(sparsity) + 1), sparsity, marker="^", color="green")
    ax.set_xlabel("After training task")
    ax.set_ylabel("Sparsity (fraction zero)")
    ax.set_title("Parameter Sparsity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(log_dir / "sparsity.png", dpi=150)
    plt.close(fig)


def _plot_parameter_growth(history: dict, log_dir: Path):
    """Total parameter count over time."""
    neuron_counts = history.get("neuron_counts", [])
    if not neuron_counts:
        return
    # Rough estimate: for each layer, out_features * in_features
    total_params = []
    for step_counts in neuron_counts:
        params = 0
        prev = 784  # input dim
        for c in step_counts:
            params += prev * c + c  # weight + bias
            prev = c
        params += prev * 10 + 10  # output layer
        total_params.append(params)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(total_params) + 1), total_params, marker="o", color="purple")
    ax.set_xlabel("After training task")
    ax.set_ylabel("Total parameters")
    ax.set_title("Parameter Growth")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(log_dir / "parameter_growth.png", dpi=150)
    plt.close(fig)


def _plot_depth_growth(history: dict, log_dir: Path):
    """Number of hidden layers over time."""
    n_layers = history.get("n_hidden_layers", [])
    if not n_layers:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(n_layers) + 1), n_layers, marker="o", color="teal", linewidth=2)
    ax.set_xlabel("After training task")
    ax.set_ylabel("Number of hidden layers")
    ax.set_title("Depth Growth (Hidden Layer Count)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, len(n_layers) + 1))
    fig.tight_layout()
    fig.savefig(log_dir / "depth_growth.png", dpi=150)
    plt.close(fig)

    # Also plot depth insertion events
    depth_insertions = history.get("depth_insertions", [])
    if depth_insertions:
        fig, ax = plt.subplots(figsize=(8, 2))
        events = [i + 1 for i, v in enumerate(depth_insertions) if v]
        if events:
            ax.eventplot(events, orientation="horizontal", colors="teal", linewidths=4)
        ax.set_xlabel("Task")
        ax.set_title("Depth Insertion Events")
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(log_dir / "depth_insertions.png", dpi=150)
        plt.close(fig)
