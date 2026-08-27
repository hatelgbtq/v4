"""
Data-driven depth-growth criteria for DEN.

Each criterion has the signature::

    (task_id: int, history: dict, model: nn.Module, config: dict) -> bool

and returns True when a new hidden layer should be inserted.

A criterion may read / write ``model.depth_growth_tracker`` dict
to persist state across tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  1.  Validation-loss plateau
# ---------------------------------------------------------------------------

def val_loss_plateau(
    task_id: int,
    history: dict,
    model: nn.Module,
    config: dict,
) -> bool:
    """Insert when validation loss has not improved for *patience* tasks.

    Config keys
    -----------
    patience : int (default 1)
        Number of consecutive tasks without improvement before inserting.
    min_delta : float (default 0.005)
        Relative improvement threshold.  An improvement smaller than
        ``min_delta * prev_loss`` counts as no improvement.
    """
    if task_id < 1:
        return False
    patience = config.get("patience", 1)
    min_delta = config.get("min_delta", 0.005)
    tracker: dict = model.depth_growth_tracker
    losses: list[float] = tracker.get("val_losses", [])
    if len(losses) < patience + 1:
        return False
    recent = losses[-patience:]
    for i in range(1, len(recent)):
        prev, cur = recent[i - 1], recent[i]
        if prev - cur > min_delta * abs(prev):
            return False
    return True


# ---------------------------------------------------------------------------
#  2.  Repeated width expansion
# ---------------------------------------------------------------------------

def repeated_expansion(
    task_id: int,
    history: dict,
    model: nn.Module,
    config: dict,
) -> bool:
    """Insert when width expansion was triggered for N consecutive tasks.

    Config keys
    -----------
    consecutive_expansions : int (default 2)
    """
    if task_id < 1:
        return False
    consecutive = config.get("consecutive_expansions", 2)
    tracker: dict = model.depth_growth_tracker
    added: list[int] = tracker.get("neurons_added", [])
    recent = added[-consecutive:]
    if len(recent) < consecutive:
        return False
    return all(n > 0 for n in recent)


# ---------------------------------------------------------------------------
#  3.  Neuron saturation
# ---------------------------------------------------------------------------

def _collect_activations(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    """Run a forward pass and return post-ReLU activation for each hidden layer.

    The batch is fed through ``model.hidden_layers`` without passing through
    the output head.  We use the **current** (un-sliced) weights so the
    activations reflect the full capacity of each layer.
    """
    activations: list[torch.Tensor] = []

    def _hook(m, _in, out):
        activations.append(out.detach().cpu())

    handles = [layer.register_forward_hook(_hook) for layer in model.hidden_layers]
    h = batch.to(device)
    if getattr(model, "embedder", None) is not None:
        h = model.embedder(h)
    h = h.view(h.size(0), -1)
    with torch.no_grad():
        for layer in model.hidden_layers:
            h = F.relu(layer(h))
    for handle in handles:
        handle.remove()
    return activations


def neuron_saturation(
    task_id: int,
    history: dict,
    model: nn.Module,
    config: dict,
) -> bool:
    """Insert when the fraction of highly-active neurons exceeds a threshold.

    A neuron is considered "saturated" if its post-ReLU activation is
    greater than ``saturation_threshold * (max activation in that layer)``.

    Config keys
    -----------
    saturation_ratio : float (default 0.02)
        Fraction of neurons that must be saturated to trigger insertion.
    saturation_threshold : float (default 0.3)
        Fraction of the layer's max activation that defines "saturated".
    """
    if task_id < 1:
        return False
    sat_ratio = config.get("saturation_ratio", 0.02)
    sat_thresh = config.get("saturation_threshold", 0.3)
    tracker: dict = model.depth_growth_tracker
    batch: torch.Tensor | None = tracker.get("probe_batch")
    device: torch.device | None = tracker.get("device")
    if batch is None or device is None:
        return False

    acts = _collect_activations(model, batch, device)
    if not acts:
        return False

    saturated_fractions = []
    for act in acts:
        if act.numel() == 0:
            continue
        flat = act.view(-1, act.size(-1))  # [batch, neurons]
        max_val = flat.max(dim=0, keepdim=True).values
        threshold = sat_thresh * max_val
        saturated = (flat > threshold).float().mean(dim=0)  # per neuron
        frac = (saturated > 0.5).float().mean().item()  # fraction of neurons
        saturated_fractions.append(frac)

    if not saturated_fractions:
        return False
    avg_sat = sum(saturated_fractions) / len(saturated_fractions)
    tracker["last_saturation_fraction"] = avg_sat
    return avg_sat > sat_ratio


# ---------------------------------------------------------------------------
#  4.  Gradient imbalance
# ---------------------------------------------------------------------------

def gradient_imbalance(
    task_id: int,
    history: dict,
    model: nn.Module,
    config: dict,
) -> bool:
    """Insert when shallow-layer gradient norms are much smaller than
    deep-layer gradient norms (vanishing-gradient signal).

    Config keys
    -----------
    imbalance_ratio : float (default 0.7)
        Ratio of deepest-layer grad norm to shallowest-layer grad norm
        above which insertion is triggered.
    """
    if task_id < 1:
        return False
    imbalance = config.get("imbalance_ratio", 0.7)
    tracker: dict = model.depth_growth_tracker
    norms: list[float] = tracker.get("grad_norm_ratios", [])
    if not norms:
        return False
    # Use the most recent task's gradient norm ratio
    latest = norms[-1]
    tracker["last_grad_norm_ratio"] = latest
    return latest > imbalance


# ---------------------------------------------------------------------------
#  5.  Representation similarity (CKA)
# ---------------------------------------------------------------------------

def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Centred Kernel Alignment (linear kernel) between two activation
    matrices of shape (batch, features)."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    XX = X @ X.T
    YY = Y @ Y.T
    hsic = (XX * YY).sum()
    sqrt_xx = (XX * XX).sum().sqrt()
    sqrt_yy = (YY * YY).sum().sqrt()
    if sqrt_xx.item() * sqrt_yy.item() == 0:
        return 0.0
    return (hsic / (sqrt_xx * sqrt_yy)).item()


def representation_similarity(
    task_id: int,
    history: dict,
    model: nn.Module,
    config: dict,
) -> bool:
    """Insert when consecutive layers have highly similar representations
    (linear CKA above threshold).

    Config keys
    -----------
    cka_threshold : float (default 0.7)
        CKA value above which layers are considered redundant.
    """
    if task_id < 1:
        return False
    cka_thresh = config.get("cka_threshold", 0.7)
    tracker: dict = model.depth_growth_tracker
    batch: torch.Tensor | None = tracker.get("probe_batch")
    device: torch.device | None = tracker.get("device")
    if batch is None or device is None:
        return False

    acts = _collect_activations(model, batch, device)
    if len(acts) < 2:
        return False

    max_sim = 0.0
    for i in range(len(acts) - 1):
        a = acts[i].view(acts[i].size(0), -1)
        b = acts[i + 1].view(acts[i + 1].size(0), -1)
        sim = _linear_cka(a, b)
        if sim > max_sim:
            max_sim = sim

    tracker["last_max_cka"] = max_sim
    return max_sim > cka_thresh


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------

CRITERIA: dict[str, callable] = {
    "val_loss_plateau": val_loss_plateau,
    "repeated_expansion": repeated_expansion,
    "neuron_saturation": neuron_saturation,
    "gradient_imbalance": gradient_imbalance,
    "representation_similarity": representation_similarity,
}
