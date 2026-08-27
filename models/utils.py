"""
Shared helper functions for the DEN model.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def knowledge_preservation_loss(
    model: nn.Module,
    lambda_reg: float,
    prev_weight_slices: dict[str, tuple[torch.Tensor, tuple]] | None = None,
) -> torch.Tensor:
    """
    L2 penalty toward previous task's weights (Eq. 2 in the paper).

    ``prev_weight_slices`` maps parameter names to ``(prev_tensor, slices)``
    where *slices* is a tuple of slices that can be applied to both the
    current and previous parameter to extract the overlapping region.
    """
    if lambda_reg == 0.0 or prev_weight_slices is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name in prev_weight_slices:
            prev, idx = prev_weight_slices[name]
            prev = prev.to(param.device)
            diff = param[idx] - prev
            loss = loss + 0.5 * diff.norm(p=2).pow(2)
    # Lightweight anchor-based L2-SP penalty: if modules expose
    # `weight_anchor` / `bias_anchor` buffers, add a small penalty
    # pulling parameters toward those anchors. This is cheaper than
    # a full EWC and reduces destructive updates to shared weights.
    anchor_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        # weight_anchor / bias_anchor are buffers on DynamicLinear / TaskOutputHead
        if hasattr(module, "weight_anchor") and hasattr(module, "weight"):
            try:
                a = module.weight_anchor.to(module.weight.device)
                anchor_loss = anchor_loss + 0.5 * (module.weight - a).pow(2).sum()
            except Exception:
                pass
        if hasattr(module, "bias_anchor") and hasattr(module, "bias"):
            try:
                a = module.bias_anchor.to(module.bias.device)
                anchor_loss = anchor_loss + 0.5 * (module.bias - a).pow(2).sum()
            except Exception:
                pass

    total = loss + 0.5 * anchor_loss
    return lambda_reg * total


def get_prev_weight_slices(
    model: nn.Module,
    prev_params: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, tuple]]:
    """
    Build a dictionary mapping parameter names to (prev_tensor, slicing)
    so that both tensors can be indexed identically before computing the
    knowledge-preservation L2 penalty.

    When the current parameter is larger (e.g. after expansion), we
    only penalise the overlapping block.
    """
    slices: dict[str, tuple[torch.Tensor, tuple]] = {}
    for name, param in model.named_parameters():
        if name in prev_params:
            prev = prev_params[name]
            if prev.shape == param.shape:
                idx = tuple(slice(None) for _ in range(param.ndim))
                slices[name] = (prev, idx)
            else:
                ndim = param.ndim
                idx = tuple(
                    slice(0, min(prev.shape[d], param.shape[d]))
                    for d in range(ndim)
                )
                slices[name] = (prev[idx], idx)
    return slices


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class accuracy."""
    return float(accuracy_score(np.argmax(labels, 1), np.argmax(preds, 1)))


def roc_auc_multilabel(preds: np.ndarray, labels: np.ndarray) -> float:
    """Mean per-class ROC-AUC (used in the original paper)."""
    scores = []
    for i in range(labels.shape[1]):
        try:
            scores.append(roc_auc_score(labels[:, i], preds[:, i]))
        except ValueError:
            scores.append(0.5)
    return float(np.mean(scores))
