"""
Dynamic network expansion utilities (DEN Section 3.2).

When selective retraining does not reduce the loss below a threshold,
the network expands by adding new hidden units.  The expansion:

  1. Adds *ex_k* random units to each hidden layer.
  2. The added units are regularised with **group-lasso** on their
     incoming weights so that unnecessary units can be pruned later.
  3. After training, rows (output neurons) whose incoming weight
     vector has norm below the threshold are removed.

IMPORTANT CONVENTION
--------------------
All weight tensors are stored in PyTorch convention:
    ``weight`` shape = (out_features, in_features)

Each **row** corresponds to one output neuron.  Group-lasso is applied
to the rows because we want to zero out entire output neurons.

References:
  - Yoon et al., "Dynamically Expandable Networks", ICLR 2018.
  - Yuan & Lin, "Model selection and estimation in regression with
    grouped variables", JRSSB 2006.
"""

import torch


def group_lasso_penalty(weight: torch.Tensor) -> torch.Tensor:
    r"""Group-lasso (L2,1) regularisation over output neurons (rows).

    .. math::
        \Omega(W) = \sum_{j} \|W_{j,:}\|_2

    Sum of L2 norms of each row (output neuron).
    """
    row_norms = weight.norm(p=2, dim=1)
    return row_norms.sum()


def group_lasso_step(
    weight: torch.Tensor,
    lambda_gl: float,
) -> torch.Tensor:
    r"""
    Proximal step for group-lasso on output neurons (rows).

    If :math:`\|W_{j,:}\|_2 < \lambda_{\text{GL}}` → set entire row to 0.
    Otherwise → :math:`W_{j,:} \leftarrow W_{j,:} - \lambda_{\text{GL}} \frac{W_{j,:}}{\|W_{j,:}\|_2}`
    """
    row_norms = weight.norm(p=2, dim=1)
    for j in range(weight.size(0)):
        if row_norms[j] < lambda_gl:
            weight[j, :] = 0.0
        else:
            weight[j, :] -= lambda_gl * (weight[j, :] / row_norms[j])
    return weight


def find_useless_new_units(
    weight: torch.Tensor,
    n_new: int,
) -> list[int]:
    """
    Among the last *n_new* **rows** of *weight* (the newly added output
    neurons), return the global row indices of those that are entirely
    zero (killed by group-lasso).

    Returns global indices relative to the full weight matrix.
    """
    if n_new <= 0:
        return []
    new_part = weight[-n_new:, :]  # last n_new rows  (shape: [n_new, in_features])
    alive = (new_part != 0).any(dim=1)  # [n_new]  – True if row has any non-zero
    dead_indices = (~alive).nonzero(as_tuple=True)[0].tolist()
    offset = weight.size(0) - n_new
    return [offset + idx for idx in dead_indices]
