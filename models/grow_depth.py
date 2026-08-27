"""
Depth-growth decision logic and layer insertion utilities.

This module extends the original DEN (which only grows width) with
the ability to dynamically insert NEW hidden layers.

The decision about WHEN to insert a layer is isolated in
``should_insert_layer()`` so that different strategies can be tested.

The **what** (layer construction) is handled by ``insert_hidden_layer()``.
The **when** (decision logic) is delegated to data-driven criteria in
``criteria.py``, selectable via the ``depth_growth_criterion`` config key.

References
----------
- Yoon et al., "Dynamically Expandable Networks", ICLR 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .criteria import CRITERIA
from .layers import DynamicLinear


# ---------------------------------------------------------------------------
#  Depth-growth decision logic
# ---------------------------------------------------------------------------

def should_insert_layer(
    task_id: int,
    history: dict,
    model: nn.Module,
    config: dict,
) -> bool:
    """Return True if a new hidden layer should be inserted before the
    next task.

    Delegates to the criterion named in ``config["depth_growth_criterion"]``.
    If the key is missing or unknown, no insertion occurs.

    Parameters
    ----------
    task_id : int
        Current task index (0-based).
    history : dict
        Training history (typically ``trainer.history``).
    model : nn.Module
        DEN model instance (needed by some criteria for activation hooks).
    config : dict
        Configuration dictionary.  Expected keys:

        - ``depth_growth_enabled`` : bool
        - ``depth_growth_criterion`` : str
          One of ``"val_loss_plateau"``, ``"repeated_expansion"``,
          ``"neuron_saturation"``, ``"gradient_imbalance"``,
          ``"representation_similarity"``.
        - Plus criterion-specific keys (see ``criteria.py``).
    """
    if not config.get("depth_growth_enabled", False):
        return False

    criterion_name = config.get("depth_growth_criterion", "")
    if not criterion_name:
        return False

    criterion_fn = CRITERIA.get(criterion_name)
    if criterion_fn is None:
        return False

    return criterion_fn(task_id, history, model, config)


# ---------------------------------------------------------------------------
#  Layer insertion
# ---------------------------------------------------------------------------

def insert_hidden_layer(
    model: nn.Module,
    task_id: int,
    insert_dim: int | None = None,
    noise_scale: float = 0.01,
) -> int:
    """Insert a new hidden layer into the model's ``hidden_layers``.

    The new layer is **appended at the end** of the current hidden stack
    and has **equal input and output dimensions** (matching the dimension
    of the last hidden layer).

    **Initialisation**
        The weight is set to the **identity matrix** plus small Gaussian
        noise, and the bias is zero.  This guarantees that immediately
        after insertion the network's input-output behaviour is nearly
        identical to before -- the new layer acts as a pass-through that
        can gradually learn new features during subsequent tasks.

        Identity initialisation is chosen because:
        - It minimally disrupts already-learned representations.
        - The new layer's output initially equals its input, so downstream
          layers (which expect the original dimension) continue to work.
        - Small noise breaks symmetry so that different units learn
          different features during subsequent training.

    Parameters
    ----------
    model : nn.Module
        A DEN model instance (must have ``hidden_layers``, ``output_heads``,
        ``n_hidden_layers`` attributes).
    task_id : int
        Current task ID  (used for the new layer's timestamp).
    insert_dim : int, optional
        If given, the new layer will have this dimension (in = out).
        Otherwise uses the dimension of the last hidden layer.
    noise_scale : float
        Standard deviation of Gaussian noise added to the identity matrix.

    Returns
    -------
    insert_idx : int
        Index at which the new layer was inserted.
    """
    n_layers = len(model.hidden_layers)
    insert_idx = n_layers  # append at the end

    # Determine in/out dimension for the new layer
    if insert_dim is None:
        insert_dim = model.hidden_layers[-1].out_features

    # Create the new layer with identity-like initialisation
    new_layer = DynamicLinear(insert_dim, insert_dim, task_id=task_id)

    with torch.no_grad():
        # Identity matrix: each output unit initially copies its input
        nn.init.eye_(new_layer.weight)
        nn.init.zeros_(new_layer.bias)
        # Small noise to break symmetry
        new_layer.weight += torch.randn_like(new_layer.weight) * noise_scale

    # Rebuild the ModuleList with the new layer inserted
    old_layers = list(model.hidden_layers)
    old_layers.insert(insert_idx, new_layer)
    model.hidden_layers = nn.ModuleList(old_layers)
    model.n_hidden_layers += 1

    # NOTE: head expansion is handled AFTER warmup in _train_subsequent_task
    # because the warmup uses the old architecture (stamp-based forward).

    return insert_idx


# ---------------------------------------------------------------------------
#  Architecture tracking
# ---------------------------------------------------------------------------

def get_architecture_summary(model: nn.Module) -> dict:
    """Return a dictionary summarising the current model architecture.

    Returns
    -------
    dict with keys:
        n_hidden_layers : int
        neurons_per_layer : list[int]
        total_params : int
    """
    neurons = [layer.out_features for layer in model.hidden_layers]
    # Estimate parameter count
    prev = model.input_dim
    total = 0
    for layer in model.hidden_layers:
        total += prev * layer.out_features + layer.out_features
        prev = layer.out_features
    for name, head in model.output_heads.items():
        total += prev * head.weight.size(0) + head.weight.size(0)
    return {
        "n_hidden_layers": model.n_hidden_layers,
        "neurons_per_layer": neurons,
        "total_params": total,
    }
