"""
Selective retraining and neuron selection logic (DEN Section 3.2).

The pipeline for a new task T > 1:

  1. Freeze all shared (hidden) layers; train only the task-T output head
     for a small number of steps (*early training*).

  2. Identify which output-head neurons have non-zero weights; then walk
     backward through the network to find hidden units that contribute
     to those output neurons.  These form the **selected sub-network**.

  3. Train the sub-network (with knowledge-preservation regularisation)
     and merge the trained weights back into the full network.

  4. If the loss after merging is still above a threshold, trigger
     **dynamic expansion** (handled in `grow.py`).

References:
  - Yoon et al., "Dynamically Expandable Networks", ICLR 2018.

IMPORTANT CONVENTION
--------------------
All weight tensors are stored in PyTorch convention:
    ``weight`` shape = (out_features, in_features)

This means:
  - ``weight[row, :]``  → incoming weights to output neuron *row*
  - ``weight[:, col]``  → outgoing weights from input feature *col*
"""

from typing import Optional

import torch


def select_active_neurons(
    shared_weights: list[dict[str, torch.Tensor]],
    task_head_weight: torch.Tensor,
    n_hidden_layers: int,
    activations: list[torch.Tensor] | None = None,
) -> tuple[list[list[int]], dict[str, torch.Tensor]]:
    """
    Identify the sub-network of active neurons for the current task.

    Walk backward from the output head:
      1. Any output-head **row** (class) with non-zero weights is active.
      2. For hidden layer *i* (going backward), a neuron *j* is active if
         the layer-*(i+1)* weight matrix has a non-zero entry at
         ``(active_output_then, j)`` — i.e. the neuron's *outgoing*
         connection to any active neuron above is non-zero.
      3. The bottom-most (first) hidden layer always keeps all its neurons
         (the input dimension is fixed).

    Returns
    -------
    active_indices : list[list[int]]
        ``active_indices[l]`` holds the active *output* neuron indices for
        hidden layer *l* (0-indexed).  The last element is the active
        output-class indices.
    sub_weights : dict[str, Tensor]
        Extracted sub-network weight / bias tensors.
    """
    # ------------------------------------------------------------------
    #  Step 1: active output classes (rows of the output-head weight)
    # ------------------------------------------------------------------
    # task_head_weight shape = (num_classes, last_hidden_dim)
    # Use a small magnitude threshold rather than exact zero comparison
    eps = 1e-4
    out_nonzero = torch.any(task_head_weight.abs() > eps, dim=1).nonzero(as_tuple=True)[0]
    out_nonzero = out_nonzero.tolist()
    if not out_nonzero:
        out_nonzero = list(range(task_head_weight.size(0)))

    # active_indices[0..n_hidden_layers-1] = active output neurons per
    # hidden layer; active_indices[n_hidden_layers] = active classes.
    active_indices: list[list[int]] = [out_nonzero]  # will be extended at front

    # ------------------------------------------------------------------
    #  Step 2: walk backward through hidden layers
    # ------------------------------------------------------------------
    # Data flow:
    #   Input → Layer0 (active[0]) → Layer1 (active[1]) → Output (active[2])
    #
    # To decide whether Layer1's output neuron *j* is active we look at
    # the *output head* weight:  if any active class has a non-zero weight
    # from neuron *j*, then neuron *j* is active.
    #
    # Concretely:  ``task_head_weight[active_classes, j] != 0``
    #   →  (n_active_classes, last_hidden_dim).any(dim=0)  →  mask[j]
    #
    # For Layer0, we then look at Layer1's weight:
    #   ``layer1_weight[active_L1, j] != 0``
    #   →  (n_active_L1, dim_L0).any(dim=0)  →  mask[j]

    for layer_idx in range(n_hidden_layers - 1, -1, -1):
        # Weight of the layer *above* the one we are checking.
        if layer_idx == n_hidden_layers - 1:
            # Check against the output head
            w_above: Optional[torch.Tensor] = task_head_weight
        else:
            w_above = shared_weights[layer_idx + 1]["weight"]

        next_active = active_indices[0]  # active neurons one level up

        if layer_idx == 0:
            # Bottom layer: all output neurons are considered active.
            this_active = list(range(shared_weights[layer_idx]["weight"].size(0)))
        else:
            # w_above has shape (out_above, in_above) where in_above
            # equals the number of output neurons of the current layer.
            #
            # We want: for each current-layer output neuron *j*, is
            # there any next_active neuron with non-zero weight coming
            # from *j*?
            #
            #   w_above[next_active, :]  →  (n_active_above, n_current_out)
            #   .any(dim=0)              →  mask over current layer outputs
                active_mask = (w_above[next_active, :].abs() > eps).any(dim=0)
                this_active = active_mask.nonzero(as_tuple=True)[0].tolist()
                # If activations are provided, include neurons that are
                # active on the probe batch even if their head-weight is small.
                if activations is not None and layer_idx < len(activations):
                    act = activations[layer_idx]
                    if act is not None and act.numel() > 0:
                        mean_act = act.mean(dim=0)
                        act_idx = (mean_act.abs() > 1e-3).nonzero(as_tuple=True)[0].tolist()
                        # union of indices
                        this_active = sorted(list(set(this_active) | set(act_idx)))
                if not this_active:
                    this_active = list(range(w_above.size(1)))

        active_indices.insert(0, this_active)

    # Now active_indices has length n_hidden_layers + 1:
    #   [layer0_out, layer1_out, ..., output_classes]

    # ------------------------------------------------------------------
    #  Step 3: build sub-network weight dict
    # ------------------------------------------------------------------
    sub_weights: dict[str, torch.Tensor] = {}

    for layer_idx in range(n_hidden_layers):
        w = shared_weights[layer_idx]["weight"]
        b = shared_weights[layer_idx]["bias"]
        out_idx = active_indices[layer_idx]        # active output neurons
        in_idx = active_indices[layer_idx - 1] if layer_idx > 0 else slice(None)

        # Handle the case where we have a slice (first layer keeps all inputs)
        if isinstance(in_idx, slice):
            sub_w = w[out_idx, :]
        else:
            sub_w = w[out_idx, :][:, in_idx]

        sub_weights[f"layer{layer_idx}/weight"] = sub_w.contiguous()
        sub_weights[f"layer{layer_idx}/bias"] = b[out_idx].contiguous()

    # Output head: keep ALL output classes, but only active input features.
    # task_head_weight shape = (num_classes, last_hidden_dim)
    in_idx = active_indices[-2]   # active last-hidden-layer output neurons
    sub_weights["output/weight"] = task_head_weight[:, in_idx].contiguous()
    sub_weights["output/bias"] = torch.zeros(task_head_weight.size(0))

    return active_indices, sub_weights


def merge_sub_network(
    full_weights: list[dict[str, torch.Tensor]],
    task_head_weight: torch.Tensor,
    task_head_bias: torch.Tensor,
    active_indices: list[list[int]],
    trained_sub_weights: dict[str, torch.Tensor],
    n_hidden_layers: int,
) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """
    Merge trained sub-network weights back into the full network.

    Only entries corresponding to active neurons are updated.
    """
    for layer_idx in range(n_hidden_layers):
        w = full_weights[layer_idx]["weight"]
        b = full_weights[layer_idx]["bias"]
        out_idx = active_indices[layer_idx]
        in_idx = active_indices[layer_idx - 1] if layer_idx > 0 else slice(None)

        if isinstance(in_idx, slice):
            w[out_idx, :] = trained_sub_weights[f"layer{layer_idx}/weight"]
        else:
            w[out_idx, :][:, in_idx] = trained_sub_weights[f"layer{layer_idx}/weight"]

        b[out_idx] = trained_sub_weights[f"layer{layer_idx}/bias"]

    # Output head: update only the selected input-feature columns.
    in_idx = active_indices[-2]
    task_head_weight[:, in_idx] = trained_sub_weights["output/weight"]

    return full_weights, task_head_weight, task_head_bias
