"""
Dynamic neural network layers that support:
- Output unit expansion (adding new neurons)
- Input unit expansion (handling growth from previous layer)
- Neuron splitting (duplicating drifted units for knowledge preservation)
- Per-neuron timestamping (tracking which task created each unit)
- Task-specific forward passes (masking to original sub-network)

Implements the core architectural ideas from the DEN paper:
"Dynamically Expandable Networks" (ICLR 2018).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicLinear(nn.Module):
    """
    A linear layer whose output dimension can grow over time.

    Supports two growth modes:
      1. Expansion:  add brand-new output units (initialized randomly).
      2. Splitting:  duplicate existing drifted units so one copy preserves
                     old knowledge and the other adapts to the new task.

    Each output neuron carries a **timestamp** --- the task ID during which
    it was created.  During inference for a particular task we only activate
    neurons whose timestamp <= task_id, which prevents semantic drift.

    Knowledge preservation anchors store the weight values *before* the
    current task's training starts, so an L2 penalty can be applied toward
    the original weights.
    """

    def __init__(self, in_features: int, out_features: int, task_id: int = 0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self._reset_parameters()

        # Anchor copies for knowledge-preservation regularisation.
        self.register_buffer("weight_anchor", self.weight.data.clone())
        self.register_buffer("bias_anchor", self.bias.data.clone())

        # Per-output-neuron timestamp: which task created / last split it.
        self.register_buffer(
            "timestamp", torch.full((out_features,), task_id, dtype=torch.long)
        )

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        task_id: int | None = None,
        out_slice: int | None = None,
        in_slice: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:         Input tensor of shape (batch, in_features).
            task_id:   Deprecated – kept for API compatibility.
            out_slice: Number of output neurons to use (slice rows).
            in_slice:  Number of input features to use (slice columns).
        """
        if out_slice is not None:
            w = self.weight[:out_slice, :in_slice] if in_slice is not None else self.weight[:out_slice, :]
            b = self.bias[:out_slice]
        else:
            w = self.weight
            b = self.bias
        return F.linear(x, w, b)

    # ------------------------------------------------------------------
    #  Expansion  (adding brand-new output units)
    # ------------------------------------------------------------------

    def expand_output_units(self, n_new: int, task_id: int):
        """
        Append *n_new* randomly-initialised output units.

        These new units receive timestamp = *task_id* and their anchor is
        zeroed so no knowledge-preservation penalty applies yet.
        """
        new_w = torch.empty(n_new, self.in_features)
        new_b = torch.empty(n_new)
        nn.init.kaiming_uniform_(new_w, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(new_b, -bound, bound)

        # --- replace parameters ---
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_w], dim=0))
        self.bias = nn.Parameter(torch.cat([self.bias.data, new_b]))
        self.out_features += n_new

        # anchors (zero-initialised → no penalty for fresh units)
        self.weight_anchor = torch.cat(
            [self.weight_anchor, torch.zeros(n_new, self.in_features)]
        )
        self.bias_anchor = torch.cat([self.bias_anchor, torch.zeros(n_new)])

        # timestamps
        self.timestamp = torch.cat(
            [self.timestamp, torch.full((n_new,), task_id, dtype=torch.long)]
        )

    def expand_input_units(self, n_new: int):
        """
        Append *n_new* columns to the weight matrix.

        Needed when the *previous* layer grew and this layer must accept
        extra input features.
        """
        new_w = torch.empty(self.out_features, n_new)
        nn.init.kaiming_uniform_(new_w, a=math.sqrt(5))
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_w], dim=1))
        self.in_features += n_new

        # anchor columns are zeroed so no penalty for new connections
        self.weight_anchor = torch.cat(
            [self.weight_anchor, torch.zeros(self.out_features, n_new)], dim=1
        )

    # ------------------------------------------------------------------
    #  Splitting  (duplicate drifted units)
    # ------------------------------------------------------------------

    def split_output_units(
        self,
        drift_indices: list[int],
        prev_weight: torch.Tensor,
        prev_bias: torch.Tensor,
        task_id: int,
    ) -> int:
        """
        Split neurons whose weights drifted beyond a threshold.

        For each drifted neuron *j*:
          - The **old** copy keeps *prev_weight[:, j]*  (timestamp = original).
          - The **new** copy gets *current weight[:, j]* (timestamp = task_id).

        Non-drifted neurons are left unchanged.

        Returns the number of units added (len(drift_indices)).
        """
        n_split = len(drift_indices)
        if n_split == 0:
            return 0

        drift_set = set(drift_indices)
        old_rows: list[torch.Tensor] = []
        new_rows: list[torch.Tensor] = []
        old_ts: list[torch.Tensor] = []
        new_ts: list[torch.Tensor] = []
        old_bias_rows: list[torch.Tensor] = []
        new_bias_rows: list[torch.Tensor] = []

        for j in range(self.out_features):
            if j in drift_set:
                # --- old copy: preserved knowledge ---
                old_rows.append(prev_weight[j : j + 1, :])
                old_bias_rows.append(prev_bias[j : j + 1])
                old_ts.append(self.timestamp[j : j + 1].clone())
                # --- new copy: adapted knowledge ---
                new_rows.append(self.weight.data[j : j + 1, :].clone())
                new_bias_rows.append(self.bias.data[j : j + 1].clone())
                new_ts.append(torch.full((1,), task_id, dtype=torch.long))
            else:
                old_rows.append(self.weight.data[j : j + 1, :].clone())
                old_bias_rows.append(self.bias.data[j : j + 1].clone())
                old_ts.append(self.timestamp[j : j + 1].clone())

        # Concatenation: first all original neurons (with drifted ones
        # replaced by their old copies), then the new copies appended.
        self.weight = nn.Parameter(torch.cat(old_rows + new_rows, dim=0))
        self.bias = nn.Parameter(torch.cat(old_bias_rows + new_bias_rows))
        # Remove old parameter from name tracking by re-assigning
        self.out_features += n_split

        # anchors  (zero for the new copies → no penalty yet)
        self.weight_anchor = torch.cat(
            [self.weight_anchor, torch.zeros(n_split, self.in_features)]
        )
        self.bias_anchor = torch.cat(
            [self.bias_anchor, torch.zeros(n_split)]
        )

        # timestamps
        self.timestamp = torch.cat(old_ts + new_ts)

        return n_split


class TaskOutputHead(nn.Module):
    """
    Task-specific output head.

    Each task gets its own linear projection from the last hidden layer to
    the output classes.  When the penultimate layer grows (more hidden units)
    every task's head is expanded with zero-initialised columns so that
    previous-task heads remain unaffected.
    """

    def __init__(self, in_features: int, num_classes: int, task_id: int):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        self.bias = nn.Parameter(torch.empty(num_classes))
        self._reset_parameters()

        self.register_buffer("weight_anchor", self.weight.data.clone())
        self.register_buffer("bias_anchor", self.bias.data.clone())

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, in_slice: int | None = None) -> torch.Tensor:
        if in_slice is not None:
            return F.linear(x, self.weight[:, :in_slice], self.bias)
        return F.linear(x, self.weight, self.bias)

    def expand_input_units(self, n_new: int):
        """Add *n_new* zero-initialised columns to accommodate growth in
        the previous layer."""
        new_w = torch.zeros(self.num_classes, n_new)
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_w], dim=1))
        self.in_features += n_new
        self.weight_anchor = torch.cat(
            [self.weight_anchor, torch.zeros(self.num_classes, n_new)], dim=1
        )
