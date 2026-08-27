"""
Unit tests for selective retraining utilities.

Uses PyTorch weight convention: weight shape = (out_features, in_features).
"""

import torch
import pytest

from models.prune import select_active_neurons


class TestSelectActiveNeurons:
    def test_all_output_units_active(self):
        """
        When all output head rows (classes) are non-zero,
        all hidden neurons should be active.
        """
        n_layers = 2
        # Layer 0: 10 output neurons, 20 input features
        # Layer 1: 5 output neurons, 10 input features (= Layer 0's outputs)
        # Output:  3 classes, 5 input features (= Layer 1's outputs)
        shared = [
            {"weight": torch.ones(10, 20), "bias": torch.ones(10)},
            {"weight": torch.ones(5, 10), "bias": torch.ones(5)},
        ]
        head_w = torch.ones(3, 5)  # all 3 classes active

        indices, sub_w = select_active_neurons(shared, head_w, n_layers)

        assert len(indices) == 3  # [layer0, layer1, output]
        assert indices[0] == list(range(10))  # bottom layer: all
        assert indices[1] == list(range(5))
        assert indices[2] == list(range(3))

        # Check sub-weight shapes are consistent
        assert sub_w["layer0/weight"].shape == (10, 20)
        assert sub_w["layer1/weight"].shape == (5, 10)
        assert sub_w["output/weight"].shape == (3, 5)

    def test_some_hidden_neurons_dead(self):
        """
        When certain hidden neurons have zero outgoing to active class,
        they should be pruned.
        """
        n_layers = 2
        # Layer 0: 10 output neurons, 784 inputs
        # Layer 1: 4 output neurons, 10 inputs (from Layer 0)
        #   - Neuron 0,1 connect to all Layer 0 outputs
        #   - Neuron 2,3 connect to none
        # Output:  2 classes, 4 inputs (from Layer 1)
        #   - Only class 0 is active
        #   - Class 0 connects to Layer 1 neurons 0,1 only

        l1_weight = torch.zeros(4, 10)
        l1_weight[0, :] = 1.0  # neuron 0 connects to all
        l1_weight[1, :] = 1.0  # neuron 1 connects to all
        # neurons 2,3 are dead (all zeros)

        out_weight = torch.zeros(2, 4)
        out_weight[0, 0] = 1.0  # class 0 uses neuron 0
        out_weight[0, 1] = 1.0  # class 0 uses neuron 1
        # class 1 is dead

        shared = [
            {"weight": torch.ones(10, 784), "bias": torch.ones(10)},
            {"weight": l1_weight, "bias": torch.ones(4)},
        ]

        indices, sub_w = select_active_neurons(shared, out_weight, n_layers)

        # Only class 0 is active
        assert indices[2] == [0]
        # Layer 1: neurons 0,1 connect to class 0, should be active
        assert indices[1] == [0, 1]
        # Layer 0 (bottom): all are active
        assert len(indices[0]) == 10

        # Check shapes
        assert sub_w["layer0/weight"].shape == (10, 784)
        assert sub_w["layer1/weight"].shape == (2, 10)
        assert sub_w["output/weight"].shape == (2, 2)  # 2 classes, 2 active features

    def test_sub_network_shapes_consistent(self):
        """Sub-network weights should form a valid MLP."""
        n_layers = 3
        shared = [
            {"weight": torch.ones(20, 784), "bias": torch.ones(20)},
            {"weight": torch.ones(15, 20), "bias": torch.ones(15)},
            {"weight": torch.ones(10, 15), "bias": torch.ones(10)},
        ]
        head_w = torch.ones(5, 10)

        indices, sub_w = select_active_neurons(shared, head_w, n_layers)

        # Verify the sub-network forms a valid chain:
        # Input → [20, 784] → [15, 20] → [10, 15] → [5, 10] → Output
        in_features = 784
        for i in range(n_layers):
            w = sub_w[f"layer{i}/weight"]
            assert w.size(1) == in_features
            in_features = w.size(0)

        out_w = sub_w["output/weight"]
        assert out_w.size(1) == in_features
        assert out_w.size(0) == 5  # 5 classes

    def test_first_layer_active_all(self):
        """Bottom layer should always have all its neurons active."""
        n_layers = 1  # single hidden layer
        shared = [
            {"weight": torch.ones(10, 784), "bias": torch.ones(10)},
        ]
        # Only class 0 is active, connecting to only neuron 0
        head_w = torch.zeros(5, 10)
        head_w[0, 0] = 1.0

        indices, sub_w = select_active_neurons(shared, head_w, n_layers)

        # Layer 0 (bottom, also the only hidden layer): all 10 neurons active
        assert len(indices[0]) == 10
        # Output: only class 0 active
        assert indices[1] == [0]

        assert sub_w["layer0/weight"].shape == (10, 784)
        assert sub_w["output/weight"].shape == (5, 10)  # 5 classes, 10 features


if __name__ == "__main__":
    pytest.main([__file__])
