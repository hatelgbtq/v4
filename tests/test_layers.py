"""
Unit tests for DynamicLinear: expansion, splitting, timestamps.
"""

import torch
import pytest

from models.layers import DynamicLinear


class TestExpandOutput:
    def test_expand_adds_neurons(self):
        layer = DynamicLinear(10, 5, task_id=0)
        assert layer.out_features == 5
        layer.expand_output_units(3, task_id=1)
        assert layer.out_features == 8
        assert layer.weight.shape == (8, 10)
        assert layer.bias.shape == (8,)

    def test_expand_updates_timestamps(self):
        layer = DynamicLinear(10, 5, task_id=0)
        layer.expand_output_units(3, task_id=2)
        assert (layer.timestamp[:5] == 0).all()
        assert (layer.timestamp[5:] == 2).all()

    def test_forward_after_expand(self):
        layer = DynamicLinear(10, 5, task_id=0)
        layer.expand_output_units(3, task_id=1)
        x = torch.randn(4, 10)
        out = layer(x)
        assert out.shape == (4, 8)


class TestExpandInput:
    def test_expand_input_adds_columns(self):
        layer = DynamicLinear(10, 5, task_id=0)
        layer.expand_input_units(3)
        assert layer.in_features == 13
        assert layer.weight.shape == (5, 13)

    def test_forward_after_input_expand(self):
        layer = DynamicLinear(10, 5, task_id=0)
        layer.expand_input_units(3)
        x = torch.randn(4, 13)
        out = layer(x)
        assert out.shape == (4, 5)


class TestSplit:
    def test_split_adds_neurons(self):
        layer = DynamicLinear(10, 5, task_id=0)
        prev_w = layer.weight.data.clone()
        prev_b = layer.bias.data.clone()
        # Simulate drift by modifying some neurons
        with torch.no_grad():
            layer.weight.data[2] += 1.0
            layer.weight.data[4] += 0.5
        n = layer.split_output_units([2, 4], prev_w, prev_b, task_id=1)
        assert n == 2
        assert layer.out_features == 7

    def test_split_preserves_old_timestamps(self):
        layer = DynamicLinear(10, 5, task_id=0)
        prev_w = layer.weight.data.clone()
        prev_b = layer.bias.data.clone()
        with torch.no_grad():
            layer.weight.data[1] += 1.0
        layer.split_output_units([1], prev_w, prev_b, task_id=2)
        # Original neuron 1 gets timestamp 0 (old copy)
        # Actually: old copy keeps timestamp 0, new copy gets task_id 2
        assert layer.timestamp[1] == 0  # old copy preserved
        assert layer.timestamp[5] == 2  # new copy gets current task

    def test_split_zero_drift(self):
        layer = DynamicLinear(10, 5, task_id=0)
        prev_w = layer.weight.data.clone()
        prev_b = layer.bias.data.clone()
        n = layer.split_output_units([], prev_w, prev_b, task_id=1)
        assert n == 0
        assert layer.out_features == 5


class TestTimestampMasking:
    def test_forward_masks_older_neurons(self):
        layer = DynamicLinear(10, 5, task_id=0)
        layer.expand_output_units(3, task_id=2)
        # Forward with out_slice=5 should only use first 5 neurons
        x = torch.randn(4, 10)
        out = layer(x, out_slice=5)
        assert out.shape == (4, 5)

    def test_forward_masks_all_neurons(self):
        layer = DynamicLinear(10, 5, task_id=0)
        x = torch.randn(4, 10)
        out = layer(x, out_slice=5)
        assert out.shape == (4, 5)


if __name__ == "__main__":
    pytest.main([__file__])
