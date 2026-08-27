"""
Unit tests for timestamp logic in DynamicLinear and selective masking.
"""

import torch
import pytest

from models.layers import DynamicLinear


class TestTimestamp:
    def test_initial_timestamp(self):
        layer = DynamicLinear(10, 5, task_id=0)
        assert (layer.timestamp == 0).all()
        assert layer.timestamp.shape == (5,)

    def test_expand_timestamp(self):
        layer = DynamicLinear(10, 3, task_id=1)
        layer.expand_output_units(2, task_id=3)
        expected = torch.tensor([1, 1, 1, 3, 3], dtype=torch.long)
        assert (layer.timestamp == expected).all()

    def test_split_timestamp(self):
        layer = DynamicLinear(10, 3, task_id=0)
        prev_w = layer.weight.data.clone()
        prev_b = layer.bias.data.clone()
        with torch.no_grad():
            layer.weight.data[1] += 2.0  # drift neuron 1
        layer.split_output_units([1], prev_w, prev_b, task_id=2)
        # Neuron 1 old copy: ts=0, new extra copy: ts=2
        assert layer.timestamp[1] == 0       # old copy preserved
        assert layer.timestamp[3] == 2       # new copy

    def test_prediction_mask(self):
        """Forward with out_slice should restrict active neurons."""
        layer = DynamicLinear(10, 3, task_id=0)
        layer.expand_output_units(2, task_id=1)
        layer.expand_output_units(1, task_id=2)

        x = torch.randn(4, 10)
        out_full = layer(x)          # all 6 neurons
        out_t0 = layer(x, out_slice=3)  # only 3 neurons
        out_t1 = layer(x, out_slice=5)  # 5 neurons

        assert out_full.shape == (4, 6)
        assert out_t0.shape == (4, 3)
        assert out_t1.shape == (4, 5)


if __name__ == "__main__":
    pytest.main([__file__])
