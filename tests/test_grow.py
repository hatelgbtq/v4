"""
Unit tests for growth utilities: group-lasso, useless unit detection.

All tests use PyTorch convention: weight shape = (out_features, in_features).
Group-lasso operates on rows (output neurons).
"""

import torch
import pytest

from models.grow import group_lasso_penalty, group_lasso_step, find_useless_new_units


class TestGroupLassoPenalty:
    def test_all_zero_rows(self):
        w = torch.zeros(5, 3)
        loss = group_lasso_penalty(w)
        assert loss.item() == 0.0

    def test_non_zero_rows(self):
        w = torch.ones(5, 3)
        loss = group_lasso_penalty(w)
        # Each of the 5 rows has norm sqrt(3) → total = 5 * sqrt(3)
        expected = 5 * (3 ** 0.5)
        assert abs(loss.item() - expected) < 1e-5


class TestGroupLassoStep:
    def test_zero_small_rows(self):
        w = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], dtype=torch.float32)  # [3, 2]
        result = group_lasso_step(w.clone(), lambda_gl=0.5)
        # Row norms: [sqrt(0.02)=0.141, sqrt(0.08)=0.283, sqrt(0.18)=0.424]
        # All < 0.5, so all rows should be zero
        assert (result == 0).all()

    def test_large_row_preserved(self):
        w = torch.tensor([[1.0, 1.0], [0.1, 0.1]], dtype=torch.float32)  # [2, 2]
        result = group_lasso_step(w.clone(), lambda_gl=0.5)
        # Row 0 norm = sqrt(2) = 1.414 > 0.5 → shrunk
        # Row 1 norm = sqrt(0.02) = 0.141 < 0.5 → zeroed
        assert result[0, 0] < 1.0  # shrunk
        assert (result[1, :] == 0).all()  # zeroed


class TestFindUselessNewUnits:
    def test_all_alive(self):
        w = torch.ones(10, 5)  # [out=10, in=5]
        dead = find_useless_new_units(w, n_new=3)
        assert dead == []

    def test_some_dead(self):
        w = torch.ones(10, 5)
        w[-2:, :] = 0.0  # last 2 rows (output neurons) are dead
        dead = find_useless_new_units(w, n_new=5)
        # Global indices: offset = 10 - 5 = 5, dead local indices [3, 4] → [8, 9]
        assert dead == [8, 9]

    def test_all_dead(self):
        w = torch.zeros(10, 5)
        dead = find_useless_new_units(w, n_new=10)
        assert dead == list(range(0, 10))


if __name__ == "__main__":
    pytest.main([__file__])
