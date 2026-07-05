"""
Unit tests for metrics module.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.metrics import calculate_jsd


def test_calculate_jsd():
    """Test that JSD returns 0 for identical distributions."""
    base_probs = torch.tensor([0.5, 0.3, 0.2])
    pert_probs = torch.tensor([0.5, 0.3, 0.2])
    jsd = calculate_jsd(base_probs, pert_probs, top_k=3)
    assert jsd == 0.0


def test_calculate_jsd_symmetric():
    """Test that JSD is symmetric."""
    base_probs = torch.tensor([0.5, 0.3, 0.2])
    pert_probs = torch.tensor([0.1, 0.6, 0.3])
    jsd1 = calculate_jsd(base_probs, pert_probs, top_k=3)
    jsd2 = calculate_jsd(pert_probs, base_probs, top_k=3)
    assert jsd1 == jsd2


def test_calculate_jsd_bounded():
    """Test that JSD is bounded between 0 and 1."""
    base_probs = torch.tensor([1.0, 0.0, 0.0])
    pert_probs = torch.tensor([0.0, 0.5, 0.5])
    jsd = calculate_jsd(base_probs, pert_probs, top_k=3)
    assert 0 <= jsd <= 1


if __name__ == "__main__":
    test_calculate_jsd()
    test_calculate_jsd_symmetric()
    test_calculate_jsd_bounded()
    print("✅ All tests passed!")