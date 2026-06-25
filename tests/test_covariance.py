import pytest
import torch

from simulacra.covariance import random_effects_covariance


def test_compound_symmetry() -> None:
    """Scalar correlation builds Q = S R S with equicorrelated off-diagonals."""
    covariance = random_effects_covariance(torch.tensor([2.0, 3.0]), 0.5)
    assert covariance.equal(torch.tensor([[4.0, 3.0], [3.0, 9.0]]))


def test_diagonal_default() -> None:
    """Default correlation 0 gives a diagonal covariance of variances."""
    covariance = random_effects_covariance(torch.tensor([2.0, 3.0]))
    assert covariance.equal(torch.tensor([[4.0, 0.0], [0.0, 9.0]]))


def test_pd_bound_raises() -> None:
    """Compound symmetry at or below -1/(q-1) is indefinite and raises."""
    with pytest.raises(ValueError, match="positive definite"):
        random_effects_covariance(torch.tensor([1.0, 1.0, 1.0]), -0.5)
