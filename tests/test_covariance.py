import pytest
import torch

from simulacra.covariance import MaternOrder, matern_kernel, random_effects_covariance


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


def test_matern_kernel_exponential() -> None:
    """order=HALF gives the exponential / OU kernel exp(-d) with unit diagonal."""
    K = matern_kernel(torch.tensor([0.0, 1.0, 2.0]), order=MaternOrder.HALF)
    expected = torch.tensor(
        [
            [1.0, 0.367879, 0.135335],
            [0.367879, 1.0, 0.367879],
            [0.135335, 0.367879, 1.0],
        ]
    )
    assert torch.allclose(K, expected, atol=1e-5)


def test_matern_kernel_three_halves() -> None:
    """order=THREE_HALVES gives (1 + sqrt(3) r) exp(-sqrt(3) r)."""
    K = matern_kernel(torch.tensor([0.0, 1.0, 2.0]), order=MaternOrder.THREE_HALVES)
    expected = torch.tensor(
        [
            [1.0, 0.483358, 0.139731],
            [0.483358, 1.0, 0.483358],
            [0.139731, 0.483358, 1.0],
        ]
    )
    assert torch.allclose(K, expected, atol=1e-5)


def test_matern_kernel_five_halves() -> None:
    """order=FIVE_HALVES gives (1 + sqrt(5) r + 5 r^2 / 3) exp(-sqrt(5) r)."""
    K = matern_kernel(torch.tensor([0.0, 1.0, 2.0]), order=MaternOrder.FIVE_HALVES)
    expected = torch.tensor(
        [
            [1.0, 0.523994, 0.138660],
            [0.523994, 1.0, 0.523994],
            [0.138660, 0.523994, 1.0],
        ]
    )
    assert torch.allclose(K, expected, atol=1e-5)


def test_matern_kernel_rbf_limit() -> None:
    """order=INFINITY gives the RBF / squared-exponential kernel exp(-d^2 / 2)."""
    K = matern_kernel(torch.tensor([0.0, 1.0, 2.0]), order=MaternOrder.INFINITY)
    expected = torch.tensor(
        [
            [1.0, 0.606531, 0.135335],
            [0.606531, 1.0, 0.606531],
            [0.135335, 0.606531, 1.0],
        ]
    )
    assert torch.allclose(K, expected, atol=1e-5)


def test_matern_kernel_length_scale() -> None:
    """length_scale stretches the correlation: doubling it raises off-diagonals."""
    K = matern_kernel(
        torch.tensor([0.0, 1.0, 2.0]), length_scale=2.0, order=MaternOrder.HALF
    )
    expected = torch.tensor(
        [
            [1.0, 0.606531, 0.367879],
            [0.606531, 1.0, 0.606531],
            [0.367879, 0.606531, 1.0],
        ]
    )
    assert torch.allclose(K, expected, atol=1e-5)


def test_matern_kernel_scales_via_random_effects_covariance() -> None:
    """matern_kernel feeds random_effects_covariance as the correlation matrix."""
    K = matern_kernel(torch.tensor([0.0, 1.0]), order=MaternOrder.HALF)
    covariance = random_effects_covariance(torch.tensor([2.0, 2.0]), correlation=K)
    expected = torch.tensor([[4.0, 1.471518], [1.471518, 4.0]])
    assert torch.allclose(covariance, expected, atol=1e-5)
