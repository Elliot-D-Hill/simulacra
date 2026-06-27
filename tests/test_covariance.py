import pytest
import torch
from jaxtyping import TypeCheckError

from simulacra.covariance import (
    MaternOrder,
    array_normal,
    matern_kernel,
    random_effects_covariance,
    sample_features,
)


def test_compound_symmetry() -> None:
    """Scalar correlation builds Q = S R S with equicorrelated off-diagonals."""
    covariance = random_effects_covariance(torch.tensor([2.0, 3.0]), 0.5)
    assert covariance.equal(torch.tensor([[4.0, 3.0], [3.0, 9.0]]))


def test_diagonal_default() -> None:
    """Default correlation 0 gives a diagonal covariance of variances."""
    covariance = random_effects_covariance(torch.tensor([2.0, 3.0]))
    assert covariance.equal(torch.tensor([[4.0, 0.0], [0.0, 9.0]]))


def test_correlation_out_of_range_rejected() -> None:
    """Scalar correlation outside [-1, 1] is not a valid correlation and is rejected."""
    with pytest.raises(TypeCheckError):
        random_effects_covariance(torch.tensor([1.0, 1.0]), 2.0)


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


def test_matern_kernel_nonpositive_length_scale_rejected() -> None:
    """length_scale must be positive; <= 0 is rejected at the call boundary."""
    with pytest.raises(TypeCheckError):
        matern_kernel(torch.tensor([0.0, 1.0]), length_scale=0.0)


def test_matern_kernel_scales_via_random_effects_covariance() -> None:
    """matern_kernel feeds random_effects_covariance as the correlation matrix."""
    K = matern_kernel(torch.tensor([0.0, 1.0]), order=MaternOrder.HALF)
    covariance = random_effects_covariance(torch.tensor([2.0, 2.0]), correlation=K)
    expected = torch.tensor([[4.0, 1.471518], [1.471518, 4.0]])
    assert torch.allclose(covariance, expected, atol=1e-5)


def test_array_normal_neutral_returns_noise() -> None:
    """Both covariances None: identity factors leave the noise unchanged."""
    noise = torch.randn(5, 2, 3)
    assert array_normal(noise).equal(noise)


def test_array_normal_colors_both_axes() -> None:
    """Diagonal row and column covariances scale each axis independently."""
    noise = torch.ones(1, 2, 2)
    row_covariance = torch.tensor([[4.0, 0.0], [0.0, 9.0]])
    column_covariance = torch.tensor([[1.0, 0.0], [0.0, 16.0]])
    X = array_normal(noise, row_covariance, column_covariance)
    expected = torch.tensor([[[2.0, 8.0], [3.0, 12.0]]])
    assert X.equal(expected)


def test_sample_features_shape_from_matern() -> None:
    """sample_features draws X [n, t, p]; t comes from the kernel, p is given."""
    X = sample_features(matern_kernel(torch.tensor([0.0, 1.0, 2.0, 3.0])), n=10, p=3)
    assert X.shape == (10, 4, 3)


def test_sample_features_recovers_kronecker() -> None:
    """Cov(vec X_n) over the (t, p) axes is feature_covariance (x) temporal_covariance."""
    n = 40000
    temporal_covariance = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
    feature_covariance = torch.tensor([[1.0, 0.6], [0.6, 1.0]])
    torch.manual_seed(0)  # type: ignore[no-untyped-call]
    X = sample_features(
        temporal_covariance, n, p=2, feature_covariance=feature_covariance
    )
    empirical = torch.cov(X.reshape(n, 4).mT)
    expected = torch.tensor(
        [
            [2.0, 1.2, 0.0, 0.0],
            [1.2, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.3],
            [0.0, 0.0, 0.3, 0.5],
        ]
    )
    assert torch.allclose(empirical, expected, atol=0.05)
