"""Covariance builders separating scale from correlation: Sigma = S R S.

Correlation constructors (compound_symmetry, lkj_correlation, matern_kernel)
return unit-diagonal R and compose with plain tensor algebra (Hadamard * keeps
the unit diagonal; + or s A + (1 - s) B add structure), scaled once by
covariance. Cross-axis structure is Kronecker via array_normal.
"""

import math
from collections.abc import Callable
from enum import Enum
from typing import Annotated

import torch
import torch.distributions as dist
from beartype.vale import Is
from jaxtyping import Float
from torch import Tensor

_cholesky: Callable[[Tensor], Tensor] = getattr(torch, "linalg").cholesky


def _is_correlation(x: float) -> bool:
    return -1.0 <= x <= 1.0


type Correlation = Annotated[float, Is[_is_correlation]]


def _is_positive(x: float) -> bool:
    return x > 0.0


type Positive = Annotated[float, Is[_is_positive]]


def compound_symmetry(q: int, rho: Correlation = 0.0) -> Float[Tensor, "q q"]:
    """Equicorrelation R = (1 - rho) I + rho J, unit diagonal."""
    return torch.eye(q) * (1.0 - rho) + rho


def covariance(
    std: Float[Tensor, "q"], correlation: Float[Tensor, "q q"] | Correlation = 0.0
) -> Float[Tensor, "q q"]:
    """Covariance Sigma = S R S, S = diag(std). Scalar correlation gives compound
    symmetry; the default 0 leaves the diagonal diag(std**2).
    """
    if not isinstance(correlation, Tensor):
        correlation = compound_symmetry(std.shape[-1], correlation)
    std_diagonal = std.diag()
    return std_diagonal @ correlation.to(std) @ std_diagonal


class MaternOrder(Enum):
    HALF = 0.5
    THREE_HALVES = 1.5
    FIVE_HALVES = 2.5
    INFINITY = math.inf


def matern_kernel(
    points: Float[Tensor, "t"],
    length_scale: Positive = 1.0,
    order: MaternOrder = MaternOrder.THREE_HALVES,
) -> Float[Tensor, "t t"]:
    """Matern correlation over points, unit diagonal (amplitude 1).

    order is the smoothness nu: HALF is the exponential / Ornstein-Uhlenbeck
    kernel, INFINITY the RBF (squared-exponential) limit. Scale to a covariance
    with covariance(std, correlation=matern_kernel(...)).
    """
    distance = (points[:, None] - points[None, :]).abs()  # [t t]
    r = distance / length_scale
    match order:
        case MaternOrder.HALF:
            return torch.exp(-r)
        case MaternOrder.THREE_HALVES:
            s = math.sqrt(3.0) * r
            return (1.0 + s) * torch.exp(-s)
        case MaternOrder.FIVE_HALVES:
            s = math.sqrt(5.0) * r
            return (1.0 + s + s**2 / 3.0) * torch.exp(-s)
        case MaternOrder.INFINITY:
            return torch.exp(-0.5 * r**2)


def lkj_correlation(q: int, concentration: Positive = 1.0) -> Float[Tensor, "q q"]:
    """Unstructured correlation sampled from the LKJ distribution; concentration 1
    is uniform over correlation matrices, larger values concentrate toward I.
    """
    distribution = dist.LKJCholesky(q, concentration)
    chol: Tensor = getattr(distribution, "sample")()  # [q q] lower-triangular
    return chol @ chol.mT


def _cholesky_factor(cov: Tensor | None, size: int) -> Tensor:
    return torch.eye(size) if cov is None else _cholesky(cov)


def array_normal(
    noise: Float[Tensor, "*batch row col"],
    row_covariance: Float[Tensor, "row row"] | None = None,
    column_covariance: Float[Tensor, "col col"] | None = None,
) -> Float[Tensor, "*batch row col"]:
    """Color iid noise into a matrix-variate normal: X = chol_row @ Z @ chol_col.mT,
    so Cov(vec X) = column_covariance (x) row_covariance. A None factor is the
    identity, leaving that axis uncorrelated.
    """
    *_, row, col = noise.shape
    chol_row = _cholesky_factor(row_covariance, row)
    chol_column = _cholesky_factor(column_covariance, col)
    return chol_row @ noise @ chol_column.mT


def sample_features(
    temporal_covariance: Float[Tensor, "t t"],
    n: int,
    p: int,
    feature_covariance: Float[Tensor, "p p"] | None = None,
) -> Float[Tensor, "n t p"]:
    """Draw features X [n, t, p] correlated over time by temporal_covariance, each
    subject an array-normal draw with Cov(vec X_n) = feature_covariance (x)
    temporal_covariance. Build the temporal kernel with matern_kernel.
    """
    t = temporal_covariance.shape[-1]
    noise = torch.randn(n, t, p)
    return array_normal(noise, temporal_covariance, feature_covariance)
