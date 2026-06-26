import math
from collections.abc import Callable
from enum import Enum

import torch
from jaxtyping import Float
from torch import Tensor

_linalg = getattr(torch, "linalg")
_cholesky: Callable[[Tensor], Tensor] = _linalg.cholesky


def random_effects_covariance(
    std: Float[Tensor, "q"], correlation: Float[Tensor, "q q"] | float = 0.0
) -> Float[Tensor, "q q"]:
    """Random-effects covariance Q = S R S, S = diag(std).

    Scalar correlation gives compound symmetry, positive definite only for
    rho > -1/(q-1).
    """
    q = std.shape[-1]
    correlation = torch.as_tensor(correlation, dtype=std.dtype)
    if correlation.ndim == 0:
        if q > 1 and correlation <= -1.0 / (q - 1):
            raise ValueError(
                f"compound-symmetry correlation {correlation.item()} must exceed "
                f"-1/(q-1) = {-1.0 / (q - 1)} for q={q} to stay positive definite"
            )
        correlation = torch.eye(q, dtype=std.dtype) * (1.0 - correlation) + correlation
    std_diagonal = std.diag()
    return std_diagonal @ correlation @ std_diagonal


class MaternOrder(Enum):
    HALF = 0.5
    THREE_HALVES = 1.5
    FIVE_HALVES = 2.5
    INFINITY = math.inf


def matern_kernel(
    points: Float[Tensor, "t"],
    length_scale: float = 1.0,
    order: MaternOrder = MaternOrder.THREE_HALVES,
) -> Float[Tensor, "t t"]:
    """Matern correlation over points, unit diagonal (amplitude 1).

    order is the smoothness nu: HALF is the exponential / Ornstein-Uhlenbeck
    kernel, INFINITY the RBF (squared-exponential) limit. Scale to a covariance
    with random_effects_covariance(std, correlation=matern_kernel(...)).
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


def _cholesky_factor(covariance: Tensor | None, size: int) -> Tensor:
    return torch.eye(size) if covariance is None else _cholesky(covariance)


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
