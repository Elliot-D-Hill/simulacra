import math
from enum import Enum

import torch
from jaxtyping import Float
from torch import Tensor


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
