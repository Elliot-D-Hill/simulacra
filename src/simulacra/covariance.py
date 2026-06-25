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
