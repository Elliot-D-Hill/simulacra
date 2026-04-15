from dataclasses import replace

import torch
import torch.distributions as dist
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .states import PredictorData


def treatment(
    data: PredictorData,
    tau: Float[Tensor, "J k"] | float | Tensor,
    gamma: Float[Tensor, "p J"] | None = None,
) -> PredictorData:
    coefficient = torch.randn(data.X.shape[-1], 2) if gamma is None else gamma
    logits = data.X @ coefficient  # [N, T, J]
    propensity = F.softmax(logits, dim=-1)  # [N, T, J]
    z = dist.Categorical(probs=propensity).sample().unsqueeze(-1)  # [N, T, 1]
    arms = propensity.shape[-1]
    indicator = F.one_hot(z.squeeze(-1), arms).to(data.eta.dtype)  # [N, T, J]
    eta = data.eta + indicator @ tau  # [N, T, k]
    return replace(data, eta=eta, treatment=z, propensity=propensity, gamma=coefficient)


def dose_response(
    data: PredictorData,
    tau: float | Tensor,
    gamma: Float[Tensor, "p 1"] | None = None,
    sigma: float | Tensor = 1.0,
) -> PredictorData:
    coefficient = torch.randn(data.X.shape[-1], 1) if gamma is None else gamma
    mean = data.X @ coefficient  # [N, T, 1]
    d = dist.Normal(mean, sigma)
    z = d.rsample()  # [N, T, 1]
    propensity = d.log_prob(z).exp()  # [N, T, 1]
    eta = data.eta + tau * z  # [N, T, k]
    return replace(data, eta=eta, treatment=z, propensity=propensity, gamma=coefficient)
