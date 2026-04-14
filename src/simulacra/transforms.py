from collections.abc import Callable
from dataclasses import replace

import torch
import torch.distributions as dist
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .states import PredictorData, Prior, RandomEffect, ResponseData


def resolve(prior: Prior, shape: tuple[int, ...] = ()) -> Tensor:
    if isinstance(prior, Tensor):
        return prior
    suffix = len(prior.batch_shape) + len(prior.event_shape)
    sample_shape = shape[:-suffix] if suffix else shape
    return (
        prior.rsample(sample_shape) if prior.has_rsample else prior.sample(sample_shape)
    )


def fixed_effects(
    X: Float[Tensor, "*D n t p"],
    beta: Float[Tensor, "*D p k"],
    points: Float[Tensor, "*D n t 1"],
) -> PredictorData:
    eta = X @ beta.unsqueeze(-3)
    return PredictorData(X=X, points=points, eta=eta, beta=beta)


def random_effects(
    data: PredictorData, levels: int, q: int, W: Prior | None, B: Prior, b: Prior
) -> PredictorData:
    *batch, n, t, k = data.eta.shape
    if W is None:
        indices = torch.arange(n) % levels
        membership = F.one_hot(indices, levels).float().unsqueeze(-2)
        membership = membership.expand(*batch, n, 1, levels)
    else:
        membership = resolve(W, (*batch, n, 1, levels))
    basis = resolve(B, (*batch, n, t, q))
    coefficient = resolve(b, (*batch, levels, q, k))
    eta = torch.einsum("...ntl,...ntr,...lrk->...ntk", membership, basis, coefficient)
    return replace(
        data,
        eta=data.eta + eta,
        random_effect=(
            *data.random_effect,
            RandomEffect(W=membership, B=basis, b=coefficient),
        ),
    )


def missing_x[S: PredictorData](data: S, proportion: float) -> S:
    mask = torch.rand_like(data.X) < proportion
    return replace(data, X=data.X.masked_fill(mask, float("nan")))


def missing_y[S: ResponseData](data: S, proportion: float) -> S:
    mask = torch.rand_like(data.y) < proportion
    return replace(data, y=data.y.masked_fill(mask, float("nan")))


def constant_y[S: ResponseData](data: S) -> S:
    return replace(data, y=data.y[..., :1, :].expand_as(data.y))


def activation(data: PredictorData, fn: Callable[[Tensor], Tensor]) -> PredictorData:
    return replace(data, eta=fn(data.eta))


def tokenize[S: PredictorData](
    data: S, vocab_size: int, weight: Prior, temperature: float | Tensor = 1.0
) -> S:
    *batch, _, _, k_in = data.eta.shape
    w = resolve(weight, (*batch, 1, k_in, vocab_size))
    logits = data.eta @ w / temperature
    tokens = dist.Categorical(logits=logits).sample()
    return replace(data, tokens=tokens, token_weight=w.squeeze(-3))
