from collections.abc import Callable
from dataclasses import replace

import torch
import torch.distributions as dist
from jaxtyping import Float
from torch import Tensor

from .states import PredictorData, RandomEffect, ResponseData


def fixed_effects(
    X: Float[Tensor, "n t p"],
    beta: Float[Tensor, "p k"],
    points: Float[Tensor, "n t 1"],
) -> PredictorData:
    eta = X @ beta
    return PredictorData(X=X, points=points, eta=eta, beta=beta)


def random_effects(
    data: PredictorData,
    W: Float[Tensor, "n t levels"],
    B: Float[Tensor, "n t q"],
    b: Float[Tensor, "levels q k"] | None = None,
) -> PredictorData:
    levels = W.shape[-1]
    q = B.shape[-1]
    k = data.eta.shape[-1]
    coefficient = torch.randn(levels, q, k) if b is None else b
    eta = torch.einsum("ntl,ntr,lrk->ntk", W, B, coefficient)
    return replace(
        data,
        eta=data.eta + eta,
        random_effect=(*data.random_effect, RandomEffect(W=W, B=B, b=coefficient)),
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
    data: S,
    vocab_size: int,
    weight: Float[Tensor, "k vocab_size"] | None = None,
    temperature: float | Tensor = 1.0,
) -> S:
    k = data.eta.shape[-1]
    w = torch.randn(k, vocab_size) if weight is None else weight
    logits = data.eta @ w / temperature
    tokens = dist.Categorical(logits=logits).sample()
    return replace(data, tokens=tokens, token_weight=w)
