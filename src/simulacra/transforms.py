from collections.abc import Callable
from dataclasses import replace

import torch
import torch.distributions as dist
from jaxtyping import Float
from torch import Tensor

from .covariance import array_normal
from .states import PredictorData, RandomEffect, ResponseData


def fixed_effects(
    X: Float[Tensor, "n t p"],
    beta: Float[Tensor, "p k"],
    points: Float[Tensor, "n t 1"],
) -> PredictorData:
    eta = X @ beta
    return PredictorData(X=X, points=points, eta=eta, beta=beta)


def _apply_random_effect(
    data: PredictorData,
    W: Float[Tensor, "n t levels"],
    B: Float[Tensor, "n t q"],
    b: Float[Tensor, "levels q k"],
) -> PredictorData:
    eta = torch.einsum("ntl,ntr,lrk->ntk", W, B, b)
    return replace(
        data,
        eta=data.eta + eta,
        random_effect=(*data.random_effect, RandomEffect(W=W, B=B, b=b)),
    )


def random_effects(
    data: PredictorData,
    W: Float[Tensor, "n t levels"],
    B: Float[Tensor, "n t q"],
    z: Float[Tensor, "levels q k"] | None = None,
    outcome_covariance: Float[Tensor, "k k"] | None = None,
    basis_covariance: Float[Tensor, "q q"] | None = None,
) -> PredictorData:
    levels = W.shape[-1]
    q = B.shape[-1]
    k = data.eta.shape[-1]
    noise = torch.randn(levels, q, k) if z is None else z
    b = array_normal(noise, basis_covariance, outcome_covariance)
    return _apply_random_effect(data, W, B, b)


def factor_effects(
    data: PredictorData,
    W: Float[Tensor, "n t levels"],
    B: Float[Tensor, "n t q"],
    loading: Float[Tensor, "q k"],
    basis_covariance: Float[Tensor, "q q"] | None = None,
) -> PredictorData:
    """Reduced-rank random effect: per-level scores u ~ N(0, basis_covariance) reach
    the k outcomes through a fixed loading, b[l, r, k] = u[l, r] * loading[r, k]. The
    factor-analytic sibling of random_effects (full outcome cov -> low-rank loading).
    """
    levels = W.shape[-1]
    q = B.shape[-1]
    u = array_normal(torch.randn(levels, q), None, basis_covariance)  # [levels q]
    b = torch.einsum("lr,rk->lrk", u, loading)  # [levels q k]
    return _apply_random_effect(data, W, B, b)


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
    data: S, weight: Float[Tensor, "k vocab_size"], temperature: float | Tensor = 1.0
) -> S:
    logits = data.eta @ weight / temperature
    tokens = dist.Categorical(logits=logits).sample()
    return replace(data, tokens=tokens, token_weight=weight)
