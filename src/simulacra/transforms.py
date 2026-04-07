from collections.abc import Callable
from dataclasses import replace

import torch
import torch.distributions as dist
from torch import Tensor

from .states import (
    CensoredData,
    EventTimeData,
    InitialData,
    PredictorData,
    ResponseData,
)

type Prior = dist.Distribution | Tensor
type FamilyFn = Callable[[PredictorData], tuple[ResponseData, dict[str, Tensor]]]


def resolve(prior: Prior, shape: tuple[int, ...] = ()) -> Tensor:
    if isinstance(prior, Tensor):
        return prior
    shape = shape[: -len(prior.event_shape) or None]
    return prior.rsample(shape) if prior.has_rsample else prior.sample(shape)


def fixed_effects(
    data: InitialData, X: Prior, beta: Prior
) -> tuple[PredictorData, dict[str, Tensor]]:
    X = resolve(X, (*data.draws, data.n, data.t, data.p))
    beta = resolve(beta, (*data.draws, 1, data.p, data.k))
    return PredictorData(X=X, eta=X @ beta), {"beta": beta.squeeze(-3)}


def random_effects(
    data: PredictorData, index: int, levels: int, q: int, W: Prior, B: Prior, b: Prior
) -> tuple[PredictorData, dict[str, Tensor]]:
    eta = data.eta
    *batch, n, t, k = eta.shape
    W = resolve(W, (*batch, n, t, levels))
    B = resolve(B, (*batch, n, t, q))
    b = resolve(b, (*batch, levels, q, k))
    eta_re = torch.einsum("...ntl,...ntr,...lrk->...ntk", W, B, b)
    params = {f"W_{index}": W, f"B_{index}": B, f"b_{index}": b}
    return replace(data, eta=eta + eta_re), params


def gaussian(
    data: PredictorData, covariance: Prior
) -> tuple[ResponseData, dict[str, Tensor]]:
    K = data.eta.shape[-1]
    cov = resolve(covariance, (K, K))
    if cov.ndim < 2:
        cov = cov * torch.eye(K)
    y = resolve(dist.MultivariateNormal(data.eta, cov))
    return ResponseData(**vars(data), y=y), {}


def poisson(data: PredictorData) -> tuple[ResponseData, dict[str, Tensor]]:
    return ResponseData(**vars(data), y=resolve(dist.Poisson(data.eta.exp()))), {}


def bernoulli(data: PredictorData) -> tuple[ResponseData, dict[str, Tensor]]:
    y = resolve(dist.Binomial(total_count=1, logits=data.eta))
    return ResponseData(**vars(data), y=y), {}


def binomial(
    data: PredictorData, num_trials: int
) -> tuple[ResponseData, dict[str, Tensor]]:
    y = resolve(dist.Binomial(total_count=num_trials, logits=data.eta))
    return ResponseData(**vars(data), y=y), {}


def negative_binomial(
    data: PredictorData, concentration: float | Tensor
) -> tuple[ResponseData, dict[str, Tensor]]:
    y = resolve(dist.NegativeBinomial(concentration, logits=data.eta))
    return ResponseData(**vars(data), y=y), {}


def gamma(
    data: PredictorData, concentration: float | Tensor
) -> tuple[ResponseData, dict[str, Tensor]]:
    y = resolve(dist.Gamma(concentration, data.eta.exp().reciprocal()))
    return ResponseData(**vars(data), y=y), {}


def log_normal(
    data: PredictorData, std: float | Tensor
) -> tuple[ResponseData, dict[str, Tensor]]:
    return ResponseData(**vars(data), y=resolve(dist.LogNormal(data.eta, std))), {}


def categorical(data: PredictorData) -> tuple[ResponseData, dict[str, Tensor]]:
    y = resolve(dist.Multinomial(total_count=1, logits=data.eta))
    return ResponseData(**vars(data), y=y), {}


def event_time(
    data: ResponseData, shape: float | Tensor
) -> tuple[EventTimeData, dict[str, Tensor]]:
    scale = data.eta.exp().reciprocal()
    event_time = resolve(dist.Weibull(scale, shape))
    return EventTimeData(**vars(data), event_time=event_time), {}


def censor_time(
    data: EventTimeData, horizon: float | Tensor
) -> tuple[CensoredData, dict[str, Tensor]]:
    clamped = torch.clamp(data.event_time, max=horizon)
    return CensoredData(**vars(data), censor_time=clamped), {}


def constant_target(
    data: PredictorData,
    family: Callable[[PredictorData], tuple[ResponseData, dict[str, Tensor]]],
) -> tuple[ResponseData, dict[str, Tensor]]:
    """Pool eta over T, sample once per subject, broadcast y back."""
    pooled = replace(data, eta=data.eta.mean(dim=-2, keepdim=True))
    result, params = family(pooled)
    return replace(result, eta=data.eta, y=result.y.expand_as(data.eta)), params


def missing_x[S: PredictorData](
    data: S, proportion: float
) -> tuple[S, dict[str, Tensor]]:
    mask = torch.rand_like(data.X) < proportion
    return replace(data, X=data.X.masked_fill(mask, float("nan"))), {}


def missing_y[S: ResponseData](
    data: S, proportion: float
) -> tuple[S, dict[str, Tensor]]:
    mask = torch.rand_like(data.y) < proportion
    return replace(data, y=data.y.masked_fill(mask, float("nan"))), {}


def tokenize[S: PredictorData](data: S, vocab_size: int) -> tuple[S, dict[str, Tensor]]:
    weights = torch.randn(data.eta.shape[-1], vocab_size)
    probs = torch.softmax(data.eta @ weights, dim=-1)
    return replace(data, tokens=resolve(dist.Categorical(probs))), {}
