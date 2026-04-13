from dataclasses import replace

import torch
import torch.distributions as dist
from torch import Tensor

from .states import PredictorData, Prior, ResponseData, promote
from .transforms import Step, resolve

type Family = Step[PredictorData, ResponseData]


def gaussian(data: PredictorData, covariance: Prior) -> ResponseData:
    K = data.eta.shape[-1]
    covariance = resolve(covariance, (K, K))
    if covariance.ndim < 2:
        covariance = covariance * torch.eye(K)
    y = resolve(dist.MultivariateNormal(data.eta, covariance))
    return promote(ResponseData, data, y=y)


def poisson(data: PredictorData) -> ResponseData:
    y = resolve(dist.Poisson(data.eta.exp()))
    return promote(ResponseData, data, y=y)


def bernoulli(data: PredictorData) -> ResponseData:
    y = resolve(dist.Binomial(total_count=1, logits=data.eta))
    return promote(ResponseData, data, y=y)


def binomial(data: PredictorData, num_trials: int) -> ResponseData:
    y = resolve(dist.Binomial(total_count=num_trials, logits=data.eta))
    return promote(ResponseData, data, y=y)


def negative_binomial(
    data: PredictorData, concentration: float | Tensor
) -> ResponseData:
    y = resolve(dist.NegativeBinomial(concentration, logits=data.eta))
    return promote(ResponseData, data, y=y)


def gamma(
    data: PredictorData, concentration: float | Tensor
) -> ResponseData:
    y = resolve(dist.Gamma(concentration, data.eta.exp().reciprocal()))
    return promote(ResponseData, data, y=y)


def log_normal(data: PredictorData, std: float | Tensor) -> ResponseData:
    y = resolve(dist.LogNormal(data.eta, std))
    return promote(ResponseData, data, y=y)


def categorical(data: PredictorData) -> ResponseData:
    y = resolve(dist.Multinomial(total_count=1, logits=data.eta))
    return promote(ResponseData, data, y=y)


def weibull(data: PredictorData, shape: float | Tensor) -> ResponseData:
    scale = data.eta.exp().reciprocal()
    y = resolve(dist.Weibull(scale, shape))
    return promote(ResponseData, data, y=y)


def log_logistic(
    data: PredictorData, shape: float | Tensor
) -> ResponseData:
    scale = data.eta.exp().reciprocal()
    u = torch.rand_like(data.eta)
    y = scale * (u / (1.0 - u)).pow(1.0 / shape)
    return promote(ResponseData, data, y=y)


def gompertz(data: PredictorData, shape: float | Tensor) -> ResponseData:
    rate = data.eta.exp()
    u = torch.rand_like(data.eta)
    y = (1.0 / shape) * torch.log1p(-shape * u.log() / rate)
    return promote(ResponseData, data, y=y)


def constant_target(data: PredictorData, family: Family) -> ResponseData:
    """Pool eta over T, sample once per subject, broadcast y back."""
    pooled = replace(data, eta=data.eta.mean(dim=-2, keepdim=True))
    result = family(pooled)
    return replace(result, eta=data.eta, y=result.y.expand_as(data.eta))
