import torch
import torch.distributions as dist
from jaxtyping import Float
from torch import Tensor

from .pipeline import Step
from .states import PredictorData, ResponseData, promote

type Family = Step[PredictorData, ResponseData]


def gaussian(
    data: PredictorData, covariance: Float[Tensor, "k k"] | None = None
) -> ResponseData:
    k = data.eta.shape[-1]
    cov = torch.eye(k) if covariance is None else covariance
    d = dist.MultivariateNormal(data.eta, cov)
    y = d.rsample()
    return promote(ResponseData, data, y=y)


def poisson(data: PredictorData) -> ResponseData:
    y = dist.Poisson(data.eta.exp()).sample()
    return promote(ResponseData, data, y=y)


def bernoulli(data: PredictorData) -> ResponseData:
    y = dist.Binomial(total_count=1, logits=data.eta).sample()
    return promote(ResponseData, data, y=y)


def binomial(data: PredictorData, num_trials: int) -> ResponseData:
    y = dist.Binomial(total_count=num_trials, logits=data.eta).sample()
    return promote(ResponseData, data, y=y)


def negative_binomial(
    data: PredictorData, concentration: float | Tensor
) -> ResponseData:
    y = dist.NegativeBinomial(concentration, logits=data.eta).sample()
    return promote(ResponseData, data, y=y)


def gamma(data: PredictorData, concentration: float | Tensor) -> ResponseData:
    y = dist.Gamma(concentration, data.eta.exp().reciprocal()).rsample()
    return promote(ResponseData, data, y=y)


def log_normal(data: PredictorData, std: float | Tensor) -> ResponseData:
    y = dist.LogNormal(data.eta, std).rsample()
    return promote(ResponseData, data, y=y)


def categorical(data: PredictorData) -> ResponseData:
    y = dist.Multinomial(total_count=1, logits=data.eta).sample()
    return promote(ResponseData, data, y=y)


def multinomial(data: PredictorData, num_trials: int) -> ResponseData:
    y = dist.Multinomial(total_count=num_trials, logits=data.eta).sample()
    return promote(ResponseData, data, y=y)


def beta(data: PredictorData, concentration: float | Tensor) -> ResponseData:
    mean = torch.sigmoid(data.eta)
    y = dist.Beta(mean * concentration, (1.0 - mean) * concentration).rsample()
    return promote(ResponseData, data, y=y)


def dirichlet(data: PredictorData, concentration: float | Tensor) -> ResponseData:
    alpha = concentration * torch.softmax(data.eta, dim=-1)
    y = dist.Dirichlet(alpha).rsample()
    return promote(ResponseData, data, y=y)


def exponential(data: PredictorData) -> ResponseData:
    y = dist.Exponential(data.eta.exp()).rsample()
    return promote(ResponseData, data, y=y)


def weibull(data: PredictorData, shape: float | Tensor) -> ResponseData:
    scale = data.eta.exp().reciprocal()
    y = dist.Weibull(scale, shape).rsample()
    return promote(ResponseData, data, y=y)


def log_logistic(data: PredictorData, shape: float | Tensor) -> ResponseData:
    scale = data.eta.exp().reciprocal()
    u = torch.rand_like(data.eta)
    y = scale * (u / (1.0 - u)).pow(1.0 / shape)
    return promote(ResponseData, data, y=y)


def gompertz(data: PredictorData, shape: float | Tensor) -> ResponseData:
    rate = data.eta.exp()
    u = torch.rand_like(data.eta)
    y = (1.0 / shape) * torch.log1p(-shape * u.log() / rate)
    return promote(ResponseData, data, y=y)
