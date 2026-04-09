from collections.abc import Callable
from dataclasses import replace

import torch
import torch.distributions as dist
from torch import Tensor

from .states import InitialData, PredictorData, ResponseData

type Prior = dist.Distribution | Tensor
type FamilyFn = Callable[[PredictorData], tuple[ResponseData, dict[str, Tensor]]]


def resolve(prior: Prior, shape: tuple[int, ...] = ()) -> Tensor:
    if isinstance(prior, Tensor):
        return prior
    suffix = len(prior.batch_shape) + len(prior.event_shape)
    sample_shape = shape[:-suffix] if suffix else shape
    return (
        prior.rsample(sample_shape) if prior.has_rsample else prior.sample(sample_shape)
    )


def fixed_effects(
    data: InitialData, X: Prior, beta: Prior
) -> tuple[PredictorData, dict[str, Tensor]]:
    basis = resolve(X, (*data.draws, data.n, data.t, data.p))
    coefficients = resolve(beta, (*data.draws, 1, data.p, data.k))
    eta = basis @ coefficients
    coordinates = torch.arange(data.t, dtype=basis.dtype).unsqueeze(-1)  # [T, 1]
    coordinates = coordinates.expand_as(basis[..., :1])  # [*draws, N, T, 1]
    return (
        PredictorData(coordinates=coordinates, X=basis, eta=eta),
        {"beta": coefficients.squeeze(-3)},
    )


def random_effects(
    data: PredictorData, levels: int, q: int, W: Prior, B: Prior, b: Prior, index: int
) -> tuple[PredictorData, dict[str, Tensor]]:
    *batch, n, t, k = data.eta.shape
    # design choice: T=1 implies membership is constant over time. For
    # non-constant longitudinal membership, a user must pass a custom W
    membership = resolve(W, (*batch, n, 1, levels))
    basis = resolve(B, (*batch, n, t, q))
    coefficients = resolve(b, (*batch, levels, q, k))
    eta_re = torch.einsum(
        "...ntl,...ntr,...lrk->...ntk", membership, basis, coefficients
    )
    params = {f"W_{index}": membership, f"B_{index}": basis, f"b_{index}": coefficients}
    return replace(data, eta=data.eta + eta_re), params


def points(
    data: PredictorData, coordinates: Prior
) -> tuple[PredictorData, dict[str, Tensor]]:
    n, t = data.X.shape[-3], data.X.shape[-2]
    match coordinates:
        case Tensor():
            if coordinates.ndim == 1:
                coordinates = coordinates.unsqueeze(-1)  # [T] -> [T, 1]
            coords = coordinates.expand(n, t, -1)
        case dist.Distribution():
            increments = resolve(coordinates, (n, t))
            coords = increments.cumsum(dim=-1).unsqueeze(-1)  # [N, T, 1]
    return replace(data, coordinates=coords), {}


def gaussian(
    data: PredictorData, covariance: Prior
) -> tuple[ResponseData, dict[str, Tensor]]:
    K = data.eta.shape[-1]
    covariance = resolve(covariance, (K, K))
    if covariance.ndim < 2:
        covariance = covariance * torch.eye(K)
    y = resolve(dist.MultivariateNormal(data.eta, covariance))
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


def weibull(
    data: PredictorData, shape: float | Tensor
) -> tuple[ResponseData, dict[str, Tensor]]:
    scale = data.eta.exp().reciprocal()
    y = resolve(dist.Weibull(scale, shape))
    return ResponseData(**vars(data), y=y), {}


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
