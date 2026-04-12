from collections.abc import Callable
from dataclasses import replace

import torch
import torch.distributions as dist
from torch import Tensor

from .states import CovariateData, InitialData, PredictorData, Prior, ResponseData

type Params = dict[str, Tensor]


def resolve(prior: Prior, shape: tuple[int, ...] = ()) -> Tensor:
    if isinstance(prior, Tensor):
        return prior
    suffix = len(prior.batch_shape) + len(prior.event_shape)
    sample_shape = shape[:-suffix] if suffix else shape
    return (
        prior.rsample(sample_shape) if prior.has_rsample else prior.sample(sample_shape)
    )


def resolve_design(data: InitialData) -> tuple[CovariateData, Params]:
    basis = resolve(data.X, (*data.draws, data.n, data.t, data.p))
    match data.coordinates:
        case Tensor():
            coords = data.coordinates
            if coords.ndim == 1:
                coords = coords.unsqueeze(-1)
            coords = coords.expand_as(basis[..., :1])
        case dist.Distribution():
            increments = resolve(data.coordinates, (*data.draws, data.n, data.t))
            coords = increments.cumsum(dim=-1).unsqueeze(-1)
    return CovariateData(X=basis, coordinates=coords), {}


def fixed_effects(
    data: CovariateData, k: int, beta: Prior
) -> tuple[PredictorData, Params]:
    *batch, _, _, p = data.X.shape
    coefficient = resolve(beta, (*batch, 1, p, k))
    eta = data.X @ coefficient
    return (
        PredictorData(X=data.X, coordinates=data.coordinates, eta=eta),
        {"beta": coefficient.squeeze(-3)},
    )


def random_effects(
    data: PredictorData, levels: int, q: int, W: Prior, B: Prior, b: Prior
) -> tuple[PredictorData, Params]:
    *batch, n, t, k = data.eta.shape
    # design choice: T=1 implies membership is constant over time. For
    # non-constant longitudinal membership, a user must pass a custom W
    membership = resolve(W, (*batch, n, 1, levels))
    basis = resolve(B, (*batch, n, t, q))
    coefficient = resolve(b, (*batch, levels, q, k))
    eta = torch.einsum("...ntl,...ntr,...lrk->...ntk", membership, basis, coefficient)
    return replace(data, eta=data.eta + eta), {
        "W": membership,
        "B": basis,
        "b": coefficient,
    }


def covariates(data: InitialData, X: Prior) -> tuple[InitialData, Params]:
    return replace(data, X=X), {}


def points(data: InitialData, coordinates: Prior) -> tuple[InitialData, Params]:
    return replace(data, coordinates=coordinates), {}


def missing_x[S: CovariateData](data: S, proportion: float) -> tuple[S, Params]:
    mask = torch.rand_like(data.X) < proportion
    return replace(data, X=data.X.masked_fill(mask, float("nan"))), {}


def missing_y[S: ResponseData](data: S, proportion: float) -> tuple[S, Params]:
    mask = torch.rand_like(data.y) < proportion
    return replace(data, y=data.y.masked_fill(mask, float("nan"))), {}


def min_max_scale[S: CovariateData](
    data: S, low: float = 0.0, high: float = 1.0
) -> tuple[S, Params]:
    X = data.X
    data_low = X.amin(dim=(-3, -2), keepdim=True)
    data_high = X.amax(dim=(-3, -2), keepdim=True)
    span = (data_high - data_low).clamp(min=1e-8)
    scaled = low + (X - data_low) / span * (high - low)
    return replace(data, X=scaled), {}


def z_score[S: CovariateData](data: S) -> tuple[S, Params]:
    X = data.X
    mean = X.mean(dim=(-3, -2), keepdim=True)
    std = X.std(dim=(-3, -2), keepdim=True).clamp(min=1e-8)
    return replace(data, X=(X - mean) / std), {}


def activation(
    data: PredictorData, fn: Callable[[Tensor], Tensor]
) -> tuple[PredictorData, Params]:
    return replace(data, eta=fn(data.eta)), {}


def projection(
    data: PredictorData, output: int, weight: Prior
) -> tuple[PredictorData, Params]:
    *batch, _, _, k_in = data.eta.shape
    w = resolve(weight, (*batch, k_in, output))
    return replace(data, eta=data.eta @ w.unsqueeze(-3)), {"weight": w}


def tokenize[S: PredictorData](data: S, vocab_size: int) -> tuple[S, Params]:
    weight = torch.randn(data.eta.shape[-1], vocab_size)
    prob = torch.softmax(data.eta @ weight, dim=-1)
    return replace(data, tokens=resolve(dist.Categorical(prob))), {}
