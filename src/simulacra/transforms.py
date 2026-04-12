from collections.abc import Callable
from dataclasses import replace

import torch
import torch.distributions as dist
from torch import Tensor

from .states import InitialData, PredictorData, ResponseData

type Prior = dist.Distribution | Tensor
type Params = dict[str, Tensor]


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
) -> tuple[PredictorData, Params]:
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
    data: PredictorData, levels: int, q: int, W: Prior, B: Prior, b: Prior
) -> tuple[PredictorData, Params]:
    *batch, n, t, k = data.eta.shape
    # design choice: T=1 implies membership is constant over time. For
    # non-constant longitudinal membership, a user must pass a custom W
    membership = resolve(W, (*batch, n, 1, levels))
    basis = resolve(B, (*batch, n, t, q))
    coefficients = resolve(b, (*batch, levels, q, k))
    eta = torch.einsum("...ntl,...ntr,...lrk->...ntk", membership, basis, coefficients)
    return replace(data, eta=data.eta + eta), {
        "W": membership,
        "B": basis,
        "b": coefficients,
    }


def points(data: PredictorData, coordinates: Prior) -> tuple[PredictorData, Params]:
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


def missing_x[S: PredictorData](data: S, proportion: float) -> tuple[S, Params]:
    mask = torch.rand_like(data.X) < proportion
    return replace(data, X=data.X.masked_fill(mask, float("nan"))), {}


def missing_y[S: ResponseData](data: S, proportion: float) -> tuple[S, Params]:
    mask = torch.rand_like(data.y) < proportion
    return replace(data, y=data.y.masked_fill(mask, float("nan"))), {}


def min_max_scale[S: PredictorData](
    data: S, low: float = 0.0, high: float = 1.0
) -> tuple[S, Params]:
    X = data.X
    data_low = X.amin(dim=(-3, -2), keepdim=True)
    data_high = X.amax(dim=(-3, -2), keepdim=True)
    span = (data_high - data_low).clamp(min=1e-8)
    scaled = low + (X - data_low) / span * (high - low)
    return replace(data, X=scaled), {}


def z_score[S: PredictorData](data: S) -> tuple[S, Params]:
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
    weights = torch.randn(data.eta.shape[-1], vocab_size)
    probs = torch.softmax(data.eta @ weights, dim=-1)
    return replace(data, tokens=resolve(dist.Categorical(probs))), {}
