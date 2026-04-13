from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import torch
import torch.distributions as dist
from torch import Tensor

from .states import CovariateData, InitialData, PredictorData, Prior, ResponseData

type Params = dict[str, Tensor]
type Step[S, T] = Callable[[S], tuple[T, Params]]
type Run[S] = Step[tuple[int, ...], S]


def chain[S, M, T](first: Step[S, M], second: Step[M, T]) -> Step[S, T]:
    def chained(data: S) -> tuple[T, Params]:
        mid, params1 = first(data)
        out, params2 = second(mid)
        return out, {**params1, **params2}

    return chained


def suffixed[S, T](fn: Step[S, T], index: int) -> Step[S, T]:
    def wrapped(data: S) -> tuple[T, Params]:
        new_data, params = fn(data)
        return new_data, {f"{k}_{index}": v for k, v in params.items()}

    return wrapped


def _format(v: object) -> str:
    if isinstance(v, Tensor):
        return f"tensor({v.item():.4g})" if v.ndim == 0 else f"Tensor{tuple(v.shape)}"
    return repr(v)


def _label(fn: Callable[..., object], **kwargs: object) -> str:
    parts = ", ".join(f"{k}={_format(v)}" for k, v in kwargs.items())
    return f"{fn.__name__}({parts})"


@dataclass(frozen=True)
class Pipeline[S]:
    run: Run[S]
    recipe: tuple[str, ...]

    def then[T](self, step: Step[S, T], label: str) -> "Pipeline[T]":
        return Pipeline(chain(self.run, step), (*self.recipe, label))

    def apply[T](
        self, fn: Callable[..., tuple[T, Params]], **kwargs: object
    ) -> "Pipeline[T]":
        step = partial(fn, **kwargs) if kwargs else fn
        return Pipeline(chain(self.run, step), (*self.recipe, _label(fn, **kwargs)))

    def __repr__(self) -> str:
        return "\n  .".join(self.recipe) or "Pipeline"


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
    data_low = data.X.amin(dim=(-3, -2), keepdim=True)
    data_high = data.X.amax(dim=(-3, -2), keepdim=True)
    span = (data_high - data_low).clamp(min=1e-8)
    scaled = low + (data.X - data_low) / span * (high - low)
    return replace(data, X=scaled), {}


def z_score[S: CovariateData](data: S) -> tuple[S, Params]:
    mean = data.X.mean(dim=(-3, -2), keepdim=True)
    std = data.X.std(dim=(-3, -2), keepdim=True).clamp(min=1e-8)
    return replace(data, X=(data.X - mean) / std), {}


def activation(
    data: PredictorData, fn: Callable[[Tensor], Tensor]
) -> tuple[PredictorData, Params]:
    return replace(data, eta=fn(data.eta)), {}


def projection(
    data: PredictorData, output: int, weight: Prior
) -> tuple[PredictorData, Params]:
    *batch, _, _, k_in = data.eta.shape
    w = resolve(weight, (*batch, 1, k_in, output))
    return replace(data, eta=data.eta @ w), {"weight": w.squeeze(-3)}


def tokenize[S: PredictorData](
    data: S, vocab_size: int, weight: Prior, temperature: float | Tensor = 1.0
) -> tuple[S, Params]:
    *batch, _, _, k_in = data.eta.shape
    w = resolve(weight, (*batch, 1, k_in, vocab_size))
    logits = data.eta @ w / temperature
    tokens = dist.Categorical(logits=logits).sample()
    return replace(data, tokens=tokens), {"weight": w.squeeze(-3)}
