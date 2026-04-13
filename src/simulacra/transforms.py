from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import torch
import torch.distributions as dist
from torch import Tensor

from .states import CovariateData, PredictorData, Prior, RandomEffect, ResponseData

type Step[S, T] = Callable[[S], T]
type Run[S] = Step[tuple[int, ...], S]


def chain[S, M, T](first: Step[S, M], second: Step[M, T]) -> Step[S, T]:
    def chained(data: S) -> T:
        return second(first(data))

    return chained


def _format(v: object) -> str:
    match v:
        case Tensor() if v.ndim == 0:
            return f"tensor({v.item():.4g})"
        case Tensor():
            return f"Tensor{tuple(v.shape)}"
        case partial():
            return label(v.func, **v.keywords)
        case _:
            return getattr(v, "__name__", repr(v))


_WIDTH = 88


def label(transform: Callable[..., object], **kwargs: object) -> str:
    parts = [f"{k}={_format(v)}" for k, v in kwargs.items()]
    single = f"{transform.__name__}({', '.join(parts)})"
    if len(single) <= _WIDTH:
        return single
    body = ",\n    ".join(parts)
    return f"{transform.__name__}(\n    {body},\n)"


@dataclass(frozen=True)
class Pipeline[S]:
    run: Run[S]
    recipe: tuple[str, ...]

    def apply[T](self, transform: Callable[..., T], **kwargs: object) -> Pipeline[T]:
        return Pipeline(
            chain(self.run, partial(transform, **kwargs)),
            (*self.recipe, label(transform, **kwargs)),
        )

    def __repr__(self) -> str:
        return "\n.".join(self.recipe) or "Pipeline"


def resolve(prior: Prior, shape: tuple[int, ...] = ()) -> Tensor:
    if isinstance(prior, Tensor):
        return prior
    suffix = len(prior.batch_shape) + len(prior.event_shape)
    sample_shape = shape[:-suffix] if suffix else shape
    return (
        prior.rsample(sample_shape) if prior.has_rsample else prior.sample(sample_shape)
    )


def fixed_effects(data: CovariateData, k: int, beta: Prior) -> PredictorData:
    *batch, _, _, p = data.X.shape
    coefficient = resolve(beta, (*batch, 1, p, k))
    eta = data.X @ coefficient
    return PredictorData(
        X=data.X, coordinates=data.coordinates, eta=eta, beta=coefficient.squeeze(-3)
    )


def random_effects(
    data: PredictorData, levels: int, q: int, W: Prior, B: Prior, b: Prior
) -> PredictorData:
    *batch, n, t, k = data.eta.shape
    # design choice: T=1 implies membership is constant over time. For
    # non-constant longitudinal membership, a user must pass a custom W
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


def missing_x[S: CovariateData](data: S, proportion: float) -> S:
    mask = torch.rand_like(data.X) < proportion
    return replace(data, X=data.X.masked_fill(mask, float("nan")))


def missing_y[S: ResponseData](data: S, proportion: float) -> S:
    mask = torch.rand_like(data.y) < proportion
    return replace(data, y=data.y.masked_fill(mask, float("nan")))


def constant_y[S: ResponseData](data: S) -> S:
    return replace(data, y=data.y[..., :1, :].expand_as(data.y))


def min_max_scale[S: CovariateData](data: S, low: float = 0.0, high: float = 1.0) -> S:
    data_low = data.X.amin(dim=(-3, -2), keepdim=True)
    data_high = data.X.amax(dim=(-3, -2), keepdim=True)
    span = (data_high - data_low).clamp(min=1e-8)
    scaled = low + (data.X - data_low) / span * (high - low)
    return replace(data, X=scaled)


def z_score[S: CovariateData](data: S) -> S:
    mean = data.X.mean(dim=(-3, -2), keepdim=True)
    std = data.X.std(dim=(-3, -2), keepdim=True).clamp(min=1e-8)
    return replace(data, X=(data.X - mean) / std)


def activation(data: PredictorData, fn: Callable[[Tensor], Tensor]) -> PredictorData:
    return replace(data, eta=fn(data.eta))


def projection(data: PredictorData, output: int, weight: Prior) -> PredictorData:
    *batch, _, _, k_in = data.eta.shape
    w = resolve(weight, (*batch, 1, k_in, output))
    return replace(
        data,
        eta=data.eta @ w,
        projection_weight=(*data.projection_weight, w.squeeze(-3)),
    )


def tokenize[S: PredictorData](
    data: S, vocab_size: int, weight: Prior, temperature: float | Tensor = 1.0
) -> S:
    *batch, _, _, k_in = data.eta.shape
    w = resolve(weight, (*batch, 1, k_in, vocab_size))
    logits = data.eta @ w / temperature
    tokens = dist.Categorical(logits=logits).sample()
    return replace(data, tokens=tokens, token_weight=w.squeeze(-3))
