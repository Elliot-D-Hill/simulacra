from dataclasses import dataclass, field

import torch.distributions as dist
from torch import Tensor

type Prior = dist.Distribution | Tensor


@dataclass(frozen=True)
class CovariateData:
    X: Tensor  # [*draws, N, T, p]
    coordinates: Tensor  # [*draws, N, T, D]


@dataclass(frozen=True)
class RandomEffect:
    W: Tensor  # [*batch, N, 1, levels]
    B: Tensor  # [*batch, N, T, q]
    b: Tensor  # [*batch, levels, q, K]


@dataclass(frozen=True)
class PredictorData(CovariateData):
    eta: Tensor  # [*draws, N, T, K]
    beta: Tensor  # [*batch, p, K]
    tokens: Tensor | None = None  # [N, T]
    token_weight: Tensor | None = None  # [*batch, K_in, vocab_size]
    random_effect: tuple[RandomEffect, ...] = ()
    projection_weight: tuple[Tensor, ...] = ()


@dataclass(frozen=True)
class ResponseData(PredictorData):
    y: Tensor = field(default_factory=Tensor)  # [N, T, K]


@dataclass(frozen=True)
class EventTimeData(ResponseData):
    event_time: Tensor = field(default_factory=Tensor)  # [N, T, K]
    censor_time: Tensor = field(default_factory=Tensor)  # [N, T, K]


@dataclass(frozen=True)
class SurvivalData(EventTimeData):
    indicator: Tensor = field(default_factory=Tensor)  # [N, T, K]
    observed_time: Tensor = field(default_factory=Tensor)  # [N, T, K]
    time_to_event: Tensor = field(default_factory=Tensor)  # [N, T, K]


@dataclass(frozen=True)
class DiscreteSurvivalData(SurvivalData):
    discrete_event_time: Tensor = field(default_factory=Tensor)  # [N, T, K, J]


def promote[T](cls: type[T], parent: object, **fields: object) -> T:
    return cls(**{**vars(parent), **fields})
