from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


def promote[T](data: object, cls: type[T], **kwargs: Any) -> T:
    return cls(**{**vars(data), **kwargs})


@dataclass(frozen=True)
class InitialData:
    draws: tuple[int, ...]
    n: int
    t: int
    p: int
    k: int


@dataclass(frozen=True)
class PredictorData:
    coordinates: Tensor  # [N, T, D]
    X: Tensor  # [N, T, p]
    eta: Tensor  # [N, T, K]
    tokens: Tensor | None = None  # [N, T]


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
