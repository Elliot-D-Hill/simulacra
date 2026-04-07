from dataclasses import dataclass, field

from torch import Tensor


@dataclass(frozen=True)
class InitialData:
    draws: tuple[int, ...]
    n: int
    t: int
    p: int
    k: int


@dataclass(frozen=True)
class PredictorData:
    X: Tensor  # [N, T, p]
    eta: Tensor  # [N, T, K]
    tokens: Tensor | None = None  # [N, T]


@dataclass(frozen=True)
class ResponseData(PredictorData):
    y: Tensor = field(default_factory=Tensor)  # [N, T, K]


@dataclass(frozen=True)
class EventTimeData(ResponseData):
    event_time: Tensor = field(default_factory=Tensor)  # [N, T, K]


@dataclass(frozen=True)
class CensoredData(EventTimeData):
    censor_time: Tensor = field(default_factory=Tensor)  # [N, T, K]
