from dataclasses import dataclass, field

import torch.distributions as dist
from torch import Tensor

type Prior = dist.Distribution | Tensor


@dataclass(frozen=True)
class InitialData:
    draws: tuple[int, ...]
    n: int
    t: int
    p: int
    X: Prior
    coordinates: Prior


@dataclass(frozen=True)
class CovariateData:
    X: Tensor  # [*draws, N, T, p]
    coordinates: Tensor  # [*draws, N, T, D]


@dataclass(frozen=True)
class PredictorData(CovariateData):
    eta: Tensor  # [*draws, N, T, K]
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


class SimulationData(dict[str, Tensor]):
    def __repr__(self) -> str:
        fields = "\n".join(
            f"    {k}: {list(v.shape)}" for k, v in self.items()
        )
        return f"SimulationData(\n{fields}\n)"
