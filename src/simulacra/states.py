from dataclasses import MISSING, dataclass, fields
from typing import cast

from jaxtyping import Float
from torch import Tensor


def _format_field(val: object) -> str:
    match val:
        case Tensor() if val.ndim == 0:
            return f"tensor({val.item():.4g})"
        case Tensor():
            return str(list(val.shape))
        case tuple():
            tensors = cast(tuple[Tensor, ...], val)
            return "(" + ", ".join(str(list(v.shape)) for v in tensors) + ",)"
        case _:
            return repr(val)


def _data_repr(self: object) -> str:
    parts = [
        f"    {f.name}: {_format_field(getattr(self, f.name))}"
        for f in fields(self)  # type: ignore[arg-type]
        if f.default is MISSING or getattr(self, f.name) is not f.default
    ]
    body = "\n".join(parts)
    return f"{type(self).__name__}(\n{body}\n)"


@dataclass(frozen=True, repr=False, kw_only=True)
class RandomEffect:
    W: Float[Tensor, "n t levels"]
    B: Float[Tensor, "n t q"]
    b: Float[Tensor, "levels q k"]
    __repr__ = _data_repr


@dataclass(frozen=True, repr=False, kw_only=True)
class PredictorData:
    X: Float[Tensor, "n t p"]
    points: Float[Tensor, "n t 1"]
    eta: Float[Tensor, "n t k"]
    beta: Float[Tensor, "p k"]
    tokens: Tensor | None = None
    token_weight: Tensor | None = None
    random_effect: tuple[RandomEffect, ...] = ()
    __repr__ = _data_repr


@dataclass(frozen=True, repr=False, kw_only=True)
class ResponseData(PredictorData):
    y: Float[Tensor, "n t k"]


@dataclass(frozen=True, repr=False, kw_only=True)
class EventTimeData(ResponseData):
    event_time: Float[Tensor, "n t k"]
    censor_time: Float[Tensor, "n t 1"]


@dataclass(frozen=True, repr=False, kw_only=True)
class SurvivalData(EventTimeData):
    indicator: Float[Tensor, "n t k"]
    observed_time: Float[Tensor, "n t k"]
    time_to_event: Float[Tensor, "n t k"]


@dataclass(frozen=True, repr=False, kw_only=True)
class DiscreteSurvivalData(SurvivalData):
    discrete_event_time: Float[Tensor, "n t k j"]


def promote[T](cls: type[T], parent: object, **fields: object) -> T:
    return cls(**{**vars(parent), **fields})
