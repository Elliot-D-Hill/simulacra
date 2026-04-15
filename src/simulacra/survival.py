import torch
import torch.distributions as dist
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .states import (
    DiscreteSurvivalData,
    EventTimeData,
    ResponseData,
    SurvivalData,
    promote,
)


def competing_risks(data: ResponseData) -> EventTimeData:
    min_time, min_idx = data.y.min(dim=-1)  # [N, T]
    is_winner = F.one_hot(min_idx, data.y.shape[-1]).bool()  # [N, T, K]
    event_time = torch.where(is_winner, data.y, torch.inf)
    censor_time = torch.where(is_winner, torch.inf, min_time.unsqueeze(-1))
    return promote(EventTimeData, data, event_time=event_time, censor_time=censor_time)


def censor(
    data: ResponseData,
    dropout: Float[Tensor, "n t 1"] | None = None,
    horizon: float | Tensor = torch.inf,
) -> SurvivalData:
    event_time = getattr(data, "event_time", data.y)
    prior_censor = getattr(data, "censor_time", torch.tensor(torch.inf))
    if dropout is None:
        n, t, _ = event_time.shape[-3:]
        dropout = dist.Exponential(1.0).sample((n, t, 1))
    rolling = data.points + horizon  # [N, T, 1]
    censor_time = torch.minimum(torch.minimum(prior_censor, dropout), rolling)
    observed_time = torch.minimum(event_time, censor_time)
    indicator = (event_time < censor_time).to(event_time.dtype)
    time_to_event = observed_time - data.points
    return promote(
        SurvivalData,
        data,
        event_time=event_time,
        censor_time=censor_time,
        indicator=indicator,
        observed_time=observed_time,
        time_to_event=time_to_event,
    )


def discretize(data: SurvivalData, boundaries: Tensor) -> DiscreteSurvivalData:
    interval_start = boundaries[:-1]  # [J]
    interval_end = boundaries[1:]  # [J]
    interval_width = interval_end - interval_start  # [J]
    tte = data.time_to_event.unsqueeze(-1)  # [..., K, 1]
    exposure = ((tte - interval_start) / interval_width).clamp(0, 1)
    in_interval = (tte > interval_start) & (tte <= interval_end)
    indicator = data.indicator.unsqueeze(-1).to(exposure.dtype)  # [..., K, 1]
    mask = indicator * in_interval.to(exposure.dtype) + (1.0 - indicator)
    discrete = exposure * mask  # [..., K, J]
    return promote(DiscreteSurvivalData, data, discrete_event_time=discrete)
