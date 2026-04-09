import torch
import torch.nn.functional as F
from torch import Tensor

from .states import (
    DiscreteSurvivalData,
    EventTimeData,
    ResponseData,
    SurvivalData,
)


def event_time(
    data: ResponseData, horizon: float | Tensor = torch.inf
) -> tuple[EventTimeData, dict[str, Tensor]]:
    latent = data.y  # [N, T, K]
    min_time, min_idx = latent.min(dim=-1)  # [N, T]
    is_winner = F.one_hot(min_idx, latent.shape[-1]).bool()  # [N, T, K]
    observed_event = torch.where(is_winner, latent, torch.inf)
    censor = torch.where(is_winner, torch.inf, min_time.unsqueeze(-1))
    censor = torch.minimum(censor, data.coordinates[..., :1] + horizon)
    return EventTimeData(**vars(data), event_time=observed_event, censor_time=censor), {}


def recurrent_events(
    data: ResponseData, horizon: float | Tensor = torch.inf
) -> tuple[EventTimeData, dict[str, Tensor]]:
    max_coord = data.coordinates[..., :1].max()
    censor = torch.full_like(data.y, max_coord.item())
    censor = torch.minimum(censor, data.coordinates[..., :1] + horizon)
    return EventTimeData(**vars(data), event_time=data.y, censor_time=censor), {}


def indicator(data: EventTimeData) -> tuple[SurvivalData, dict[str, Tensor]]:
    coords = data.coordinates[..., :1]  # [N, T, 1] or [T, 1]
    observed_time = torch.minimum(data.event_time, data.censor_time)
    ind = (data.event_time < data.censor_time).to(data.event_time.dtype)
    tte = observed_time - coords
    return (
        SurvivalData(
            **vars(data), indicator=ind, observed_time=observed_time, time_to_event=tte
        ),
        {},
    )


def discretize(
    data: SurvivalData, boundaries: Tensor
) -> tuple[DiscreteSurvivalData, dict[str, Tensor]]:
    interval_start = boundaries[:-1]  # [J]
    interval_end = boundaries[1:]  # [J]
    interval_width = interval_end - interval_start  # [J]
    tte = data.time_to_event.unsqueeze(-1)  # [..., K, 1]
    exposure = ((tte - interval_start) / interval_width).clamp(0, 1)
    in_interval = (tte > interval_start) & (tte <= interval_end)
    ind = data.indicator.unsqueeze(-1).to(exposure.dtype)  # [..., K, 1]
    mask = ind * in_interval.to(exposure.dtype) + (1.0 - ind)
    discrete = exposure * mask  # [..., K, J]
    return DiscreteSurvivalData(**vars(data), discrete_event_time=discrete), {}
