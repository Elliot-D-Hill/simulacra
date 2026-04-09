import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import Tensor

from .states import (
    DiscreteSurvivalData,
    EventTimeData,
    ResponseData,
    SurvivalData,
)
from .transforms import Prior, resolve


def competing_risks(
    data: ResponseData,
) -> tuple[EventTimeData, dict[str, Tensor]]:
    latent = data.y  # [N, T, K]
    min_time, min_idx = latent.min(dim=-1)  # [N, T]
    is_winner = F.one_hot(min_idx, latent.shape[-1]).bool()  # [N, T, K]
    event = torch.where(is_winner, latent, torch.inf)
    censor = torch.where(is_winner, torch.inf, min_time.unsqueeze(-1))
    return EventTimeData(**vars(data), event_time=event, censor_time=censor), {}


def censor(
    data: ResponseData,
    dropout: Prior | None = None,
    *,
    horizon: float | Tensor = torch.inf,
) -> tuple[SurvivalData, dict[str, Tensor]]:
    if not isinstance(data, EventTimeData):
        data = EventTimeData(
            **vars(data),
            event_time=data.y,
            censor_time=torch.full_like(data.y, torch.inf),
        )
    if dropout is None:
        t_max = data.coordinates[..., -1:, :1].clamp(min=1.0)  # [*batch, N, 1, 1]
        dropout = dist.Uniform(torch.zeros(()), t_max)
    absolute = resolve(dropout, (*data.event_time.shape[:-2], 1, 1))
    rolling = data.coordinates[..., :1] + horizon  # [*batch, N, T, 1]
    ct = torch.minimum(data.censor_time, absolute)
    ct = torch.minimum(ct, rolling)
    coords = data.coordinates[..., :1]  # [N, T, 1]
    observed_time = torch.minimum(data.event_time, ct)
    ind = (data.event_time < ct).to(data.event_time.dtype)
    time_to_event = observed_time - coords
    return SurvivalData(
        **{**vars(data), "censor_time": ct},
        indicator=ind,
        observed_time=observed_time,
        time_to_event=time_to_event,
    ), {}


def discretize(
    data: SurvivalData, boundaries: Tensor
) -> tuple[DiscreteSurvivalData, dict[str, Tensor]]:
    interval_start = boundaries[:-1]  # [J]
    interval_end = boundaries[1:]  # [J]
    interval_width = interval_end - interval_start  # [J]
    time_to_event = data.time_to_event.unsqueeze(-1)  # [..., K, 1]
    exposure = ((time_to_event - interval_start) / interval_width).clamp(0, 1)
    in_interval = (time_to_event > interval_start) & (time_to_event <= interval_end)
    ind = data.indicator.unsqueeze(-1).to(exposure.dtype)  # [..., K, 1]
    mask = ind * in_interval.to(exposure.dtype) + (1.0 - ind)
    discrete = exposure * mask  # [..., K, J]
    return DiscreteSurvivalData(**vars(data), discrete_event_time=discrete), {}
