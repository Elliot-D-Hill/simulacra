import torch
import torch.distributions as dist
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .states import DiscreteSurvivalData, ResponseData, SurvivalData, promote


def censor(
    data: ResponseData,
    dropout: Float[Tensor, "n t 1"] | None = None,
    horizon: float | Tensor = torch.inf,
) -> SurvivalData:
    """Race the k cause-specific times in y against the censoring time (dropout, capped
    at points + horizon); the soonest wins. cause is the winner's index, or k when
    censored. Single-event TTE is the k = 1 case.
    """
    n, t, k = data.y.shape
    if dropout is None:
        dropout = dist.Exponential(1.0).sample((n, t, 1))
    censor_time = torch.minimum(dropout, data.points + horizon)  # [n t 1]
    competitors = torch.cat(
        [data.y, censor_time], dim=-1
    )  # [n t k+1]; censor is column k
    observed_time, cause = competitors.min(
        dim=-1, keepdim=True
    )  # [n t 1]; cause 0..k-1, k censored
    indicator = F.one_hot(cause.squeeze(-1), k + 1)[..., :k].to(data.y.dtype)  # [n t k]
    time_to_event = observed_time - data.points  # [n t 1]
    return promote(
        SurvivalData,
        data,
        censor_time=censor_time,
        time_to_event=time_to_event,
        indicator=indicator,
        cause=cause,
    )


def discretize(data: SurvivalData, boundaries: Tensor) -> DiscreteSurvivalData:
    interval_start = boundaries[:-1]  # [J]
    interval_end = boundaries[1:]  # [J]
    interval_width = interval_end - interval_start  # [J]
    tte = data.time_to_event.unsqueeze(-1)  # [n t 1 1]
    exposure = ((tte - interval_start) / interval_width).clamp(0, 1)  # [n t 1 j]
    in_interval = (tte > interval_start) & (tte <= interval_end)  # [n t 1 j]
    indicator = data.indicator.unsqueeze(-1).to(exposure.dtype)  # [n t k 1]
    mask = indicator * in_interval.to(exposure.dtype) + (1.0 - indicator)  # [n t k j]
    discrete = (
        exposure * mask
    )  # [n t k j]; shared tte broadcasts against the per-cause indicator
    return promote(DiscreteSurvivalData, data, discrete_event_time=discrete)
