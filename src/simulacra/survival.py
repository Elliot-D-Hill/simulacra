import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .states import DiscreteSurvivalData, ResponseData, SurvivalData, promote


def censor(
    data: ResponseData,
    censor_time: Float[Tensor, "n t 1"] | None = None,
    horizon: float | Tensor = torch.inf,
) -> SurvivalData:
    """Race the k cause-specific times in y against the censoring time; the soonest wins.
    cause is the winner's index, or k when censored. censor_time defaults to the
    administrative deadline points + horizon (horizon = inf is no censoring), and an
    explicit tensor is capped by it. Single-event TTE is the k = 1 case.
    """
    k = data.y.shape[-1]
    deadline = data.points + horizon  # [n t 1]; admin cutoff, inf = none
    censor_time = (
        deadline if censor_time is None else torch.minimum(censor_time, deadline)
    )
    competitors = torch.cat(
        [data.y, censor_time], dim=-1
    )  # [n t k+1]; censor is column k
    observed_time, cause = competitors.min(
        dim=-1, keepdim=True
    )  # [n t 1]; cause k censored
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
    )  # [n t k j]; shared tte broadcasts against per-cause indicator
    return promote(DiscreteSurvivalData, data, discrete_event_time=discrete)
