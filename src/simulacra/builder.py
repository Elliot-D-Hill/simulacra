from collections.abc import Callable
from typing import Any, Final, NoReturn, Self

import torch
import torch.distributions as dist
from torch import Tensor

from .family import (
    bernoulli,
    binomial,
    categorical,
    exponential,
    gamma,
    gaussian,
    gompertz,
    log_logistic,
    log_normal,
    negative_binomial,
    poisson,
    weibull,
)
from .graph import Graph, build_graph, guide, step
from .pipeline import Pipeline, label
from .states import (
    CovariateData,
    DiscreteSurvivalData,
    EventTimeData,
    PredictorData,
    Prior,
    ResponseData,
    SurvivalData,
)
from .survival import censor, competing_risks, discretize
from .transforms import (
    activation,
    constant_y,
    fixed_effects,
    min_max_scale,
    missing_x,
    missing_y,
    projection,
    random_effects,
    resolve,
    tokenize,
    z_score,
)

UNIT_VARIANCE: Final[Tensor] = torch.tensor(1.0)
UNIT_NORMAL: Final[dist.Normal] = dist.Normal(0.0, UNIT_VARIANCE)
EXP1: Final[dist.Exponential] = dist.Exponential(1.0)


class _Pipeline[S]:
    def __init__(self, pipeline: Pipeline[S]) -> None:
        self._pipeline = pipeline

    def __repr__(self) -> str:
        return "\n.".join(self._pipeline.recipe) or type(self).__name__

    def __getattr__(self, name: str) -> NoReturn:
        raise guide(self, name, GRAPH)

    def _step[B: _Pipeline[Any]](
        self, cls: type[B], transform: Callable[..., Any], **kwargs: object
    ) -> B:
        return cls(self._pipeline.apply(transform, **kwargs))

    def draw(self, draws: int | None = None, seed: int | None = None) -> S:
        if seed is not None:
            torch.manual_seed(seed)  # type: ignore[no-untyped-call]
        batch = (draws,) if draws is not None else ()
        return self._pipeline.run(batch)


class Simulation(_Pipeline[CovariateData]):
    @step
    def z_score(self) -> Simulation:
        return self._step(Simulation, z_score)

    @step
    def min_max_scale(self, low: float = 0.0, high: float = 1.0) -> Simulation:
        return self._step(Simulation, min_max_scale, low=low, high=high)

    @step
    def fixed_effects(self, k: int = 1, beta: Prior = UNIT_NORMAL) -> Predictor:
        return self._step(Predictor, fixed_effects, k=k, beta=beta)


def _effective_shape(prior: Prior) -> tuple[int, ...]:
    match prior:
        case Tensor():
            return tuple(prior.shape)
        case dist.Distribution():
            return tuple(prior.batch_shape) + tuple(prior.event_shape)


def _check_axis(prior: Prior, axis: int, expected: int, param: str, name: str) -> None:
    shape = _effective_shape(prior)
    if -axis > len(shape):
        return
    observed = shape[axis]
    if observed != 1 and observed != expected:
        raise ValueError(
            f"{name} prior shape {shape} has size {observed} at axis {axis}, "
            f"expected {param}={expected} (or 1 to broadcast)."
        )


def _resolve_dim(
    explicit: int | None, sources: tuple[tuple[Prior, int], ...], default: int
) -> int:
    if explicit is not None:
        return explicit
    candidates = {
        _effective_shape(prior)[axis]
        for prior, axis in sources
        if -axis <= len(_effective_shape(prior)) and _effective_shape(prior)[axis] != 1
    }
    if len(candidates) > 1:
        raise ValueError(f"priors disagree on size: {sorted(candidates)}")
    return next(iter(candidates), default)


def simulate(
    n: int | None = None,
    t: int | None = None,
    p: int | None = None,
    X: Prior = UNIT_NORMAL,
    points: Prior = EXP1,
) -> Simulation:
    t_axis = -1 if isinstance(points, Tensor) and points.ndim == 1 else -2
    n = _resolve_dim(n, ((X, -3),), default=1)
    t = _resolve_dim(t, ((X, -2), (points, t_axis)), default=1)
    p = _resolve_dim(p, ((X, -1),), default=1)
    _check_axis(X, -3, n, "n", "X")
    _check_axis(X, -2, t, "t", "X")
    _check_axis(X, -1, p, "p", "X")
    _check_axis(points, t_axis, t, "t", "points")

    def run(draws: tuple[int, ...]) -> CovariateData:
        basis = resolve(X, (*draws, n, t, p))
        match points:
            case Tensor():
                pts = points
                if pts.ndim == 1:
                    pts = pts.unsqueeze(-1)
                pts = pts.expand_as(basis[..., :1])
            case dist.Distribution():
                increments = resolve(points, (*draws, n, t))
                pts = increments.cumsum(dim=-1).unsqueeze(-1)
        return CovariateData(X=basis, points=pts)

    return Simulation(
        Pipeline(run=run, recipe=(label(simulate, n=n, t=t, p=p, X=X, points=points),))
    )


class _ResponsePipeline[S: ResponseData](_Pipeline[S]):
    @step
    def missing_x(self, proportion: float) -> Self:
        return self._step(type(self), missing_x, proportion=proportion)

    @step
    def missing_y(self, proportion: float) -> Self:
        return self._step(type(self), missing_y, proportion=proportion)

    @step
    def constant_y(self) -> Self:
        return self._step(type(self), constant_y)


class Response(_ResponsePipeline[ResponseData]): ...


class _CensorPipeline[S: ResponseData](_ResponsePipeline[S]):
    @step
    def censor(
        self, dropout: Prior = EXP1, *, horizon: float | Tensor = torch.inf
    ) -> Survival:
        return self._step(Survival, censor, dropout=dropout, horizon=horizon)


class PositiveSupportResponse(_CensorPipeline[ResponseData]):
    @step
    def competing_risks(self) -> CompetingResponse:
        return self._step(CompetingResponse, competing_risks)


class CompetingResponse(_CensorPipeline[EventTimeData]): ...


class Survival(_ResponsePipeline[SurvivalData]):
    @step
    def discretize(self, boundaries: Tensor) -> DiscreteSurvival:
        return self._step(DiscreteSurvival, discretize, boundaries=boundaries)


class DiscreteSurvival(_ResponsePipeline[DiscreteSurvivalData]): ...


class Predictor(_Pipeline[PredictorData]):
    @step
    def random_effects(
        self,
        levels: int,
        q: int = 1,
        *,
        W: Prior | None = None,
        B: Prior = UNIT_NORMAL,
        b: Prior = UNIT_NORMAL,
    ) -> Predictor:
        return self._step(Predictor, random_effects, levels=levels, q=q, W=W, B=B, b=b)

    @step
    def activation(self, fn: Callable[[Tensor], Tensor] = torch.relu) -> Predictor:
        return self._step(Predictor, activation, fn=fn)

    @step
    def projection(self, output: int, weight: Prior = UNIT_NORMAL) -> Predictor:
        return self._step(Predictor, projection, output=output, weight=weight)

    @step
    def tokenize(
        self,
        vocab_size: int,
        weight: Prior = UNIT_NORMAL,
        temperature: float | Tensor = 1.0,
    ) -> Predictor:
        return self._step(
            Predictor,
            tokenize,
            vocab_size=vocab_size,
            weight=weight,
            temperature=temperature,
        )

    @step
    def gaussian(self, covariance: Prior = UNIT_VARIANCE) -> Response:
        return self._step(Response, gaussian, covariance=covariance)

    @step
    def poisson(self) -> Response:
        return self._step(Response, poisson)

    @step
    def bernoulli(self) -> Response:
        return self._step(Response, bernoulli)

    @step
    def binomial(self, num_trials: int = 1) -> Response:
        return self._step(Response, binomial, num_trials=num_trials)

    @step
    def negative_binomial(self, concentration: float | Tensor) -> Response:
        return self._step(Response, negative_binomial, concentration=concentration)

    @step
    def categorical(self) -> Response:
        return self._step(Response, categorical)

    @step
    def gamma(self, concentration: float | Tensor) -> PositiveSupportResponse:
        return self._step(PositiveSupportResponse, gamma, concentration=concentration)

    @step
    def log_normal(self, std: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._step(PositiveSupportResponse, log_normal, std=std)

    @step
    def weibull(self, shape: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._step(PositiveSupportResponse, weibull, shape=shape)

    @step
    def exponential(self) -> PositiveSupportResponse:
        return self._step(PositiveSupportResponse, exponential)

    @step
    def log_logistic(self, shape: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._step(PositiveSupportResponse, log_logistic, shape=shape)

    @step
    def gompertz(self, shape: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._step(PositiveSupportResponse, gompertz, shape=shape)


GRAPH: Final[Graph] = build_graph(
    Simulation,
    Predictor,
    Response,
    PositiveSupportResponse,
    CompetingResponse,
    Survival,
    DiscreteSurvival,
)
