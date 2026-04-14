from collections.abc import Callable
from typing import Any, Final, NoReturn, Self

import torch
import torch.distributions as dist
from jaxtyping import Float
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
    missing_x,
    missing_y,
    random_effects,
    resolve,
    tokenize,
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


def _materialize(prior: Prior, shape: tuple[int, ...]) -> Tensor:
    return prior.expand(shape) if isinstance(prior, Tensor) else resolve(prior, shape)


def simulate(
    X: Float[Tensor, "*D n t p"], beta: Prior = UNIT_NORMAL, *, points: Prior = EXP1
) -> Predictor:
    *_, n, t, p = X.shape
    k = beta.shape[-1] if isinstance(beta, Tensor) else 1

    def run(draws: tuple[int, ...]) -> PredictorData:
        batch = (*draws, *X.shape[:-3])
        x = X.expand(*batch, n, t, p)
        coefficient = _materialize(beta, (*batch, p, k))
        if isinstance(points, Tensor):
            pts = points.expand(*batch, n, t, 1)
        else:
            increments = resolve(points, (*batch, n, t))
            pts = increments.cumsum(dim=-1).unsqueeze(-1)
        return fixed_effects(X=x, beta=coefficient, points=pts)

    return Predictor(
        Pipeline(run=run, recipe=(label(simulate, X=X, beta=beta, points=points),))
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
    Predictor,
    Response,
    PositiveSupportResponse,
    CompetingResponse,
    Survival,
    DiscreteSurvival,
)
