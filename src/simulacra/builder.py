from collections.abc import Callable
from typing import Any, Final, NoReturn, Self

import torch
import torch.distributions as dist
from jaxtyping import Float
from torch import Tensor

from .causal import dose_response, treatment
from .family import (
    bernoulli,
    beta,
    binomial,
    categorical,
    dirichlet,
    exponential,
    gamma,
    gaussian,
    gompertz,
    log_logistic,
    log_normal,
    multinomial,
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
    tokenize,
)


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

    def draw(self, seed: int | None = None) -> S:
        if seed is not None:
            torch.manual_seed(seed)  # type: ignore[no-untyped-call]
        return self._pipeline.run()


def _default_points(n: int, t: int) -> Tensor:
    return dist.Exponential(1.0).sample((n, t, 1)).cumsum(dim=-2)


def simulate(
    X: Float[Tensor, "n t p"],
    beta: Float[Tensor, "p k"] | None = None,
    points: Float[Tensor, "n t 1"] | None = None,
) -> Predictor:
    n, t, p = X.shape

    def run() -> PredictorData:
        coefficients = torch.randn(p, 1) if beta is None else beta
        pts = _default_points(n, t) if points is None else points
        return fixed_effects(X=X, beta=coefficients, points=pts)

    beta_label = beta if beta is not None else (p, 1)
    points_label = points if points is not None else (n, t, 1)
    return Predictor(
        Pipeline(
            run=run,
            recipe=(label(simulate, X=X, beta=beta_label, points=points_label),),
        )
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
        self,
        dropout: Float[Tensor, "n t 1"] | None = None,
        horizon: float | Tensor = torch.inf,
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
        W: Float[Tensor, "n t levels"],
        B: Float[Tensor, "n t q"],
        b: Float[Tensor, "levels q k"] | None = None,
    ) -> Predictor:
        return self._step(Predictor, random_effects, W=W, B=B, b=b)

    @step
    def activation(self, fn: Callable[[Tensor], Tensor] = torch.relu) -> Predictor:
        return self._step(Predictor, activation, fn=fn)

    @step
    def tokenize(
        self, weight: Float[Tensor, "k vocab_size"], temperature: float | Tensor = 1.0
    ) -> Predictor:
        return self._step(Predictor, tokenize, weight=weight, temperature=temperature)

    @step
    def treatment(
        self,
        tau: Float[Tensor, "J k"] | float | Tensor,
        gamma: Float[Tensor, "p J"] | None = None,
    ) -> Predictor:
        return self._step(Predictor, treatment, tau=tau, gamma=gamma)

    @step
    def dose_response(
        self,
        tau: float | Tensor,
        gamma: Float[Tensor, "p 1"] | None = None,
        sigma: float | Tensor = 1.0,
    ) -> Predictor:
        return self._step(Predictor, dose_response, tau=tau, gamma=gamma, sigma=sigma)

    @step
    def gaussian(self, covariance: Float[Tensor, "k k"] | None = None) -> Response:
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
    def multinomial(self, num_trials: int) -> Response:
        return self._step(Response, multinomial, num_trials=num_trials)

    @step
    def beta(self, concentration: float | Tensor) -> Response:
        return self._step(Response, beta, concentration=concentration)

    @step
    def dirichlet(self, concentration: float | Tensor) -> Response:
        return self._step(Response, dirichlet, concentration=concentration)

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
