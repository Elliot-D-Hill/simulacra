from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Final, NoReturn, Self, overload

import torch
import torch.distributions as dist
from torch import Tensor

from .family import (
    Family,
    bernoulli,
    binomial,
    categorical,
    constant_target,
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
from .states import (
    CovariateData,
    DiscreteSurvivalData,
    EventTimeData,
    InitialData,
    PredictorData,
    Prior,
    ResponseData,
    SurvivalData,
)
from .survival import EXP1, censor, competing_risks, discretize
from .transforms import (
    Pipeline,
    activation,
    chain,
    covariates,
    fixed_effects,
    label,
    min_max_scale,
    missing_x,
    missing_y,
    points,
    projection,
    random_effects,
    resolve_design,
    tokenize,
    z_score,
)

UNIT_VARIANCE: Final[Tensor] = torch.tensor(1.0)
UNIT_NORMAL: Final[dist.Normal] = dist.Normal(0.0, UNIT_VARIANCE)


def _identity[T](x: T) -> T:
    return x


class Covariate:
    def __init__(self, pipeline: Pipeline[CovariateData]) -> None:
        self._pipeline = pipeline

    def __repr__(self) -> str:
        return "\n  .".join(self._pipeline.recipe) or type(self).__name__

    def __getattr__(self, name: str) -> NoReturn:
        raise guide(self, name, GRAPH)

    @step
    def z_score(self) -> Self:
        return type(self)(self._pipeline.apply(z_score))

    @step
    def min_max_scale(self, low: float = 0.0, high: float = 1.0) -> Self:
        return type(self)(self._pipeline.apply(min_max_scale, low=low, high=high))

    @step
    def fixed_effects(self, k: int = 1, beta: Prior = UNIT_NORMAL) -> Predictor:
        return Predictor(self._pipeline.apply(fixed_effects, k=k, beta=beta))


class Simulation:
    def __init__(self, n: int, t: int = 1, p: int = 1) -> None:
        coordinates = torch.arange(t, dtype=torch.float).unsqueeze(-1)
        initial = InitialData(
            draws=(), n=n, t=t, p=p, X=UNIT_NORMAL, coordinates=coordinates
        )
        self._pipeline = Pipeline(
            run=lambda draws: replace(initial, draws=draws),
            recipe=(f"Simulation(n={n}, t={t}, p={p})",),
        )

    @classmethod
    def _from_pipeline(cls, pipeline: Pipeline[InitialData]) -> Self:
        obj = cls.__new__(cls)
        obj._pipeline = pipeline
        return obj

    def __repr__(self) -> str:
        return "\n  .".join(self._pipeline.recipe) or type(self).__name__

    def __getattr__(self, name: str) -> NoReturn:
        raise guide(self, name, GRAPH)

    @step
    def covariates(self, X: Prior = UNIT_NORMAL) -> Self:
        return type(self)._from_pipeline(self._pipeline.apply(covariates, X=X))

    @step
    def points(self, coordinates: Prior = EXP1) -> Self:
        return type(self)._from_pipeline(
            self._pipeline.apply(points, coordinates=coordinates)
        )

    @step
    def z_score(self) -> Covariate:
        return Covariate(
            self._pipeline.then(chain(resolve_design, z_score), label(z_score))
        )

    @step
    def min_max_scale(self, low: float = 0.0, high: float = 1.0) -> Covariate:
        step_fn = chain(resolve_design, partial(min_max_scale, low=low, high=high))
        return Covariate(
            self._pipeline.then(step_fn, label(min_max_scale, low=low, high=high))
        )

    @step
    def fixed_effects(self, k: int = 1, beta: Prior = UNIT_NORMAL) -> Predictor:
        step_fn = chain(resolve_design, partial(fixed_effects, k=k, beta=beta))
        return Predictor(
            self._pipeline.then(step_fn, label(fixed_effects, k=k, beta=beta))
        )


class _Pipeline[S: PredictorData]:
    def __init__(self, pipeline: Pipeline[S]) -> None:
        self._pipeline = pipeline

    def __repr__(self) -> str:
        return "\n  .".join(self._pipeline.recipe) or type(self).__name__

    def __getattr__(self, name: str) -> NoReturn:
        raise guide(self, name, GRAPH)

    def draw(self, draws: int | None = None, seed: int | None = None) -> S:
        if seed is not None:
            torch.manual_seed(seed)  # type: ignore[no-untyped-call]
        batch = (draws,) if draws is not None else ()
        return self._pipeline.run(batch)


class _ResponsePipeline[S: ResponseData](_Pipeline[S]):
    @step
    def missing_x(self, proportion: float) -> Self:
        return type(self)(self._pipeline.apply(missing_x, proportion=proportion))

    @step
    def missing_y(self, proportion: float) -> Self:
        return type(self)(self._pipeline.apply(missing_y, proportion=proportion))


class Response(_ResponsePipeline[ResponseData]): ...


class PositiveSupportResponse(_ResponsePipeline[ResponseData]):
    @step
    def competing_risks(self) -> CompetingResponse:
        return CompetingResponse(self._pipeline.apply(competing_risks))

    @step
    def censor(
        self, dropout: Prior = EXP1, *, horizon: float | Tensor = torch.inf
    ) -> Survival:
        return Survival(self._pipeline.apply(censor, dropout=dropout, horizon=horizon))


class CompetingResponse(_ResponsePipeline[EventTimeData]):
    @step
    def censor(
        self, dropout: Prior = EXP1, *, horizon: float | Tensor = torch.inf
    ) -> Survival:
        return Survival(self._pipeline.apply(censor, dropout=dropout, horizon=horizon))


class Survival(_ResponsePipeline[SurvivalData]):
    @step
    def discretize(self, boundaries: Tensor) -> DiscreteSurvival:
        return DiscreteSurvival(self._pipeline.apply(discretize, boundaries=boundaries))


class DiscreteSurvival(_ResponsePipeline[DiscreteSurvivalData]): ...


class _FamilyPipeline(_Pipeline[PredictorData]):
    def __init__(
        self,
        pipeline: Pipeline[PredictorData],
        wrap: Callable[[Family], Family] = _identity,
    ) -> None:
        super().__init__(pipeline)
        self._wrap = wrap

    @overload
    def _family(
        self,
        fn: Callable[..., ResponseData],
        cls: type[Response] = ...,
        **kwargs: object,
    ) -> Response: ...
    @overload
    def _family(
        self,
        fn: Callable[..., ResponseData],
        cls: type[PositiveSupportResponse],
        **kwargs: object,
    ) -> PositiveSupportResponse: ...
    def _family(
        self,
        fn: Callable[..., ResponseData],
        cls: type[_ResponsePipeline[ResponseData]] = Response,
        **kwargs: object,
    ) -> _ResponsePipeline[ResponseData]:
        family = partial(fn, **kwargs)
        return cls(self._pipeline.then(self._wrap(family), label(fn, **kwargs)))

    @step
    def gaussian(self, covariance: Prior = UNIT_VARIANCE) -> Response:
        return self._family(gaussian, covariance=covariance)

    @step
    def poisson(self) -> Response:
        return self._family(poisson)

    @step
    def bernoulli(self) -> Response:
        return self._family(bernoulli)

    @step
    def binomial(self, num_trials: int = 1) -> Response:
        return self._family(binomial, num_trials=num_trials)

    @step
    def negative_binomial(self, concentration: float | Tensor) -> Response:
        return self._family(negative_binomial, concentration=concentration)

    @step
    def categorical(self) -> Response:
        return self._family(categorical)

    @step
    def gamma(self, concentration: float | Tensor) -> PositiveSupportResponse:
        return self._family(gamma, PositiveSupportResponse, concentration=concentration)

    @step
    def log_normal(self, std: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._family(log_normal, PositiveSupportResponse, std=std)

    @step
    def weibull(self, shape: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._family(weibull, PositiveSupportResponse, shape=shape)

    @step
    def exponential(self) -> PositiveSupportResponse:
        return self._family(exponential, PositiveSupportResponse)

    @step
    def log_logistic(self, shape: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._family(log_logistic, PositiveSupportResponse, shape=shape)

    @step
    def gompertz(self, shape: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._family(gompertz, PositiveSupportResponse, shape=shape)


class Predictor(_FamilyPipeline):
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
        # design choice: default to soft level assignments
        w = W or dist.Dirichlet(torch.ones(levels))
        return Predictor(
            self._pipeline.apply(random_effects, levels=levels, q=q, W=w, B=B, b=b)
        )

    @step
    def activation(self, fn: Callable[[Tensor], Tensor] = torch.relu) -> Predictor:
        return Predictor(
            self._pipeline.then(partial(activation, fn=fn), label(activation))
        )

    @step
    def projection(self, output: int, weight: Prior = UNIT_NORMAL) -> Predictor:
        return Predictor(self._pipeline.apply(projection, output=output, weight=weight))

    @step
    def tokenize(
        self,
        vocab_size: int,
        weight: Prior = UNIT_NORMAL,
        temperature: float | Tensor = 1.0,
    ) -> Predictor:
        return Predictor(
            self._pipeline.apply(
                tokenize, vocab_size=vocab_size, weight=weight, temperature=temperature
            )
        )

    @step
    def constant_target(self) -> ConstantPredictor:
        recipe = (*self._pipeline.recipe, label(constant_target))
        return ConstantPredictor(Pipeline(self._pipeline.run, recipe))


class ConstantPredictor(_FamilyPipeline):
    def __init__(self, pipeline: Pipeline[PredictorData]) -> None:
        super().__init__(pipeline, wrap=lambda f: partial(constant_target, family=f))


GRAPH: Final[Graph] = build_graph(
    Simulation,
    Covariate,
    Predictor,
    ConstantPredictor,
    Response,
    PositiveSupportResponse,
    CompetingResponse,
    Survival,
    DiscreteSurvival,
)
