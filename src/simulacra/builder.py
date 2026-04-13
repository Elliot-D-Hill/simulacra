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
    SimulationData,
    SimulationParams,
    SurvivalData,
)
from .survival import EXP1, censor, competing_risks, discretize
from .transforms import (
    Pipeline,
    label,
    activation,
    chain,
    covariates,
    fixed_effects,
    min_max_scale,
    missing_x,
    missing_y,
    points,
    projection,
    random_effects,
    resolve_design,
    suffixed,
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
    def fixed_effects(self, k: int = 1, beta: Prior = UNIT_NORMAL) -> "Predictor":
        return Predictor(self._pipeline.apply(fixed_effects, k=k, beta=beta))


class Simulation:
    def __init__(self, n: int, t: int = 1, p: int = 1) -> None:
        coordinates = torch.arange(t, dtype=torch.float).unsqueeze(-1)
        initial = InitialData(
            draws=(), n=n, t=t, p=p, X=UNIT_NORMAL, coordinates=coordinates
        )
        self._pipeline = Pipeline(
            run=lambda draws: (replace(initial, draws=draws), {}),
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
    def fixed_effects(self, k: int = 1, beta: Prior = UNIT_NORMAL) -> "Predictor":
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

    def draw(
        self, draws: int | None = None, seed: int | None = None
    ) -> tuple[SimulationData, SimulationParams]:
        if seed is not None:
            torch.manual_seed(seed)  # type: ignore[no-untyped-call]
        batch = (draws,) if draws is not None else ()
        data, params = self._pipeline.run(batch)
        tensor_data = {k: v for k, v in vars(data).items() if v is not None}
        squeezed = {k: v.squeeze(-2) for k, v in tensor_data.items()}
        return SimulationData(squeezed), SimulationParams(params)


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
    def competing_risks(self) -> "CompetingResponse":
        return CompetingResponse(self._pipeline.apply(competing_risks))

    @step
    def censor(
        self, dropout: Prior = EXP1, *, horizon: float | Tensor = torch.inf
    ) -> "Survival":
        return Survival(self._pipeline.apply(censor, dropout=dropout, horizon=horizon))


class CompetingResponse(_ResponsePipeline[EventTimeData]):
    @step
    def censor(
        self, dropout: Prior = EXP1, *, horizon: float | Tensor = torch.inf
    ) -> "Survival":
        return Survival(self._pipeline.apply(censor, dropout=dropout, horizon=horizon))


class Survival(_ResponsePipeline[SurvivalData]):
    @step
    def discretize(self, boundaries: Tensor) -> "DiscreteSurvival":
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
    def _apply(self, family: Family, label: str) -> Response: ...
    @overload
    def _apply(
        self, family: Family, label: str, cls: type[PositiveSupportResponse]
    ) -> PositiveSupportResponse: ...
    def _apply(
        self,
        family: Family,
        label: str,
        cls: type[_ResponsePipeline[ResponseData]] = Response,
    ) -> _ResponsePipeline[ResponseData]:
        return cls(self._pipeline.then(self._wrap(family), label))

    @step
    def gaussian(self, covariance: Prior = UNIT_VARIANCE) -> Response:
        return self._apply(
            partial(gaussian, covariance=covariance),
            label(gaussian, covariance=covariance),
        )

    @step
    def poisson(self) -> Response:
        return self._apply(poisson, label(poisson))

    @step
    def bernoulli(self) -> Response:
        return self._apply(bernoulli, label(bernoulli))

    @step
    def binomial(self, num_trials: int = 1) -> Response:
        return self._apply(
            partial(binomial, num_trials=num_trials),
            label(binomial, num_trials=num_trials),
        )

    @step
    def negative_binomial(self, concentration: float | Tensor) -> Response:
        return self._apply(
            partial(negative_binomial, concentration=concentration),
            label(negative_binomial, concentration=concentration),
        )

    @step
    def categorical(self) -> Response:
        return self._apply(categorical, label(categorical))

    @step
    def gamma(self, concentration: float | Tensor) -> PositiveSupportResponse:
        return self._apply(
            partial(gamma, concentration=concentration),
            label(gamma, concentration=concentration),
            PositiveSupportResponse,
        )

    @step
    def log_normal(self, std: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._apply(
            partial(log_normal, std=std),
            label(log_normal, std=std),
            PositiveSupportResponse,
        )

    @step
    def weibull(self, shape: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._apply(
            partial(weibull, shape=shape),
            label(weibull, shape=shape),
            PositiveSupportResponse,
        )

    @step
    def exponential(self) -> PositiveSupportResponse:
        return self._apply(
            partial(weibull, shape=1.0),
            label(weibull, shape=1.0),
            PositiveSupportResponse,
        )

    @step
    def log_logistic(self, shape: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._apply(
            partial(log_logistic, shape=shape),
            label(log_logistic, shape=shape),
            PositiveSupportResponse,
        )

    @step
    def gompertz(self, shape: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._apply(
            partial(gompertz, shape=shape),
            label(gompertz, shape=shape),
            PositiveSupportResponse,
        )


class Predictor(_FamilyPipeline):
    def __init__(
        self,
        pipeline: Pipeline[PredictorData],
        *,
        re_count: int = 0,
        proj_count: int = 0,
    ) -> None:
        super().__init__(pipeline)
        self._re_count = re_count
        self._proj_count = proj_count

    @step
    def random_effects(
        self,
        levels: int,
        q: int = 1,
        *,
        W: Prior | None = None,
        B: Prior = UNIT_NORMAL,
        b: Prior = UNIT_NORMAL,
    ) -> "Predictor":
        # design choice: default to soft level assignments
        w = W or dist.Dirichlet(torch.ones(levels))
        step_fn = suffixed(
            partial(random_effects, levels=levels, q=q, W=w, B=B, b=b), self._re_count
        )
        return Predictor(
            self._pipeline.then(
                step_fn, label(random_effects, levels=levels, q=q, W=w, B=B, b=b)
            ),
            re_count=self._re_count + 1,
            proj_count=self._proj_count,
        )

    @step
    def activation(self, fn: Callable[[Tensor], Tensor] = torch.relu) -> "Predictor":
        return Predictor(
            self._pipeline.then(partial(activation, fn=fn), label(activation)),
            re_count=self._re_count,
            proj_count=self._proj_count,
        )

    @step
    def projection(self, output: int, weight: Prior = UNIT_NORMAL) -> "Predictor":
        step_fn = suffixed(
            partial(projection, output=output, weight=weight), self._proj_count
        )
        return Predictor(
            self._pipeline.then(
                step_fn, label(projection, output=output, weight=weight)
            ),
            re_count=self._re_count,
            proj_count=self._proj_count + 1,
        )

    @step
    def tokenize(
        self,
        vocab_size: int,
        weight: Prior = UNIT_NORMAL,
        temperature: float | Tensor = 1.0,
    ) -> "Predictor":
        step_fn = partial(
            tokenize, vocab_size=vocab_size, weight=weight, temperature=temperature
        )
        return Predictor(
            self._pipeline.then(step_fn, label(tokenize, vocab_size=vocab_size)),
            re_count=self._re_count,
            proj_count=self._proj_count,
        )

    @step
    def constant_target(self) -> "ConstantPredictor":
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
