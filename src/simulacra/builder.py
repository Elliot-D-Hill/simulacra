from collections.abc import Callable
from dataclasses import replace
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
    Params,
    Step,
    activation,
    covariates,
    fixed_effects,
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

type Run[S] = Callable[[tuple[int, ...]], tuple[S, Params]]


def _label(fn: Callable[..., object], **kwargs: object) -> str:
    def _format(v: object) -> str:
        if isinstance(v, Tensor):
            return (
                f"tensor({v.item():.4g})" if v.ndim == 0 else f"Tensor{tuple(v.shape)}"
            )
        return repr(v)

    parts = ", ".join(f"{k}={_format(v)}" for k, v in kwargs.items())
    return f"{fn.__name__}({parts})"


def _compose[S, T](prev: Run[S], step: Step[S, T]) -> Run[T]:
    def run(draws: tuple[int, ...]) -> tuple[T, Params]:
        data, params = prev(draws)
        new_data, new_params = step(data)
        return new_data, {**params, **new_params}

    return run


def _suffixed[S, T](fn: Step[S, T], index: int) -> Step[S, T]:
    def wrapped(data: S) -> tuple[T, Params]:
        new_data, params = fn(data)
        return new_data, {f"{k}_{index}": v for k, v in params.items()}

    return wrapped


def _identity[T](x: T) -> T:
    return x


class Covariate:
    def __init__(self, run: Run[CovariateData], recipe: tuple[str, ...] = ()) -> None:
        self._run = run
        self._recipe = recipe

    def __repr__(self) -> str:
        return "\n  .".join(self._recipe) or type(self).__name__

    def __getattr__(self, name: str) -> NoReturn:
        raise guide(self, name, GRAPH)

    @step
    def z_score(self) -> Self:
        return type(self)(
            _compose(self._run, z_score), (*self._recipe, _label(z_score))
        )

    @step
    def min_max_scale(self, low: float = 0.0, high: float = 1.0) -> Self:
        return type(self)(
            _compose(self._run, lambda data: min_max_scale(data, low, high)),
            (*self._recipe, _label(min_max_scale, low=low, high=high)),
        )

    @step
    def fixed_effects(self, k: int = 1, beta: Prior = UNIT_NORMAL) -> Predictor:
        return Predictor(
            _compose(self._run, lambda data: fixed_effects(data, k, beta)),
            (*self._recipe, _label(fixed_effects, k=k, beta=beta)),
        )


class Simulation:
    def __init__(self, n: int, t: int = 1, p: int = 1) -> None:
        coordinates = torch.arange(t, dtype=torch.float).unsqueeze(-1)
        initial = InitialData(
            draws=(), n=n, t=t, p=p, X=UNIT_NORMAL, coordinates=coordinates
        )
        self._run: Run[InitialData] = lambda draws: (replace(initial, draws=draws), {})
        self._recipe = (f"Simulation(n={n}, t={t}, p={p})",)

    @classmethod
    def _from_run(cls, run: Run[InitialData], recipe: tuple[str, ...]) -> Self:
        obj = cls.__new__(cls)
        obj._run = run
        obj._recipe = recipe
        return obj

    def __repr__(self) -> str:
        return "\n  .".join(self._recipe) or type(self).__name__

    def __getattr__(self, name: str) -> NoReturn:
        raise guide(self, name, GRAPH)

    @step
    def covariates(self, X: Prior = UNIT_NORMAL) -> Self:
        return type(self)._from_run(
            _compose(self._run, lambda data: covariates(data, X)),
            (*self._recipe, _label(covariates, X=X)),
        )

    @step
    def points(self, coordinates: Prior = EXP1) -> Self:
        return type(self)._from_run(
            _compose(self._run, lambda data: points(data, coordinates)),
            (*self._recipe, _label(points, coordinates=coordinates)),
        )

    @step
    def z_score(self) -> Covariate:
        return Covariate(
            _compose(_compose(self._run, resolve_design), z_score),
            (*self._recipe, _label(z_score)),
        )

    @step
    def min_max_scale(self, low: float = 0.0, high: float = 1.0) -> Covariate:
        return Covariate(
            _compose(
                _compose(self._run, resolve_design),
                lambda data: min_max_scale(data, low, high),
            ),
            (*self._recipe, _label(min_max_scale, low=low, high=high)),
        )

    @step
    def fixed_effects(self, k: int = 1, beta: Prior = UNIT_NORMAL) -> Predictor:
        return Predictor(
            _compose(
                _compose(self._run, resolve_design),
                lambda data: fixed_effects(data, k, beta),
            ),
            (*self._recipe, _label(fixed_effects, k=k, beta=beta)),
        )


class _Pipeline[S: PredictorData]:
    def __init__(self, run: Run[S], recipe: tuple[str, ...] = ()) -> None:
        self._run = run
        self._recipe = recipe

    def __repr__(self) -> str:
        return "\n  .".join(self._recipe) or type(self).__name__

    def __getattr__(self, name: str) -> NoReturn:
        raise guide(self, name, GRAPH)

    def draw(
        self, draws: int | None = None, seed: int | None = None
    ) -> tuple[SimulationData, SimulationParams]:
        if seed is not None:
            torch.manual_seed(seed)  # type: ignore[no-untyped-call]
        batch = (draws,) if draws is not None else ()
        data, params = self._run(batch)
        tensor_data = {k: v for k, v in vars(data).items() if v is not None}
        squeezed = {k: v.squeeze(-2) for k, v in tensor_data.items()}
        return SimulationData(squeezed), SimulationParams(params)


class _ResponsePipeline[S: ResponseData](_Pipeline[S]):
    @step
    def missing_x(self, proportion: float) -> Self:
        return type(self)(
            _compose(self._run, lambda data: missing_x(data, proportion)),
            (*self._recipe, _label(missing_x, proportion=proportion)),
        )

    @step
    def missing_y(self, proportion: float) -> Self:
        return type(self)(
            _compose(self._run, lambda data: missing_y(data, proportion)),
            (*self._recipe, _label(missing_y, proportion=proportion)),
        )


class Response(_ResponsePipeline[ResponseData]): ...


class PositiveSupportResponse(_ResponsePipeline[ResponseData]):
    @step
    def competing_risks(self) -> "CompetingResponse":
        return CompetingResponse(
            _compose(self._run, competing_risks),
            (*self._recipe, _label(competing_risks)),
        )

    @step
    def censor(
        self, dropout: Prior = EXP1, *, horizon: float | Tensor = torch.inf
    ) -> "Survival":
        return Survival(
            _compose(self._run, lambda data: censor(data, dropout, horizon=horizon)),
            (*self._recipe, _label(censor, dropout=dropout, horizon=horizon)),
        )


class CompetingResponse(_ResponsePipeline[EventTimeData]):
    @step
    def censor(
        self, dropout: Prior = EXP1, *, horizon: float | Tensor = torch.inf
    ) -> "Survival":
        return Survival(
            _compose(self._run, lambda data: censor(data, dropout, horizon=horizon)),
            (*self._recipe, _label(censor, dropout=dropout, horizon=horizon)),
        )


class Survival(_ResponsePipeline[SurvivalData]):
    @step
    def discretize(self, boundaries: Tensor) -> DiscreteSurvival:
        return DiscreteSurvival(
            _compose(self._run, lambda data: discretize(data, boundaries)),
            (*self._recipe, _label(discretize, boundaries=boundaries)),
        )


class DiscreteSurvival(_ResponsePipeline[DiscreteSurvivalData]): ...


class _FamilyPipeline(_Pipeline[PredictorData]):
    def __init__(
        self,
        run: Run[PredictorData],
        recipe: tuple[str, ...] = (),
        wrap: Callable[[Family], Family] = _identity,
    ) -> None:
        super().__init__(run, recipe)
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
        return cls(_compose(self._run, self._wrap(family)), (*self._recipe, label))

    @step
    def gaussian(self, covariance: Prior = UNIT_VARIANCE) -> Response:
        return self._apply(
            lambda data: gaussian(data, covariance),
            _label(gaussian, covariance=covariance),
        )

    @step
    def poisson(self) -> Response:
        return self._apply(poisson, _label(poisson))

    @step
    def bernoulli(self) -> Response:
        return self._apply(bernoulli, _label(bernoulli))

    @step
    def binomial(self, num_trials: int = 1) -> Response:
        return self._apply(
            lambda data: binomial(data, num_trials),
            _label(binomial, num_trials=num_trials),
        )

    @step
    def negative_binomial(self, concentration: float | Tensor) -> Response:
        return self._apply(
            lambda data: negative_binomial(data, concentration),
            _label(negative_binomial, concentration=concentration),
        )

    @step
    def categorical(self) -> Response:
        return self._apply(categorical, _label(categorical))

    @step
    def gamma(self, concentration: float | Tensor) -> PositiveSupportResponse:
        return self._apply(
            lambda data: gamma(data, concentration),
            _label(gamma, concentration=concentration),
            PositiveSupportResponse,
        )

    @step
    def log_normal(self, std: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._apply(
            lambda data: log_normal(data, std),
            _label(log_normal, std=std),
            PositiveSupportResponse,
        )

    @step
    def weibull(self, shape: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._apply(
            lambda data: weibull(data, shape),
            _label(weibull, shape=shape),
            PositiveSupportResponse,
        )

    @step
    def exponential(self) -> PositiveSupportResponse:
        return self._apply(
            lambda data: weibull(data, 1.0),
            _label(weibull, shape=1.0),
            PositiveSupportResponse,
        )

    @step
    def log_logistic(self, shape: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._apply(
            lambda data: log_logistic(data, shape),
            _label(log_logistic, shape=shape),
            PositiveSupportResponse,
        )

    @step
    def gompertz(self, shape: float | Tensor = 1.0) -> PositiveSupportResponse:
        return self._apply(
            lambda data: gompertz(data, shape),
            _label(gompertz, shape=shape),
            PositiveSupportResponse,
        )


class Predictor(_FamilyPipeline):
    def __init__(self, run: Run[PredictorData], recipe: tuple[str, ...] = ()) -> None:
        super().__init__(run, recipe)
        self._re_count: int = 0
        self._proj_count: int = 0

    def _chain(self, run: Run[PredictorData], recipe: tuple[str, ...]) -> Predictor:
        result = Predictor(run, recipe)
        result._re_count = self._re_count
        result._proj_count = self._proj_count
        return result

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
        index = self._re_count
        # design choice: default to soft level assignments
        w = W or dist.Dirichlet(torch.ones(levels))
        result = self._chain(
            _compose(
                self._run,
                _suffixed(lambda data: random_effects(data, levels, q, w, B, b), index),
            ),
            (*self._recipe, _label(random_effects, levels=levels, q=q, W=w, B=B, b=b)),
        )
        result._re_count = index + 1
        return result

    @step
    def activation(self, fn: Callable[[Tensor], Tensor] = torch.relu) -> Predictor:
        return self._chain(
            _compose(self._run, lambda data: activation(data, fn)),
            (*self._recipe, _label(activation)),
        )

    @step
    def projection(self, output: int, weight: Prior = UNIT_NORMAL) -> Predictor:
        index = self._proj_count
        result = self._chain(
            _compose(
                self._run,
                _suffixed(lambda data: projection(data, output, weight), index),
            ),
            (*self._recipe, _label(projection, output=output, weight=weight)),
        )
        result._proj_count = index + 1
        return result

    @step
    def tokenize(
        self,
        vocab_size: int,
        weight: Prior = UNIT_NORMAL,
        temperature: float | Tensor = 1.0,
    ) -> Predictor:
        return self._chain(
            _compose(
                self._run,
                lambda data: tokenize(data, vocab_size, weight, temperature),
            ),
            (*self._recipe, _label(tokenize, vocab_size=vocab_size)),
        )

    @step
    def constant_target(self) -> "ConstantPredictor":
        return ConstantPredictor(self._run, (*self._recipe, _label(constant_target)))


class ConstantPredictor(_FamilyPipeline):
    def __init__(self, run: Run[PredictorData], recipe: tuple[str, ...] = ()) -> None:
        super().__init__(
            run, recipe, wrap=lambda f: lambda data: constant_target(data, f)
        )


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
