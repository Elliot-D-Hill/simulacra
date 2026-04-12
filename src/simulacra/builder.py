from collections.abc import Callable
from dataclasses import replace
from typing import Final, NoReturn, Self, overload

import torch
import torch.distributions as dist
from tensordict import TensorDict
from torch import Tensor

from .graph import Graph, build_graph, guide, step
from .families import (
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
from .states import (
    DiscreteSurvivalData,
    EventTimeData,
    InitialData,
    PredictorData,
    ResponseData,
    SurvivalData,
)
from .survival import censor, competing_risks, discretize
from .transforms import (
    Params,
    Prior,
    fixed_effects,
    missing_x,
    missing_y,
    points,
    random_effects,
    tokenize,
)

UNIT_VARIANCE: Final[Tensor] = torch.tensor(1.0)
UNIT_NORMAL: Final[dist.Normal] = dist.Normal(0.0, UNIT_VARIANCE)
EXP1: Final[dist.Exponential] = dist.Exponential(1.0)

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


def _compose[S, T](prev: Run[S], step: Callable[[S], tuple[T, Params]]) -> Run[T]:
    def run(draws: tuple[int, ...]) -> tuple[T, Params]:
        data, params = prev(draws)
        new_data, new_params = step(data)
        return new_data, {**params, **new_params}

    return run


def _identity[T](x: T) -> T:
    return x


class _Pipeline[S: PredictorData]:
    def __init__(self, run: Run[S], recipe: tuple[str, ...] = ()) -> None:
        self._run = run
        self._recipe = recipe

    def __repr__(self) -> str:
        return "\n  ".join(self._recipe) or type(self).__name__

    def __getattr__(self, name: str) -> NoReturn:
        raise guide(self, name, GRAPH)

    @step
    def missing_x(self, proportion: float) -> Self:
        return type(self)(
            _compose(self._run, lambda data: missing_x(data, proportion)),
            (*self._recipe, _label(missing_x, proportion=proportion)),
        )

    @step
    def tokenize(self, vocab_size: int) -> Self:
        return type(self)(
            _compose(self._run, lambda data: tokenize(data, vocab_size)),
            (*self._recipe, _label(tokenize, vocab_size=vocab_size)),
        )

    def draw(
        self, draws: int | None = None, seed: int | None = None
    ) -> tuple[dict[str, Tensor], Params]:
        if seed is not None:
            torch.manual_seed(seed)  # type: ignore[no-untyped-call]
        batch = (draws,) if draws is not None else ()
        data, params = self._run(batch)
        tensor_data = {k: v for k, v in vars(data).items() if v is not None}
        n, t = data.X.shape[-3], data.X.shape[-2]
        td = TensorDict(tensor_data, batch_size=(*batch, n, t))
        # TODO: casting to dict is a patch due to TensorDict poor access typing
        return dict(td.squeeze(-1)), params


class _ResponsePipeline[S: ResponseData](_Pipeline[S]):
    @step
    def missing_y(self, proportion: float) -> Self:
        return type(self)(
            _compose(self._run, lambda data: missing_y(data, proportion)),
            (*self._recipe, _label(missing_y, proportion=proportion)),
        )


class Response(_ResponsePipeline[ResponseData]): ...


class PositiveSupportResponse(_ResponsePipeline[ResponseData]):
    @step
    def competing_risks(self) -> CompetingResponse:
        return CompetingResponse(
            _compose(self._run, competing_risks),
            (*self._recipe, _label(competing_risks)),
        )

    @step
    def censor(
        self, dropout: Prior | None = None, *, horizon: float | Tensor = torch.inf
    ) -> Survival:
        return Survival(
            _compose(self._run, lambda data: censor(data, dropout, horizon=horizon)),
            (*self._recipe, _label(censor, dropout=dropout, horizon=horizon)),
        )


class CompetingResponse(_ResponsePipeline[EventTimeData]):
    @step
    def censor(
        self, dropout: Prior | None = None, *, horizon: float | Tensor = torch.inf
    ) -> Survival:
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


class Simulation:
    def __init__(self, n: int, t: int, p: int, k: int) -> None:
        initial = InitialData(draws=(), n=n, t=t, p=p, k=k)
        self._run: Run[InitialData] = lambda draws: (replace(initial, draws=draws), {})
        self._recipe = (f"Simulation(n={n}, t={t}, p={p}, k={k})",)

    def __repr__(self) -> str:
        return "\n  ".join(self._recipe) or type(self).__name__

    def __getattr__(self, name: str) -> NoReturn:
        raise guide(self, name, GRAPH)

    @step
    def fixed_effects(
        self, X: Prior = UNIT_NORMAL, beta: Prior = UNIT_NORMAL
    ) -> Predictor:
        return Predictor(
            _compose(self._run, lambda data: fixed_effects(data, X, beta)),
            (*self._recipe, _label(fixed_effects, X=X, beta=beta)),
        )


class Predictor(_FamilyPipeline):
    def __init__(self, run: Run[PredictorData], recipe: tuple[str, ...] = ()) -> None:
        super().__init__(run, recipe)
        self._re_count: int = 0

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
        result = Predictor(
            _compose(
                self._run, lambda data: random_effects(data, levels, q, w, B, b, index)
            ),
            (*self._recipe, _label(random_effects, levels=levels, q=q, W=w, B=B, b=b)),
        )
        result._re_count = index + 1
        return result

    @step
    def constant_target(self) -> ConstantPredictor:
        return ConstantPredictor(self._run, (*self._recipe, _label(constant_target)))

    @step
    def points(self, coordinates: Prior = EXP1) -> Predictor:
        result = Predictor(
            _compose(self._run, lambda data: points(data, coordinates)),
            (*self._recipe, _label(points, coordinates=coordinates)),
        )
        result._re_count = self._re_count
        return result


class ConstantPredictor(_FamilyPipeline):
    def __init__(self, run: Run[PredictorData], recipe: tuple[str, ...] = ()) -> None:
        super().__init__(
            run, recipe, wrap=lambda f: lambda data: constant_target(data, f)
        )


GRAPH: Final[Graph] = build_graph(
    Simulation,
    Predictor,
    ConstantPredictor,
    Response,
    PositiveSupportResponse,
    CompetingResponse,
    Survival,
    DiscreteSurvival,
)
