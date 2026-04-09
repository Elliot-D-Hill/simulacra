from collections.abc import Callable
from dataclasses import replace
from typing import Final, Self

import torch
import torch.distributions as dist
from tensordict import TensorDict
from torch import Tensor

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
    FamilyFn,
    Prior,
    bernoulli,
    binomial,
    categorical,
    constant_target,
    fixed_effects,
    gamma,
    gaussian,
    log_normal,
    missing_x,
    missing_y,
    negative_binomial,
    points,
    poisson,
    random_effects,
    tokenize,
    weibull,
)

UNIT_VARIANCE: Final[Tensor] = torch.tensor(1.0)
UNIT_NORMAL: Final[dist.Normal] = dist.Normal(0.0, UNIT_VARIANCE)
EXP1: Final[dist.Exponential] = dist.Exponential(1.0)

type Run[S] = Callable[[tuple[int, ...]], tuple[S, dict[str, Tensor]]]


def _label(fn: Callable[..., object], **kwargs: object) -> str:
    def _format(v: object) -> str:
        if isinstance(v, Tensor):
            return (
                f"tensor({v.item():.4g})" if v.ndim == 0 else f"Tensor{tuple(v.shape)}"
            )
        return repr(v)

    parts = ", ".join(f"{k}={_format(v)}" for k, v in kwargs.items())
    return f"{fn.__name__}({parts})"


def _compose[S, T](
    prev: Run[S], step: Callable[[S], tuple[T, dict[str, Tensor]]]
) -> Run[T]:
    def run(draws: tuple[int, ...]) -> tuple[T, dict[str, Tensor]]:
        data, params = prev(draws)
        new_data, new_params = step(data)
        return new_data, {**params, **new_params}

    return run


class _Step[S]:
    def __init__(self, run: Run[S], recipe: tuple[str, ...] = ()) -> None:
        self._run = run
        self._recipe = recipe

    def __repr__(self) -> str:
        return "\n  ".join(self._recipe) or type(self).__name__


class _HasX[S: PredictorData](_Step[S]):
    def missing_x(self, proportion: float) -> Self:
        return type(self)(
            _compose(self._run, lambda data: missing_x(data, proportion)),
            (*self._recipe, _label(missing_x, proportion=proportion)),
        )

    def tokenize(self, vocab_size: int) -> Self:
        return type(self)(
            _compose(self._run, lambda data: tokenize(data, vocab_size)),
            (*self._recipe, _label(tokenize, vocab_size=vocab_size)),
        )

    def draw(
        self, draws: int | None = None, seed: int | None = None
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        if seed is not None:
            torch.manual_seed(seed)  # type: ignore[no-untyped-call]
        batch = (draws,) if draws is not None else ()
        data, params = self._run(batch)
        tensor_data = {k: v for k, v in vars(data).items() if v is not None}
        n, t = data.X.shape[-3], data.X.shape[-2]
        td = TensorDict(tensor_data, batch_size=(*batch, n, t))
        # TODO: casting to dict is a patch due to TensorDict poor access typing
        return dict(td.squeeze(-1)), params


def _identity[T](x: T) -> T:
    return x


class _HasFamily(_HasX[PredictorData]):
    def __init__(
        self,
        run: Run[PredictorData],
        recipe: tuple[str, ...] = (),
        wrap: Callable[[FamilyFn], FamilyFn] = _identity,
    ) -> None:
        super().__init__(run, recipe)
        self._wrap = wrap

    def _apply(self, family: FamilyFn, label: str) -> Response:
        return Response(_compose(self._run, self._wrap(family)), (*self._recipe, label))

    def gaussian(self, covariance: Prior = UNIT_VARIANCE) -> Response:
        return self._apply(
            lambda data: gaussian(data, covariance),
            _label(gaussian, covariance=covariance),
        )

    def poisson(self) -> Response:
        return self._apply(poisson, _label(poisson))

    def bernoulli(self) -> Response:
        return self._apply(bernoulli, _label(bernoulli))

    def binomial(self, num_trials: int = 1) -> Response:
        return self._apply(
            lambda data: binomial(data, num_trials),
            _label(binomial, num_trials=num_trials),
        )

    def negative_binomial(self, concentration: float | Tensor) -> Response:
        return self._apply(
            lambda data: negative_binomial(data, concentration),
            _label(negative_binomial, concentration=concentration),
        )

    def gamma(self, concentration: float | Tensor) -> Response:
        return self._apply(
            lambda data: gamma(data, concentration),
            _label(gamma, concentration=concentration),
        )

    def log_normal(self, std: float | Tensor = 1.0) -> Response:
        return self._apply(
            lambda data: log_normal(data, std), _label(log_normal, std=std)
        )

    def categorical(self) -> Response:
        return self._apply(categorical, _label(categorical))

    def weibull(self, shape: float | Tensor = 1.0) -> WeibullResponse:
        return WeibullResponse(
            _compose(self._run, self._wrap(lambda data: weibull(data, shape))),
            (*self._recipe, _label(weibull, shape=shape)),
        )


class Simulation(_Step[InitialData]):
    def __init__(self, n: int, t: int, p: int, k: int) -> None:
        initial = InitialData(draws=(), n=n, t=t, p=p, k=k)
        super().__init__(
            lambda draws: (replace(initial, draws=draws), {}),
            (f"Simulation(n={n}, t={t}, p={p}, k={k})",),
        )

    def fixed_effects(
        self, X: Prior = UNIT_NORMAL, beta: Prior = UNIT_NORMAL
    ) -> Predictor:
        return Predictor(
            _compose(self._run, lambda data: fixed_effects(data, X, beta)),
            (*self._recipe, _label(fixed_effects, X=X, beta=beta)),
        )


class Predictor(_HasFamily):
    def __init__(self, run: Run[PredictorData], recipe: tuple[str, ...] = ()) -> None:
        super().__init__(run, recipe)
        self._re_count: int = 0

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

    def constant_target(self) -> ConstantPredictor:
        return ConstantPredictor(self._run, (*self._recipe, _label(constant_target)))

    def points(self, coordinates: Prior = EXP1) -> Predictor:
        result = Predictor(
            _compose(self._run, lambda data: points(data, coordinates)),
            (*self._recipe, _label(points, coordinates=coordinates)),
        )
        result._re_count = self._re_count
        return result


class ConstantPredictor(_HasFamily):
    def __init__(self, run: Run[PredictorData], recipe: tuple[str, ...] = ()) -> None:
        super().__init__(
            run, recipe, wrap=lambda f: lambda data: constant_target(data, f)
        )


class _HasY[S: ResponseData](_HasX[S]):
    def missing_y(self, proportion: float) -> Self:
        return type(self)(
            _compose(self._run, lambda data: missing_y(data, proportion)),
            (*self._recipe, _label(missing_y, proportion=proportion)),
        )


class Response(_HasY[ResponseData]): ...


class WeibullResponse(_HasY[ResponseData]):
    def competing_risks(self) -> CompetingResponse:
        return CompetingResponse(
            _compose(self._run, competing_risks),
            (*self._recipe, _label(competing_risks)),
        )

    def censor(
        self, dropout: Prior | None = None, *, horizon: float | Tensor = torch.inf
    ) -> Survival:
        return Survival(
            _compose(self._run, lambda data: censor(data, dropout, horizon=horizon)),
            (*self._recipe, _label(censor, dropout=dropout, horizon=horizon)),
        )


class CompetingResponse(_HasY[EventTimeData]):
    def censor(
        self, dropout: Prior | None = None, *, horizon: float | Tensor = torch.inf
    ) -> Survival:
        return Survival(
            _compose(self._run, lambda data: censor(data, dropout, horizon=horizon)),
            (*self._recipe, _label(censor, dropout=dropout, horizon=horizon)),
        )


class Survival(_HasY[SurvivalData]):
    def discretize(self, boundaries: Tensor) -> DiscreteSurvival:
        return DiscreteSurvival(
            _compose(self._run, lambda data: discretize(data, boundaries)),
            (*self._recipe, _label(discretize, boundaries=boundaries)),
        )


class DiscreteSurvival(_HasY[DiscreteSurvivalData]): ...
