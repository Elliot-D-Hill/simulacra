from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from torch import Tensor

type Step[S, T] = Callable[[S], T]
type Run[S] = Callable[[], S]

_WIDTH = 88


def _format(v: object) -> str:
    match v:
        case Tensor() if v.ndim == 0:
            return f"tensor({v.item():.4g})"
        case Tensor():
            return f"Tensor{tuple(v.shape)}"
        case tuple():
            return f"Tensor{v}"
        case partial():
            return label(v.func, **v.keywords)
        case _:
            return getattr(v, "__name__", repr(v))


def label(transform: Callable[..., object], **kwargs: object) -> str:
    parts = [f"{k}={_format(v)}" for k, v in kwargs.items() if v is not None]
    single = f"{transform.__name__}({', '.join(parts)})"
    if len(single) <= _WIDTH:
        return single
    body = ",\n    ".join(parts)
    return f"{transform.__name__}(\n    {body},\n)"


@dataclass(frozen=True)
class Pipeline[S]:
    run: Run[S]
    recipe: tuple[str, ...]

    def apply[T](self, transform: Callable[..., T], **kwargs: object) -> Pipeline[T]:
        first = self.run

        def composed() -> T:
            return transform(first(), **kwargs)

        return Pipeline(composed, (*self.recipe, label(transform, **kwargs)))

    def __repr__(self) -> str:
        return "\n.".join(self.recipe) or "Pipeline"
