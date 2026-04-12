"""Transition graph introspection for the builder pipeline.

Provides a ``@step`` marker decorator and a ``build_graph`` function that
extracts the valid transition graph from type annotations at import time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self, get_type_hints


def step[F: Callable[..., Any]](fn: F) -> F:
    """Mark a method as a pipeline transition. Zero runtime cost."""
    fn._is_step = True  # type: ignore[attr-defined]
    return fn


@dataclass(frozen=True)
class Transition:
    """A single valid state transition in the pipeline."""

    source: type
    method: str
    target: type | None  # None means Self (target = source at call time)

    def __repr__(self) -> str:
        target_name = "Self" if self.target is None else self.target.__name__
        return f"{self.source.__name__}.{self.method}() -> {target_name}"


@dataclass(frozen=True)
class Graph:
    """The pipeline's transition graph, queryable as data."""

    transitions: tuple[Transition, ...]

    def from_state(self, state: type) -> tuple[Transition, ...]:
        """Transitions leaving *state*."""
        return tuple(t for t in self.transitions if t.source is state)

    def to_state(self, state: type) -> tuple[Transition, ...]:
        """Transitions entering *state*."""
        return tuple(t for t in self.transitions if t.target is state)

    def methods_on(self, state: type) -> frozenset[str]:
        """Method names valid on *state*."""
        return frozenset(t.method for t in self.from_state(state))

    def states(self) -> frozenset[type]:
        """All state types in the graph."""
        sources = {t.source for t in self.transitions}
        targets = {t.target for t in self.transitions if t.target is not None}
        return frozenset(sources | targets)

    def all_methods(self) -> frozenset[str]:
        """All method names across all states."""
        return frozenset(t.method for t in self.transitions)

    def owners_of(self, method: str) -> frozenset[str]:
        """Class names that have *method* as a transition."""
        return frozenset(
            t.source.__name__ for t in self.transitions if t.method == method
        )


def build_graph(*state_classes: type) -> Graph:
    """Extract the transition graph from ``@step``-decorated methods.

    Walks each class's MRO so inherited methods (e.g. ``missing_x`` from
    ``_Pipeline``) register for every concrete class.  Methods returning
    ``Self`` are recorded with ``target=None``; methods returning a non-class
    type (like ``draw``'s ``tuple``) are skipped.
    """
    known = {cls.__name__ for cls in state_classes}
    transitions: list[Transition] = []
    for cls in state_classes:
        seen: set[str] = set()
        for klass in cls.__mro__:
            for name, method in vars(klass).items():
                if name in seen:
                    continue
                seen.add(name)
                if not (callable(method) and getattr(method, "_is_step", False)):
                    continue
                hints = get_type_hints(method)
                target = hints.get("return")
                if target is Self:
                    transitions.append(Transition(cls, name, None))
                elif isinstance(target, type) and target.__name__ in known:
                    transitions.append(Transition(cls, name, target))
    return Graph(
        tuple(sorted(transitions, key=lambda t: (t.source.__name__, t.method)))
    )


def guide(instance: object, name: str, graph: Graph) -> AttributeError:
    """Build a guided ``AttributeError`` for an invalid method call.

    Returns (not raises) the exception so callers can ``raise guide(...)``.
    """
    current = type(instance).__name__
    if name not in graph.all_methods():
        return AttributeError(name)
    owners = sorted(graph.owners_of(name) - {current})
    return AttributeError(
        f"{current} has no method {name}(); "
        f"available from: {', '.join(owners)}"
    )
