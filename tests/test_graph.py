import pytest
import torch

from simulacra import (
    GRAPH,
    CompetingResponse,
    DiscreteSurvival,
    PositiveSupportResponse,
    Predictor,
    Response,
    Survival,
    simulate,
)

ALL_STATES: set[type] = {
    CompetingResponse,
    DiscreteSurvival,
    PositiveSupportResponse,
    Predictor,
    Response,
    Survival,
}


def test_graph_contains_all_concrete_classes() -> None:
    assert GRAPH.states() == ALL_STATES


def test_predictor_transitions() -> None:
    expected = {
        "random_effects",
        "activation",
        "tokenize",
        "treatment",
        "dose_response",
        "gaussian",
        "poisson",
        "bernoulli",
        "binomial",
        "negative_binomial",
        "categorical",
        "multinomial",
        "beta",
        "dirichlet",
        "gamma",
        "log_normal",
        "weibull",
        "exponential",
        "log_logistic",
        "gompertz",
    }
    assert GRAPH.methods_on(Predictor) == expected


def test_response_transitions() -> None:
    expected = {"missing_x", "missing_y", "constant_y"}
    assert GRAPH.methods_on(Response) == expected


def test_positive_support_response_transitions() -> None:
    expected = {"competing_risks", "censor", "missing_x", "missing_y", "constant_y"}
    assert GRAPH.methods_on(PositiveSupportResponse) == expected


def test_competing_response_transitions() -> None:
    expected = {"censor", "missing_x", "missing_y", "constant_y"}
    assert GRAPH.methods_on(CompetingResponse) == expected


def test_survival_transitions() -> None:
    expected = {"discretize", "missing_x", "missing_y", "constant_y"}
    assert GRAPH.methods_on(Survival) == expected


def test_discrete_survival_transitions() -> None:
    expected = {"missing_x", "missing_y", "constant_y"}
    assert GRAPH.methods_on(DiscreteSurvival) == expected


def test_same_type_transitions_are_self() -> None:
    predictor_targets = {t.method: t.target for t in GRAPH.from_state(Predictor)}
    assert predictor_targets["random_effects"] is None
    assert predictor_targets["activation"] is None
    assert predictor_targets["tokenize"] is None


def test_family_targets() -> None:
    targets = {t.method: t.target for t in GRAPH.from_state(Predictor)}
    assert targets["gaussian"] is Response
    assert targets["weibull"] is PositiveSupportResponse


def test_to_state_survival() -> None:
    sources = {t.source for t in GRAPH.to_state(Survival)}
    assert sources == {Survival, PositiveSupportResponse, CompetingResponse}


def test_owners_of_censor() -> None:
    assert GRAPH.owners_of("censor") == {"CompetingResponse", "PositiveSupportResponse"}


def test_draw_not_in_graph() -> None:
    assert "draw" not in GRAPH.all_methods()


def test_private_methods_not_in_graph() -> None:
    for transition in GRAPH.transitions:
        assert not transition.method.startswith("_")


def test_response_censor_guides(dims: tuple[int, int, int, int]) -> None:
    N, T, p, k = dims
    resp = simulate(torch.randn(N, T, p), torch.randn(p, k)).gaussian()
    with pytest.raises(
        AttributeError,
        match=r"Response has no method censor\(\).*CompetingResponse.*PositiveSupportResponse",
    ):
        resp.censor()  # type: ignore[attr-defined]


def test_survival_gaussian_guides(dims: tuple[int, int, int, int]) -> None:
    N, T, p, k = dims
    surv = (
        simulate(torch.randn(N, T, p), torch.randn(p, k)).weibull().censor(horizon=2.0)
    )
    with pytest.raises(
        AttributeError, match=r"Survival has no method gaussian\(\).*Predictor"
    ):
        surv.gaussian()  # type: ignore[attr-defined]


def test_unknown_method_no_guidance(dims: tuple[int, int, int, int]) -> None:
    N, T, p, k = dims
    resp = simulate(torch.randn(N, T, p), torch.randn(p, k)).gaussian()
    with pytest.raises(AttributeError, match="^foobar$"):
        resp.foobar()  # type: ignore[attr-defined]


def test_hasattr_still_returns_false(dims: tuple[int, int, int, int]) -> None:
    N, T, p, k = dims
    resp = simulate(torch.randn(N, T, p), torch.randn(p, k)).gaussian()
    assert not hasattr(resp, "censor")
    assert not hasattr(resp, "competing_risks")
    assert not hasattr(resp, "discretize")
    assert not hasattr(resp, "random_effects")


def test_valid_method_still_works(dims: tuple[int, int, int, int]) -> None:
    N, T, p, k = dims
    resp = simulate(torch.randn(N, T, p), torch.randn(p, k)).gaussian()
    resp_with_missing = resp.missing_y(0.1)
    assert type(resp_with_missing).__name__ == "Response"
