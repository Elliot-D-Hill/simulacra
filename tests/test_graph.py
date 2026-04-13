import pytest

from simulacra import (
    GRAPH,
    CompetingResponse,
    ConstantPredictor,
    Covariate,
    DiscreteSurvival,
    PositiveSupportResponse,
    Predictor,
    Response,
    Simulation,
    Survival,
)

ALL_STATES: set[type] = {
    CompetingResponse,
    ConstantPredictor,
    Covariate,
    DiscreteSurvival,
    PositiveSupportResponse,
    Predictor,
    Response,
    Simulation,
    Survival,
}


def test_graph_contains_all_concrete_classes() -> None:
    assert GRAPH.states() == ALL_STATES


def test_simulation_transitions() -> None:
    expected = {"z_score", "min_max_scale", "fixed_effects"}
    assert GRAPH.methods_on(Simulation) == expected


def test_covariate_transitions() -> None:
    expected = {"z_score", "min_max_scale", "fixed_effects"}
    assert GRAPH.methods_on(Covariate) == expected


def test_predictor_transitions() -> None:
    expected = {
        "random_effects",
        "activation",
        "projection",
        "constant_target",
        "tokenize",
        "gaussian",
        "poisson",
        "bernoulli",
        "binomial",
        "negative_binomial",
        "categorical",
        "gamma",
        "log_normal",
        "weibull",
        "exponential",
        "log_logistic",
        "gompertz",
    }
    assert GRAPH.methods_on(Predictor) == expected


def test_constant_predictor_transitions() -> None:
    expected = {
        "gaussian",
        "poisson",
        "bernoulli",
        "binomial",
        "negative_binomial",
        "categorical",
        "gamma",
        "log_normal",
        "weibull",
        "exponential",
        "log_logistic",
        "gompertz",
    }
    assert GRAPH.methods_on(ConstantPredictor) == expected


def test_response_transitions() -> None:
    expected = {"missing_x", "missing_y"}
    assert GRAPH.methods_on(Response) == expected


def test_positive_support_response_transitions() -> None:
    expected = {"competing_risks", "censor", "missing_x", "missing_y"}
    assert GRAPH.methods_on(PositiveSupportResponse) == expected


def test_competing_response_transitions() -> None:
    expected = {"censor", "missing_x", "missing_y"}
    assert GRAPH.methods_on(CompetingResponse) == expected


def test_survival_transitions() -> None:
    expected = {"discretize", "missing_x", "missing_y"}
    assert GRAPH.methods_on(Survival) == expected


def test_discrete_survival_transitions() -> None:
    expected = {"missing_x", "missing_y"}
    assert GRAPH.methods_on(DiscreteSurvival) == expected


def test_self_transitions_resolve_to_source() -> None:
    """Self targets are stored as None, not the base class."""
    covariate_transitions = {(t.method, t.target) for t in GRAPH.from_state(Covariate)}
    assert ("z_score", None) in covariate_transitions
    assert ("min_max_scale", None) in covariate_transitions


def test_family_targets() -> None:
    targets = {t.method: t.target for t in GRAPH.from_state(Predictor)}
    assert targets["gaussian"] is Response
    assert targets["weibull"] is PositiveSupportResponse
    assert targets["random_effects"] is Predictor
    assert targets["constant_target"] is ConstantPredictor


def test_to_state_survival() -> None:
    sources = {t.source for t in GRAPH.to_state(Survival)}
    assert sources == {PositiveSupportResponse, CompetingResponse}


def test_owners_of_censor() -> None:
    assert GRAPH.owners_of("censor") == {"CompetingResponse", "PositiveSupportResponse"}


def test_draw_not_in_graph() -> None:
    assert "draw" not in GRAPH.all_methods()


def test_private_methods_not_in_graph() -> None:
    for transition in GRAPH.transitions:
        assert not transition.method.startswith("_")


def test_response_censor_guides(dims: tuple[int, int, int, int]) -> None:
    N, T, p, k = dims
    resp = Simulation(N, T, p).fixed_effects(k=k).gaussian()
    with pytest.raises(
        AttributeError,
        match=r"Response has no method censor\(\).*CompetingResponse.*PositiveSupportResponse",
    ):
        resp.censor()  # type: ignore[attr-defined]


def test_survival_gaussian_guides(dims: tuple[int, int, int, int]) -> None:
    N, T, p, k = dims
    surv = Simulation(N, T, p).fixed_effects(k=k).weibull().censor(horizon=2.0)
    with pytest.raises(
        AttributeError, match=r"Survival has no method gaussian\(\).*Predictor"
    ):
        surv.gaussian()  # type: ignore[attr-defined]


def test_simulation_gaussian_guides(dims: tuple[int, int, int, int]) -> None:
    N, T, p, _ = dims
    sim = Simulation(N, T, p)
    with pytest.raises(AttributeError, match=r"Simulation has no method gaussian\(\)"):
        sim.gaussian()  # type: ignore[attr-defined]


def test_unknown_method_no_guidance(dims: tuple[int, int, int, int]) -> None:
    N, T, p, k = dims
    resp = Simulation(N, T, p).fixed_effects(k=k).gaussian()
    with pytest.raises(AttributeError, match="^foobar$"):
        resp.foobar()  # type: ignore[attr-defined]


def test_hasattr_still_returns_false(dims: tuple[int, int, int, int]) -> None:
    N, T, p, k = dims
    resp = Simulation(N, T, p).fixed_effects(k=k).gaussian()
    assert not hasattr(resp, "censor")
    assert not hasattr(resp, "competing_risks")
    assert not hasattr(resp, "discretize")
    assert not hasattr(resp, "random_effects")


def test_valid_method_still_works(dims: tuple[int, int, int, int]) -> None:
    N, T, p, k = dims
    resp = Simulation(N, T, p).fixed_effects(k=k).gaussian()
    resp_with_missing = resp.missing_y(0.1)
    assert type(resp_with_missing).__name__ == "Response"
