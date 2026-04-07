import torch

from simulacra import Simulation


# --- type-state enforcement ---


def test_simulation_has_no_downstream_methods(
    dims: tuple[int, int, int, int],
) -> None:
    """Simulation exposes only fixed_effects."""
    N, T, p, k = dims
    sim = Simulation(N, T, p, k)
    assert not hasattr(sim, "missing_x")
    assert not hasattr(sim, "tokenize")
    assert not hasattr(sim, "missing_y")
    assert not hasattr(sim, "gaussian")
    assert not hasattr(sim, "draw")


def test_predictor_methods(dims: tuple[int, int, int, int]) -> None:
    """Predictor has X-transforms and draw, but not missing_y."""
    N, T, p, k = dims
    pred = Simulation(N, T, p, k).fixed_effects()
    assert hasattr(pred, "missing_x")
    assert hasattr(pred, "tokenize")
    assert hasattr(pred, "draw")
    assert not hasattr(pred, "missing_y")


def test_response_methods(dims: tuple[int, int, int, int]) -> None:
    """Response has both X- and Y-transforms."""
    N, T, p, k = dims
    resp = Simulation(N, T, p, k).fixed_effects().gaussian()
    assert hasattr(resp, "missing_x")
    assert hasattr(resp, "tokenize")
    assert hasattr(resp, "missing_y")


def test_event_time_methods(dims: tuple[int, int, int, int]) -> None:
    """EventTime has X- and Y-transforms."""
    N, T, p, k = dims
    et = Simulation(N, T, p, k).fixed_effects().gaussian().event_time()
    assert hasattr(et, "missing_x")
    assert hasattr(et, "missing_y")


def test_censored_methods(dims: tuple[int, int, int, int]) -> None:
    """Censored has X- and Y-transforms."""
    N, T, p, k = dims
    cens = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .gaussian()
        .event_time()
        .censor_time(horizon=2.0)
    )
    assert hasattr(cens, "missing_x")
    assert hasattr(cens, "missing_y")


def test_constant_predictor_methods(dims: tuple[int, int, int, int]) -> None:
    """ConstantPredictor has families and X-transforms, but no RE or double-pooling."""
    N, T, p, k = dims
    seq = Simulation(N, T, p, k).fixed_effects().constant_target()
    assert not hasattr(seq, "constant_target")
    assert not hasattr(seq, "random_effects")
    assert hasattr(seq, "gaussian")
    assert hasattr(seq, "missing_x")
    assert hasattr(seq, "draw")


# --- pipeline ---


def test_reproducibility(dims: tuple[int, int, int, int]) -> None:
    """Same seed produces identical draws."""
    N, T, p, k = dims
    data1, _ = Simulation(N, T, p, k).fixed_effects().gaussian().draw(seed=0)
    data2, _ = Simulation(N, T, p, k).fixed_effects().gaussian().draw(seed=0)
    assert data1["y"].equal(data2["y"])


def test_immutability(dims: tuple[int, int, int, int]) -> None:
    """Branching from a shared base reuses X but produces different y."""
    N, T, p, k = dims
    base = Simulation(N, T, p, k).fixed_effects()
    da, _ = base.gaussian().draw(seed=0)
    db, _ = base.poisson().draw(seed=0)
    assert da["X"].equal(db["X"])
    assert not da["y"].equal(db["y"])


def test_base_shape(dims: tuple[int, int, int, int]) -> None:
    """draws=None gives [N, T, ...] shape."""
    N, T, p, k = dims
    data, _ = Simulation(N, T, p, k).fixed_effects().gaussian().draw(seed=0)
    assert data["X"].shape == (N, T, p)
    assert data["y"].shape == (N, T, k)


def test_draws_shape(dims: tuple[int, int, int, int]) -> None:
    """draws=D adds a leading dimension to data and params."""
    N, T, p, k = dims
    D = 7
    sim = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .gaussian()
        .event_time()
        .censor_time(horizon=2.0)
    )
    data, params = sim.draw(seed=0, draws=D)
    assert data["X"].shape == (D, N, T, p)
    assert params["beta"].shape == (D, p, k)
    assert data["y"].shape == (D, N, T, k)
    assert data["event_time"].shape == (D, N, T, k)
    assert data["censor_time"].shape == (D, N, T, k)


def test_draws_independent(dims: tuple[int, int, int, int]) -> None:
    """Draws along the leading dimension are independent."""
    N, T, p, k = dims
    data, _ = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .gaussian()
        .event_time()
        .censor_time(horizon=2.0)
        .draw(seed=0, draws=7)
    )
    assert not data["y"][0].equal(data["y"][1])


def test_draws_none_base_shape(dims: tuple[int, int, int, int]) -> None:
    """draws=None on a full pipeline gives base shape."""
    N, T, p, k = dims
    data, _ = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .gaussian()
        .event_time()
        .censor_time(horizon=2.0)
        .draw(seed=0)
    )
    assert data["y"].shape == (N, T, k)


def test_concrete_tensor_prior(dims: tuple[int, int, int, int]) -> None:
    """A concrete tensor passed as a prior flows through unchanged."""
    N, T, p, k = dims
    X_fixed = torch.ones(N, T, p)
    data, _ = Simulation(N, T, p, k).fixed_effects(X=X_fixed).gaussian().draw(seed=0)
    assert data["X"].equal(X_fixed)


def test_full_chain(dims: tuple[int, int, int, int]) -> None:
    """Full pipeline produces all expected keys."""
    N, T, p, k = dims
    data, params = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .gaussian()
        .missing_y(0.2)
        .event_time(shape=1.5)
        .censor_time(horizon=3.0)
        .tokenize(vocab_size=50)
        .missing_x(0.2)
        .draw(seed=1)
    )
    assert "y" in data and "event_time" in data and "censor_time" in data
    assert "beta" in params
