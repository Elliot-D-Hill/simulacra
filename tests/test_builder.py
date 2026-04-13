import torch

from simulacra import simulate

# --- type-state enforcement ---


def test_simulation_has_no_downstream_methods(dims: tuple[int, int, int, int]) -> None:
    """simulate() exposes only covariate transforms and fixed_effects."""
    N, T, p, _ = dims
    sim = simulate(N, T, p)
    assert not hasattr(sim, "missing_x")
    assert not hasattr(sim, "tokenize")
    assert not hasattr(sim, "missing_y")
    assert not hasattr(sim, "gaussian")
    assert not hasattr(sim, "draw")


def test_predictor_methods(dims: tuple[int, int, int, int]) -> None:
    """Predictor has tokenize and draw, but not missing_x or missing_y."""
    N, T, p, k = dims
    pred = simulate(N, T, p).fixed_effects(k=k)
    assert hasattr(pred, "tokenize")
    assert hasattr(pred, "draw")
    assert not hasattr(pred, "missing_x")
    assert not hasattr(pred, "missing_y")


def test_response_methods(dims: tuple[int, int, int, int]) -> None:
    """Response has missing_x and missing_y."""
    N, T, p, k = dims
    resp = simulate(N, T, p).fixed_effects(k=k).gaussian()
    assert hasattr(resp, "missing_x")
    assert hasattr(resp, "missing_y")
    assert not hasattr(resp, "tokenize")


def test_positive_support_response_methods(dims: tuple[int, int, int, int]) -> None:
    """PositiveSupportResponse has survival methods and missingness transforms."""
    N, T, p, k = dims
    resp = simulate(N, T, p).fixed_effects(k=k).weibull()
    assert hasattr(resp, "competing_risks")
    assert hasattr(resp, "censor")
    assert hasattr(resp, "missing_x")
    assert hasattr(resp, "missing_y")


def test_all_positive_support_families_have_survival_methods(
    dims: tuple[int, int, int, int],
) -> None:
    """All positive-support families expose censor and competing_risks."""
    N, T, p, k = dims
    base = simulate(N, T, p).fixed_effects(k=k)
    families = [
        base.gamma(concentration=2.0),
        base.log_normal(),
        base.weibull(),
        base.exponential(),
        base.log_logistic(),
        base.gompertz(),
    ]
    for resp in families:
        assert hasattr(resp, "competing_risks")
        assert hasattr(resp, "censor")


def test_general_response_lacks_survival_methods(
    dims: tuple[int, int, int, int],
) -> None:
    """General families do not expose survival methods."""
    N, T, p, k = dims
    resp = simulate(N, T, p).fixed_effects(k=k).gaussian()
    assert not hasattr(resp, "competing_risks")
    assert not hasattr(resp, "censor")


def test_survival_methods(dims: tuple[int, int, int, int]) -> None:
    """Survival has missingness transforms and discretize."""
    N, T, p, k = dims
    surv = simulate(N, T, p).fixed_effects(k=k).weibull().censor(horizon=2.0)
    assert hasattr(surv, "missing_x")
    assert hasattr(surv, "missing_y")
    assert hasattr(surv, "discretize")


def test_constant_y_available_on_response(dims: tuple[int, int, int, int]) -> None:
    """constant_y is available on all response types."""
    N, T, p, k = dims
    resp = simulate(N, T, p).fixed_effects(k=k).gaussian()
    assert hasattr(resp, "constant_y")


def test_self_transition_preserves_methods(dims: tuple[int, int, int, int]) -> None:
    """z_score() returns a Simulation with the same API."""
    N, T, p, _ = dims
    sim = simulate(N, T, p).z_score()
    assert hasattr(sim, "z_score")
    assert hasattr(sim, "min_max_scale")
    assert hasattr(sim, "fixed_effects")
    assert not hasattr(sim, "gaussian")
    assert not hasattr(sim, "draw")


# --- pipeline ---


def test_reproducibility(dims: tuple[int, int, int, int]) -> None:
    """Same seed produces identical draws."""
    N, T, p, k = dims
    data1 = simulate(N, T, p).fixed_effects(k=k).gaussian().draw(seed=0)
    data2 = simulate(N, T, p).fixed_effects(k=k).gaussian().draw(seed=0)
    assert data1.y.equal(data2.y)


def test_immutability(dims: tuple[int, int, int, int]) -> None:
    """Branching from a shared base reuses X but produces different y."""
    N, T, p, k = dims
    base = simulate(N, T, p).fixed_effects(k=k)
    da = base.gaussian().draw(seed=0)
    db = base.poisson().draw(seed=0)
    assert da.X.equal(db.X)
    assert not da.y.equal(db.y)


def test_base_shape(dims: tuple[int, int, int, int]) -> None:
    """draws=None gives [N, T, ...] shape."""
    N, T, p, k = dims
    data = simulate(N, T, p).fixed_effects(k=k).gaussian().draw(seed=0)
    assert data.X.shape == (N, T, p)
    assert data.y.shape == (N, T, k)


def test_draws_shape(dims: tuple[int, int, int, int]) -> None:
    """draws=D adds a leading dimension."""
    N, T, p, k = dims
    D = 7
    data = (
        simulate(N, T, p)
        .fixed_effects(k=k)
        .weibull()
        .censor(horizon=2.0)
        .draw(seed=0, draws=D)
    )
    assert data.X.shape == (D, N, T, p)
    assert data.beta.shape == (D, p, k)
    assert data.y.shape == (D, N, T, k)
    assert data.event_time.shape == (D, N, T, k)


def test_draws_independent(dims: tuple[int, int, int, int]) -> None:
    """Draws along the leading dimension are independent."""
    N, T, p, k = dims
    data = (
        simulate(N, T, p)
        .fixed_effects(k=k)
        .weibull()
        .censor(horizon=2.0)
        .draw(seed=0, draws=7)
    )
    assert not data.y[0].equal(data.y[1])


def test_draws_none_base_shape(dims: tuple[int, int, int, int]) -> None:
    """draws=None on a full pipeline gives base shape."""
    N, T, p, k = dims
    data = (
        simulate(N, T, p).fixed_effects(k=k).weibull().censor(horizon=2.0).draw(seed=0)
    )
    assert data.y.shape == (N, T, k)


def test_concrete_tensor_prior(dims: tuple[int, int, int, int]) -> None:
    """A concrete tensor passed as X via covariates flows through unchanged."""
    N, T, p, k = dims
    X_fixed = torch.ones(N, T, p)
    data = simulate(N, T, p, X=X_fixed).fixed_effects(k=k).gaussian().draw(seed=0)
    assert data.X.equal(X_fixed)


def test_full_chain(dims: tuple[int, int, int, int]) -> None:
    """Full pipeline produces all expected fields."""
    N, T, p, k = dims
    data = (
        simulate(N, T, p)
        .fixed_effects(k=k)
        .tokenize(vocab_size=50)
        .weibull(shape=1.5)
        .censor(horizon=3.0)
        .missing_x(0.2)
        .missing_y(0.2)
        .draw(seed=1)
    )
    assert data.y.numel() > 0
    assert data.event_time.numel() > 0
    assert data.censor_time.numel() > 0
    assert data.beta.numel() > 0
