import pytest
import torch
import torch.distributions as dist

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


def test_points_tensor_mismatch_raises() -> None:
    """A tensor points with wrong T raises ValueError naming both dims."""
    with pytest.raises(ValueError, match=r"points.*size 10.*t=5"):
        simulate(n=4, t=5, p=3, points=torch.zeros(10))


def test_points_tensor_2d_mismatch_raises() -> None:
    """A rank-2+ tensor whose second-to-last axis mismatches t raises."""
    with pytest.raises(ValueError, match=r"points.*size 4.*t=5"):
        simulate(n=4, t=5, p=3, points=torch.zeros(4, 1))


def test_points_tensor_1d_match_succeeds() -> None:
    """1-D tensor of length t is accepted and broadcast to (..., t, 1)."""
    data = (
        simulate(n=4, t=5, p=3, points=torch.arange(5.0))
        .fixed_effects(k=2)
        .gaussian()
        .draw(seed=0)
    )
    assert data.points.shape == (4, 5, 1)


def test_points_tensor_2d_match_succeeds() -> None:
    """Rank-2 tensor of shape (t, 1) is accepted."""
    grid = torch.arange(5.0).unsqueeze(-1)
    data = (
        simulate(n=4, t=5, p=3, points=grid).fixed_effects(k=2).gaussian().draw(seed=0)
    )
    assert data.points.shape == (4, 5, 1)


def test_points_tensor_leading_batch_dims_match_succeeds() -> None:
    """A pre-batched grid with leading singleton dims and correct T succeeds."""
    grid = torch.zeros(1, 5, 1)
    data = (
        simulate(n=4, t=5, p=3, points=grid).fixed_effects(k=2).gaussian().draw(seed=0)
    )
    assert data.points.shape == (4, 5, 1)


def test_points_distribution_default_succeeds() -> None:
    """Distribution points bypass the Tensor shape check."""
    data = (
        simulate(n=4, t=5, p=3, points=dist.Exponential(2.0))
        .fixed_effects(k=2)
        .gaussian()
        .draw(seed=0)
    )
    assert data.points.shape == (4, 5, 1)


def test_points_validation_fails_at_call_time() -> None:
    """The ValueError fires at simulate() call, not at draw()."""
    with pytest.raises(ValueError):
        simulate(n=4, t=5, p=3, points=torch.zeros(7))


def test_X_tensor_wrong_n_raises() -> None:
    """X tensor with mismatched N axis raises."""
    with pytest.raises(ValueError, match=r"X.*size 7.*n=4"):
        simulate(n=4, t=5, p=3, X=torch.zeros(7, 5, 3))


def test_X_tensor_wrong_t_raises() -> None:
    """X tensor with mismatched T axis raises."""
    with pytest.raises(ValueError, match=r"X.*size 9.*t=5"):
        simulate(n=4, t=5, p=3, X=torch.zeros(4, 9, 3))


def test_X_tensor_wrong_p_raises() -> None:
    """X tensor with mismatched p axis raises."""
    with pytest.raises(ValueError, match=r"X.*size 2.*p=3"):
        simulate(n=4, t=5, p=3, X=torch.zeros(4, 5, 2))


def test_X_tensor_full_rank_match_succeeds() -> None:
    """X tensor with full (n, t, p) shape succeeds."""
    data = (
        simulate(n=4, t=5, p=3, X=torch.ones(4, 5, 3))
        .fixed_effects(k=2)
        .gaussian()
        .draw(seed=0)
    )
    assert data.X.shape == (4, 5, 3)


def test_X_tensor_broadcast_singletons_succeeds() -> None:
    """X tensor with singleton axes broadcasts cleanly."""
    data = (
        simulate(n=4, t=5, p=3, X=torch.ones(1, 5, 1))
        .fixed_effects(k=2)
        .gaussian()
        .draw(seed=0)
    )
    assert data.X.shape == (1, 5, 1)


def test_X_tensor_upstream_batch_dims_succeed() -> None:
    """X tensor with extra leading batch dims passes trailing-axis check."""
    X = torch.ones(2, 4, 5, 3)
    data = simulate(n=4, t=5, p=3, X=X).fixed_effects(k=2).gaussian().draw(seed=0)
    assert data.X.shape == (2, 4, 5, 3)


def test_X_distribution_wrong_batch_shape_raises() -> None:
    """A distribution whose batch_shape disagrees with (n, t, p) is caught."""
    with pytest.raises(ValueError, match=r"X.*size 7.*p=3"):
        simulate(n=4, t=5, p=3, X=dist.Normal(torch.zeros(7), 1.0))


def test_points_distribution_wrong_batch_shape_raises() -> None:
    """A distribution points whose batch_shape disagrees with t is caught."""
    with pytest.raises(ValueError, match=r"points.*size 7.*t=5"):
        simulate(n=4, t=5, p=3, points=dist.Exponential(torch.ones(7, 1)))


def test_dim_defaults_are_one_when_all_unspecified() -> None:
    """simulate() with no args and scalar priors collapses to (1, 1, 1)."""
    data = simulate().fixed_effects(k=2).gaussian().draw(seed=0)
    assert data.X.shape == (1, 1, 1)


def test_t_inferred_from_1d_points_tensor() -> None:
    """A 1-D points tensor sets t from its length when t is omitted."""
    data = (
        simulate(points=torch.arange(10.0)).fixed_effects(k=2).gaussian().draw(seed=0)
    )
    assert data.points.shape == (1, 10, 1)


def test_t_inferred_from_2d_points_tensor() -> None:
    """A rank-2 points tensor of shape (T, 1) sets t from axis -2."""
    data = (
        simulate(points=torch.arange(10.0).unsqueeze(-1))
        .fixed_effects(k=2)
        .gaussian()
        .draw(seed=0)
    )
    assert data.points.shape == (1, 10, 1)


def test_all_dims_inferred_from_X_tensor() -> None:
    """A full-rank X tensor sets n, t, p simultaneously."""
    data = simulate(X=torch.ones(4, 5, 3)).fixed_effects(k=2).gaussian().draw(seed=0)
    assert data.X.shape == (4, 5, 3)


def test_t_inference_agrees_across_X_and_points() -> None:
    """Consistent T across X and points succeeds."""
    data = (
        simulate(X=torch.ones(4, 5, 3), points=torch.arange(5.0))
        .fixed_effects(k=2)
        .gaussian()
        .draw(seed=0)
    )
    assert data.X.shape == (4, 5, 3)
    assert data.points.shape == (4, 5, 1)


def test_priors_disagree_raises() -> None:
    """X and points with disagreeing T raise a domain-level ValueError."""
    with pytest.raises(ValueError, match=r"disagree"):
        simulate(X=torch.ones(4, 5, 3), points=torch.arange(7.0))


def test_explicit_int_overrides_inference() -> None:
    """Explicit int is honored; disagreement with a prior goes through _check_axis."""
    with pytest.raises(ValueError, match=r"points.*size 10.*t=5"):
        simulate(t=5, points=torch.arange(10.0))


def test_n_inferred_from_X_with_t_explicit() -> None:
    """Mixed explicit+inferred dims: X sets n; t explicit."""
    data = (
        simulate(t=5, X=torch.ones(4, 5, 3)).fixed_effects(k=2).gaussian().draw(seed=0)
    )
    assert data.X.shape == (4, 5, 3)


def test_dim_inferred_from_distribution_batch_shape() -> None:
    """A distribution with non-trivial batch_shape contributes to inference."""
    X = dist.Normal(torch.zeros(4, 5, 3), 1.0)
    data = simulate(X=X).fixed_effects(k=2).gaussian().draw(seed=0)
    assert data.X.shape == (4, 5, 3)
