import pytest
import torch

from simulacra import simulate

# --- type-state enforcement ---


def test_predictor_methods(dims: tuple[int, int, int, int]) -> None:
    """Predictor has tokenize and draw, but not missing_x or missing_y."""
    N, T, p, _ = dims
    pred = simulate(torch.randn(N, T, p))
    assert hasattr(pred, "tokenize")
    assert hasattr(pred, "draw")
    assert not hasattr(pred, "missing_x")
    assert not hasattr(pred, "missing_y")


def test_response_methods(dims: tuple[int, int, int, int]) -> None:
    """Response has missing_x and missing_y."""
    N, T, p, _ = dims
    resp = simulate(torch.randn(N, T, p)).gaussian()
    assert hasattr(resp, "missing_x")
    assert hasattr(resp, "missing_y")
    assert not hasattr(resp, "tokenize")


def test_positive_support_response_methods(dims: tuple[int, int, int, int]) -> None:
    """PositiveSupportResponse has survival methods and missingness transforms."""
    N, T, p, _ = dims
    resp = simulate(torch.randn(N, T, p)).weibull()
    assert hasattr(resp, "competing_risks")
    assert hasattr(resp, "censor")
    assert hasattr(resp, "missing_x")
    assert hasattr(resp, "missing_y")


def test_all_positive_support_families_have_survival_methods(
    dims: tuple[int, int, int, int],
) -> None:
    """All positive-support families expose censor and competing_risks."""
    N, T, p, _ = dims
    base = simulate(torch.randn(N, T, p))
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
    N, T, p, _ = dims
    resp = simulate(torch.randn(N, T, p)).gaussian()
    assert not hasattr(resp, "competing_risks")
    assert not hasattr(resp, "censor")


def test_survival_methods(dims: tuple[int, int, int, int]) -> None:
    """Survival has missingness transforms and discretize."""
    N, T, p, _ = dims
    surv = simulate(torch.randn(N, T, p)).weibull().censor(horizon=2.0)
    assert hasattr(surv, "missing_x")
    assert hasattr(surv, "missing_y")
    assert hasattr(surv, "discretize")


def test_constant_y_available_on_response(dims: tuple[int, int, int, int]) -> None:
    """constant_y is available on all response types."""
    N, T, p, _ = dims
    resp = simulate(torch.randn(N, T, p)).gaussian()
    assert hasattr(resp, "constant_y")


# --- pipeline ---


def test_reproducibility(dims: tuple[int, int, int, int]) -> None:
    """Same seed produces identical draws."""
    N, T, p, _ = dims
    X = torch.randn(N, T, p)
    data1 = simulate(X).gaussian().draw(seed=0)
    data2 = simulate(X).gaussian().draw(seed=0)
    assert data1.y.equal(data2.y)


def test_immutability(dims: tuple[int, int, int, int]) -> None:
    """Branching from a shared base reuses X but produces different y."""
    N, T, p, _ = dims
    base = simulate(torch.randn(N, T, p))
    da = base.gaussian().draw(seed=0)
    db = base.poisson().draw(seed=0)
    assert da.X.equal(db.X)
    assert not da.y.equal(db.y)


def test_base_shape(dims: tuple[int, int, int, int]) -> None:
    """draws=None gives [N, T, ...] shape."""
    N, T, p, _ = dims
    data = simulate(torch.randn(N, T, p)).gaussian().draw(seed=0)
    assert data.X.shape == (N, T, p)
    assert data.y.shape == (N, T, 1)


def test_explicit_beta_sets_k(dims: tuple[int, int, int, int]) -> None:
    """Passing an explicit beta tensor with (p, k) sets the output dim."""
    N, T, p, k = dims
    beta = torch.randn(p, k)
    data = simulate(torch.randn(N, T, p), beta).gaussian().draw(seed=0)
    assert data.y.shape == (N, T, k)
    assert data.beta.shape == (p, k)


def test_draws_shape(dims: tuple[int, int, int, int]) -> None:
    """draws=D adds a leading dimension."""
    N, T, p, k = dims
    D = 7
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .weibull()
        .censor(horizon=2.0)
        .draw(seed=0, draws=D)
    )
    assert data.beta.shape == (D, p, k)
    assert data.y.shape == (D, N, T, k)
    assert data.event_time.shape == (D, N, T, k)


def test_draws_independent(dims: tuple[int, int, int, int]) -> None:
    """Draws along the leading dimension are independent."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .weibull()
        .censor(horizon=2.0)
        .draw(seed=0, draws=7)
    )
    assert not data.y[0].equal(data.y[1])


def test_draws_none_base_shape(dims: tuple[int, int, int, int]) -> None:
    """draws=None on a full pipeline gives base shape."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .weibull()
        .censor(horizon=2.0)
        .draw(seed=0)
    )
    assert data.y.shape == (N, T, k)


def test_concrete_X_flows_through(dims: tuple[int, int, int, int]) -> None:
    """A concrete X tensor flows through the pipeline unchanged."""
    N, T, p, _ = dims
    X_fixed = torch.ones(N, T, p)
    data = simulate(X_fixed).gaussian().draw(seed=0)
    assert data.X.equal(X_fixed)


def test_full_chain(dims: tuple[int, int, int, int]) -> None:
    """Full pipeline produces all expected fields."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
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


# --- shape contracts enforced by jaxtyping ---


def test_X_beta_p_mismatch_raises() -> None:
    """X.p and beta.p must agree; caught when aligning beta to X's p."""
    with pytest.raises(RuntimeError, match=r"expand|size"):
        simulate(torch.randn(4, 5, 3), torch.randn(7, 2)).gaussian().draw(seed=0)


def test_points_1d_tensor_no_auto_promotion() -> None:
    """A 1-D points tensor is rejected; user must pass correct (n, t, 1) shape."""
    with pytest.raises(RuntimeError, match=r"expand|size"):
        simulate(torch.randn(4, 5, 3), points=torch.arange(5.0)).gaussian().draw(seed=0)


def test_X_leading_batch_dims_flow_through() -> None:
    """X with leading batch dims propagates through the pipeline and combines with draws."""
    D_pre = 7
    base = simulate(torch.randn(D_pre, 10, 5, 3)).gaussian()
    assert base.draw().y.shape == (D_pre, 10, 5, 1)
    assert base.draw(draws=4).y.shape == (4, D_pre, 10, 5, 1)


def test_points_correct_shape_succeeds() -> None:
    """A points tensor with shape (n, t, 1) is accepted."""
    grid = torch.arange(5.0).reshape(1, 5, 1).expand(4, 5, 1)
    data = simulate(torch.randn(4, 5, 3), points=grid).gaussian().draw(seed=0)
    assert data.points.shape == (4, 5, 1)
