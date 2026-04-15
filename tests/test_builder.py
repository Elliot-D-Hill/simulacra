import pytest
import torch

from simulacra import simulate

# --- type-state enforcement ---


def test_predictor_methods(dims: tuple[int, int, int, int]) -> None:
    """Predictor has tokenize and draw, but not missing_x or missing_y."""
    N, T, p, k = dims
    pred = simulate(torch.randn(N, T, p), torch.randn(p, k))
    assert hasattr(pred, "tokenize")
    assert hasattr(pred, "draw")
    assert not hasattr(pred, "missing_x")
    assert not hasattr(pred, "missing_y")


def test_response_methods(dims: tuple[int, int, int, int]) -> None:
    """Response has missing_x and missing_y."""
    N, T, p, k = dims
    resp = simulate(torch.randn(N, T, p), torch.randn(p, k)).gaussian()
    assert hasattr(resp, "missing_x")
    assert hasattr(resp, "missing_y")
    assert not hasattr(resp, "tokenize")


def test_positive_support_response_methods(dims: tuple[int, int, int, int]) -> None:
    """PositiveSupportResponse has survival methods and missingness transforms."""
    N, T, p, k = dims
    resp = simulate(torch.randn(N, T, p), torch.randn(p, k)).weibull()
    assert hasattr(resp, "competing_risks")
    assert hasattr(resp, "censor")
    assert hasattr(resp, "missing_x")
    assert hasattr(resp, "missing_y")


def test_all_positive_support_families_have_survival_methods(
    dims: tuple[int, int, int, int],
) -> None:
    """All positive-support families expose censor and competing_risks."""
    N, T, p, k = dims
    base = simulate(torch.randn(N, T, p), torch.randn(p, k))
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
    resp = simulate(torch.randn(N, T, p), torch.randn(p, k)).gaussian()
    assert not hasattr(resp, "competing_risks")
    assert not hasattr(resp, "censor")


def test_survival_methods(dims: tuple[int, int, int, int]) -> None:
    """Survival has missingness transforms and discretize."""
    N, T, p, k = dims
    surv = (
        simulate(torch.randn(N, T, p), torch.randn(p, k)).weibull().censor(horizon=2.0)
    )
    assert hasattr(surv, "missing_x")
    assert hasattr(surv, "missing_y")
    assert hasattr(surv, "discretize")


def test_constant_y_available_on_response(dims: tuple[int, int, int, int]) -> None:
    """constant_y is available on all response types."""
    N, T, p, k = dims
    resp = simulate(torch.randn(N, T, p), torch.randn(p, k)).gaussian()
    assert hasattr(resp, "constant_y")


# --- pipeline ---


def test_reproducibility(dims: tuple[int, int, int, int]) -> None:
    """Same seed produces identical draws."""
    N, T, p, k = dims
    X = torch.randn(N, T, p)
    beta = torch.randn(p, k)
    data1 = simulate(X, beta).gaussian().draw(seed=0)
    data2 = simulate(X, beta).gaussian().draw(seed=0)
    assert data1.y.equal(data2.y)


def test_immutability(dims: tuple[int, int, int, int]) -> None:
    """Branching from a shared base reuses X but produces different y."""
    N, T, p, k = dims
    base = simulate(torch.randn(N, T, p), torch.randn(p, k))
    da = base.gaussian().draw(seed=0)
    db = base.poisson().draw(seed=0)
    assert da.X.equal(db.X)
    assert not da.y.equal(db.y)


def test_base_shape(dims: tuple[int, int, int, int]) -> None:
    """Default pipeline gives [N, T, k] shape."""
    N, T, p, k = dims
    data = simulate(torch.randn(N, T, p), torch.randn(p, k)).gaussian().draw(seed=0)
    assert data.X.shape == (N, T, p)
    assert data.y.shape == (N, T, k)


def test_explicit_beta_sets_k(dims: tuple[int, int, int, int]) -> None:
    """beta tensor's last dim sets the output dim."""
    N, T, p, k = dims
    beta = torch.randn(p, k)
    data = simulate(torch.randn(N, T, p), beta).gaussian().draw(seed=0)
    assert data.y.shape == (N, T, k)
    assert data.beta.shape == (p, k)


def test_concrete_X_flows_through(dims: tuple[int, int, int, int]) -> None:
    """A concrete X tensor flows through the pipeline unchanged."""
    N, T, p, k = dims
    X_fixed = torch.ones(N, T, p)
    data = simulate(X_fixed, torch.randn(p, k)).gaussian().draw(seed=0)
    assert data.X.equal(X_fixed)


def test_full_chain(dims: tuple[int, int, int, int]) -> None:
    """Full pipeline produces all expected fields."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .tokenize(weight=torch.randn(k, 50))
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
    """X.p and beta.p must agree; caught by jaxtyping at fixed_effects boundary."""
    with pytest.raises(Exception):
        simulate(torch.randn(4, 5, 3), torch.randn(7, 2)).gaussian().draw(seed=0)


def test_points_1d_tensor_no_auto_promotion() -> None:
    """A 1-D points tensor is rejected; user must pass correct (n, t, 1) shape."""
    with pytest.raises(Exception):
        simulate(
            torch.randn(4, 5, 3), torch.randn(3, 2), points=torch.arange(5.0)
        ).gaussian().draw(seed=0)


def test_X_leading_batch_dims_rejected() -> None:
    """4-D X is rejected; library is strictly single-realization."""
    with pytest.raises(Exception):
        simulate(torch.randn(7, 4, 5, 3), torch.randn(3, 2)).gaussian().draw(seed=0)


def test_beta_none_defaults_to_k1_random(dims: tuple[int, int, int, int]) -> None:
    """simulate without beta defaults to random (p, 1) — one-liner preserved."""
    N, T, p, _ = dims
    data = simulate(torch.randn(N, T, p)).gaussian().draw(seed=0)
    assert data.beta.shape == (p, 1)
    assert data.y.shape == (N, T, 1)


def test_points_correct_shape_succeeds() -> None:
    """A points tensor with shape (n, t, 1) is accepted."""
    grid = torch.arange(5.0).reshape(5, 1).unsqueeze(0).expand(4, 5, 1)
    data = (
        simulate(torch.randn(4, 5, 3), torch.randn(3, 2), points=grid)
        .gaussian()
        .draw(seed=0)
    )
    assert data.points.shape == (4, 5, 1)
