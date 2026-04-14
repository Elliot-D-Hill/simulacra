import torch
import torch.distributions as dist

from simulacra import PredictorData, simulate
from simulacra.transforms import random_effects

# --- custom points ---


def test_custom_points(dims: tuple[int, int, int, int]) -> None:
    """Custom points with the correct (n, t, 1) shape survive through the pipeline."""
    N, T, p, _ = dims
    grid = torch.linspace(0.0, 10.0, T).reshape(1, T, 1).expand(N, T, 1)
    data = simulate(torch.randn(N, T, p), points=grid).gaussian().draw(seed=0)
    assert data.points.equal(grid)


# --- missing data ---


def test_missing_x(dims: tuple[int, int, int, int]) -> None:
    """missing_x injects NaNs into X."""
    N, T, p, _ = dims
    data = simulate(torch.randn(N, T, p)).gaussian().missing_x(0.3).draw(seed=1)
    assert data.X.isnan().any()


def test_missing_y(dims: tuple[int, int, int, int]) -> None:
    """missing_y injects NaNs into y."""
    N, T, p, _ = dims
    data = simulate(torch.randn(N, T, p)).poisson().missing_y(0.3).draw(seed=1)
    assert data.y.shape == (N, T, 1)
    assert data.y.isnan().any()


# --- event / censor time ---


def test_event_time(dims: tuple[int, int, int, int]) -> None:
    """Weibull response produces positive event times with correct shape."""
    N, T, p, _ = dims
    data = simulate(torch.randn(N, T, p)).weibull(shape=1.0).draw(seed=1)
    assert data.y.shape == (N, T, 1)
    assert (data.y > 0).all()


def test_log_logistic_event_time(dims: tuple[int, int, int, int]) -> None:
    """Log-logistic produces positive event times with correct shape."""
    N, T, p, _ = dims
    data = simulate(torch.randn(N, T, p)).log_logistic(shape=2.0).draw(seed=1)
    assert data.y.shape == (N, T, 1)
    assert (data.y > 0).all()


def test_gompertz_event_time(dims: tuple[int, int, int, int]) -> None:
    """Gompertz produces positive event times with correct shape."""
    N, T, p, _ = dims
    data = simulate(torch.randn(N, T, p)).gompertz(shape=0.5).draw(seed=1)
    assert data.y.shape == (N, T, 1)
    assert (data.y > 0).all()


def test_exponential_positive_support(dims: tuple[int, int, int, int]) -> None:
    """Exponential samples are strictly positive."""
    N, T, p, _ = dims
    d = simulate(torch.randn(N, T, p)).exponential().draw(seed=42)
    assert d.y.shape == (N, T, 1)
    assert (d.y > 0).all()


def test_gamma_survival_pipeline(dims: tuple[int, int, int, int]) -> None:
    """Gamma response works through the full survival pipeline."""
    N, T, p, _ = dims
    data = (
        simulate(torch.randn(N, T, p))
        .gamma(concentration=2.0)
        .censor(horizon=2.0)
        .draw(seed=1)
    )
    assert data.indicator.numel() > 0
    assert data.time_to_event.numel() > 0


def test_censor_time(dims: tuple[int, int, int, int]) -> None:
    """censor bounds observed times relative to points plus horizon."""
    N, T, p, _ = dims
    horizon = 2.0
    data = simulate(torch.randn(N, T, p)).weibull().censor(horizon=horizon).draw(seed=1)
    bound = data.points[..., :1] + horizon
    assert (data.censor_time <= bound).all()


def test_censor_scalar_dropout_k_outcomes(dims: tuple[int, int, int, int]) -> None:
    """Scalar dropout: censor_time is (n, t, 1) regardless of k (per-subject censoring)."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .weibull()
        .censor(horizon=2.0)
        .draw(seed=1)
    )
    assert data.y.shape == (N, T, k)
    assert data.event_time.shape == (N, T, k)
    assert data.censor_time.shape == (N, T, 1)
    assert data.observed_time.shape == (N, T, k)


# --- random effects ---


def test_default_membership_shapes(dims: tuple[int, int, int, int]) -> None:
    """Default W is round-robin with T=1 (constant over time)."""
    N, T, p, k = dims
    L, q = 3, 2
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .random_effects(levels=L, q=q)
        .gaussian()
        .draw(seed=0)
    )
    re = data.random_effect[0]
    assert re.W.shape == (N, 1, L)
    assert re.B.shape == (N, T, q)
    assert re.b.shape == (L, q, k)
    assert data.eta.shape == (N, T, k)


def test_default_membership_round_robin(dims: tuple[int, int, int, int]) -> None:
    """Default W is one-hot round-robin: each row selects exactly one level."""
    N, T, p, k = dims
    levels = 3
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .random_effects(levels=levels, q=2)
        .gaussian()
        .draw(seed=0)
    )
    W = data.random_effect[0].W
    assert torch.allclose(W.sum(-1), torch.ones(N, 1))
    assert ((W == 0) | (W == 1)).all()
    expected = torch.arange(N) % levels
    assert torch.equal(W.squeeze(-2).argmax(-1), expected)


def test_default_membership_constant_across_time(
    dims: tuple[int, int, int, int],
) -> None:
    """Default W has T=1, implying constant membership via broadcasting."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .random_effects(levels=3, q=2)
        .gaussian()
        .draw(seed=0)
    )
    assert data.random_effect[0].W.shape[1] == 1


def test_random_effects_with_draws(dims: tuple[int, int, int, int]) -> None:
    """Draws dimension propagates through random effects."""
    N, T, p, k = dims
    D, L, q = 7, 3, 2
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .random_effects(levels=L, q=q)
        .gaussian()
        .draw(seed=0, draws=D)
    )
    re = data.random_effect[0]
    assert re.W.shape == (D, N, 1, L)
    assert re.B.shape == (D, N, T, q)
    assert re.b.shape == (D, L, q, k)
    assert data.eta.shape == (D, N, T, k)


def test_explicit_dirichlet_membership(dims: tuple[int, int, int, int]) -> None:
    """Explicit Dirichlet prior yields positive soft membership that sums to 1."""
    N, T, p, k = dims
    L = 3
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .random_effects(levels=L, q=2, W=dist.Dirichlet(torch.ones(L)))
        .gaussian()
        .draw(seed=0)
    )
    W = data.random_effect[0].W
    assert W.shape == (N, 1, L)
    assert torch.allclose(W.sum(-1), torch.ones(N, 1))
    assert (W > 0).all()


def test_multiple_random_effects(dims: tuple[int, int, int, int]) -> None:
    """Two RE terms produce independently indexed params that accumulate in eta."""
    N, T, p, k = dims
    L, q = 3, 2
    L2, q2 = 2, 3
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .random_effects(levels=L, q=q)
        .random_effects(levels=L2, q=q2)
        .gaussian()
        .draw(seed=0)
    )
    re0 = data.random_effect[0]
    assert re0.W.shape == (N, 1, L)
    assert re0.B.shape == (N, T, q)
    assert re0.b.shape == (L, q, k)
    re1 = data.random_effect[1]
    assert re1.W.shape == (N, 1, L2)
    assert re1.B.shape == (N, T, q2)
    assert re1.b.shape == (L2, q2, k)
    assert data.eta.shape == (N, T, k)


def test_einsum_equivalence(dims: tuple[int, int, int, int]) -> None:
    """Manual einsum matches the random_effects transform."""
    N, T, p, k = dims
    L, q = 3, 2
    torch.manual_seed(42)  # type: ignore[no-untyped-call]
    indices = torch.arange(N) % L
    W_test = torch.nn.functional.one_hot(indices, L).unsqueeze(1).float()  # (N, 1, L)
    B_test = torch.randn(N, T, q)
    b_test = torch.randn(L, q, k)
    eta_manual = torch.einsum("ntl,ntr,lrk->ntk", W_test, B_test, b_test)
    points = torch.arange(T, dtype=torch.float).reshape(1, T, 1).expand(N, T, 1)
    base_data = PredictorData(
        X=torch.zeros(N, T, p),
        points=points,
        eta=torch.zeros(N, T, k),
        beta=torch.zeros(p, k),
    )
    re_data = random_effects(base_data, L, q, W_test, B_test, b_test)
    assert eta_manual.equal(re_data.eta)


# --- activation ---


def test_activation_relu_clips_negatives(dims: tuple[int, int, int, int]) -> None:
    """ReLU activation zeroes negative values in eta."""
    N, T, p, _ = dims
    data = simulate(torch.randn(N, T, p)).activation().gaussian().draw(seed=0)
    assert (data.eta >= 0).all()


def test_activation_custom_function(dims: tuple[int, int, int, int]) -> None:
    """Custom activation applies the given function to eta."""
    N, T, p, _ = dims
    data = simulate(torch.randn(N, T, p)).activation(torch.abs).gaussian().draw(seed=0)
    assert (data.eta >= 0).all()


# --- constant y ---


def test_constant_y_gaussian(dims: tuple[int, int, int, int]) -> None:
    """constant_y broadcasts y so it is identical at every timepoint."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .gaussian()
        .constant_y()
        .draw(seed=0)
    )
    assert data.y.shape == (N, T, k)
    assert data.X.shape == (N, T, p)
    assert data.eta.shape == (N, T, k)
    for i in range(1, T):
        assert data.y[:, 0, :].equal(data.y[:, i, :])


def test_constant_y_poisson(dims: tuple[int, int, int, int]) -> None:
    """constant_y works with poisson family."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .poisson()
        .constant_y()
        .draw(seed=0)
    )
    for i in range(1, T):
        assert data.y[:, 0, :].equal(data.y[:, i, :])


def test_constant_y_bernoulli(dims: tuple[int, int, int, int]) -> None:
    """constant_y works with bernoulli family."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .bernoulli()
        .constant_y()
        .draw(seed=0)
    )
    for i in range(1, T):
        assert data.y[:, 0, :].equal(data.y[:, i, :])


def test_constant_y_with_draws(dims: tuple[int, int, int, int]) -> None:
    """Draws dimension works with constant_y."""
    N, T, p, k = dims
    D = 7
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .gaussian()
        .constant_y()
        .draw(seed=0, draws=D)
    )
    assert data.y.shape == (D, N, T, k)
    for i in range(1, T):
        assert data.y[:, :, 0, :].equal(data.y[:, :, i, :])


def test_normal_y_varies_along_time(dims: tuple[int, int, int, int]) -> None:
    """Without constant_y, y varies along T."""
    N, T, p, _ = dims
    data = simulate(torch.randn(N, T, p)).gaussian().draw(seed=0)
    assert not data.y[:, 0, :].equal(data.y[:, 1, :])
