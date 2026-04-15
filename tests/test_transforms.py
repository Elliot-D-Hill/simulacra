import torch
import torch.distributions as dist
import torch.nn.functional as F

from simulacra import PredictorData, simulate
from simulacra.transforms import random_effects

# --- custom points ---


def test_custom_points(dims: tuple[int, int, int, int]) -> None:
    """Custom points with the correct (n, t, 1) shape survive through the pipeline."""
    N, T, p, k = dims
    grid = torch.linspace(0.0, 10.0, T).reshape(1, T, 1).expand(N, T, 1)
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k), points=grid)
        .gaussian()
        .draw(seed=0)
    )
    assert data.points.equal(grid)


# --- missing data ---


def test_missing_x(dims: tuple[int, int, int, int]) -> None:
    """missing_x injects NaNs into X."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .gaussian()
        .missing_x(0.3)
        .draw(seed=1)
    )
    assert data.X.isnan().any()


def test_missing_y(dims: tuple[int, int, int, int]) -> None:
    """missing_y injects NaNs into y."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .poisson()
        .missing_y(0.3)
        .draw(seed=1)
    )
    assert data.y.shape == (N, T, k)
    assert data.y.isnan().any()


# --- event / censor time ---


def test_event_time(dims: tuple[int, int, int, int]) -> None:
    """Weibull response produces positive event times with correct shape."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .weibull(shape=1.0)
        .draw(seed=1)
    )
    assert data.y.shape == (N, T, k)
    assert (data.y > 0).all()


def test_log_logistic_event_time(dims: tuple[int, int, int, int]) -> None:
    """Log-logistic produces positive event times with correct shape."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .log_logistic(shape=2.0)
        .draw(seed=1)
    )
    assert data.y.shape == (N, T, k)
    assert (data.y > 0).all()


def test_gompertz_event_time(dims: tuple[int, int, int, int]) -> None:
    """Gompertz produces positive event times with correct shape."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .gompertz(shape=0.5)
        .draw(seed=1)
    )
    assert data.y.shape == (N, T, k)
    assert (data.y > 0).all()


def test_exponential_positive_support(dims: tuple[int, int, int, int]) -> None:
    """Exponential samples are strictly positive."""
    N, T, p, k = dims
    d = simulate(torch.randn(N, T, p), torch.randn(p, k)).exponential().draw(seed=42)
    assert d.y.shape == (N, T, k)
    assert (d.y > 0).all()


def test_gamma_survival_pipeline(dims: tuple[int, int, int, int]) -> None:
    """Gamma response works through the full survival pipeline."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .gamma(concentration=2.0)
        .censor(horizon=2.0)
        .draw(seed=1)
    )
    assert data.indicator.numel() > 0
    assert data.time_to_event.numel() > 0


def test_censor_time(dims: tuple[int, int, int, int]) -> None:
    """censor bounds observed times relative to points plus horizon."""
    N, T, p, k = dims
    horizon = 2.0
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .weibull()
        .censor(horizon=horizon)
        .draw(seed=1)
    )
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


def test_dropout_none_defaults_to_exponential(dims: tuple[int, int, int, int]) -> None:
    """censor with dropout=None samples (n, t, 1) from Exponential(1) internally."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .weibull()
        .censor(horizon=10.0)
        .draw(seed=0)
    )
    assert data.censor_time.shape == (N, T, 1)
    assert (data.censor_time > 0).all()


def test_dropout_explicit_tensor(dims: tuple[int, int, int, int]) -> None:
    """censor accepts an explicit dropout tensor of shape (n, t, 1)."""
    N, T, p, k = dims
    dropout = torch.full((N, T, 1), 0.5)
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .weibull()
        .censor(dropout=dropout, horizon=10.0)
        .draw(seed=0)
    )
    assert data.censor_time.shape == (N, T, 1)


# --- random effects ---


def test_explicit_dirichlet_membership(dims: tuple[int, int, int, int]) -> None:
    """User-supplied Dirichlet sample as W flows through unchanged."""
    N, T, p, k = dims
    L, q = 3, 2
    W = dist.Dirichlet(torch.ones(L)).sample((N, T))
    B = torch.randn(N, T, q)
    b = torch.randn(L, q, k)
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .random_effects(W=W, B=B, b=b)
        .gaussian()
        .draw(seed=0)
    )
    re_W = data.random_effect[0].W
    assert re_W.shape == (N, T, L)
    assert torch.allclose(re_W.sum(-1), torch.ones(N, T))
    assert (re_W > 0).all()


def test_multiple_random_effects(dims: tuple[int, int, int, int]) -> None:
    """Two RE terms produce independently indexed params that accumulate in eta."""
    N, T, p, k = dims
    L, q = 3, 2
    L2, q2 = 2, 3
    W1 = F.one_hot(torch.arange(N) % L, L).float().unsqueeze(1).expand(N, T, L)
    W2 = F.one_hot(torch.arange(N) % L2, L2).float().unsqueeze(1).expand(N, T, L2)
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .random_effects(W=W1, B=torch.randn(N, T, q), b=torch.randn(L, q, k))
        .random_effects(W=W2, B=torch.randn(N, T, q2), b=torch.randn(L2, q2, k))
        .gaussian()
        .draw(seed=0)
    )
    re0 = data.random_effect[0]
    assert re0.W.shape == (N, T, L)
    assert re0.B.shape == (N, T, q)
    assert re0.b.shape == (L, q, k)
    re1 = data.random_effect[1]
    assert re1.W.shape == (N, T, L2)
    assert re1.B.shape == (N, T, q2)
    assert re1.b.shape == (L2, q2, k)
    assert data.eta.shape == (N, T, k)


def test_einsum_equivalence(dims: tuple[int, int, int, int]) -> None:
    """Manual einsum matches the random_effects transform."""
    N, T, p, k = dims
    L, q = 3, 2
    torch.manual_seed(42)  # type: ignore[no-untyped-call]
    indices = torch.arange(N) % L
    W = F.one_hot(indices, L).float().unsqueeze(1).expand(N, T, L)  # (N, T, L)
    B = torch.randn(N, T, q)
    b = torch.randn(L, q, k)
    eta_manual = torch.einsum("ntl,ntr,lrk->ntk", W, B, b)
    points = torch.arange(T, dtype=torch.float).reshape(1, T, 1).expand(N, T, 1)
    base_data = PredictorData(
        X=torch.zeros(N, T, p),
        points=points,
        eta=torch.zeros(N, T, k),
        beta=torch.zeros(p, k),
    )
    re_data = random_effects(base_data, W, B, b)
    assert eta_manual.equal(re_data.eta)


# --- activation ---


def test_activation_relu_clips_negatives(dims: tuple[int, int, int, int]) -> None:
    """ReLU activation zeroes negative values in eta."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .activation()
        .gaussian()
        .draw(seed=0)
    )
    assert (data.eta >= 0).all()


def test_activation_custom_function(dims: tuple[int, int, int, int]) -> None:
    """Custom activation applies the given function to eta."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .activation(torch.abs)
        .gaussian()
        .draw(seed=0)
    )
    assert (data.eta >= 0).all()


# --- gaussian covariance defaults ---


def test_covariance_none_defaults_to_eye(dims: tuple[int, int, int, int]) -> None:
    """gaussian with covariance=None uses torch.eye(k) internally."""
    N, T, p, k = dims
    data = simulate(torch.randn(N, T, p), torch.randn(p, k)).gaussian().draw(seed=0)
    assert data.y.shape == (N, T, k)


def test_covariance_explicit_scaled_identity(dims: tuple[int, int, int, int]) -> None:
    """gaussian accepts an explicit (k, k) covariance tensor."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .gaussian(covariance=torch.eye(k) * 4.0)
        .draw(seed=0)
    )
    assert data.y.shape == (N, T, k)


# --- beta / dirichlet / multinomial ---


def test_beta_response(dims: tuple[int, int, int, int]) -> None:
    """Beta samples lie strictly in (0, 1) with shape (N, T, k)."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .beta(concentration=5.0)
        .draw(seed=0)
    )
    assert data.y.shape == (N, T, k)
    assert (data.y > 0).all()
    assert (data.y < 1).all()


def test_dirichlet_response(dims: tuple[int, int, int, int]) -> None:
    """Dirichlet samples lie on the simplex with shape (N, T, k)."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .dirichlet(concentration=5.0)
        .draw(seed=0)
    )
    assert data.y.shape == (N, T, k)
    assert (data.y > 0).all()
    assert torch.allclose(data.y.sum(-1), torch.ones(N, T))


def test_multinomial_response(dims: tuple[int, int, int, int]) -> None:
    """Multinomial samples sum to num_trials with shape (N, T, k)."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .multinomial(num_trials=7)
        .draw(seed=0)
    )
    assert data.y.shape == (N, T, k)
    assert (data.y >= 0).all()
    assert data.y.sum(-1).eq(7).all()


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


def test_normal_y_varies_along_time(dims: tuple[int, int, int, int]) -> None:
    """Without constant_y, y varies along T."""
    N, T, p, k = dims
    data = simulate(torch.randn(N, T, p), torch.randn(p, k)).gaussian().draw(seed=0)
    assert not data.y[:, 0, :].equal(data.y[:, 1, :])
