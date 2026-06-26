import pytest
import torch
import torch.distributions as dist
import torch.nn.functional as F

from simulacra import PredictorData, ResponseData, simulate
from simulacra.covariance import MaternOrder, matern_kernel
from simulacra.survival import censor
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


def test_censor_competing_risks_resolution() -> None:
    """Hardcoded race: cell 0's cause 0 is soonest (event); cell 1's censoring time beats
    every cause (censored, sentinel cause k = 2, all-zero indicator)."""
    data = ResponseData(
        X=torch.zeros(1, 2, 1),
        points=torch.zeros(1, 2, 1),
        eta=torch.zeros(1, 2, 2),
        beta=torch.zeros(1, 2),
        y=torch.tensor([[[2.0, 5.0], [5.0, 8.0]]]),
    )
    result = censor(data, dropout=torch.tensor([[[3.0], [3.0]]]))
    assert result.cause.equal(torch.tensor([[[0], [2]]]))
    assert result.time_to_event.equal(torch.tensor([[[2.0], [3.0]]]))
    assert result.indicator.equal(torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]))
    assert result.censor_time.equal(torch.tensor([[[3.0], [3.0]]]))


def test_censor_tie_resolves_to_event() -> None:
    """A cause time equal to the censoring time resolves to the event: the cause column
    precedes the censor column, so the arg-min (lowest index) picks the cause."""
    data = ResponseData(
        X=torch.zeros(1, 1, 1),
        points=torch.zeros(1, 1, 1),
        eta=torch.zeros(1, 1, 2),
        beta=torch.zeros(1, 2),
        y=torch.tensor([[[3.0, 8.0]]]),
    )
    result = censor(data, dropout=torch.tensor([[[3.0]]]))
    assert result.cause.equal(torch.tensor([[[0]]]))
    assert result.indicator.equal(torch.tensor([[[1.0, 0.0]]]))


def test_censor_single_event_degeneracy() -> None:
    """k = 1 reduces to standard single-event TTE: event when y precedes censoring
    (cause 0), else censored (sentinel cause 1, all-zero indicator)."""
    data = ResponseData(
        X=torch.zeros(1, 2, 1),
        points=torch.zeros(1, 2, 1),
        eta=torch.zeros(1, 2, 1),
        beta=torch.zeros(1, 1),
        y=torch.tensor([[[2.0], [5.0]]]),
    )
    result = censor(data, dropout=torch.tensor([[[3.0], [3.0]]]))
    assert result.cause.equal(torch.tensor([[[0], [1]]]))
    assert result.time_to_event.equal(torch.tensor([[[2.0], [3.0]]]))
    assert result.indicator.equal(torch.tensor([[[1.0], [0.0]]]))


def test_censor_competing_risks_shapes(dims: tuple[int, int, int, int]) -> None:
    """Competing-risks censor collapses the k causes to one outcome per cell: scalar
    time_to_event/censor_time/cause (n, t, 1) and a (n, t, k) cause indicator."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .weibull()
        .censor(horizon=2.0)
        .draw(seed=1)
    )
    assert data.time_to_event.shape == (N, T, 1)
    assert data.censor_time.shape == (N, T, 1)
    assert data.cause.shape == (N, T, 1)
    assert data.cause.dtype == torch.long
    assert data.indicator.shape == (N, T, k)


def test_censor_indicator_one_hot_or_censored(dims: tuple[int, int, int, int]) -> None:
    """Each cell is exactly one cause or censored: the indicator row sums to 1 (event) or
    0 (censored), with 0/1 entries."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .weibull()
        .censor(horizon=2.0)
        .draw(seed=1)
    )
    row_sum = data.indicator.sum(-1)
    assert ((row_sum == 0) | (row_sum == 1)).all()
    assert ((data.indicator == 0) | (data.indicator == 1)).all()


def test_censor_discretize_pipeline(dims: tuple[int, int, int, int]) -> None:
    """censor then discretize yields per-cause discretized exposure (n, t, k, j) >= 0."""
    N, T, p, k = dims
    boundaries = torch.tensor([0.0, 1.0, 2.0, 4.0])
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .weibull()
        .censor(horizon=3.0)
        .discretize(boundaries)
        .draw(seed=1)
    )
    assert data.discrete_event_time.shape == (N, T, k, 3)
    assert (data.discrete_event_time >= 0).all()


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
    z = torch.randn(L, q, k)
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .random_effects(W=W, B=B, z=z)
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
        .random_effects(W=W1, B=torch.randn(N, T, q), z=torch.randn(L, q, k))
        .random_effects(W=W2, B=torch.randn(N, T, q2), z=torch.randn(L2, q2, k))
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
    z = torch.randn(L, q, k)
    eta_manual = torch.einsum("ntl,ntr,lrk->ntk", W, B, z)
    points = torch.arange(T, dtype=torch.float).reshape(1, T, 1).expand(N, T, 1)
    base_data = PredictorData(
        X=torch.zeros(N, T, p),
        points=points,
        eta=torch.zeros(N, T, k),
        beta=torch.zeros(p, k),
    )
    re_data = random_effects(base_data, W, B, z)
    assert eta_manual.equal(re_data.eta)


@pytest.fixture
def re_base() -> PredictorData:
    """Zero-filled predictor state with k=2 outcomes for random-effects draws."""
    return PredictorData(
        X=torch.zeros(1, 1, 1),
        points=torch.zeros(1, 1, 1),
        eta=torch.zeros(1, 1, 2),
        beta=torch.zeros(1, 2),
    )


def test_random_effects_identity_covariance_matches_iid(re_base: PredictorData) -> None:
    """No covariance reproduces the iid standard-normal coefficient draw."""
    L, q, k = 3, 2, 2
    W = torch.zeros(1, 1, L)
    B = torch.zeros(1, 1, q)
    torch.manual_seed(0)  # type: ignore[no-untyped-call]
    drawn = random_effects(re_base, W, B).random_effect[-1].b
    torch.manual_seed(0)  # type: ignore[no-untyped-call]
    expected = torch.randn(L, q, k)
    assert drawn.equal(expected)


def test_random_effects_covariance_recovers_kronecker(re_base: PredictorData) -> None:
    """Empirical covariance of vec(b) across levels matches Q (x) Sigma_K."""
    L = 40000
    basis_covariance = torch.tensor([[2.0, 0.0], [0.0, 0.5]])
    outcome_covariance = torch.tensor([[1.0, 0.6], [0.6, 1.0]])
    W = torch.zeros(1, 1, L)
    B = torch.zeros(1, 1, 2)
    torch.manual_seed(0)  # type: ignore[no-untyped-call]
    re = random_effects(
        re_base,
        W,
        B,
        outcome_covariance=outcome_covariance,
        basis_covariance=basis_covariance,
    )
    empirical = torch.cov(re.random_effect[-1].b.reshape(L, 4).mT)
    expected = torch.tensor(
        [
            [2.0, 1.2, 0.0, 0.0],
            [1.2, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.3],
            [0.0, 0.0, 0.3, 0.5],
        ]
    )
    assert torch.allclose(empirical, expected, atol=0.05)


def test_random_effects_outcome_covariance_only(re_base: PredictorData) -> None:
    """outcome_covariance alone correlates the k outcomes on the right axis."""
    L = 40000
    outcome_covariance = torch.tensor([[1.0, 0.6], [0.6, 1.0]])
    W = torch.zeros(1, 1, L)
    B = torch.zeros(1, 1, 1)
    torch.manual_seed(0)  # type: ignore[no-untyped-call]
    re = random_effects(re_base, W, B, outcome_covariance=outcome_covariance)
    empirical = torch.cov(re.random_effect[-1].b.reshape(L, 2).mT)
    assert torch.allclose(empirical, outcome_covariance, atol=0.05)


def test_random_effects_colors_explicit_noise(re_base: PredictorData) -> None:
    """Explicit z is scaled by the covariance: b = z @ chol(outcome).mT."""
    W = torch.zeros(1, 1, 1)
    B = torch.zeros(1, 1, 1)
    z = torch.tensor([[[1.0, 1.0]]])
    outcome_covariance = torch.tensor([[4.0, 0.0], [0.0, 9.0]])
    re = random_effects(re_base, W, B, z=z, outcome_covariance=outcome_covariance)
    assert re.random_effect[-1].b.equal(torch.tensor([[[2.0, 3.0]]]))


def test_random_effects_basis_covariance_only(re_base: PredictorData) -> None:
    """basis_covariance alone correlates the q basis columns, leaving outcomes iid."""
    L = 40000
    basis_covariance = torch.tensor([[1.0, 0.6], [0.6, 1.0]])
    W = torch.zeros(1, 1, L)
    B = torch.zeros(1, 1, 2)
    torch.manual_seed(0)  # type: ignore[no-untyped-call]
    re = random_effects(re_base, W, B, basis_covariance=basis_covariance)
    empirical = torch.cov(re.random_effect[-1].b[:, :, 0].mT)
    assert torch.allclose(empirical, basis_covariance, atol=0.05)


def test_random_intercept_constant_over_time(dims: tuple[int, int, int, int]) -> None:
    """A random intercept (q=1) with no fixed effect is identical across timepoints."""
    N, T, p, k = dims
    W = F.one_hot(torch.arange(N), N).float().unsqueeze(1).expand(N, T, N)
    B = torch.ones(N, T, 1)
    data = (
        simulate(torch.randn(N, T, p), torch.zeros(p, k))
        .random_effects(W=W, B=B)
        .gaussian()
        .draw(seed=0)
    )
    for i in range(1, T):
        assert data.eta[:, 0, :].equal(data.eta[:, i, :])


def test_ehr_frailty_weibull_sequences() -> None:
    """Per-patient frailty with correlated causes and per-cause Weibull shapes
    yields valid event sequences with the frailty recorded."""
    N, T, p, k = 50, 6, 3, 3
    W = F.one_hot(torch.arange(N), N).float().unsqueeze(1).expand(N, T, N)
    B = torch.ones(N, T, 1)
    outcome_covariance = torch.tensor(
        [[1.0, 0.4, 0.4], [0.4, 1.0, 0.4], [0.4, 0.4, 1.0]]
    )
    shape = torch.tensor([0.5, 1.0, 2.0])
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .random_effects(W=W, B=B, outcome_covariance=outcome_covariance)
        .weibull(shape=shape)
        .draw(seed=0)
    )
    assert data.y.shape == (N, T, k)
    assert (data.y > 0).all()
    assert data.random_effect[-1].b.shape == (N, 1, k)


def test_identity_time_basis_yields_per_timepoint_effect(
    dims: tuple[int, int, int, int],
) -> None:
    """An identity time basis (B = I_T, q = T) routes coefficient column t to
    timepoint t, so eta equals the coefficients per subject. This is the mechanism
    that turns a basis_covariance into a temporal covariance over the eta."""
    N, T, p, k = dims
    W = F.one_hot(torch.arange(N), N).float().unsqueeze(1).expand(N, T, N)
    B = torch.eye(T).unsqueeze(0).expand(N, T, T)
    z = torch.randn(N, T, k)
    data = (
        simulate(torch.zeros(N, T, p), torch.zeros(p, k))
        .random_effects(W=W, B=B, z=z)
        .gaussian()
        .draw(seed=0)
    )
    assert data.eta.equal(z)


def test_matern_basis_covariance_recovers_temporal_kernel() -> None:
    """basis_covariance = matern_kernel(grid) over an identity time basis gives the
    coefficients a stationary temporal covariance K_T: the OU dynamic-frailty form."""
    levels, T, k = 40000, 5, 1
    grid = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    K_T = matern_kernel(grid, order=MaternOrder.HALF)
    base = PredictorData(
        X=torch.zeros(1, 1, 1),
        points=torch.zeros(1, 1, 1),
        eta=torch.zeros(1, 1, k),
        beta=torch.zeros(1, k),
    )
    W = torch.zeros(1, 1, levels)
    B = torch.zeros(1, 1, T)
    torch.manual_seed(0)  # type: ignore[no-untyped-call]
    re = random_effects(base, W, B, basis_covariance=K_T)
    empirical = torch.cov(re.random_effect[-1].b[:, :, 0].mT)
    assert torch.allclose(empirical, K_T, atol=0.05)


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
