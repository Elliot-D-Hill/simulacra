import torch
import torch.distributions as dist

from simulacra import PredictorData, Simulation
from simulacra.transforms import random_effects

# --- scaling ---


def test_min_max_scale_range(dims: tuple[int, int, int, int]) -> None:
    """min_max_scale maps X into [0, 1] by default."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p).min_max_scale().fixed_effects(k=k).gaussian().draw(seed=0)
    )
    assert data.X.min() >= -1e-6
    assert data.X.max() <= 1.0 + 1e-6


def test_min_max_scale_custom_range(dims: tuple[int, int, int, int]) -> None:
    """min_max_scale respects custom low/high bounds."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p)
        .min_max_scale(low=-1.0, high=1.0)
        .fixed_effects(k=k)
        .gaussian()
        .draw(seed=0)
    )
    assert data.X.min() >= -1.0 - 1e-6
    assert data.X.max() <= 1.0 + 1e-6


# def test_z_score_moments(dims: tuple[int, int, int, int]) -> None:
#     """z_score centers each covariate column to mean~0, std~1."""
#     N, T, p, k = dims
#     data = Simulation(N, T, p).z_score().fixed_effects(k=k).gaussian().draw(seed=0)
#     column_mean = data.X.mean(dim=(-3, -2))
#     column_std = data.X.std(dim=(-3, -2))
#     assert torch.allclose(column_mean, torch.zeros(p), atol=1e-5)
#     assert torch.allclose(column_std, torch.ones(p), atol=1e-1)


def test_min_max_scale_with_draws(dims: tuple[int, int, int, int]) -> None:
    """Draws dimension propagates through min_max_scale."""
    N, T, p, k = dims
    D = 7
    data = (
        Simulation(N, T, p)
        .min_max_scale()
        .fixed_effects(k=k)
        .gaussian()
        .draw(seed=0, draws=D)
    )
    assert data.X.shape == (D, N, T, p)
    assert data.X.min() >= -1e-6
    assert data.X.max() <= 1.0 + 1e-6


def test_z_score_with_draws(dims: tuple[int, int, int, int]) -> None:
    """Draws dimension propagates through z_score."""
    N, T, p, k = dims
    D = 7
    data = (
        Simulation(N, T, p)
        .z_score()
        .fixed_effects(k=k)
        .gaussian()
        .draw(seed=0, draws=D)
    )
    assert data.X.shape == (D, N, T, p)


# --- covariate stage ---


def test_covariates_then_scaling(dims: tuple[int, int, int, int]) -> None:
    """Custom X via covariates flows through z_score into fixed_effects."""
    N, T, p, k = dims
    X_custom = torch.randn(N, T, p) * 10.0 + 50.0
    data = (
        Simulation(N, T, p)
        .covariates(X=X_custom)
        .z_score()
        .fixed_effects(k=k)
        .gaussian()
        .draw(seed=0)
    )
    column_mean = data.X.mean(dim=(-3, -2))
    assert torch.allclose(column_mean, torch.zeros(p), atol=1e-5)


def test_chained_scalings(dims: tuple[int, int, int, int]) -> None:
    """min_max_scale then z_score compose on the Covariate stage."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p)
        .min_max_scale()
        .z_score()
        .fixed_effects(k=k)
        .gaussian()
        .draw(seed=0)
    )
    column_mean = data.X.mean(dim=(-3, -2))
    assert torch.allclose(column_mean, torch.zeros(p), atol=1e-5)


def test_custom_coordinates(dims: tuple[int, int, int, int]) -> None:
    """Custom coordinates via points survive through the pipeline."""
    N, T, p, k = dims
    custom_coords = torch.linspace(0.0, 10.0, T)
    data = (
        Simulation(N, T, p)
        .points(coordinates=custom_coords)
        .fixed_effects(k=k)
        .gaussian()
        .draw(seed=0)
    )
    expected = custom_coords.unsqueeze(-1).expand(N, T, 1)
    assert data.coordinates.equal(expected)


def test_covariates_and_points_order_independent(
    dims: tuple[int, int, int, int],
) -> None:
    """covariates and points can be called in either order."""
    N, T, p, k = dims
    X_custom = torch.randn(N, T, p)
    custom_coords = torch.linspace(0.0, 10.0, T)
    d1 = (
        Simulation(N, T, p)
        .covariates(X=X_custom)
        .points(coordinates=custom_coords)
        .fixed_effects(k=k)
        .gaussian()
        .draw(seed=0)
    )
    d2 = (
        Simulation(N, T, p)
        .points(coordinates=custom_coords)
        .covariates(X=X_custom)
        .fixed_effects(k=k)
        .gaussian()
        .draw(seed=0)
    )
    assert d1.X.equal(d2.X)
    assert d1.coordinates.equal(d2.coordinates)
    assert d1.y.equal(d2.y)


# --- missing data ---


def test_missing_x(dims: tuple[int, int, int, int]) -> None:
    """missing_x injects NaNs into X."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p).fixed_effects(k=k).gaussian().missing_x(0.3).draw(seed=1)
    )
    assert data.X.isnan().any()


def test_missing_y(dims: tuple[int, int, int, int]) -> None:
    """missing_y injects NaNs into y."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p).fixed_effects(k=k).poisson().missing_y(0.3).draw(seed=1)
    )
    assert data.y.shape == (N, T, k)
    assert data.y.isnan().any()


# --- event / censor time ---


def test_event_time(dims: tuple[int, int, int, int]) -> None:
    """Weibull response produces positive event times with correct shape."""
    N, T, p, k = dims
    data = Simulation(N, T, p).fixed_effects(k=k).weibull(shape=1.0).draw(seed=1)
    assert data.y.shape == (N, T, k)
    assert (data.y > 0).all()


def test_log_logistic_event_time(dims: tuple[int, int, int, int]) -> None:
    """Log-logistic produces positive event times with correct shape."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p).fixed_effects(k=k).log_logistic(shape=2.0).draw(seed=1)
    )
    assert data.y.shape == (N, T, k)
    assert (data.y > 0).all()


def test_gompertz_event_time(dims: tuple[int, int, int, int]) -> None:
    """Gompertz produces positive event times with correct shape."""
    N, T, p, k = dims
    data = Simulation(N, T, p).fixed_effects(k=k).gompertz(shape=0.5).draw(seed=1)
    assert data.y.shape == (N, T, k)
    assert (data.y > 0).all()


def test_exponential_equals_weibull_shape_one(dims: tuple[int, int, int, int]) -> None:
    """Exponential is Weibull with shape=1."""
    N, T, p, k = dims
    d1 = Simulation(N, T, p).fixed_effects(k=k).exponential().draw(seed=42)
    d2 = Simulation(N, T, p).fixed_effects(k=k).weibull(shape=1.0).draw(seed=42)
    assert d1.y.equal(d2.y)


def test_gamma_survival_pipeline(dims: tuple[int, int, int, int]) -> None:
    """Gamma response works through the full survival pipeline."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p)
        .fixed_effects(k=k)
        .gamma(concentration=2.0)
        .censor(horizon=2.0)
        .draw(seed=1)
    )
    assert data.indicator.numel() > 0
    assert data.time_to_event.numel() > 0


def test_censor_time(dims: tuple[int, int, int, int]) -> None:
    """censor bounds observed times relative to coordinates plus horizon."""
    N, T, p, k = dims
    horizon = 2.0
    data = (
        Simulation(N, T, p)
        .fixed_effects(k=k)
        .weibull()
        .censor(horizon=horizon)
        .draw(seed=1)
    )
    bound = data.coordinates[..., :1] + horizon
    assert (data.censor_time <= bound).all()


# --- random effects ---


def test_default_membership_shapes(dims: tuple[int, int, int, int]) -> None:
    """Default W is Dirichlet with T=1 (constant over time)."""
    N, T, p, k = dims
    L, q = 3, 2
    data = (
        Simulation(N, T, p)
        .fixed_effects(k=k)
        .random_effects(levels=L, q=q)
        .gaussian()
        .draw(seed=0)
    )
    re = data.random_effect[0]
    assert re.W.shape == (N, 1, L)
    assert re.B.shape == (N, T, q)
    assert re.b.shape == (L, q, k)
    assert data.eta.shape == (N, T, k)


def test_default_membership_dirichlet(dims: tuple[int, int, int, int]) -> None:
    """Default W is Dirichlet: rows sum to 1 with all-positive entries."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p)
        .fixed_effects(k=k)
        .random_effects(levels=3, q=2)
        .gaussian()
        .draw(seed=0)
    )
    W = data.random_effect[0].W
    assert torch.allclose(W.sum(-1), torch.ones(N, 1))
    assert (W > 0).all()


def test_default_membership_constant_across_time(
    dims: tuple[int, int, int, int],
) -> None:
    """Default W has T=1, implying constant membership via broadcasting."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p)
        .fixed_effects(k=k)
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
        Simulation(N, T, p)
        .fixed_effects(k=k)
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
        Simulation(N, T, p)
        .fixed_effects(k=k)
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
        Simulation(N, T, p)
        .fixed_effects(k=k)
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
    N, T, _, k = dims
    L, q = 3, 2
    torch.manual_seed(42)  # type: ignore[no-untyped-call]
    indices = torch.arange(N) % L
    W_test = (
        torch.nn.functional.one_hot(indices, L).unsqueeze(1).float().expand(N, T, L)
    )
    B_test = torch.randn(N, T, q)
    b_test = torch.randn(L, q, k)
    eta_manual = torch.einsum("ntl,ntr,lrk->ntk", W_test, B_test, b_test)
    coordinates = torch.arange(T, dtype=torch.float).unsqueeze(-1).expand(N, T, 1)
    base_data = PredictorData(
        X=torch.empty(0), coordinates=coordinates, eta=torch.zeros(N, T, k)
    )
    re_data = random_effects(base_data, L, q, W_test, B_test, b_test)
    assert eta_manual.equal(re_data.eta)


# --- activation ---


def test_activation_relu_clips_negatives(dims: tuple[int, int, int, int]) -> None:
    """ReLU activation zeroes negative values in eta."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p).fixed_effects(k=k).activation().gaussian().draw(seed=0)
    )
    assert (data.eta >= 0).all()


def test_activation_custom_function(dims: tuple[int, int, int, int]) -> None:
    """Custom activation applies the given function to eta."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p)
        .fixed_effects(k=k)
        .activation(torch.abs)
        .gaussian()
        .draw(seed=0)
    )
    assert (data.eta >= 0).all()


# --- projection ---


def test_projection_shape(dims: tuple[int, int, int, int]) -> None:
    """Projection changes eta's last dimension and returns indexed weight."""
    N, T, p, k = dims
    output = 8
    data = (
        Simulation(N, T, p)
        .fixed_effects(k=k)
        .projection(output)
        .gaussian()
        .draw(seed=0)
    )
    assert data.eta.shape == (N, T, output)
    assert len(data.projection_weight) == 1
    assert data.projection_weight[0].shape == (k, output)


def test_projection_with_draws(dims: tuple[int, int, int, int]) -> None:
    """Draws dimension propagates through projection."""
    N, T, p, k = dims
    D, output = 7, 8
    data = (
        Simulation(N, T, p)
        .fixed_effects(k=k)
        .projection(output)
        .gaussian()
        .draw(seed=0, draws=D)
    )
    assert data.eta.shape == (D, N, T, output)
    assert data.projection_weight[0].shape == (D, k, output)


def test_multiple_projections(dims: tuple[int, int, int, int]) -> None:
    """Two projections produce independently indexed weight params."""
    N, T, p, k = dims
    mid, output = 8, 4
    data = (
        Simulation(N, T, p)
        .fixed_effects(k=k)
        .projection(mid)
        .projection(output)
        .gaussian()
        .draw(seed=0)
    )
    assert data.projection_weight[0].shape == (k, mid)
    assert data.projection_weight[1].shape == (mid, output)


# --- MLP pipeline ---


def test_mlp_pipeline(dims: tuple[int, int, int, int]) -> None:
    """Activation and projection compose into an MLP-like pipeline."""
    N, T, p, _ = dims
    hidden, output = 16, 3
    data = (
        Simulation(N, T, p)
        .fixed_effects(k=hidden)
        .activation()
        .projection(output)
        .activation()
        .gaussian()
        .draw(seed=0)
    )
    assert data.eta.shape == (N, T, output)
    assert data.y.shape == (N, T, output)
    assert (data.eta >= 0).all()
    assert len(data.projection_weight) == 1


# --- constant target ---


def test_constant_target_gaussian(dims: tuple[int, int, int, int]) -> None:
    """constant_target pools eta so y is identical at every timepoint."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p).fixed_effects(k=k).constant_target().gaussian().draw(seed=0)
    )
    assert data.y.shape == (N, T, k)
    assert data.X.shape == (N, T, p)
    assert data.eta.shape == (N, T, k)
    for i in range(1, T):
        assert data.y[:, 0, :].equal(data.y[:, i, :])


def test_constant_target_poisson(dims: tuple[int, int, int, int]) -> None:
    """constant_target works with poisson family."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p).fixed_effects(k=k).constant_target().poisson().draw(seed=0)
    )
    for i in range(1, T):
        assert data.y[:, 0, :].equal(data.y[:, i, :])


def test_constant_target_bernoulli(dims: tuple[int, int, int, int]) -> None:
    """constant_target works with bernoulli family."""
    N, T, p, k = dims
    data = (
        Simulation(N, T, p)
        .fixed_effects(k=k)
        .constant_target()
        .bernoulli()
        .draw(seed=0)
    )
    for i in range(1, T):
        assert data.y[:, 0, :].equal(data.y[:, i, :])


def test_constant_target_with_draws(dims: tuple[int, int, int, int]) -> None:
    """Draws dimension works with constant_target."""
    N, T, p, k = dims
    D = 7
    data = (
        Simulation(N, T, p)
        .fixed_effects(k=k)
        .constant_target()
        .gaussian()
        .draw(seed=0, draws=D)
    )
    assert data.y.shape == (D, N, T, k)
    for i in range(1, T):
        assert data.y[:, :, 0, :].equal(data.y[:, :, i, :])


def test_normal_y_varies_along_time(dims: tuple[int, int, int, int]) -> None:
    """Without constant_target, y varies along T."""
    N, T, p, k = dims
    data = Simulation(N, T, p).fixed_effects(k=k).gaussian().draw(seed=0)
    assert not data.y[:, 0, :].equal(data.y[:, 1, :])
