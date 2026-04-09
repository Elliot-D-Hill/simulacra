import torch
import torch.distributions as dist

from simulacra import PredictorData, Simulation
from simulacra.transforms import random_effects


# --- missing data ---


def test_missing_x(dims: tuple[int, int, int, int]) -> None:
    """missing_x injects NaNs into X."""
    N, T, p, k = dims
    data, _ = (
        Simulation(N, T, p, k).fixed_effects().missing_x(0.3).gaussian().draw(seed=1)
    )
    assert data["X"].isnan().any()


def test_missing_y(dims: tuple[int, int, int, int]) -> None:
    """missing_y injects NaNs into y."""
    N, T, p, k = dims
    data, _ = (
        Simulation(N, T, p, k).fixed_effects().poisson().missing_y(0.3).draw(seed=1)
    )
    assert data["y"].shape == (N, T, k)
    assert data["y"].isnan().any()


# --- event / censor time ---


def test_event_time(dims: tuple[int, int, int, int]) -> None:
    """Weibull response produces positive event times with correct shape."""
    N, T, p, k = dims
    data, _ = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .weibull(shape=1.0)
        .draw(seed=1)
    )
    assert data["y"].shape == (N, T, k)
    assert (data["y"] > 0).all()


def test_log_logistic_event_time(dims: tuple[int, int, int, int]) -> None:
    """Log-logistic produces positive event times with correct shape."""
    N, T, p, k = dims
    data, _ = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .log_logistic(shape=2.0)
        .draw(seed=1)
    )
    assert data["y"].shape == (N, T, k)
    assert (data["y"] > 0).all()


def test_gompertz_event_time(dims: tuple[int, int, int, int]) -> None:
    """Gompertz produces positive event times with correct shape."""
    N, T, p, k = dims
    data, _ = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .gompertz(shape=0.5)
        .draw(seed=1)
    )
    assert data["y"].shape == (N, T, k)
    assert (data["y"] > 0).all()


def test_exponential_equals_weibull_shape_one(
    dims: tuple[int, int, int, int],
) -> None:
    """Exponential is Weibull with shape=1."""
    N, T, p, k = dims
    d1, _ = Simulation(N, T, p, k).fixed_effects().exponential().draw(seed=42)
    d2, _ = Simulation(N, T, p, k).fixed_effects().weibull(shape=1.0).draw(seed=42)
    assert d1["y"].equal(d2["y"])


def test_gamma_survival_pipeline(dims: tuple[int, int, int, int]) -> None:
    """Gamma response works through the full survival pipeline."""
    N, T, p, k = dims
    data, _ = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .gamma(concentration=2.0)
        .censor(horizon=2.0)
        .draw(seed=1)
    )
    assert "indicator" in data
    assert "time_to_event" in data


def test_censor_time(dims: tuple[int, int, int, int]) -> None:
    """censor bounds observed times relative to coordinates plus horizon."""
    N, T, p, k = dims
    horizon = 2.0
    data, _ = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .weibull()
        .censor(horizon=horizon)
        .draw(seed=1)
    )
    bound = data["coordinates"][..., :1] + horizon
    assert (data["censor_time"] <= bound).all()


# --- random effects ---


def test_default_membership_shapes(dims: tuple[int, int, int, int]) -> None:
    """Default W is Dirichlet with T=1 (constant over time)."""
    N, T, p, k = dims
    L, q = 3, 2
    data, params = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .random_effects(levels=L, q=q)
        .gaussian()
        .draw(seed=0)
    )
    assert params["W_0"].shape == (N, 1, L)
    assert params["B_0"].shape == (N, T, q)
    assert params["b_0"].shape == (L, q, k)
    assert data["eta"].shape == (N, T, k)


def test_default_membership_dirichlet(dims: tuple[int, int, int, int]) -> None:
    """Default W is Dirichlet: rows sum to 1 with all-positive entries."""
    N, T, p, k = dims
    _, params = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .random_effects(levels=3, q=2)
        .gaussian()
        .draw(seed=0)
    )
    assert torch.allclose(params["W_0"].sum(-1), torch.ones(N, 1))
    assert (params["W_0"] > 0).all()


def test_default_membership_constant_across_time(
    dims: tuple[int, int, int, int],
) -> None:
    """Default W has T=1, implying constant membership via broadcasting."""
    N, T, p, k = dims
    _, params = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .random_effects(levels=3, q=2)
        .gaussian()
        .draw(seed=0)
    )
    assert params["W_0"].shape[1] == 1


def test_random_effects_with_draws(dims: tuple[int, int, int, int]) -> None:
    """Draws dimension propagates through random effects."""
    N, T, p, k = dims
    D, L, q = 7, 3, 2
    data, params = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .random_effects(levels=L, q=q)
        .gaussian()
        .draw(seed=0, draws=D)
    )
    assert params["W_0"].shape == (D, N, 1, L)
    assert params["B_0"].shape == (D, N, T, q)
    assert params["b_0"].shape == (D, L, q, k)
    assert data["eta"].shape == (D, N, T, k)


def test_explicit_dirichlet_membership(dims: tuple[int, int, int, int]) -> None:
    """Explicit Dirichlet prior yields positive soft membership that sums to 1."""
    N, T, p, k = dims
    L = 3
    _, params = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .random_effects(levels=L, q=2, W=dist.Dirichlet(torch.ones(L)))
        .gaussian()
        .draw(seed=0)
    )
    assert params["W_0"].shape == (N, 1, L)
    assert torch.allclose(params["W_0"].sum(-1), torch.ones(N, 1))
    assert (params["W_0"] > 0).all()


def test_multiple_random_effects(dims: tuple[int, int, int, int]) -> None:
    """Two RE terms produce independently indexed params that accumulate in eta."""
    N, T, p, k = dims
    L, q = 3, 2
    L2, q2 = 2, 3
    data, params = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .random_effects(levels=L, q=q)
        .random_effects(levels=L2, q=q2)
        .gaussian()
        .draw(seed=0)
    )
    assert params["W_0"].shape == (N, 1, L)
    assert params["B_0"].shape == (N, T, q)
    assert params["b_0"].shape == (L, q, k)
    assert params["W_1"].shape == (N, 1, L2)
    assert params["B_1"].shape == (N, T, q2)
    assert params["b_1"].shape == (L2, q2, k)
    assert data["eta"].shape == (N, T, k)


def test_einsum_equivalence(dims: tuple[int, int, int, int]) -> None:
    """Manual einsum matches the random_effects transform."""
    N, T, _, k = dims
    L, q = 3, 2
    torch.manual_seed(42)  # type: ignore[no-untyped-call]
    indices = torch.arange(N) % L
    W_test = torch.nn.functional.one_hot(indices, L).unsqueeze(1).float().expand(N, T, L)
    B_test = torch.randn(N, T, q)
    b_test = torch.randn(L, q, k)
    eta_manual = torch.einsum("ntl,ntr,lrk->ntk", W_test, B_test, b_test)
    coordinates = torch.arange(T, dtype=torch.float).unsqueeze(-1).expand(N, T, 1)
    base_data = PredictorData(coordinates=coordinates, X=torch.empty(0), eta=torch.zeros(N, T, k))
    re_data, _ = random_effects(base_data, L, q, W_test, B_test, b_test, 0)
    assert eta_manual.equal(re_data.eta)


# --- constant target ---


def test_constant_target_gaussian(dims: tuple[int, int, int, int]) -> None:
    """constant_target pools eta so y is identical at every timepoint."""
    N, T, p, k = dims
    data, _ = (
        Simulation(N, T, p, k).fixed_effects().constant_target().gaussian().draw(seed=0)
    )
    assert data["y"].shape == (N, T, k)
    assert data["X"].shape == (N, T, p)
    assert data["eta"].shape == (N, T, k)
    for i in range(1, T):
        assert data["y"][:, 0, :].equal(data["y"][:, i, :])


def test_constant_target_poisson(dims: tuple[int, int, int, int]) -> None:
    """constant_target works with poisson family."""
    N, T, p, k = dims
    data, _ = (
        Simulation(N, T, p, k).fixed_effects().constant_target().poisson().draw(seed=0)
    )
    for i in range(1, T):
        assert data["y"][:, 0, :].equal(data["y"][:, i, :])


def test_constant_target_bernoulli(dims: tuple[int, int, int, int]) -> None:
    """constant_target works with bernoulli family."""
    N, T, p, k = dims
    data, _ = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .constant_target()
        .bernoulli()
        .draw(seed=0)
    )
    for i in range(1, T):
        assert data["y"][:, 0, :].equal(data["y"][:, i, :])


def test_constant_target_with_draws(dims: tuple[int, int, int, int]) -> None:
    """Draws dimension works with constant_target."""
    N, T, p, k = dims
    D = 7
    data, _ = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .constant_target()
        .gaussian()
        .draw(seed=0, draws=D)
    )
    assert data["y"].shape == (D, N, T, k)
    for i in range(1, T):
        assert data["y"][:, :, 0, :].equal(data["y"][:, :, i, :])


def test_normal_y_varies_along_time(dims: tuple[int, int, int, int]) -> None:
    """Without constant_target, y varies along T."""
    N, T, p, k = dims
    data, _ = Simulation(N, T, p, k).fixed_effects().gaussian().draw(seed=0)
    assert not data["y"][:, 0, :].equal(data["y"][:, 1, :])
