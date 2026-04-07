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
    """event_time produces positive values with correct shape."""
    N, T, p, k = dims
    data, _ = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .gaussian()
        .event_time(shape=1.0)
        .draw(seed=1)
    )
    assert data["event_time"].shape == (N, T, k)
    assert (data["event_time"] > 0).all()


def test_censor_time(dims: tuple[int, int, int, int]) -> None:
    """censor_time clamps event times at the horizon."""
    N, T, p, k = dims
    data, _ = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .gaussian()
        .event_time()
        .censor_time(horizon=2.0)
        .draw(seed=1)
    )
    assert (data["censor_time"] <= 2.0).all()


# --- random effects ---


def test_default_membership_shapes(dims: tuple[int, int, int, int]) -> None:
    """Default W is one-hot with correct indexed param shapes."""
    N, T, p, k = dims
    L, q = 3, 2
    data, params = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .random_effects(levels=L, q=q)
        .gaussian()
        .draw(seed=0)
    )
    assert params["W_0"].shape == (N, T, L)
    assert params["B_0"].shape == (N, T, q)
    assert params["b_0"].shape == (L, q, k)
    assert data["eta"].shape == (N, T, k)


def test_default_membership_one_hot(dims: tuple[int, int, int, int]) -> None:
    """Default W is one-hot: rows sum to 1 with a single 1 per row."""
    N, T, p, k = dims
    _, params = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .random_effects(levels=3, q=2)
        .gaussian()
        .draw(seed=0)
    )
    assert (params["W_0"].sum(-1) == 1.0).all()
    assert (params["W_0"].max(-1).values == 1.0).all()


def test_default_membership_constant_across_time(
    dims: tuple[int, int, int, int],
) -> None:
    """Same subject has the same group assignment at every timepoint."""
    N, T, p, k = dims
    _, params = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .random_effects(levels=3, q=2)
        .gaussian()
        .draw(seed=0)
    )
    assert (params["W_0"][:, 0, :] == params["W_0"][:, 1, :]).all()


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
    assert params["W_0"].shape == (D, N, T, L)
    assert params["B_0"].shape == (D, N, T, q)
    assert params["b_0"].shape == (D, L, q, k)
    assert data["eta"].shape == (D, N, T, k)


def test_soft_membership_dirichlet(dims: tuple[int, int, int, int]) -> None:
    """Dirichlet prior yields positive soft membership that sums to 1."""
    N, T, p, k = dims
    L = 3
    _, params = (
        Simulation(N, T, p, k)
        .fixed_effects()
        .random_effects(levels=L, q=2, W=dist.Dirichlet(torch.ones(L)))
        .gaussian()
        .draw(seed=0)
    )
    assert params["W_0"].shape == (N, T, L)
    assert torch.allclose(params["W_0"].sum(-1), torch.ones(N, T))
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
    assert params["W_0"].shape == (N, T, L)
    assert params["B_0"].shape == (N, T, q)
    assert params["b_0"].shape == (L, q, k)
    assert params["W_1"].shape == (N, T, L2)
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
    base_data = PredictorData(X=torch.empty(0), eta=torch.zeros(N, T, k))
    re_data, _ = random_effects(base_data, 0, L, q, W_test, B_test, b_test)
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
