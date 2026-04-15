import pytest
import torch

from simulacra import simulate


def test_binary_treatment_shape(dims: tuple[int, int, int, int]) -> None:
    """Default gamma=None produces J=2 binary shapes."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .treatment(tau=torch.zeros(2, k))
        .draw(seed=0)
    )
    assert data.treatment is not None
    assert data.propensity is not None
    assert data.gamma is not None
    assert data.treatment.shape == (N, T, 1)
    assert data.propensity.shape == (N, T, 2)
    assert data.gamma.shape == (p, 2)


def test_treatment_values_in_arm_range(dims: tuple[int, int, int, int]) -> None:
    """For J arms, treatment indices lie in {0, ..., J-1}."""
    N, T, p, k = dims
    J = 4
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .treatment(tau=torch.zeros(J, k), gamma=torch.randn(p, J))
        .draw(seed=0)
    )
    assert data.treatment is not None
    assert data.treatment.min().item() >= 0
    assert data.treatment.max().item() < J


def test_propensity_rows_sum_to_one(dims: tuple[int, int, int, int]) -> None:
    """Softmax propensity rows sum to 1 across arms."""
    N, T, p, k = dims
    J = 3
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .treatment(tau=torch.zeros(J, k), gamma=torch.randn(p, J))
        .draw(seed=0)
    )
    assert data.propensity is not None
    assert torch.allclose(data.propensity.sum(dim=-1), torch.ones(N, T))


def test_propensity_in_unit_interval(dims: tuple[int, int, int, int]) -> None:
    """Softmax propensities lie strictly in (0, 1)."""
    N, T, p, k = dims
    J = 3
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .treatment(tau=torch.zeros(J, k), gamma=torch.randn(p, J))
        .draw(seed=0)
    )
    assert data.propensity is not None
    assert (data.propensity > 0).all()
    assert (data.propensity < 1).all()


def test_zero_gamma_gives_uniform_propensity(dims: tuple[int, int, int, int]) -> None:
    """gamma=zeros(p, J) produces uniform propensity 1/J over arms (softmax(0s))."""
    N, T, p, k = dims
    J = 3
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .treatment(tau=torch.zeros(J, k), gamma=torch.zeros(p, J))
        .draw(seed=0)
    )
    assert data.propensity is not None
    expected = torch.full((N, T, J), 1.0 / J)
    assert torch.allclose(data.propensity, expected)


def test_tau_zero_preserves_eta(dims: tuple[int, int, int, int]) -> None:
    """tau=zeros leaves eta at its pre-treatment value X @ beta."""
    N, T, p, k = dims
    X = torch.randn(N, T, p)
    beta = torch.randn(p, k)
    data = (
        simulate(X, beta)
        .treatment(tau=torch.zeros(2, k), gamma=torch.randn(p, 2))
        .draw(seed=0)
    )
    assert torch.allclose(data.eta, X @ beta)


def test_binary_rr_reduction(dims: tuple[int, int, int, int]) -> None:
    """J=2 with γ[:, 0]=0 and τ=[[0], [ATE]] recovers classical RR: η += Z * ATE."""
    N, T, p, k = dims
    X = torch.randn(N, T, p)
    beta = torch.randn(p, k)
    gamma = torch.cat((torch.zeros(p, 1), torch.randn(p, 1)), dim=-1)
    ate = torch.randn(k)
    tau = torch.stack((torch.zeros(k), ate), dim=0)
    data = simulate(X, beta).treatment(tau=tau, gamma=gamma).draw(seed=0)
    assert data.treatment is not None
    expected = X @ beta + data.treatment.to(data.eta.dtype) * ate
    assert torch.allclose(data.eta, expected)


def test_multi_arm_tau_shifts_eta_per_arm(dims: tuple[int, int, int, int]) -> None:
    """For J=3, η shifts by τ[j] wherever Z==j."""
    N, T, p, k = dims
    X = torch.randn(N, T, p)
    beta = torch.randn(p, k)
    gamma = torch.randn(p, 3)
    tau = torch.randn(3, k)
    data = simulate(X, beta).treatment(tau=tau, gamma=gamma).draw(seed=0)
    assert data.treatment is not None
    expected = X @ beta + tau[data.treatment.squeeze(-1)]
    assert torch.allclose(data.eta, expected)


def test_seed_reproducibility(dims: tuple[int, int, int, int]) -> None:
    """Same seed produces identical treatment indices."""
    N, T, p, k = dims
    X = torch.randn(N, T, p)
    beta = torch.randn(p, k)
    gamma = torch.randn(p, 3)
    tau = torch.zeros(3, k)
    first = simulate(X, beta).treatment(tau=tau, gamma=gamma).draw(seed=42)
    second = simulate(X, beta).treatment(tau=tau, gamma=gamma).draw(seed=42)
    assert first.treatment is not None
    assert second.treatment is not None
    assert first.treatment.equal(second.treatment)


def test_treatment_then_gaussian(dims: tuple[int, int, int, int]) -> None:
    """Full pipeline simulate -> treatment -> gaussian produces y of shape (N, T, k)."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .treatment(tau=torch.zeros(2, k))
        .gaussian()
        .draw(seed=0)
    )
    assert data.y.shape == (N, T, k)


def test_tau_required(dims: tuple[int, int, int, int]) -> None:
    """Calling .treatment() without tau raises TypeError."""
    N, T, p, k = dims
    pipeline = simulate(torch.randn(N, T, p), torch.randn(p, k))
    with pytest.raises(TypeError):
        pipeline.treatment()  # type: ignore[call-arg]


def test_dose_response_shape(dims: tuple[int, int, int, int]) -> None:
    """dose_response produces continuous treatment and propensity of shape (N, T, 1)."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .dose_response(tau=0.5)
        .draw(seed=0)
    )
    assert data.treatment is not None
    assert data.propensity is not None
    assert data.gamma is not None
    assert data.treatment.shape == (N, T, 1)
    assert data.propensity.shape == (N, T, 1)
    assert data.gamma.shape == (p, 1)
    assert data.treatment.dtype.is_floating_point


def test_dose_response_propensity_positive(dims: tuple[int, int, int, int]) -> None:
    """Gaussian density propensity is strictly positive."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .dose_response(tau=0.0)
        .draw(seed=0)
    )
    assert data.propensity is not None
    assert (data.propensity > 0).all()


def test_dose_response_tau_zero_preserves_eta(dims: tuple[int, int, int, int]) -> None:
    """tau=0 leaves eta at its pre-treatment value X @ beta."""
    N, T, p, k = dims
    X = torch.randn(N, T, p)
    beta = torch.randn(p, k)
    data = (
        simulate(X, beta).dose_response(tau=0.0, gamma=torch.randn(p, 1)).draw(seed=0)
    )
    assert torch.allclose(data.eta, X @ beta)


def test_dose_response_tau_shifts_eta_by_z(dims: tuple[int, int, int, int]) -> None:
    """eta equals eta_pre + tau * z element-wise."""
    N, T, p, k = dims
    X = torch.randn(N, T, p)
    beta = torch.randn(p, k)
    tau = torch.randn(k)
    data = (
        simulate(X, beta).dose_response(tau=tau, gamma=torch.randn(p, 1)).draw(seed=0)
    )
    assert data.treatment is not None
    assert torch.allclose(data.eta, X @ beta + tau * data.treatment)


def test_dose_response_seed_reproducibility(dims: tuple[int, int, int, int]) -> None:
    """Same seed produces identical continuous treatment draws."""
    N, T, p, k = dims
    X = torch.randn(N, T, p)
    beta = torch.randn(p, k)
    gamma = torch.randn(p, 1)
    first = simulate(X, beta).dose_response(tau=0.5, gamma=gamma).draw(seed=42)
    second = simulate(X, beta).dose_response(tau=0.5, gamma=gamma).draw(seed=42)
    assert first.treatment is not None
    assert second.treatment is not None
    assert first.treatment.equal(second.treatment)


def test_dose_response_then_gaussian(dims: tuple[int, int, int, int]) -> None:
    """Full pipeline simulate -> dose_response -> gaussian produces y of shape (N, T, k)."""
    N, T, p, k = dims
    data = (
        simulate(torch.randn(N, T, p), torch.randn(p, k))
        .dose_response(tau=0.5)
        .gaussian()
        .draw(seed=0)
    )
    assert data.y.shape == (N, T, k)
