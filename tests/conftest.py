import pytest


@pytest.fixture
def dims() -> tuple[int, int, int, int]:
    """(N, T, p, k) used across all test modules."""
    return 4, 5, 3, 2
