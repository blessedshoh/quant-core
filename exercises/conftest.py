import pytest

from exercises import black_scholes


@pytest.fixture
def impl():
    """Point the shared tests at YOUR implementation instead of the engine's."""
    return black_scholes
