import pytest

from quantcore import black_scholes


@pytest.fixture
def impl():
    """The Black-Scholes implementation under test.

    Here it is the real engine. `exercises/conftest.py` replaces it with your own
    version, so the same tests check your work.
    """
    return black_scholes
