"""The promise that makes committed tapes worth having: replay needs nothing.

No keys, no network, no accounts. If any of that stops being true these tests fail, and they fail
here rather than on a contributor's first clone.
"""

from __future__ import annotations

import os

import pytest

from .conftest import _PLACEHOLDER, _PROVIDER_KEYS

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("name", _PROVIDER_KEYS)
def test_a_blank_credential_counts_as_absent(name: str) -> None:
    """CI sets provider keys to the EMPTY STRING deliberately, to prove replay needs none.

    An empty value is present-but-useless: `os.environ.setdefault` leaves it in place, the client
    constructor then raises "Missing credentials", and the whole replayed suite fails because of the
    guard against live calls rather than anything to do with the tapes. Blank must count as absent.
    """
    value = os.environ.get(name, "")
    assert value.strip(), f"{name} is blank — the placeholder did not apply"


def test_the_placeholder_is_never_mistaken_for_a_real_key() -> None:
    """`requires()` gates RECORDING on real credentials. If the placeholder ever read as real, a
    recording run would proceed with a fake key and fail somewhere far from the cause."""
    from .conftest import requires

    real = {n for n in _PROVIDER_KEYS if os.environ.get(n, _PLACEHOLDER) != _PLACEHOLDER}
    if real:
        pytest.skip(f"real credentials present ({len(real)}); nothing to prove about placeholders")
    with pytest.raises(BaseException, match="recording needs"):
        requires(*_PROVIDER_KEYS)
