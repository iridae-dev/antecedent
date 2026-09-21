"""Ctrl-C reaches native runs, and Python callback exceptions keep their identity."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent
from antecedent import _native


def _series(n: int = 120, k: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    names = [f"v{i}" for i in range(k)]
    return names, [rng.normal(size=n) for _ in names]


def _static(n: int = 120, k: int = 4, seed: int = 0):
    return _series(n, k, seed)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT delivery via os.kill is POSIX")
def test_sigint_interrupts_a_running_native_discovery():
    """A slow CI callback keeps the native run busy; Ctrl-C must end it promptly.

    Before the interrupt watcher, the GIL-released run never returned to CPython's signal
    machinery, so the KeyboardInterrupt only surfaced after the whole discovery finished.
    """
    names, columns = _static()

    def slow_independent(_columns, queries):
        time.sleep(0.05 * max(1, len(queries)))
        return [(0.0, 1.0) for _ in queries]

    timer = threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT))
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            _native.discover_pcmci(
                names, columns, max_lag=3, alpha=0.05, fdr=False, seed=1, ci=slow_independent
            )
    finally:
        timer.cancel()
    # The full run needs thousands of 50 ms callbacks; a prompt exit is seconds, not minutes.
    assert time.monotonic() - started < 30.0


def test_a_cancelled_token_stops_discovery_with_the_cancelled_error():
    names, columns = _series()
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(antecedent.errors.CausalCancelledError):
        _native.discover_pcmci(names, columns, max_lag=2, seed=1, cancel=token)


def test_callback_exception_is_chained_not_stringified():
    names, columns = _static()

    def boom(_columns, _queries):
        raise ValueError("ci exploded")

    with pytest.raises(_native.CausalError) as raised:
        _native.discover_pc(names, columns, ci=boom, fdr=False, seed=1)
    cause = raised.value.__cause__
    assert isinstance(cause, ValueError)
    assert str(cause) == "ci exploded"
    assert cause.__traceback__ is not None


def test_keyboard_interrupt_in_a_callback_is_not_swallowed_by_except_causalerror():
    names, columns = _static()

    def interrupted(_columns, _queries):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        try:
            _native.discover_pc(names, columns, ci=interrupted, fdr=False, seed=1)
        except _native.CausalError:  # a retry loop like this used to eat the interrupt
            pytest.fail("KeyboardInterrupt was converted into a CausalError")


class _Mechanism:
    def __init__(self, extra: int = 0):
        self.extra = extra

    def sample_noise(self, n):
        return np.zeros(n + self.extra)

    def evaluate(self, parents, noise):
        return np.zeros(len(noise))


def _do(mechanism, seed=1):
    rng = np.random.default_rng(3)
    z = rng.normal(size=60)
    t = z + rng.normal(size=60)
    y = t + rng.normal(size=60)
    return _native.sample_do(
        ["z", "t", "y"],
        [z, t, y],
        [("z", "t"), ("t", "y")],
        "t",
        1.0,
        20,
        seed=seed,
        mechanism_wrappers={"y": mechanism},
    )


def test_mechanism_output_longer_than_requested_is_a_shape_error():
    assert _do(_Mechanism()).n_draws == 20
    with pytest.raises(_native.CausalError, match="expected exactly 20"):
        _do(_Mechanism(extra=3))


class _Seeded:
    def sample_noise(self, n, rng):
        return rng.normal(size=n)

    def evaluate(self, parents, noise):
        return np.asarray(noise, dtype=np.float64)


def test_seed_governs_a_mechanism_that_accepts_an_rng():
    a = np.asarray(_do(_Seeded(), seed=7).draws)
    b = np.asarray(_do(_Seeded(), seed=7).draws)
    c = np.asarray(_do(_Seeded(), seed=8).draws)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
