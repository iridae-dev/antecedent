"""Batch noise replay is reproducible: across calls, encodings of the noise and fresh processes.

``sample_noise`` returns the exogenous noise behind a set of draws. ``replay_noise`` evaluates
the fitted model on retained noise, so replaying under the same policy reproduces the draws bit
for bit and replaying under another policy gives that policy's counterfactual on the same units.
"""

from __future__ import annotations

import io
import subprocess
import sys
import textwrap

import antecedent
import numpy as np
import pytest

NAMES = ["price", "demand"]
EDGES = [("price", "demand")]
N_ROWS = 4000

# A fresh interpreter refits the saved columns and replays the saved noise.
FRESH_PROCESS = textwrap.dedent(
    """
    import sys

    import antecedent
    import numpy as np

    folder = sys.argv[1]
    data = np.load(folder + "/data.npz")
    model = antecedent.model.fit_gcm(
        ["price", "demand"], [data["price"], data["demand"]], [("price", "demand")]
    )
    noise = np.load(folder + "/noise.npy")
    expected = np.load(folder + "/draws.npy")
    replayed = np.array([np.asarray(c) for c in model.replay_noise(noise).draws])
    sys.exit(0 if np.array_equal(replayed, expected) else 1)
    """
)


def _columns() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2929415)
    price = 11.0 + rng.normal(scale=2.0, size=N_ROWS)
    demand = 50.0 - 0.8 * price + rng.normal(scale=1.0, size=N_ROWS)
    return price, demand


def _matrix(result) -> np.ndarray:
    return np.array([np.asarray(column) for column in result.draws])


@pytest.fixture(scope="module")
def fitted():
    price, demand = _columns()
    return antecedent.model.fit_gcm(NAMES, [price, demand], EDGES), price, demand


def test_replaying_the_sampled_noise_reproduces_the_draws_bit_for_bit(fitted):
    model, _, _ = fitted
    for interventions in ({}, {"price": 10.0}):
        draws, noise = model.sample_noise(interventions, N_ROWS, seed=11)
        assert noise.shape == (len(NAMES), N_ROWS)
        replayed = model.replay_noise(noise, interventions)
        np.testing.assert_array_equal(_matrix(replayed), _matrix(draws))


def test_replay_does_not_depend_on_how_the_noise_array_is_encoded(fitted):
    model, _, _ = fitted
    draws, noise = model.sample_noise({}, N_ROWS, seed=12)
    expected = _matrix(draws)

    buffer = io.BytesIO()
    np.save(buffer, noise)
    buffer.seek(0)
    variants = {
        "npy round trip": np.load(buffer),
        "fortran order": np.asfortranarray(noise),
        "raw bytes": np.frombuffer(noise.tobytes(), dtype=np.float64).reshape(noise.shape),
        "list of rows": np.array([list(row) for row in noise]),
    }
    for label, encoded in variants.items():
        np.testing.assert_array_equal(_matrix(model.replay_noise(encoded)), expected, err_msg=label)


def test_replaying_the_noise_under_another_policy_is_that_policys_counterfactual(fitted):
    model, _, _ = fitted
    factual, noise = model.sample_noise({}, N_ROWS, seed=13)
    counterfactual = _matrix(model.replay_noise(noise, {"price": 10.0}))
    price, demand = _matrix(factual)
    assert (counterfactual[0] == 10.0).all()
    # The demand mechanism is affine in price on the same noise, so the change in demand is
    # proportional to the change in price with one slope for every unit.
    slopes = (counterfactual[1] - demand) / (10.0 - price)
    assert np.ptp(slopes) < 1e-6
    assert slopes[0] == pytest.approx(-0.8, abs=0.05)


def test_replay_is_bit_reproducible_in_a_fresh_process(fitted, tmp_path):
    model, price, demand = fitted
    draws, noise = model.sample_noise({}, N_ROWS, seed=14)
    np.savez(tmp_path / "data.npz", price=price, demand=demand)
    np.save(tmp_path / "noise.npy", noise)
    np.save(tmp_path / "draws.npy", _matrix(draws))
    completed = subprocess.run(
        [sys.executable, "-I", "-c", FRESH_PROCESS, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
