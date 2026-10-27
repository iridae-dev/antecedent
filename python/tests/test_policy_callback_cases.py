"""Batch callbacks and portable policy artifacts on the simulated pricing chain.

Each test is one acceptance case of the callback and portable-artifact areas, run on the stated
population (``_pricing_chain.pricing_chain``): a lognormal price with ``demand = 50 - 0.8 price +
noise`` and a noisy revenue. Callbacks are transforms of ``price`` that read its generated value;
the refusals all happen before a draw is returned.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import antecedent
import numpy as np
import pytest
from antecedent import Execution
from antecedent.errors import CausalError, CausalUnsupportedError
from antecedent.intervention_callbacks import (
    GENERATED,
    CallbackProgram,
    InterventionCallback,
    decode_programs,
    recover_callback,
    required_callbacks,
    sample_do,
)
from antecedent.intervention_programs import (
    decode_policy_program,
    encode_policy_program,
    generated,
    maximum,
    policy_program_identity,
    sample_policy,
    transform,
)

from _pricing_chain import fit_pricing_chain

NAMES = ["price", "demand", "revenue"]
TESTS = str(Path(__file__).resolve().parent)

SOURCE = """
def discounted(price):
    return np.clip(0.85 * price, 10.0, None)
"""


def _discount(function, **declaration):
    callback = InterventionCallback(function, inputs=["price"], name="discounted", **declaration)
    return CallbackProgram("price", callback, reads=[GENERATED], mode="transform")


def _refusal(code, program, fitted, rows):
    result = None
    with pytest.raises(CausalUnsupportedError) as raised:
        result = sample_do(fitted, [program], rows, execution=Execution(seed=3))
    assert result is None, "no partial result is returned"
    assert raised.value.reason_code == code, raised.value
    return str(raised.value)


def _bits(draws):
    return np.ascontiguousarray(draws).view(np.uint64)


def test_a_numpy_callback_is_the_same_policy_as_its_expression_and_runs_once_per_batch():
    rows = 100_000
    fitted = fit_pricing_chain(rows, 1885751)
    calls = []

    def clipped(price):
        calls.append((price.dtype, price.shape))
        return np.clip(0.85 * price, 10.0, None)

    by_callback = sample_do(
        fitted, [_discount(clipped, lower=10.0, upper=1.0e6)], rows, execution=Execution(seed=11)
    )
    by_expression = sample_policy(
        fitted,
        [transform("price", maximum(10, 0.85 * generated()))],
        rows,
        execution=antecedent.Execution(seed=11),
    )
    np.testing.assert_array_equal(_bits(by_callback.draws), _bits(by_expression.draws))
    assert calls == [(np.dtype("float64"), (rows,))], "once per batch, not once per row"
    assert by_callback.draws[NAMES.index("price")].min() >= 10.0


def test_a_callback_returning_the_wrong_number_of_rows_names_both_lengths():
    fitted = fit_pricing_chain(1000, 7850999)
    message = _refusal(
        "policy_callback_output_invalid", _discount(lambda price: price[:-1]), fitted, 1000
    )
    assert "999" in message and "1000" in message, message


def test_a_callback_returning_nan_reports_the_first_offending_row():
    fitted = fit_pricing_chain(1000, 4007062)
    message = _refusal(
        "policy_callback_output_invalid",
        _discount(lambda price: np.full_like(price, np.nan)),
        fitted,
        1000,
    )
    assert "row 0" in message and "non-finite" in message, message


def test_a_callback_returning_strings_names_the_declared_and_observed_dtype():
    fitted = fit_pricing_chain(1000, 2001332)
    message = _refusal(
        "policy_callback_output_invalid",
        _discount(lambda price: np.array(["expensive"] * len(price))),
        fitted,
        1000,
    )
    assert "float64" in message and "<U9" in message, message


def test_an_exception_in_the_callback_carries_its_name_and_message_and_returns_nothing():
    fitted = fit_pricing_chain(1000, 3022533)

    def raising(price):
        raise ValueError("no demand model for region 7")

    message = _refusal("policy_callback_failed", _discount(raising), fitted, 1000)
    assert "discounted" in message and "no demand model for region 7" in message, message
    assert "ValueError" in message


def test_a_callback_reading_a_column_it_did_not_declare_is_refused_naming_the_column():
    fitted = fit_pricing_chain(1000, 1500505)
    callback = InterventionCallback(lambda price: price, inputs=["price"], name="reads_demand")
    program = CallbackProgram("revenue", callback, reads=["demand"])
    message = _refusal("policy_dependency_unavailable", program, fitted, 1000)
    assert "demand" in message and "reads_demand" in message, message


def test_a_missing_runtime_dependency_is_refused_naming_it_and_the_callback():
    with pytest.raises(CausalUnsupportedError) as raised:
        InterventionCallback(
            lambda price: price,
            inputs=["price"],
            name="needs_a_package",
            dependencies={"antecedent-no-such-distribution": "1.0.0"},
        )
    assert raised.value.reason_code == "policy_callback_unrecoverable"
    message = str(raised.value)
    assert "antecedent-no-such-distribution" in message and "needs_a_package" in message, message


def test_an_interactive_callback_is_refused_for_export_but_stays_usable_in_session():
    fitted = fit_pricing_chain(1000, 3873110)
    program = _discount(lambda price: np.maximum(0.85 * price, 10.0))
    with pytest.raises(CausalUnsupportedError) as raised:
        encode_policy_program(NAMES, [program])
    assert raised.value.reason_code == "policy_callback_not_portable"
    message = str(raised.value)
    assert "discounted" in message and "no verifiable recoverable implementation" in message
    result = sample_do(fitted, [program], 1000, execution=Execution(seed=3))
    assert result.n_draws == 1000 and result.draws[NAMES.index("price")].min() >= 10.0


def test_a_callback_with_an_unpinned_dependency_is_refused_for_export_naming_it():
    callback = InterventionCallback.from_source(
        SOURCE,
        entry_point="discounted",
        inputs=["price"],
        dependencies={"packaging": None},
    )
    assert callback.recoverable
    program = CallbackProgram("price", callback, reads=[GENERATED], mode="transform")
    with pytest.raises(CausalUnsupportedError) as raised:
        encode_policy_program(NAMES, [program])
    assert raised.value.reason_code == "policy_callback_not_portable"
    assert "packaging" in str(raised.value) and "pinned" in str(raised.value)
    with pytest.raises(CausalUnsupportedError):
        callback.recovery_record()


_CALLBACK_REPLAY = """
import hashlib, sys
sys.path.insert(0, sys.argv[1])
from _pricing_chain import fit_pricing_chain
from antecedent import Execution
from antecedent.intervention_callbacks import decode_programs, recover_callback, required_callbacks, sample_do

artifact = open(sys.argv[2], "rb").read()
source = open(sys.argv[3]).read()
fitted = fit_pricing_chain(int(sys.argv[4]), int(sys.argv[5]))
callbacks = [recover_callback(source, record) for record in required_callbacks(artifact)]
programs = decode_programs(["price", "demand", "revenue"], artifact, callbacks)
draws = sample_do(fitted, list(programs), int(sys.argv[4]), execution=Execution(seed=7)).draws
print(hashlib.sha256(draws.tobytes()).hexdigest())
"""


def _run(script, *args):
    done = subprocess.run(
        [sys.executable, "-c", script, TESTS, *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_a_portable_callback_artifact_replays_bit_for_bit_in_a_fresh_process(tmp_path):
    rows, seed = 10_000, 6736828
    fitted = fit_pricing_chain(rows, seed)
    callback = InterventionCallback.from_source(
        SOURCE, entry_point="discounted", inputs=["price"], lower=10.0, upper=1.0e6
    )
    program = CallbackProgram("price", callback, reads=[GENERATED], mode="transform")
    artifact = encode_policy_program(NAMES, [program])

    # The artifact records the implementation hash and the dependency versions, never the code.
    (record,) = required_callbacks(artifact)
    digest = hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()
    assert record.source_sha256 == digest and digest.encode() in artifact
    assert dict(record.dependencies)["numpy"] == np.__version__
    assert b"np.clip" not in artifact

    path, source = tmp_path / "policy.bin", tmp_path / "callback.py"
    path.write_bytes(artifact)
    source.write_text(SOURCE)
    local = hashlib.sha256(
        sample_do(fitted, [program], rows, execution=Execution(seed=7)).draws.tobytes()
    ).hexdigest()
    assert _run(_CALLBACK_REPLAY, path, source, rows, seed) == local

    # A changed implementation refuses replay, whichever way it arrives.
    with pytest.raises(CausalUnsupportedError) as changed:
        recover_callback(SOURCE.replace("0.85", "0.9"), record)
    assert changed.value.reason_code == "policy_callback_unrecoverable"
    assert "source_sha256" in str(changed.value)
    other = InterventionCallback.from_source(
        SOURCE.replace("0.85", "0.9"),
        entry_point="discounted",
        inputs=["price"],
        lower=10.0,
        upper=1.0e6,
    )
    with pytest.raises(CausalUnsupportedError) as swapped:
        decode_programs(NAMES, artifact, [other])
    assert swapped.value.reason_code == "policy_callback_unrecoverable"
    assert "source_sha256" in str(swapped.value)
    with pytest.raises(CausalUnsupportedError) as missing:
        decode_programs(NAMES, artifact, [])
    assert missing.value.reason_code == "policy_callback_unrecoverable"
    with pytest.raises(CausalUnsupportedError) as unsupplied:
        decode_policy_program(NAMES, artifact)
    assert unsupplied.value.reason_code == "policy_callback_unrecoverable"
    forged = artifact.replace(digest.encode(), ("0" * 64).encode())
    with pytest.raises(CausalError):
        decode_programs(NAMES, forged, [callback])


_CLOSED_REPLAY = """
import hashlib, sys
sys.path.insert(0, sys.argv[1])
from _pricing_chain import fit_pricing_chain
import antecedent
from antecedent.intervention_programs import decode_policy_program, policy_program_identity, sample_policy

names = ["price", "demand", "revenue"]
programs = decode_policy_program(names, open(sys.argv[2], "rb").read())
fitted = fit_pricing_chain(int(sys.argv[3]), int(sys.argv[4]))
draws = sample_policy(fitted, programs, int(sys.argv[3]), execution=antecedent.Execution(seed=5)).draws
print(policy_program_identity(names, programs), hashlib.sha256(draws.tobytes()).hexdigest())
"""


def test_a_saved_policy_replays_identically_in_three_fresh_processes(tmp_path):
    rows, seed = 10_000, 3999629
    fitted = fit_pricing_chain(rows, seed)
    programs = [transform("price", maximum(10, 0.85 * generated()))]
    saved = encode_policy_program(NAMES, programs)
    identity = policy_program_identity(NAMES, programs)
    expected = hashlib.sha256(
        sample_policy(
            fitted, programs, rows, execution=antecedent.Execution(seed=5)
        ).draws.tobytes()
    ).hexdigest()
    path = tmp_path / "policy.bin"
    path.write_bytes(saved)

    replays = [_run(_CLOSED_REPLAY, path, rows, seed) for _ in range(3)]
    assert len(set(replays)) == 1, replays
    replayed_identity, replayed_draws = replays[0].split()
    assert replayed_identity == identity, "the replay identity is the one recorded at save"
    assert replayed_draws == expected
    assert decode_policy_program(NAMES, saved) == tuple(programs)
