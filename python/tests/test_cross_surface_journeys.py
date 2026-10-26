"""Workflow journeys that cross the plan, the report, the artifact and the host surface.

Each analysis family already has a suite of its own. These journeys follow one result through
every surface a user or a host reads: the prepared plan before any estimate, the readable report,
the exported artifact reloaded by an independent consumer in a fresh interpreter, the identity a
caller retains independently of the bytes, and the host record of the claim. They then refresh
the data, keep the earlier artifact, and read a summary-only form that serves no rows.

The Gaussian lines are the cross-surface fixture's: ``y = 1 + 2t`` and its one-step horizon
counterpart, so every cell's contrast is known. The attribution routes use the lifecycle suite's
own chains.
"""

from __future__ import annotations

import json
import subprocess
import sys

import antecedent as ant
import numpy as np
import pytest
from antecedent import host
from antecedent.errors import (
    CausalSerializationError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.outcomes import require_success
from antecedent.transport import advanced as transport
from antecedent.transport.advanced import consume_period_refresh_artifact

from _repo_text import REPO_ROOT, load_json
from test_attribution_lifecycle import (
    _anomaly_query,
    _change_query,
    _dag,
    _outlier_chain,
    _two_period_chain,
)
from test_temporal_extensions import SERIES, period_window, started

CASES = load_json(REPO_ROOT / "conformance" / "cross_surface" / "slice_cases.json")
GAUSSIAN = CASES["gaussian"]

_REFUSALS = (CausalSerializationError, CausalUnsupportedError, CausalValueError)
_CELLS = [(False, False), (False, True), (True, False), (True, True)]


def _cell_id(cell: tuple[bool, bool]) -> str:
    horizon, bayesian = cell
    return f"{'horizon' if horizon else 'static'}-{'bayesian' if bayesian else 'frequentist'}"


def _tables(horizon: bool) -> tuple[dict[str, np.ndarray], tuple]:
    """The fixture's line as data, with its graph and query."""
    if horizon:
        data = {
            "t": np.asarray(GAUSSIAN["horizon_treatment"], dtype=float),
            "y": np.asarray(GAUSSIAN["horizon_outcome"], dtype=float),
        }
        return data, (
            [("t", 1, "y", 0)],
            ant.PulseEffect("t", "y", treatment_lag=1, horizon_steps=1),
        )
    data = {
        "t": np.asarray(GAUSSIAN["treatment"], dtype=float),
        "y": np.asarray(GAUSSIAN["outcome"], dtype=float),
    }
    return data, ([("t", "y")], ant.AverageEffect("t", "y"))


def _inference(bayesian: bool):
    if not bayesian:
        return ant.Frequentist()
    return ant.Bayesian(
        n_draws=GAUSSIAN["posterior_draws"],
        prior_scale=GAUSSIAN["prior_scale"],
        backend="conjugate",
    )


def _prepare(horizon: bool, bayesian: bool):
    data, (graph, query) = _tables(horizon)
    prepared = ant.prepare(
        data,
        graph=graph,
        query=query,
        inference=_inference(bayesian),
        refute="none",
        bootstrap=0,
        seed=1,
    )
    return prepared, data


def _tolerance(horizon: bool, bayesian: bool) -> float:
    if bayesian:
        return 0.1
    return 1e-7 if horizon else 1e-9


def _fact(markdown: str, key: str) -> str:
    prefix = f"- **{key}:** "
    for line in markdown.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    raise AssertionError(f"no fact {key!r} in\n{markdown}")


@pytest.mark.parametrize("cell", _CELLS, ids=_cell_id)
def test_one_result_reads_the_same_through_plan_report_artifact_and_host(cell) -> None:
    horizon, bayesian = cell
    prepared, data = _prepare(horizon, bayesian)
    view = prepared.plan_view()
    assert prepared.last_result is None, "reading the plan executes nothing"
    assert (view.uncertainty.reading == "posterior") is bayesian

    before = prepared.report("markdown")
    assert "a contract before execution" in before
    assert "## Claim" not in before

    result = require_success(prepared.estimate(data, seed=1))
    assert result.estimate.ate == pytest.approx(
        GAUSSIAN["effect"], abs=_tolerance(horizon, bayesian)
    )
    assert prepared.plan_view().identity == view.identity, "estimating keeps the program"

    after = prepared.report("markdown")
    assert "## Claim" in after
    assert _fact(after, "Inference").lower() == ("bayesian" if bayesian else "frequentist")
    assert after == prepared.report("markdown"), "reading the report twice changes nothing"

    # The host reads the same claim the report prints, and a causal effect is a causal record.
    inspection = result.inspect()
    claim = inspection.contract["claim"]
    record = host.claim_record(claim)
    assert record["causal"] is True
    assert record["status"] == host.claim_status(claim) == host.inspection_status(inspection)
    assert record["status"] in {"supported", "partial"}

    # The independent consumer keeps the answer, and the retained identity binds it.
    artifact = bytes(result.export(artifact_id=_cell_id(cell)))
    loaded = require_success(ant.load(artifact))
    assert loaded.acceptance.recognized
    assert loaded.program_id == result.program_id
    assert loaded.claim_id == result.claim_id
    assert loaded.answer == result.answer
    bound = require_success(ant.load(artifact, expected={"claim_id": result.claim_id}))
    assert bound.claim_id == result.claim_id
    with pytest.raises(CausalSerializationError, match="expected claim_id differs"):
        require_success(ant.load(artifact, expected={"claim_id": "0" * 64}))


_CHILD = """
import json, sys
from pathlib import Path
import antecedent as ant

def forbidden(*args, **kwargs):
    raise AssertionError('a fresh-process load cannot analyze or prepare')
ant.analyze = forbidden
ant.prepare = forbidden
loaded = ant.load(Path(sys.argv[1]).read_bytes())
print(json.dumps({
    "program_id": loaded.program_id,
    "claim_id": loaded.claim_id,
    "kind": loaded.answer.kind,
    "status": loaded.acceptance.status,
    "recognized": loaded.acceptance.recognized,
}))
"""


def _fresh_process(path) -> dict:
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, str(path)], check=True, capture_output=True, text=True
    )
    return json.loads(child.stdout)


def _routes() -> dict:
    routes = {}
    for horizon, bayesian in _CELLS:
        prepared, data = _prepare(horizon, bayesian)
        routes[_cell_id((horizon, bayesian))] = require_success(prepared.estimate(data, seed=1))
    routes["anomaly-attribution"] = require_success(
        ant.analyze(_outlier_chain(), graph=_dag(), query=_anomaly_query())
    )
    routes["change-attribution"] = require_success(
        ant.analyze(_two_period_chain(), graph=_dag(), query=_change_query())
    )
    return routes


def test_every_route_artifact_reloads_in_a_fresh_process_with_its_identities(tmp_path) -> None:
    for name, result in _routes().items():
        path = tmp_path / f"{name}.ant"
        path.write_bytes(bytes(result.export(artifact_id=name)))
        seen = _fresh_process(path)
        assert seen["recognized"] is True, name
        assert seen["status"] in {"verified", "sealed"}, name
        assert seen["program_id"] == result.program_id, name
        assert seen["claim_id"] == result.claim_id, name
        assert seen["kind"] == result.answer.kind, name


def test_a_changed_byte_is_refused_or_never_a_different_recognized_answer() -> None:
    prepared, data = _prepare(False, False)
    result = require_success(prepared.estimate(data, seed=1))
    artifact = bytes(result.export())
    step = max(1, len(artifact) // 48)
    refused = 0
    for position in range(0, len(artifact), step):
        changed = bytearray(artifact)
        changed[position] ^= 0x01
        try:
            other = require_success(ant.load(bytes(changed)))
        except Exception:  # noqa: BLE001 - every refusal type is the expected outcome
            refused += 1
            continue
        if other.acceptance.recognized and other.acceptance.status in {"verified", "sealed"}:
            assert other.claim_id == result.claim_id, position
            assert other.answer == result.answer, position
    assert refused, "no flipped byte was ever refused"
    # A truncated artifact is never read as a shorter answer.
    with pytest.raises(_REFUSALS):
        require_success(ant.load(artifact[: len(artifact) // 2]))


def test_a_refresh_moves_the_data_not_the_program_and_leaves_the_earlier_artifact_intact(
    tmp_path,
) -> None:
    for horizon, bayesian in _CELLS:
        label = _cell_id((horizon, bayesian))
        prepared, data = _prepare(horizon, bayesian)
        result = require_success(prepared.estimate(data, seed=1))
        program = prepared.plan_view().identity
        earlier = bytes(result.export(artifact_id=label))
        earlier_path = tmp_path / f"{label}-earlier.ant"
        earlier_path.write_bytes(earlier)
        before = _fresh_process(earlier_path)

        if horizon:
            # A level shift of the outcome keeps the one-step contrast.
            moved = {"t": data["t"], "y": data["y"] + 0.5}
            truth = GAUSSIAN["effect"]
        else:
            # One more unit of slope moves the contrast.
            moved = {"t": data["t"], "y": data["y"] + data["t"]}
            truth = GAUSSIAN["effect"] + 1.0
        refreshed = require_success(prepared.refresh(moved, seed=1))
        tolerance = 2 * _tolerance(horizon, bayesian) if bayesian else 1e-6
        assert refreshed.estimate.ate == pytest.approx(truth, abs=tolerance), label
        assert refreshed.program_id == result.program_id, label
        assert prepared.plan_view().identity == program, label
        assert refreshed.data_snapshot_id != result.data_snapshot_id, label
        assert bytes(refreshed.export(artifact_id=label)) != earlier, label

        # The earlier artifact does not follow the refresh, in this process or a fresh one.
        assert _fresh_process(earlier_path) == before, label
        assert require_success(ant.load(earlier)).claim_id == result.claim_id, label
        report = require_success(ant.load(earlier)).inspect()
        assert report.data_snapshot_id == result.data_snapshot_id, label
        later = require_success(
            ant.load(
                bytes(refreshed.export(artifact_id=label)),
                expected={"data_snapshot_id": refreshed.data_snapshot_id},
            )
        )
        assert later.program_id == result.program_id, label
        with pytest.raises(CausalSerializationError, match="expected data_snapshot_id differs"):
            require_success(
                ant.load(
                    bytes(refreshed.export(artifact_id=label)),
                    expected={"data_snapshot_id": result.data_snapshot_id},
                )
            )


def test_a_refused_refresh_leaves_the_prepared_study_as_it_was() -> None:
    prepared, data = _prepare(False, False)
    first = require_success(prepared.estimate(data, seed=1))
    program = prepared.plan_view().identity
    wrong_schema = {"t": data["t"], "w": data["y"]}
    with pytest.raises(_REFUSALS):
        require_success(prepared.refresh(wrong_schema, seed=1))
    assert prepared.plan_view().identity == program
    again = require_success(prepared.estimate(data, seed=1))
    assert again.estimate.ate == first.estimate.ate
    assert again.data_snapshot_id == first.data_snapshot_id


_SUMMARY_CHILD = """
import json, sys
from pathlib import Path
from antecedent.transport import advanced as transport

summary_bytes = Path(sys.argv[1]).read_bytes()
ledger_bytes = Path(sys.argv[2]).read_bytes()

def refusal(call):
    try:
        call()
    except Exception as error:
        return [type(error).__name__, str(error)]
    return None

summary = transport.PeriodRefreshSummary.consume(summary_bytes)
out = {
    "retention": summary.retention,
    "n": summary.n,
    "beta": list(summary.beta),
    "residual_variance": summary.residual_variance,
    "ops": {
        "replay_rows": refusal(summary.replay_rows),
        "refit_from_rows": refusal(summary.refit_from_rows),
        "evict_row": refusal(lambda: summary.evict_row(0)),
        "set_mask": refusal(lambda: summary.set_mask(0, True)),
    },
    "after_refusals_n": summary.n,
    "summary_as_ledger": refusal(lambda: transport.consume_period_refresh_artifact(summary_bytes)),
    "ledger_as_summary": refusal(lambda: transport.PeriodRefreshSummary.consume(ledger_bytes)),
}
ledger = transport.consume_period_refresh_artifact(ledger_bytes)
out["ledger_rows_used"] = ledger.rows_used
out["ledger_beta"] = list(ledger.least_squares.beta)
print(json.dumps(out))
"""


def _read_summary_in_a_fresh_process(tmp_path, summary: bytes, ledger: bytes) -> dict:
    summary_path = tmp_path / "summary.ant"
    ledger_path = tmp_path / "ledger.ant"
    summary_path.write_bytes(summary)
    ledger_path.write_bytes(ledger)
    child = subprocess.run(
        [sys.executable, "-c", _SUMMARY_CHILD, str(summary_path), str(ledger_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(child.stdout)


def _operation_message(name: str) -> str:
    return name.replace("in_place", "in-place").replace("_", " ")


_UNSERVED = {
    "replay_rows": "row_reconstruction",
    "refit_from_rows": "refit_from_retained_rows",
    "evict_row": "exact_row_eviction",
    "set_mask": "in_place_weight_or_mask_update",
}


@pytest.mark.parametrize("invariant", [True, False], ids=["extended", "restarted"])
def test_a_summary_only_artifact_serves_a_fresh_process_and_refuses_the_rows_it_dropped(
    tmp_path, invariant
) -> None:
    refresh = started(invariant=invariant)
    outcome = require_success(refresh.refresh(SERIES[6:12], period_window(6, 12, "snap_2")))
    assert outcome.outcome == ("extended" if invariant else "restarted")
    live = refresh.least_squares
    seen = _read_summary_in_a_fresh_process(
        tmp_path, refresh.export_summary_only(), bytes(refresh.export())
    )

    assert seen["retention"] == "summary-only"
    assert seen["n"] == live.n
    assert seen["beta"] == pytest.approx(list(live.beta), abs=1e-9)
    assert seen["residual_variance"] == pytest.approx(live.residual_variance, abs=1e-9)
    for operation, refused in seen["ops"].items():
        assert refused is not None, f"{operation} was served from a summary"
        assert refused[0] == "TemporalExtensionRefusal", operation
        assert "summary-only cannot serve " + _operation_message(_UNSERVED[operation]) in refused[1]
    assert seen["after_refusals_n"] == live.n
    # Neither form is read as the other.
    assert seen["summary_as_ledger"] is not None
    assert seen["ledger_as_summary"] is not None
    # The ledger the rows stayed in recomputes the same fit that the summary served.
    assert seen["ledger_rows_used"] == live.n
    assert seen["ledger_beta"] == pytest.approx(list(live.beta), abs=1e-9)
    assert consume_period_refresh_artifact(bytes(refresh.export())).rows_used == live.n
    assert (
        require_success(transport.PeriodRefreshSummary.consume(refresh.export_summary_only())).n
        == live.n
    )


def test_a_declared_invariant_extension_and_a_restart_serve_different_summaries(tmp_path) -> None:
    extended = started(invariant=True)
    require_success(extended.refresh(SERIES[6:12], period_window(6, 12, "snap_2")))
    restarted = started(invariant=False)
    require_success(restarted.refresh(SERIES[6:12], period_window(6, 12, "snap_2")))
    pooled = require_success(transport.PeriodRefreshSummary.consume(extended.export_summary_only()))
    fresh_start = require_success(
        transport.PeriodRefreshSummary.consume(restarted.export_summary_only())
    )
    assert pooled.n > fresh_start.n, "the restart dropped the first period's rows"
    assert pooled.source_digest != fresh_start.source_digest
