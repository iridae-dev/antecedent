"""Checked empirical histories: SCM truth, retention, source replay and closed intervals."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from antecedent.execution import Execution, ResourceLimits
from antecedent.outcomes import require_success
from antecedent.recalc import RecalcRefusal, Utility
from antecedent.recalc_capabilities import (
    Family,
    Operation,
    RetainedKind,
    require_adapter,
    retained_kind,
)
from antecedent.recalc_temporal import (
    TemporalEffect,
    TemporalHistory,
    TemporalRequest,
    TemporalResponse,
    TemporalSession,
    TemporalUnit,
    consume_temporal_result,
)


def request():
    units = []
    for s0 in range(2):
        for a1 in range(2):
            for a2 in range(2):
                for ul in range(4):
                    for uy in range(20):
                        l2 = int(ul < s0 + a1 + 1)
                        y = float(uy < 2 + 4 * a1 + 3 * l2 + 5 * a2 + 2 * s0 + 2 * s0 * a1)
                        units.append(
                            TemporalUnit(
                                len(units),
                                (
                                    TemporalHistory(0, s0, a1, l2, a2, y),
                                    TemporalHistory(2, s0, a1, l2, a2, y),
                                ),
                            )
                        )
    return TemporalRequest(
        tuple((a, b) for a in range(5) for b in range(a + 1, 5)),
        units,
        (0, 5),
        "enumerated-dependent-source",
        (0.2, 0.8),
        "given-target-law",
        TemporalEffect((1, 1), (0, 0)),
        Utility(2, 0.5),
    )


def truth(a1, a2, p1):
    # Integrate structural noise probabilities, independently of empirical factor fitting.
    return 0.1375 + 0.2375 * a1 + 0.25 * a2 + 0.1375 * p1 + 0.1 * p1 * a1


def test_temporal_accepts_one_hundred_thousand_history_rows_as_columns():
    from antecedent.errors import CausalValueError

    base = request()
    cells = [unit.histories[0] for unit in base.units]
    # 6,400 units of 16 non-overlapping windows each: 102,400 history rows, ten copies of every
    # cell, so the empirical factors and the SCM truth are those of the 640-unit request.
    units = tuple(
        TemporalUnit(
            len(cells) * copy + index,
            tuple(TemporalHistory(2 * step, h.s0, h.a1, h.l2, h.a2, h.y) for step in range(16)),
        )
        for copy in range(10)
        for index, h in enumerate(cells)
    )
    big = replace(base, units=units, period=(0, 40))
    assert sum(len(unit.histories) for unit in units) == 102_400
    result = require_success(TemporalSession().execute(big, execution=Execution(seed=41)))
    assert result.means == pytest.approx((truth(1, 1, 0.8), truth(0, 0, 0.8)), abs=1e-9)
    with pytest.raises(CausalValueError, match="recalc.memory_budget_exceeded"):
        require_success(
            TemporalSession().execute(
                big,
                execution=Execution(seed=41, limits=ResourceLimits(memory_bytes=1_000_000)),
            )
        )


def test_temporal_actual_history_effect_and_response_scm_truth_and_full_rerun():
    req = request()
    session = TemporalSession()
    result = require_success(session.execute(req, execution=Execution(seed=41)))
    assert result.means == pytest.approx((truth(1, 1, 0.8), truth(0, 0, 0.8)), abs=1e-12)
    assert result.law.ate == pytest.approx(0.5675, abs=1e-12)
    assert result.law.std_error is None
    assert result.receipt.totals.factor_builds > 0
    assert result.receipt.totals.program_compilations > 0
    assert result.receipt.totals.integrations > 0
    assert result.receipt.totals.model_fits == result.receipt.totals.fold_fits == 0
    assert (
        require_success(TemporalSession().execute(req, execution=Execution(seed=41))).means
        == result.means
    )
    response = replace(req, functional=TemporalResponse((1, 1)))
    result = require_success(session.execute(response, execution=Execution(seed=41)))
    assert isinstance(result.functional, TemporalResponse)
    assert result.law.ate == pytest.approx(truth(1, 1, 0.8), abs=1e-12)
    assert result.means == pytest.approx((result.law.ate,), abs=1e-12)
    assert (
        require_success(TemporalSession().execute(response, execution=Execution(seed=41))).means
        == result.means
    )


def test_temporal_utility_target_and_functional_reuse_actual_mechanisms():
    req = request()
    session = TemporalSession()
    first = require_success(session.execute(req, execution=Execution(seed=41)))
    assert (
        require_success(session.execute(req, execution=Execution(seed=41))).receipt.totals.total
        == 0
    )
    utility = require_success(
        session.execute(replace(req, utility=Utility(3, 1)), execution=Execution(seed=41))
    )
    assert utility.receipt.totals.total == utility.receipt.totals.decisions == 1
    assert utility.law == first.law
    target = replace(req, initial_state=(0.5, 0.5), initial_state_id="new-target")
    result = require_success(session.execute(target, execution=Execution(seed=41)))
    assert result.receipt.totals.factor_builds == 0
    assert result.law.ate == pytest.approx(truth(1, 1, 0.5) - truth(0, 0, 0.5), abs=1e-12)
    assert (
        require_success(TemporalSession().execute(target, execution=Execution(seed=41))).law
        == result.law
    )


def test_temporal_source_history_changes_refit_and_match_fresh_execution():
    req = request()
    session = TemporalSession()
    first = require_success(session.execute(req, execution=Execution(seed=41)))
    changed = replace(
        req,
        units=tuple(
            TemporalUnit(
                unit.unit_id, tuple(replace(history, y=1 - history.y) for history in unit.histories)
            )
            for unit in req.units
        ),
        snapshot_id="changed-source",
    )
    result = require_success(session.execute(changed, execution=Execution(seed=41)))
    assert result.receipt.totals.factor_builds > 0
    assert result.law.ate == pytest.approx(-first.law.ate, abs=1e-12)
    assert (
        require_success(TemporalSession().execute(changed, execution=Execution(seed=41))).means
        == result.means
    )


def test_temporal_unmeasured_history_protocol_preserves_source_state():
    session = TemporalSession()
    require_success(session.execute(request(), execution=Execution(seed=41)))
    ids = session.identities
    with pytest.raises(RecalcRefusal) as error:
        session.dependent_interval()
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "cell_not_licensed",
        "temporal_interval.route_frozen",
        "inference",
    )
    assert session.identities == ids
    assert (
        require_success(
            session.execute(request(), execution=Execution(seed=41))
        ).receipt.totals.total
        == 0
    )


def test_temporal_exact_scope_refusals_preserve_native_state():
    req = request()
    session = TemporalSession()
    require_success(session.execute(req, execution=Execution(seed=41)))
    ids = session.identities
    for changed in (
        replace(req, horizon=3),
        replace(req, selection_targets=(2,)),
        replace(req, bidirected=((0, 4),)),
    ):
        with pytest.raises(RecalcRefusal) as error:
            require_success(session.execute(changed, execution=Execution(seed=41)))
        assert (error.value.reason_code, error.value.detail, error.value.stage) == (
            "route_not_supported",
            "recalc.temporal_scope_unsupported",
            "identification",
        )
        assert session.identities == ids
    assert (
        require_success(session.execute(req, execution=Execution(seed=41))).receipt.totals.total
        == 0
    )


def test_temporal_native_capabilities_and_bounded_histories():
    from antecedent.errors import CausalValueError

    session = TemporalSession()
    assert retained_kind(session) == RetainedKind.READABLE
    require_success(session.execute(request(), execution=Execution(seed=41)))
    assert retained_kind(session) == RetainedKind.LIVE_HISTORY
    assert require_adapter(Family.TEMPORAL, Operation.TARGET, session).state_type is TemporalSession
    unit = request().units[0]
    many = replace(request(), units=(unit,) * 4097)
    with pytest.raises(CausalValueError) as error:
        require_success(
            session.execute(
                many,
                execution=Execution(seed=41, limits=ResourceLimits(memory_bytes=4096)),
            )
        )
    assert error.value.reason_code == "invalid_argument"
    assert "recalc.memory_budget_exceeded" in str(error.value)


def test_temporal_source_artifact_independent_fresh_process_rebuild(tmp_path):
    req = request()
    session = TemporalSession()
    result = require_success(session.execute(req, execution=Execution(seed=41)))
    artifact = tmp_path / "histories.artifact"
    receipt = tmp_path / "histories.receipt"
    artifact.write_bytes(bytes(session.export(execution=Execution(seed=999))))
    receipt.write_bytes(bytes(result.receipt.export()))
    replay = require_success(
        consume_temporal_result(artifact.read_bytes(), execution=Execution(seed=41))
    )
    assert replay.result.means == result.means
    assert replay.result.receipt.totals.factor_builds > 0
    assert (
        require_success(
            replay.session.execute(req, execution=Execution(seed=41))
        ).receipt.totals.total
        == 0
    )
    with pytest.raises(RecalcRefusal) as error:
        require_success(
            consume_temporal_result(artifact.read_bytes(), execution=Execution(seed=42))
        )
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "invalid_argument",
        "recalc.temporal_artifact_seed_mismatch",
        "score_artifact",
    )
    code = "from antecedent.outcomes import require_success\nfrom antecedent.execution import Execution\nimport json, pathlib, sys\nsys.path.insert(0, sys.argv[3])\nfrom test_recalc_temporal import request\nfrom antecedent.recalc import RecalcReceipt, ResumeContext, RecalcRefusal\nfrom antecedent.recalc_temporal import TemporalSession, consume_temporal_result\nreplay = require_success(consume_temporal_result(pathlib.Path(sys.argv[1]).read_bytes(), execution=Execution(seed=41)))\nreceipt = require_success(RecalcReceipt.consume(pathlib.Path(sys.argv[2]).read_bytes()))\nflags = require_success(TemporalSession.resume(receipt, ResumeContext(portable_fit=True, portable_scores=True)))\ntry:\n    require_success(flags.execute(request(), execution=Execution(seed=41)))\n    raise AssertionError('flags restored native mechanisms')\nexcept RecalcRefusal as error:\n    refused = (error.reason_code, error.detail, error.stage)\nraw = require_success(require_success(TemporalSession.resume(receipt, ResumeContext(supplied_data=True))).execute(request(), execution=Execution(seed=41)))\nprint(json.dumps({'means': replay.result.means, 'point': replay.result.law.ate, 'rebuild': replay.result.receipt.totals.factor_builds, 'reuse': require_success(replay.session.execute(request(), execution=Execution(seed=41))).receipt.totals.total, 'raw': raw.receipt.totals.factor_builds, 'refused': refused}))\n"
    child = subprocess.run(
        [sys.executable, "-c", code, str(artifact), str(receipt), str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(child.stdout)
    assert observed["means"] == list(result.means)
    assert observed["point"] == result.law.ate and observed["rebuild"] > 0 and observed["raw"] > 0
    assert observed["reuse"] == 0
    assert observed["refused"] == [
        "score_table_unavailable",
        "recalc.unavailable_data",
        "data_snapshot",
    ]


def test_temporal_typed_unit_ownership_and_source_support_refusals():
    req = request()
    session = TemporalSession()
    require_success(session.execute(req, execution=Execution(seed=41)))
    ids = session.identities
    duplicate = replace(req, units=(req.units[0], req.units[0]))
    with pytest.raises(RecalcRefusal) as error:
        require_success(session.execute(duplicate, execution=Execution(seed=41)))
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "route_not_supported",
        "recalc.temporal_history_unsupported",
        "data_snapshot",
    )
    assert session.identities == ids
    gap = replace(
        req,
        units=tuple(unit for unit in req.units if unit.histories[0].s0 == 0),
        snapshot_id="missing-initial-state",
    )
    with pytest.raises(RecalcRefusal) as error:
        require_success(session.execute(gap, execution=Execution(seed=41)))
    assert (error.value.reason_code, error.value.detail, error.value.stage) == (
        "transport_support_failure",
        "recalc.temporal_support_failure",
        "law",
    )
    assert session.identities == ids
    assert (
        require_success(session.execute(req, execution=Execution(seed=41))).receipt.totals.total
        == 0
    )
