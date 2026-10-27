"""A previous-release artifact adopts current digests and refuses a fresh-process reuse.

The Python half of the Rust cross-surface journey. The receipts are real: an in-process rerun
that changes only the utility, and a fresh-process resume that recomputes every derived stage.
Each digest is computed by the current reader, a stale one is the previous release's four-byte
encoding, and the route artifact is the frozen previous-release fixture carried onto the
methods of the committed route manifest.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from antecedent import release_migration as rm
from antecedent.errors import CausalSerializationError, CausalUnsupportedError
from antecedent.execution import Execution
from antecedent.outcomes import require_success
from antecedent.recalc import (
    RecalcNoLiveState,
    RecalcRequest,
    RecalcSession,
    ResumeContext,
    Utility,
)

from _repo_text import REPO_ROOT, load_toml

ROW_IDS = [7, 11, 13]
LEVEL = 0.0
SEED = 61
N = 300
ROUTE_FIXTURE = (
    REPO_ROOT
    / "crates"
    / "antecedent-io"
    / "tests"
    / "fixtures"
    / "previous_release_routes_v1.json"
)
DERIVED = ("identification", "score_artifact", "law", "decision")


def request() -> RecalcRequest:
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal(N)
    p = 1.0 / (1.0 + np.exp(-(-0.2 + 0.8 * z)))
    t = (rng.random(N) < p).astype(np.float64)
    y = (2.0 + 0.8 * z) * t + z + 0.3 * rng.standard_normal(N)
    return RecalcRequest(
        data={"t": t, "y": y, "z": z},
        edges=(("z", "t"), ("z", "y"), ("t", "y")),
        treatment="t",
        outcome="y",
        utility=Utility(2.0, 0.5),
    )


def artifact(
    score: bytes | None = None, target: bytes | None = None, receipt: bytes = b""
) -> bytes:
    return rm.previous_release_artifact(
        row_ids=ROW_IDS,
        score_row_digest=rm.score_row_digest(ROW_IDS) if score is None else score,
        target_digest=rm.target_digest(LEVEL) if target is None else target,
        receipt=receipt,
    )


def in_process_receipt() -> bytes:
    base = request()
    session = RecalcSession()
    require_success(session.execute(base, execution=Execution(seed=SEED)))
    return bytes(
        require_success(
            session.execute(
                replace(base, utility=Utility(3.0, 0.1)), execution=Execution(seed=SEED)
            )
        ).receipt.export()
    )


def fresh_process_receipt() -> bytes:
    base = request()
    first = RecalcSession()
    require_success(first.execute(base, execution=Execution(seed=SEED)))
    resumed = require_success(
        RecalcSession.resume(first.identities, ResumeContext(supplied_data=True))
    )
    return bytes(
        require_success(resumed.execute(base, execution=Execution(seed=SEED))).receipt.export()
    )


def test_a_stale_score_digest_or_a_foreign_target_digest_is_refused_with_the_remedy() -> None:
    narrow = rm.narrow_score_row_digest(ROW_IDS)
    assert narrow != rm.score_row_digest(ROW_IDS)
    with pytest.raises(CausalUnsupportedError, match="recompute it from the row ids"):
        rm.consume_previous_release_artifact(artifact(score=narrow), intervention_level=LEVEL)

    other = rm.target_digest(1.0)
    assert other != rm.target_digest(LEVEL)
    with pytest.raises(CausalUnsupportedError, match="recompute it from the target"):
        rm.consume_previous_release_artifact(artifact(target=other), intervention_level=LEVEL)
    # The same carried digest is current only for the target that produced it.
    adopted = rm.consume_previous_release_artifact(
        artifact(target=other, receipt=in_process_receipt()), intervention_level=1.0
    )
    assert adopted.target_digest == other

    # A digest of other row ids is not widened into the artifact's rows either.
    with pytest.raises(CausalUnsupportedError):
        rm.consume_previous_release_artifact(
            artifact(score=rm.score_row_digest([7, 11, 14])), intervention_level=LEVEL
        )


def test_an_in_process_receipt_reuses_derived_stages_and_recomputes_only_the_decision() -> None:
    adopted = rm.consume_previous_release_artifact(
        artifact(receipt=in_process_receipt()), intervention_level=LEVEL
    )
    assert adopted.score_row_digest == rm.score_row_digest(ROW_IDS)
    assert adopted.target_digest == rm.target_digest(LEVEL)
    assert adopted.boundary == "in_process"
    for stage in ("identification", "score_artifact", "law"):
        assert adopted.status(stage).startswith("reused"), stage
    assert adopted.status("decision") == "recomputed(upstream:utility<-utility:modified)"


def test_a_fresh_process_receipt_recomputes_every_derived_stage() -> None:
    adopted = rm.consume_previous_release_artifact(
        artifact(receipt=fresh_process_receipt()), intervention_level=LEVEL
    )
    assert adopted.boundary == "fresh_process"
    for stage in DERIVED:
        assert adopted.status(stage) == "recomputed(fresh_process)", stage


def test_a_receipt_that_reuses_a_derived_stage_in_a_fresh_process_is_refused() -> None:
    # The Rust suite forges a receipt that claims reuse of a derived stage; here the honest
    # path shows no such receipt can be produced by a session.
    # A session cannot claim portable scores it was never handed, so no such receipt exists.
    base = request()
    first = RecalcSession()
    require_success(first.execute(base, execution=Execution(seed=SEED)))
    claimed = ResumeContext(portable_scores=True, supplied_data=True)
    with pytest.raises(RecalcNoLiveState) as caught:
        require_success(
            require_success(RecalcSession.resume(first.identities, claimed)).execute(
                base, execution=Execution(seed=SEED)
            )
        )
    assert caught.value.stage == "score_artifact"


def test_bytes_that_are_not_a_previous_release_object_are_not_decoded() -> None:
    with pytest.raises(CausalSerializationError):
        rm.consume_previous_release_artifact(b"[]", intervention_level=LEVEL)
    with pytest.raises(CausalSerializationError):
        rm.consume_previous_release_artifact(
            artifact(receipt=b"not a receipt")[:-1], intervention_level=LEVEL
        )
    with pytest.raises(CausalUnsupportedError, match="receipt was not loaded"):
        rm.consume_previous_release_artifact(
            artifact(receipt=in_process_receipt()[:40]), intervention_level=LEVEL
        )


def manifest_methods() -> list[rm.CurrentMethod]:
    manifest = load_toml(REPO_ROOT / "parity" / "capabilities" / "route_manifest.toml")
    return [
        rm.CurrentMethod(
            id=m["id"],
            route=m["route"],
            estimator=m["estimator"],
            design=m["design"],
            inference=m["inference"],
            status=m["status"],
        )
        for m in manifest["method"]
    ]


def test_the_frozen_previous_route_artifact_gets_an_explicit_outcome_for_every_record() -> None:
    migrated = rm.consume_previous_route_records(ROUTE_FIXTURE.read_bytes(), manifest_methods())
    assert len(migrated) == 10
    assert not [m.id for m in migrated if m.is_break], "no fixture record is a break"
    by_id = {m.id: m for m in migrated}
    assert by_id["licensed.AverageEffect.Dag"].outcome == "split"
    assert by_id["closed.AnomalyAttribution.Admg"].outcome in {"one_to_one", "split"}

    response = by_id["licensed.InterventionResponse.CoDetermined"]
    assert response.outcome == "split"
    assert sum(m.narrowed for m in response.methods) == 1
    assert any(not m.narrowed and m.status == "licensed" for m in response.methods)

    temporal = by_id["licensed.AnomalyAttribution.TemporalDag"]
    assert temporal.outcome == "split"
    assert [axis for m in temporal.methods for axis in m.axes] == [
        "temporal frequentist",
        "temporal bayesian",
    ]

    execution_only = by_id["gcm.attribute.temporal.bayesian"]
    assert execution_only.outcome == "one_to_one"
    assert execution_only.methods[0].axes == ("temporal bayesian",)
    assert execution_only.unserved_axes == ("static bayesian",)


def test_a_removed_route_reports_its_break_beside_the_carried_records() -> None:
    routes = (
        b'{"routes":['
        b'{"id":"licensed.AverageEffect.Dag","status":"licensed",'
        b'"design_time_model":"static","inference":"Frequentist"},'
        b'{"id":"licensed.RemovedQuery.Dag","status":"licensed",'
        b'"design_time_model":"static","inference":"Frequentist"},'
        b'{"id":"graphless.removed.design","status":"graphless",'
        b'"design_time_model":"static","inference":"unlisted"},'
        b'{"id":"licensed.AverageEffect.Dag","status":"withdrawn",'
        b'"design_time_model":"static","inference":"Frequentist"}]}'
    )
    migrated = rm.consume_previous_route_records(routes, manifest_methods())
    assert [m.outcome for m in migrated] == ["split", "break", "retired", "break"]
    assert migrated[1].kind == "route_without_current_method"
    assert migrated[1].message is not None and "method that replaced it" in migrated[1].message
    assert migrated[3].kind == "unknown_status"
    with pytest.raises(CausalSerializationError):
        rm.consume_previous_route_records(b'{"routes":[{"id":"x"}]}', manifest_methods())
