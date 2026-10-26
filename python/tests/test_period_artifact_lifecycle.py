"""Canonical period artifacts retain independent results, never live sessions."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from hashlib import sha256

import antecedent as ac
import pytest
from antecedent.artifacts import ArtifactExpectation, ExportedArtifact
from antecedent.errors import CausalSerializationError, CausalTypeError, CausalValueError
from antecedent.outcomes import require_success
from antecedent.transport.advanced import (
    PeriodPrior,
    PeriodRefresh,
    PeriodRefreshSummary,
    PeriodRowCounts,
    TemporalExtensionRefusal,
)
from test_temporal_extensions import SERIES, period_window, started

LEDGER = "ANTECEDENT-PERIOD-REFRESH"
SUMMARY = "ANTECEDENT-PERIOD-REFRESH-SUMMARY"
BETA = [-0.3917647058823548, 0.5058823529411782, 1.3882352941176475]
POSTERIOR = [0.2272942518191619, 0.16164821678711794, 0.8370005826277753]


def _artifact(session, summary):
    return session.export(retention="summary-only") if summary else session.export()


@pytest.mark.parametrize("summary", [False, True], ids=["ledger", "summary"])
def test_period_export_preserves_historical_native_bytes_and_explicit_expectation(summary):
    session = started(invariant=True)
    native_export = (
        session._native.export_summary_only if summary else session._native.export_ledger
    )
    original, refusal = native_export()
    assert refusal is None
    artifact = _artifact(session, summary)
    assert isinstance(artifact, ExportedArtifact)
    assert artifact.kind == (SUMMARY if summary else LEDGER)
    assert bytes(artifact) == bytes(original)
    assert not hasattr(session, "export_summary_only")
    if not summary:
        assert bytes(session.export(retention="full")) == bytes(original)
    assert bytes(artifact).startswith(artifact.kind.encode() + b"\x01")
    assert artifact.expectation == ArtifactExpectation(
        artifact.kind, {"expected_encoding_sha256": sha256(bytes(original)).hexdigest()}
    )
    unbound = ac.load(artifact)
    bound = ac.load(bytes(artifact), expected=artifact.expectation)
    assert unbound.accepted and bound.accepted
    assert not unbound.acceptance.identity_bound
    assert bound.acceptance.identity_bound
    assert bound.acceptance.verified
    assert bound.acceptance.replayable is (not summary)
    assert not bound.acceptance.resumable
    assert bound.study is None
    assert bound.inspect().stage == "consumed"
    assert bytes(bound.export()) == bytes(artifact)
    result = require_success(bound)
    if summary:
        assert result.retention == "summary-only"
        assert result.n == 4
        assert result.source == "v@snap_1"
        assert result.row_counts == PeriodRowCounts(rows=4, live_rows=4, effective_n=4)
        assert result.beta == pytest.approx(BETA, abs=1e-9)
        assert result.residual_variance == pytest.approx(session.least_squares.residual_variance)
    else:
        assert result.rows_used == 4
        assert result.live_snapshots == ("snap_1",)
        assert result.least_squares.beta == pytest.approx(BETA, abs=1e-9)
        assert result.posterior.mean == pytest.approx(POSTERIOR, abs=1e-9)
        assert result.posterior.shape == 5.0
        assert result.posterior.rate == pytest.approx(2.1763865921233685, abs=1e-9)
    assert not hasattr(result, "refresh")
    assert not hasattr(result, "_prepared")
    assert not hasattr(result, "_data")


@pytest.mark.parametrize("summary", [False, True], ids=["ledger", "summary"])
def test_period_artifact_cannot_silently_bind_partial_or_foreign_identity(summary):
    artifact = _artifact(started(invariant=True), summary)
    with pytest.raises(CausalValueError, match="differs"):
        ac.load(artifact, expected={"expected_encoding_sha256": "0" * 64})
    with pytest.raises(CausalValueError, match="expected_encoding_sha256"):
        ac.load(artifact, expected={"prior_digest": "not-a-complete-record-identity"})
    with pytest.raises(CausalValueError, match="hexadecimal"):
        ac.load(artifact, expected={"expected_encoding_sha256": "invalid"})
    with pytest.raises(CausalTypeError, match="string"):
        ac.load(artifact, expected={"expected_encoding_sha256": 12})
    foreign_kind = LEDGER if summary else SUMMARY
    with pytest.raises(CausalValueError, match="kind differs"):
        ac.load(artifact, expected=ArtifactExpectation(foreign_kind, artifact.expectation.identity))


@pytest.mark.parametrize("summary", [False, True], ids=["ledger", "summary"])
def test_matching_encoding_expectation_still_invokes_period_semantic_verification(summary):
    data = bytes(_artifact(started(invariant=True), summary))
    changed = data[:-1] + bytes([data[-1] ^ 1])
    with pytest.raises((CausalSerializationError, TemporalExtensionRefusal)):
        require_success(
            ac.load(changed, expected={"expected_encoding_sha256": sha256(changed).hexdigest()})
        )
    with pytest.raises((CausalSerializationError, TemporalExtensionRefusal)):
        require_success(ac.load(data[:-5]))


def test_loaded_summary_refuses_row_operations_without_mutating_its_answer():
    session = started(invariant=True)
    summary = require_success(ac.load(session.export(retention="summary-only")))
    before = (summary.n, summary.beta, summary.residual_variance, summary.source_digest)
    for operation in (
        summary.replay_rows,
        summary.refit_from_rows,
        lambda: summary.evict_row(0),
        lambda: summary.set_mask(0, True),
    ):
        with pytest.raises(TemporalExtensionRefusal, match="summary-only cannot serve"):
            operation()
        assert (summary.n, summary.beta, summary.residual_variance, summary.source_digest) == before
    # Live sessions advance explicitly; already loaded records retain their own native state.
    require_success(session.refresh(SERIES[6:12], period_window(6, 12, "snap_2")))
    assert session.least_squares.n == 10
    assert (summary.n, summary.beta, summary.residual_variance, summary.source_digest) == before


@pytest.mark.parametrize("summary", [False, True], ids=["ledger", "summary"])
def test_period_artifact_loads_in_a_fresh_process_without_session_or_dataset(tmp_path, summary):
    artifact = _artifact(started(invariant=True), summary)
    path = tmp_path / "period.ant"
    path.write_bytes(bytes(artifact))
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from hashlib import sha256
from pathlib import Path
import sys
import antecedent as ac
from antecedent.outcomes import require_success
data = Path(sys.argv[1]).read_bytes()
summary = sys.argv[2] == 'summary'
loaded = ac.load(data, expected={'expected_encoding_sha256': sha256(data).hexdigest()})
assert loaded.accepted
assert loaded.study is None
assert loaded.acceptance.verified and loaded.acceptance.identity_bound
assert loaded.acceptance.replayable is (not summary)
assert not loaded.acceptance.resumable
assert loaded.inspect().stage == 'consumed'
record = require_success(loaded)
if summary:
    assert record.source == 'v@snap_1'
    assert record.row_counts.to_dict() == {'rows': 4, 'live_rows': 4, 'effective_n': 4}
assert not hasattr(record, 'refresh')
assert not hasattr(record, '_prepared') and not hasattr(record, '_data')
assert (record.n if summary else record.rows_used) == 4
beta = record.beta if summary else record.least_squares.beta
assert all(abs(a-b) < 1e-9 for a,b in zip(beta, [-0.3917647058823548, 0.5058823529411782, 1.3882352941176475]))
assert bytes(loaded.export()) == data
""",
            str(path),
            "summary" if summary else "ledger",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr


@pytest.mark.parametrize("retention", [None, True, 12, []])
def test_period_export_rejects_nonstring_retention_before_native_export(retention):
    session = started(invariant=True)
    session._native = object()  # Any attempted native export would raise AttributeError.
    with pytest.raises(CausalTypeError, match="retention"):
        session.export(retention=retention)


@pytest.mark.parametrize("retention", ["", "summary_only", "rows", "Full"])
def test_period_export_rejects_unknown_retention_without_changing_session(retention):
    session = started(invariant=True)
    before = session.least_squares
    with pytest.raises(CausalValueError, match="retention"):
        session.export(retention=retention)
    assert session.least_squares == before


def test_period_summary_row_counts_are_readonly_and_inspect_without_native_work(monkeypatch):
    summary = require_success(ac.load(started(invariant=True).export(retention="summary-only")))
    counts = summary.row_counts

    def forbid_native_payload(*args):
        raise AssertionError("count inspection must not read or refit its native parent")

    monkeypatch.setattr(PeriodRefreshSummary, "_payload", forbid_native_payload)
    assert counts.to_dict() == {"rows": 4, "live_rows": 4, "effective_n": 4}
    assert counts.inspect().details == counts.to_dict()
    assert len(repr(counts)) < 160
    assert "effective_n" in repr(counts)
    with pytest.raises(FrozenInstanceError):
        counts.rows = 100
    with pytest.raises(AttributeError):
        summary.row_counts = counts


def test_singular_period_fit_exports_a_summary_without_inventing_coefficients():
    # Every lagged regressor is [1, 2]: three observations cannot identify two coefficients.
    session = PeriodRefresh.start(
        [2.0, 2.0, 2.0, 2.0],
        series_id="v",
        lags=1,
        prior=PeriodPrior.isotropic(2, variance=1.0, shape=3.0, rate=2.0),
        window=period_window(0, 4, "flat"),
    )
    loaded = ac.load(session.export(retention="summary-only"))
    summary = require_success(loaded)
    assert loaded.acceptance.verified
    assert summary.source == "v@flat"
    assert summary.n == 3
    assert summary.row_counts == PeriodRowCounts(rows=3, live_rows=3, effective_n=3)
    assert summary.beta is None
    assert summary.residual_variance is None
    # A proper prior still yields a posterior; the full ledger promises an identified LS fit.
    posterior = session.posterior
    assert posterior.n == 3
    # With X rows [1, 2], y=2 and prior precision I, the posterior mean
    # solves [[4, 6], [6, 13]] beta = [6, 12], giving [3/8, 3/4].
    assert posterior.mean == pytest.approx([0.375, 0.75], abs=1e-12)
    assert posterior.precision == pytest.approx([4.0, 6.0, 6.0, 13.0], abs=1e-12)
    assert posterior.shape == 4.5
    assert posterior.rate == pytest.approx(2.375, abs=1e-12)
    assert session.least_squares.n == 3
    assert session.least_squares.beta is None
    assert session.least_squares.residual_variance is None
    assert session.events[0].snapshot_id == "flat"
    assert session.events[0].outcome == "start"
    assert len(session.prior_digest) == 64
    with pytest.raises(TemporalExtensionRefusal, match="least-squares fit is singular"):
        session.export()


def test_exactly_identified_period_fit_has_coefficients_without_residual_degrees_of_freedom():
    # Two lagged rows [1, 1] -> 2 and [1, 2] -> 4 give intercept=0, slope=2, n=p=2.
    session = PeriodRefresh.start(
        [1.0, 2.0, 4.0],
        series_id="v",
        lags=1,
        prior=PeriodPrior.isotropic(2, variance=1.0, shape=3.0, rate=2.0),
        window=period_window(0, 3, "two-rows"),
    )
    summary = require_success(ac.load(session.export(retention="summary-only")))
    assert summary.source == "v@two-rows"
    assert summary.n == 2
    assert summary.row_counts == PeriodRowCounts(rows=2, live_rows=2, effective_n=2)
    assert summary.beta == pytest.approx([0.0, 2.0], abs=1e-12)
    assert summary.residual_variance is None
    ledger = require_success(ac.load(session.export()))
    assert ledger.least_squares.beta == pytest.approx([0.0, 2.0], abs=1e-12)
    assert ledger.least_squares.residual_variance is None


def test_period_row_counts_are_exported_from_the_transport_domain():
    from antecedent.transport import advanced

    assert advanced.PeriodRowCounts is PeriodRowCounts
    assert "PeriodRowCounts" in advanced.__all__


def test_live_period_properties_request_only_their_native_read_model():
    session = started(invariant=True)
    native = session._native
    calls = []

    class ReadModels:
        def posterior_json(self):
            calls.append("posterior")
            return native.posterior_json()

        def least_squares_json(self):
            calls.append("least_squares")
            return native.least_squares_json()

        def events_json(self):
            calls.append("events")
            return native.events_json()

        def prior_digest(self):
            calls.append("prior_digest")
            return native.prior_digest()

    # No fallback payload, ledger export or numerical-operation method exists on
    # this forwarding surface. Each property must request only its own record.
    session._native = ReadModels()
    digest = session.prior_digest
    assert calls == ["prior_digest"]
    posterior, least_squares = session.posterior, session.least_squares
    assert calls == ["prior_digest", "posterior", "least_squares"]
    assert posterior.n == least_squares.n == 4
    assert session.events[0].snapshot_id == "snap_1"
    assert calls[-1] == "events"
    for _ in range(3):
        assert session.prior_digest == digest
        assert session.posterior == posterior
        assert session.least_squares == least_squares
    assert calls.count("events") == 1


def test_period_read_models_follow_mutations_and_preserve_prior_snapshots():
    session = started(invariant=True)
    before = session.posterior
    original_digest = session.prior_digest
    original_events = session.events
    original_bytes = bytes(session.export())
    with pytest.raises(TemporalExtensionRefusal):
        session.refresh(SERIES[6:12], period_window(0, 6, "overlap"))
    assert bytes(session.export()) == original_bytes
    assert session.posterior == before and session.events == original_events

    prior = PeriodPrior.isotropic(3, variance=2.0, shape=4.0, rate=3.0)
    session.reprior(prior)
    assert session.prior_digest != original_digest
    assert session.posterior.n == before.n
    assert session.posterior.mean != before.mean
    assert session.events == original_events
    assert session.check_against_fresh()
    assert session.summary_under(prior) == before.n
    session.refresh(SERIES[6:10], period_window(6, 10, "next"))
    assert session.posterior.n == before.n + 4
    assert session.least_squares.n == before.n + 4
    assert len(session.events) == len(original_events) + 1
    assert before.n == 4 and len(original_events) == 1
    loaded = require_success(ac.load(session.export()))
    # Independent replay accumulates retained rows instead of incrementally
    # combining period sufficient statistics; compare their numerical answers.
    posterior = session.posterior
    assert loaded.posterior.n == posterior.n
    assert loaded.posterior.shape == posterior.shape
    assert loaded.posterior.mean == pytest.approx(posterior.mean, rel=1e-12, abs=1e-12)
    assert loaded.posterior.precision == pytest.approx(posterior.precision, rel=1e-12, abs=1e-12)
    assert loaded.posterior.rate == pytest.approx(posterior.rate, rel=1e-12, abs=1e-12)
    least_squares = session.least_squares
    assert loaded.least_squares.n == least_squares.n
    assert loaded.least_squares.beta == pytest.approx(least_squares.beta, rel=1e-12, abs=1e-12)
    assert loaded.least_squares.residual_variance == pytest.approx(
        least_squares.residual_variance, rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize("summary", [False, True], ids=["ledger", "summary"])
def test_period_frame_inspection_is_bounded_and_requires_independent_consumption(
    monkeypatch, summary
):
    from antecedent import _native, artifacts
    from antecedent.transport import _period_refresh

    artifact = _artifact(started(invariant=True), summary)
    data = bytes(artifact)

    def forbid_semantic_consumer(*args, **kwargs):
        raise AssertionError("artifact inspection must not replay or verify period statistics")

    class ForbiddenSummaryConsumer:
        consume = staticmethod(forbid_semantic_consumer)

    # Cover the native public entry points and their already-imported domain aliases.
    monkeypatch.setattr(_native, "consume_period_refresh_artifact", forbid_semantic_consumer)
    monkeypatch.setattr(_period_refresh, "_consume_ledger", forbid_semantic_consumer)
    monkeypatch.setattr(_native, "NativePeriodRefreshSummary", ForbiddenSummaryConsumer)
    monkeypatch.setattr(_period_refresh, "_NativeSummary", ForbiddenSummaryConsumer)
    inspected = artifacts.inspect(artifact)
    assert inspected.kind == (SUMMARY if summary else LEDGER)
    assert inspected.metadata == {
        "encoded_payload_bytes": len(data) - len(inspected.kind.encode()) - 1,
        "validation": "requires_consumer",
    }
    assert inspected.acceptance.status == "unavailable"
    assert not inspected.acceptance.verified
    assert not inspected.acceptance.identity_bound
    assert not inspected.acceptance.replayable
    assert not inspected.acceptance.resumable
    assert len(repr(inspected.metadata)) < 160
    assert len(repr(inspected)) < 400
    assert "snap_1" not in repr(inspected)
    assert bytes(inspected.export()) == data


@pytest.mark.parametrize("invariant", [False, True], ids=["restart", "extend"])
def test_live_period_state_preserves_canonical_metadata_across_mutations(invariant):
    from antecedent import _native

    session = started(invariant=invariant)

    def compare_with_independent_ledger():
        consumed, refusal = _native.consume_period_refresh_artifact(bytes(session.export()))
        assert refusal is None
        consumed = json.loads(consumed)
        posterior, refusal = session._native.posterior_json()
        assert refusal is None
        assert json.loads(posterior) == consumed["posterior"]
        assert json.loads(session._native.least_squares_json()) == consumed["least_squares"]
        assert json.loads(session._native.events_json()) == consumed["events"]
        assert session.prior_digest == consumed["prior_digest"]
        assert session.posterior.n == consumed["rows_used"]
        assert consumed["series_id"] == "v"
        assert consumed["lags"] == 2
        return consumed

    initial = compare_with_independent_ledger()
    assert len(initial["blocks"]) == 1
    require_success(session.refresh(SERIES[6:12], period_window(6, 12, "snap_2")))
    advanced = compare_with_independent_ledger()
    assert len(advanced["blocks"]) == (2 if invariant else 1)
    session.evict_row(3 if invariant else 9)
    evicted = compare_with_independent_ledger()
    assert sum(block["rows"] for block in evicted["blocks"]) == advanced["rows_used"] - 1
    assert evicted["blocks"] != advanced["blocks"]
    assert all(len(block["digest"]) == 64 for block in evicted["blocks"])
