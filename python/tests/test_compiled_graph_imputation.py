"""Bounded native graph-imputation lifecycle, replay and refusal, without calibration claims."""

import math

import pytest
from antecedent import Execution, ResourceLimits
from antecedent.missing import (
    GraphSelectionImputationRefusal,
    PreparedGraphImputation,
    consume_compiled_graph_imputation,
)
from antecedent.state import CancellationToken


def _prepared(execution=None):
    return PreparedGraphImputation(
        {"t": [0, 1] * 6, "y": [0.2, 1.3, -0.1, 0.8, 0.3, 1.0, None, 1.4, -0.3, 0.9, 0.1, 1.1]},
        treatment="t",
        outcome="y",
        graph_support="treatment_root_direct_effect",
        domains={"t": "binary", "y": "continuous"},
        imputations=2,
        iterations=2,
        n_draws=100,
        prior_scale=1.0,
        execution=execution,
    )


def _consume(artifact, **overrides):
    expected = {"program": artifact.program, "source": artifact.source, "receipt": artifact.receipt}
    expected.update(overrides)
    return consume_compiled_graph_imputation(artifact.data, **expected)


def test_retained_execution_replays_from_the_exported_source():
    prepared = _prepared()
    result = prepared.execute()
    assert math.isfinite(result.mean) and result.diagnostic_sd > 0
    artifact = prepared.export()
    replayed = _consume(artifact)
    assert replayed.receipt == result.receipt
    assert replayed.mean == result.mean


def test_replay_refuses_changed_source_and_wrong_expected_identity():
    prepared = _prepared()
    prepared.execute()
    artifact = prepared.export()
    changed = bytearray(artifact.data)
    changed[-1] ^= 1
    with pytest.raises(GraphSelectionImputationRefusal):
        consume_compiled_graph_imputation(
            bytes(changed),
            program=artifact.program,
            source=artifact.source,
            receipt=artifact.receipt,
        )
    with pytest.raises(GraphSelectionImputationRefusal):
        _consume(artifact, receipt=bytes(32))
    with pytest.raises(GraphSelectionImputationRefusal):
        consume_compiled_graph_imputation(
            b"candidate", program=bytes(32), source=bytes(32), receipt=bytes(32)
        )


def test_export_requires_an_executed_point():
    with pytest.raises(GraphSelectionImputationRefusal):
        _prepared().export()


def test_declaration_refuses_missing_domains_and_unbounded_shape():
    with pytest.raises(ValueError, match="declare one domain"):
        PreparedGraphImputation(
            {"t": [0, 1, 0, 1], "y": [0, 1, 0, 1]},
            treatment="t",
            outcome="y",
            graph_support="treatment_root_direct_effect",
            domains={},
        )
    with pytest.raises(ValueError, match="source shape"):
        PreparedGraphImputation(
            {name: [0, 1] * 3 for name in "tyabcd"},
            treatment="t",
            outcome="y",
            graph_support="treatment_root_direct_effect",
            domains={name: "binary" for name in "tyabcd"},
        )


def _calibrated_or_refused(call):
    # Licensing is a record lookup at measurement: with no covering record the route
    # refuses, and once one exists the same call returns a calibrated result.
    try:
        return call()
    except GraphSelectionImputationRefusal as refused:
        assert refused.code == "cell_not_licensed"
        return None


def test_interval_is_calibrated_or_refused_where_no_record_covers_the_execution():
    result = _prepared().execute()
    for call in (result.interval, result.measured):
        measured = _calibrated_or_refused(call)
        assert measured is None or measured.scalars


def test_consumed_result_keeps_the_measured_route():
    prepared = _prepared()
    prepared.execute()
    replayed = _consume(prepared.export())
    for call in (replayed.interval, replayed.measured):
        measured = _calibrated_or_refused(call)
        assert measured is None or measured.scalars


def test_wider_shapes_and_the_parent_admitting_support_are_accepted():
    columns = {
        "t": [0, 1] * 8,
        "y": [0.2, 1.3, -0.1, 0.8, 0.3, 1.0, None, 1.4, -0.3, 0.9, 0.1, 1.1, 0.4, 1.2, 0.0, 0.7],
        "a": [0.1, -0.2, 0.4, 0.3, -0.5, 0.2, 0.0, 0.6, -0.1, 0.3, 0.2, -0.4, 0.5, 0.1, -0.3, 0.2],
        "b": [-0.3, 0.2, 0.1, -0.2, 0.4, 0.0, 0.3, -0.1, 0.2, 0.5, -0.4, 0.1, 0.0, -0.2, 0.3, 0.1],
    }
    prepared = PreparedGraphImputation(
        columns,
        treatment="t",
        outcome="y",
        graph_support="treatment_direct_effect",
        domains={"t": "binary", "y": "continuous", "a": "continuous", "b": "continuous"},
        imputations=2,
        iterations=2,
        n_draws=100,
    )
    result = prepared.execute()
    assert math.isfinite(result.mean) and result.diagnostic_sd > 0


def test_preparation_observes_resource_and_cancellation_policy():
    with pytest.raises(GraphSelectionImputationRefusal) as refused:
        _prepared(Execution(limits=ResourceLimits(memory_bytes=1)))
    assert refused.value.code == "execution_resource_limit"
    cancel = CancellationToken()
    cancel.cancel()
    with pytest.raises(GraphSelectionImputationRefusal) as refused:
        _prepared(Execution(cancel=cancel))
    assert refused.value.code == "cancelled_no_claim"


def test_prepared_handle_retains_original_execution_policy():
    cancel = CancellationToken()
    prepared = _prepared(Execution(cancel=cancel))
    cancel.cancel()
    with pytest.raises(GraphSelectionImputationRefusal) as refused:
        prepared.execute()
    assert refused.value.code == "cancelled_no_claim"
    with pytest.raises(GraphSelectionImputationRefusal) as refused:
        consume_compiled_graph_imputation(
            b"candidate",
            program=bytes(32),
            source=bytes(32),
            receipt=bytes(32),
            execution=Execution(cancel=cancel),
        )
    assert refused.value.code == "cancelled_no_claim"


def test_support_must_be_explicit_and_must_match_the_native_cell():
    data = {"t": [0, 1, 0, 1], "y": [0.1, 1.1, 0.2, 1.3]}
    with pytest.raises(TypeError, match="graph_support"):
        PreparedGraphImputation(
            data, treatment="t", outcome="y", domains={"t": "binary", "y": "continuous"}
        )
    with pytest.raises(GraphSelectionImputationRefusal):
        PreparedGraphImputation(
            data,
            treatment="t",
            outcome="y",
            graph_support="arbitrary",
            domains={"t": "binary", "y": "continuous"},
        )
