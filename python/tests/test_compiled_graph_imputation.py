"""Bounded native graph-imputation lifecycle and refusal, without calibration claims."""

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


def test_retained_execution_is_closed_before_licensing():
    prepared = _prepared()
    with pytest.raises(GraphSelectionImputationRefusal):
        prepared.export()
    with pytest.raises(GraphSelectionImputationRefusal):
        prepared.execute()
    with pytest.raises(GraphSelectionImputationRefusal):
        consume_compiled_graph_imputation(
            b"candidate",
            program=bytes(32),
            source=bytes(32),
            receipt=bytes(32),
        )


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
            {"t": [0, 1] * 3, "y": [0, 1] * 3, "a": [0, 1] * 3, "b": [0, 1] * 3},
            treatment="t",
            outcome="y",
            graph_support="treatment_root_direct_effect",
            domains={"t": "binary", "y": "binary", "a": "binary", "b": "binary"},
        )


def test_python_interval_request_refuses_before_licensing():
    from antecedent.missing import _ConsumedGraphImputation

    with pytest.raises(GraphSelectionImputationRefusal) as refused:
        _ConsumedGraphImputation().interval()
    assert refused.value.code == "cell_not_licensed"


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
