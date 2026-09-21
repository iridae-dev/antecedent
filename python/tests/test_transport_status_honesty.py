"""Regression tests: transport results must publish only native scientific status.

Covers four related defects where the Python wrap layer authored a status
instead of reading it from the native execution: a fabricated analytic SE
(``python-pkg-1``), an unauthenticated export/load round trip
(``python-pkg-2``), a catalog-search error rendered as "identified"
(``python-pkg-3``), and identification/validation defaults that look like a
pass when nothing was actually checked (``python-pkg-4``).
"""

from __future__ import annotations

import base64
import json
import math

import pytest
from antecedent import Admg, AverageEffect, analyze, identify, prepare, transport
from antecedent.results import AnalysisResult
from antecedent.transport import advanced
from antecedent.transport._wrap import _RehydratedStudy, wrap_transport_result


def _graph():
    return Admg.from_edges(["x", "y"], [("x", "y")])


def _single_source():
    return transport.Evidence(
        source=transport.Source(
            "source", kind="experimental", interventions=["x"], sampling="independent"
        ),
        target_sampling="representative_sample",
    )


def _ate():
    return transport.Transport(
        AverageEffect("x", "y"),
        target="target",
        evidence=_single_source(),
    )


def _exact():
    return transport.ExactTransportData(
        (
            transport.ExactDiscreteLaw(
                "source",
                "source",
                (("y", (0.0, 1.0)),),
                (0.5, 0.5),
                "v1",
                interventions=(("x", 0.0),),
            ),
            transport.ExactDiscreteLaw(
                "source",
                "source",
                (("y", (0.0, 1.0)),),
                (0.2, 0.8),
                "v1",
                interventions=(("x", 1.0),),
            ),
        )
    )


def _statistical():
    return transport.StatisticalTransportData(
        samples=(
            transport.RegimeSample(
                "source",
                "source",
                "v1",
                {"y": [0.0] * 50 + [1.0] * 50},
                interventions=(("x", 0.0),),
            ),
            transport.RegimeSample(
                "source",
                "source",
                "v1",
                {"y": [0.0] * 20 + [1.0] * 80},
                interventions=(("x", 1.0),),
            ),
        )
    )


# --- python-pkg-1: fabricated se_analytic = 0.0 -----------------------------

# No response-family query reachable through the public ``Transport`` surface
# lowers to the plug-in "scalar" shape today: ``ResponseCurve`` refuses a
# grid shorter than two points, and every single-point derivative query
# (``PointDerivative``, ``Elasticity``, ``AverageDerivative``, ...) declares
# its ``at=`` as a plain float, which ``lower_question`` does not accept
# either. So the plug-in ``ExactTransportDistribution`` /
# ``StatisticalTransportDistribution`` branch of ``wrap_transport_result`` is
# exercised directly here, the same way ``prepare_transport`` would reach it
# for a shape it already produces.


def _exact_specialist():
    graph = _graph()
    identified = advanced.identify_classical(
        graph,
        advanced.SelectionDiagram("source", "target", []),
        outcomes=["y"],
        treatments=["x"],
    )
    catalog = advanced.EvidenceCatalog(
        regimes=[advanced.EvidenceRegime("obs", "target", measured=["x", "y"])]
    )
    law = advanced.ExactDiscreteLaw(
        "target",
        "obs",
        (("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
        (0.4, 0.1, 0.15, 0.35),
        "snapshot-1",
    )
    specialist = advanced.evaluate_exact(
        identified, catalog, advanced.ExactTransportData((law,)), at={"x": 1.0}
    )
    return identified, specialist


def _statistical_specialist():
    graph = _graph()
    identified = advanced.identify_classical(
        graph,
        advanced.SelectionDiagram("source", "target", []),
        outcomes=["y"],
        treatments=["x"],
    )
    catalog = advanced.EvidenceCatalog(
        environments=[
            advanced.Environment(
                "source",
                [
                    advanced.VariableCoordinate("x", "binary"),
                    advanced.VariableCoordinate("y", "binary"),
                ],
            ),
            advanced.Environment(
                "target",
                [
                    advanced.VariableCoordinate("x", "binary"),
                    advanced.VariableCoordinate("y", "binary"),
                ],
            ),
        ],
        regimes=[
            advanced.EvidenceRegime(
                "trial", "source", kind="experimental", interventions=["x"], measured=["y"]
            ),
        ],
        bindings=[
            advanced.RegimeBinding(
                "trial", "v1", sampling="independent", dependence="independent_studies"
            ),
        ],
        target_sampling="representative_sample",
    )
    sample = advanced.RegimeSample(
        "source", "trial", "v1", {"y": [0.0] * 20 + [1.0] * 80}, interventions=(("x", 1.0),)
    )
    data = advanced.StatisticalTransportData(samples=(sample,))
    study = prepare(
        data,
        query=advanced.StatisticalTransportQuery(
            identified, catalog, {"x": 1.0}, bootstrap=39, seed=7
        ),
    )
    return identified, study.estimate()


def test_exact_scalar_never_reports_a_fabricated_analytic_se():
    identified, specialist = _exact_specialist()
    study = _RehydratedStudy(_ate(), {"identified": identified, "shape": "scalar"})
    result = wrap_transport_result(study, specialist)
    assert isinstance(result, AnalysisResult)
    # 0.0 is a real, finite, wrong standard error; a complete exact law makes
    # no sampling claim at all, so nothing may be reported as computed.
    assert not math.isfinite(result.estimate.se_analytic)
    assert result.estimate.se_bootstrap is None
    assert result.transport.uncertainty_reason == "exact_supplied_law_no_sampling_uncertainty"
    report = result.inspect()
    assert report.uncertainty.available is False


def test_statistical_scalar_reports_native_interval_not_a_fabricated_se():
    identified, specialist = _statistical_specialist()
    study = _RehydratedStudy(_ate(), {"identified": identified, "shape": "scalar"})
    result = wrap_transport_result(study, specialist)
    assert isinstance(result, AnalysisResult)
    assert not math.isfinite(result.estimate.se_analytic)
    # Either the bootstrap-inferred interval survives, or the native reason
    # it was withheld does (e.g. too few replicates survived resampling);
    # either way this is real native uncertainty reasoning, never a fake SE.
    assert result.transport.uncertainty_reason is not None
    if result.transport.interval is not None:
        lower, upper = result.transport.interval
        assert lower <= result.estimate.ate <= upper


def test_exact_contrast_never_reports_a_fabricated_analytic_se():
    result = analyze(_exact(), graph=_graph(), query=_ate())
    assert isinstance(result, AnalysisResult)
    assert result.answer.kind == "point"
    assert not math.isfinite(result.estimate.se_analytic)


# --- python-pkg-4(b): validation defaults to passed=True when nothing ran --


def test_transport_validation_never_claims_passed_when_nothing_ran():
    result = analyze(_exact(), graph=_graph(), query=_ate())
    assert result.validation.ran is False
    assert result.validation.passed is False


# --- python-pkg-4(a): identification never defaults to identified ----------


def test_identification_view_fails_closed_with_no_native_stage():
    from antecedent.transport._wrap import _identification_view

    query = _ate()
    view = _identification_view(None, query)
    assert view.status == "NotIdentified"
    view_empty_stage = _identification_view({}, query)
    assert view_empty_stage.status == "NotIdentified"


# --- python-pkg-3: catalog-search error must not read as "identified" ------


def test_catalog_search_failure_is_reported_incomplete_not_identified():
    from antecedent.transport import _day1

    def _boom(*_args, **_kwargs):
        raise ValueError("transport.identification_budget")

    original = _day1.inspect_catalog
    _day1.inspect_catalog = _boom
    try:
        ident = identify(graph=_graph(), query=_ate())
    finally:
        _day1.inspect_catalog = original

    report = ident.inspect()
    # Before the fix this silently became support.available=True,
    # summary="not_estimated", reason=None -- a stalled search read as "no
    # evidence is missing". It must instead be reported as unknown, and the
    # detail must name the incomplete search rather than inventing "identified".
    assert report.support.available is False
    assert report.support.reason == "catalog_search_incomplete"
    assert "did not finish" in report.support.payload["detail"]


def test_catalog_search_unrecognized_error_propagates():
    from antecedent.transport import _day1

    def _boom(*_args, **_kwargs):
        raise RuntimeError("a genuine bug, not a budget/cancellation refusal")

    original = _day1.inspect_catalog
    _day1.inspect_catalog = _boom
    try:
        with pytest.raises(RuntimeError, match="genuine bug"):
            identify(graph=_graph(), query=_ate())
    finally:
        _day1.inspect_catalog = original


# --- python-pkg-2: export/load must verify through the native consumer -----


def test_transport_export_is_verified_natively_not_a_bare_json_blob():
    from antecedent import load

    # ``AverageEffect`` round-trips through ``_query_to_dict``/``_query_from_dict``
    # (unlike ``PointDerivative``, which those helpers do not reconstruct);
    # its contrast shape exercises the ``TransportResponseGrid`` export path.
    result = analyze(_exact(), graph=_graph(), query=_ate())
    blob = result.export()
    assert blob.startswith(b"ANTECEDENT-TRANSPORT-VIEW\x01")
    prefix = b"ANTECEDENT-TRANSPORT-VIEW\x01"
    payload = json.loads(blob[len(prefix) :])
    # The point estimate and identification status must not be sitting in
    # the envelope as trusted plaintext: they must come from a native
    # artifact that a hand edit cannot forge.
    assert "ate" not in payload
    assert "identification" not in payload
    assert "identification_artifact" in payload
    assert "specialist_artifact" in payload

    # Hand-editing the lineage metadata (never a scientific claim) is fine.
    payload["provider"] = "tampered_but_harmless"
    loaded = load(prefix + json.dumps(payload).encode())
    assert loaded.answer.value == pytest.approx(result.answer.value)
    assert loaded.identification.status == result.identification.status

    # Corrupting the native certificate must not silently keep the old
    # identification status -- it must fail rather than pass through.
    corrupt = dict(payload)
    corrupt["identification_artifact"] = base64.b64encode(b"not a real certificate").decode("ascii")
    with pytest.raises(ValueError):
        load(prefix + json.dumps(corrupt).encode())

    # Dropping the certificate entirely must fail closed (unidentified),
    # never keep reporting the original identified status.
    dropped = dict(payload)
    del dropped["identification_artifact"]
    reloaded = load(prefix + json.dumps(dropped).encode())
    assert reloaded.identification.status == "NotIdentified"
