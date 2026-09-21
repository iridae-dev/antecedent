"""Wrap T5–T9 specialist results as Analysis / CausalResponseView."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import CausalUnsupportedError
from ..results import (
    AnalysisResult,
    CausalResponseView,
    EstimateView,
    IdentificationView,
    PerformanceView,
    ResponseUncertainty,
    ResponseView,
    SupportDiagnostic,
    SupportReport,
    ValidationView,
)
from ..results._execution import Answer
from ._day1 import Evidence, Source, Transport, missing_evidence_detail
from ._impl import (
    EmpiricalTable,
    ExactTransportDistribution,
    LearnedCategorical,
    LearnedTrialEstimate,
    StatisticalTransportDistribution,
    TransportResponseGrid,
    TrialAipw,
)

VIEW_PREFIX = b"ANTECEDENT-TRANSPORT-VIEW\x01"


@dataclass(frozen=True, slots=True)
class TransportSection:
    """Lineage for a transported claim. Specialist view stays on ``distribution``."""

    formula: str | None = None
    provider: str | None = None
    bindings: tuple[str, ...] = ()
    replicate_ids: tuple[int, ...] = ()
    distribution: Any = None
    unavailable: str | None = None
    shape: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "provider": self.provider,
            "bindings": list(self.bindings),
            "replicate_ids": list(self.replicate_ids),
            "unavailable": self.unavailable,
            "shape": self.shape,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TransportSection:
        return cls(
            formula=raw.get("formula"),
            provider=raw.get("provider"),
            bindings=tuple(raw.get("bindings") or ()),
            replicate_ids=tuple(raw.get("replicate_ids") or ()),
            unavailable=raw.get("unavailable"),
            shape=raw.get("shape"),
        )


def _provider_name(provider: object) -> str:
    if isinstance(provider, EmpiricalTable) or provider in (None, "plugin"):
        return "empirical_table"
    if isinstance(provider, LearnedCategorical):
        return "learned_categorical"
    if isinstance(provider, TrialAipw):
        return "trial_aipw"
    return type(provider).__name__


def _identification_view(stage: Mapping[str, Any] | None, query: Transport) -> IdentificationView:
    identified = None if stage is None else stage.get("identified")
    status = "NonparametricallyIdentified"
    method = "identify.transport_sid"
    rules: list[str] = []
    if identified is not None:
        if identified.outcome != "identified":
            status = "NotIdentified"
        rules = list(identified.rules)
        method = "identify.transport_sid"
    return IdentificationView(
        status=status,
        method=method,
        adjustment_set=[],
        assumption_count=len(rules),
        derivation_step_count=len(rules),
    )


def _empty_validation() -> ValidationView:
    return ValidationView(passed=True, ran=False, count=0)


def _empty_performance() -> PerformanceView:
    return PerformanceView()


class _ExportAdapter:
    def __init__(self, result: Any) -> None:
        self._result = result

    def export_contracted_artifact(self, artifact_id: str = "analysis-result") -> bytes:
        return encode_transport_view(self._result)

    def export(self) -> bytes:
        return encode_transport_view(self._result)


def _attach(result: Any, study: Any, specialist: Any, section: TransportSection) -> Any:
    object.__setattr__(result, "query", getattr(study, "_query", None))
    object.__setattr__(result, "_execution", _ExportAdapter(result))
    object.__setattr__(result, "_prepared", study)
    return result


def _query_to_dict(query: Transport) -> dict[str, Any]:
    question = query.question
    return {
        "question": {
            "type": type(question).__name__,
            "treatment": getattr(question, "treatment", None),
            "outcome": getattr(question, "outcome", None),
            "grid": list(getattr(question, "grid", ()) or ()),
            "control_level": getattr(question, "control_level", None),
            "active_level": getattr(question, "active_level", None),
        },
        "target": query.target,
        "selections": list(query.selections),
        "evidence": {
            "target_sampling": query.evidence.target_sampling,
            "sources": [
                {
                    "identity": source.identity,
                    "kind": source.kind,
                    "interventions": list(source.interventions),
                    "sampling": source.sampling,
                    "dependence": source.dependence,
                    "selections": list(source.selections),
                    "measured": None if source.measured is None else list(source.measured),
                }
                for source in query.evidence.sources
            ],
        },
    }


def _query_from_dict(raw: Mapping[str, Any] | None) -> Transport | None:
    if not raw:
        return None
    from ..query import AverageEffect, ResponseCurve

    body = raw["question"]
    if body["type"] == "AverageEffect":
        kwargs: dict[str, Any] = {}
        if body.get("control_level") is not None:
            kwargs["control_level"] = body["control_level"]
        if body.get("active_level") is not None:
            kwargs["active_level"] = body["active_level"]
        question: Any = AverageEffect(body["treatment"], body["outcome"], **kwargs)
    else:
        question = ResponseCurve(
            body["treatment"], body["outcome"], grid=body.get("grid") or [0.0, 1.0]
        )
    sources = [
        Source(
            item["identity"],
            kind=item["kind"],
            interventions=item.get("interventions") or (),
            sampling=item["sampling"],
            dependence=item.get("dependence"),
            selections=item.get("selections") or (),
            measured=item.get("measured"),
        )
        for item in raw["evidence"]["sources"]
    ]
    return Transport(
        question,
        raw["target"],
        Evidence(source=sources, target_sampling=raw["evidence"]["target_sampling"]),
        selections=raw.get("selections") or (),
    )


def encode_transport_view(result: Any) -> bytes:
    section = result.transport
    query = getattr(result, "query", None)
    identification = result.identification
    estimate = getattr(result, "estimate", None)
    payload: dict[str, Any] = {
        "shape": section.shape,
        "transport": section.to_dict(),
        "query": _query_to_dict(query) if isinstance(query, Transport) else None,
        "identification": {
            "status": identification.status,
            "method": identification.method,
            "adjustment_set": list(identification.adjustment_set),
            "assumption_count": identification.assumption_count,
            "derivation_step_count": identification.derivation_step_count,
        },
        "ate": None if estimate is None else getattr(estimate, "ate", None),
        "estimator_id": None if estimate is None else getattr(estimate, "estimator_id", None),
        "method": None if estimate is None else getattr(estimate, "method", None),
        "response": None,
        "support": None,
        "diagnostics": list(getattr(result, "diagnostics", None) or ()),
    }
    response = getattr(result, "response", None)
    if response is not None:
        payload["response"] = {
            "treatments": list(response.treatments),
            "outcomes": list(response.outcomes),
            "points": [list(point) for point in response.points],
            "values": [list(value) for value in response.values],
        }
        support = result.support
        payload["support"] = {
            "status": support.status,
            "query_region": support.query_region,
            "warnings": list(getattr(support, "warnings", ()) or ()),
            "point_status": list(support.point_status) if support.point_status else None,
        }
    return VIEW_PREFIX + json.dumps(payload).encode()


def decode_transport_view(encoded: bytes) -> AnalysisResult | CausalResponseView:
    payload = json.loads(encoded[len(VIEW_PREFIX) :])
    section = TransportSection.from_dict(payload["transport"])
    query = _query_from_dict(payload.get("query"))
    view = IdentificationView(
        status=payload["identification"]["status"],
        method=payload["identification"]["method"],
        adjustment_set=list(payload["identification"]["adjustment_set"]),
        assumption_count=payload["identification"]["assumption_count"],
        derivation_step_count=payload["identification"]["derivation_step_count"],
    )
    if payload.get("response") is not None:
        raw = payload["response"]
        support = payload.get("support") or {}
        result: Any = CausalResponseView(
            estimand=None if query is None else query.question,
            response=ResponseView(
                treatments=list(raw["treatments"]),
                outcomes=list(raw["outcomes"]),
                points=list(raw["points"]),
                values=list(raw["values"]),
            ),
            estimate=[row[0] for row in raw["values"]],
            uncertainty=ResponseUncertainty(kind="none"),
            support=SupportReport(
                status=support.get("status") or "supported",
                query_region=support.get("query_region") or {},
                warnings=tuple(support.get("warnings") or ()),
                point_status=tuple(support["point_status"])
                if support.get("point_status")
                else None,
            ),
            identification=view,
            transport=section,
        )
    else:
        result = AnalysisResult(
            identification=view,
            estimate=EstimateView(
                ate=payload.get("ate"),
                se_analytic=0.0,
                se_bootstrap=None,
                estimator_id=payload.get("estimator_id") or "transport.empirical_table",
                method=payload.get("method") or "transport.plugin",
            ),
            posterior=None,
            validation=_empty_validation(),
            performance=_empty_performance(),
            diagnostics=list(payload.get("diagnostics") or ()),
            provenance={"operation_ids": ["estimate.transport"]},
            transport=section,
        )
    object.__setattr__(result, "query", query)
    object.__setattr__(result, "_execution", _ExportAdapter(result))
    return result


def _unavailable_result(
    study: Any,
    query: Transport,
    detail: str,
    *,
    shape: str,
    stage: Mapping[str, Any] | None,
) -> AnalysisResult | CausalResponseView:
    identified = None if stage is None else stage.get("identified")
    section = TransportSection(
        formula=None if identified is None else identified.formula,
        provider=_provider_name(None if stage is None else stage.get("provider")),
        unavailable=detail,
        shape=shape,
    )
    view = _identification_view(stage, query)
    if shape == "grid":
        result = CausalResponseView(
            estimand=query.question,
            response=None,
            estimate=None,
            uncertainty=ResponseUncertainty(kind="none"),
            support=SupportReport(status="outside_empirical_support", query_region={}),
            identification=view,
            diagnostics=(detail,),
            transport=section,
        )
        return _attach(result, study, None, section)
    result = AnalysisResult(
        identification=view,
        estimate=EstimateView(
            ate=None,
            se_analytic=0.0,
            se_bootstrap=None,
            estimator_id="transport.empirical_table",
            method="unavailable",
        ),
        posterior=None,
        validation=_empty_validation(),
        performance=_empty_performance(),
        diagnostics=[detail],
        provenance={"operation_ids": ["identify.transport_sid"]},
        transport=section,
    )
    return _attach(result, study, None, section)


def _bindings(stage: Mapping[str, Any] | None) -> tuple[str, ...]:
    catalog = None if stage is None else stage.get("catalog")
    if catalog is None:
        return ()
    return tuple(f"{binding.regime}:{binding.snapshot_identity}" for binding in catalog.bindings)


def _replicates(specialist: Any) -> tuple[int, ...]:
    uncertainty = getattr(specialist, "uncertainty", None)
    if uncertainty is not None and getattr(uncertainty, "replicate_ids", None):
        return tuple(uncertainty.replicate_ids)
    if isinstance(specialist, LearnedTrialEstimate):
        return tuple(index for index, _value in specialist.replicates)
    return ()


def wrap_transport_result(study: Any, specialist: Any) -> AnalysisResult | CausalResponseView:
    query = study._query
    if not isinstance(query, Transport):
        return specialist
    stage = getattr(study, "_transport_stage", None) or {}
    shape = stage.get("shape") or "scalar"
    provider = _provider_name(stage.get("provider"))
    identified = stage.get("identified")
    formula = getattr(specialist, "formula", None) or (
        None if identified is None else identified.formula
    )
    section = TransportSection(
        formula=formula,
        provider=provider,
        bindings=_bindings(stage),
        replicate_ids=_replicates(specialist),
        distribution=specialist,
        shape=shape,
    )
    view = _identification_view(stage, query)
    if isinstance(specialist, LearnedTrialEstimate):
        result = AnalysisResult(
            identification=view,
            estimate=EstimateView(
                ate=specialist.estimate,
                se_analytic=0.0,
                se_bootstrap=None,
                estimator_id="transport.trial_aipw",
                method="trial.aipw",
            ),
            posterior=None,
            validation=_empty_validation(),
            performance=_empty_performance(),
            diagnostics=[],
            provenance={"operation_ids": ["estimate.trial_to_target"]},
            transport=section,
        )
        return _attach(result, study, specialist, section)
    if isinstance(specialist, TransportResponseGrid):
        return _wrap_grid(study, query, specialist, section, view, stage)
    if isinstance(specialist, (ExactTransportDistribution, StatisticalTransportDistribution)):
        if shape == "contrast":
            return _unavailable_result(
                study,
                query,
                "AverageEffect contrast requires a two-point grid execution",
                shape="contrast",
                stage=stage,
            )
        outcome = query.question.outcome  # type: ignore[union-attr]
        value = specialist.mean(outcome)
        result = AnalysisResult(
            identification=view,
            estimate=EstimateView(
                ate=value,
                se_analytic=0.0,
                se_bootstrap=None,
                estimator_id=f"transport.{provider}",
                method="transport.plugin",
            ),
            posterior=None,
            validation=_empty_validation(),
            performance=_empty_performance(),
            diagnostics=[],
            provenance={"operation_ids": ["estimate.transport"]},
            transport=section,
        )
        return _attach(result, study, specialist, section)
    return specialist


def _wrap_grid(
    study: Any,
    query: Transport,
    grid: TransportResponseGrid,
    section: TransportSection,
    view: IdentificationView,
    stage: Mapping[str, Any],
) -> AnalysisResult | CausalResponseView:
    worlds = list(stage.get("worlds") or ())
    outcome = query.question.outcome  # type: ignore[union-attr]
    treatment = query.question.treatment  # type: ignore[union-attr]
    points: list[list[float]] = []
    values: list[list[float]] = []
    statuses: list[str] = []
    diagnostics: list[SupportDiagnostic] = []
    warnings: list[str] = []
    for index, row in enumerate(grid.points):
        status = row["status"]
        at = worlds[index] if index < len(worlds) else {}
        coordinate = float(at.get(treatment, index))
        if status != "available":
            statuses.append("outside_empirical_support")
            detail = str(row.get("detail") or status)
            diagnostics.append(
                SupportDiagnostic(id=f"grid:{index}", values=(coordinate,), detail=detail)
            )
            warnings.append(detail)
            continue
        points.append([coordinate])
        values.append([float(row["means"][outcome])])
        statuses.append("supported")
    if stage.get("shape") == "contrast":
        if len(values) != 2 or any(status != "supported" for status in statuses):
            detail = missing_evidence_detail(stage["identified"], stage["catalog"], query=query)
            return _unavailable_result(
                study,
                query,
                detail,
                shape="contrast",
                stage=stage,
            )
        contrast = grid.contrast(1, 0, outcome)
        result = AnalysisResult(
            identification=view,
            estimate=EstimateView(
                ate=float(contrast.estimate),
                se_analytic=0.0,
                se_bootstrap=None,
                estimator_id=f"transport.{section.provider}",
                method="transport.contrast",
            ),
            posterior=None,
            validation=_empty_validation(),
            performance=_empty_performance(),
            diagnostics=[],
            provenance={"operation_ids": ["estimate.transport"]},
            transport=section,
        )
        return _attach(result, study, grid, section)
    if not points:
        return _unavailable_result(
            study,
            query,
            warnings[0] if warnings else "every requested grid point is unbound",
            shape="grid",
            stage=stage,
        )
    support_status = (
        "supported"
        if all(item == "supported" for item in statuses)
        else "outside_empirical_support"
    )
    region = {treatment: (min(p[0] for p in points), max(p[0] for p in points))} if points else {}
    result = CausalResponseView(
        estimand=query.question,
        response=ResponseView(
            treatments=[treatment],
            outcomes=[outcome],
            points=points,
            values=values,
        ),
        estimate=[row[0] for row in values],
        uncertainty=ResponseUncertainty(kind="none"),
        support=SupportReport(
            status=support_status,
            query_region=region,
            diagnostics=diagnostics,
            warnings=warnings,
            point_status=tuple(statuses) if statuses else None,
        ),
        identification=view,
        transport=section,
    )
    return _attach(result, study, grid, section)


def unavailable_from_stage(study: Any) -> AnalysisResult | CausalResponseView:
    query = study._query
    stage = getattr(study, "_transport_stage", None) or {}
    identified = stage.get("identified")
    catalog = stage.get("catalog")
    shape = stage.get("shape") or "scalar"
    if identified is None:
        raise CausalUnsupportedError("transport study has no identification stage")
    if identified.outcome in {"identified", "missing_evidence"}:
        detail = missing_evidence_detail(identified, catalog, query=query)
    else:
        from ._day1 import not_certified_detail

        detail = not_certified_detail(identified, query=query)
    return _unavailable_result(study, query, detail, shape=shape, stage=stage)


def answer_from_transport(result: Any) -> Answer | None:
    section = getattr(result, "transport", None)
    if section is None or section.unavailable is None:
        return None
    return Answer("unavailable", detail=section.unavailable)
