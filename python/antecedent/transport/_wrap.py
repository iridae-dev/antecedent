"""Wrap T5–T9 specialist results as Analysis / CausalResponseView."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ..errors import CausalSerializationError, CausalUnsupportedError
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
from ._day1 import Evidence, Source, Transport, lower_question, missing_evidence_detail
from ._impl import (
    EmpiricalTable,
    ExactTransportDistribution,
    LearnedCategorical,
    LearnedTrialEstimate,
    StatisticalTransportDistribution,
    TransportResponseGrid,
    TrialAipw,
    _exact_distribution,
    _learned_trial,
    _response_grid,
    _statistical_distribution,
    consume_exact,
    consume_identification,
    consume_response_grid,
    consume_statistical,
)

#: Never a computed number: :func:`fmt_se` and the uncertainty-availability
#: check in ``results/_execution.py`` both treat a non-finite ``se_analytic``
#: as withheld, so this is how an unwritten analytic SE is spelled without
#: inventing ``0.0`` (a real, wrong, finite standard error).
_NO_ANALYTIC_SE = math.nan

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
    #: Native sampling interval for the reported number, when one was licensed
    #: (``LearnedTrialEstimate.interval``, a plug-in ``TransportContrast.interval``,
    #: or the matching entry in ``TransportUncertainty.mean_intervals``). ``None``
    #: means no interval was computed, not that it rounds to a point.
    interval: tuple[float, float] | None = None
    #: Why no interval exists (e.g. ``exact_supplied_law_no_sampling_uncertainty``),
    #: read from the native uncertainty/contrast reason; never invented.
    uncertainty_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "provider": self.provider,
            "bindings": list(self.bindings),
            "replicate_ids": list(self.replicate_ids),
            "unavailable": self.unavailable,
            "shape": self.shape,
            "interval": list(self.interval) if self.interval is not None else None,
            "uncertainty_reason": self.uncertainty_reason,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TransportSection:
        interval = raw.get("interval")
        return cls(
            formula=raw.get("formula"),
            provider=raw.get("provider"),
            bindings=tuple(raw.get("bindings") or ()),
            replicate_ids=tuple(raw.get("replicate_ids") or ()),
            unavailable=raw.get("unavailable"),
            shape=raw.get("shape"),
            interval=tuple(interval) if interval else None,
            uncertainty_reason=raw.get("uncertainty_reason"),
        )


def _provider_name(provider: object) -> str:
    if isinstance(provider, EmpiricalTable) or provider in (None, "plugin"):
        return "empirical_table"
    if isinstance(provider, LearnedCategorical):
        return "learned_categorical"
    if isinstance(provider, TrialAipw):
        return "trial_aipw"
    if isinstance(provider, str):
        # Already a resolved name (e.g. round-tripped through export/load).
        return provider
    return type(provider).__name__


def _identification_view(stage: Mapping[str, Any] | None, query: Transport) -> IdentificationView:
    identified = None if stage is None else stage.get("identified")
    method = "identify.transport_sid"
    if identified is None:
        # No native identification stage travelled with this result: never
        # default to identified when nothing is actually known. Fails closed,
        # the same way an unrecognized status string closes in ``_verdict.py``.
        return IdentificationView(
            status="NotIdentified",
            method=method,
            adjustment_set=[],
            assumption_count=0,
            derivation_step_count=0,
        )
    status = (
        "NonparametricallyIdentified" if identified.outcome == "identified" else "NotIdentified"
    )
    rules = list(identified.rules)
    return IdentificationView(
        status=status,
        method=method,
        adjustment_set=[],
        assumption_count=len(rules),
        derivation_step_count=len(rules),
    )


def _empty_validation() -> ValidationView:
    # Rust's aggregate rule never claims a pass when nothing ran
    # (``estimation.py``'s quoted rule: "never claim pass when nothing ran").
    return ValidationView(passed=False, ran=False, count=0)


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


class _RehydratedStudy:
    """Just enough of a study for :func:`wrap_transport_result` to rebuild a view.

    Carries a verified native identification (and, for a computed answer, the
    verified native specialist) recovered from an exported artifact — never a
    JSON-authored status. ``catalog`` stays ``None``: evidence bindings are
    lineage, not re-derivable from the exported artifact, and are restored
    from the plain-text metadata afterwards rather than re-fabricated here.
    """

    def __init__(self, query: Transport | None, stage: Mapping[str, Any]) -> None:
        self._query = query
        self._transport_stage = stage


def _identification_bytes(result: Any) -> bytes | None:
    study = getattr(result, "_prepared", None)
    stage = getattr(study, "_transport_stage", None) or {}
    identified = stage.get("identified")
    if identified is None:
        return None
    return bytes(identified._native.export())


def _specialist_bytes(section: TransportSection) -> bytes | None:
    distribution = section.distribution
    export = getattr(distribution, "export", None)
    if export is None:
        return None
    return bytes(export())


def _consume_specialist_artifact(encoded: bytes) -> Any:
    """Verify one of the native specialist exports and rebuild its display object.

    Dispatches on the artifact's own magic prefix — the same four kinds
    :func:`antecedent._workflow.load` already recognizes for a bare specialist
    export — so a transported result's numbers are always re-derived from a
    native consumer, never trusted from unauthenticated JSON.
    """
    if encoded.startswith(b"ANTECEDENT-LEARNED-TRIAL\x01"):
        from .. import _native

        native = _native.consume_learned_trial(encoded)
        return _learned_trial(native, native.last_result())
    if encoded.startswith(b"ANTECEDENT-EXACT-TRANSPORT\x01"):
        prepared = consume_exact(encoded)
        return _exact_distribution(prepared._native, prepared._native.last_result())
    if encoded.startswith(b"ANTECEDENT-STATISTICAL-TRANSPORT\x01"):
        prepared = consume_statistical(encoded)
        return _statistical_distribution(prepared._native, prepared._native.last_result())
    if encoded.startswith(b"ANTECEDENT-TRANSPORT-GRID\x01"):
        prepared = consume_response_grid(encoded)
        return _response_grid(prepared._native, prepared._native.last_result())
    raise CausalSerializationError("unrecognized transport specialist artifact")


def encode_transport_view(result: Any) -> bytes:
    """Export a transported result as its native artifacts plus lineage metadata.

    The identification certificate and the specialist's own native export are
    the sole authority for status and numbers on reload (see
    :func:`decode_transport_view`); everything else here (provider label,
    bindings, the query) is descriptive lineage that can be lost or rewritten
    without a tampered artifact ever being able to claim a stronger status
    than the one its native certificate carries.
    """
    section = result.transport
    query = getattr(result, "query", None)
    payload: dict[str, Any] = {
        "query": _query_to_dict(query) if isinstance(query, Transport) else None,
        "provider": section.provider,
        "bindings": list(section.bindings),
        "unavailable": section.unavailable,
    }
    identification_bytes = _identification_bytes(result)
    if identification_bytes is not None:
        payload["identification_artifact"] = base64.b64encode(identification_bytes).decode("ascii")
    specialist_bytes = _specialist_bytes(section)
    if specialist_bytes is not None:
        payload["specialist_artifact"] = base64.b64encode(specialist_bytes).decode("ascii")
    return VIEW_PREFIX + json.dumps(payload).encode()


def decode_transport_view(encoded: bytes) -> AnalysisResult | CausalResponseView:
    """Rebuild a transported result from its verified native artifacts.

    Status, identification and every reported number come from
    ``consume_identification``/the matching native specialist consumer — the
    same functions a live analysis uses — never from the JSON envelope
    directly, so editing the envelope by hand cannot upgrade what the result
    claims.
    """
    payload = json.loads(encoded[len(VIEW_PREFIX) :])
    query = _query_from_dict(payload.get("query"))
    identification_artifact = payload.get("identification_artifact")
    identified = (
        consume_identification(base64.b64decode(identification_artifact))
        if identification_artifact is not None
        else None
    )
    shape, worlds = lower_question(query.question) if query is not None else (None, ())
    stage = {
        "identified": identified,
        "catalog": None,
        "bound": None,
        "shape": shape,
        "worlds": worlds,
        "provider": payload.get("provider"),
        "graph": None,
    }
    study = _RehydratedStudy(query, stage)
    specialist_artifact = payload.get("specialist_artifact")
    if specialist_artifact is not None:
        specialist = _consume_specialist_artifact(base64.b64decode(specialist_artifact))
        result = wrap_transport_result(study, specialist)
    else:
        detail = payload.get("unavailable") or (
            "no evidence travelled with this artifact; identification status alone is verified"
        )
        result = _unavailable_result(study, query, detail, shape=shape or "scalar", stage=stage)
    bindings = payload.get("bindings")
    if bindings:
        object.__setattr__(result, "transport", replace(result.transport, bindings=tuple(bindings)))
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
            se_analytic=_NO_ANALYTIC_SE,
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
        section = replace(
            section, interval=specialist.interval, uncertainty_reason=specialist.uncertainty_reason
        )
        result = AnalysisResult(
            identification=view,
            estimate=EstimateView(
                ate=specialist.estimate,
                se_analytic=_NO_ANALYTIC_SE,
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
        interval, uncertainty_reason = _plugin_uncertainty(specialist, outcome)
        section = replace(section, interval=interval, uncertainty_reason=uncertainty_reason)
        result = AnalysisResult(
            identification=view,
            estimate=EstimateView(
                ate=value,
                se_analytic=_NO_ANALYTIC_SE,
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


def _plugin_uncertainty(
    specialist: Any, outcome: str
) -> tuple[tuple[float, float] | None, str | None]:
    """Native interval for one outcome mean, or the native reason it is withheld.

    ``ExactTransportDistribution.uncertainty`` is always ``None`` (a complete
    law makes no sampling claim); a ``StatisticalTransportDistribution`` may
    withhold uncertainty for reasons of its own (``uncertainty.reason``).
    """
    uncertainty = getattr(specialist, "uncertainty", None)
    if uncertainty is None:
        return None, "exact_supplied_law_no_sampling_uncertainty"
    for name, lower, upper in uncertainty.mean_intervals or ():
        if name == outcome:
            return (lower, upper), uncertainty.reason
    return None, uncertainty.reason


def _contrast_unavailable_detail(
    stage: Mapping[str, Any], query: Transport, warnings: list[str]
) -> str:
    """Explain an unsupported contrast without ever crashing on the way there.

    ``missing_evidence_detail`` re-runs the bounded catalog search when it is
    not given one; that search can itself fail (the same budget/cancellation
    outcomes handled in ``identification_from_transport``), which must
    degrade to the native per-point detail rather than raise past this
    diagnostic and hide the real (already-known) reason the contrast is
    unavailable.
    """
    identified = stage.get("identified")
    catalog = stage.get("catalog")
    if identified is not None and catalog is not None:
        try:
            return missing_evidence_detail(identified, catalog, query=query)
        except Exception:
            pass
    return warnings[0] if warnings else "every requested grid point is unbound"


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
            detail = _contrast_unavailable_detail(stage, query, warnings)
            return _unavailable_result(
                study,
                query,
                detail,
                shape="contrast",
                stage=stage,
            )
        contrast = grid.contrast(1, 0, outcome)
        section = replace(section, interval=contrast.interval, uncertainty_reason=contrast.reason)
        result = AnalysisResult(
            identification=view,
            estimate=EstimateView(
                ate=float(contrast.estimate),
                se_analytic=_NO_ANALYTIC_SE,
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
