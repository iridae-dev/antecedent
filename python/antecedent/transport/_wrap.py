"""Wrap T5–T9 specialist results as Analysis / CausalResponseView."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..errors import CausalSerializationError, CausalUnsupportedError
from ..results import (
    AnalysisResult,
    CausalResponseView,
    EstimateView,
    IdentificationView,
    PerformanceView,
    ReasoningSlots,
    ResponseUncertainty,
    ResponseView,
    SlotView,
    SupportDiagnostic,
    SupportReport,
    ValidationView,
)
from ..results._execution import Answer
from ._day1 import (
    Evidence,
    Source,
    Transport,
    _inner_phrase,
    lower_question,
    missing_evidence_detail,
    not_certified_detail,
    transport_stage,
)
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
from ._restricted import (
    SCOPE as Z_SCOPE,
)
from ._restricted import (
    RestrictedTransportExecution,
    RestrictedTransportIdentification,
    consume_restricted_artifacts,
    identification_from_snapshot,
)

#: Never a computed number: :func:`fmt_se` and the uncertainty-availability
#: check in ``results/_execution.py`` both treat a non-finite ``se_analytic``
#: as withheld, so this is how an unwritten analytic SE is spelled without
#: inventing ``0.0`` (a real, wrong, finite standard error).
_NO_ANALYTIC_SE = math.nan

VIEW_PREFIX = b"ANTECEDENT-TRANSPORT-VIEW\x01"
Z_TRANSPORT_PREFIX = b"ANTECEDENT-Z-TRANSPORT\x01"


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


def _is_restricted(obj: Any) -> bool:
    return getattr(obj, "scope", None) == Z_SCOPE


def _identification_bytes(stage: Mapping[str, Any]) -> bytes | None:
    """The classical identification certificate, when a native one travelled."""
    identified = stage.get("identified")
    if identified is None or getattr(identified, "_native", None) is None:
        return None
    return bytes(identified._native.export())


def _identification_snapshot(stage: Mapping[str, Any]) -> bytes | None:
    """The z-transport failure snapshot for a non-identified restricted decision.

    An identified restricted result needs no separate certificate: each
    z-transport artifact embeds the checked proof, and consuming it rechecks
    that proof.
    """
    identified = stage.get("identified")
    catalog = stage.get("catalog")
    if (
        not isinstance(identified, RestrictedTransportIdentification)
        or identified.outcome == "identified"
        or identified.stage is None
        or catalog is None
    ):
        return None
    return bytes(identified.stage.failure_snapshot(catalog))


def _specialist_artifacts(distribution: Any) -> list[bytes]:
    if distribution is None:
        return []
    artifacts = getattr(distribution, "artifacts", None)
    if artifacts is not None:
        return [bytes(item) for item in artifacts]
    export = getattr(distribution, "export", None)
    if export is None:
        return []
    return [bytes(export())]


def _consume_specialist_artifact(encoded: bytes) -> Any:
    """Verify one of the native specialist exports and rebuild its display object.

    Dispatches on the artifact's own magic prefix — the same kinds
    :func:`antecedent._workflow.load` already recognizes for a bare specialist
    export — so a transported result's numbers are always re-derived from a
    native consumer, never trusted from unauthenticated JSON.
    """
    if encoded.startswith(b"ANTECEDENT-LEARNED-TRIAL\x01"):
        from .. import _native

        native = _native.consume_learned_trial(encoded)
        return _learned_trial(native, native.last_result())
    if encoded.startswith(b"ANTECEDENT-EXACT-TRANSPORT\x01"):
        exact_prepared = consume_exact(encoded)
        return _exact_distribution(exact_prepared._native, exact_prepared._native.last_result())
    if encoded.startswith(b"ANTECEDENT-STATISTICAL-TRANSPORT\x01"):
        statistical_prepared = consume_statistical(encoded)
        return _statistical_distribution(
            statistical_prepared._native, statistical_prepared._native.last_result()
        )
    if encoded.startswith(b"ANTECEDENT-TRANSPORT-GRID\x01"):
        grid_prepared = consume_response_grid(encoded)
        return _response_grid(grid_prepared._native, grid_prepared._native.last_result())
    if encoded.startswith(Z_TRANSPORT_PREFIX):
        return consume_restricted_artifacts([encoded])
    raise CausalSerializationError("unrecognized transport specialist artifact")


def encode_envelope(
    *,
    query: Transport | None,
    provider: str | None,
    bindings: Sequence[str],
    unavailable: str | None,
    identification_bytes: bytes | None,
    identification_snapshot: bytes | None,
    specialist_artifacts: Sequence[bytes],
) -> bytes:
    """The transport-view envelope: native artifacts plus descriptive lineage."""
    payload: dict[str, Any] = {
        "query": _query_to_dict(query) if isinstance(query, Transport) else None,
        "provider": provider,
        "bindings": list(bindings),
        "unavailable": unavailable,
    }
    if identification_bytes is not None:
        payload["identification_artifact"] = base64.b64encode(identification_bytes).decode("ascii")
    if identification_snapshot is not None:
        payload["identification_snapshot"] = base64.b64encode(identification_snapshot).decode(
            "ascii"
        )
    encoded = [base64.b64encode(item).decode("ascii") for item in specialist_artifacts]
    if len(encoded) == 1:
        payload["specialist_artifact"] = encoded[0]
    elif encoded:
        payload["specialist_artifacts"] = encoded
    return VIEW_PREFIX + json.dumps(payload).encode()


def encode_transport_view(result: Any) -> bytes:
    """Export a transported result as its native artifacts plus lineage metadata.

    The identification certificate (or z-transport failure snapshot) and the
    specialist's own native exports are the sole authority for status and
    numbers on reload (see :func:`decode_transport_view`); everything else
    here (provider label, bindings, the query) is descriptive lineage that can
    be lost or rewritten without a tampered artifact ever being able to claim
    a stronger status than the one its native certificate carries. A
    restricted grid carries one z-transport artifact per target assignment.
    """
    section = result.transport
    stage = getattr(getattr(result, "_prepared", None), "_transport_stage", None) or {}
    return encode_envelope(
        query=getattr(result, "query", None),
        provider=section.provider,
        bindings=section.bindings,
        unavailable=section.unavailable,
        identification_bytes=_identification_bytes(stage),
        identification_snapshot=_identification_snapshot(stage),
        specialist_artifacts=_specialist_artifacts(section.distribution),
    )


def encode_restricted_execution(study: Any, execution: RestrictedTransportExecution) -> bytes:
    """The loadable view of one restricted execution, from its prepared study."""
    stage = getattr(study, "_transport_stage", None) or {}
    return encode_envelope(
        query=getattr(study, "_query", None),
        provider=_provider_name(stage.get("provider")),
        bindings=_bindings(stage),
        unavailable=None,
        identification_bytes=None,
        identification_snapshot=None,
        specialist_artifacts=_specialist_artifacts(execution),
    )


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
    identification_snapshot = payload.get("identification_snapshot")
    identified: Any = None
    if identification_artifact is not None:
        identified = consume_identification(base64.b64decode(identification_artifact))
    elif identification_snapshot is not None:
        identified = identification_from_snapshot(base64.b64decode(identification_snapshot))
    shape, worlds = lower_question(query.question) if query is not None else (None, [])
    artifacts = [
        base64.b64decode(item)
        for item in (
            [payload["specialist_artifact"]]
            if payload.get("specialist_artifact") is not None
            else payload.get("specialist_artifacts") or []
        )
    ]
    specialist: Any = None
    if artifacts and all(item.startswith(Z_TRANSPORT_PREFIX) for item in artifacts):
        specialist = consume_restricted_artifacts(
            artifacts, worlds if len(worlds) == len(artifacts) else ()
        )
        identified = RestrictedTransportIdentification(
            outcome="identified",
            reason=None,
            rules=specialist.rules,
            formula=Z_SCOPE,
            source=None,
        )
    elif len(artifacts) == 1:
        specialist = _consume_specialist_artifact(artifacts[0])
    elif artifacts:
        raise CausalSerializationError(
            "a transported view carries several artifacts only for a z-transport grid"
        )
    stage = transport_stage(
        identified=identified,
        catalog=None,
        bound=None,
        shape=shape,
        worlds=worlds,
        provider=payload.get("provider"),
        graph=None,
    )
    study = _RehydratedStudy(query, stage)
    if specialist is not None:
        result = wrap_transport_result(study, specialist)
    else:
        if not isinstance(query, Transport):
            raise CausalSerializationError(
                "transported view artifact is missing its Transport query"
            )
        detail = payload.get("unavailable") or (
            "no evidence travelled with this artifact; identification status alone is verified"
        )
        result = _unavailable_result(study, query, detail, shape=shape or "scalar", stage=stage)
    bindings = payload.get("bindings")
    if bindings:
        object.__setattr__(result, "transport", replace(result.transport, bindings=tuple(bindings)))
    return result


def _scalar_result(
    study: Any,
    specialist: Any,
    section: TransportSection,
    view: IdentificationView,
    *,
    ate: float | None,
    estimator_id: str,
    method: str,
    operation: str,
    diagnostics: Sequence[str] = (),
) -> AnalysisResult:
    """The one scalar ``AnalysisResult`` scaffold behind every transported number."""
    result = AnalysisResult(
        identification=view,
        estimate=EstimateView(
            ate=ate,
            se_analytic=_NO_ANALYTIC_SE,
            se_bootstrap=None,
            estimator_id=estimator_id,
            method=method,
        ),
        posterior=None,
        validation=_empty_validation(),
        performance=_empty_performance(),
        diagnostics=list(diagnostics),
        provenance={"operation_ids": [operation]},
        transport=section,
        reasoning=_reasoning_from_specialist(specialist),
    )
    return _attach(result, study, specialist, section)


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
        grid_result = CausalResponseView(
            estimand=query.question,
            response=None,
            estimate=None,
            uncertainty=ResponseUncertainty(kind="none"),
            support=SupportReport(status="outside_empirical_support", query_region={}),
            identification=view,
            diagnostics=(detail,),
            transport=section,
        )
        return _attach(grid_result, study, None, section)
    return _scalar_result(
        study,
        None,
        section,
        view,
        ate=None,
        estimator_id="transport.empirical_table",
        method="unavailable",
        operation="identify.transport_sid",
        diagnostics=[detail],
    )


def _reasoning_from_specialist(specialist: Any) -> ReasoningSlots | None:
    """Four-slot reasoning read from the specialist's own native ``inspect()``.

    ``AnalysisResult``/``CausalResponseView`` default their ``reasoning`` to
    ``None``, which ``ExecutionResult.inspect()`` then treats as an empty
    contract (every slot ``unavailable:missing``). The specialist distribution
    already carries a native four-slot report (identification, support,
    uncertainty, assumptions); this copies it over so the outer wrapped result
    reports the same slots instead of a manufactured "missing" one.
    """
    inspect = getattr(specialist, "inspect", None)
    if specialist is None or inspect is None:
        return None
    report = inspect()

    def slot(name: str) -> SlotView:
        model = getattr(report, name)
        return SlotView(model.available, model.reason, model.summary, dict(model.payload))

    return ReasoningSlots(
        identification=slot("identification"),
        support=slot("support"),
        uncertainty=slot("uncertainty"),
        assumptions=slot("assumptions"),
        program_id=getattr(report, "program_id", None),
        claim_id=getattr(report, "claim_id", None),
        target_id=getattr(report, "target_id", None),
        identification_id=getattr(report, "identification_id", None),
        identification_product_id=getattr(report, "identification_product_id", None),
        inference_binding_id=getattr(report, "inference_binding_id", None),
        observation_id=getattr(report, "observation_id", None),
        data_snapshot_id=getattr(report, "data_snapshot_id", None),
        execution_id=getattr(report, "execution_id", None),
        score_reuse_id=getattr(report, "score_reuse_id", None),
        target_weights_id=getattr(report, "target_weights_id", None),
        contract=dict(report.contract)
        if isinstance(getattr(report, "contract", None), Mapping)
        else None,
    )


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


@dataclass(frozen=True, slots=True)
class _GridRow:
    """One grid point as the view builder sees it: a coordinate and a mean, or a reason."""

    coordinate: float
    mean: float | None
    detail: str | None = None


def _grid_view(
    study: Any,
    query: Transport,
    specialist: Any,
    section: TransportSection,
    view: IdentificationView,
    stage: Mapping[str, Any],
    rows: Sequence[_GridRow],
) -> AnalysisResult | CausalResponseView:
    """The one ``CausalResponseView`` scaffold behind every transported grid."""
    treatment = query.question.treatment  # type: ignore[union-attr]
    outcome = query.question.outcome  # type: ignore[union-attr]
    points: list[list[float]] = []
    values: list[list[float]] = []
    statuses: list[str] = []
    diagnostics: list[SupportDiagnostic] = []
    warnings: list[str] = []
    for index, row in enumerate(rows):
        if row.mean is None:
            detail = row.detail or "unavailable"
            statuses.append("outside_empirical_support")
            diagnostics.append(
                SupportDiagnostic(id=f"grid:{index}", values=(row.coordinate,), detail=detail)
            )
            warnings.append(detail)
            continue
        points.append([row.coordinate])
        values.append([row.mean])
        statuses.append("supported")
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
    region = {treatment: (min(p[0] for p in points), max(p[0] for p in points))}
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
            point_status=tuple(statuses),
        ),
        identification=view,
        transport=section,
        reasoning=_reasoning_from_specialist(specialist),
    )
    return _attach(result, study, specialist, section)


def _wrap_restricted(
    study: Any,
    query: Transport,
    specialist: RestrictedTransportExecution,
    section: TransportSection,
    view: IdentificationView,
    stage: Mapping[str, Any],
) -> AnalysisResult | CausalResponseView:
    outcome = query.question.outcome  # type: ignore[union-attr]
    treatment = query.question.treatment  # type: ignore[union-attr]
    shape = stage.get("shape") or "scalar"
    points = list(specialist.points)
    if shape == "contrast":
        if len(points) != 2 or any(point.status != "available" for point in points):
            return _unavailable_result(
                study,
                query,
                "restricted-experiment contrast needs both treatment levels",
                shape="contrast",
                stage=stage,
            )
        estimate = float(points[1].means[outcome]) - float(points[0].means[outcome])
        section = replace(section, interval=None, uncertainty_reason="no_interval_reported")
        return _scalar_result(
            study,
            specialist,
            section,
            view,
            ate=estimate,
            estimator_id=f"transport.{section.provider}",
            method="transport.contrast",
            operation="estimate.transport",
        )
    if shape == "scalar":
        interval, uncertainty_reason = _plugin_uncertainty(specialist, outcome)
        section = replace(section, interval=interval, uncertainty_reason=uncertainty_reason)
        return _scalar_result(
            study,
            specialist,
            section,
            view,
            ate=specialist.mean(outcome),
            estimator_id=f"transport.{section.provider}",
            method="transport.plugin",
            operation="estimate.transport",
        )
    rows = [
        _GridRow(
            coordinate=float(point.at.get(treatment, index)),
            mean=float(point.means[outcome]) if point.status == "available" else None,
            detail=None if point.status == "available" else point.status,
        )
        for index, point in enumerate(points)
    ]
    return _grid_view(study, query, specialist, section, view, stage, rows)


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
    if isinstance(specialist, RestrictedTransportExecution):
        return _wrap_restricted(study, query, specialist, section, view, stage)
    if isinstance(specialist, LearnedTrialEstimate):
        section = replace(
            section, interval=specialist.interval, uncertainty_reason=specialist.uncertainty_reason
        )
        return _scalar_result(
            study,
            specialist,
            section,
            view,
            ate=specialist.estimate,
            estimator_id="transport.trial_aipw",
            method="trial.aipw",
            operation="estimate.trial_to_target",
        )
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
        interval, uncertainty_reason = _plugin_uncertainty(specialist, outcome)
        section = replace(section, interval=interval, uncertainty_reason=uncertainty_reason)
        return _scalar_result(
            study,
            specialist,
            section,
            view,
            ate=specialist.mean(outcome),
            estimator_id=f"transport.{provider}",
            method="transport.plugin",
            operation="estimate.transport",
        )
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
    rows = []
    for index, row in enumerate(grid.points):
        at = worlds[index] if index < len(worlds) else {}
        available = row["status"] == "available"
        rows.append(
            _GridRow(
                coordinate=float(at.get(treatment, index)),
                mean=float(row["means"][outcome]) if available else None,
                detail=None if available else str(row.get("detail") or row["status"]),
            )
        )
    if stage.get("shape") == "contrast":
        warnings = [row.detail for row in rows if row.detail is not None]
        if len(rows) != 2 or warnings:
            detail = _contrast_unavailable_detail(stage, query, warnings)
            return _unavailable_result(study, query, detail, shape="contrast", stage=stage)
        contrast = grid.contrast(1, 0, outcome)
        section = replace(section, interval=contrast.interval, uncertainty_reason=contrast.reason)
        return _scalar_result(
            study,
            grid,
            section,
            view,
            ate=float(contrast.estimate),
            estimator_id=f"transport.{section.provider}",
            method="transport.contrast",
            operation="estimate.transport",
        )
    return _grid_view(study, query, grid, section, view, stage, rows)


def unavailable_from_stage(study: Any) -> AnalysisResult | CausalResponseView:
    query = study._query
    stage = getattr(study, "_transport_stage", None) or {}
    identified = stage.get("identified")
    catalog = stage.get("catalog")
    shape = stage.get("shape") or "scalar"
    if identified is None:
        raise CausalUnsupportedError("transport study has no identification stage")
    if _is_restricted(identified):
        if identified.outcome == "missing_evidence":
            detail = identified.detail or (
                f"{_inner_phrase(query)} is identified for {query.target}, "
                "but a cited joint is unbound."
            )
        else:
            detail = not_certified_detail(identified, query=query)
        return _unavailable_result(study, query, detail, shape=shape, stage=stage)
    if identified.outcome in {"identified", "missing_evidence"}:
        if catalog is None:
            raise CausalUnsupportedError("transport study has no bound evidence catalog")
        detail = missing_evidence_detail(identified, catalog, query=query)
    else:
        detail = not_certified_detail(identified, query=query)
    return _unavailable_result(study, query, detail, shape=shape, stage=stage)


def answer_from_transport(result: Any) -> Answer | None:
    section = getattr(result, "transport", None)
    if section is None or section.unavailable is None:
        return None
    return Answer("unavailable", detail=section.unavailable)
