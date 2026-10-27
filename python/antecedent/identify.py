"""Data-free identification followed by immutable preparation and execution.

``identify()`` retains the resolved query, graph, strategy and available
certificate. ``Identification.prepare(data)`` enters the canonical prepared
lifecycle; ``Identification.estimate(data)`` prepares and executes that same
lifecycle. Native preparation checks identification again because the identify-only
certificate is not a retained executable native proof.

This module wraps the existing identify-only native implementation in
``estimation``. It also projects legacy identify-only and completed-analysis
views without introducing an independent identification algorithm.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from ._claim import identification_statement
from ._freeze import thaw
from ._prepared_identification import PreparedIdentification
from ._verdict import describe_status, verdict_for
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .estimation import IdentifyResult
from .estimation import identify as _identify_native
from .execution import Execution
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag, TieredBackground
from .ids import Estimator, Identifier, Refute
from .inference import Bayesian, Frequentist
from .query import (
    AverageDerivative,
    AverageEffect,
    ConditionalEffect,
    Counterfactual,
    DirectionalDerivative,
    Elasticity,
    InterventionResponse,
    MediationEffect,
    NestedCounterfactual,
    PointDerivative,
    PulseEffect,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
    SustainedEffect,
    TemporalMediationEffect,
)
from .results import IdentificationView

if TYPE_CHECKING:
    from .outcomes import Outcome
    from .prepared import PreparedAnalysis


class _DefaultRefute(Enum):
    DEFAULT = "default"


def _assumption_line(item: object) -> str | None:
    if isinstance(item, str) and item.strip():
        return item.strip()
    if not isinstance(item, Mapping):
        return None
    tag = item.get("assumption", item.get("id", item.get("kind")))
    if isinstance(tag, Mapping):
        description = tag.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
        tag = tag.get("id") or tag.get("kind") or next(iter(tag), None)
    if tag is None:
        return None
    label = str(tag).replace("_", " ")
    extras = [
        str(item[key])
        for key in ("source", "status")
        if isinstance(item.get(key), str) and item[key]
    ]
    return f"{label} ({', '.join(extras)})" if extras else label


def _certificate_cases(certificate: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not certificate:
        return []
    cases = certificate.get("cases")
    if not isinstance(cases, (list, tuple)):
        return []
    return [case for case in cases if isinstance(case, Mapping)]


def _assumption_statements(certificate: Mapping[str, Any] | None) -> tuple[str, ...]:
    seen: list[str] = []
    for case in _certificate_cases(certificate):
        identification = case.get("identification")
        raw = (
            identification.get("required_assumptions")
            if isinstance(identification, Mapping)
            else None
        )
        entries = raw.get("entries", raw) if isinstance(raw, Mapping) else raw
        if not isinstance(entries, (list, tuple)):
            continue
        for item in entries:
            line = _assumption_line(item)
            if line and line not in seen:
                seen.append(line)
    return tuple(seen)


def _derivation_statements(certificate: Mapping[str, Any] | None) -> tuple[str, ...]:
    seen: list[str] = []
    for case in _certificate_cases(certificate):
        identification = case.get("identification")
        steps = identification.get("derivation") if isinstance(identification, Mapping) else None
        if not isinstance(steps, (list, tuple)):
            continue
        for step in steps:
            if isinstance(step, str) and step.strip() and step not in seen:
                seen.append(step.strip())
                continue
            if not isinstance(step, Mapping):
                continue
            detail = step.get("detail") or step.get("rule")
            if isinstance(detail, str) and detail.strip() and detail not in seen:
                seen.append(detail.strip())
    return tuple(seen)


@dataclass(frozen=True)
class Identification:
    """A resolved identification strategy, staged for ``.prepare()`` / ``.estimate()``.

    Produced by :func:`identify` or :meth:`from_view`. Typed-structure
    certificates retain each case's assumptions, derivation and search diagnostics;
    aggregate counts remain unset when there is no single point certificate.
    :attr:`statement`, :attr:`verdict`, :attr:`assumption_statements`, and
    :attr:`derivation_statements` are human-readable state on this object —
    not a notebook renderer. Conceptually immutable, like
    :class:`antecedent.AcceptedGraph`.
    """

    status: str
    method: str
    adjustment_set: Sequence[str]
    graph: (
        Dag
        | Admg
        | Cpdag
        | Pag
        | TemporalDag
        | TemporalCpdag
        | TemporalPag
        | TieredBackground
        | Sequence[tuple[str, str]]
    )
    query: (
        AverageEffect
        | ResponseCurve
        | InterventionResponse
        | ConditionalEffect
        | PulseEffect
        | SustainedEffect
        | PointDerivative
        | Elasticity
        | SemiElasticity
        | AverageDerivative
        | DirectionalDerivative
        | ResponseJacobian
        | MediationEffect
        | NestedCounterfactual
        | TemporalMediationEffect
        | Counterfactual
        | Any
    )
    names: Sequence[str] | None = None
    identifier: str | None = None
    assumption_count: int | None = None
    derivation_step_count: int | None = None
    # Complete native certificate: per-completion wire records, coordinates,
    # weights, derivations, assumptions, expression arenas, and search limits.
    certificate: Mapping[str, Any] | None = None
    design: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("adjustment_set", "names", "certificate", "graph", "query", "design"):
            from ._study_lifecycle import retain

            object.__setattr__(self, name, retain(getattr(self, name)))

    @classmethod
    def from_prepared(cls, payload: Mapping[str, Any], *, query: Any, graph: Any) -> Identification:
        names = tuple(payload.get("names", ()))
        cases = payload.get("cases", ())
        first = cases[0].get("identification", {}) if cases else {}
        estimands = first.get("estimands", ())
        adjustment = estimands[0].get("adjustment_set", ()) if estimands else ()

        def variable_name(value: Any) -> str:
            index = value if isinstance(value, int) else value.get("raw", value.get("variable", 0))
            return names[index] if 0 <= index < len(names) else str(index)

        design = not cases and (
            payload.get("graph_class") in ("randomized_trial", "RandomizedTrial")
            or type(query).__name__ == "InterferenceQuery"
        )
        import re

        from ._verdict import VERDICTS

        statuses = {re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower(): key for key in VERDICTS}
        status = statuses.get(str(payload.get("status")), payload.get("status"))
        method = payload.get("method") or (estimands[0].get("method") if estimands else None)
        return cls(
            status=status or ("DesignBased" if design else "NotIdentified"),
            method=method or ("design_based" if design else "unavailable"),
            adjustment_set=[variable_name(v) for v in adjustment],
            graph=graph,
            query=query,
            names=list(names),
            identifier=payload.get("method"),
            assumption_count=len(first.get("required_assumptions", ())) if cases else None,
            derivation_step_count=len(first.get("derivation", ())) if cases else None,
            certificate=None if design else dict(payload),
            design={
                "basis": "declared_design",
                "query_family": type(query).__name__,
                "outcome": getattr(
                    query, "outcome", getattr(getattr(query, "contrast", None), "outcome", None)
                ),
                "reasoning": payload.get("reasoning", {}),
                "limitations": (
                    "Design declaration does not establish a graph certificate or add an estimation or inference license.",
                ),
            }
            if design
            else None,
        )

    @property
    def scenario_adjustment_sets(self) -> tuple[tuple[str, ...], ...]:
        """Named retained adjustment scenarios; these are not exhaustive graph-class bounds."""
        return tuple(
            tuple(coordinate["name"] for coordinate in coordinates)
            for case in _certificate_cases(self.certificate)
            for coordinates in case.get("adjustment_coordinates", ())
        )

    @property
    def completion_keys(self) -> list[int]:
        """Stable temporal-class completion fingerprints for :class:`ClassPrior`."""
        certificate = self.certificate or {}
        keys = certificate.get("completion_keys")
        if keys:
            return [int(key) for key in keys]
        return [
            int(case["fingerprint"])
            for case in certificate.get("cases", [])
            if "fingerprint" in case
        ]

    def __bool__(self) -> bool:
        """``True`` when the single verdict table (:mod:`antecedent._verdict`) reports identified."""
        return self.verdict == "identified"

    @property
    def verdict(self) -> str:
        """Stable human label: identified, not identified, partial, or graph-dependent."""
        return "design based" if self.design is not None else verdict_for(self.status)

    @property
    def qualified_verdict(self) -> str:
        """:attr:`verdict` plus the restriction it holds under, when there is one.

        ``IdentifiedUnderParametricRestrictions`` reads
        ``"identified under parametric restrictions"``; priors never upgrade
        identification, so ``IdentifiedUnderPriorRestrictions`` keeps its
        qualifier too.
        """
        return "design based" if self.design is not None else describe_status(self.status)

    @property
    def assumption_statements(self) -> tuple[str, ...]:
        """Readable assumption lines retained from the identification certificate."""
        return (
            ("The declared design and its required scientific checks govern this analysis.",)
            if self.design is not None
            else _assumption_statements(self.certificate)
        )

    @property
    def derivation_statements(self) -> tuple[str, ...]:
        """Readable derivation steps retained from the identification certificate."""
        return _derivation_statements(self.certificate)

    def claim(self) -> str:
        """Same sentence as :attr:`statement` — the claim before estimation."""
        return self.statement

    @property
    def statement(self) -> str:
        """One-sentence identification state. This is data, not a display hook."""
        return (
            f"{type(self.query).__name__} for {self.design.get('outcome') or 'the declared quantity'} uses the declared design; scientific standing and inference licenses are reported by estimation."
            if self.design is not None
            else identification_statement(self.query, self.status, self.method, self.adjustment_set)
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe identification state, including the human-readable fields."""
        return {
            "status": self.status,
            "verdict": self.verdict,
            "qualified_verdict": self.qualified_verdict,
            "statement": self.statement,
            "method": self.method,
            "adjustment_set": list(self.adjustment_set),
            "identifier": self.identifier,
            "assumption_count": self.assumption_count,
            "derivation_step_count": self.derivation_step_count,
            "assumption_statements": list(self.assumption_statements),
            "derivation_statements": list(self.derivation_statements),
            "certificate": thaw(self.certificate),
            "design": thaw(self.design),
        }

    def __repr__(self) -> str:
        return f"<Identification {self.statement}>"

    def _repr_html_(self) -> str:
        from .results._html import _identification_repr_html

        return (
            self.inspect()._repr_html_()
            if self.design is not None
            else _identification_repr_html(self)
        )

    def to_identify_result(self) -> IdentifyResult:
        """Convert down to the legacy identify-only result shape.

        ``IdentifyResult`` stays exported for callers that only want that
        narrower shape; this keeps it a projection of ``Identification``
        rather than a second, independently-computed value.
        """
        return IdentifyResult(
            status=self.status,
            method=self.method,
            adjustment_set=list(self.adjustment_set),
            query=self.query,
        )

    @classmethod
    def from_view(
        cls,
        view: IdentificationView,
        *,
        graph: Dag | Admg | Sequence[tuple[str, str]],
        query: AverageEffect,
        names: Sequence[str] | None = None,
        identifier: str | None = None,
    ) -> Identification:
        """Build from a completed analysis's identification section.

        Unlike :func:`identify` alone, a view sourced from a full estimate
        carries ``assumption_count`` / ``derivation_step_count``.
        """
        return cls(
            status=view.status,
            method=view.method,
            adjustment_set=list(view.adjustment_set),
            graph=graph,
            query=query,
            names=list(names) if names is not None else None,
            identifier=identifier,
            assumption_count=view.assumption_count,
            derivation_step_count=view.derivation_step_count,
        )

    def inspect(self) -> Any:
        """Identification / support / assumptions slots. No provider access."""

        from ._claim import identification_statement
        from .results._report import InspectionReport, SlotModel
        from .transport import Transport

        if self.design is not None:
            unavailable = SlotModel(available=False, reason="not_executed", summary="not estimated")
            return InspectionReport(
                identification=SlotModel(
                    available=True, summary="design based", payload=self.design
                ),
                support=unavailable,
                uncertainty=unavailable,
                assumptions=SlotModel(
                    available=True,
                    summary="declared",
                    payload={"statements": self.assumption_statements},
                ),
                stage="prepared",
                family="DesignIdentification",
                details=self.to_dict(),
                operations=("inspect", "estimate"),
            )
        certificate = self.certificate or {}
        outcome = certificate.get("outcome") or self.status
        identified = self.verdict == "identified"
        missing = outcome == "missing_evidence" or certificate.get("missing_detail")
        support_detail = certificate.get("missing_detail")
        if missing and not support_detail:
            support_detail = (
                "The formula is identified, but a required joint is unbound; "
                "bind that snapshot before estimate."
            )
        not_certified = outcome in {"not_certified", "proven_non_transportable"}
        engine = certificate.get("engine") or {
            "formula": certificate.get("formula"),
            "catalog_search": certificate.get("catalog_search"),
            "rules": certificate.get("rules") or [],
        }
        if isinstance(self.query, Transport):
            statement = identification_statement(
                self.query, self.status, self.method, self.adjustment_set
            )
            return InspectionReport(
                identification=SlotModel(
                    available=identified,
                    reason=None if identified else str(outcome),
                    summary=statement,
                    payload={"outcome": outcome, "method": self.method, "engine": engine},
                ),
                support=SlotModel(
                    available=identified and not missing,
                    # ``outcome`` is exact (``missing_evidence`` or
                    # ``catalog_search_incomplete``): a stalled search is
                    # reported as unknown, never folded into "no evidence
                    # is missing".
                    reason=None if not missing else str(outcome),
                    summary="search_incomplete"
                    if outcome == "catalog_search_incomplete"
                    else "unbound_factor"
                    if missing
                    else "not_estimated"
                    if identified
                    else "unavailable",
                    payload={"detail": support_detail, "engine": engine},
                ),
                uncertainty=SlotModel(
                    available=False,
                    reason="not_estimated",
                    summary="unavailable",
                ),
                assumptions=SlotModel(
                    available=True,
                    summary="declared",
                    payload={
                        "statements": list(self.assumption_statements),
                        "not_certified": certificate.get("not_certified_detail")
                        if not_certified
                        else None,
                    },
                ),
            )
        return InspectionReport(
            identification=SlotModel(
                available=identified,
                reason=None if identified else str(outcome),
                summary=self.status,
                payload={
                    "outcome": outcome,
                    "formula": certificate.get("formula"),
                    "rules": certificate.get("rules") or [],
                    "method": self.method,
                },
            ),
            support=SlotModel(
                available=identified and not missing,
                reason=None if not missing else "missing_evidence",
                summary="unbound_factor"
                if missing
                else "not_estimated"
                if identified
                else "unavailable",
                payload={
                    "detail": support_detail,
                    "catalog_search": certificate.get("catalog_search"),
                },
            ),
            uncertainty=SlotModel(
                available=False,
                reason="not_estimated",
                summary="unavailable",
            ),
            assumptions=SlotModel(
                available=True,
                summary="declared",
                payload={
                    "statements": list(self.assumption_statements),
                    "not_certified": certificate.get("not_certified_detail")
                    if not_certified
                    else None,
                },
            ),
        )

    def prepare(
        self,
        data: Any,
        *,
        inference: Frequentist | Bayesian | None = None,
        execution: Execution | None = None,
        estimator: str | Estimator | None = None,
        estimator_config: Mapping[str, Any] | None = None,
        refute: (
            bool | Refute | Literal["full", "placebo", "none", "cheap"] | None | _DefaultRefute
        ) = _DefaultRefute.DEFAULT,
        class_prior: Any = None,
        max_completions: int | None = None,
        participant: str | None = None,
        missing: Any = None,
        population_registry: Any = None,
        validators: Sequence[Any] | Mapping[str, Any] | None = None,
        regimes: Sequence[int] | None = None,
        provider: Any = None,
        sampling_design: str | None = None,
        retain_joint_horizons: bool = False,
        on_refusal: Literal["refuse", "describe"] = "refuse",
        evidence_candidates: Sequence[Any] | None = None,
        outcome_units: str | None = None,
        quantity_population: str = "target",
        quantity_transform: str = "identity",
        dose_units: str | None = None,
    ) -> PreparedAnalysis[Any]:
        """Own data and prepare this strategy through the common lifecycle.

        The native preparation checks identification again: the identify-only
        certificate is retained on this stage, but is not a reusable native proof.
        Statistical budgets belong to ``inference``; execution settings belong
        to ``Execution``. Omitted refutation keeps the ordinary-stage default
        disabled and the transport-stage default unspecified; explicit options
        are validated by the common preparation path.
        """
        from .prepared import Reuse, prepare
        from .transport import Transport

        resolved_refute = refute
        if refute is _DefaultRefute.DEFAULT:
            resolved_refute = None if isinstance(self.query, Transport) else False
        graph = self.graph
        if self.names is not None and isinstance(graph, (list, tuple)):
            graph = Dag.from_edges(list(self.names), list(graph))
        study = prepare(
            data,
            query=self.query,
            graph=graph,
            identifier=None if isinstance(graph, TieredBackground) else self.identifier,
            inference=inference,
            execution=execution,
            estimator=estimator,
            estimator_config=estimator_config,
            refute=resolved_refute,
            class_prior=class_prior,
            max_completions=max_completions,
            participant=participant,
            missing=missing,
            population_registry=population_registry,
            validators=validators,
            regimes=regimes,
            provider=provider,
            sampling_design=sampling_design,
            retain_joint_horizons=retain_joint_horizons,
            on_refusal=on_refusal,
            evidence_candidates=evidence_candidates,
            outcome_units=outcome_units,
            quantity_population=quantity_population,
            quantity_transform=quantity_transform,
            dose_units=dose_units,
        )
        return replace(
            study,
            reuse=Reuse(
                retained=("declarations",),
                invalidated=("identification", "program", "fits", "scores"),
                preparation="reprepared",
            ),
        )

    def estimate(
        self,
        data: Any,
        *,
        inference: Frequentist | Bayesian | None = None,
        execution: Execution | None = None,
        estimator: str | Estimator | None = None,
        estimator_config: Mapping[str, Any] | None = None,
        refute: (
            bool | Refute | Literal["full", "placebo", "none", "cheap"] | None | _DefaultRefute
        ) = _DefaultRefute.DEFAULT,
        class_prior: Any = None,
        max_completions: int | None = None,
        participant: str | None = None,
        missing: Any = None,
        population_registry: Any = None,
        validators: Sequence[Any] | Mapping[str, Any] | None = None,
        regimes: Sequence[int] | None = None,
        provider: Any = None,
        sampling_design: str | None = None,
        retain_joint_horizons: bool = False,
        on_refusal: Literal["refuse", "describe"] = "refuse",
        evidence_candidates: Sequence[Any] | None = None,
        outcome_units: str | None = None,
        quantity_population: str = "target",
        quantity_transform: str = "identity",
        dose_units: str | None = None,
    ) -> Outcome[Any]:
        """Prepare and execute through the common immutable lifecycle."""
        from .prepared import estimate

        return estimate(
            self.prepare(
                data,
                inference=inference,
                execution=execution,
                estimator=estimator,
                estimator_config=estimator_config,
                refute=refute,
                class_prior=class_prior,
                max_completions=max_completions,
                participant=participant,
                missing=missing,
                population_registry=population_registry,
                validators=validators,
                regimes=regimes,
                provider=provider,
                sampling_design=sampling_design,
                retain_joint_horizons=retain_joint_horizons,
                on_refusal=on_refusal,
                evidence_candidates=evidence_candidates,
                outcome_units=outcome_units,
                quantity_population=quantity_population,
                quantity_transform=quantity_transform,
                dose_units=dose_units,
            )
        )


_STRUCTURE_GRAPHS = (Cpdag, Pag, TemporalDag, TemporalCpdag, TemporalPag)


def _uses_structure_identify(graph: object, query: object) -> bool:
    if isinstance(graph, _STRUCTURE_GRAPHS):
        return True
    if isinstance(graph, Dag) and isinstance(
        query, (ConditionalEffect, PulseEffect, SustainedEffect, InterventionResponse)
    ):
        return True
    return isinstance(query, (ResponseCurve, InterventionResponse)) and isinstance(
        graph, (Cpdag, Pag)
    )


class _IdentifyStructureKwargs(TypedDict, total=False):
    modifiers: list[str]
    policy: str
    treatment_lag: int
    horizon_steps: int
    active_level: float
    window: tuple[int, int] | None
    treatments: list[str]
    horizons: list[int]
    max_history_lag: int | None
    response_grid: list[float]
    response_steps: list[tuple[str, str, list[float]]]


def _identify_typed_graph(
    *,
    graph: Dag
    | Admg
    | Cpdag
    | Pag
    | TemporalDag
    | TemporalCpdag
    | TemporalPag
    | TieredBackground
    | Sequence[tuple[str, str]],
    supplied: object,
    query: object,
    identifier: str | None,
    names: Sequence[str] | None,
) -> Identification:
    from ._native import _identify_structure

    if not isinstance(supplied, (Dag, Cpdag, Pag, TemporalDag, TemporalCpdag, TemporalPag)):
        raise TypeError("typed-graph identify() requires a supported graph class")
    if isinstance(query, (ResponseCurve, InterventionResponse)):
        from .observation import Complete

        if (not query.is_temporal) and (
            query.observation is not None and not isinstance(query.observation, Complete)
        ):
            raise CausalUnsupportedError(
                "typed-graph response identification supports complete-observation static queries only"
            )
    if isinstance(query, AverageEffect):
        kind = "average_effect"
        treatment, outcome = query.treatment, query.outcome
        extra: _IdentifyStructureKwargs = {}
    elif isinstance(query, ResponseCurve) and getattr(query, "is_temporal", False):
        kind = "temporal_response"
        treatment, outcome = query.treatment, query.outcome
        extra = {
            "treatment_lag": query.treatment_lag,
            "policy": query.policy,
            "horizons": list(query.horizons or [1]),
            "max_history_lag": query.max_history_lag,
            "response_grid": list(query.grid),
            "horizon_steps": int((getattr(query, "horizons", None) or [1])[0]),
        }
    elif isinstance(query, ResponseCurve):
        kind = "response"
        treatment, outcome = query.treatment, query.outcome
        extra = {}
    elif isinstance(query, TemporalMediationEffect):
        kind = "temporal_mediation"
        treatment, outcome = query.treatment, query.outcome
        extra = {
            "treatments": [query.mediator],
            "horizons": list(query.horizons or [1]),
            "policy": query.contrast,
            "horizon_steps": int((query.horizons or [1])[0]),
        }
    elif isinstance(query, InterventionResponse) and query.is_temporal:
        from .intervention import encode_temporal_steps

        supplied_steps = query.intervention
        specs = (
            list(supplied_steps)
            if isinstance(supplied_steps, Sequence) and not isinstance(supplied_steps, (str, bytes))
            else [supplied_steps]
        )
        steps = [step for spec in specs for step in encode_temporal_steps(spec)]
        if not steps:
            raise CausalValueError("InterventionResponse requires at least one intervention")
        kind = "temporal_response"
        treatment, outcome = steps[0][0], query.outcome
        extra = {
            "response_steps": steps,
            "policy": query.policy,
            "treatment_lag": query.treatment_lag,
            "horizons": list(query.horizons or [1]),
            "max_history_lag": query.max_history_lag,
        }
    elif isinstance(query, InterventionResponse):
        kind = "intervention_response"
        spec = query.intervention
        specs = (
            list(spec)
            if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes))
            else [spec]
        )
        if not specs:
            raise CausalValueError("InterventionResponse requires at least one intervention")
        treatments = []
        for item in specs:
            variable = getattr(item, "variable", None)
            if not isinstance(variable, str):
                raise TypeError("InterventionResponse identify() needs a treatment variable")
            treatments.append(variable)
        treatment = treatments[0]
        outcome = query.outcome
        extra = {"treatments": treatments}
    elif isinstance(query, ConditionalEffect):
        kind = "conditional"
        treatment, outcome = query.treatment, query.outcome
        extra = {"modifiers": list(query.modifiers)}
    elif isinstance(query, PulseEffect):
        kind = "pulse"
        treatment, outcome = query.treatment, query.outcome
        extra = {
            "policy": "pulse",
            "treatment_lag": query.treatment_lag,
            "horizon_steps": query.horizon_steps,
            "active_level": query.active_level,
            "max_history_lag": query.max_history_lag,
        }
    elif isinstance(query, SustainedEffect):
        kind = "sustained"
        treatment, outcome = query.treatment, query.outcome
        extra = {
            "policy": "sustained",
            "treatment_lag": query.treatment_lag,
            "horizon_steps": query.horizon_steps,
            "active_level": query.active_level,
            "max_history_lag": query.max_history_lag,
            "window": query.window,
        }
    else:
        raise TypeError(f"typed-graph identify() does not support {type(query).__name__}")
    raw = _identify_structure(
        supplied,
        kind,
        treatment,
        outcome,
        identifier=identifier,
        include_details=True,
        **extra,
    )
    if not isinstance(raw, str):
        raise TypeError("native identification certificate was not returned")
    certificate = json.loads(raw)
    status, method = certificate["status"], certificate["method"]
    cases = certificate["cases"]
    sets = [case["adjustment_coordinates"][0] for case in cases if case["adjustment_coordinates"]]
    shared = sets[0] if sets and all(item == sets[0] for item in sets) else []
    adjustment = list(dict.fromkeys(item["name"] for item in shared))
    resolved_names = list(names) if names is not None else None
    if resolved_names is None and isinstance(supplied, (Dag, Cpdag, Pag)):
        nodes = supplied.nodes()
        if nodes and isinstance(nodes[0], str):
            resolved_names = list(nodes)
    return Identification(
        status=status,
        method=method,
        adjustment_set=list(adjustment),
        graph=graph,
        query=query,
        names=resolved_names,
        identifier=identifier or method,
        certificate=certificate,
    )


def _identify_tiered(
    graph: TieredBackground,
    query: object,
    names: Sequence[str] | None,
    identifier: str | None,
) -> Identification:
    from ._native import _identify_tiered_structure
    from .intervention import Set
    from .observation import Complete, _assumption_kwargs
    from .population import coerce_target_population
    from .query import coerce_outcome_functional

    if identifier is not None:
        raise CausalUnsupportedError(
            "TieredBackground selects its own identifier; omit identifier",
            reason_code="option_not_applicable",
        )
    resolved_names = (
        list(names) if names is not None else [name for tier in graph.tiers for name in tier]
    )
    spec: dict[str, Any] = {
        "outcome": getattr(query, "outcome", None),
        "target_population": coerce_target_population(getattr(query, "target_population", None)),
        "outcome_functional": coerce_outcome_functional(getattr(query, "outcome_functional", None)),
    }
    if isinstance(query, AverageEffect):
        spec.update(
            kind="average",
            treatment=query.treatment,
            control_level=query.control_level,
            active_level=query.active_level,
        )
    elif isinstance(query, InterventionResponse):
        if query.is_temporal or query.feedback is not None:
            raise CausalUnsupportedError(
                "tiered identification requires a static joint hard intervention",
                reason_code="option_not_applicable",
            )
        if query.observation is not None and not isinstance(query.observation, Complete):
            raise CausalUnsupportedError(
                "tiered joint identification requires complete observation",
                reason_code="option_not_applicable",
            )
        interventions = query.intervention
        interventions = (
            list(interventions) if isinstance(interventions, Sequence) else [interventions]
        )
        hard_interventions: list[tuple[str, float]] = []
        for item in interventions:
            if not isinstance(item, Set):
                raise CausalUnsupportedError(
                    "tiered joint identification requires hard Set interventions",
                    reason_code="option_not_applicable",
                )
            hard_interventions.append((item.variable, item.value))
        spec.update(
            kind="joint",
            interventions=hard_interventions,
            observation_assumptions=[
                _assumption_kwargs(item) for item in query.observation_assumptions
            ],
        )
    else:
        raise CausalUnsupportedError(
            "TieredBackground identification supports AverageEffect or joint InterventionResponse",
            reason_code="option_not_applicable",
        )
    certificate = json.loads(
        _identify_tiered_structure(
            resolved_names,
            [list(tier) for tier in graph.tiers],
            str(graph.within_tier),
            spec,
        )
    )
    sets = [
        coordinates
        for case in certificate["cases"]
        for coordinates in case["adjustment_coordinates"]
    ]
    shared = sets[0] if sets and all(item == sets[0] for item in sets) else []
    return Identification(
        status=certificate["status"],
        method=certificate["method"],
        adjustment_set=[item["name"] for item in shared],
        graph=graph,
        query=query,
        names=resolved_names,
        identifier=certificate["method"],
        certificate=certificate,
    )


def identify(
    *,
    graph: Dag
    | Admg
    | Cpdag
    | Pag
    | TemporalDag
    | TemporalCpdag
    | TemporalPag
    | TieredBackground
    | Sequence[tuple[str, str]],
    query: AverageEffect
    | ResponseCurve
    | InterventionResponse
    | ConditionalEffect
    | PulseEffect
    | SustainedEffect
    | PointDerivative
    | Elasticity
    | SemiElasticity
    | AverageDerivative
    | DirectionalDerivative
    | ResponseJacobian
    | MediationEffect
    | NestedCounterfactual
    | Counterfactual
    | TemporalMediationEffect
    | Any,
    names: Sequence[str] | None = None,
    identifier: str | Identifier | None = None,
) -> Identification:
    """Identify without estimating; returns a stageable :class:`Identification`.

    ``TieredBackground`` identifies without data. Co-determined tiers return
    their checked closure; unknown tiers retain two declared orientation
    scenarios, rather than bounds over every possible graph. Joint hard
    interventions use the co-determined treatment-set closure.

    Same parameters as the one-shot identify-only call
    (:func:`antecedent.estimation.identify`, still available unchanged for
    callers that only want the ``IdentifyResult`` shape) — the difference is
    the return type carries enough state to continue into ``.prepare()`` /
    ``.estimate()`` while retaining the resolved strategy and query.

    Pass ``names`` when ``graph`` is an edge list (variable order). With a
    typed graph, names are taken from ``graph.nodes()``.

    ``ResponseCurve`` uses the same pairwise backdoor identification contract
    as the complete-observation response estimator; the original curve query
    is retained so ``.estimate(data)`` executes the requested grid.
    Staged ``ResponseCurve`` identification on a ``Dag`` (or directed edge
    list) uses pairwise backdoor; ``Admg`` is refused on that path. ``Cpdag``
    and ``Pag`` response, ``ConditionalEffect`` on ``Dag`` / ``Cpdag`` /
    ``Pag``, and temporal pulse / sustained on temporal classes use the
    typed-graph identifier (generalized adjustment or temporal backdoor).
    Joint ``InterventionResponse`` uses a common adjustment set certified for
    every target in each graph completion. This is sufficient adjustment
    identification, not general response ID.

    For ``AverageEffect``, accepts an ``Admg`` as well as a ``Dag``. Prefer an
    ``Admg`` whenever a confounder is unmeasured: a ``Dag`` cannot express
    "this variable is not observable", so a latent common cause flattened into
    one is identified by adjusting on a variable no study can measure.
    ``Dag.latent_project(observed)`` builds the ``Admg``.
    """
    identifier_s = str(identifier) if isinstance(identifier, Identifier) else identifier
    from .transport import Transport
    from .transport._day1 import identification_from_transport

    if isinstance(query, Transport):
        if not isinstance(graph, Admg):
            raise CausalTypeError("transport.Transport identify requires graph=Admg(...)")
        return identification_from_transport(graph, query)

    from .accepted_graph import AcceptedGraph as _AcceptedGraph

    supplied = graph.graph if isinstance(graph, _AcceptedGraph) else graph
    if isinstance(supplied, TieredBackground):
        return _identify_tiered(supplied, query, names, identifier_s)
    if _uses_structure_identify(supplied, query):
        return _identify_typed_graph(
            graph=graph,
            supplied=supplied,
            query=query,
            identifier=identifier_s,
            names=names,
        )
    if isinstance(
        query,
        (
            MediationEffect,
            NestedCounterfactual,
            Counterfactual,
            PointDerivative,
            Elasticity,
            SemiElasticity,
            AverageDerivative,
            DirectionalDerivative,
            ResponseJacobian,
        ),
    ):
        from ._native import PreparedAnalysis as NativePrepared
        from .accepted_graph import AcceptedGraph
        from .estimation import _static_edges

        supplied = graph.graph if isinstance(graph, AcceptedGraph) else graph
        if not isinstance(supplied, (Dag, list, tuple)):
            raise TypeError("these staged kinds require a Dag")
        resolved_names = (
            list(names)
            if names is not None
            else list(supplied.nodes())
            if isinstance(supplied, Dag)
            else None
        )
        if resolved_names is None:
            raise ValueError("names is required with an edge list")
        if isinstance(query, (DirectionalDerivative, ResponseJacobian)):
            treatments = list(query.treatments)
            outcomes = list(query.outcomes)
        else:
            treatments = [query.treatment]
            outcomes = [query.outcome]
        at = getattr(query, "at", None)
        if isinstance(query, (DirectionalDerivative, ResponseJacobian)):
            at = [at[t] for t in treatments] if isinstance(at, Mapping) else list(query.at)
        elif at is not None:
            at = [at]
        direction = getattr(query, "direction", None)
        if direction is not None:
            direction = (
                [direction[t] for t in treatments]
                if isinstance(direction, Mapping)
                else list(direction)
            )
        scale = (
            "log_log"
            if isinstance(query, Elasticity)
            else "log_" + query.log_scale
            if isinstance(query, SemiElasticity)
            else "identity"
        )
        status, method, adjustment, strategy = NativePrepared.identify_existing(
            resolved_names,
            _static_edges(supplied),
            query.kind,
            treatments,
            outcomes,
            mediators=(
                list(query.mediators)
                if isinstance(query, MediationEffect)
                else [query.mediator]
                if isinstance(query, NestedCounterfactual)
                else []
            ),
            contrast=getattr(query, "contrast", "mediated"),
            control_level=getattr(query, "control_level", 0.0),
            active_level=getattr(query, "active_level", 1.0),
            at=at,
            direction=direction,
            order=getattr(query, "order", 1),
            scale=scale,
            weighting=getattr(query, "weighting", None) or "observed",
        )
        if identifier_s not in (None, strategy):
            raise ValueError(f"{query.kind} requires identifier={strategy}")
        return Identification(status, method, adjustment, graph, query, resolved_names, strategy)
    if not isinstance(query, (AverageEffect, ResponseCurve)):
        raise TypeError("staged identify() supports AverageEffect and ResponseCurve queries")
    if isinstance(query, ResponseCurve) and isinstance(graph, Admg):
        raise TypeError("staged ResponseCurve identification currently requires a Dag")
    if isinstance(query, ResponseCurve) and identifier_s not in (None, "response.backdoor"):
        raise ValueError(
            "staged ResponseCurve identification requires identifier='response.backdoor'"
        )
    identification_query = (
        AverageEffect(treatment=query.treatment, outcome=query.outcome)
        if isinstance(query, ResponseCurve)
        else query
    )
    # The identify-only native API still accepts the scalar contrast shape. The
    # pairwise backdoor search is identical, but its ATE strategy id must not be
    # confused with the response strategy retained by this staged object.
    native_identifier = "backdoor.adjustment" if isinstance(query, ResponseCurve) else identifier
    if isinstance(graph, (Cpdag, Pag, TemporalDag, TemporalCpdag, TemporalPag, TieredBackground)):
        raise TypeError("this query is not supported on the supplied graph class")
    result = _identify_native(
        graph=graph,
        query=identification_query,
        names=names,
        identifier=native_identifier,
    )
    return Identification(
        status=result.status,
        method=result.method,
        adjustment_set=list(result.adjustment_set),
        graph=graph,
        query=query,
        names=list(names) if names is not None else None,
        identifier="response.backdoor" if isinstance(query, ResponseCurve) else identifier_s,
    )


@dataclass(frozen=True, slots=True)
class BinaryIvBounds:
    """Sharp Balke–Pearl interval for ``E[Y(1)-Y(0)]`` under a binary IV."""

    lower: float
    upper: float
    method: str = "identify.binary_iv_bounds"


def binary_iv_bounds(cells: Sequence[Sequence[float]]) -> BinaryIvBounds:
    """Sharp response-type bounds on the ATE from the observed binary-IV law.

    ``cells`` is a 2×4 nested sequence: one arm per instrument value ``Z∈{0,1}``,
    with cell order ``(Y,D)=(0,0),(1,0),(0,1),(1,1)``. This is a contrast bound,
    not a continuous-response curve estimator.
    """
    import math

    from .errors import CausalEstimateError, CausalIdentifyError
    from .outcomes import require_success
    from .prepared import analyze
    from .quasi import IvBounds
    from .results import AnalysisResult

    arms = [[float(cell) for cell in arm] for arm in cells]
    if len(arms) != 2 or any(len(arm) != 4 for arm in arms):
        raise CausalValueError(
            "cells must be a 2x4 nested sequence: one arm per instrument value, "
            "cells (Y,D)=(0,0),(1,0),(0,1),(1,1)"
        )
    if any(not math.isfinite(cell) or cell < 0.0 for arm in arms for cell in arm) or any(
        abs(sum(arm) - 1.0) > 1e-9 for arm in arms
    ):
        raise CausalIdentifyError(
            "binary-IV cells must be finite, non-negative, and sum to one by Z arm"
        )
    # The observed law is a weighted eight-row table of the three coded columns: the compiled
    # IvBounds study reads the law from it exactly as it reads a sample.
    instrument = [z for z in (0.0, 1.0) for _ in range(4)]
    outcome = [float(y) for _ in (0, 1) for y, _d in ((0, 0), (1, 0), (0, 1), (1, 1))]
    treatment = [float(d) for _ in (0, 1) for _y, d in ((0, 0), (1, 0), (0, 1), (1, 1))]
    weight = [cell for arm in arms for cell in arm]
    try:
        result = require_success(
            analyze(
                {"z": instrument, "d": treatment, "y": outcome, "w": weight},
                query=IvBounds(outcome="y", treatment="d", instrument="z", weight="w"),
            )
        )
    except (CausalEstimateError, CausalUnsupportedError) as error:
        # A law the instrument premises cannot produce is an identification failure, named so.
        if "instrumental inequality" in str(error):
            raise CausalIdentifyError(str(error)) from error
        raise
    if not isinstance(result, AnalysisResult):
        raise CausalIdentifyError("the compiled IV bounds study returned no identified set")
    section = result.iv_bounds
    if section is None:
        raise CausalIdentifyError("the compiled IV bounds study returned no identified set")
    return BinaryIvBounds(lower=float(section.lower), upper=float(section.upper))


__all__ = [
    "BinaryIvBounds",
    "Identification",
    "IdentifyResult",
    "PreparedIdentification",
    "binary_iv_bounds",
    "identify",
]
