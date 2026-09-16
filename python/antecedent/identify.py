"""The staged identification surface: ``identify()`` → ``estimate()`` → ``validate()``.

:func:`identify` runs identification only (no estimate) and returns an
:class:`Identification` — enough state to continue into a strategy-consistent
``.estimate()`` / ``.validate()`` workflow. The one-shot analysis pipeline
deterministically rechecks identification before estimation. This is the
single class that replaces both ``estimation.IdentifyResult`` (identify-only
shape) and ``results.IdentificationView`` (the identification section of a
full :class:`antecedent.AnalysisResult`) — see :meth:`Identification.from_view`
and :meth:`Identification.to_identify_result` for the two conversions that
keep those existing shapes reachable rather than deleting them outright.

This module imports ``estimation``; ``estimation`` must never import this
module (that would be a cycle). Because of that direction, ``estimation.identify()``
keeps its own native-calling implementation unchanged — :func:`identify` here
wraps it rather than the reverse, so there remains exactly one code path that
calls the native identify-only entry point.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from ._verdict import describe_status, verdict_for
from .errors import CausalUnsupportedError, CausalValueError
from .estimation import IdentifyResult
from .estimation import identify as _identify_native
from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag
from .ids import Estimator, Identifier, Latency, Refute
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
    PointDerivative,
    PulseEffect,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
    SustainedEffect,
    TemporalMediationEffect,
)
from .results import IdentificationView


def _query_phrase(query: object) -> str:
    name = type(query).__name__
    treatment = getattr(query, "treatment", None)
    outcome = getattr(query, "outcome", None)
    if isinstance(treatment, str) and isinstance(outcome, str):
        mediators = getattr(query, "mediators", None)
        if mediators:
            via = ", ".join(str(item) for item in mediators)
            return f"{name} of {outcome} from {treatment} via {via}"
        modifier = getattr(query, "modifier", None)
        if isinstance(modifier, str):
            return f"{name} of {outcome} from {treatment} given {modifier}"
        return f"{name} of {outcome} from {treatment}"
    return name


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
    if not isinstance(cases, list):
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
        if not isinstance(entries, list):
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
        if not isinstance(steps, list):
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
    """A resolved identification strategy, staged for ``.estimate()`` / ``.validate()``.

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
    adjustment_set: list[str]
    graph: (
        Dag
        | Admg
        | Cpdag
        | Pag
        | TemporalDag
        | TemporalCpdag
        | TemporalPag
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
        | TemporalMediationEffect
        | Counterfactual
    )
    names: list[str] | None = None
    identifier: str | None = None
    assumption_count: int | None = None
    derivation_step_count: int | None = None
    # Complete native certificate: per-completion wire records, coordinates,
    # weights, derivations, assumptions, expression arenas, and search limits.
    certificate: dict[str, Any] | None = None

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
        return verdict_for(self.status)

    @property
    def qualified_verdict(self) -> str:
        """:attr:`verdict` plus the restriction it holds under, when there is one.

        ``IdentifiedUnderParametricRestrictions`` reads
        ``"identified under parametric restrictions"``; priors never upgrade
        identification, so ``IdentifiedUnderPriorRestrictions`` keeps its
        qualifier too.
        """
        return describe_status(self.status)

    @property
    def assumption_statements(self) -> tuple[str, ...]:
        """Readable assumption lines retained from the identification certificate."""
        return _assumption_statements(self.certificate)

    @property
    def derivation_statements(self) -> tuple[str, ...]:
        """Readable derivation steps retained from the identification certificate."""
        return _derivation_statements(self.certificate)

    @property
    def statement(self) -> str:
        """One-sentence identification state. This is data, not a display hook."""
        query = _query_phrase(self.query)
        verdict = self.verdict
        if verdict == "not identified":
            return f"{query} is not identified."
        if verdict == "graph-dependent":
            phrase = f"{query} is graph-dependent, not a single identified effect"
        else:
            phrase = f"{query} is {self.qualified_verdict}"
        method = self.method.strip() if self.method else ""
        if method and method.lower() not in {"none", "unavailable"}:
            phrase = f"{phrase} by {method}"
        if self.adjustment_set:
            adjusted = ", ".join(self.adjustment_set)
            phrase = f"{phrase}, adjusting for {adjusted}"
        return f"{phrase}."

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
        }

    def __repr__(self) -> str:
        return f"<Identification {self.statement}>"

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

    def estimate(
        self,
        data: Mapping[str, Any] | Any,
        *,
        inference: Frequentist | Bayesian | None = None,
        estimator: str | Estimator | None = None,
        estimator_config: Mapping[str, Any] | None = None,
        refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = False,
        seed: int = 1,
        bootstrap: int | None = None,
        threads: int = 1,
        latency: Latency | Literal["interactive", "standard", "report"] | None = None,
    ) -> Any:
        """Estimate the effect on ``data`` using this identification's strategy.

        Dispatches through :func:`antecedent.analyze` with ``identifier=``
        fixed to the strategy this identification already resolved, so the
        estimate is consistent with what :func:`identify` reported rather
        than letting ``analyze`` pick a strategy on its own. There is no
        separate native "estimate against a precomputed identification" entry
        point — ``analyze`` recomputes identification internally as part of
        its own pipeline, deterministically arriving at the same strategy.
        """
        from ._analyze import analyze

        return analyze(
            data,
            query=self.query,
            graph=self.graph,
            inference=inference,
            identifier=self.identifier,
            estimator=estimator,
            estimator_config=estimator_config,
            refute=refute,
            seed=seed,
            bootstrap=bootstrap,
            threads=threads,
            latency=latency,
        )

    def validate(
        self,
        data: Mapping[str, Any] | Any,
        *,
        refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = "cheap",
        seed: int = 1,
        threads: int = 1,
    ) -> Any:
        """Run the refutation/validation suite against this identification.

        ``refute`` defaults to the scalar ``"cheap"`` suite. Function-valued
        response queries have no licensed scalar refutation state and raise
        :class:`~antecedent.errors.CausalUnsupportedError`. A literal ``True``
        is not accepted by :func:`antecedent.analyze`; it does not name a
        validation contract.
        """
        return self.estimate(data, refute=refute, seed=seed, threads=threads)


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
    modifier: str
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
    | Sequence[tuple[str, str]],
    supplied: object,
    query: object,
    identifier: str | None,
    names: Sequence[str] | None,
) -> Identification:
    from ._native import identify_structure as _identify_structure

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
        extra = {"modifier": query.modifier}
    elif isinstance(query, PulseEffect):
        kind = "pulse"
        treatment, outcome = query.treatment, query.outcome
        extra = {
            "policy": "pulse",
            "treatment_lag": query.treatment_lag,
            "horizon_steps": query.horizon_steps,
            "active_level": query.active_level,
        }
    elif isinstance(query, SustainedEffect):
        kind = "sustained"
        treatment, outcome = query.treatment, query.outcome
        extra = {
            "policy": "sustained",
            "treatment_lag": query.treatment_lag,
            "horizon_steps": query.horizon_steps,
            "active_level": query.active_level,
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


def identify(
    *,
    graph: Dag
    | Admg
    | Cpdag
    | Pag
    | TemporalDag
    | TemporalCpdag
    | TemporalPag
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
    | Counterfactual
    | TemporalMediationEffect,
    names: Sequence[str] | None = None,
    identifier: str | Identifier | None = None,
) -> Identification:
    """Identify without estimating; returns a stageable :class:`Identification`.

    ``TieredBackground`` is Rust-only in 1.5 (``identify_tiered`` /
    ``identify_tiered_joint``). This Python entry does not accept a tier rule.

    Same parameters as the one-shot identify-only call
    (:func:`antecedent.estimation.identify`, still available unchanged for
    callers that only want the ``IdentifyResult`` shape) — the difference is
    the return type carries enough state to continue into ``.estimate()`` /
    ``.validate()`` while retaining the resolved strategy and query.

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
    from .accepted_graph import AcceptedGraph as _AcceptedGraph

    supplied = graph.graph if isinstance(graph, _AcceptedGraph) else graph
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
            mediators=list(query.mediators) if isinstance(query, MediationEffect) else [],
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
    if isinstance(graph, (Cpdag, Pag, TemporalDag, TemporalCpdag, TemporalPag)):
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


def estimate(
    identification: Identification,
    data: Mapping[str, Any] | Any,
    *,
    inference: Frequentist | Bayesian | None = None,
    estimator: str | Estimator | None = None,
    refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = False,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int = 1,
    latency: Latency | Literal["interactive", "standard", "report"] | None = None,
) -> Any:
    """Module-level mirror of :meth:`Identification.estimate`.

    The stub this replaces took ``(identification, *, graph, query, names,
    identifier)`` — all four of ``graph``/``query``/``names``/``identifier``
    are redundant once ``identification`` already carries them (that's the
    entire point of staging: continue without re-supplying what ``identify``
    already resolved), so they're dropped here. The stub also never accepted
    a ``data`` argument at all, despite obviously needing tabular data to run
    estimation — added as a required positional parameter.
    """
    return identification.estimate(
        data,
        inference=inference,
        estimator=estimator,
        refute=refute,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        latency=latency,
    )


def validate(
    identification: Identification,
    data: Mapping[str, Any] | Any,
    *,
    refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = "cheap",
    seed: int = 1,
    threads: int = 1,
) -> Any:
    """Module-level mirror of :meth:`Identification.validate`. See :func:`estimate`."""
    return identification.validate(data, refute=refute, seed=seed, threads=threads)


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
    from ._native import binary_iv_ate_bounds as _binary_iv_ate_bounds

    lower, upper = _binary_iv_ate_bounds([list(arm) for arm in cells])
    return BinaryIvBounds(lower=float(lower), upper=float(upper))


__all__ = [
    "BinaryIvBounds",
    "Identification",
    "IdentifyResult",
    "binary_iv_bounds",
    "estimate",
    "identify",
    "validate",
]
