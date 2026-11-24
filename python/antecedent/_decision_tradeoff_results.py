"""The result of a decision across trade-off ranges and the records it carries. Public names
are exported from :mod:`antecedent.decision`."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from ._artifact_protocol import ExportedArtifact, artifact_export, artifact_from_receipt
from ._decision_result import DecisionResult, Standing, _standing
from ._decision_value import Measure, _raise
from ._native import export_preference_decision as _export_decision
from .decision import ConstraintExclusion, PreferenceRefusal, _not_replayable
from .joint_distribution import JointDistributionArtifact
from .outcomes import Answer

if TYPE_CHECKING:
    from .decision import Contract

# --------------------------------------------------------------------------- result


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One action across the admissible trade-offs.

    Ranges are over declared assumptions, never confidence intervals. On the exact route
    ``value_range`` is exact; on the certified route ``value_range``, ``worst_case_range`` and
    ``worst_case_regret_range`` are enclosures and ``best_fraction`` is the volume fraction of
    the set on which the action is certified best. ``measure_share`` is a measure's weight, not
    a probability unless the measure is one.
    """

    id: str
    admissible: bool
    exclusions: tuple[ConstraintExclusion, ...]
    value_range: tuple[float, float] | None
    necessarily_best: bool | None
    uniquely_best: bool | None
    dominated_by: tuple[str, ...]
    possibly_best: bool | None
    worst_case_regret: float | None
    measure_share: float | None
    measure_value: float | None
    best_fraction: float | None
    worst_case_range: tuple[float, float] | None
    worst_case_regret_range: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class SwitchThreshold:
    """Where the order of a pair changes, when it depends on a single trade-off."""

    parameter: str
    value: float
    first_leads_above: bool


@dataclass(frozen=True, slots=True)
class Switch:
    """A pair of actions whose order changes inside the admissible set.

    ``favours_first_at`` and ``favours_second_at`` are admissible trade-off values on each side;
    ``threshold`` is the value where the order changes when one parameter decides it.
    """

    first: str
    second: str
    margin_range: tuple[float, float]
    favours_first_at: Mapping[str, float]
    favours_second_at: Mapping[str, float]
    threshold: SwitchThreshold | None


@dataclass(frozen=True, slots=True)
class Region:
    """One box of trade-off values the certified route kept: ``best`` when one action is proven
    best everywhere in it, otherwise the ``candidates`` still possibly best."""

    lower: Mapping[str, float]
    upper: Mapping[str, float]
    best: str | None
    candidates: tuple[str, ...]
    touches_relation_boundary: bool


@dataclass(frozen=True, slots=True)
class StructureOutcome:
    """What became of one causal structure: ``evaluated`` (with its own verdict),
    ``unidentified`` or ``unevaluated`` (with the reason)."""

    id: str
    probability: float | None
    status: Literal["evaluated", "unidentified", "unevaluated"]
    reason: str | None
    kind: str | None
    selected: str | None


@dataclass(frozen=True, slots=True)
class ClassSampling:
    """The draw of one equivalence class whose members were sampled rather than enumerated.

    ``scheme`` is ``"clique_picking"`` (a CPDAG, exactly uniform from an exact count) or
    ``"circle_endpoint_rejection"`` (a PAG, exactly uniform over its maximal ancestral graphs
    within a search budget). ``class_size`` is exact when ``class_size_exact``, otherwise a
    lower bound (the distinct members seen). Members are drawn with replacement from the
    stream ``(seed, class_key)``.
    """

    class_key: int
    scheme: str
    class_size: int
    class_size_exact: bool
    drawn: int
    distinct: int
    seed: int
    attempts: int | None
    assumption: str | None


@dataclass(frozen=True, slots=True)
class SampledMass:
    """What became of the draws of one sampled class, as masses under the declared measure.

    Each of ``identified``, ``unidentified`` and ``unevaluated`` is ``(estimate, lower, upper)``:
    the share of the draws with an exact Clopper-Pearson interval, scaled by the class mass and
    never renormalized.
    """

    class_key: int
    class_mass: float
    drawn: int
    identified: tuple[float, float, float]
    unidentified: tuple[float, float, float]
    unevaluated: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SampledMargin:
    """The expected-value margin of ``first`` over ``second`` at one vertex of the admissible
    trade-offs, on the sampled measure, with its Monte Carlo standard error (``None`` when a
    sampled class drew a single member)."""

    first: str
    second: str
    vertex: Mapping[str, float]
    margin: float
    standard_error: float | None


def _class_sampling(wire: Mapping[str, Any]) -> ClassSampling:
    size = wire["class_size"]
    return ClassSampling(
        class_key=int(wire["class_key"]),
        scheme=str(wire["scheme"]),
        class_size=int(size["count"]),
        class_size_exact=size["kind"] == "exact",
        drawn=int(wire["drawn"]),
        distinct=int(wire["distinct"]),
        seed=int(wire["seed"]),
        attempts=None if wire.get("attempts") is None else int(wire["attempts"]),
        assumption=wire.get("assumption"),
    )


def _triple(estimate: Mapping[str, Any]) -> tuple[float, float, float]:
    return (float(estimate["estimate"]), float(estimate["lower"]), float(estimate["upper"]))


def _pair(value: Any) -> tuple[float, float] | None:
    return None if value is None else (float(value[0]), float(value[1]))


def _verdict_action(verdict: Mapping[str, Any]) -> str | None:
    action = verdict.get("action")
    return None if action is None else str(action)


class TradeoffDecision(DecisionResult):
    """The decision across the admissible trade-offs (and structures, when supplied).

    ``kind`` is the verdict in snake case: ``necessarily_best`` (one action strictly ahead
    everywhere), ``never_beaten`` (never behind, tying some rival somewhere),
    ``preference_dependent`` (the choice changes inside the set; see :attr:`switches`),
    ``worst_case_choice``, ``minimax_regret_choice``, ``measure_choice``, ``tied``,
    ``undetermined`` (certified route only), ``report_only``, ``no_admissible_action``, and
    across structures ``structures_disagree`` or ``insufficient_science``. Every range in it is
    over declared assumptions, not a confidence interval.

    When an equivalence class was sampled under a declared uniform measure (see
    :meth:`antecedent.inference.ClassPrior.sample`) no verdict over every member is given:
    ``best_on_sampled_measure`` or ``undetermined_on_sampled_measure`` under
    ``bayes_over_structures`` (a Monte Carlo estimate; see :attr:`margins`), and
    ``best_on_every_sampled_member`` or ``per_sampled_member`` under ``maximin`` and
    ``require_invariant_best_action``. :attr:`sampling` and :attr:`sampled_masses` state the
    draw and its masses.
    """

    def __init__(
        self,
        contract: Any,
        source: Any,
        body: Mapping[str, Any],
        measure: Mapping[str, Any] | None,
        options: Mapping[str, Any],
    ) -> None:
        self._contract = contract
        self._source = source
        self._body = body
        self._measure_wire = measure
        self._options = dict(options)
        self._report = {"kind": self.kind, "label": self.route, "status": self.selected}

    # -- verdict -----------------------------------------------------------------

    @property
    def _table(self) -> Mapping[str, Any] | None:
        """The single-source table: the body itself, or the combination across structures."""
        if "structural_verdict" in self._body:
            combined = self._body.get("combined")
            return combined if isinstance(combined, Mapping) else None
        return self._body

    @property
    def _verdict(self) -> Mapping[str, Any]:
        structural = self._body.get("structural_verdict")
        if structural is None:
            return dict(self._body["verdict"])
        if structural["kind"] == "combined":
            return dict(structural["verdict"])
        return dict(structural)

    @property
    def kind(self) -> str:
        return str(self._verdict["kind"])

    @property
    def selected(self) -> str | None:
        """The action the verdict chooses, or ``None`` when it chooses none."""
        return _verdict_action(self._verdict)

    @property
    def choice(self) -> str | None:
        """The action the verdict chooses (the same as :attr:`selected`)."""
        return self.selected

    @property
    def standing(self) -> Standing:
        """The verdict in the vocabulary every decision result shares."""
        return _standing(self._body)

    @property
    def answer(self) -> Answer:
        """A structured answer whose ``detail`` is the verdict kind (``"necessarily_best"``, ...)."""
        return Answer(
            "structured",
            detail=self.kind,
            structured={"selected": self.selected, "contract_identity": self.contract_identity},
        )

    @property
    def statement(self) -> str:
        """The verdict in plain words."""
        # The statement is computed natively with the verdict; no sentence is written here.
        return str(self._body["statement"])

    def explain(self) -> str:
        """The statement, followed by the route and the assumptions it stands on."""
        text_ = self.statement
        text_ += f" Answered by the {self.route} route"
        if self.undecided_fraction:
            text_ += f", with a volume fraction of {self.undecided_fraction:.3g} left undecided"
        text_ += "."
        if self.assumptions:
            text_ += " It stands on: " + ", ".join(self.assumptions) + "."
        return text_

    @property
    def route(self) -> Literal["exact", "certified"]:
        return self._body["route"]

    @property
    def route_reason(self) -> str | None:
        """The stable label naming why this route answered, when known."""
        reason = self._body.get("route_reason")
        return None if reason is None else str(reason)

    @property
    def effective_tie_tolerance(self) -> float | None:
        """The tie tolerance the verdict applied: margins within it counted as ties.

        On the exact route it is the declared ``tie_tolerance`` or, when larger, the numerical
        error bound of the margins (the rounding of the expected values plus the linear-program
        tolerances times the margin's slope). On the certified route it is the declared tolerance
        on boxes, whose enclosures are outward-rounded, and under a weighted policy the declared
        tolerance plus the widest measure-weighted margin enclosure. ``None`` when no single table
        was combined (structures that disagree, or too little science to combine them).
        """
        table = self._table
        value = None if table is None else table.get("effective_tie_tolerance")
        return None if value is None else float(value)

    @property
    def contract_identity(self) -> str:
        return str(self._body["contract_identity"])

    @property
    def policy(self) -> str:
        return str(self._body["policy"])

    # -- tables ------------------------------------------------------------------

    @property
    def parameters(self) -> tuple[str, ...]:
        table = self._table
        return () if table is None else tuple(table["parameter_ids"])

    @property
    def parameter_ranges(self) -> Mapping[str, tuple[float, float]]:
        """Each parameter's range over the admissible set (declared ranges cut by relations)."""
        table = self._table
        if table is None or table.get("parameter_ranges") is None:
            return MappingProxyType({})
        return MappingProxyType(
            {
                name: (float(lo), float(hi))
                for name, (lo, hi) in zip(self.parameters, table["parameter_ranges"], strict=False)
            }
        )

    def _point(self, values: Sequence[float]) -> Mapping[str, float]:
        return MappingProxyType(
            {name: float(v) for name, v in zip(self.parameters, values, strict=False)}
        )

    @property
    def actions(self) -> tuple[ActionRecord, ...]:
        table = self._table
        if table is None:
            return ()
        return tuple(
            ActionRecord(
                id=a["id"],
                admissible=bool(a["admissible"]),
                exclusions=tuple(
                    ConstraintExclusion(e["constraint_id"], e["probability"], e["required"])
                    for e in a["exclusions"]
                ),
                value_range=_pair(a.get("value_range", a.get("value_bounds"))),
                necessarily_best=a.get("necessarily_best"),
                uniquely_best=a.get("uniquely_best"),
                dominated_by=tuple(a.get("dominated_by") or ()),
                possibly_best=a.get("possibly_best"),
                worst_case_regret=a.get("worst_case_regret"),
                measure_share=a.get("measure_share"),
                measure_value=a.get("measure_value"),
                best_fraction=a.get("best_fraction"),
                worst_case_range=_pair(a.get("worst_case_bounds")),
                worst_case_regret_range=_pair(a.get("worst_case_regret_bounds")),
            )
            for a in table["actions"]
        )

    @property
    def switches(self) -> tuple[Switch, ...]:
        """Pairs whose order changes inside the set (exact route), each with a threshold when
        one parameter decides it."""
        table = self._table
        if table is None:
            return ()
        out = []
        for s in table.get("switches") or ():
            threshold = s.get("threshold")
            out.append(
                Switch(
                    first=s["first"],
                    second=s["second"],
                    margin_range=(float(s["margin_range"][0]), float(s["margin_range"][1])),
                    favours_first_at=self._point(s["favours_first_at"]),
                    favours_second_at=self._point(s["favours_second_at"]),
                    threshold=None
                    if threshold is None
                    else SwitchThreshold(
                        threshold["parameter_id"],
                        float(threshold["value"]),
                        bool(threshold["first_leads_above"]),
                    ),
                )
            )
        return tuple(out)

    @property
    def undecided_fraction(self) -> float | None:
        """Volume fraction of the set the certified route left undecided; ``None`` on the exact
        route, which decides every point."""
        table = self._table
        value = None if table is None else table.get("undecided_fraction")
        return None if value is None else float(value)

    @property
    def regions(self) -> tuple[Region, ...]:
        """The boxes the certified route kept (empty on the exact route)."""
        table = self._table
        if table is None:
            return ()
        out = []
        for r in table.get("regions") or ():
            status = r["status"]
            out.append(
                Region(
                    lower=self._point(r["lower"]),
                    upper=self._point(r["upper"]),
                    best=status.get("action") if status["kind"] == "best" else None,
                    candidates=tuple(status.get("candidates") or ()),
                    touches_relation_boundary=bool(r["touches_relation_boundary"]),
                )
            )
        return tuple(out)

    @property
    def structures(self) -> tuple[StructureOutcome, ...]:
        """Per-structure outcomes when the decision was across structures (else empty)."""
        out = []
        for atom in self._body.get("structures") or ():
            result = atom.get("result")
            verdict = None if result is None else result["verdict"]
            out.append(
                StructureOutcome(
                    id=atom["id"],
                    probability=atom.get("probability"),
                    status=atom["status"],
                    reason=atom.get("reason"),
                    kind=None if verdict is None else verdict["kind"],
                    selected=None if verdict is None else _verdict_action(verdict),
                )
            )
        return tuple(out)

    def _sampled(self) -> Mapping[str, Any]:
        sampled = self._body.get("sampled")
        return sampled if isinstance(sampled, Mapping) else {}

    @property
    def sampling(self) -> tuple[ClassSampling, ...]:
        """The draw of every class that was sampled rather than enumerated (else empty)."""
        return tuple(_class_sampling(wire) for wire in self._sampled().get("sampling") or ())

    @property
    def sampled_masses(self) -> tuple[SampledMass, ...]:
        """Identified, unidentified and unevaluated mass of every sampled class (else empty)."""
        return tuple(
            SampledMass(
                class_key=int(mass["class_key"]),
                class_mass=float(mass["class_mass"]),
                drawn=int(mass["drawn"]),
                identified=_triple(mass["identified"]),
                unidentified=_triple(mass["unidentified"]),
                unevaluated=_triple(mass["unevaluated"]),
            )
            for mass in self._sampled().get("masses") or ()
        )

    @property
    def margins(self) -> tuple[SampledMargin, ...]:
        """Pairwise expected-value margins at each vertex of the admissible trade-offs with
        their Monte Carlo standard errors, for ``bayes_over_structures`` over sampled classes
        on the exact route (else empty)."""
        return tuple(
            SampledMargin(
                first=str(margin["first"]),
                second=str(margin["second"]),
                vertex=self._point(margin["vertex"]),
                margin=float(margin["margin"]),
                standard_error=None
                if margin.get("standard_error") is None
                else float(margin["standard_error"]),
            )
            for margin in self._sampled().get("margins") or ()
        )

    def _mass(self, name: str) -> float | None:
        value = self._body.get(name)
        return None if value is None else float(value)

    @property
    def unidentified_mass(self) -> float | None:
        return self._mass("unidentified_mass")

    @property
    def unevaluated_mass(self) -> float | None:
        return self._mass("unevaluated_mass")

    @property
    def measure(self) -> Measure | None:
        """The measure the decision was read with, if any."""
        wire = self._measure_wire or self._body.get("measure")
        if wire is None:
            return None
        ids = [p["id"] for p in self._contract._preferences()["model"]["parameters"]]
        return Measure._from_wire(wire, ids)

    @property
    def assumptions(self) -> tuple[str, ...]:
        """Parameter assumptions, relation origins and the measure origin the verdict stands on."""
        table = self._table
        found = list((table or {}).get("assumptions") or ())
        for atom in self._body.get("structures") or ():
            result = atom.get("result")
            if result is not None:
                found.extend(result.get("assumptions") or ())
        return tuple(sorted(set(found)))

    @property
    def draw_share_best_everywhere(self) -> float | None:
        """Weight of the draws in which the necessarily-best action also has the highest
        per-draw value at every vertex of the admissible set.

        A point diagnostic of the supplied law, not a calibrated probability that the action
        is best. ``None`` for a mean source, when no action is necessarily best, when vertex
        enumeration exceeds its cap, on the certified route and for structural combinations.
        """
        table = self._table
        value = None if table is None else table.get("draw_share_best_everywhere")
        return None if value is None else float(value)

    @property
    def n_draws(self) -> int:
        table = self._table
        return 0 if table is None else int(table.get("n_draws", 0))

    @property
    def contract(self) -> Contract | Any:
        """The contract as declared (a :class:`~antecedent.decision.Contract`, or the
        :class:`antecedent.preferences.Contract` it was declared with)."""
        return self._contract

    @property
    def source(self) -> Any:
        return self._source

    def to_dict(self) -> dict[str, Any]:
        """The native result, with the verdict kind, the selected action and the statement."""
        body = json.loads(json.dumps(self._body))
        body.update({"kind": self.kind, "selected": self.selected, "statement": self.statement})
        return dict(body)

    @artifact_export
    def export(self, *, artifact_id: str = "preference-decision") -> ExportedArtifact:
        """The decision bound to its contract identity and the digest of its draws, replayable.

        Only a decision on aligned joint draws is exported; a mean-source or structural
        decision refuses with ``route_not_supported``.
        """
        if not isinstance(self._source, JointDistributionArtifact):
            raise PreferenceRefusal(
                _not_replayable(
                    type(self._source).__name__,
                    "export a decision across trade-offs evaluated on aligned joint draws",
                )
            )
        data, refusal = _export_decision(
            json.dumps(self._contract._tradeoff_wire()),
            None if self._measure_wire is None else json.dumps(self._measure_wire),
            json.dumps(self._options),
            self._source._native,
            artifact_id,
        )
        _raise(refusal)
        assert data is not None
        return artifact_from_receipt(self, bytes(data))

    def __repr__(self) -> str:
        return f"<TradeoffDecision {self.kind} {self.selected!r} ({self.route})>"


__all__ = [
    "ActionRecord",
    "ClassSampling",
    "Region",
    "SampledMargin",
    "SampledMass",
    "StructureOutcome",
    "Switch",
    "SwitchThreshold",
    "TradeoffDecision",
]
