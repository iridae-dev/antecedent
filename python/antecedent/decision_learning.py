"""Optional: learn the trade-off ranges of a value from recorded choices.

:mod:`antecedent.decision` decides with trade-offs you state. This module is the optional
extra: when past choices (or stated comparisons from an elicitation exercise) are recorded, it
narrows the declared ranges to the values compatible with them, or estimates the trade-offs by
reading the choices as noisy. What it produces is reviewed and accepted once, then plugs into a
decision as ``decision.Contract(value=accepted, ...)``::

    from antecedent import decision, decision_learning as dl

    records = dl.Records(
        [
            dl.Record(
                "r1",
                decider="clinic_a",
                options=[
                    dl.Option("treat", means={"benefit": 2.0, "harm": 0.5}),
                    dl.Option("wait", means={"benefit": 0.0, "harm": 0.0}),
                ],
                chosen="treat",
                source=dl.Forecast("forecast-2026-03"),
                observation_rows=("ledger:row-17",),
            ),
        ],
        population="target",
        pooling=dl.Pooled("one_shared_set_of_preferences"),
    )
    accepted = dl.CompatibleSet().accept(value, records)
    print(accepted.pending)                      # what a reviewer should look at
    accepted = accepted.acknowledge(*accepted.pending)
    contract = decision.Contract(value=accepted, actions=..., target_population="target")

What each result is and is not:

* A compatible set is an assumption range: the declared ranges cut by the relation each choice
  implies (the chosen option was valued at least as highly, within a tolerance). It is not a
  probability statement. Contradictory records are reported with the tolerance that would
  reconcile them, never silently dropped. When the value is not linear in some trade-offs (a
  risk attitude inside ``exp``, a breakpoint inside a kink) the set is found by subdividing those
  shape parameters: its ranges are outer ranges (``basis="enclosure"``), the kept boxes are in
  ``LearnedSet.region``, and when neither compatibility nor contradiction can be proven within
  the box budget the outcome is :class:`Undetermined`.
* A choice-model estimate reads each choice as noisy (logit). Its intervals come from a normal
  approximation whose calibration status is resolved from the shipped coverage table
  (``not_measured`` until a coverage record exists), and its measure over the trade-offs is
  draws from that approximation, not a posterior. Shape parameters are estimated by profiling
  the likelihood over their declared ranges; they carry profile likelihood-ratio intervals, and
  a shape parameter the choices barely inform, or whose estimate sits on a declared bound, is
  refused rather than reported.
* An option that existed but could not be chosen (``Option(..., unavailable=constraint_id)``)
  teaches nothing about preferences.
* Pooling several decision-makers into one set of preferences is itself a named assumption
  (:class:`Pooled`); by default each decider is read separately (:class:`PerDecider`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from ._artifact_protocol import (
    ArtifactAcceptance,
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._decision_sources import _mean_wire
from ._decision_tradeoff import resolve_tradeoff_source
from ._decision_value import Measure, Structures, Value, _raise
from ._native import accept_preferences_choice_model as _accept_choice_model
from ._native import accept_preferences_compatible_set as _accept_compatible_set
from ._native import accept_preferences_declared as _accept_declared
from ._native import consume_accepted_preferences as _consume_accepted
from ._native import estimate_preference_choice_model as _estimate_choice_model
from ._native import learn_preference_set as _learn_set
from ._native import preference_analysis_options as _analysis_options
from ._scientific import ScientificResult
from ._validate import real, text
from .decision import PreferenceRefusal
from .errors import CausalTypeError, CausalValueError
from .execution import Execution, resolve_execution
from .joint_distribution import JointDistributionArtifact
from .outcomes import outcome_operation

if TYPE_CHECKING:
    from .decision import Contract

# --------------------------------------------------------------------------- options


def _floats(values: Any, what: str) -> tuple[float, ...]:
    return tuple(real(v, what, finite=True) for v in values)


@dataclass(frozen=True, slots=True, eq=False)
class Option:
    """One alternative at a recorded choice, with what was expected of it.

    Give exactly one of ``means`` (expected value per attribute id), ``draws`` (aligned joint
    draws per attribute id, with optional ``weights``) or ``unavailable`` (the id of the
    constraint that made it impossible to choose; such an option teaches nothing).
    """

    id: str
    means: Mapping[str, float] | None = None
    draws: Mapping[str, Sequence[float]] | None = None
    weights: Sequence[float] | None = None
    unavailable: str | None = None

    def __post_init__(self) -> None:
        text(self.id, "Option.id", non_empty=True)
        given = [x is not None for x in (self.means, self.draws, self.unavailable)]
        if sum(given) != 1:
            raise CausalValueError("an Option gives means=, draws= or unavailable=, exactly one")
        if self.weights is not None and self.draws is None:
            raise CausalValueError("Option.weights weigh draws=")
        if self.means is not None:
            object.__setattr__(
                self,
                "means",
                MappingProxyType(
                    {
                        k: real(v, f"Option {self.id!r} mean", finite=True)
                        for k, v in self.means.items()
                    }
                ),
            )
        if self.draws is not None:
            object.__setattr__(
                self,
                "draws",
                MappingProxyType(
                    {k: _floats(v, f"Option {self.id!r} draw") for k, v in self.draws.items()}
                ),
            )
        if self.weights is not None:
            object.__setattr__(self, "weights", _floats(self.weights, "an option draw weight"))
        if self.unavailable is not None:
            text(self.unavailable, "Option.unavailable", non_empty=True)

    def _wire(self, attributes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        names = [a["id"] for a in attributes]
        if self.unavailable is not None:
            return {
                "id": self.id,
                "belief": {"kind": "unavailable", "constraint": self.unavailable},
            }
        given = self.means if self.means is not None else self.draws
        assert given is not None
        if sorted(given) != sorted(names):
            raise CausalValueError(
                f"Option {self.id!r} gives a value for every attribute {names}, no more"
            )
        if self.means is not None:
            return {
                "id": self.id,
                "belief": {"kind": "means", "values": [self.means[n] for n in names]},
            }
        assert self.draws is not None
        columns = [self.draws[n] for n in names]
        if len({len(c) for c in columns}) != 1:
            raise CausalValueError(
                f"Option {self.id!r} gives the same number of draws per attribute"
            )
        rows = [value for row in zip(*columns, strict=True) for value in row]
        return {
            "id": self.id,
            "belief": {
                "kind": "draws",
                "rows": rows,
                "weights": None if self.weights is None else list(self.weights),
            },
        }

    @classmethod
    def from_result(cls, contract: Contract, source: Any, **binding: Any) -> tuple[Option, ...]:
        """What was expected of each action of ``contract``, read from an analysis source.

        ``contract`` is a :class:`antecedent.decision.Contract` with ``value=``; ``source`` is
        what its ``evaluate`` reads
        across trade-offs (aligned joint draws, a mean source, or a supported analysis result
        with the same ``program=``, ``outcome_units=``, ``dose_units=`` and ``population=``
        keywords). Each quantity input is located exactly as the decision locates it; a fixed
        input is taken as declared. Use the options in a :class:`Record` whose source is
        :class:`AnalysisAsOf`.
        """
        resolved = resolve_tradeoff_source(contract, source, **binding)
        contract_json = json.dumps(contract._tradeoff_wire())
        if isinstance(resolved, JointDistributionArtifact):
            body, refusal = _analysis_options(contract_json, resolved._native, None)
        elif isinstance(resolved, Structures):
            raise CausalTypeError("Option.from_result reads one source, not Structures")
        else:
            body, refusal = _analysis_options(contract_json, None, json.dumps(_mean_wire(resolved)))
        _raise(refusal)
        assert body is not None
        names = [a["id"] for a in contract._tradeoff_wire()["model"]["attributes"]]
        out = []
        for option in json.loads(body)["options"]:
            belief = option["belief"]
            if belief["kind"] == "means":
                out.append(cls(option["id"], means=dict(zip(names, belief["values"], strict=True))))
            else:
                rows, k = belief["rows"], len(names)
                out.append(
                    cls(
                        option["id"],
                        draws={name: rows[j::k] for j, name in enumerate(names)},
                        weights=belief.get("weights"),
                    )
                )
        return tuple(out)


# --------------------------------------------------------------------------- belief sources


@dataclass(frozen=True, slots=True)
class Forecast:
    """The expectations were a forecast recorded before the choice."""

    source_id: str

    def _wire(self) -> dict[str, Any]:
        return {"kind": "forecast", "source_id": self.source_id}


@dataclass(frozen=True, slots=True)
class AnalysisAsOf:
    """The expectations were an analysis result as it stood at a date."""

    result_identity: str
    as_of: str

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "analysis_as_of",
            "result_identity": self.result_identity,
            "as_of": self.as_of,
        }


@dataclass(frozen=True, slots=True)
class Realized:
    """The outcomes later realized, read as what was expected under a named assumption."""

    assumption: str

    def _wire(self) -> dict[str, Any]:
        return {"kind": "realized", "assumption": self.assumption}


@dataclass(frozen=True, slots=True)
class Stated:
    """A stated comparison: what someone said they would choose among hypothetical bundles in
    an elicitation exercise, not what they chose. Read as evidence only under ``assumption``
    (that the statement reflects the preferences the person would act on). Every option of a
    stated record is given by ``means`` (a bundle is known exactly)."""

    elicitation: str
    assumption: str

    def _wire(self) -> dict[str, Any]:
        return {"kind": "stated", "elicitation": self.elicitation, "assumption": self.assumption}


BeliefSource = Forecast | AnalysisAsOf | Realized | Stated


@dataclass(frozen=True, slots=True, eq=False)
class Record:
    """One recorded choice (or stated comparison): the options, the one chosen, where the
    expectations came from, and the evidence-ledger rows it rests on (each row counts once)."""

    id: str
    decider: str
    options: tuple[Option, ...]
    chosen: str
    source: BeliefSource
    observation_rows: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", tuple(self.options))
        object.__setattr__(self, "observation_rows", tuple(self.observation_rows))
        if any(not isinstance(o, Option) for o in self.options):
            raise CausalTypeError("Record.options holds Option values")
        if not isinstance(self.source, (Forecast, AnalysisAsOf, Realized, Stated)):
            raise CausalTypeError("Record.source is a Forecast, AnalysisAsOf, Realized or Stated")

    def _wire(self, attributes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "id": self.id,
            "decider": self.decider,
            "options": [o._wire(attributes) for o in self.options],
            "chosen": self.chosen,
            "source": self.source._wire(),
            "observation_rows": list(self.observation_rows),
        }


@dataclass(frozen=True, slots=True)
class PerDecider:
    """Read each decision-maker's choices separately (the default)."""

    def _wire(self) -> dict[str, Any]:
        return {"kind": "per_decider"}


@dataclass(frozen=True, slots=True)
class Pooled:
    """Read every choice as coming from one set of preferences; the assumption is named."""

    assumption: str

    def _wire(self) -> dict[str, Any]:
        return {"kind": "pooled", "assumption": self.assumption}


@dataclass(frozen=True, slots=True, eq=False)
class Records:
    """Recorded choices of decision-makers in one population, and how they are pooled."""

    records: tuple[Record, ...]
    population: str
    pooling: PerDecider | Pooled = PerDecider()

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        if any(not isinstance(r, Record) for r in self.records):
            raise CausalTypeError("Records.records holds Record values")
        if not isinstance(self.pooling, (PerDecider, Pooled)):
            raise CausalTypeError("Records.pooling is PerDecider() or Pooled(assumption)")

    def _wire(self, model: Mapping[str, Any]) -> str:
        attributes = model["attributes"]
        return json.dumps(
            {
                "attributes": [{"id": a["id"], "units": a["units"]} for a in attributes],
                "population": self.population,
                "records": [r._wire(attributes) for r in self.records],
                "pooling": self.pooling._wire(),
            }
        )


def _model_wire(model: Value) -> dict[str, Any]:
    if not isinstance(model, Value):
        raise CausalTypeError("learning reads a decision.Value")
    return model._wire()


def _records(records: Records) -> Records:
    if not isinstance(records, Records):
        raise CausalTypeError("learning reads decision_learning.Records")
    return records


# --------------------------------------------------------------------------- compatible set


@dataclass(frozen=True, slots=True)
class ParameterRange:
    """One parameter's range before (``declared``) and after (``learned``) the records.

    An assumption range, not a probability statement. ``basis`` is ``linear_program`` (the exact
    smallest and largest admissible value) or ``enclosure`` (an outer range under a value with
    shape parameters: it holds every admissible value and possibly more)."""

    parameter: str
    declared: tuple[float, float]
    learned: tuple[float, float]
    varied: bool
    narrowed: bool
    basis: Literal["linear_program", "enclosure"] = "linear_program"


@dataclass(frozen=True, slots=True)
class RegionCell:
    """One kept box of shape-parameter values.

    ``status`` is ``inside`` (every shape value of the box admits trade-offs compatible with every
    record) or ``boundary`` (neither proven inside nor proven outside). ``linear`` is the outer
    range of each remaining trade-off over the box."""

    shape: Mapping[str, tuple[float, float]]
    status: Literal["inside", "boundary"]
    linear: Mapping[str, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class Region:
    """The shape-parameter boxes that cover a compatible set (or that were left undecided)."""

    shape_parameters: tuple[str, ...]
    linear_parameters: tuple[str, ...]
    cells: tuple[RegionCell, ...]
    boxes_used: int
    budget_exhausted: bool


def _region(wire: Mapping[str, Any] | None, names: Sequence[str]) -> Region | None:
    if wire is None:
        return None
    shape = tuple(names[i] for i in wire["shape_parameters"])
    linear = tuple(names[i] for i in wire["linear_parameters"])
    cells = tuple(
        RegionCell(
            MappingProxyType(
                {
                    name: (float(lo), float(hi))
                    for name, lo, hi in zip(shape, c["shape_lower"], c["shape_upper"], strict=True)
                }
            ),
            c["status"],
            MappingProxyType(
                {
                    name: (float(pair[0]), float(pair[1]))
                    for name, pair in zip(linear, c["linear_outer"], strict=True)
                }
            ),
        )
        for c in wire["cells"]
    )
    return Region(shape, linear, cells, int(wire["boxes_used"]), bool(wire["budget_exhausted"]))


@dataclass(frozen=True, slots=True)
class LearnedSet:
    """The set of trade-offs compatible with the records of one decider (or pooled).

    ``binding`` lists the ``(record, rejected option)`` choices that bound the set; ``tolerance``
    is how far below a rejected option a chosen one was allowed to fall."""

    decider: str | None
    ranges: tuple[ParameterRange, ...]
    binding: tuple[tuple[str, str], ...]
    tolerance: float
    n_records: int
    assumptions: tuple[str, ...]
    model_identity: str
    records_digest: str
    region: Region | None = None

    def range(self, parameter: str) -> tuple[float, float]:
        """The learned range of one parameter."""
        for r in self.ranges:
            if r.parameter == parameter:
                return r.learned
        raise CausalValueError(f"no parameter {parameter!r}")


@dataclass(frozen=True, slots=True)
class Conflict:
    """Records that cannot all hold under the declared model: a smallest (or irreducible) set of
    them, and the smallest tolerance at which every one becomes compatible.

    Under a value with shape parameters the conflict is proven by subdivision; ``kind`` is
    ``not_minimised`` when it could not be proven that fewer of these records are compatible, and
    ``tolerance_needed`` is a tolerance at which the records are compatible at the best shape
    value examined."""

    decider: str | None
    records: tuple[str, ...]
    kind: Literal["smallest", "irreducible", "not_minimised"]
    tolerance_needed: float


@dataclass(frozen=True, slots=True)
class Undetermined:
    """Records whose compatibility with the declared model is proven neither way: no box of
    shape values was proven compatible and not every box was proven incompatible before the
    subdivision stopped. ``region`` holds the undecided boxes; a larger box budget or a smaller
    minimum width may settle it."""

    decider: str | None
    n_records: int
    tolerance: float
    region: Region | None
    model_identity: str
    records_digest: str


def _set_outcome(
    wire: Mapping[str, Any], decider: str | None, names: Sequence[str] = ()
) -> LearnedSet | Conflict | Undetermined:
    if wire["kind"] == "contradictory":
        c = wire["conflict"]
        return Conflict(decider, tuple(c["records"]), c["kind"], float(c["tolerance_needed"]))
    if wire["kind"] == "undetermined":
        u = wire["undetermined"]
        return Undetermined(
            decider=u.get("decider"),
            n_records=int(u["n_records"]),
            tolerance=float(u["tolerance"]),
            region=_region(u["region"], names) if names else None,
            model_identity=u["model_identity"],
            records_digest=u["records_digest"],
        )
    learned = wire["learned"]
    names = [r["parameter_id"] for r in learned["ranges"]]
    return LearnedSet(
        decider=learned.get("decider"),
        ranges=tuple(
            ParameterRange(
                r["parameter_id"],
                (float(r["declared"][0]), float(r["declared"][1])),
                (float(r["learned"][0]), float(r["learned"][1])),
                bool(r["varied_in_records"]),
                bool(r["narrowed"]),
                r.get("basis", "linear_program"),
            )
            for r in learned["ranges"]
        ),
        binding=tuple((b["record_id"], b["rejected_option"]) for b in learned["binding"]),
        tolerance=float(learned["tolerance"]),
        n_records=int(learned["n_records"]),
        assumptions=tuple(learned["assumptions"]),
        model_identity=learned["model_identity"],
        records_digest=learned["records_digest"],
        region=_region(learned.get("region"), names),
    )


class CompatibleSetResult(ScientificResult):
    """The compatible set for the pooled records and for each decider, and every conflict."""

    def __init__(self, body: Mapping[str, Any], parameter_ids: Sequence[str] = ()) -> None:
        self._body = body
        names = tuple(parameter_ids)
        pooled = body.get("pooled")
        self.pooled: LearnedSet | Conflict | Undetermined | None = (
            None if pooled is None else _set_outcome(pooled, None, names)
        )
        self.per_decider: Mapping[str, LearnedSet | Conflict | Undetermined] = MappingProxyType(
            {
                d["decider"]: _set_outcome(d["outcome"], d["decider"], names)
                for d in body["per_decider"]
            }
        )
        overlap = body.get("overlap")
        self.common: LearnedSet | Conflict | Undetermined | None = (
            None if overlap is None else _set_outcome(overlap["common"], None, names)
        )
        self.disagreeing: tuple[str, ...] = () if overlap is None else tuple(overlap["disagreeing"])

    @property
    def conflicts(self) -> tuple[Conflict, ...]:
        """Every contradictory outcome, pooled first."""
        found = [self.pooled, *self.per_decider.values(), self.common]
        return tuple(item for item in found if isinstance(item, Conflict))

    @property
    def undetermined(self) -> tuple[Undetermined, ...]:
        """Every outcome proven neither compatible nor contradictory, pooled first."""
        found = [self.pooled, *self.per_decider.values(), self.common]
        return tuple(item for item in found if isinstance(item, Undetermined))

    def to_dict(self) -> dict[str, Any]:
        return dict(json.loads(json.dumps(self._body)))

    def __repr__(self) -> str:
        return (
            f"<CompatibleSetResult pooled={type(self.pooled).__name__} "
            f"deciders={len(self.per_decider)} conflicts={len(self.conflicts)}>"
        )


@dataclass(frozen=True, slots=True)
class CompatibleSet:
    """Learn the trade-offs compatible with the records.

    Assumption: each chosen option was valued at least as highly as every available rejected
    option, within ``tolerance`` utility units, under the declared model and the recorded
    expectations. ``conflict_search_size`` bounds the exhaustive search for a smallest conflict.
    Under a value with shape parameters, ``max_boxes`` bounds the boxes of shape values examined
    and ``min_width_fraction`` stops splitting a box once every shape parameter's width is at most
    that fraction of its declared range.
    """

    tolerance: float = 0.0
    conflict_search_size: int | None = None
    max_boxes: int | None = None
    min_width_fraction: float | None = None

    def _options(self) -> str:
        wire: dict[str, Any] = {"tolerance": real(self.tolerance, "tolerance")}
        if self.conflict_search_size is not None:
            wire["conflict_search_size"] = int(self.conflict_search_size)
        if self.max_boxes is not None:
            wire["max_boxes"] = int(self.max_boxes)
        if self.min_width_fraction is not None:
            wire["min_width_fraction"] = real(self.min_width_fraction, "min_width_fraction")
        return json.dumps(wire)

    @outcome_operation
    def learn(
        self, model: Value, records: Records, *, execution: Execution | None = None
    ) -> CompatibleSetResult:
        """The full outcome: per decider, pooled (when pooled) and every conflict. The learner
        reads the execution's cancellation token at its work boundaries; a cancelled run is
        refused with ``cancelled_no_claim``."""
        execution = resolve_execution(execution)
        wire = _model_wire(model)
        body, refusal = _learn_set(
            json.dumps(wire), _records(records)._wire(wire), self._options(), execution.cancel
        )
        _raise(refusal)
        assert body is not None
        return CompatibleSetResult(json.loads(body), [p["id"] for p in wire["parameters"]])

    def accept(
        self,
        model: Value,
        records: Records,
        *,
        decider: str | None = None,
        execution: Execution | None = None,
    ) -> AcceptedPreferences:
        """Learn and accept: ``decider=None`` takes the pooled outcome (records must be
        :class:`Pooled`), a name takes that decider's own. Contradictory records refuse. The
        learner reads the execution's cancellation token at its work boundaries."""
        execution = resolve_execution(execution)
        wire = _model_wire(model)
        native, refusal = _accept_compatible_set(
            json.dumps(wire),
            _records(records)._wire(wire),
            decider,
            self._options(),
            execution.cancel,
        )
        _raise(refusal)
        return AcceptedPreferences(native)


# --------------------------------------------------------------------------- choice model


@dataclass(frozen=True, slots=True)
class ParameterEstimate:
    """One trade-off's point estimate and interval.

    The interval comes from a normal approximation (``interval_method="wald"``) or, for a shape
    parameter, from the profile likelihood ratio (``"profile_likelihood"``); its calibration
    status is the estimate's ``interval_calibration``.
    ``inside_declared`` says whether the point estimate lies in the declared range."""

    parameter: str
    estimate: float
    standard_error: float
    interval: tuple[float, float]
    declared: tuple[float, float]
    inside_declared: bool
    interval_method: Literal["wald", "profile_likelihood"] = "wald"


@dataclass(frozen=True, slots=True, eq=False)
class ChoiceEstimate:
    """Trade-offs estimated from choices read as noisy (logit in the declared value).

    ``consistency`` is how sharply choices follow value. ``measure`` holds draws from the normal
    approximation to the estimate's sampling distribution, kept inside the declared set; it is
    not a posterior. ``interval_calibration`` is resolved from the shipped coverage table:
    ``measured`` only when a current coverage record measured this construction, otherwise
    ``not_measured``."""

    identity: str
    decider: str | None
    consistency: float
    consistency_standard_error: float
    parameters: tuple[ParameterEstimate, ...]
    not_learnable: tuple[str, ...]
    log_likelihood: float
    null_log_likelihood: float
    n_records: int
    n_deciders: int
    covariance: str
    level: float
    interval_calibration: str
    measure: Measure
    share_outside_declared: float
    assumptions: tuple[str, ...]
    shape_parameters: tuple[str, ...] = ()


def _estimate(wire: Mapping[str, Any], parameter_ids: Sequence[str]) -> ChoiceEstimate:
    return ChoiceEstimate(
        identity=wire["identity"],
        decider=wire.get("decider"),
        consistency=float(wire["consistency"]),
        consistency_standard_error=float(wire["consistency_standard_error"]),
        parameters=tuple(
            ParameterEstimate(
                p["parameter_id"],
                float(p["estimate"]),
                float(p["standard_error"]),
                (float(p["interval"][0]), float(p["interval"][1])),
                (float(p["declared"][0]), float(p["declared"][1])),
                bool(p["inside_declared"]),
                p.get("interval_method", "wald"),
            )
            for p in wire["parameters"]
        ),
        not_learnable=tuple(wire["not_learnable"]),
        log_likelihood=float(wire["log_likelihood"]),
        null_log_likelihood=float(wire["null_log_likelihood"]),
        n_records=int(wire["n_records"]),
        n_deciders=int(wire["n_deciders"]),
        covariance=str(wire["covariance_kind"]),
        level=float(wire["level"]),
        interval_calibration=str(wire["interval_calibration"]),
        measure=Measure._from_wire(wire["measure"], parameter_ids),
        share_outside_declared=float(wire["share_outside_declared"]),
        assumptions=tuple(wire["assumptions"]),
        shape_parameters=tuple(wire.get("shape_parameters", ())),
    )


class ChoiceModelResult(ScientificResult):
    """The pooled estimate (pooled records) or one outcome per decider: an estimate, or the
    refusal of that decider's fit."""

    def __init__(self, body: Mapping[str, Any], parameter_ids: Sequence[str]) -> None:
        self._body = body
        pooled = body.get("pooled")
        self.pooled: ChoiceEstimate | None = (
            None if pooled is None else _estimate(pooled, parameter_ids)
        )
        per: dict[str, ChoiceEstimate | PreferenceRefusal] = {}
        for item in body["per_decider"]:
            if item.get("estimate") is not None:
                per[item["decider"]] = _estimate(item["estimate"], parameter_ids)
            else:
                per[item["decider"]] = PreferenceRefusal(item["refusal"])
        self.per_decider: Mapping[str, ChoiceEstimate | PreferenceRefusal] = MappingProxyType(per)

    def to_dict(self) -> dict[str, Any]:
        return dict(json.loads(json.dumps(self._body)))

    def __repr__(self) -> str:
        return (
            f"<ChoiceModelResult pooled={self.pooled is not None} deciders={len(self.per_decider)}>"
        )


@dataclass(frozen=True, slots=True)
class ShapeSearch:
    """How the shape parameters of a value are searched when estimating from noisy choices.

    ``grid_points`` profile evaluations seed the search (the box centre, then a Sobol sequence),
    ``max_dimensions`` caps the number of shape parameters searched, ``profile_points`` sets the
    grid each shape parameter's profile is checked on, and ``max_refinements`` bounds each local
    refinement."""

    grid_points: int = 64
    max_dimensions: int = 3
    profile_points: int = 33
    max_refinements: int = 200


@dataclass(frozen=True, slots=True)
class ChoiceModel:
    """Estimate the trade-offs from choices read as noisy.

    ``level`` is the two-sided level of each interval (calibration from the shipped coverage
    table), ``draws`` the
    number of draws in the estimate's measure, ``seed`` their generator seed. Records holding
    both stated comparisons and recorded choices in one fit are refused. ``shape_search``
    controls the search over shape parameters, when the value has any.
    """

    level: float = 0.95
    draws: int = 2000
    seed: int = 0
    shape_search: ShapeSearch | None = None

    def _options(self) -> str:
        wire: dict[str, Any] = {
            "level": real(self.level, "level"),
            "n_draws": int(self.draws),
            "seed": int(self.seed),
        }
        if self.shape_search is not None:
            if not isinstance(self.shape_search, ShapeSearch):
                raise CausalTypeError("shape_search= is a ShapeSearch")
            wire["shape_grid_points"] = int(self.shape_search.grid_points)
            wire["shape_max_dimensions"] = int(self.shape_search.max_dimensions)
            wire["shape_profile_points"] = int(self.shape_search.profile_points)
            wire["shape_max_refinements"] = int(self.shape_search.max_refinements)
        return json.dumps(wire)

    @outcome_operation
    def learn(
        self, model: Value, records: Records, *, execution: Execution | None = None
    ) -> ChoiceModelResult:
        """The pooled estimate, or one outcome per decider. The estimator reads the
        execution's cancellation token at its work boundaries; a cancelled run is refused with
        ``cancelled_no_claim``."""
        execution = resolve_execution(execution)
        wire = _model_wire(model)
        body, refusal = _estimate_choice_model(
            json.dumps(wire), _records(records)._wire(wire), self._options(), execution.cancel
        )
        _raise(refusal)
        assert body is not None
        return ChoiceModelResult(json.loads(body), [p["id"] for p in wire["parameters"]])

    def accept(
        self,
        model: Value,
        records: Records,
        *,
        decider: str | None = None,
        execution: Execution | None = None,
    ) -> AcceptedPreferences:
        """Estimate and accept; the accepted preferences carry the estimate's measure. The
        estimator reads the execution's cancellation token at its work boundaries."""
        execution = resolve_execution(execution)
        wire = _model_wire(model)
        native, refusal = _accept_choice_model(
            json.dumps(wire),
            _records(records)._wire(wire),
            decider,
            self._options(),
            execution.cancel,
        )
        _raise(refusal)
        return AcceptedPreferences(native)


# --------------------------------------------------------------------------- accepted


@dataclass(frozen=True, slots=True)
class ReviewItem:
    """Something to look at before relying on accepted preferences.

    ``kind`` is ``parameter_not_varied`` (no record moved it; it keeps its declared range),
    ``parameter_not_narrowed``, ``tolerance_applied``, ``parameter_not_learnable``,
    ``estimate_outside_declared``, ``draws_outside_declared`` or ``stated_comparisons_used``
    (the records include stated comparisons: what someone said they would choose, not what
    they chose); ``parameter`` names the parameter and ``amount`` the tolerance, the dropped
    share or the number of stated comparisons, when there is one. A parameter the model fixes
    (``lower == upper``) is a value stated outright and raises no item."""

    kind: str
    parameter: str | None = None
    amount: float | None = None

    def _wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"kind": self.kind}
        if self.parameter is not None:
            wire["parameter_id"] = self.parameter
        if self.kind == "tolerance_applied":
            wire["tolerance"] = self.amount
        if self.kind == "draws_outside_declared":
            wire["share"] = self.amount
        if self.kind == "stated_comparisons_used":
            if self.amount is None or self.amount != int(self.amount):
                raise CausalValueError("a stated_comparisons_used item counts whole comparisons")
            wire["count"] = int(self.amount)
        return wire

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> ReviewItem:
        amount = wire.get("tolerance", wire.get("share", wire.get("count")))
        return cls(
            wire["kind"], wire.get("parameter_id"), None if amount is None else float(amount)
        )


class AcceptedPreferences:
    """Reviewed, versioned preferences reused across decisions.

    Immutable: :meth:`acknowledge` and :meth:`relearn` return a new handle with the version
    increased by one; nothing relearns implicitly. Pass it as ``value=`` to
    :class:`antecedent.decision.Contract`; an estimate's measure travels with it.
    """

    def __init__(self, native: Any) -> None:
        self._native = native
        self._replayed = False
        self._summary: Mapping[str, Any] = json.loads(native.summary_json())

    @classmethod
    def declared(cls, model: Value) -> AcceptedPreferences:
        """Accept the declared ranges as they stand (version 1, nothing pending)."""
        native, refusal = _accept_declared(json.dumps(_model_wire(model)))
        _raise(refusal)
        return cls(native)

    @property
    def identity(self) -> str:
        return str(self._native.identity)

    @property
    def version(self) -> int:
        return int(self._native.version)

    @property
    def basis(self) -> Literal["declared", "compatible_set", "choice_estimate"]:
        return self._summary["basis"]["kind"]

    @property
    def decider(self) -> str | None:
        return self._summary.get("decider")

    @property
    def _parameter_ids(self) -> list[str]:
        return [p["id"] for p in self._summary["declared_model"]["parameters"]]

    @property
    def pending(self) -> tuple[ReviewItem, ...]:
        """Review items not yet acknowledged."""
        return tuple(ReviewItem._from_wire(w) for w in self._summary["pending"])

    @property
    def acknowledged(self) -> tuple[ReviewItem, ...]:
        return tuple(ReviewItem._from_wire(w) for w in self._summary["acknowledged"])

    @property
    def ranges(self) -> Mapping[str, tuple[float, float]]:
        """The accepted range of each parameter: learned for a compatible set, declared
        otherwise. Assumption ranges, not probability intervals."""
        basis = self._summary["basis"]
        if basis["kind"] == "compatible_set":
            return MappingProxyType(
                {
                    r["parameter_id"]: (float(r["learned"][0]), float(r["learned"][1]))
                    for r in basis["learned"]["ranges"]
                }
            )
        return MappingProxyType(
            {
                p["id"]: (float(p["lower"]), float(p["upper"]))
                for p in self._summary["model"]["parameters"]
            }
        )

    @property
    def learned(self) -> LearnedSet | None:
        basis = self._summary["basis"]
        if basis["kind"] != "compatible_set":
            return None
        learned = _set_outcome({"kind": "learned", "learned": basis["learned"]}, self.decider)
        assert isinstance(learned, LearnedSet)
        return learned

    @property
    def estimate(self) -> ChoiceEstimate | None:
        basis = self._summary["basis"]
        if basis["kind"] != "choice_estimate":
            return None
        return _estimate(basis["estimate"], self._parameter_ids)

    @property
    def measure(self) -> Measure | None:
        """The estimate's measure over the trade-offs (not a posterior), or ``None``."""
        wire = self._summary.get("measure")
        return None if wire is None else Measure._from_wire(wire, self._parameter_ids)

    @property
    def assumptions(self) -> tuple[str, ...]:
        return tuple(self._summary["assumptions"])

    @property
    def stated_comparisons(self) -> int:
        """How many of the records the preferences were learned from are stated comparisons."""
        return int(self._summary.get("stated_comparisons", 0))

    def acknowledge(self, *items: ReviewItem) -> AcceptedPreferences:
        """A new handle (version + 1) with ``items`` acknowledged; each must be pending."""
        if any(not isinstance(item, ReviewItem) for item in items):
            raise CausalTypeError("acknowledge takes ReviewItem values from .pending")
        native, refusal = self._native.acknowledge(json.dumps([i._wire() for i in items]))
        _raise(refusal)
        return AcceptedPreferences(native)

    def relearn(
        self,
        records: Records,
        *,
        decider: str | None = None,
        using: CompatibleSet | ChoiceModel | None = None,
        execution: Execution | None = None,
    ) -> AcceptedPreferences:
        """Relearn from ``records`` with the stored options (or ``using``): version + 1,
        acknowledgements cleared. The learner reads the execution's cancellation token at its
        work boundaries; a cancelled run is refused with ``cancelled_no_claim``."""
        execution = resolve_execution(execution)
        if using is None:
            learning = self._summary.get("learning")
            if learning is None:
                raise CausalValueError(
                    "declared preferences have no learning options; pass using=CompatibleSet(...) "
                    "or using=ChoiceModel(...)"
                )
            if learning["kind"] == "compatible_set":
                using = CompatibleSet(
                    learning["tolerance"],
                    learning["conflict_search_size"],
                    learning.get("max_boxes"),
                    learning.get("min_width_fraction"),
                )
            else:
                search = ShapeSearch(
                    learning.get("shape_grid_points", 64),
                    learning.get("shape_max_dimensions", 3),
                    learning.get("shape_profile_points", 33),
                    learning.get("shape_max_refinements", 200),
                )
                using = ChoiceModel(
                    learning["level"], learning["n_draws"], learning["seed"], search
                )
        wire = _records(records)._wire(self._summary["declared_model"])
        if isinstance(using, CompatibleSet):
            native, refusal = self._native.relearn_compatible_set(
                wire, decider, using._options(), execution.cancel
            )
        elif isinstance(using, ChoiceModel):
            native, refusal = self._native.relearn_choice_model(
                wire, decider, using._options(), execution.cancel
            )
        else:
            raise CausalTypeError("using= is a CompatibleSet or a ChoiceModel")
        _raise(refusal)
        return AcceptedPreferences(native)

    def _preference_wire(self) -> dict[str, Any]:
        """What a contract reads from ``value=``: the model and the
        measure, if any."""
        return {"model": self._summary["model"], "measure": self._summary.get("measure")}

    @artifact_export
    def export(self, *, artifact_id: str = "accepted-preferences") -> ExportedArtifact:
        """The checksummed record, loadable under :attr:`identity` (see :func:`load`)."""
        data, refusal = self._native.export(artifact_id)
        _raise(refusal)
        assert data is not None
        return artifact_from_receipt(self, bytes(data))

    def to_dict(self) -> dict[str, Any]:
        return dict(json.loads(json.dumps(self._summary)))

    def __repr__(self) -> str:
        return (
            f"<AcceptedPreferences {self.basis} v{self.version} pending={len(self.pending)} "
            f"{self.identity[:12]}>"
        )


_LOAD_PROOF = (
    "Loads only under the identity the consumer retained, recomputed from the decoded content; "
    "with records, a learned basis is relearned and must be reproduced."
)


def _load_acceptance(result: AcceptedPreferences, bound: bool) -> ArtifactAcceptance:
    """Replayed when the basis was recomputed (records given, or nothing to learn), else loaded."""
    replayed = bool(getattr(result, "_replayed", False))
    return ArtifactAcceptance(
        "replayed" if replayed else "loaded", bound, True, False, proof=_LOAD_PROOF
    )


@artifact_consumer(
    proof=_LOAD_PROOF,
    replayable=True,
    identity=("expected_identity",),
    input="data",
    acceptance=_load_acceptance,
)
def load(
    data: bytes,
    *,
    expected_identity: str,
    records: Records | None = None,
    execution: Execution | None = None,
) -> AcceptedPreferences:
    """Load accepted preferences under an identity retained separately from the bytes.

    With ``records``, a learned basis is relearned from them with the stored options and must be
    reproduced; a declared basis needs no records. The relearning reads the execution's
    cancellation token; a cancelled replay is refused with ``cancelled_no_claim``.
    """
    execution = resolve_execution(execution)
    data = container_bytes(data)
    records_json = None
    if records is not None:
        # The record set is stated over the declared attributes, which the stored record names.
        unverified, refusal = _consume_accepted(data, expected_identity, None)
        _raise(refusal)
        assert unverified is not None
        model = json.loads(unverified.summary_json())["declared_model"]
        records_json = _records(records)._wire(model)
    native, refusal = _consume_accepted(data, expected_identity, records_json, execution.cancel)
    _raise(refusal)
    accepted = AcceptedPreferences(native)
    accepted._replayed = records is not None or accepted.basis == "declared"
    return accepted


__all__ = [
    "AcceptedPreferences",
    "AnalysisAsOf",
    "BeliefSource",
    "ChoiceEstimate",
    "ChoiceModel",
    "ChoiceModelResult",
    "CompatibleSet",
    "CompatibleSetResult",
    "Conflict",
    "Forecast",
    "LearnedSet",
    "Option",
    "ParameterEstimate",
    "ParameterRange",
    "PerDecider",
    "Pooled",
    "Realized",
    "Record",
    "Records",
    "Region",
    "RegionCell",
    "ReviewItem",
    "ShapeSearch",
    "Stated",
    "Undetermined",
    "load",
]
