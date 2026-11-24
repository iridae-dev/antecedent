"""Compile a decision contract with the analysis that estimates its quantities: one prepared
study, several prepared studies or a compiled design. Public names are exported from
:mod:`antecedent.decision`."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ._decision_tradeoff import _options, tradeoff_measure
from ._decision_tradeoff_results import ClassSampling, TradeoffDecision, _class_sampling
from ._decision_value import Measure, _raise
from ._native import preference_compile_across_studies as _compile_across
from ._native import preference_compile_for_study as _compile_for_study
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from ._validate import count
from .decision import Decision, DecisionRefusal
from .errors import CausalTypeError, CausalValueError
from .execution import Execution, resolve_execution
from .joint_distribution import JointDistributionArtifact, ScientificQuantity
from .outcomes import outcome_operation, require_success


@dataclass(frozen=True, slots=True)
class CompiledQuantity:
    """One quantity the actions read, bound to the analysis that targets it.

    ``source`` is the position of the study in the compiled source (0 for one study or a
    design); ``read_by`` lists the ``(action, input)`` of every input that reads it.
    """

    quantity: ScientificQuantity
    source: int
    read_by: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class TradeOffRange:
    """One trade-off as the contract and the compiled plan state it: its units, its declared
    range (``lower == upper`` for a fixed value) and the assumption the range stands on."""

    id: str
    units: str
    lower: float
    upper: float
    assumption: str


@dataclass(frozen=True, slots=True)
class JointLawInput:
    """One column of a study's law that the joint law of several studies reads: the study's
    position, the canonical identity of the quantity and the chain stage it becomes."""

    study: int
    quantity: str
    stage: str


@dataclass(frozen=True, slots=True)
class JointLawPlan:
    """The joint law separately estimated outcomes are joined into, declared at compile.

    ``dependence`` is ``distinct_evidence`` or ``shared_evidence``; ``draws`` aligned rows are
    drawn under ``seed``; ``cross_blocks`` names the study pairs whose cross covariance was
    declared; ``inputs`` are the columns the chain reads, in output order.
    """

    dependence: str
    draws: int
    seed: int
    cross_blocks: tuple[tuple[int, int], ...]
    inputs: tuple[JointLawInput, ...]


@dataclass(frozen=True, slots=True)
class DesignLawPlan:
    """The resampling law a compiled design draws when the decision needs one law of its
    outcomes: ``static_regime_refits`` or ``independent_unit_bootstrap``, the unit resampled
    (``subject``, ``cluster``, ``pair`` or ``sequence``), the number of refits, their seed and
    the declared sampling premise (required for an independent-unit bootstrap)."""

    kind: str
    unit: str
    replicates: int
    seed: int
    sampling: str | None


def trade_off_ranges(parameters: Sequence[Mapping[str, Any]]) -> tuple[TradeOffRange, ...]:
    """Each trade-off of a value wire as a :class:`TradeOffRange`."""
    return tuple(
        TradeOffRange(
            id=str(p["id"]),
            units=str(p["units"]),
            lower=float(p["lower"]),
            upper=float(p["upper"]),
            assumption=str(p["assumption"]),
        )
        for p in parameters
    )


def _check_compile_arguments(
    source_kind: str,
    *,
    reads: Mapping[str, bool],
    given: Mapping[str, Any],
    seed: int | None,
    draws_anything: bool,
) -> None:
    """Refuse an argument the compile source does not read, instead of dropping it."""
    unread = [f"{name}=" for name, value in given.items() if value is not None and not reads[name]]
    if seed is not None and not draws_anything:
        unread.append("seed=")
    if unread:
        raise CausalValueError(
            f"{source_kind} does not read {', '.join(unread)}: draws= and shared= apply to "
            "several studies, refits= and sampling= to a compiled design, and seed= only when "
            "draws, shared covariance or refits are drawn",
            reason_code="option_not_applicable",
        )


def _coordinate_count(study: Any) -> int | None:
    """The number of coordinates of a study's mean response curve (its grid), when known."""
    query = getattr(study, "query", None)
    grid = getattr(query, "grid", None)
    if grid is None or getattr(query, "is_temporal", False):
        return None
    return len(grid)


def _cross_blocks(shared: Any, studies: Sequence[Any]) -> list[dict[str, Any]]:
    """The declared cross-study covariance blocks as the native wire: each pair of study
    positions with its first < second and its block flattened row-major."""
    if not isinstance(shared, Mapping):
        raise CausalTypeError(
            "shared= maps a pair of study positions (i, j) to a 2-D covariance block"
        )
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for key, block in shared.items():
        if not (
            isinstance(key, tuple)
            and len(key) == 2
            and all(isinstance(k, int) and not isinstance(k, bool) for k in key)
        ):
            raise CausalTypeError(f"shared= keys are pairs of study positions (i, j), not {key!r}")
        i, j = key
        if i == j or not (0 <= i < len(studies) and 0 <= j < len(studies)):
            raise CausalValueError(
                f"shared= key {key} names two different studies by their positions "
                f"0..{len(studies) - 1} in the source"
            )
        try:
            matrix = np.asarray(block, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise CausalValueError(
                f"shared[{key}] is a 2-D array of numbers (rows of study {i}'s coordinates by "
                f"columns of study {j}'s): {error}"
            ) from None
        rows, columns = _coordinate_count(studies[i]), _coordinate_count(studies[j])
        if (
            matrix.ndim != 2
            or (rows is not None and matrix.shape[0] != rows)
            or (columns is not None and matrix.shape[1] != columns)
        ):
            expected = tuple("?" if n is None else n for n in (rows, columns))
            raise CausalValueError(
                f"shared[{key}] has shape {matrix.shape}; it needs {expected}: one row per "
                f"coordinate of study {i} and one column per coordinate of study {j}, in grid order"
            )
        if i > j:
            i, j, matrix = j, i, matrix.T
        if (i, j) in blocks:
            raise CausalValueError(f"shared= declares the pair of studies {i} and {j} twice")
        blocks[(i, j)] = matrix
    return [
        {"first": i, "second": j, "covariance": [float(v) for v in matrix.reshape(-1)]}
        for (i, j), matrix in sorted(blocks.items())
    ]


def _refuse(form: str) -> Any:
    """The refusal class of a compile: an ordinary contract refuses as a decision, a contract
    across trade-offs as a decision across preferences."""
    if form == "ordinary":
        return lambda payload: raise_refusal(payload, DecisionRefusal)
    return _raise


def compile_contract(
    contract: Any,
    source: Any,
    *,
    draws: int | None = None,
    seed: int | None = None,
    shared: Mapping[tuple[int, int], Any] | None = None,
    refits: int | None = None,
    sampling: str | None = None,
    execution: Execution | None = None,
) -> CompiledDecision:
    """Bind ``contract`` to the analysis in ``source`` and compile it (see
    :meth:`antecedent.decision.Contract.compile`).

    ``contract`` decides as an ordinary contract when its ``_compile_form()`` is ``ordinary``:
    its ordinary wire travels as ``{"ordinary": wire}``. Otherwise its trade-off wire travels as
    it is.
    """
    from ._preference_design import compile_with_design, is_compiled_design
    from .design.ranking import _bound_native
    from .transport._controls import call_native

    source = require_success(source)
    execution = resolve_execution(execution)
    design = is_compiled_design(source)
    several = isinstance(source, (list, tuple))
    _check_compile_arguments(
        "a compiled design" if design else "several studies" if several else "one study",
        reads={"draws": several, "shared": several, "refits": design, "sampling": design},
        given={"draws": draws, "shared": shared, "refits": refits, "sampling": sampling},
        seed=seed,
        draws_anything=(several and (draws is not None or shared is not None))
        or (design and refits is not None),
    )
    seed_value = execution.seed if seed is None else count(seed, "seed", minimum=0)
    form = contract._compile_form()
    wire = {"ordinary": contract._wire()} if form == "ordinary" else contract._tradeoff_wire()
    contract_json = json.dumps(wire)
    if design:
        compiled, refusal = compile_with_design(
            source,
            contract_json,
            refits=refits,
            seed=seed_value,
            sampling=sampling,
            execution=execution,
        )
    elif several:
        studies = [require_success(study) for study in source]
        natives = [_bound_native(study) for study in studies]
        if shared is not None:
            kind = "shared_evidence"
        elif draws is not None:
            kind = "distinct_evidence"
        else:
            kind = "undeclared"
        dependence = {
            "kind": kind,
            "draws": 2000 if draws is None else count(draws, "draws", minimum=0),
            "seed": seed_value,
            "cross": _cross_blocks(shared or {}, studies),
        }
        compiled, refusal = call_native(
            execution,
            _compile_across,
            natives,
            contract_json,
            json.dumps(dependence),
            seed="seed",
        )
    else:
        compiled, refusal = call_native(
            execution, _compile_for_study, _bound_native(source), contract_json, seed="seed"
        )
    _refuse(form)(refusal)
    assert compiled is not None
    return CompiledDecision(contract, compiled, form=form)


class CompiledDecision(ScientificResult):
    """A decision contract compiled with the analysis that estimates it.

    Before anything is fitted it states: :attr:`contract_identity` and :attr:`bound_identity`;
    :attr:`quantities`, each quantity the actions read with the study that targets it;
    :attr:`representation`, the source the contract needs (``"mean"`` or ``"joint_draws"``),
    and :attr:`carrier`, what the plans demand of the analysis; :attr:`trade_offs`, each
    trade-off's declared range and assumption (empty for an ordinary contract);
    :attr:`joint_law`, the declared joint law of separately estimated outcomes;
    :attr:`design_law`, the resampling law of a compiled design; and :attr:`unidentified_mass`
    and :attr:`class_sampling` across structures. :meth:`inspect` returns everything the
    compile states, the compiled plans included, as nested mappings. :meth:`execute` runs each
    plan once and returns the decision: a :class:`~antecedent.decision.Decision` for an
    ordinary contract (compiled with a study of one structure, several studies or a design; a
    study over a graph posterior refuses with
    ``preference_compile.ordinary_structures_not_compiled``), and a
    :class:`~antecedent.decision.TradeoffDecision` across trade-offs.
    """

    def __init__(
        self, contract: Any, native: Any, *, form: Literal["ordinary", "tradeoff"] = "tradeoff"
    ) -> None:
        self._contract = contract
        self._native = native
        self._form = form

    @property
    def contract(self) -> Any:
        """The contract as declared."""
        return self._contract

    @property
    def form(self) -> Literal["ordinary", "tradeoff"]:
        """``ordinary`` when the compiled decision is an ordinary contract, else ``tradeoff``."""
        return self._form

    @property
    def bound_identity(self) -> str:
        """Identity of the contract bound to the analysis's causal contracts."""
        return str(self._native.bound_identity())

    def inspect(self) -> Mapping[str, Any]:
        """What the compile states, read without fitting anything."""
        return MappingProxyType(json.loads(self._native.describe()))

    @property
    def plans(self) -> tuple[Mapping[str, Any], ...]:
        """Every compiled plan, each with the decision in its evidence layer."""
        return tuple(
            MappingProxyType(plan) for plan in json.loads(self._native.describe())["plans"]
        )

    def _stated(self) -> Mapping[str, Any] | None:
        """The decision facts of the first plan that carries them."""
        for plan in json.loads(self._native.describe())["plans"]:
            decision = (plan.get("evidence") or {}).get("decision") or {}
            preference = decision.get("preference")
            if isinstance(preference, Mapping):
                return preference
        return None

    @property
    def contract_identity(self) -> str:
        """Canonical digest of the contract (the same as its ``identity``)."""
        identity = self.inspect().get("contract_identity")
        return self._contract.identity if identity is None else str(identity)

    @property
    def route(self) -> Literal["exact", "certified"] | None:
        """The route a decision across trade-offs will be answered by; ``None`` for an
        ordinary contract."""
        route = self.inspect()["demand"].get("route")
        return None if route is None else route

    @property
    def representation(self) -> str:
        """The source the contract needs of the analysis: ``"mean"`` when means answer it
        (an affine value and an expectation), otherwise ``"joint_draws"``."""
        return str(self.inspect()["demand"]["representation"])

    @property
    def carrier(self) -> str | None:
        """What each compiled plan demands of its analysis (``"mean"``, ``"decision_draws"``,
        ...)."""
        carrier = self.inspect()["demand"].get("carrier")
        return None if carrier is None else str(carrier)

    @property
    def quantities(self) -> tuple[CompiledQuantity, ...]:
        """Each quantity the actions read, in the compile's order, with its source study."""
        return tuple(
            CompiledQuantity(
                quantity=ScientificQuantity._from_wire(q["quantity"]),
                source=int(q["source"]),
                read_by=tuple((str(a), str(b)) for a, b in q["read_by"]),
            )
            for q in self.inspect()["demand"]["quantities"]
        )

    @property
    def trade_offs(self) -> tuple[TradeOffRange, ...]:
        """Each trade-off with its declared range, as the plan states it; empty for an ordinary
        contract."""
        if self._form == "ordinary":
            return ()
        stated = self._stated()
        parameters = (
            stated["parameters"]
            if stated is not None
            else self._contract._preferences()["model"]["parameters"]
        )
        return trade_off_ranges(parameters)

    @property
    def parameter_ranges(self) -> Mapping[str, tuple[float, float]]:
        """Each trade-off's declared range, by id."""
        return MappingProxyType({t.id: (t.lower, t.upper) for t in self.trade_offs})

    @property
    def joint_law(self) -> JointLawPlan | None:
        """The declared joint law of separately estimated outcomes, or ``None`` when one
        study, a design or the outcomes' means answer the decision."""
        joint = self.inspect().get("joint_law")
        if joint is None:
            return None
        return JointLawPlan(
            dependence=str(joint["dependence"]),
            draws=int(joint["draws"]),
            seed=int(joint["seed"]),
            cross_blocks=tuple((int(a), int(b)) for a, b in joint.get("cross_blocks") or ()),
            inputs=tuple(
                JointLawInput(
                    study=int(column["study"]),
                    quantity=str(column["quantity"]),
                    stage=str(column["stage"]),
                )
                for column in joint["inputs"]
            ),
        )

    @property
    def design_law(self) -> DesignLawPlan | None:
        """The resampling law a compiled design draws, or ``None`` when its arm means answer
        the decision or the source is not a design."""
        stated = self._stated()
        law = None if stated is None else stated.get("design_law")
        if law is None:
            return None
        return DesignLawPlan(
            kind=str(law["kind"]),
            unit=str(law["unit"]),
            replicates=int(law["replicates"]),
            seed=int(law["seed"]),
            sampling=None if law.get("sampling") is None else str(law["sampling"]),
        )

    @property
    def unidentified_mass(self) -> float:
        """Share of the structural mass under which the quantities are not identified."""
        return float(self.inspect()["unidentified_mass"])

    @property
    def class_sampling(self) -> tuple[ClassSampling, ...]:
        """The draw of every equivalence class sampled rather than enumerated (else empty)."""
        return tuple(_class_sampling(wire) for wire in self.inspect().get("class_sampling") or ())

    def _summary_fields(self) -> dict[str, Any]:
        described = self.inspect()
        return {
            "bound_identity": described["bound_identity"],
            "representation": described["demand"]["representation"],
            "structures": len(described["structures"]),
        }

    def to_dict(self) -> dict[str, Any]:
        return dict(json.loads(self._native.describe()))

    @outcome_operation
    def execute(
        self,
        measure: Measure | None = None,
        execution: Execution | None = None,
        *,
        tie_tolerance: float | None = None,
        vertex_cap: int | None = None,
        max_boxes: int | None = None,
        min_width_fraction: float | None = None,
    ) -> Any:
        """Run every compiled plan once and decide.

        An ordinary contract returns a :class:`~antecedent.decision.Decision`; ``measure`` and
        the tuning options apply only across trade-offs and are refused with
        ``option_not_applicable`` otherwise. Across trade-offs it returns a
        :class:`~antecedent.decision.TradeoffDecision`; ``measure`` and the tuning options are
        those of :meth:`antecedent.decision.Contract.evaluate`. The decision reads the
        execution's cancellation token at its work boundaries.
        """
        from .transport._controls import call_native

        options = _options(tie_tolerance, vertex_cap, max_boxes, min_width_fraction)
        if self._form == "ordinary":
            unread = [
                f"{name}="
                for name, value in (("measure", measure), *options.items())
                if value is not None
            ]
            if unread:
                raise CausalValueError(
                    f"{', '.join(unread)} apply to a decision across trade-offs; this compiled "
                    "decision is an ordinary contract",
                    reason_code="option_not_applicable",
                )
            body, refusal, law = call_native(
                execution, self._native.execute, None, json.dumps({}), seed="seed"
            )
            raise_refusal(refusal, DecisionRefusal)
            assert body is not None
            return self._ordinary_result(json.loads(body), law)
        measure_wire = tradeoff_measure(self._contract, measure)
        body, refusal, law = call_native(
            execution,
            self._native.execute,
            None if measure_wire is None else json.dumps(measure_wire),
            json.dumps(options),
            seed="seed",
        )
        _raise(refusal)
        assert body is not None
        source = self if law is None else JointDistributionArtifact._from_native(law)
        return TradeoffDecision(self._contract, source, json.loads(body), measure_wire, options)

    def _ordinary_result(self, body: Mapping[str, Any], law: Any) -> Any:
        ordinary = self._contract._ordinary()
        source = self if law is None else JointDistributionArtifact._from_native(law)
        return Decision(ordinary, source, body, declared=self._contract)

    def __repr__(self) -> str:
        return f"<CompiledDecision {self.bound_identity[:12]}>"


__all__ = [
    "CompiledDecision",
    "CompiledQuantity",
    "DesignLawPlan",
    "JointLawInput",
    "JointLawPlan",
    "TradeOffRange",
]
