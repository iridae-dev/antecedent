"""The contract of a decision across trade-off ranges, its evaluation and its compile
against prepared studies, and the joint laws and mean sources it reads. Public names are
exported from :mod:`antecedent.preferences`."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np

from ._native import decide_preferences_joint as _decide_joint
from ._native import decide_preferences_means as _decide_means
from ._native import decide_preferences_structures as _decide_structures
from ._native import preference_chain_joint_law as _chain_joint_law
from ._native import preference_claim_joint_artifact as _claim_joint_artifact
from ._native import preference_claim_source as _claim_source
from ._native import preference_compile_across_studies as _compile_across
from ._native import preference_compile_for_study as _compile_for_study
from ._native import preference_contract_describe as _describe_contract
from ._native import preference_experiment_quantity as _experiment_quantity
from ._native import preference_experiment_source as _experiment_source
from ._native import preference_joint_law_of as _joint_law_of
from ._preference_declarations import (
    Action,
    Measure,
    Policy,
    StructuralPolicy,
    Structures,
    ValueModel,
    _constraint_node,
    _raise,
)
from ._preference_results import (
    ClassSampling,
    CompiledQuantity,
    DesignLawPlan,
    JointLawInput,
    JointLawPlan,
    PreferenceDecision,
    TradeOffRange,
    _class_sampling,
)
from ._scientific import ScientificResult
from ._validate import count, text
from .decision import Constraint, Criterion
from .decision import StructuralPolicy as StructuralPolicyKind
from .errors import CausalTypeError, CausalValueError
from .execution import Execution, resolve_execution
from .external import BoundExternalClaim
from .joint_distribution import JointDistributionArtifact, ScientificQuantity
from .outcomes import outcome_operation, require_success
from .results.sources import MeanSource

if TYPE_CHECKING:
    from .program_claims import ProgramBinding


def _options(
    tie_tolerance: float | None,
    vertex_cap: int | None,
    max_boxes: int | None,
    min_width_fraction: float | None,
) -> dict[str, Any]:
    values = {
        "tie_tolerance": tie_tolerance,
        "vertex_cap": vertex_cap,
        "max_boxes": max_boxes,
        "min_width_fraction": min_width_fraction,
    }
    return {key: value for key, value in values.items() if value is not None}


def _mean_wire(source: MeanSource | BoundExternalClaim) -> dict[str, Any]:
    if isinstance(source, BoundExternalClaim):
        identity = source.identity_fields
        return {
            "coordinates": [q._wire() for q in source.quantities],
            "means": [float(v) for v in source.values],
            "provider_id": str(identity["provider_id"]),
            "snapshot_id": str(identity["snapshot_id"]),
            "causal_contract_id": str(identity["causal_contract_id"]),
            "rng_id": "none:mean_grid",
        }
    source._validated_original()
    return {
        "coordinates": [q._wire() for q in source.coordinates],
        "means": list(source.means),
        "provider_id": source.provider_id,
        "snapshot_id": source.snapshot_id,
        "causal_contract_id": source.causal_contract_id,
        "rng_id": source.rng_id,
    }


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


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Contract:
    """A decision whose value comes from a model with explicit trade-offs.

    Give the model as ``model=`` (a :class:`ValueModel`), or as ``preferences=``: any object
    exposing ``_preference_wire()``, such as a :class:`ValueModel` or the accepted preferences
    of :mod:`antecedent.preference_learning` (which then also supply their measure, if any).
    ``criterion`` and ``constraints`` are those of :mod:`antecedent.decision`; a constraint reads
    attributes (build one with :func:`constraint`, or with ``decision.x(k)`` for attribute
    ``k``). ``policy`` says how the answer across the trade-offs is consumed.
    """

    actions: tuple[Action, ...]
    target_population: str
    model: ValueModel | None = None
    preferences: Any = None
    criterion: Criterion = field(default_factory=Criterion.expected_utility)
    policy: Policy = field(default_factory=Policy.invariant_best)
    constraints: tuple[Constraint, ...] = ()
    horizon: int = 0
    structural_policy: StructuralPolicy | StructuralPolicyKind = field(
        default_factory=StructuralPolicy.report_only
    )

    def __post_init__(self) -> None:
        if (self.model is None) == (self.preferences is None):
            raise CausalValueError("a Contract takes model= or preferences=, exactly one")
        if self.model is not None and not isinstance(self.model, ValueModel):
            raise CausalTypeError("Contract.model is a ValueModel")
        if self.preferences is not None and not callable(
            getattr(self.preferences, "_preference_wire", None)
        ):
            raise CausalTypeError(
                "Contract.preferences is a ValueModel or accepted preferences "
                "(an object with _preference_wire())"
            )
        if isinstance(self.actions, (str, bytes)) or not isinstance(self.actions, Iterable):
            raise CausalTypeError("Contract.actions is a sequence of Action")
        actions = tuple(self.actions)
        if any(not isinstance(a, Action) for a in actions):
            raise CausalTypeError("Contract.actions holds only Action values")
        constraints = tuple(self.constraints)
        if any(not isinstance(c, Constraint) for c in constraints):
            raise CausalTypeError("Contract.constraints holds decision.Constraint values")
        if not isinstance(self.criterion, Criterion):
            raise CausalTypeError("Contract.criterion is a decision.Criterion")
        if not isinstance(self.policy, Policy):
            raise CausalTypeError("Contract.policy is a Policy such as Policy.maximin()")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise CausalTypeError("Contract.horizon is an integer")
        structural = self.structural_policy
        if isinstance(structural, str):
            if structural not in get_args(StructuralPolicyKind):
                raise CausalValueError(
                    f"Contract.structural_policy is a StructuralPolicy or one of "
                    f"{get_args(StructuralPolicyKind)}"
                )
            structural = StructuralPolicy(structural)
        elif not isinstance(structural, StructuralPolicy):
            raise CausalTypeError(
                "Contract.structural_policy is a StructuralPolicy such as "
                "StructuralPolicy.maximin(), or its kind as a string"
            )
        object.__setattr__(self, "structural_policy", structural)
        text(self.target_population, "Contract.target_population", non_empty=True)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "constraints", constraints)

    @property
    def _structural_policy(self) -> StructuralPolicy:
        policy = self.structural_policy
        return policy if isinstance(policy, StructuralPolicy) else StructuralPolicy(policy)

    def _preferences(self) -> dict[str, Any]:
        source = self.model if self.model is not None else self.preferences
        return dict(source._preference_wire())

    def _wire(self) -> dict[str, Any]:
        model = self._preferences()["model"]
        attributes = {a["id"]: k for k, a in enumerate(model["attributes"])}
        return {
            "model": model,
            "actions": [a._wire(model["attributes"]) for a in self.actions],
            "criterion": self.criterion._wire(),
            "constraints": [
                {
                    **c._wire(),
                    "expr": _constraint_node(c.expr._wire_value, attributes),
                }
                for c in self.constraints
            ],
            "target_population": self.target_population,
            "horizon": self.horizon,
            "structural_policy": self._structural_policy.kind,
            "preference_policy": self.policy.kind,
        }

    def _described(self) -> dict[str, Any]:
        body, refusal = _describe_contract(json.dumps(self._wire()))
        _raise(refusal)
        assert body is not None
        return dict(json.loads(body))

    @property
    def identity(self) -> str:
        """Canonical digest of the contract, to retain for replay."""
        return str(self._described()["identity"])

    @property
    def route(self) -> Literal["exact", "certified"]:
        """``exact`` (linear programs) or ``certified`` (interval subdivision)."""
        return self._described()["route"]

    @property
    def parameter_ids(self) -> tuple[str, ...]:
        return tuple(p["id"] for p in self._preferences()["model"]["parameters"])

    def _measure(self, measure: Measure | None) -> dict[str, Any] | None:
        carried = self._preferences().get("measure")
        if measure is None:
            return carried
        if carried is not None:
            raise CausalValueError(
                "these accepted preferences carry their own measure; do not pass measure= too"
            )
        if not isinstance(measure, Measure):
            raise CausalTypeError("measure= is a preferences.Measure")
        return measure._wire(self._preferences()["model"]["parameters"])

    def _source(
        self,
        source: Any,
        *,
        program: ProgramBinding | None = None,
        outcome_units: str | None = None,
        dose_units: str | None = None,
        population: str = "target",
    ) -> Any:
        """The source to decide on: joint draws, a mean source or structures."""
        from .decision import _result_mean_source
        from .results.sources import adapt_mean_source, is_mean_producer

        source = require_success(source)
        if isinstance(source, Structures):
            return source
        if isinstance(source, (list, tuple)):
            return joint_law(*source)
        from .decision_evidence import JointEvidence

        if isinstance(source, JointEvidence):
            return experiment_means(source)
        from .propagation_chain import PropagationChain

        if isinstance(source, PropagationChain):
            return chain_law(source)
        if is_mean_producer(source):
            source = adapt_mean_source(source)
        cohort_source = getattr(source, "_cohort_mean_source", None)
        if cohort_source is not None:
            source = cohort_source()
        from .results._execution import ResultAPI

        if isinstance(source, ResultAPI):
            wire = json.dumps(self._wire())
            source = _result_mean_source(
                tuple(q for action in self.actions for q in action.quantities),
                lambda claim: claim._decision_source(
                    lambda: _claim_source(wire, claim._native),
                    lambda: _claim_joint_artifact(wire, claim._native),
                ),
                source,
                program=program,
                outcome_units=outcome_units,
                dose_units=dose_units,
                population=population,
            )
        if isinstance(source, (JointDistributionArtifact, MeanSource, BoundExternalClaim)):
            return source
        raise CausalTypeError(
            "preferences.Contract.evaluate reads a JointDistributionArtifact (aligned draws), a "
            "sequence of them whose rows pair, a PropagationChain with joint outputs, the "
            "JointEvidence of one experiment (its arm means), a "
            "MeanSource or BoundExternalClaim (means), Structures, or a supported analysis "
            f"result, not {type(source).__name__}"
        )

    @outcome_operation
    def compile(
        self,
        source: Any,
        *,
        draws: int | None = None,
        seed: int | None = None,
        shared: Mapping[tuple[int, int], Any] | None = None,
        refits: int | None = None,
        sampling: str | None = None,
        execution: Execution | None = None,
    ) -> CompiledDecision:
        """Bind this contract to the analysis that estimates its quantities and compile it.

        This is the way in. ``source`` is the analysis, prepared with
        :func:`antecedent.prepare`: one prepared study (its mean response curve, under its
        structure or each structure of its graph posterior), a sequence of prepared studies
        (one outcome each), or a compiled design of several outcomes
        (``joint_panel.prepare_native_joint_design``). Every quantity an action reads must be
        a coordinate the analysis targets. The decision is lowered into the analysis's own
        plan, so the :class:`CompiledDecision` states the quantities, the source the contract
        needs, the trade-off ranges and the identification before anything is fitted; what
        cannot be answered (a quantity not identified, a source the analysis cannot supply,
        studies that share observations without a declared covariance) is refused here.
        :meth:`CompiledDecision.execute` then runs each plan once.

        Several studies meet by their means when means suffice. A value that needs one law
        of their outcomes needs ``draws=`` (the aligned rows the propagation chain draws)
        and, for studies that share observations, ``shared=``: the declared covariance
        between the estimates of each such pair, keyed ``(i, j)`` by the studies' positions
        in ``source``. Each block is a 2-D array-like with one row per coordinate of study
        ``i`` and one column per coordinate of study ``j``, both in grid order; a block keyed
        ``(j, i)`` is read as its transpose. A static-regime design reads its arm means, or
        with ``refits=`` (at least 99) its aligned whole-subject refits. A cluster-randomized
        design or a matched-pair or switchback experiment reads its arm means; randomization
        alone does not identify the dependence of its arm-mean estimators across arms, so a
        value that needs one law of its outcomes also needs ``sampling=``, the assumption id
        of the premise that its independent units (clusters, pairs or sequences) were sampled
        independently from a larger population, with ``refits=`` for the independent-unit
        bootstrap. The premise enters the plan and the result's assumptions.

        ``seed`` seeds the chain's draws and the refits; by default it is the seed of
        ``execution`` (``Execution().seed`` when none is given), so ``execution=`` alone
        reproduces a compile. An argument the source does not read refuses with
        ``option_not_applicable`` rather than being dropped: ``draws=`` and ``shared=`` are
        read only for several studies, ``refits=`` and ``sampling=`` only for a design, and
        ``seed=`` only when something is drawn (``draws=``, ``shared=`` or ``refits=``).
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
        contract_json = json.dumps(self._wire())
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
        _raise(refusal)
        assert compiled is not None
        return CompiledDecision(self, compiled)

    @outcome_operation
    def evaluate(
        self,
        source: Any,
        *,
        measure: Measure | None = None,
        tie_tolerance: float | None = None,
        vertex_cap: int | None = None,
        max_boxes: int | None = None,
        min_width_fraction: float | None = None,
        program: ProgramBinding | None = None,
        outcome_units: str | None = None,
        dose_units: str | None = None,
        population: str = "target",
        execution: Execution | None = None,
    ) -> PreferenceDecision:
        """Decide across every admissible value of the trade-offs, on a source already in hand.

        The way in is :meth:`compile`, which binds the contract to the analysis, shows the plan
        and refuses what cannot be answered before anything is fitted. This is the lower layer
        that route calls: it reads a source the caller already holds.

        ``source`` is what :meth:`antecedent.decision.Contract.evaluate` reads (aligned joint
        draws; a mean source or bound external claim; a supported analysis result, read as its
        mean source with ``program=``, ``outcome_units=``, ``dose_units=`` and ``population=``
        as there), or :class:`Structures` to decide across causal structures as well. Several
        separately estimated outcomes are read as one law: a sequence of
        :class:`~antecedent.joint_distribution.JointDistributionArtifact` whose rows already pair
        (see :func:`joint_law`), or a :class:`~antecedent.propagation_chain.PropagationChain`
        whose declared joint outputs join them under the chain's declared dependence (see
        :func:`chain_law`). The :class:`~antecedent.decision_evidence.JointEvidence` of one
        experiment is read as its arm means (:func:`experiment_means`); name each action's
        inputs with :func:`experiment_quantity`. Means answer only a value affine in the
        attributes; a kinked or otherwise nonlinear term refuses.

        A mean source answers only the exact route (an affine value and an expectation
        criterion); anything else refuses with ``preference_evaluation.joint_draws_required``.
        ``measure`` weighs trade-off values for :meth:`Policy.weighted` and reports shares; it
        is refused when the accepted preferences already carry one. ``tie_tolerance``,
        ``vertex_cap``, ``max_boxes`` and ``min_width_fraction`` tune the routes. The decision
        reads the execution's cancellation token at its work boundaries; a cancelled run is
        refused with ``cancelled_no_claim``.
        """
        execution = resolve_execution(execution)
        resolved = self._source(
            source,
            program=program,
            outcome_units=outcome_units,
            dose_units=dose_units,
            population=population,
        )
        measure_wire = self._measure(measure)
        options = _options(tie_tolerance, vertex_cap, max_boxes, min_width_fraction)
        contract_json = json.dumps(self._wire())
        measure_json = None if measure_wire is None else json.dumps(measure_wire)
        options_json = json.dumps(options)
        if isinstance(resolved, Structures):
            claims = [c._wire() for c in resolved.claims]
            sources = [None if c.source is None else c.source._native for c in resolved.claims]
            body, refusal = _decide_structures(
                contract_json,
                measure_json,
                options_json,
                resolved.kind,
                json.dumps(claims),
                sources,
                execution.cancel,
            )
        elif isinstance(resolved, JointDistributionArtifact):
            body, refusal = _decide_joint(
                contract_json, measure_json, options_json, resolved._native, execution.cancel
            )
        else:
            body, refusal = _decide_means(
                contract_json,
                measure_json,
                options_json,
                json.dumps(_mean_wire(resolved)),
                execution.cancel,
            )
        _raise(refusal)
        assert body is not None
        return PreferenceDecision(self, resolved, json.loads(body), measure_wire, options)


class CompiledDecision(ScientificResult):
    """A decision across preferences compiled with the analysis that estimates it.

    Before anything is fitted it states: :attr:`contract_identity` and :attr:`bound_identity`;
    :attr:`quantities`, each quantity the actions read with the study that targets it;
    :attr:`representation`, the source the contract needs (``"mean"`` or ``"joint_draws"``),
    and :attr:`carrier`, what the plans demand of the analysis; :attr:`trade_offs`, each
    parameter's declared range and assumption; :attr:`joint_law`, the declared joint law of
    separately estimated outcomes; :attr:`design_law`, the resampling law of a compiled
    design; and :attr:`unidentified_mass` and :attr:`class_sampling` across structures.
    :meth:`inspect` returns everything the compile states, the compiled plans included, as
    nested mappings. :meth:`execute` runs each plan once and returns the
    :class:`PreferenceDecision`.
    """

    def __init__(self, contract: Contract, native: Any) -> None:
        self._contract = contract
        self._native = native

    @property
    def contract(self) -> Contract:
        return self._contract

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
        """Canonical digest of the preference contract (the same as :attr:`Contract.identity`)."""
        identity = self.inspect().get("contract_identity")
        return self._contract.identity if identity is None else str(identity)

    @property
    def route(self) -> Literal["exact", "certified"]:
        """The route the decision will be answered by."""
        return self.inspect()["demand"]["route"]

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
        """Each trade-off parameter with its declared range, as the plan states it."""
        stated = self._stated()
        parameters = (
            stated["parameters"]
            if stated is not None
            else self._contract._preferences()["model"]["parameters"]
        )
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

    @property
    def parameter_ranges(self) -> Mapping[str, tuple[float, float]]:
        """Each trade-off parameter's declared range, by id."""
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
    ) -> PreferenceDecision:
        """Run every compiled plan once and decide across the admissible trade-offs.

        ``measure`` and the tuning options are those of :meth:`Contract.evaluate`. The decision
        reads the execution's cancellation token at its work boundaries.
        """
        from .transport._controls import call_native

        measure_wire = self._contract._measure(measure)
        options = _options(tie_tolerance, vertex_cap, max_boxes, min_width_fraction)
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
        return PreferenceDecision(self._contract, source, json.loads(body), measure_wire, options)

    def __repr__(self) -> str:
        return f"<CompiledDecision {self.bound_identity[:12]}>"


def joint_law(*laws: JointDistributionArtifact) -> JointDistributionArtifact:
    """One joint law from laws whose rows already pair.

    Rows pair when every law has joint alignment and the same source, provider, generator,
    snapshot, causal contract, number of draws, meaning, weights, calibration and trust: row
    ``i`` of each is the same draw. Laws that were sampled separately do not pair and refuse
    (``joint_law_required``); join those under a declared dependence with a propagation chain
    and :func:`chain_law`. Nothing is resampled or re-estimated.
    """
    laws = tuple(require_success(law) for law in laws)
    if not laws or any(not isinstance(law, JointDistributionArtifact) for law in laws):
        raise CausalTypeError("joint_law takes one or more JointDistributionArtifact values")
    native, refusal = _joint_law_of([law._native for law in laws])
    _raise(refusal)
    assert native is not None
    return JointDistributionArtifact._from_native(native)


def chain_law(chain: Any) -> JointDistributionArtifact:
    """The joint output law of a sealed propagation chain, as aligned joint draws.

    The chain's record is resealed (every stage recomputed under its declared dependence) and
    its declared ``joint_outputs`` become the columns, so separately fitted results enter one
    decision with the dependence the chain states, not an assumed independence. A chain read
    neither as a posterior nor as a predictive law, or without joint outputs, refuses.
    """
    from .propagation_chain import PropagationChain

    chain = require_success(chain)
    if not isinstance(chain, PropagationChain):
        raise CausalTypeError(f"chain_law takes a PropagationChain, not {type(chain).__name__}")
    native, refusal = _chain_joint_law(chain._record, None)
    _raise(refusal)
    assert native is not None
    return JointDistributionArtifact._from_native(native)


def _evidence_wire(evidence: Any) -> str:
    from .decision_evidence import JointEvidence

    evidence = require_success(evidence)
    if not isinstance(evidence, JointEvidence):
        raise CausalTypeError(
            f"an experiment is a decision_evidence.JointEvidence, not {type(evidence).__name__}"
        )
    return json.dumps(dict(evidence.wire))


def experiment_quantity(
    evidence: Any,
    outcome: str,
    arm: Literal["treated", "control"],
    *,
    execution: Any = None,
) -> ScientificQuantity:
    """The coordinate of one outcome of an experiment in one arm, for an :class:`Action` input.

    ``evidence`` is the :class:`~antecedent.decision_evidence.JointEvidence` of one experiment;
    ``outcome`` names one of its outcome components and ``arm`` is ``treated`` or ``control``.
    The coordinate is the arm mean (functional ``mean``) in the experiment's population,
    regime and horizon, as the experiment's mean source (:func:`experiment_means`) carries it.
    """
    from ._panel_execution import run_panel

    if arm not in ("treated", "control"):
        raise CausalValueError("arm is 'treated' or 'control'")
    body, refusal = run_panel(
        execution, _experiment_quantity, _evidence_wire(evidence), outcome, arm == "treated"
    )
    _raise(refusal)
    assert body is not None
    return ScientificQuantity._from_wire(json.loads(body))


def experiment_means(evidence: Any, *, execution: Any = None) -> MeanSource:
    """The arm means of every outcome of one experiment, as one mean source.

    The experiment is rebuilt and checked against its compiled source first. Arm means are
    point estimates read as means only under exposure declared free of between-unit
    interference; any other exposure refuses. A mean source carries no joint law, so it answers
    the exact route (a value affine in the attributes, an expectation criterion) only.
    """
    from ._panel_execution import run_panel

    wire = _evidence_wire(evidence)
    body, refusal = run_panel(execution, _experiment_source, wire)
    _raise(refusal)
    assert body is not None
    fields_ = json.loads(body)
    return MeanSource(
        coordinates=tuple(ScientificQuantity._from_wire(q) for q in fields_["coordinates"]),
        means=tuple(fields_["means"]),
        provider_id=fields_["provider_id"],
        snapshot_id=fields_["snapshot_id"],
        causal_contract_id=fields_["causal_contract_id"],
        rng_id=fields_["rng_id"],
        evidence=(json.loads(wire)["evidence_id"],),
    )
