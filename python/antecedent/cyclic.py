# SPDX-License-Identifier: MIT OR Apache-2.0
"""Nonlinear cyclic equilibrium SCMs with retained Rust execution and replay.

Equations have the form ``x[i] = intercept + activation(weights @ x) + u[i]``.
Noise covariance is joint and may be correlated. Fitting requires independent
observational/interventional equilibrium replicates with known hard clamps.
Answers are conditional on the specified/fitted model; no nonparametric causal
identification or nominal interval calibration is implied.

Typed worlds name variables (:class:`World` of ``Set``/``Shift``/``Soft`` and
:class:`InitialState`) on :class:`LinearCyclicModel`, :meth:`EquilibriumModel.worlds` and
:meth:`OdeSystem.worlds`; :meth:`PreparedCyclicFamily.interventions` returns them keyed by
world id. :func:`prepare_experimental_recovery` recovers a linear law from clamp experiments
under a declared draw origin, and ``infer`` on :class:`OdeSystem`,
:class:`OrnsteinUhlenbeck` and :class:`OuMechanisms` attaches bootstrap, observed-information
or posterior uncertainty to a dynamic fit.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from . import _native
from ._artifact_protocol import (
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._cyclic_read_models import CyclicPage as CyclicPage
from ._cyclic_read_models import RecoveredCyclicLaw as RecoveredCyclicLaw
from ._cyclic_typed import (
    AsymptoticNormal,
    CyclicInterventions,
    DynamicBootstrap,
    DynamicMethod,
    DynamicPosterior,
    DynamicQuantity,
    DynamicQuantitySelector,
    EvidenceSchema,
    ExperimentalFitOptions,
    ExperimentRegime,
    FlatCoefficients,
    InitialState,
    InverseWishartDisturbance,
    KnownDisturbance,
    LinearCyclicModel,
    NormalCoefficients,
    ObservedInformation,
    OdeState,
    OuMechanismCovariance,
    OuMechanismMean,
    OuStationaryCovariance,
    OuStationaryMean,
    RecoveredDrawSelector,
    RecoveredResponseSelector,
    RecoveredWorldAnswer,
    RecoveryBootstrap,
    RecoveryPosterior,
    RecoveryPrior,
    RecoveryUncertainty,
    SamplingDesign,
    World,
    WorldAnswer,
    WorldSelector,
    _limits,
    _seed,
    _with_execution,
    prepare_experimental_recovery,
)
from ._cyclic_typed import EvidenceVariable as EvidenceVariable
from ._freeze import freeze, thaw
from ._scientific import ScientificResult
from .errors import CausalError, CausalValueError, StructuredRefusal
from .execution import Execution
from .intervention import Set, Shift, Soft
from .outcomes import (
    Refusal,
    is_scientific_refusal,
    outcome_from_error,
    outcome_operation,
    require_success,
)


def _reseed(request, seed):
    if "seed" in request:
        request["seed"] = seed
    for key in ("sampling", "policy", "inference", "method", "request"):
        if isinstance(request.get(key), dict):
            _reseed(request[key], seed)


class CyclicRefusal(StructuredRefusal):
    """A cyclic-model request refused for a scientific or portability reason.

    A :class:`~antecedent.errors.StructuredRefusal`: ``reason_code`` (alias ``code``) is the
    registered runtime-refusal code and :attr:`detail` the stable ``<family>.<slot>`` detail.
    Malformed arguments raise :class:`~antecedent.errors.CausalValueError` instead, and
    refusals reported by the native layer arrive as the shared causal error classes with their
    ``reason=<code>:`` prefix read into ``reason_code``. :attr:`next_attempt` is the
    sentence naming what to declare instead, when the refusal names one; it is also the
    :attr:`remedy` unless a separate remedy is given.
    """

    def __init__(
        self,
        message: str = "",
        *,
        reason_code: str | None = None,
        detail: str | None = None,
        remedy: str | None = None,
        next_attempt: str | None = None,
    ) -> None:
        text = f"{detail}: {message}" if detail else message
        super().__init__(
            {
                "code": reason_code,
                "detail": detail or "",
                "message": message,
                "remedy": remedy if remedy is not None else next_attempt,
            },
            text=text,
        )
        #: What to declare instead, when the refusal names it.
        self.next_attempt: str | None = next_attempt


def _cyclic_numeric_rows(values, width):
    import numpy as np

    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise CausalValueError("cyclic buffers must contain real numbers")
    array = np.asarray(array, dtype=np.float64)
    if len(values) == 0:
        return array.reshape(0, width)
    if array.ndim != 2 or array.shape[1] != width:
        raise CausalValueError("cyclic numeric buffer shape")
    return array


@dataclass(frozen=True)
class Equation:
    """Named additive-noise mechanism, optionally a compositional expression."""

    id: str
    intercept: float
    weights: tuple[float, ...]
    activation: Literal["identity", "tanh", "sin", "relu"] = "tanh"
    expression: dict[str, Any] | None = None
    parameters: tuple[float, ...] = ()

    @classmethod
    def compositional(
        cls,
        id: str,
        order: int,
        expression: dict[str, Any],
        *,
        parameters: tuple[float, ...] = (),
        intercept: float = 0.0,
    ) -> Equation:
        """Build a portable expression: add/multiply/tanh/sin/softplus and reads.

        Leaves are ``{"op": "variable", "variable": 0}``, ``parameter`` reads,
        or finite ``constant`` values. Add uses ``terms``; multiply uses
        ``factors``; unary operations use ``argument``. Rust checks bindings,
        expression limits and domain-wide derivative bounds before execution.
        """
        return cls(id, intercept, (0.0,) * order, "identity", expression, parameters)


@dataclass(frozen=True)
class SolvePolicy:
    """Numerical policy and solver admission; initialization never supplies a selection law.

    The budget fields bound a sparse solve instead of a fixed variable count. Exceeding one
    is a refusal that names its own code (``solver_jacobian_nnz_budget``,
    ``solver_block_size_budget``, ``solver_factor_fill_budget``,
    ``solver_iteration_budget``, ``solver_wall_clock_budget`` or, beyond the dense
    disturbance-covariance admission, ``solver_model_order_budget``). A numerical failure
    inside the budgets (non-convergence, singular Jacobian) stays a retained world outcome.

    ``max_nnz`` bounds the structural nonzeros of the free Jacobian, ``max_block_size`` the
    order of one strongly connected block, ``max_factor_fill`` the sparse-LU entries of one factorization, ``max_total_iterations``
    the iterations summed over blocks and ``max_wall_clock_ms`` a deadline (``0`` is already
    expired). Memory is a physical control: set it on
    ``Execution(limits=ResourceLimits(...))``, which the solver's estimated bytes are admitted
    against. ``condition_warning`` flags ``ill_conditioned`` in the solve evidence when the
    Jacobian condition estimate exceeds it; ``spectral_iterations`` bounds the power iteration
    that bounds each block's spectral radius.
    """

    tolerance: float = 1e-10
    max_iterations: int = 1000
    require_contraction: bool = True
    max_nnz: int = 8_000_000
    max_block_size: int = 4096
    max_factor_fill: int = 16_000_000
    max_total_iterations: int = 10_000_000
    max_wall_clock_ms: int | None = None
    condition_warning: float = 1e8
    spectral_iterations: int = 200


@dataclass(frozen=True)
class FitOptions:
    """Local coordinate likelihood fit with fixed joint noise covariance."""

    initial_step: float = 0.2
    tolerance: float = 1e-5
    max_sweeps: int = 500
    l1_penalty: float = 0.0


@dataclass(frozen=True)
class CyclicResult(ScientificResult):
    """Immutable native-owned scientific output; materialization is explicit."""

    _native: Any = field(repr=False)
    execution_facts: Any = field(default=None, repr=False, compare=False)
    _prepared: Any = field(default=None, repr=False, compare=False)
    execution: Execution = field(default_factory=Execution, repr=False, compare=False)
    _metadata: Any = field(default=None, init=False, repr=False, compare=False)
    _inspection: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from ._cyclic_read_models import result_inspection
        from .errors import CausalTypeError

        if not isinstance(self._native, _native._CyclicExecution):
            raise CausalTypeError("CyclicResult requires a retained native execution owner")
        if not isinstance(self.execution, Execution):
            raise CausalTypeError("execution must be an Execution")
        metadata = json.loads(self._native.metadata_json())
        object.__setattr__(self, "_metadata", freeze(metadata))
        object.__setattr__(
            self,
            "_inspection",
            result_inspection(metadata, stage="consumed" if metadata["loaded"] else "executed"),
        )

    def _summary_fields(self) -> dict[str, Any]:
        return {
            name: self._metadata.get(name)
            for name in ("identity", "family", "claim", "calibration", "world_count")
        }

    @property
    def identity(self) -> str:
        """Exact native request identity, read without report conversion."""
        return str(self._native.identity())

    @property
    def identities(self):
        contract = self._metadata.get("contract") or {}
        return freeze(
            {
                **contract.get("identities", {}),
                **{
                    key: contract[key]
                    for key in ("cyclic_solution", "recovery", "lowered_program_identity")
                    if contract.get(key) is not None
                },
                "source_identity": self._metadata.get("source_identity"),
            }
        )

    @property
    def provenance(self):
        return freeze(
            {
                "source_identity": self._metadata.get("source_identity"),
                "scope": self._metadata["scope"],
                "native_contract": self._metadata.get("contract"),
                "native_contract_is_summary": self._metadata.get("contract") is not None,
                "graph_metadata": self._metadata["diagnostics"].get("graph"),
            }
        )

    def inspect(self):
        return self._inspection

    @_with_execution
    def to_dict(self, *, execution: Execution | None = None) -> dict[str, Any]:
        """Explicitly materialize the complete native execution report."""
        return dict(json.loads(self._native.report_json()))

    @artifact_export
    @_with_execution
    def export(self, *, execution: Execution | None = None) -> ExportedArtifact:
        """Encode the retained native result only at this explicit boundary."""
        assert execution is not None  # _with_execution resolves the retained policy.
        return artifact_from_receipt(self, self._native.export(**_limits(execution)))

    def assumptions(self, *, offset: int = 0, limit: int = 64):
        from ._cyclic_read_models import native_page

        return native_page(self._native.assumptions_json, "assumptions", offset, limit)

    @property
    def law(self):
        from ._cyclic_read_models import RecoveredCyclicLaw

        summary = json.loads(self._native.recovery_summary_json())
        if summary is None:
            from .errors import CausalTypeError

            raise CausalTypeError("this cyclic result has no experimental recovered law")
        return RecoveredCyclicLaw(
            self._native,
            self.execution,
            summary,
            _stage="consumed" if self._metadata["loaded"] else "executed",
        )

    @outcome_operation
    def interventions(self):
        """Select native worlds without fitting, solving or serializing them again."""
        names = getattr(self._prepared, "variable_names", None)
        if names is None:
            names = self._native.variable_names()
        return CyclicInterventions._from_native(
            self._native, names, self.execution_facts, self._prepared, execution=self.execution
        )

    def numeric(
        self,
        world: int,
        *,
        row: int = 0,
        draw: int = 0,
        counterfactual: bool = False,
        field: str = "values",
    ):
        """Borrow one selected native readonly buffer, retaining its exact owner."""
        selected = self._native.world_values(
            world, row=row, draw=draw, counterfactual=counterfactual, field=field
        )
        return None if selected is None else memoryview(selected)

    @outcome_operation
    @_with_execution
    def evidence(self, schema, selectors, *, execution: Execution | None = None):
        """Project retained evidence with explicit quantities, units and population."""
        names = getattr(self._prepared, "variable_names", None)
        if names is None:
            names = self._native.variable_names()
        wire = [item._wire(names) if hasattr(item, "_wire") else item for item in selectors]
        assert execution is not None  # _with_execution resolves the retained policy.
        return CyclicEvidence(
            self._native.evidence(
                json.dumps(
                    schema._wire() if isinstance(schema, EvidenceSchema) else schema,
                    allow_nan=False,
                ),
                json.dumps(wire, allow_nan=False),
                **_limits(execution),
            ),
            execution=execution,
            _prepared=self._prepared,
        )


@outcome_operation
@_with_execution
@artifact_consumer(
    proof="cyclic.consume_artifact: independent native equilibrium request/identity verification and numerical refit/replay",
    replayable=True,
)
def consume_artifact(
    artifact: bytes,
    *,
    expected_identity: str,
    execution: Execution | None = None,
) -> CyclicResult:
    """Check identity/integrity and recompute in Rust, including any retained fit."""
    assert execution is not None
    limits = _limits(execution)
    artifact = container_bytes(artifact)
    owner, facts = _native.consume_cyclic_system(artifact, expected_identity, **limits)
    return CyclicResult(owner, facts, execution=execution)


@dataclass(frozen=True)
class EquilibriumModel:
    """Serializable nonlinear mechanisms and row-major disturbance covariance."""

    equations: tuple[Equation, ...]
    noise_covariance: tuple[float, ...]

    @_with_execution
    def isolate(
        self,
        domain: list[dict[str, float]],
        noise: list[float],
        *,
        clamps: list[dict[str, Any]] | None = None,
        max_boxes: int = 10_000,
        minimum_width: float = 1e-8,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Bounded root isolation for eligible polynomial mechanism compositions.

        The compiler preserves exact coefficients, including FMA product tails.
        Smooth non-polynomial operations and ReLU domains crossing a kink refuse
        this path. Unresolved boxes remain explicit; no root probabilities are
        inferred from their counts or solver initialization.
        """
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "equilibrium_roots",
                    "model": asdict(self),
                    "domain": domain,
                    "noise": noise,
                    "clamps": clamps or [],
                    "policy": {"max_boxes": max_boxes, "minimum_width": minimum_width},
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def fit_adam(
        self,
        parameters: list[dict[str, Any]],
        training: list[dict[str, Any]],
        validation: list[dict[str, Any]],
        *,
        policy: AdamPolicy | None = None,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Projected nonlinear likelihood fitting with explicit validation stopping.

        Known intervention targets and all free-system Jacobians remain in the
        likelihood. Validation stopping, training-score stationarity and epoch
        exhaustion are separate facts. Final test regimes must stay outside this
        request; a minibatch seed never supplies an equilibrium selection law.
        """
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "adam_fit",
                    "model": asdict(self),
                    "parameters": parameters,
                    "training": training,
                    "validation": validation,
                    "policy": dict(asdict(policy or AdamPolicy()), seed=_seed()),
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def fit_gradient(
        self,
        parameters: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        *,
        learning_rate: float = 0.1,
        score_tolerance: float = 1e-6,
        max_iterations: int = 1000,
        l1_penalty: float = 0.0,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Fit local expression/legacy parameters with the exact Jacobian score."""
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "gradient_fit",
                    "model": asdict(self),
                    "parameters": parameters,
                    "rows": rows,
                    "policy": {
                        "learning_rate": learning_rate,
                        "score_tolerance": score_tolerance,
                        "max_iterations": max_iterations,
                        "l1_penalty": l1_penalty,
                    },
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def prepare(
        self,
        worlds: list[list[dict[str, Any]]],
        noise: list[list[float]],
        *,
        initial: list[float] | None = None,
        factual: list[float] | None = None,
        policy: SolvePolicy | None = None,
        parameters: list[dict[str, Any]] | None = None,
        rows: list[dict[str, Any]] | None = None,
        fit_options: FitOptions | None = None,
        execution: Execution | None = None,
    ) -> PreparedCyclicSystem:
        """Retain aligned intervention worlds and optional joint-likelihood fit.

        A clamp is ``{"variable": index, "value": number}``. A data row is
        ``{"values": [...], "clamps": [...]}``; parameter coordinates are
        ``{"kind": "weight", "equation": i, "parent": j}`` or
        ``{"kind": "intercept", "equation": i}``.
        """
        assert execution is not None
        if (parameters is None) != (rows is None):
            raise CausalValueError("parameters and rows must be supplied together")
        fit = None
        if parameters is not None:
            fit = {
                "parameters": parameters,
                "rows": rows,
                "options": asdict(fit_options or FitOptions()),
            }
        request = {
            "model": asdict(self),
            "fit": fit,
            "worlds": worlds,
            "noise": noise,
            "initial": initial if initial is not None else [0.0] * len(self.equations),
            "policy": asdict(policy or SolvePolicy()),
            "factual": factual,
        }
        return PreparedCyclicSystem._from_request(request, execution=execution)

    @_with_execution
    def worlds(
        self,
        worlds: list[World],
        noise: list[list[float]],
        *,
        variables: list[str] | None = None,
        initial: list[float] | None = None,
        policy: SolvePolicy | None = None,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Typed worlds over shared disturbance draws, named by variable.

        ``variables`` defaults to the equation ids: equation ``k`` is the mechanism of
        variable ``k``. Every world solves each row of ``noise`` (one joint disturbance
        realization per row, in the variables' units); a failed draw keeps its slot and its
        share of failed mass. ``initial`` is a numerical guess, never a causal initial state
        or a selection law. Compilation solves nothing; a world the family cannot take is
        refused with its next attempt.
        """
        assert execution is not None
        from ._cyclic_typed import equilibrium_worlds

        return equilibrium_worlds(
            self,
            worlds,
            noise,
            variables=variables,
            initial=initial,
            policy=policy,
            execution=execution,
        )


@dataclass(frozen=True)
class PreparedCyclicSystem:
    """Frozen retained request; Rust validates and executes its complete program."""

    request_json: str = field(repr=False)
    execution: Execution = field(default_factory=Execution, repr=False, compare=False)
    _handle: Any = field(init=False, repr=False, compare=False)

    def __repr__(self) -> str:
        return "PreparedCyclicSystem()"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_handle",
            self.execution._call(_native.prepare_cyclic_system, self.request_json),
        )

    @classmethod
    @_with_execution
    def _from_request(
        cls,
        request: dict[str, Any],
        *,
        execution: Execution | None = None,
    ):
        assert execution is not None
        limits = _limits(execution)
        dimension = len(request["model"]["equations"])
        noise = _cyclic_numeric_rows(request["noise"], dimension)
        metadata = dict(request, noise=[])
        rows = [] if request["fit"] is None else request["fit"]["rows"]
        observations = _cyclic_numeric_rows([row["values"] for row in rows], dimension)
        if request["fit"] is not None:
            metadata["fit"] = dict(request["fit"], rows=[dict(row, values=[]) for row in rows])
        handle = _native.prepare_cyclic_system_buffers(
            json.dumps(metadata, allow_nan=False),
            noise,
            observations,
            **limits,
        )
        result = object.__new__(cls)
        object.__setattr__(result, "request_json", None)
        object.__setattr__(result, "_handle", handle)
        return result

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if name == "request_json" and value is None:
            value = object.__getattribute__(self, "_handle").request_json()
            object.__setattr__(self, name, value)
        return value

    def inspect(self) -> dict[str, Any]:
        """Inspect the common native prepared handle's fit count and SCC schedules."""
        result: dict[str, Any] = json.loads(self._handle.inspect())
        return result

    def contract(self) -> dict[str, Any]:
        """The contract of this handle, derived through the ``system`` family it is carried as.

        Returns the layered identities (target, identification, identification product,
        program, inference binding, observation, data snapshot) and the
        ``cyclic.solution.v1`` tag as lowercase hex, the identification kind (always
        ``specified_model_conditional`` here) and the typed layer payloads a family export
        writes. Deriving it compiles schedules and does not fit. The contract grants no
        scientific license.
        """
        result: dict[str, Any] = json.loads(self._handle.contract())
        return result

    @_with_execution
    def to_family(
        self,
        *,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """The same frozen request as a ``system`` family program.

        The family shares rebinding, evidence projection, keyed replay and fit reuse with
        every other family. Compilation does not fit; both views share their native fit
        owner, including a fit first performed through either view. Equation ``k`` is
        named by its equation id.
        """
        assert execution is not None
        handle = self._handle.to_family(
            **_limits(execution),
        )
        request = json.loads(self.request_json)
        return PreparedCyclicFamily._from_handle(
            json.dumps({"family": "system", "request": request}, allow_nan=False),
            tuple(equation["id"] for equation in request["model"]["equations"]),
            handle,
            execution,
        )

    @_with_execution
    def reconfigure(
        self,
        worlds: list[list[dict[str, Any]]] | None = None,
        noise: list[list[float]] | None = None,
        *,
        initial: list[float] | None = None,
        model: EquilibriumModel | None = None,
        policy: SolvePolicy | None = None,
        execution: Execution | None = None,
    ) -> PreparedCyclicSystem:
        """Branch scientific inputs and share only compatible native fitting inputs."""
        assert execution is not None
        request = json.loads(self.request_json)
        _reseed(request, execution.seed)
        for key, value in (("worlds", worlds), ("noise", noise), ("initial", initial)):
            if value is not None:
                request[key] = value
        if model is not None:
            request["model"] = asdict(model)
        if policy is not None:
            request["policy"] = asdict(policy)
        result = object.__new__(PreparedCyclicSystem)
        object.__setattr__(result, "request_json", None)
        object.__setattr__(
            result, "_handle", self._handle.rebind(json.dumps(request, allow_nan=False))
        )
        return result

    @_with_execution
    def rebind(
        self, rows: list[dict[str, Any]], *, execution: Execution | None = None
    ) -> PreparedCyclicSystem:
        """Replace training observations through a new preparation and refit."""
        assert execution is not None
        observations = _cyclic_numeric_rows(
            [row["values"] for row in rows], len(self._handle.initial_state())
        )
        handle = self._handle.refresh_buffers(
            json.dumps([dict(row, values=[]) for row in rows], allow_nan=False),
            observations,
            **_limits(execution),
        )
        result = object.__new__(PreparedCyclicSystem)
        object.__setattr__(result, "request_json", None)
        object.__setattr__(result, "_handle", handle)
        return result

    @outcome_operation
    @_with_execution
    def execute(
        self,
        *,
        execution: Execution | None = None,
    ) -> CyclicResult:
        """Execute the retained native product and produce its independent replay artifact."""
        assert execution is not None
        owner, facts = self._handle.execute(**_limits(execution))
        return CyclicResult(owner, facts, self, execution)

    @outcome_operation
    @_with_execution
    def posterior(
        self,
        *,
        prior_sd: float,
        chains: int = 4,
        warmup: int = 500,
        draws: int = 1000,
        proposal_scale: float = 0.05,
        execution: Execution | None = None,
    ) -> dict[str, Any]:
        """Conditional-structure posterior with an explicit normal parameter prior.

        Prior support is restricted to the certified likelihood domain. Inspect
        ``diagnostics_pass`` before interpreting draws; structural-selection and
        covariance uncertainty are not included.
        """
        assert execution is not None
        seed = _seed()
        result: dict[str, Any] = json.loads(
            _native.cyclic_parameter_posterior(
                self.request_json,
                prior_sd,
                chains,
                warmup,
                draws,
                proposal_scale,
                seed,
                **_limits(execution),
            )
        )
        return result


@dataclass(frozen=True)
class OdePolicy:
    """Adaptive continuous-time local-error and work policy."""

    absolute_tolerance: float = 1e-9
    relative_tolerance: float = 1e-7
    initial_step: float = 0.01
    max_steps: int = 100_000


@dataclass(frozen=True)
class PreparedCyclicFamily:
    """Retained dynamics/constraint request; no static SCM coercion is implied."""

    request_json: str = field(repr=False)
    variable_names: tuple[str, ...] | None = field(default=None, repr=False, compare=False)
    execution: Execution = field(default_factory=Execution, repr=False, compare=False)
    _handle: Any = field(default=None, init=False, repr=False, compare=False)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False, compare=False)
    _inspection_facts: Any = field(default=None, init=False, repr=False, compare=False)
    _retained_law_declaration: Any = field(default=None, init=False, repr=False, compare=False)
    _scientific_refusal: Refusal | None = field(default=None, init=False, repr=False, compare=False)

    def __repr__(self) -> str:
        return "PreparedCyclicFamily()"

    def __post_init__(self) -> None:
        try:
            self.execution._call(self._prepare, execution=self.execution)
        except CausalError as error:
            if not is_scientific_refusal(error):
                raise
            object.__setattr__(self, "_scientific_refusal", outcome_from_error(error).refusal)
        self._cache_inspection_facts()

    @property
    def refusal(self) -> Refusal | None:
        """The native scientific preparation refusal, if no program was licensed."""
        return self._scientific_refusal

    @classmethod
    def _from_refused(cls, request_json, variable_names, refusal, execution, *, retained_law=None):
        """Retain the same immutable declarations and actual typed native refusal."""
        result = object.__new__(cls)
        for name, value in (
            ("request_json", request_json),
            ("variable_names", variable_names),
            ("execution", execution),
            ("_handle", None),
            ("_lock", threading.Lock()),
            ("_retained_law_declaration", retained_law),
            ("_scientific_refusal", refusal),
        ):
            object.__setattr__(result, name, value)
        result._cache_inspection_facts()
        return result

    @classmethod
    def _from_retained_law(cls, law, worlds, names, outcome, handle, execution):
        result = object.__new__(cls)
        for name, value in (
            ("request_json", None),
            ("variable_names", tuple(names)),
            ("execution", execution),
            ("_handle", handle),
            ("_lock", threading.Lock()),
            ("_retained_law_declaration", (law, tuple(worlds), tuple(names), outcome)),
            ("_scientific_refusal", None),
            (
                "_inspection_facts",
                freeze(
                    {
                        "draw_origin": law._metadata["draw_origin"],
                        "law_meaning": law._metadata["law_meaning"],
                        "retained_law": True,
                        "worlds": tuple(world.id for world in worlds),
                        "draw_count": law._metadata["draw_count"],
                        "failed_mass": law._metadata["failed_mass"],
                    }
                ),
            ),
        ):
            object.__setattr__(result, name, value)
        return result

    def _cache_inspection_facts(self) -> None:
        from ._cyclic_typed import family_facts

        if self._scientific_refusal is not None:
            family = (
                "experimental_recovery"
                if self._retained_law_declaration is not None
                else json.loads(self.request_json)["family"]
            )
            object.__setattr__(
                self,
                "_inspection_facts",
                freeze(
                    {
                        "family": family,
                        "stage": "preparation_refused",
                        "identity": None,
                        "identity_unavailable": "native_program_not_prepared",
                        "variable_count": None
                        if self.variable_names is None
                        else len(self.variable_names),
                        "applicable_operations": ("inspect", "estimate", "reconfigure")
                        if self._retained_law_declaration is not None
                        else ("inspect", "estimate", "reconfigure", "rebind"),
                    }
                ),
            )
            return
        native = json.loads(self._handle.inspect())
        facts = family_facts(json.loads(self.request_json), None, native.get("design_unresolved"))
        object.__setattr__(self, "_inspection_facts", freeze(facts))

    @classmethod
    def _from_handle(cls, request_json, variable_names, handle, execution):
        result = object.__new__(cls)
        for name, value in (
            ("request_json", request_json),
            ("variable_names", variable_names),
            ("execution", execution),
            ("_handle", handle),
            ("_lock", threading.Lock()),
            ("_retained_law_declaration", None),
            ("_scientific_refusal", None),
        ):
            object.__setattr__(result, name, value)
        result._cache_inspection_facts()
        return result

    def inspect(self) -> dict[str, Any]:
        """Inspect retained work; this preflight never fits or executes a model.

        Beyond identity, family and retained bytes, the inspection names the draw origin and
        its evidence meaning (``sampling_law``, ``parameter_posterior``, or None for supplied
        draws), the unresolved directions of a recovered law, the request fields whose change
        invalidates a retained fit (``fit_dependencies``; ``*`` is the whole request), and the
        world ids of a typed-world family.

        For experimental recovery, ``design_unresolved`` and ``unresolved_directions``
        keep the clamp design's original unresolved rows. A completed execution's
        actual recovered law, draw statuses and unresolved directions belong to
        ``result.law``; execution never rewrites the preparation's declarations.
        """
        if self._scientific_refusal is not None:
            return dict(thaw(self._inspection_facts), refusal=self._scientific_refusal)
        facts = dict(json.loads(self._handle.inspect()))
        facts.update(thaw(self._inspection_facts))
        if self.variable_names is not None:
            facts["variables"] = list(self.variable_names)
        return facts

    @_with_execution
    @outcome_operation
    def interventions(
        self,
        *,
        execution: Execution | None = None,
    ) -> Any:
        """Execute and return the intervention worlds keyed by scientific world id.

        Returns :class:`CyclicInterventions`: a :class:`WorldAnswer` per world of a specified
        model (certificate, solve route, failed and withheld mass) or a
        :class:`RecoveredWorldAnswer` per world of a recovered law (draw origin kept). A family
        without intervention worlds raises :class:`CyclicRefusal` naming the typed families.
        """
        assert execution is not None
        limits = _limits(execution)
        with self._lock:
            self._prepare(execution=execution)
            owner, facts = self._handle.execute(**limits)
        return CyclicInterventions._from_native(
            owner, self.variable_names, facts, self, execution=execution
        )

    @_with_execution
    @outcome_operation
    def evidence(
        self,
        schema: EvidenceSchema | dict[str, Any],
        selectors: list[Any],
        *,
        execution: Execution | None = None,
    ) -> CyclicEvidence:
        """Execute the retained program and project its native evidence.

        Schema declares ``id``, ``population`` and ordered ``(name, units)``
        variables. Typed selectors bind the actual world, time and uncertainty
        port. Original diagnostics and failed slots remain part of the evidence.
        Selectors are wire mappings or typed selectors (:class:`WorldSelector`,
        :class:`RecoveredResponseSelector`, :class:`RecoveredDrawSelector`,
        :class:`DynamicQuantitySelector`); a recovered law's port follows its draw origin.
        """
        assert execution is not None
        limits = _limits(execution)
        wire = [
            item._wire(self.variable_names) if hasattr(item, "_wire") else item
            for item in selectors
        ]
        with self._lock:
            self._prepare(execution=execution)
        return CyclicEvidence(
            self._handle.evidence(
                json.dumps(
                    schema._wire() if isinstance(schema, EvidenceSchema) else schema,
                    allow_nan=False,
                ),
                json.dumps(wire, allow_nan=False),
                **limits,
            ),
            execution=execution,
            _prepared=self,
        )

    @_with_execution
    @outcome_operation
    def identified_functional(
        self,
        *,
        execution: Execution | None = None,
    ) -> IdentifiedCyclicFunctional:
        """Retain the checked functional and its exact graph/assumption source."""
        assert execution is not None
        if self._scientific_refusal is not None:
            self._scientific_refusal.raise_error()
        if self._retained_law_declaration is not None:
            raise CausalValueError("a recovered law is not a graph-identification preparation")
        source = json.loads(self.request_json)
        if source.get("family") != "graph" or source["request"].get("operation") not in (
            "identify",
            "adjust",
            "calculus",
        ):
            raise CausalValueError("expected a graph identification preparation")
        result = require_success(self.execute(execution=execution))
        answer = result.to_dict()["answer"]
        # Serde's tagged answer stores the operation report in `result`.
        graph = answer.get("result", answer.get("answer", answer))
        if (
            graph.get("status") not in ("identified", "verified_rule")
            or graph.get("program") is None
            or graph.get("fully_observational") is False
        ):
            # The request was valid; the native procedure did not supply an
            # observational functional. Preserve its exact standing and proof
            # trace rather than reporting an argument error or nonidentifiability.
            raise StructuredRefusal(
                {
                    "code": "effect_not_identified",
                    "detail": "cyclic.observational_functional_unavailable",
                    "stage": "identified_functional",
                    "message": "the native cyclic procedure did not return an observational functional",
                    "offending": result.identity,
                    "native_identification": graph,
                    "source_identity": result.identity,
                    "remedy": "inspect the retained native status, theorem premises and search limits before choosing another identification request",
                }
            )
        return IdentifiedCyclicFunctional(source["request"], result.identity, graph["program"])

    @_with_execution
    def contract(
        self,
        *,
        execution: Execution | None = None,
    ) -> dict[str, Any]:
        """The contract of this handle, derived without fitting, sampling or solving.

        Returns the layered identities (target, identification, identification product,
        program, inference binding, observation, data snapshot), the ``cyclic.solution.v1``
        tag and, for an experimental recovery, the recovery identity, all as lowercase hex;
        the identification kind (``specified_model_conditional``, ``experimental_recovery``
        or ``cyclic_ioscm``, never a graphical product); and the typed layer payloads the
        family export writes in its contract section. The contract grants no scientific
        license.
        """
        assert execution is not None
        if self._scientific_refusal is not None:
            return {
                "available": False,
                "reason": "native_program_not_prepared",
                "refusal": self._scientific_refusal,
            }
        result: dict[str, Any] = json.loads(self._native_handle(execution=execution).contract())
        return result

    @_with_execution
    def _native_handle(self, *, execution: Execution | None = None) -> Any:
        """The compiled native handle, compiling it first when needed (no fit)."""
        assert execution is not None
        with self._lock:
            self._prepare(execution=execution)
            return self._handle

    @_with_execution
    def rebind(
        self, data: dict[str, Any], *, execution: Execution | None = None
    ) -> PreparedCyclicFamily:
        """Replace declared observation fields on a new native preparation."""
        assert execution is not None
        if self._retained_law_declaration is not None:
            raise CausalValueError(
                "a retained law has no rebindable observations; prepare the original experiment declarations to fit new data"
            )
        request = json.loads(self.request_json)
        target = request["request"] if request["family"] == "system" else request
        if request["family"] == "dynamic_inference":
            target = request["problem"]["problem"]
        for key, value in data.items():
            if key not in {
                "rows",
                "regimes",
                "samples",
                "observations",
                "training",
                "validation",
                "heldout",
                "noise",
                "data",
            }:
                raise CausalValueError(f"{key!r} is not an observation field")
            if key == "rows" and request["family"] == "system":
                if target["fit"] is None:
                    raise CausalValueError("rows require a fitted system")
                target["fit"]["rows"] = value
            elif (
                request["family"] == "dynamic_inference"
                and key == "observations"
                and "data" in target
            ):
                target["data"] = value
            elif key in target:
                target[key] = value
            else:
                raise CausalValueError(f"this family has no {key!r} observations")
        return self._branch(json.dumps(request, allow_nan=False), execution=execution)

    @_with_execution
    def reconfigure(
        self,
        *,
        worlds: list[World] | None = None,
        scientific: dict[str, Any] | None = None,
        method: DynamicMethod | None = None,
        uncertainty: RecoveryUncertainty | None = None,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Branch scientific declarations or named worlds, retaining compatible fits."""
        assert execution is not None
        if self._retained_law_declaration is not None:
            law, original_worlds, names, original_outcome = self._retained_law_declaration
            if method is not None or uncertainty is not None:
                raise CausalValueError(
                    "changing a retained law's inference requires the original experiment preparation"
                )
            declarations = {} if scientific is None else dict(scientific)
            if set(declarations) - {"outcome"}:
                raise CausalValueError("retained-law reconfiguration supports worlds and outcome")
            return law.prepare(
                original_worlds if worlds is None else worlds,
                variables=names,
                outcome=declarations.get("outcome", original_outcome),
                execution=execution,
            )
        from ._cyclic_typed import _worlds

        request = json.loads(self.request_json)
        _reseed(request, execution.seed)
        if method is not None:
            from ._cyclic_typed import _method_wire

            if request["family"] != "dynamic_inference":
                raise CausalValueError("method requires a dynamic inference family")
            request["method"] = _method_wire(method, request["method"]["design"])
        if uncertainty is not None:
            from ._cyclic_typed import _inference_wire

            if request["family"] != "experimental_recovery" or self.variable_names is None:
                raise CausalValueError("uncertainty requires an experimental recovery")
            observed = {unit for regime in request["regimes"] for unit in regime["units"]}
            retained = [
                name
                for i, name in enumerate(self.variable_names)
                if i in observed and i not in request["dropped"]
            ]
            design = request["inference"].get("design", request["options"].get("sampling_design"))
            request["inference"] = _inference_wire(uncertainty, design, retained)
        if scientific:
            for key, value in scientific.items():
                if key not in request or key in {
                    "family",
                    "rows",
                    "regimes",
                    "observations",
                    "training",
                    "validation",
                }:
                    raise CausalValueError(
                        f"{key!r} is not a scientific declaration of this family"
                    )
                request[key] = value
        if worlds is not None:
            if self.variable_names is None or "worlds" not in request:
                raise CausalValueError("this family has no named intervention worlds")
            request["worlds"] = _worlds(
                worlds, {name: i for i, name in enumerate(self.variable_names)}
            )
        return self._branch(json.dumps(request, allow_nan=False), execution=execution)

    @_with_execution
    def _branch(
        self, request_json: str, *, execution: Execution | None = None
    ) -> PreparedCyclicFamily:
        """Recompile changed inputs, reusing only compatible statistical products."""
        assert execution is not None
        if self._handle is None:
            return PreparedCyclicFamily(request_json, self.variable_names, execution)
        try:
            with self._lock:
                handle = self._handle.rebind(request_json)
        except CausalError as error:
            if not is_scientific_refusal(error):
                raise
            refusal = outcome_from_error(error).refusal
            return PreparedCyclicFamily._from_refused(
                request_json, self.variable_names, refusal, execution
            )
        return PreparedCyclicFamily._from_handle(
            request_json, self.variable_names, handle, execution
        )

    @_with_execution
    def _prepare(
        self,
        *,
        execution: Execution | None = None,
    ) -> None:
        assert execution is not None
        limits = _limits(execution)
        if self._scientific_refusal is not None:
            self._scientific_refusal.raise_error()
        if self._handle is None:
            object.__setattr__(
                self,
                "_handle",
                _native.prepare_cyclic_family(self.request_json, **limits),
            )

    @outcome_operation
    @_with_execution
    def execute(
        self,
        *,
        execution: Execution | None = None,
    ) -> CyclicResult:
        """Execute the retained program and encode its result, reusing compatible fits."""
        assert execution is not None
        limits = _limits(execution)
        with self._lock:
            self._prepare(execution=execution)
            owner, facts = self._handle.execute(**limits)
        return CyclicResult(owner, facts, self, execution)


@_with_execution
def prepare_recovered_linear(
    nodes: list[int],
    draws: list[dict[str, Any]],
    worlds: list[dict[str, Any]],
    *,
    outcome: int,
    source_id: str,
    certifying_evidence: list[str],
    execution: Execution | None = None,
) -> PreparedCyclicFamily:
    """Prepare a recovered linear law on the existing cyclic family handle.

    ``draws`` are the retained coefficient and disturbance law, including failed
    and inadmissible draws. ``worlds`` are hard clamps ``{"clamp_nodes": [...],
    "clamp_values": [...]}`` indexed into ``nodes``. Preparation does not fit.
    The law stays conditional on the clamp-experiment recovery that produced it;
    compilation grants no scientific license.
    """
    assert execution is not None
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "recovered_linear",
                "nodes": nodes,
                "draws": draws,
                "worlds": worlds,
                "outcome": outcome,
                "source_id": source_id,
                "certifying_evidence": certifying_evidence,
            },
            allow_nan=False,
        ),
        execution=execution,
    )


@outcome_operation
@_with_execution
@artifact_consumer(
    proof="cyclic.consume_family_artifact: independent native family contract/identity verification and numerical refit/replay",
    replayable=True,
)
def consume_family_artifact(
    artifact: bytes,
    *,
    expected_identity: str,
    execution: Execution | None = None,
) -> CyclicResult:
    """Recompute a dynamic/constraint answer with independent identity checking."""
    assert execution is not None
    limits = _limits(execution)
    artifact = container_bytes(artifact)
    owner, facts = _native.consume_cyclic_family(artifact, expected_identity, **limits)
    return CyclicResult(owner, facts, execution=execution)


@dataclass(frozen=True)
class OdeSystem:
    """Drift ``intercept + activation(weights @ state) - decay * state``."""

    equations: tuple[Equation, ...]
    decay: tuple[float, ...]

    @_with_execution
    def prepare(
        self,
        initial: list[float],
        times: list[float],
        *,
        schedule: list[dict[str, Any]] | None = None,
        policy: OdePolicy | None = None,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Retain an initial-value problem and time-indexed sustained clamps."""
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "ode",
                    "model": asdict(self),
                    "initial": initial,
                    "times": times,
                    "schedule": schedule or [],
                    "policy": asdict(policy or OdePolicy()),
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def worlds(
        self,
        initial: list[float],
        times: list[float],
        worlds: list[World],
        *,
        variables: list[str] | None = None,
        initial_laws: list[str] | None = None,
        policy: OdePolicy | None = None,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Typed worlds of this initial-value problem, named by variable.

        ``variables`` defaults to the equation ids and ``initial_laws`` (the declared
        initial-state law of each state, replaced by :class:`InitialState`) to the variable
        names. A ``Set`` is a sustained clamp from the first time; a ``Soft`` replaces or
        changes the named mechanism. A disturbance ``Shift`` is refused with its next attempt:
        this family has no structural disturbance. Trajectories are numerical integrations,
        never certified solutions; times are in the system's own units.
        """
        assert execution is not None
        from ._cyclic_typed import ode_worlds

        return ode_worlds(
            self,
            initial,
            times,
            worlds,
            variables=variables,
            initial_laws=initial_laws,
            policy=policy,
            execution=execution,
        )

    @_with_execution
    def infer(
        self,
        parameters: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        *,
        method: DynamicMethod,
        sampling_design: SamplingDesign,
        quantities: list[DynamicQuantity] | tuple[()] = (),
        integration: OdePolicy | None = None,
        learning_rate: float = 0.1,
        score_tolerance: float = 1e-6,
        max_iterations: int = 1000,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Fit drift once, then attach sampling-law, information or posterior uncertainty.

        ``observations`` are independent trajectories (the sampling units), as in
        :meth:`fit`. ``method`` is :class:`DynamicBootstrap` (whole-procedure refits),
        :class:`ObservedInformation` (delta method, only for a regular likelihood) or
        :class:`DynamicPosterior`. ``quantities`` (:class:`OdeState`) are evaluated at every
        replicate or draw. The point fit is shared across methods: rebinding only the method
        or quantities reuses it. ``sampling_design`` is the study's declared design; the
        bootstrap resamples the unit it names. Structure and observation variances are
        specified, not estimated.
        """
        assert execution is not None
        from ._cyclic_typed import infer_ode

        return infer_ode(
            self,
            parameters,
            observations,
            method=method,
            quantities=quantities,
            sampling_design=sampling_design,
            integration=integration,
            learning_rate=learning_rate,
            score_tolerance=score_tolerance,
            max_iterations=max_iterations,
            execution=execution,
        )

    @_with_execution
    def fit(
        self,
        parameters: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        *,
        integration: OdePolicy | None = None,
        learning_rate: float = 0.1,
        score_tolerance: float = 1e-6,
        max_iterations: int = 1000,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Fit drift jointly using analytic sensitivities and known observation variances."""
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "ode_fit",
                    "model": asdict(self),
                    "parameters": parameters,
                    "observations": observations,
                    "integration": asdict(integration or OdePolicy()),
                    "policy": {
                        "learning_rate": learning_rate,
                        "score_tolerance": score_tolerance,
                        "max_iterations": max_iterations,
                    },
                },
                allow_nan=False,
            ),
            execution=execution,
        )


@dataclass(frozen=True)
class OrnsteinUhlenbeck:
    """Linear stochastic drift about a center, with diffusion covariance."""

    drift: tuple[float, ...]
    center: tuple[float, ...]
    diffusion: tuple[float, ...]

    @_with_execution
    def stationary(
        self,
        *,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Request stationary covariance with a checked Lyapunov stability witness."""
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "ou_stationary",
                    "model": asdict(self),
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def transient(
        self,
        initial_mean: list[float],
        initial_covariance: list[float],
        times: list[float],
        *,
        policy: OdePolicy | None = None,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Retain the explicit initial law and finite-horizon moment request."""
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "ou_transient",
                    "model": asdict(self),
                    "initial_mean": initial_mean,
                    "initial_covariance": initial_covariance,
                    "times": times,
                    "policy": asdict(policy or OdePolicy()),
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def fit(
        self,
        parameters: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        *,
        policy: FitOptions | None = None,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Fit stationary drift/center under known regimes and specified diffusion.

        A numerical fit does not establish that stationary observations identify
        every requested drift coordinate.
        """
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "ou_fit",
                    "model": asdict(self),
                    "parameters": parameters,
                    "rows": rows,
                    "policy": asdict(policy or FitOptions()),
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def infer(
        self,
        parameters: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        *,
        method: DynamicMethod,
        sampling_design: SamplingDesign,
        quantities: list[DynamicQuantity] | tuple[()] = (),
        options: FitOptions | None = None,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Fit stationary drift/center once, then attach its declared uncertainty.

        ``rows`` are independent stationary replicates under known sustained clamps (the
        sampling units). ``method`` is :class:`DynamicBootstrap`,
        :class:`ObservedInformation` or :class:`DynamicPosterior`; ``quantities``
        (:class:`OuStationaryMean`, :class:`OuStationaryCovariance`) are evaluated at every
        replicate or draw. Diffusion is specified, not estimated. An observed-information
        covariance exists only for a regular likelihood; otherwise its reason is reported.
        """
        assert execution is not None
        from ._cyclic_typed import infer_ou

        return infer_ou(
            self,
            parameters,
            rows,
            method=method,
            quantities=quantities,
            sampling_design=sampling_design,
            options=options,
            execution=execution,
        )


@dataclass(frozen=True)
class PolynomialSystem:
    """Named polynomial residuals with explicit finite variable domains.

    Each equation is ``{"id": name, "terms": [{"coefficient": c,
    "powers": [p0, ...]}, ...]}``; each domain is ``{"lower": a, "upper": b}``.
    """

    variables: tuple[str, ...]
    equations: tuple[dict[str, Any], ...]
    domain: tuple[dict[str, float], ...]

    def replace_equation(self, equation: str, variable: int, value: float) -> PolynomialSystem:
        """Replace one named equation by ``variable=value``, retaining other equations."""
        if not 0 <= variable < len(self.variables):
            raise CausalValueError("variable index out of range")
        matches = [i for i, item in enumerate(self.equations) if item["id"] == equation]
        if len(matches) != 1:
            raise CausalValueError("equation identity must resolve uniquely")
        powers = [0] * len(self.variables)
        powers[variable] = 1
        equations = list(self.equations)
        equations[matches[0]] = {
            "id": equation,
            "terms": [
                {"coefficient": 1.0, "powers": powers},
                {"coefficient": -value, "powers": [0] * len(self.variables)},
            ],
        }
        return PolynomialSystem(self.variables, tuple(equations), self.domain)

    @_with_execution
    def prepare(
        self,
        *,
        max_boxes: int = 10_000,
        minimum_width: float = 1e-8,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Retain bounded root isolation; unresolved boxes never become certified roots."""
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "polynomial",
                    "model": asdict(self),
                    "policy": {"max_boxes": max_boxes, "minimum_width": minimum_width},
                },
                allow_nan=False,
            ),
            execution=execution,
        )


@dataclass(frozen=True)
class CyclicGraph:
    """Original directed feedback and distinct latent-confounding edges."""

    nodes: int
    directed: tuple[tuple[int, int], ...]
    bidirected: tuple[tuple[int, int], ...] = ()

    @_with_execution
    def identify(
        self,
        outcomes: list[int],
        treatments: list[int],
        *,
        io_scm: bool,
        scc_product_density: bool,
        continuous: bool = True,
        max_steps: int = 1000,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Generalized cyclic ID with explicit theorem premises and bounded search."""
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "graph",
                    "request": {
                        "operation": "identify",
                        "graph": asdict(self),
                        "outcomes": outcomes,
                        "treatments": treatments,
                        "io_scm": io_scm,
                        "scc_product_density": scc_product_density,
                        "continuous": continuous,
                        "max_steps": max_steps,
                    },
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def calculus(
        self,
        rule: Literal["observation", "exchange", "action"],
        x: list[int],
        y: list[int],
        z: list[int],
        w: list[int],
        *,
        io_scm: bool,
        standard_measurable_spaces: bool,
        max_checks: int = 1000,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Check Theorem 7.2 with indicators, fixed inputs and distinct bounded failure.

        The functional retains source and executable roots. A verified rewrite
        with remaining intervention factors does not become an identified effect.
        """
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "graph",
                    "request": {
                        "operation": "calculus",
                        "graph": asdict(self),
                        "rule": rule,
                        "x": x,
                        "y": y,
                        "z": z,
                        "w": w,
                        "io_scm": io_scm,
                        "standard_measurable_spaces": standard_measurable_spaces,
                        "max_checks": max_checks,
                    },
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def adjust(
        self,
        outcomes: list[int],
        treatments: list[int],
        adjustment: list[int],
        *,
        io_scm: bool,
        standard_measurable_spaces: bool,
        continuous: bool = True,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Derive the executable cyclic backdoor adjustment functional when licensed."""
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "graph",
                    "request": {
                        "operation": "adjust",
                        "graph": asdict(self),
                        "outcomes": outcomes,
                        "treatments": treatments,
                        "adjustment": adjustment,
                        "continuous": continuous,
                        "io_scm": io_scm,
                        "standard_measurable_spaces": standard_measurable_spaces,
                    },
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def backdoor(
        self,
        outcomes: list[int],
        treatments: list[int],
        adjustment: list[int],
        *,
        io_scm: bool,
        scc_product_density: bool,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Check the sufficient sigma-backdoor criterion; failure is not impossibility."""
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "graph",
                    "request": {
                        "operation": "backdoor",
                        "graph": asdict(self),
                        "outcomes": outcomes,
                        "treatments": treatments,
                        "adjustment": adjustment,
                        "io_scm": io_scm,
                        "scc_product_density": scc_product_density,
                    },
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def oracle_discovery(
        self,
        *,
        simple_scm: bool,
        sigma_faithful: bool,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Exact sigma-CI reference reproduction; this does not learn from samples."""
        assert execution is not None
        return _discovery_request(
            [[0.0] * 8 for _ in range(self.nodes)],
            simple_scm=simple_scm,
            sigma_faithful=sigma_faithful,
            independent_equilibrium_replicates=True,
            oracle=asdict(self),
            ci="parcorr",
            alpha=0.05,
            max_cond_size=self.nodes,
            pds_max_nodes=10_000,
        )


@_with_execution
def _discovery_request(
    columns: list[list[float]],
    *,
    simple_scm: bool,
    sigma_faithful: bool,
    independent_equilibrium_replicates: bool,
    oracle: dict[str, Any] | None,
    ci: str,
    alpha: float,
    max_cond_size: int,
    pds_max_nodes: int,
    execution: Execution | None = None,
) -> PreparedCyclicFamily:
    assert execution is not None
    seed = _seed()
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "graph",
                "request": {
                    "operation": "discover",
                    "seed": seed,
                    "columns": columns,
                    "oracle": oracle,
                    "ci": ci,
                    "alpha": alpha,
                    "max_cond_size": max_cond_size,
                    "pds_max_nodes": pds_max_nodes,
                    "simple_scm": simple_scm,
                    "sigma_faithful": sigma_faithful,
                    "independent_equilibrium_replicates": independent_equilibrium_replicates,
                },
            },
            allow_nan=False,
        ),
        execution=execution,
    )


@_with_execution
def discover_cyclic(
    columns: list[list[float]],
    *,
    simple_scm: bool,
    sigma_faithful: bool,
    independent_equilibrium_replicates: bool,
    ci: str = "parcorr",
    alpha: float = 0.05,
    max_cond_size: int | None = None,
    pds_max_nodes: int = 10_000,
    execution: Execution | None = None,
) -> PreparedCyclicFamily:
    """FCI for cyclic equivalence classes with the chosen existing CI procedure.

    The CI procedure must suit the data-generating family. Partial correlation
    is not an independence oracle for arbitrary nonlinear systems. All engine
    truncation diagnostics and the PAG inducing-path interpretation are retained.
    """
    assert execution is not None
    return _discovery_request(
        columns,
        simple_scm=simple_scm,
        sigma_faithful=sigma_faithful,
        independent_equilibrium_replicates=independent_equilibrium_replicates,
        oracle=None,
        ci=ci,
        alpha=alpha,
        max_cond_size=len(columns) if max_cond_size is None else max_cond_size,
        pds_max_nodes=pds_max_nodes,
        execution=execution,
    )


@_with_execution
def learn_structure(
    model: EquilibriumModel,
    training: list[dict[str, Any]],
    *,
    heldout: list[dict[str, Any]] | None = None,
    threshold: float = 0.05,
    policy: FitOptions | None = None,
    execution: Execution | None = None,
    bootstrap_replicates: int = 0,
    design: Literal["fixed_assignments", "randomized_within_targets"] = "fixed_assignments",
) -> PreparedCyclicFamily:
    """Select a sparse parametric equilibrium topology and refit it jointly.

    Supply a positive L1 penalty. Held-out intervention regimes evaluate the final
    fit and never tune selection. Uncertainty conditional on the selected model
    must not be read as uncertainty for the complete selection procedure.
    """
    assert execution is not None
    seed = _seed()
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "structure_fit",
                "seed": seed,
                "model": asdict(model),
                "training": training,
                "heldout": heldout or [],
                "threshold": threshold,
                "bootstrap_replicates": bootstrap_replicates,
                "design": design,
                "policy": asdict(policy or FitOptions(l1_penalty=1.0)),
            },
            allow_nan=False,
        ),
        execution=execution,
    )


@dataclass(frozen=True)
class SamplingPolicy:
    """Retained independent chains, warmup, proposal scale, and replay seed."""

    chains: int = 4
    warmup: int = 500
    draws: int = 1000
    proposal_scale: float = 0.05


@_with_execution
def posterior_responses(
    model: EquilibriumModel,
    parameters: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    worlds: list[list[dict[str, Any]]],
    noise: list[list[float]],
    *,
    prior_sd: float,
    initial: list[float] | None = None,
    sampling: SamplingPolicy | None = None,
    solve_policy: SolvePolicy | None = None,
    execution: Execution | None = None,
) -> PreparedCyclicFamily:
    """Joint parameter/noise-aligned response program with a replayable posterior.

    Results retain world, parameter-draw, and supplied disturbance-row axes.
    Supplied disturbances are not automatically a population distribution.
    """
    assert execution is not None
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "parameter_posterior",
                "model": asdict(model),
                "parameters": parameters,
                "rows": rows,
                "prior_sd": prior_sd,
                "sampling": dict(asdict(sampling or SamplingPolicy()), seed=_seed()),
                "worlds": worlds,
                "noise": noise,
                "initial": initial if initial is not None else [0.0] * len(model.equations),
                "solve_policy": asdict(solve_policy or SolvePolicy()),
            },
            allow_nan=False,
        ),
        execution=execution,
    )


@_with_execution
def partial_counterfactual(
    model: EquilibriumModel,
    observations: list[float | None],
    worlds: list[list[dict[str, Any]]],
    *,
    measurement_variances: list[float],
    initial: list[float],
    sampling: SamplingPolicy | None = None,
    solve_policy: SolvePolicy | None = None,
    execution: Execution | None = None,
) -> PreparedCyclicFamily:
    """Abduce under an explicit exact/noisy observation model and reuse disturbances.

    This conditions on a specified mechanism/noise model. Parameter uncertainty
    is separate from latent-state/measurement uncertainty.
    """
    assert execution is not None
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "partial_counterfactual",
                "model": asdict(model),
                "observations": observations,
                "measurement_variances": measurement_variances,
                "initial": initial,
                "sampling": dict(asdict(sampling or SamplingPolicy()), seed=_seed()),
                "worlds": worlds,
                "solve_policy": asdict(solve_policy or SolvePolicy()),
            },
            allow_nan=False,
        ),
        execution=execution,
    )


@dataclass(frozen=True)
class BipartiteCausalGraph:
    """Equation-variable graph; numerical solvability is a separate question."""

    variables: int
    exogenous: tuple[int, ...]
    equations: tuple[tuple[int, ...], ...]

    @_with_execution
    def b_separation(
        self,
        x: int,
        y: int,
        conditioning: list[int],
        *,
        replacement: tuple[int, int] | None = None,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Compute causal ordering and determinism-aware graphical B-separation.

        Markov interpretation requires independent exogenous inputs and measurable
        unique cluster solutions; matching alone establishes neither.
        """
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "graph",
                    "request": {
                        "operation": "bipartite",
                        **asdict(self),
                        "x": x,
                        "y": y,
                        "conditioning": conditioning,
                        "replacement": replacement,
                    },
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def domain_invariance(
        self,
        changed_equations: list[int],
        outcomes: list[int],
        conditioning: list[int],
        *,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Check the BGCM domain-selector sufficient invariance criterion.

        Compatible unique cluster solutions must hold in every domain; only named
        equations may change. A failed separation check does not disprove invariance.
        """
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "graph",
                    "request": {
                        "operation": "domain_invariance",
                        **asdict(self),
                        "changed_equations": changed_equations,
                        "outcomes": outcomes,
                        "conditioning": conditioning,
                    },
                },
                allow_nan=False,
            ),
            execution=execution,
        )


@_with_execution
def fit_polynomial_candidates(
    candidates: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    *,
    max_boxes: int = 10_000,
    minimum_width: float = 1e-8,
    execution: Execution | None = None,
) -> PreparedCyclicFamily:
    """Fit a finite equation-parameter set under explicit root-selection/measurement laws.

    Each candidate names a polynomial model and geometric root regions with
    declared probabilities. A finite-set optimum certificate requires disjoint
    outward-rounded scores; unresolved roots/rankings retain a following attempt.
    """
    assert execution is not None
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "polynomial_fit",
                "candidates": candidates,
                "measurements": measurements,
                "policy": {"max_boxes": max_boxes, "minimum_width": minimum_width},
            },
            allow_nan=False,
        ),
        execution=execution,
    )


@_with_execution
def joint_counterfactual(
    model: EquilibriumModel,
    parameters: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    observations: list[float | None],
    measurement_variances: list[float],
    worlds: list[list[dict[str, Any]]],
    *,
    prior_sd: float,
    initial: list[float],
    sampling: SamplingPolicy | None = None,
    solve_policy: SolvePolicy | None = None,
    execution: Execution | None = None,
) -> PreparedCyclicFamily:
    """Update parameters and factual states jointly, then reuse each noise draw.

    Training and factual measurement evidence both enter the posterior. Structure
    and covariance are specified. Inspect joint chain diagnostics before use.
    """
    assert execution is not None
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "joint_counterfactual",
                "model": asdict(model),
                "parameters": parameters,
                "rows": rows,
                "prior_sd": prior_sd,
                "observations": observations,
                "measurement_variances": measurement_variances,
                "worlds": worlds,
                "initial": initial,
                "sampling": dict(asdict(sampling or SamplingPolicy()), seed=_seed()),
                "solve_policy": asdict(solve_policy or SolvePolicy()),
            },
            allow_nan=False,
        ),
        execution=execution,
    )


@_with_execution
def action_worlds(
    model: EquilibriumModel,
    worlds: list[dict[str, Any]],
    noise: list[list[float]],
    *,
    factual: list[float] | None = None,
    initial: list[float] | None = None,
    policy: SolvePolicy | None = None,
    execution: Execution | None = None,
) -> PreparedCyclicFamily:
    """Retain named clamps, mechanism changes/replacements and additive shocks.

    Each world has ``id`` and ``actions``. Actions name stable ``mechanism`` IDs;
    kinds are ``clamp``, ``replace_mechanism``, ``change_mechanism``, and
    ``additive_shock``. Replacements/changes preserve the original disturbances;
    shocks add to them after factual abduction. Clamps remove the mechanism.
    Failed numerical/certificate attempts retain their original response slots.
    """
    assert execution is not None
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "equilibrium_actions",
                "model": asdict(model),
                "worlds": worlds,
                "noise": noise,
                "factual": factual,
                "initial": initial if initial is not None else [0.0] * len(model.equations),
                "policy": asdict(policy or SolvePolicy()),
            },
            allow_nan=False,
        ),
        execution=execution,
    )


@_with_execution
def bootstrap_responses(
    model: EquilibriumModel,
    parameters: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    worlds: list[list[dict[str, Any]]],
    noise: list[list[float]],
    *,
    design: Literal["fixed_assignments", "randomized_within_targets"] = "fixed_assignments",
    replicates: int = 100,
    execution: Execution | None = None,
    policy: FitOptions | None = None,
    initial: list[float] | None = None,
    solve_policy: SolvePolicy | None = None,
) -> PreparedCyclicFamily:
    """Resample IID rows within known intervention regimes and propagate fits.

    Failed fits/response attempts retain their positions. This conditions on the
    declared structure and covariance; it does not claim calibrated coverage.
    """
    assert execution is not None
    seed = _seed()
    return PreparedCyclicFamily(
        json.dumps(
            {
                "family": "frequentist_bootstrap",
                "design": design,
                "model": asdict(model),
                "parameters": parameters,
                "rows": rows,
                "worlds": worlds,
                "noise": noise,
                "replicates": replicates,
                "seed": seed,
                "policy": asdict(policy or FitOptions()),
                "initial": initial if initial is not None else [0.0] * len(model.equations),
                "solve_policy": asdict(solve_policy or SolvePolicy()),
            },
            allow_nan=False,
        ),
        execution=execution,
    )


@dataclass(frozen=True)
class PolynomialWorld:
    """Equation-specific actions; clamps remove the replaced disturbance."""

    id: str
    replacements: tuple[dict[str, Any], ...] = ()
    clamps: tuple[dict[str, Any], ...] = ()
    shocks: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PolynomialExogenousAtom:
    """Declared joint disturbances and geometric branch choices across all worlds.

    Relative weights define a law independently of root-search order. Measurement
    kernels map world IDs to category/relative-probability maps. Each training unit
    has its own atom; one query unit reuses its atom across every counterfactual.
    """

    id: str
    weight: float
    disturbances: tuple[float, ...]
    selections: dict[str, list[dict[str, float]]]
    measurements: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class PolynomialCausalCandidate:
    """Finite parameter candidate with a declared prior and conditional exogenous law."""

    id: str
    model: PolynomialSystem
    atoms: tuple[PolynomialExogenousAtom, ...]
    prior_weight: float = 1.0


@dataclass(frozen=True)
class PolynomialObservation:
    """Partial state event at declared measurement resolution and/or noisy category.

    Bounds are observation events, separate from solver tolerance. A category
    requires an explicit measurement kernel in every exogenous atom.
    """

    world: str
    bounds: tuple[dict[str, float] | None, ...]
    label: str | None = None


@dataclass(frozen=True)
class PolynomialCausalModel:
    """Finite joint-selection SCM: fitting, abduction, action, prediction and replay.

    A candidate is shared across training units. Disturbances and branch selectors
    are independent between units conditional on that candidate, and shared across
    the potential worlds of one unit. Undefined/unresolved response mass is retained.
    """

    candidates: tuple[PolynomialCausalCandidate, ...]
    worlds: tuple[PolynomialWorld, ...]

    @_with_execution
    def prepare(
        self,
        factual: PolynomialObservation,
        *,
        training: tuple[PolynomialObservation, ...] = (),
        max_boxes: int = 10_000,
        minimum_width: float = 1e-8,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Condition the finite parameter/selection law and retain coupled world states."""
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "polynomial_causal",
                    "program": {
                        **asdict(self),
                        "training": [asdict(item) for item in training],
                        "factual": asdict(factual),
                        "policy": {"max_boxes": max_boxes, "minimum_width": minimum_width},
                    },
                },
                allow_nan=False,
            ),
            execution=execution,
        )


@dataclass(frozen=True)
class IdentifiedCyclicFunctional:
    """Checked cyclic functional bound to an independent identification source."""

    source: dict[str, Any]
    source_identity: str
    program: dict[str, Any]

    @_with_execution
    def evaluate_gaussian(
        self,
        law: dict[str, Any],
        assignments: list[list[tuple[int, float]]],
        *,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Evaluate continuous identified densities with analytic Gaussian marginalization.

        ``law`` is ``{kind: specified, mean: [...], covariance: [...]}`` or
        ``{kind: fit, rows: [...], independent_observational_replicates: True}``.
        Fitting is an unregularized IID observational Gaussian likelihood plug-in,
        conditional on the original graphical premises; no latent recovery or
        statistical interval claim is added. Discrete sources refuse this path.
        """
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "gaussian_functional",
                    "source": self.source,
                    "source_identity": self.source_identity,
                    "program": thaw(self.program),
                    "law": law,
                    "assignments": assignments,
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def evaluate(
        self,
        support: list[list[float]],
        weights: list[float],
        assignments: list[list[tuple[int, float]]],
        *,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Consume a finite joint law or empirical table; retain positivity failures.

        Variable zero varies fastest in the joint weights. Known probabilities
        and empirical counts are distinct scientific inputs; this operation
        evaluates their declared law without adding a nominal uncertainty claim.
        """
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "functional",
                    "source": self.source,
                    "source_identity": self.source_identity,
                    "program": thaw(self.program),
                    "law": {"support": support, "weights": weights},
                    "assignments": assignments,
                },
                allow_nan=False,
            ),
            execution=execution,
        )


@dataclass(frozen=True)
class LocalMechanism:
    """Pure local callback returning ``(value, derivatives_in_read_order)``.

    The callback receives only its declared inputs. Purity and derivative
    correctness are caller obligations. Its code and external allocations are
    process-local; independent artifacts require serializable expressions.
    """

    id: str
    reads: tuple[int, ...]
    callback: Any


@dataclass(frozen=True)
class LocalCyclicModel:
    """Process-local additive-noise feedback model with explicit derivatives."""

    mechanisms: tuple[LocalMechanism, ...]

    @outcome_operation
    @_with_execution
    def execute(
        self,
        noise: list[float],
        clamps: list[dict[str, Any]],
        *,
        initial: list[float] | None = None,
        factual: list[float] | None = None,
        tolerance: float = 1e-10,
        max_iterations: int = 1000,
        max_evaluations: int = 100_000,
        execution: Execution | None = None,
    ) -> dict[str, Any]:
        """Numerical SCC solve and optional shared-noise counterfactual.

        Initialization is numerical. It supplies neither uniqueness nor an
        equilibrium selection law, and no probabilistic root selection is made.
        Rust checks residuals, derivative shapes, cancellation and call budgets.
        """
        assert execution is not None
        limits = _limits(execution)
        request = {
            "mechanisms": [(m.id, m.reads) for m in self.mechanisms],
            "noise": noise,
            "clamps": clamps,
            "initial": initial if initial is not None else [0.0] * len(self.mechanisms),
            "factual": factual,
            "policy": {
                "tolerance": tolerance,
                "max_iterations": max_iterations,
                "max_evaluations": max_evaluations,
            },
        }
        return json.loads(
            _native.execute_local_cyclic(
                json.dumps(request, allow_nan=False),
                [m.callback for m in self.mechanisms],
                **limits,
            )
        )

    @artifact_export
    def export(self) -> ExportedArtifact:
        """Refuse to present arbitrary local callback code as a portable model."""
        raise CyclicRefusal(
            "local callback code is not portable; use serializable built-in expressions",
            reason_code="policy_callback_not_portable",
            detail="cyclic_local.callback_not_portable",
            remedy="declare the mechanisms as serializable built-in expressions",
        )


@dataclass(frozen=True)
class CyclicEvidence:
    """Native source-bound contract envelopes, uncertainty ports and decision inputs."""

    _handle: Any = field(repr=False)
    execution: Execution = field(default_factory=Execution, repr=False, compare=False)
    _prepared: Any = field(default=None, repr=False, compare=False)
    _metadata: Any = field(default=None, init=False, repr=False, compare=False)
    _inspection: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from .errors import CausalTypeError
        from .outcomes import Answer
        from .results._report import InspectionReport, SlotModel

        if not isinstance(self._handle, _native._CyclicEvidence):
            raise CausalTypeError("CyclicEvidence requires a retained native evidence owner")
        metadata = json.loads(self._handle.metadata_json(offset=0, limit=0))
        object.__setattr__(self, "_metadata", freeze(metadata))
        unavailable = SlotModel(
            available=False, reason="select_native_evidence_columns", summary="unavailable"
        )
        object.__setattr__(
            self,
            "_inspection",
            InspectionReport(
                identification=SlotModel(
                    available=False, reason="source_bound_projection", summary=metadata["scope"]
                ),
                support=unavailable,
                uncertainty=unavailable,
                assumptions=unavailable,
                answer=Answer("structured"),
                stage="executed" if self._prepared is not None else "consumed",
                family="cyclic_evidence",
                details={key: value for key, value in metadata.items() if key != "columns"},
                operations=(
                    "inspect",
                    "columns",
                    "to_dict",
                    "export",
                    "joint_law",
                    "decide_means",
                    "decide_enclosures",
                ),
            ),
        )

    def __repr__(self) -> str:
        return (
            f"CyclicEvidence(identity={self.identity!r}, columns={self._metadata['column_count']})"
        )

    def inspect(self):
        """Bounded native source and projection metadata, without samples or columns."""
        return self._inspection

    @_with_execution
    def to_dict(self, *, execution: Execution | None = None) -> dict[str, Any]:
        """Explicitly convert the complete native projection and original failed slots."""
        return dict(json.loads(self._handle.report()))

    def columns(self, *, offset: int = 0, limit: int = 64):
        """Explicitly page original quantity and uncertainty-port metadata."""
        from ._cyclic_read_models import native_page

        return native_page(self._handle.metadata_json, "columns", offset, limit)

    @property
    def quantities(self) -> tuple[Any, ...]:
        """Typed scientific coordinates issued by the native evidence projection."""
        from .joint_distribution import ScientificQuantity

        quantities: list[ScientificQuantity] = []
        page = self.columns()
        while page is not None:
            quantities.extend(
                ScientificQuantity.from_wire(column["quantity"]) for column in page.items
            )
            page = page.next()
        return tuple(quantities)

    @property
    def identity(self) -> str:
        """Source/schema/world/time/selector identity for independent consumption."""
        return str(self._handle.identity())

    @property
    def provenance(self):
        return freeze(
            {
                key: self._metadata[key]
                for key in (
                    "identity",
                    "source_identity",
                    "scope",
                    "source_bridge",
                    "recovery_origins",
                )
                if key in self._metadata
            }
        )

    @artifact_export
    @_with_execution
    def export(
        self,
        *,
        execution: Execution | None = None,
    ) -> ExportedArtifact:
        """Export original source and projection, preserving replay and uncertainty meaning."""
        assert execution is not None
        limits = _limits(execution)
        return artifact_from_receipt(self, bytes(self._handle.export(**limits)))

    @_with_execution
    @outcome_operation
    def joint_law(
        self,
        *others: Any,
        dependence: Literal["aligned", "independent"] | None = None,
        execution: Execution | None = None,
    ) -> Any:
        """Compose explicit native columns with their verified scientific draw pairing.

        Version-three recovered functional posterior columns keep exact weights.
        Shared observations cannot be declared independent; equal draw counts do
        not establish pairing. Sampling laws, Gaussian means, failed slots and
        numerical mass bounds remain distinct from outcome distributions.
        """
        from .errors import CausalTypeError, CausalValueError
        from .joint_distribution import JointDistributionArtifact
        from .outcomes import Outcome, require_success

        assert execution is not None  # _with_execution resolves the retained policy.
        if dependence not in (None, "aligned", "independent"):
            raise CausalValueError("dependence must be aligned or independent")
        if len(others) > 256:
            raise CausalValueError("at most 256 additional evidence products")
        selected = tuple(
            require_success(item) if isinstance(item, Outcome) else item for item in others
        )
        if any(not isinstance(item, CyclicEvidence) for item in selected):
            raise CausalTypeError("joint_law inputs must be CyclicEvidence outcomes or values")
        return JointDistributionArtifact._from_native(
            self._handle.joint_law(
                others=[item._handle for item in selected],
                dependence=dependence,
                **_limits(execution),
            )
        )

    @outcome_operation
    @_with_execution
    def decide_enclosures(
        self, contract: Any, *, execution: Execution | None = None
    ) -> dict[str, Any]:
        """Conservative robust decisions on model solution/scenario outer enclosures.

        Unresolved root boxes remain in the set; undefined selected-state mass
        refuses. This introduces neither equilibrium probabilities nor a
        nonparametric identification or statistical coverage claim.
        """
        assert execution is not None
        return dict(json.loads(self._handle.decide_enclosures(json.dumps(contract._wire()))))

    @outcome_operation
    @_with_execution
    def decide_means(self, contract: Any, *, execution: Execution | None = None) -> dict[str, Any]:
        """Evaluate an existing decision Contract on point/mean evidence only."""
        assert execution is not None
        return dict(json.loads(self._handle.decide_means(json.dumps(contract._wire()))))


@outcome_operation
@_with_execution
@artifact_consumer(
    proof="cyclic.consume_evidence: independent native source replay and complete evidence schema/selector/identity verification",
    replayable=True,
)
def consume_evidence(
    artifact: bytes,
    *,
    expected_identity: str,
    execution: Execution | None = None,
) -> CyclicEvidence:
    """Independently replay a cyclic source and recheck its entire evidence projection."""
    assert execution is not None
    limits = _limits(execution)
    artifact = container_bytes(artifact)
    return CyclicEvidence(
        _native.consume_cyclic_evidence(artifact, expected_identity, **limits),
        execution=execution,
    )


@dataclass(frozen=True)
class OuMechanisms:
    """Affine stochastic feedback ``dX=(M X+b)dt+L dW`` with named equations.

    Forcing is distinct from the stationary mean. Replacements change exactly
    one equation's drift, forcing and diffusion-factor row, preserving all
    untargeted mechanisms. The stationary law is rechecked in every world.
    """

    equations: tuple[str, ...]
    drift: tuple[float, ...]
    forcing: tuple[float, ...]
    diffusion_factor: tuple[float, ...]

    @_with_execution
    def prepare(
        self,
        worlds: list[list[dict[str, Any]]],
        *,
        regimes: list[dict[str, Any]] | None = None,
        parameters: list[dict[str, Any]] | None = None,
        learning_rate: float = 0.1,
        score_tolerance: float = 1e-6,
        max_iterations: int = 1000,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Evaluate or jointly fit independent stationary intervention regimes.

        A replacement supplies ``equation`` (stable id), ``drift`` (full row),
        ``forcing`` and ``diffusion_factor`` (full row). Known target mechanisms
        stay fixed during fitting; baseline coefficients are shared everywhere
        else. Parameters are ``drift`` (equation,parent), ``forcing`` (equation)
        or ``diffusion`` (equation,source). Identifiability is a separate question.
        """
        assert execution is not None
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "ou_mechanisms",
                    "model": asdict(self),
                    "worlds": worlds,
                    "regimes": regimes or [],
                    "parameters": parameters or [],
                    "policy": {
                        "learning_rate": learning_rate,
                        "score_tolerance": score_tolerance,
                        "max_iterations": max_iterations,
                    },
                },
                allow_nan=False,
            ),
            execution=execution,
        )

    @_with_execution
    def infer(
        self,
        parameters: list[dict[str, Any]],
        regimes: list[dict[str, Any]],
        *,
        method: DynamicMethod,
        sampling_design: SamplingDesign,
        quantities: list[DynamicQuantity] | tuple[()] = (),
        learning_rate: float = 0.1,
        score_tolerance: float = 1e-6,
        max_iterations: int = 1000,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Fit shared mechanisms once, then attach their declared uncertainty.

        ``regimes`` are equation-replacement regimes ``{"id", "replacements", "values"}``;
        each observation row is one sampling unit. ``quantities``
        (:class:`OuMechanismMean`, :class:`OuMechanismCovariance`) are stationary laws after
        equation replacements, evaluated at every replicate or draw. Known target mechanisms
        stay fixed; identifiability of the estimated coordinates is a separate question.
        """
        assert execution is not None
        from ._cyclic_typed import infer_ou_mechanisms

        return infer_ou_mechanisms(
            self,
            parameters,
            regimes,
            method=method,
            quantities=quantities,
            sampling_design=sampling_design,
            learning_rate=learning_rate,
            score_tolerance=score_tolerance,
            max_iterations=max_iterations,
            execution=execution,
        )

    @_with_execution
    def fit_trajectories(
        self,
        trajectories: Sequence[OuTrajectory],
        parameters: list[dict[str, Any]],
        *,
        observation: OuObservationModel | None = None,
        predictions: Sequence[OuPredictionDesign] = (),
        stationary_worlds: Sequence[OuStationaryDesign] = (),
        learning_rate: float = 0.1,
        score_tolerance: float = 1e-6,
        max_iterations: int = 1000,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Fit shared mechanisms from continuous trajectories by the exact likelihood.

        Between observations the Gaussian transition is exact (matrix exponential and the
        van Loan covariance), so irregular intervals and known piecewise-constant
        intervention schedules carry no discretization error. A Kalman filter evaluates the
        likelihood under the declared ``observation`` model (default: every component
        observed without noise), which covers partial observation; unobserved components
        are reported with their filtered uncertainty.

        The result keeps three records separate: ``fitting`` (optimizer convergence),
        ``identification`` (``identified``, ``not_identified`` or ``model_conditional``,
        with observability and sampling-interval aliasing checks) and ``numerics``
        (matrix-exponential accuracy, conditioning, stability of the drift). A design that
        cannot identify a requested coordinate is refused with
        ``dynamics_not_identified``. ``predictions`` are held-out intervention trajectories
        and ``stationary_worlds`` sustained interventions; each carries process variability
        and the fitted parameters' sampling uncertainty (delta method on the observed
        information) as separate fields. Coordinates are ``drift``, ``forcing`` or
        ``diffusion`` as for :meth:`prepare`; use a lower-triangular or diagonal diffusion
        parameterization, since only ``L L^T`` is identified.
        """
        assert execution is not None
        observed = observation or OuObservationModel.direct(len(self.equations))
        return PreparedCyclicFamily(
            json.dumps(
                {
                    "family": "ou_trajectory_fit",
                    "model": asdict(self),
                    "observation": asdict(observed),
                    "parameters": parameters,
                    "trajectories": [item._wire() for item in trajectories],
                    "policy": {
                        "learning_rate": learning_rate,
                        "score_tolerance": score_tolerance,
                        "max_iterations": max_iterations,
                    },
                    "predictions": [item._wire() for item in predictions],
                    "stationary_worlds": [item._wire() for item in stationary_worlds],
                },
                allow_nan=False,
            ),
            execution=execution,
        )


@dataclass(frozen=True)
class OuObservationModel:
    """Linear Gaussian observation model ``y = H x + offset + noise`` of a hidden OU state.

    ``matrix`` is the row-major ``m x n`` matrix ``H``; ``noise_standard_deviation`` holds
    the known noise level of each observed row. A state component with a zero column of
    ``H`` is unobserved: it is reported with its filtered uncertainty, never imputed.
    """

    matrix: tuple[float, ...]
    noise_standard_deviation: tuple[float, ...]
    offset: tuple[float, ...] = ()

    @classmethod
    def direct(cls, dimension: int, noise_standard_deviation: float = 0.0) -> OuObservationModel:
        """Observe every one of ``dimension`` state components directly."""
        matrix = [0.0] * (dimension * dimension)
        for i in range(dimension):
            matrix[i * dimension + i] = 1.0
        return cls(tuple(matrix), (float(noise_standard_deviation),) * dimension)


@dataclass(frozen=True)
class OuScheduleSegment:
    """From ``start`` until the next segment, these equation replacements are in force.

    A replacement is ``{"equation", "drift", "forcing", "diffusion_factor"}`` exactly as
    for :meth:`OuMechanisms.prepare`; no replacements denotes the baseline mechanism.
    """

    start: float
    replacements: tuple[dict[str, Any], ...] = ()


def _schedule_wire(schedule: Sequence[OuScheduleSegment]) -> list[dict[str, Any]]:
    return [{"start": item.start, "replacements": list(item.replacements)} for item in schedule]


_BASELINE_SCHEDULE = (OuScheduleSegment(0.0),)
_STATIONARY_START: dict[str, Any] = {"kind": "stationary"}


@dataclass(frozen=True)
class OuTrajectory:
    """One unit observed at strictly increasing, possibly irregular ``times``.

    ``values[k]`` is the observed vector at ``times[k]``. ``schedule`` is the known
    piecewise-constant intervention schedule (starts strictly increasing, the first at or
    before ``times[0]``). ``initial`` is ``{"kind": "stationary"}`` for the stationary law
    of the first segment, or ``{"kind": "gaussian", "mean", "covariance"}``.
    """

    id: str
    times: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    schedule: tuple[OuScheduleSegment, ...] = _BASELINE_SCHEDULE
    initial: dict[str, Any] = field(default_factory=lambda: dict(_STATIONARY_START))

    def _wire(self) -> dict[str, Any]:
        if len(self.times) != len(self.values):
            raise CausalValueError("an OU trajectory needs one observed vector per time")
        return {
            "id": self.id,
            "schedule": _schedule_wire(self.schedule),
            "initial": dict(self.initial),
            "observations": [
                {"time": t, "values": list(v)} for t, v in zip(self.times, self.values, strict=True)
            ],
        }


@dataclass(frozen=True)
class OuPredictionDesign:
    """A held-out intervention trajectory whose state law is predicted at ``times``."""

    id: str
    times: tuple[float, ...]
    schedule: tuple[OuScheduleSegment, ...] = _BASELINE_SCHEDULE
    initial: dict[str, Any] = field(default_factory=lambda: dict(_STATIONARY_START))

    def _wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "schedule": _schedule_wire(self.schedule),
            "initial": dict(self.initial),
            "times": list(self.times),
        }


@dataclass(frozen=True)
class OuStationaryDesign:
    """A sustained intervention world whose stationary law is predicted."""

    id: str
    replacements: tuple[dict[str, Any], ...] = ()

    def _wire(self) -> dict[str, Any]:
        return {"id": self.id, "replacements": list(self.replacements)}


# ---------------------------------------------------------------------------
# Nonlinear stochastic differential equations with diagonal noise
# ---------------------------------------------------------------------------


class SdeExpression:
    """Portable expressions for SDE drift and diffusion, written with names.

    ``state(name)`` reads a state, ``input(name)`` an external input ``u(t)`` and
    ``parameter(name)`` a named parameter; the system resolves names to coordinates.
    Supported operations are exactly the portable mechanism language: ``add``,
    ``multiply``, ``tanh``, ``sin`` and ``softplus`` (no division or square root:
    divide by a known constant by multiplying with its reciprocal, and take a square
    root with ``SdeDiffusion(..., square_root=True)``).
    """

    @staticmethod
    def constant(value: float) -> dict[str, Any]:
        """A finite constant."""
        return {"op": "constant", "value": float(value)}

    @staticmethod
    def state(name: str) -> dict[str, Any]:
        """The state named ``name``."""
        return {"op": "state", "name": name}

    @staticmethod
    def input(name: str) -> dict[str, Any]:
        """The external input named ``name`` evaluated at the current time."""
        return {"op": "input", "name": name}

    @staticmethod
    def parameter(name: str) -> dict[str, Any]:
        """The parameter named ``name``."""
        return {"op": "parameter_ref", "name": name}

    @staticmethod
    def add(*terms: dict[str, Any]) -> dict[str, Any]:
        """The sum of one or more terms."""
        return {"op": "add", "terms": list(terms)}

    @staticmethod
    def multiply(*factors: dict[str, Any]) -> dict[str, Any]:
        """The product of one or more factors."""
        return {"op": "multiply", "factors": list(factors)}

    @staticmethod
    def negate(term: dict[str, Any]) -> dict[str, Any]:
        """Minus ``term``."""
        return SdeExpression.multiply(SdeExpression.constant(-1.0), term)

    @staticmethod
    def tanh(argument: dict[str, Any]) -> dict[str, Any]:
        """Smooth bounded saturation."""
        return {"op": "tanh", "argument": argument}

    @staticmethod
    def sin(argument: dict[str, Any]) -> dict[str, Any]:
        """Smooth non-monotone response."""
        return {"op": "sin", "argument": argument}

    @staticmethod
    def softplus(argument: dict[str, Any]) -> dict[str, Any]:
        """Stable smooth rectifier."""
        return {"op": "softplus", "argument": argument}


def _resolve_expression(
    expression: dict[str, Any],
    states: dict[str, int],
    inputs: dict[str, int],
    parameters: dict[str, int],
) -> dict[str, Any]:
    op = expression.get("op")
    if op == "state":
        return {"op": "variable", "variable": _named(states, expression["name"], "state")}
    if op == "input":
        return {
            "op": "variable",
            "variable": len(states) + _named(inputs, expression["name"], "input"),
        }
    if op == "parameter_ref":
        return {
            "op": "parameter",
            "parameter": _named(parameters, expression["name"], "parameter"),
        }
    if op in ("add", "multiply"):
        key = "terms" if op == "add" else "factors"
        return {
            "op": op,
            key: [
                _resolve_expression(item, states, inputs, parameters) for item in expression[key]
            ],
        }
    if op in ("tanh", "sin", "softplus"):
        return {
            "op": op,
            "argument": _resolve_expression(expression["argument"], states, inputs, parameters),
        }
    if op in ("constant", "variable", "parameter"):
        return dict(expression)
    raise CausalValueError(f"unknown SDE expression operation {op!r}")


def _named(table: dict[str, int], name: str, kind: str) -> int:
    if name not in table:
        raise CausalValueError(f"unknown SDE {kind} {name!r}; declared: {sorted(table)}")
    return table[name]


@dataclass(frozen=True)
class SdeInput:
    """A time-dependent external input ``u(t)``, a declared schedule and not a state.

    ``interpolation`` is ``"piecewise_constant"`` (the value at the latest knot not after
    ``t``) or ``"linear"``; both are constant outside the knots. ``envelope`` is the closed
    range the regularity certificate assumes for this input in every world; it defaults to
    the range of ``values``. A world may replace the schedule only inside the envelope.
    """

    name: str
    times: tuple[float, ...]
    values: tuple[float, ...]
    interpolation: Literal["piecewise_constant", "linear"] = "piecewise_constant"
    envelope: tuple[float, float] | None = None

    def _wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "name": self.name,
            "interpolation": self.interpolation,
            "times": list(self.times),
            "values": list(self.values),
        }
        if self.envelope is not None:
            wire["envelope"] = list(self.envelope)
        return wire


@dataclass(frozen=True)
class SdeBound:
    """An interval for one state; ``"reflect"`` mirrors excursions, ``"clamp"`` projects them."""

    lower: float | None = None
    upper: float | None = None
    rule: Literal["reflect", "clamp"] = "reflect"

    def _wire(self) -> dict[str, Any]:
        return {"lower": self.lower, "upper": self.upper, "rule": self.rule}


@dataclass(frozen=True)
class SdeDiffusion:
    """Diagonal diffusion of one state: the expression, or ``sqrt(max(expression, 0))``."""

    expression: dict[str, Any]
    square_root: bool = False


@dataclass(frozen=True)
class SdeAction:
    """Hold state ``variable`` at ``values[j]`` from ``times[j]`` until the next time.

    Action times must lie on the declared simulation step grid (an off-grid action is
    refused rather than silently shifted).
    """

    variable: str | int
    times: tuple[float, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class SdeWorld:
    """A named intervention regime: component holds and replaced input schedules."""

    id: str
    actions: tuple[SdeAction, ...] = ()
    inputs: tuple[SdeInput, ...] = ()

    def _wire(self, states: dict[str, int]) -> dict[str, Any]:
        return {
            "id": self.id,
            "actions": [
                {
                    "variable": a.variable
                    if isinstance(a.variable, int)
                    else _named(states, a.variable, "state"),
                    "times": list(a.times),
                    "values": list(a.values),
                }
                for a in self.actions
            ],
            "inputs": [item._wire() for item in self.inputs],
        }


@dataclass(frozen=True)
class SdeContrast:
    """Treated minus control, simulated with shared random numbers."""

    id: str
    treated: str
    control: str


@dataclass(frozen=True)
class SdeSimulation:
    """Declared step, horizon, scheme, ensemble size and seed (all semantic, replayable).

    The horizon is ``start + steps * step``; moments are recorded every ``output_every``
    steps. ``check_step`` also runs the half-step refinement on the same Brownian paths and
    reports the numerical-approximation error; ``weak_tolerance`` is the relative tolerance
    of its convergence flag. ``retain_final_samples`` keeps each path's final state.
    """

    step: float
    steps: int
    paths: int
    seed: int
    start: float = 0.0
    scheme: Literal["euler_maruyama", "milstein"] = "euler_maruyama"
    output_every: int = 1
    weak_tolerance: float = 0.02
    check_step: bool = True
    retain_final_samples: bool = False

    def _wire(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SdeObservationModel:
    """Linear Gaussian observation ``y = H x + offset + noise`` of a hidden SDE state.

    Declaring one switches the fit to the extended-Kalman innovation likelihood, which
    covers noise and partial observation and is always a model-conditional approximation.
    """

    matrix: tuple[float, ...]
    noise_standard_deviation: tuple[float, ...]
    offset: tuple[float, ...] = ()

    @classmethod
    def direct(cls, dimension: int, noise_standard_deviation: float) -> SdeObservationModel:
        """Observe every state component with the same noise level."""
        matrix = [0.0] * (dimension * dimension)
        for i in range(dimension):
            matrix[i * dimension + i] = 1.0
        return cls(tuple(matrix), (float(noise_standard_deviation),) * dimension)


@dataclass(frozen=True)
class SdeTrajectory:
    """One unit observed at strictly increasing ``times`` under a known intervention."""

    id: str
    times: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    intervention: SdeWorld | None = None

    def _wire(self, states: dict[str, int]) -> dict[str, Any]:
        if len(self.times) != len(self.values):
            raise CausalValueError("an SDE trajectory needs one observed vector per time")
        wire: dict[str, Any] = {
            "id": self.id,
            "observations": [
                {"time": t, "values": list(v)} for t, v in zip(self.times, self.values, strict=True)
            ],
        }
        if self.intervention is not None:
            wire["intervention"] = self.intervention._wire(states)
        return wire


@dataclass(frozen=True)
class StochasticDifferentialEquation:
    """``dX = f(X, u(t), theta) dt + g(X, u(t), theta) dW`` with diagonal, state-dependent noise.

    ``drift`` and ``diffusion`` hold one expression per state (:class:`SdeExpression`);
    ``parameters`` maps parameter names to their declared values; ``bounds`` maps state
    names to :class:`SdeBound` (reflecting or clamping, e.g. nonnegative counts).

    Native validation certifies finite Lipschitz constants of both coefficients over the
    declared state box by interval derivative arithmetic. A nonlinearity that multiplies
    unbounded states is refused (``sde.unbounded_growth``): declare bounds. A square-root
    diffusion that reaches zero is admitted but certified only as Holder-1/2, which the
    certificate records. :meth:`simulate` and :meth:`fit` are bounded, replayable, and keep
    Monte Carlo error, step-halving error and parameter uncertainty in separate fields.
    """

    variables: tuple[str, ...]
    drift: tuple[dict[str, Any], ...]
    diffusion: tuple[SdeDiffusion, ...]
    parameters: dict[str, float]
    inputs: tuple[SdeInput, ...] = ()
    bounds: dict[str, SdeBound] = field(default_factory=dict)

    @classmethod
    def epidemic(
        cls,
        population: float = 1000.0,
        *,
        season: SdeInput | None = None,
        beta: float = 0.6,
        amplitude: float = 0.3,
        gamma: float = 0.25,
    ) -> StochasticDifferentialEquation:
        """The documented SIR system with a seasonal input and state-dependent noise.

        ``flux = beta (1 + amplitude u(t)) S I / N`` is the nonlinear transmission rate;
        ``dS = -flux dt``, ``dI = (flux - gamma I) dt``, ``dR = gamma I dt``. Each state
        carries independent square-root noise with variance equal to the rates of the events
        that change it (``flux``, ``flux + gamma I`` and ``gamma I``): a diagonal
        chemical-Langevin approximation that ignores the correlation the shared infection
        event induces between ``S`` and ``I``. States are reflected into ``[0, N]``.
        """
        e = SdeExpression
        season = season or SdeInput(
            "season",
            (0.0, 5.0, 10.0, 15.0, 20.0),
            (-1.0, 0.0, 1.0, 0.0, -1.0),
            "linear",
            (-1.0, 1.0),
        )
        flux = e.multiply(
            e.parameter("beta"),
            e.add(e.constant(1.0), e.multiply(e.parameter("amplitude"), e.input(season.name))),
            e.state("S"),
            e.state("I"),
            e.constant(1.0 / population),
        )
        recovery = e.multiply(e.parameter("gamma"), e.state("I"))
        return cls(
            variables=("S", "I", "R"),
            drift=(e.negate(flux), e.add(flux, e.negate(recovery)), recovery),
            diffusion=(
                SdeDiffusion(flux, square_root=True),
                SdeDiffusion(e.add(flux, recovery), square_root=True),
                SdeDiffusion(recovery, square_root=True),
            ),
            parameters={"beta": beta, "amplitude": amplitude, "gamma": gamma},
            inputs=(season,),
            bounds={name: SdeBound(0.0, population) for name in ("S", "I", "R")},
        )

    def _tables(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        return (
            {name: i for i, name in enumerate(self.variables)},
            {item.name: i for i, item in enumerate(self.inputs)},
            {name: i for i, name in enumerate(self.parameters)},
        )

    def _model_wire(self) -> dict[str, Any]:
        states, inputs, parameters = self._tables()
        if not (len(self.drift) == len(self.diffusion) == len(self.variables)):
            raise CausalValueError("an SDE needs one drift and one diffusion per state")
        unknown = set(self.bounds) - set(states)
        if unknown:
            raise CausalValueError(f"bounds for unknown states {sorted(unknown)}")
        return {
            "variables": list(self.variables),
            "inputs": [item._wire() for item in self.inputs],
            "parameters": [{"name": n, "value": float(v)} for n, v in self.parameters.items()],
            "drift": [_resolve_expression(d, states, inputs, parameters) for d in self.drift],
            "diffusion": [
                {
                    "expression": _resolve_expression(d.expression, states, inputs, parameters),
                    "square_root": d.square_root,
                }
                for d in self.diffusion
            ],
            "boundaries": [
                self.bounds[name]._wire() if name in self.bounds else SdeBound()._wire()
                for name in self.variables
            ],
        }

    def _initial(self, initial: Sequence[float] | dict[str, float]) -> list[float]:
        if isinstance(initial, dict):
            states = self._tables()[0]
            if set(initial) != set(states):
                raise CausalValueError("initial must name every state exactly once")
            return [float(initial[name]) for name in self.variables]
        return [float(v) for v in initial]

    def _request(
        self,
        initial: Sequence[float] | dict[str, float],
        simulation: SdeSimulation,
        interventions: Sequence[SdeWorld],
        contrasts: Sequence[SdeContrast],
        extra: dict[str, Any],
    ) -> str:
        states = self._tables()[0]
        return json.dumps(
            {
                "family": "sde_analysis",
                "model": self._model_wire(),
                "initial": self._initial(initial),
                "simulation": simulation._wire(),
                "interventions": [w._wire(states) for w in interventions],
                "contrasts": [asdict(c) for c in contrasts],
                **extra,
            },
            allow_nan=False,
        )

    @_with_execution
    def simulate(
        self,
        initial: Sequence[float] | dict[str, float],
        simulation: SdeSimulation,
        *,
        interventions: Sequence[SdeWorld] = (),
        contrasts: Sequence[SdeContrast] = (),
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Simulate the declared system at its declared parameters.

        The implicit ``"baseline"`` world is always first, then ``interventions`` in order.
        Every world uses the same Brownian increments path by path, so ``contrasts`` are
        paired differences with their own Monte Carlo standard error. Each world reports
        mean, variance, joint covariance across states, Monte Carlo standard error and, with
        ``check_step``, the effect of halving the step. Specified-model results: nothing here
        is estimated.
        """
        assert execution is not None
        return PreparedCyclicFamily(
            self._request(initial, simulation, interventions, contrasts, {}), execution=execution
        )

    @_with_execution
    def fit(
        self,
        trajectories: Sequence[SdeTrajectory],
        parameters: Sequence[str | int],
        initial: Sequence[float] | dict[str, float],
        simulation: SdeSimulation,
        *,
        observation: SdeObservationModel | None = None,
        filter_prior: tuple[Sequence[float], Sequence[float]] | None = None,
        interventions: Sequence[SdeWorld] = (),
        contrasts: Sequence[SdeContrast] = (),
        learning_rate: float = 0.1,
        score_tolerance: float = 1e-6,
        max_iterations: int = 1000,
        coarse_flag: float = 0.1,
        coarse_refusal: float = 0.5,
        variance_floor: float = 1e-12,
        filter_step: float = 0.0,
        parameter_draws: int = 0,
        parameter_seed: int = 0,
        execution: Execution | None = None,
    ) -> PreparedCyclicFamily:
        """Fit named parameters from trajectories, then simulate at the fit.

        Without ``observation`` every state is observed directly and the likelihood is the
        Euler Gaussian transition quasi-likelihood. Its accuracy depends on the observation
        interval times the local relaxation rate: above ``coarse_flag`` the fit is reported
        as a model-conditional approximation, above ``coarse_refusal`` it is refused
        (``sde_fit.observation_interval_too_coarse``). With ``observation`` (and the hidden
        state's Gaussian ``filter_prior`` ``(mean, covariance)`` at each first observation)
        the likelihood is the extended-Kalman innovation likelihood of the Euler-discretized
        model, substepped by ``filter_step``; it is always flagged as a linearization.

        The result keeps ``fitting`` (optimizer), ``identification`` (observed information)
        and ``discretization`` (interval-rate evidence) apart. Unidentified parameters are
        refused as ``dynamics_not_identified``. ``parameter_draws`` (0 or 2..512) propagates
        the fit's sampling covariance through the simulation by common-random-number
        ensembles; that spread is reported separately from Monte Carlo and step error.
        """
        assert execution is not None
        states, _, parameter_table = self._tables()
        coordinates = [
            p if isinstance(p, int) else _named(parameter_table, p, "parameter") for p in parameters
        ]
        extra: dict[str, Any] = {
            "parameters": coordinates,
            "trajectories": [item._wire(states) for item in trajectories],
            "fit_policy": {
                "optimizer": {
                    "learning_rate": learning_rate,
                    "score_tolerance": score_tolerance,
                    "max_iterations": max_iterations,
                    "l1_penalty": 0.0,
                },
                "coarse_flag": coarse_flag,
                "coarse_refusal": coarse_refusal,
                "variance_floor": variance_floor,
                "filter_step": filter_step,
            },
            "parameter_draws": parameter_draws,
            "parameter_seed": parameter_seed,
        }
        if observation is not None:
            extra["observation"] = asdict(observation)
        if filter_prior is not None:
            extra["filter_prior"] = {
                "mean": list(filter_prior[0]),
                "covariance": list(filter_prior[1]),
            }
        return PreparedCyclicFamily(
            self._request(initial, simulation, interventions, contrasts, extra),
            execution=execution,
        )


@dataclass(frozen=True)
class AdamPolicy:
    """Bounded projected Adam, with explicit independent validation stopping."""

    learning_rate: float = 0.01
    batch_size: int = 256
    max_epochs: int = 200
    minimum_epochs: int = 20
    patience: int = 10
    relative_tolerance: float = 1e-5
    score_tolerance: float = 1e-3
    l1_penalty: float = 0.001
    contraction_margin: float = 0.95


__all__ = [
    "AdamPolicy",
    "AsymptoticNormal",
    "BipartiteCausalGraph",
    "CyclicEvidence",
    "CyclicGraph",
    "CyclicInterventions",
    "CyclicPage",
    "CyclicRefusal",
    "CyclicResult",
    "RecoveredCyclicLaw",
    "DynamicBootstrap",
    "DynamicMethod",
    "DynamicPosterior",
    "DynamicQuantity",
    "DynamicQuantitySelector",
    "Equation",
    "EquilibriumModel",
    "ExperimentRegime",
    "ExperimentalFitOptions",
    "FitOptions",
    "FlatCoefficients",
    "IdentifiedCyclicFunctional",
    "InitialState",
    "InverseWishartDisturbance",
    "KnownDisturbance",
    "LinearCyclicModel",
    "LocalCyclicModel",
    "LocalMechanism",
    "NormalCoefficients",
    "ObservedInformation",
    "OdePolicy",
    "OdeState",
    "OdeSystem",
    "OrnsteinUhlenbeck",
    "OuMechanismCovariance",
    "OuMechanismMean",
    "OuMechanisms",
    "OuObservationModel",
    "OuPredictionDesign",
    "OuScheduleSegment",
    "OuStationaryCovariance",
    "OuStationaryDesign",
    "OuStationaryMean",
    "OuTrajectory",
    "PolynomialCausalCandidate",
    "PolynomialCausalModel",
    "PolynomialExogenousAtom",
    "PolynomialObservation",
    "PolynomialSystem",
    "PolynomialWorld",
    "PreparedCyclicFamily",
    "PreparedCyclicSystem",
    "RecoveredDrawSelector",
    "RecoveredResponseSelector",
    "RecoveredWorldAnswer",
    "RecoveryBootstrap",
    "RecoveryPosterior",
    "RecoveryPrior",
    "RecoveryUncertainty",
    "SamplingDesign",
    "SamplingPolicy",
    "Set",
    "SdeAction",
    "SdeBound",
    "SdeContrast",
    "SdeDiffusion",
    "SdeExpression",
    "SdeInput",
    "SdeObservationModel",
    "SdeSimulation",
    "SdeTrajectory",
    "SdeWorld",
    "StochasticDifferentialEquation",
    "Shift",
    "Soft",
    "SolvePolicy",
    "World",
    "WorldAnswer",
    "WorldSelector",
    "action_worlds",
    "bootstrap_responses",
    "consume_artifact",
    "consume_evidence",
    "consume_family_artifact",
    "discover_cyclic",
    "fit_polynomial_candidates",
    "joint_counterfactual",
    "learn_structure",
    "partial_counterfactual",
    "posterior_responses",
    "prepare_experimental_recovery",
    "prepare_recovered_linear",
]
