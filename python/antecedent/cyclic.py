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
from collections.abc import Mapping
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
    _seed,
    _with_execution,
    prepare_experimental_recovery,
)
from ._cyclic_typed import EvidenceVariable as EvidenceVariable
from ._freeze import freeze, thaw
from ._scientific import ScientificResult
from .errors import CausalValueError, StructuredRefusal
from .execution import Execution
from .intervention import Set, Shift, Soft
from .outcomes import outcome_operation, require_success


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
    """Numerical policy; initialization never supplies a selection law."""

    tolerance: float = 1e-10
    max_iterations: int = 1000
    require_contraction: bool = True


@dataclass(frozen=True)
class FitOptions:
    """Local coordinate likelihood fit with fixed joint noise covariance."""

    initial_step: float = 0.2
    tolerance: float = 1e-5
    max_sweeps: int = 500
    l1_penalty: float = 0.0


@dataclass(frozen=True)
class CyclicResult(ScientificResult):
    """Model-conditional worlds, diagnostics, and independent replay artifact."""

    report: Mapping[str, Any] = field(repr=False)
    artifact: bytes = field(repr=False)
    execution_facts: Any = field(default=None, repr=False, compare=False)
    _prepared: Any = field(default=None, repr=False, compare=False)
    _contract: Any = field(default=None, repr=False, compare=False)
    _inspection: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "report", freeze(self.report))
        object.__setattr__(self, "_contract", freeze(self._contract))
        object.__setattr__(self, "_inspection", self._make_inspection())

    def _summary_fields(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "family": self.report.get("answer", {}).get("family"),
            "claim": self.report.get("claim"),
            "calibration": self.report.get("calibration"),
        }

    def to_dict(self) -> dict[str, Any]:
        """Explicit mutable JSON-compatible copy of the native execution report."""
        return thaw(self.report)

    @property
    def identities(self):
        contract = self._contract or {}
        native = self._native_report_metadata()
        return freeze(
            {
                **contract.get("identities", {}),
                **{key: native[key] for key in ("source_identity",) if key in native},
                **{
                    key: contract[key]
                    for key in ("cyclic_solution", "recovery", "lowered_program_identity")
                    if contract.get(key) is not None
                },
            }
        )

    @property
    def provenance(self):
        """Retained native contract coordinates, without a reconstructed source history."""
        native = self._native_report_metadata()
        return freeze(
            {
                **({"native_contract": self._contract} if self._contract else {}),
                **{
                    key: native[key]
                    for key in ("source_identity", "derivation", "scope")
                    if key in native
                },
            }
        )

    def _native_report_metadata(self):
        native = self.report.get("answer", {}).get("result", self.report)
        return native if isinstance(native, Mapping) else self.report

    def inspect(self):
        """Return retained native metadata without serializing a payload or running work."""
        return self._inspection

    def _make_inspection(self):
        from .outcomes import Answer
        from .results._report import InspectionReport, SlotModel

        contract = self._contract or {}
        layers = contract.get("layers", {})
        identification = layers.get("identification")
        inference = layers.get("inference_binding")
        target = layers.get("target")
        # Only native metadata is exposed: no certificate, interval or assumption is
        # inferred from successful numerical execution or from a mean-valued target.
        unavailable = SlotModel(
            available=False, reason="not_retained_by_native_report", summary="unavailable"
        )
        native = self._native_report_metadata()
        assumptions = native.get("assumptions")
        if assumptions is None and target is not None:
            assumptions = target.get("assumptions")
            if assumptions is None and "targets" in target:
                assumptions = tuple(
                    item["assumptions"] for item in target["targets"] if "assumptions" in item
                )
        diagnostics = native.get("diagnostics", ())
        if not isinstance(diagnostics, (list, tuple)):
            diagnostics = ()
        ids = {
            f"{key}_id": value
            for key, value in contract.get("identities", {}).items()
            if value is not None
        }
        return InspectionReport(
            identification=SlotModel(
                available=True, summary=contract["identification_kind"], payload=identification
            )
            if identification
            else unavailable,
            support=unavailable,
            uncertainty=SlotModel(
                available=inference is not None and inference.get("method") != "none",
                summary=str(inference.get("method")) if inference else "unavailable",
                reason="no_uncertainty_claim"
                if inference and inference.get("method") == "none"
                else None
                if inference
                else "not_retained_by_native_report",
                payload=inference or {},
            ),
            assumptions=SlotModel(
                available=True, summary="native assumptions", payload={"assumptions": assumptions}
            )
            if assumptions is not None
            else unavailable,
            contract=contract or None,
            target=target,
            inference_binding=inference,
            answer=Answer("structured"),
            calibration=self.report.get("calibration"),
            diagnostics=tuple(item for item in diagnostics if isinstance(item, str)),
            native_diagnostics=freeze(native.get("diagnostics")),
            details={
                **self._summary_fields(),
                **{
                    key: native[key]
                    for key in ("derivation", "scope", "source_identity")
                    if key in native
                },
            },
            family=contract.get("family", self.report.get("answer", {}).get("family", "system")),
            stage="executed" if self._prepared is not None else "consumed",
            operations=("inspect", "to_dict", "export"),
            **ids,
        )

    @property
    def identity(self) -> str:
        """Request digest to retain separately from the artifact."""
        return str(self.report["identity"])

    @artifact_export
    def export(self) -> ExportedArtifact:
        """Return the checksummed native artifact."""
        return artifact_from_receipt(self, self.artifact)


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
    max_bytes = None
    cancel = None
    artifact = container_bytes(artifact)
    report, facts = _native.consume_cyclic_system(
        artifact, expected_identity, max_bytes=max_bytes, cancel=cancel
    )
    return CyclicResult(json.loads(report), artifact, facts)


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
        max_bytes = None
        cancel = None
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
            max_bytes=max_bytes,
            cancel=cancel,
        )
        result = object.__new__(cls)
        object.__setattr__(result, "request_json", None)
        object.__setattr__(result, "_handle", handle)
        object.__setattr__(result, "_use_buffers", noise.size * len(request["worlds"]) >= 64)
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
            max_bytes=None,
            cancel=None,
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
        object.__setattr__(result, "_use_buffers", self.__dict__.get("_use_buffers", False))
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
            max_bytes=None,
            cancel=None,
        )
        result = object.__new__(PreparedCyclicSystem)
        object.__setattr__(result, "request_json", None)
        object.__setattr__(result, "_handle", handle)
        object.__setattr__(result, "_use_buffers", self.__dict__.get("_use_buffers", False))
        return result

    @outcome_operation
    @_with_execution
    def execute(
        self,
        *,
        execution: Execution | None = None,
    ) -> CyclicResult:
        """Execute the retained native product and produce its independent replay artifact."""
        # Tiny results are cheaper to decode directly; bulk values use NumPy buffers.
        assert execution is not None
        if not self.__dict__.get("_use_buffers", False):
            report, artifact, facts = self._handle.execute(
                max_bytes=None,
                cancel=None,
            )
            return CyclicResult(json.loads(report), artifact, facts, self, self.contract())
        report, artifact, values, counterfactuals, width, facts = self._handle.execute_buffers(
            max_bytes=None,
            cancel=None,
        )
        report = json.loads(report)
        offset = 0
        for world in report["worlds"]:
            for row in world:
                row["values"] = values[offset : offset + width].tolist()
                offset += width
        offset = 0
        for row in report["counterfactuals"] or []:
            row["values"] = counterfactuals[offset : offset + width].tolist()
            offset += width
        return CyclicResult(report, artifact, facts, self, self.contract())

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
                max_bytes=None,
                cancel=None,
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
    _law: Any = field(default=None, init=False, repr=False, compare=False)
    _inspection_facts: Any = field(default=None, init=False, repr=False, compare=False)

    def __repr__(self) -> str:
        return "PreparedCyclicFamily()"

    def __post_init__(self) -> None:
        self.execution._call(self._prepare, execution=self.execution)
        self._cache_inspection_facts()

    def _cache_inspection_facts(self) -> None:
        from ._cyclic_typed import family_facts

        native = json.loads(self._handle.inspect())
        facts = family_facts(
            json.loads(self.request_json), self._law, native.get("design_unresolved")
        )
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
            ("_law", None),
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

        For an experimental recovery, ``design_unresolved`` lists the coefficient rows the
        clamp design leaves unresolved (``{"target", "missing_clamps"}`` by unit id), read from
        the native compiler before any fit; it is None for other families. Before a fit on this
        handle ``unresolved_directions`` reports those rows; after one it reports the recovered
        law's directions, which also name rank deficiencies the fit found.
        """
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
        max_bytes = None
        cancel = None
        from ._cyclic_typed import _next_attempt_refusal

        with self._lock:
            self._prepare(execution=execution)
            try:
                payload, facts = self._handle.interventions(max_bytes=max_bytes, cancel=cancel)
            except ValueError as error:
                refusal = _next_attempt_refusal(error, "cyclic_family.interventions")
                if refusal is None:
                    raise
                raise refusal from error
        return CyclicInterventions._from_native(
            json.loads(payload), self.variable_names, facts, self
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
        max_bytes = None
        cancel = None
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
                max_bytes=max_bytes,
                cancel=cancel,
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
        source = json.loads(self.request_json)
        if source.get("family") != "graph" or source["request"].get("operation") not in (
            "identify",
            "adjust",
            "calculus",
        ):
            raise CausalValueError("expected a graph identification preparation")
        result = require_success(self.execute(execution=execution))
        answer = result.report["answer"]
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
        with self._lock:
            handle = self._handle.rebind(request_json)
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
        max_bytes = None
        cancel = None
        if self._handle is None:
            object.__setattr__(
                self,
                "_handle",
                _native.prepare_cyclic_family(
                    self.request_json, max_bytes=max_bytes, cancel=cancel
                ),
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
        max_bytes = None
        cancel = None
        with self._lock:
            self._prepare(execution=execution)
            report, artifact, facts = self._handle.execute(max_bytes=max_bytes, cancel=cancel)
            decoded = json.loads(report)
            answer = decoded.get("answer", {})
            if answer.get("family") == "experimental_recovery":
                object.__setattr__(self, "_law", answer["result"]["law"])
                self._cache_inspection_facts()
        return CyclicResult(decoded, artifact, facts, self, self.contract())


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
    max_bytes = None
    cancel = None
    artifact = container_bytes(artifact)
    report, facts = _native.consume_cyclic_family(
        artifact, expected_identity, max_bytes=max_bytes, cancel=cancel
    )
    return CyclicResult(json.loads(report), artifact, facts)


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
        max_bytes = None
        cancel = None
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
                max_bytes=max_bytes,
                cancel=cancel,
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

    @property
    def report(self) -> dict[str, Any]:
        """Return derived scientific coordinates, original diagnostics and failed slots."""
        return dict(json.loads(self._handle.report()))

    @property
    def quantities(self) -> tuple[Any, ...]:
        """Typed scientific coordinates issued by the native evidence projection."""
        from .joint_distribution import ScientificQuantity

        return tuple(
            ScientificQuantity.from_wire(column["quantity"]) for column in self.report["columns"]
        )

    @property
    def identity(self) -> str:
        """Source/schema/world/time/selector identity for independent consumption."""
        return str(self.report["identity"])

    @artifact_export
    @_with_execution
    def export(
        self,
        *,
        execution: Execution | None = None,
    ) -> ExportedArtifact:
        """Export original source and projection, preserving replay and uncertainty meaning."""
        assert execution is not None
        max_bytes = None
        cancel = None
        return artifact_from_receipt(
            self, bytes(self._handle.export(max_bytes=max_bytes, cancel=cancel))
        )

    @_with_execution
    @outcome_operation
    def joint_law(
        self,
        *,
        execution: Execution | None = None,
    ) -> Any:
        """Use the existing joint-distribution/decision API for complete aligned laws.

        Bootstrap laws, numerical root sets, finite mass bounds, uncertified roots
        and failed posterior slots cannot be silently coerced to an outcome law.
        """
        assert execution is not None
        max_bytes = None
        cancel = None
        from .joint_distribution import JointDistributionArtifact

        return JointDistributionArtifact._from_native(
            self._handle.joint_law(max_bytes=max_bytes, cancel=cancel)
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
    max_bytes = None
    cancel = None
    artifact = container_bytes(artifact)
    return CyclicEvidence(
        _native.consume_cyclic_evidence(
            artifact, expected_identity, max_bytes=max_bytes, cancel=cancel
        )
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
    "CyclicRefusal",
    "CyclicResult",
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
    "OuStationaryCovariance",
    "OuStationaryMean",
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
