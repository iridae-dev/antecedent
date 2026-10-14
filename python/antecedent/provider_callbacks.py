"""Arbitrary law, posterior and nested callbacks.

A callback is a function Antecedent does not own: a law, a posterior or a composition of them that
the library can ask questions of but cannot inspect. This module declares one, plans and executes
requests against it, and exports and resumes its results::

    from antecedent import provider_callbacks as pc
    from antecedent.joint_distribution import ScientificQuantity

    outcome = ScientificQuantity.of(
        "outcome", "systolic", units="mmHg", population="trial", regime="do(a=1)"
    )
    declaration = pc.CallbackDeclaration(
        "gaussian_law", "law", pc.ProviderIdentity("lab", "gaussian", "1.0", "snap", "req"),
        outcome, "estimand:outcome_law", "interventional_predictive", pc.Support.real(),
        operations=("mean", "cdf", "quantile"),
        dependencies=pc.Dependencies.complete([pc.Dependency("model", "parameters", "mu=2")]),
        replay=pc.Replay.deterministic_pure(),
        trust=pc.Trust.externally_attested("lab"),
        evidence=(),
    )
    session = pc.CallbackSession()
    result = session.execute(provider, declaration, pc.Request(pc.Query.mean()))

What each part means
--------------------
* Mean, sample, cdf, quantile and posterior update are negotiated separately: one never implies
  another. A mean does not supply a law; draws do not supply a distribution function.
* The replay contract (deterministic pure, seeded stochastic, stateful, side-effecting) decides
  when a stored answer is reused. Opaque or undeclared dependencies make reuse ineligible; a
  stateful callback is reused only while the state identity it reports is unchanged; a
  side-effecting callback is never invoked while planning or verifying and is retried only under
  its declared idempotent contract.
* An answer is accepted only as a well-formed law of the declared quantity (finite, in support,
  a distribution function in [0, 1] and non-decreasing). A malformed law is refused whole.
* A result is ``diagnostic`` or ``licensed_external``; neither is native. Metadata cannot mint
  native trust, a joint law or an identification licence, and a callback supplies no interval or
  posterior guarantee (:func:`refuse_interval` and the other ``refuse_*`` functions).

Code execution
--------------
A provider is a Python object *you* pass to :meth:`CallbackSession.execute` or
:meth:`CallbackSession.resume`; its ``invoke`` runs only there, in your process, with your
authority, and only after the library has checked the declaration, the operation, the provider's
identity and the reuse plan. Planning (:meth:`CallbackSession.plan`), :func:`consume_callback_result`
and :func:`inspect_callback_result` never call a provider: an artifact carries data only and
cannot name, select or trigger a callable. Nothing is pickled. Values cross to Rust as JSON and
IEEE bit patterns and are validated there. See ``docs/callback-security.md``.

Every identity, decision and refusal is computed in Rust; refusals are :class:`CallbackRefusal`
(a :class:`~antecedent.errors.CausalUnsupportedError`) with a registered ``reason_code``, a
``callback_protocol.*`` ``detail`` and a ``remedy``.
"""

from __future__ import annotations

import json
import random
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from . import _native
from ._artifact_protocol import bytes_export
from ._freeze import thaw
from ._refusal import StructuredRefusal
from ._scientific import ScientificResult
from ._validate import count as _count
from .errors import CausalTypeError, CausalValueError
from .scientific_bridge import _quantity_wire

if TYPE_CHECKING:
    from ._artifact_protocol import ExportedArtifact

__all__ = [
    "CallbackConsumed",
    "CallbackContext",
    "CallbackDeclaration",
    "CallbackExecution",
    "CallbackExecutionRefusal",
    "CallbackPlanDecision",
    "CallbackRefusal",
    "CallbackResume",
    "CallbackSession",
    "CallbackToken",
    "Dependencies",
    "Dependency",
    "ExecutionPolicy",
    "ExportedCallbackResult",
    "FunctionProvider",
    "Input",
    "Output",
    "ProviderCallback",
    "ProviderIdentity",
    "Query",
    "Replay",
    "Request",
    "Support",
    "Trust",
    "consume_callback_result",
    "inspect_callback_result",
    "refuse_identification_licence",
    "refuse_interval",
    "refuse_joint_law",
    "refuse_native_trust",
    "refuse_posterior_guarantee",
    "values_digest",
]

#: Re-exported cooperative cancellation token passed to a provider's ``invoke``.
CallbackToken = _native.CallbackToken

KINDS = ("law", "posterior", "nested")
OPERATIONS = ("mean", "sample", "cdf", "quantile", "posterior_update")
DEPENDENCY_KINDS = (
    "data",
    "model",
    "environment",
    "file",
    "remote_service",
    "closure_state",
)
PHASES = ("planning", "verification", "execution")
MEANINGS = (
    "parameter_posterior",
    "causal_functional_posterior",
    "posterior_predictive",
    "interventional_predictive",
    "estimator_sampling",
    "bootstrap",
    "empirical_outcome",
)
_UNCERTAIN_OPERATIONS = frozenset({"sample", "cdf", "quantile", "posterior_update"})


KindsName = Literal["law", "posterior", "nested"]
OperationsName = Literal["mean", "sample", "cdf", "quantile", "posterior_update"]
DependencyKindsName = Literal[
    "data", "model", "environment", "file", "remote_service", "closure_state"
]
PhasesName = Literal["planning", "verification", "execution"]
MeaningsName = Literal[
    "parameter_posterior",
    "causal_functional_posterior",
    "posterior_predictive",
    "interventional_predictive",
    "estimator_sampling",
    "bootstrap",
    "empirical_outcome",
]


class CallbackRefusal(StructuredRefusal):
    """A typed refusal of the callback route (registered ``reason_code``, ``detail``, ``remedy``)."""

    def __init__(self, refusal: Mapping[str, Any], *, text: str | None = None) -> None:
        super().__init__(refusal, text=text)
        #: What was expected, when a comparison failed.
        self.expected: str | None = refusal.get("expected")
        #: What was supplied, when a comparison failed.
        self.supplied: str | None = refusal.get("supplied")


class CallbackExecutionRefusal(CallbackRefusal):
    """A refused or failed execution, with what actually ran."""

    def __init__(
        self, refusal: Mapping[str, Any], *, invocations: int, request_attempts: int
    ) -> None:
        super().__init__(refusal)
        #: Provider invocations the failed call made before it ended.
        self.invocations = invocations
        #: Provider invocations of this request so far, the failed call included.
        self.request_attempts = request_attempts


def _raise(payload: str | None) -> None:
    if payload is None:
        return
    wire = json.loads(payload)
    if isinstance(wire, dict) and "refusal" in wire:
        raise CallbackExecutionRefusal(
            wire["refusal"],
            invocations=int(wire["invocations"]),
            request_attempts=int(wire["request_attempts"]),
        )
    raise CallbackRefusal(wire)


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalTypeError(f"{what} must be a non-empty string")
    return value


def _choice(value: object, what: str, allowed: Iterable[str]) -> str:
    text = _name(value, what)
    options = tuple(allowed)
    if text not in options:
        raise CausalValueError(f"{what} must be one of {options}, got {text!r}")
    return text


def _bits(value: float) -> int:
    return int(struct.unpack("<Q", struct.pack("<d", float(value)))[0])


def _dumps(value: object) -> str:
    return json.dumps(thaw(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


# -- declaration -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    """Provider, object, version, snapshot and exact-request identity of a provider object."""

    provider_id: str
    object_id: str
    version_id: str
    snapshot_id: str
    request_id: str

    def _wire(self) -> dict[str, str]:
        return {
            "provider_id": _name(self.provider_id, "provider_id"),
            "object_id": _name(self.object_id, "object_id"),
            "version_id": _name(self.version_id, "version_id"),
            "snapshot_id": _name(self.snapshot_id, "snapshot_id"),
            "request_id": _name(self.request_id, "request_id"),
        }


@dataclass(frozen=True, slots=True)
class Support:
    """The numeric values a quantity takes."""

    kind: str
    lower: float | None = None
    upper: float | None = None

    @classmethod
    def real(cls) -> Support:
        """Any finite real."""
        return cls("real")

    @classmethod
    def non_negative(cls) -> Support:
        """Zero or positive."""
        return cls("non_negative")

    @classmethod
    def binary(cls) -> Support:
        """Zero or one."""
        return cls("binary")

    @classmethod
    def interval(cls, lower: float, upper: float) -> Support:
        """The closed interval ``[lower, upper]``."""
        return cls("interval", float(lower), float(upper))

    def _wire(self) -> dict[str, Any]:
        return {"kind": self.kind, "lower": self.lower, "upper": self.upper}


@dataclass(frozen=True, slots=True)
class Dependency:
    """One dependency at the version or digest it was used at."""

    kind: str
    name: str
    version: str

    def _wire(self) -> dict[str, str]:
        return {
            "kind": _choice(self.kind, "dependency kind", DEPENDENCY_KINDS),
            "name": _name(self.name, "dependency name"),
            "version": _name(self.version, "dependency version"),
        }


@dataclass(frozen=True, slots=True)
class Dependencies:
    """Whether the callback's dependencies are fully declared."""

    status: str
    items: tuple[Dependency, ...] = ()

    @classmethod
    def complete(cls, items: Iterable[Dependency]) -> Dependencies:
        """Every dependency listed (an empty list says there are none)."""
        return cls("complete", tuple(items))

    @classmethod
    def opaque(cls) -> Dependencies:
        """The provider has dependencies it will not describe: reuse is ineligible."""
        return cls("opaque")

    @classmethod
    def undeclared(cls) -> Dependencies:
        """The producer did not say: reuse is ineligible."""
        return cls("undeclared")

    def _wire(self) -> dict[str, Any]:
        return {"status": self.status, "items": [d._wire() for d in self.items]}


@dataclass(frozen=True, slots=True)
class Replay:
    """The randomness, state and side-effect contract."""

    kind: str
    stream_id: str | None = None
    exposes_state_identity: bool | None = None
    retry: str | None = None
    max_attempts: int | None = None

    @classmethod
    def deterministic_pure(cls) -> Replay:
        """A pure function of the declared dependencies and the request."""
        return cls("deterministic_pure")

    @classmethod
    def seeded_stochastic(cls, stream_id: str) -> Replay:
        """Draws from the stream with this semantic identity, derived from the seed."""
        return cls("seeded_stochastic", stream_id=_name(stream_id, "stream_id"))

    @classmethod
    def stateful(cls, *, exposes_state_identity: bool) -> Replay:
        """Depends on state that changes; reuse needs an exposed state identity."""
        return cls("stateful", exposes_state_identity=bool(exposes_state_identity))

    @classmethod
    def side_effecting(cls, *, max_attempts: int | None = None) -> Replay:
        """Has effects outside the process. ``max_attempts`` declares an idempotent retry."""
        if max_attempts is None:
            return cls("side_effecting", retry="no_retry")
        return cls(
            "side_effecting",
            retry="idempotent_key",
            max_attempts=_count(
                max_attempts,
                "max_attempts",
                minimum=0,
                maximum=2**64 - 1,
                reason_code="invalid_argument",
            ),
        )

    def _wire(self) -> dict[str, Any]:
        _choice(
            self.kind,
            "replay kind",
            ("deterministic_pure", "seeded_stochastic", "stateful", "side_effecting"),
        )
        return {
            "kind": self.kind,
            "stream_id": self.stream_id,
            "exposes_state_identity": self.exposes_state_identity,
            "retry": self.retry,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True, slots=True)
class Trust:
    """The trust the provider claims. There is no native level."""

    level: str
    attestor: str | None = None
    receipt_digest: str | None = None

    @classmethod
    def unverified(cls) -> Trust:
        """No claim."""
        return cls("unverified")

    @classmethod
    def externally_attested(cls, attestor: str) -> Trust:
        """The supplier asserts it; no numerical check was performed here."""
        return cls("externally_attested", attestor=_name(attestor, "attestor"))

    @classmethod
    def verified_extension(cls, receipt_digest: str) -> Trust:
        """The exact provider contract passed object-level verification, in this receipt."""
        return cls("verified_extension", receipt_digest=_name(receipt_digest, "receipt_digest"))

    def _wire(self) -> dict[str, Any]:
        return {
            "level": _choice(
                self.level,
                "trust level",
                ("unverified", "externally_attested", "verified_extension"),
            ),
            "attestor": self.attestor,
            "receipt_digest": self.receipt_digest,
        }


@dataclass(frozen=True, slots=True)
class Input:
    """One input of a callback: its quantity, the operation consumed and who supplies it."""

    name: str
    quantity: Any
    semantic_identity: str
    support: Support
    operation: str
    uncertainty: str | None = None
    callback_identity: str | None = None
    evidence: tuple[str, ...] | None = None

    @classmethod
    def from_callback(cls, name: str, inner: CallbackDeclaration, operation: str = "mean") -> Input:
        """An input supplied by another callback (a nested callback)."""
        _choice(operation, "operation", OPERATIONS)
        return cls(
            name,
            inner.quantity,
            inner.semantic_identity,
            inner.support,
            operation,
            uncertainty=inner.meaning if operation in _UNCERTAIN_OPERATIONS else None,
            callback_identity=inner.identity,
            evidence=inner.effective_evidence(),
        )

    def _wire(self) -> dict[str, Any]:
        supplied = self.callback_identity is not None
        return {
            "name": _name(self.name, "input name"),
            "quantity": _quantity_wire(self.quantity, f"input {self.name} quantity"),
            "semantic_identity": _name(self.semantic_identity, "semantic_identity"),
            "support": self.support._wire(),
            "operation": _choice(self.operation, "operation", OPERATIONS),
            "uncertainty": self.uncertainty,
            "supplier": "callback" if supplied else "external",
            "callback_identity": self.callback_identity,
            "evidence": {
                "declared": self.evidence is not None and supplied,
                "sources": sorted(self.evidence or ()) if supplied else [],
            },
        }


@dataclass(frozen=True, slots=True)
class CallbackDeclaration:
    """A callback as the protocol declares it (an instance of the interface-contract schema).

    ``evidence`` lists the evidence-ledger source identities (hex) the callback rests on; ``None``
    leaves its overlap with other evidence unknown. ``update`` is
    ``(candidate_id, prior_id, observation_quantity)`` and is required exactly when
    ``posterior_update`` is an operation.
    """

    component: str
    kind: str
    provider: ProviderIdentity
    quantity: Any
    semantic_identity: str
    meaning: str
    support: Support
    operations: tuple[str, ...]
    dependencies: Dependencies
    replay: Replay
    trust: Trust
    evidence: tuple[str, ...] | None = None
    update: tuple[str, str, Any] | None = None
    inputs: tuple[Input, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(sorted(set(self.operations))))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if self.evidence is not None:
            object.__setattr__(self, "evidence", tuple(sorted(set(self.evidence))))

    def _wire(self) -> dict[str, Any]:
        update = None
        if self.update is not None:
            candidate, prior, observation = self.update
            update = {
                "candidate_id": _name(candidate, "candidate_id"),
                "prior_id": _name(prior, "prior_id"),
                "observation": _quantity_wire(observation, "update observation"),
            }
        return {
            "component": _name(self.component, "component"),
            "kind": _choice(self.kind, "kind", KINDS),
            "identity": self.provider._wire(),
            "quantity": _quantity_wire(self.quantity, "quantity"),
            "semantic_identity": _name(self.semantic_identity, "semantic_identity"),
            "meaning": _choice(self.meaning, "meaning", MEANINGS),
            "support": self.support._wire(),
            "operations": [_choice(o, "operation", OPERATIONS) for o in self.operations],
            "update": update,
            "inputs": [i._wire() for i in self.inputs],
            "dependencies": self.dependencies._wire(),
            "replay": self.replay._wire(),
            "trust": self.trust._wire(),
            "evidence": {
                "declared": self.evidence is not None,
                "sources": list(self.evidence or ()),
            },
        }

    def _native(self) -> dict[str, Any]:
        wire, refusal = _native.callback_declaration_identity(_dumps(self._wire()))
        _raise(refusal)
        assert wire is not None
        return json.loads(wire)

    @property
    def identity(self) -> str:
        """Canonical identity (hex): binds dependency versions, replay contract and provider."""
        return str(self._native()["identity"])

    @property
    def contract_identity(self) -> str:
        """Identity (hex) of the declaration's projection into the interface contract."""
        return str(self._native()["contract_identity"])

    def effective_evidence(self) -> tuple[str, ...] | None:
        """The evidence the output rests on, inner callbacks included; ``None`` if unknown."""
        evidence = self._native()["effective_evidence"]
        return tuple(evidence["sources"]) if evidence["declared"] else None

    def validate(self) -> str:
        """Check the declaration is well formed and return its identity.

        Raises:
            CallbackRefusal: the first defect.
        """
        return self.identity


# -- requests ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Query:
    """One question put to a callback."""

    operation: str
    draws: int | None = None
    points: tuple[float, ...] = ()
    observation: float | None = None

    @classmethod
    def mean(cls) -> Query:
        """The mean."""
        return cls("mean")

    @classmethod
    def sample(cls, draws: int) -> Query:
        """Marginal draws."""
        return cls(
            "sample",
            draws=_count(
                draws, "draws", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
            ),
        )

    @classmethod
    def cdf(cls, thresholds: Sequence[float]) -> Query:
        """The distribution function at finite, strictly increasing thresholds."""
        return cls("cdf", points=tuple(float(t) for t in thresholds))

    @classmethod
    def quantile(cls, levels: Sequence[float]) -> Query:
        """The quantile function at levels strictly inside (0, 1), strictly increasing."""
        return cls("quantile", points=tuple(float(p) for p in levels))

    @classmethod
    def posterior_update(cls, observation: float, draws: int = 0) -> Query:
        """The posterior after one possible observation."""
        return cls(
            "posterior_update",
            draws=_count(
                draws, "draws", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
            ),
            observation=float(observation),
        )

    def _wire(self) -> dict[str, Any]:
        return {
            "operation": _choice(self.operation, "operation", OPERATIONS),
            "draws": self.draws,
            "points": list(self.points),
            "observation": self.observation,
        }


@dataclass(frozen=True, slots=True)
class Request:
    """A query with its seed, the digests of supplied inputs and an idempotency key."""

    query: Query
    seed: int = 0
    input_digests: Mapping[str, str] = field(default_factory=dict)
    idempotency_key: str | None = None

    def _wire(self) -> dict[str, Any]:
        if (
            not 0
            <= _count(
                self.seed, "seed", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
            )
            < 2**64
        ):
            raise CausalValueError("seed must be an unsigned 64-bit integer")
        return {
            "query": self.query._wire(),
            "seed": _count(
                self.seed, "seed", minimum=0, maximum=2**64 - 1, reason_code="invalid_argument"
            ),
            "input_digests": {str(k): str(v) for k, v in self.input_digests.items()},
            "idempotency_key": self.idempotency_key,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    """The explicit execution policy of one call.

    ``wall_limit_ms`` is enforced after the call returns (the late result is discarded; an
    in-process callable is not interrupted). ``allow_side_effects`` is an opt-in for one request.
    """

    max_invocations: int = 1
    max_output_values: int = 1_000_001
    wall_limit_ms: int | None = None
    allow_side_effects: bool = False
    phase: str = "execution"

    def _wire(self) -> dict[str, Any]:
        return {
            "max_invocations": _count(
                self.max_invocations,
                "max_invocations",
                minimum=0,
                maximum=2**64 - 1,
                reason_code="invalid_argument",
            ),
            "max_output_values": _count(
                self.max_output_values,
                "max_output_values",
                minimum=0,
                maximum=2**64 - 1,
                reason_code="invalid_argument",
            ),
            "wall_limit_ms": self.wall_limit_ms,
            "allow_side_effects": bool(self.allow_side_effects),
            "phase": _choice(self.phase, "phase", PHASES),
        }


@dataclass(frozen=True, slots=True)
class Output:
    """What a provider returns: the operation answered and its values.

    ``values`` is the mean (one value), the draws, the distribution function values, the quantiles,
    or for a posterior update the posterior mean followed by its draws.
    """

    operation: str
    values: tuple[float, ...]

    @classmethod
    def mean(cls, value: float) -> Output:
        """A mean."""
        return cls("mean", (float(value),))

    @classmethod
    def sample(cls, draws: Iterable[float]) -> Output:
        """Marginal draws."""
        return cls("sample", tuple(float(v) for v in draws))

    @classmethod
    def cdf(cls, values: Iterable[float]) -> Output:
        """Distribution function values, one per threshold."""
        return cls("cdf", tuple(float(v) for v in values))

    @classmethod
    def quantile(cls, values: Iterable[float]) -> Output:
        """Quantiles, one per level."""
        return cls("quantile", tuple(float(v) for v in values))

    @classmethod
    def posterior_update(cls, mean: float, draws: Iterable[float] = ()) -> Output:
        """The updated posterior mean and draws."""
        return cls("posterior_update", (float(mean), *(float(v) for v in draws)))


@dataclass(frozen=True, slots=True)
class CallbackContext:
    """What one invocation is given.

    ``stream_seed`` is present only for a seeded callback: the seed of its semantic stream, fresh
    for every invocation, so :meth:`rng` replays bit for bit. ``inputs`` maps an input name to the
    accepted output of the callback that supplied it.
    """

    stream_seed: int | None
    attempt: int
    idempotency_key: str | None
    inputs: Mapping[str, Output]
    token: Any

    def rng(self) -> random.Random:
        """The deterministic generator of a seeded callback.

        Raises:
            CausalValueError: the callback is not seeded stochastic.
        """
        if self.stream_seed is None:
            raise CausalValueError("only a seeded stochastic callback is given a stream")
        return random.Random(self.stream_seed)

    @property
    def cancelled(self) -> bool:
        """Whether cancellation was requested."""
        return bool(self.token.is_cancelled())


@runtime_checkable
class ProviderCallback(Protocol):
    """A provider the caller supplies. It runs only when passed to ``execute`` or ``resume``."""

    identity: ProviderIdentity

    def invoke(self, query: Query, context: CallbackContext) -> Output | Mapping[str, Any]:
        """One invocation; raise to report a failure (no claim is issued)."""
        ...


@dataclass(slots=True)
class FunctionProvider:
    """A provider from a function ``fn(query, context) -> Output``.

    ``state`` is an optional zero-argument function returning the state identity a stateful
    callback exposes through the protocol (a version counter, a weights digest).
    """

    identity: ProviderIdentity
    fn: Callable[[Query, CallbackContext], Output | Mapping[str, Any]]
    state: Callable[[], str | None] | None = None

    def invoke(self, query: Query, context: CallbackContext) -> Output | Mapping[str, Any]:
        """Call the function."""
        return self.fn(query, context)

    def state_identity(self) -> str | None:
        """The exposed state identity, when there is one."""
        return None if self.state is None else self.state()


# -- results -----------------------------------------------------------------------------------


@dataclass(repr=False, frozen=True, slots=True)
class CallbackExecution(ScientificResult):
    """An accepted answer."""

    output: Output
    output_digest: str
    invocations: int
    reused: bool
    recomputed_because: str | None
    standing: str
    declaration_identity: str
    request_identity: str
    state_identity: str | None
    history_index: int | None
    _session: Any = field(default=None, repr=False, compare=False)
    _owner_token: Any = field(default=None, repr=False, compare=False)

    @bytes_export
    def export(self, artifact_id: str = "callback-result") -> bytes:
        if self._session is None:
            raise CausalValueError("this execution has no owning session; use artifact resume")
        return self._session.export(self, artifact_id)

    def artifact(self, artifact_id: str = "callback-result") -> ExportedArtifact[CallbackConsumed]:
        from ._artifact_protocol import exported_artifact

        return exported_artifact(self._session, self, artifact_id=artifact_id)


@dataclass(frozen=True, slots=True)
class CallbackPlanDecision:
    """What planning decides: ``reuse``, ``recompute`` (with a reason) or ``refuse``."""

    plan: str
    reason: str | None = None
    refusal: CallbackRefusal | None = None


@dataclass(frozen=True, slots=True)
class CallbackResume:
    """The outcome of resuming a stored result: ``answered`` or ``unavailable``."""

    kind: str
    execution: CallbackExecution | None = None
    record: int | None = None
    refusal: CallbackRefusal | None = None


@dataclass(frozen=True, slots=True)
class ExportedCallbackResult:
    """A sealed ``callback_result_v1`` artifact and its identity."""

    artifact: bytes
    identity: str


@dataclass(repr=False, frozen=True, slots=True)
class CallbackConsumed(ScientificResult):
    """A consumed artifact: every identity recomputed. Nothing was executed."""

    identity: str
    declaration_identity: str
    request_identity: str
    stage: str
    values: tuple[float, ...]
    invocations: int
    state_identity: str | None
    reason: str | None
    declaration: Mapping[str, Any]
    request: Mapping[str, Any]


def _execution(wire: Mapping[str, Any], session: Any = None) -> CallbackExecution:
    output = wire["output"]
    return CallbackExecution(
        output=Output(output["operation"], tuple(float(v) for v in output["values"])),
        output_digest=wire["output_digest"],
        invocations=int(wire["invocations"]),
        reused=bool(wire["reused"]),
        recomputed_because=wire["recomputed_because"],
        standing=wire["standing"],
        declaration_identity=wire["declaration_identity"],
        request_identity=wire["request_identity"],
        state_identity=wire["state_identity"],
        history_index=wire["history_index"],
        _session=session,
        _owner_token=None if session is None else session._owner_token,
    )


def _provider_parts(provider: Any) -> tuple[str, Callable[..., str], Callable[[], Any] | None]:
    identity = getattr(provider, "identity", None)
    invoke = getattr(provider, "invoke", None)
    if not isinstance(identity, ProviderIdentity) or not callable(invoke):
        raise CausalTypeError(
            "a provider needs an `identity` (ProviderIdentity) and an `invoke(query, context)`"
        )
    state_fn = getattr(provider, "state_identity", None)

    def call(query_json: str, context_json: str, token: Any) -> str:
        wire = json.loads(query_json)
        query = Query(
            wire["operation"],
            wire["draws"],
            tuple(wire["points"]),
            wire["observation"],
        )
        raw = json.loads(context_json)
        context = CallbackContext(
            stream_seed=raw["stream_seed"],
            attempt=int(raw["attempt"]),
            idempotency_key=raw["idempotency_key"],
            inputs={
                name: Output(v["operation"], tuple(float(x) for x in v["values"]))
                for name, v in raw["inputs"].items()
            },
            token=token,
        )
        answer = invoke(query, context)
        if isinstance(answer, Mapping):
            answer = Output(str(answer["operation"]), tuple(answer["values"]))
        if not isinstance(answer, Output):
            raise CausalTypeError("a provider returns an Output")
        return json.dumps(
            {"operation": answer.operation, "values_bits": [_bits(v) for v in answer.values]}
        )

    return _dumps(identity._wire()), call, state_fn if callable(state_fn) else None


class CallbackSession:
    """Execution state of one process: the history of results and the invocation counts."""

    def __init__(self) -> None:
        self._owner_token = object()
        self._handle = _native.CallbackSessionHandle()

    @property
    def invocations(self) -> int:
        """Provider invocations so far, failed attempts included (a reuse counts zero)."""
        return int(self._handle.invocations())

    @property
    def state_probes(self) -> int:
        """State-identity probes so far; a probe reads reported state and is not an invocation."""
        return int(self._handle.state_probes())

    def request_attempts(self, request_identity: str) -> int:
        """Provider invocations of one request identity."""
        return int(self._handle.request_attempts(request_identity))

    def __len__(self) -> int:
        return int(self._handle.history_len())

    def record(self, index: int) -> dict[str, Any]:
        """One history record (declaration, request, stage, values) as a mapping."""
        return json.loads(self._handle.record_json(index))

    def dependents_of(self, identity: str) -> tuple[int, ...]:
        """History indices of results that consumed the callback with this declaration identity."""
        return tuple(self._handle.dependents_of(identity))

    def plan(
        self,
        declaration: CallbackDeclaration,
        request: Request,
        *,
        current_state: str | None = None,
    ) -> CallbackPlanDecision:
        """Decide reuse from declarations and stored records. Takes no provider: it cannot run one."""
        wire, refusal = self._handle.plan(
            _dumps(declaration._wire()), _dumps(request._wire()), current_state
        )
        _raise(refusal)
        assert wire is not None
        decoded = json.loads(wire)
        return CallbackPlanDecision(
            decoded["plan"],
            decoded.get("reason"),
            CallbackRefusal(decoded["refusal"]) if "refusal" in decoded else None,
        )

    def execute(
        self,
        provider: ProviderCallback,
        declaration: CallbackDeclaration,
        request: Request,
        *,
        inputs: Mapping[str, CallbackExecution] | None = None,
        policy: ExecutionPolicy | None = None,
        token: Any = None,
    ) -> CallbackExecution:
        """Answer ``request`` from the stored result or by invoking ``provider``.

        The provider's ``invoke`` runs here and only here, after the declaration, the policy phase,
        the negotiated operation, the provider's identity, the inputs and the reuse plan have been
        checked. ``inputs`` are the accepted executions of the callbacks this one consumes.

        Raises:
            CallbackExecutionRefusal: refused or failed; carries the invocation counts.
        """
        identity_json, invoke, state_fn = _provider_parts(provider)
        wire, refusal = self._handle.execute(
            identity_json,
            invoke,
            state_fn,
            _dumps(declaration._wire()),
            _dumps(request._wire()),
            _dumps(self._input_indices(inputs)),
            _dumps((policy or ExecutionPolicy())._wire()),
            token if token is not None else CallbackToken(),
        )
        _raise(refusal)
        assert wire is not None
        return _execution(json.loads(wire), self)

    def resume(
        self,
        artifact: bytes,
        *,
        expected_identity: str | None = None,
        provider: ProviderCallback | None = None,
        inputs: Mapping[str, CallbackExecution] | None = None,
        policy: ExecutionPolicy | None = None,
        token: Any = None,
    ) -> CallbackResume:
        """Resume a stored result in this process.

        The artifact is consumed first (every identity recomputed). With a ``provider`` that is the
        declared object the request is answered by reuse or recomputation under the replay
        contract; with none the stage is recorded ``unavailable`` and nothing runs. A provider
        that is not the declared object is refused.

        Raises:
            CallbackExecutionRefusal: the artifact or the provider is refused.
        """
        identity_json = invoke = state_fn = None
        if provider is not None:
            identity_json, invoke, state_fn = _provider_parts(provider)
        wire, refusal = self._handle.resume(
            bytes(artifact),
            expected_identity,
            identity_json,
            invoke,
            state_fn,
            _dumps(self._input_indices(inputs)),
            _dumps((policy or ExecutionPolicy())._wire()),
            token if token is not None else CallbackToken(),
        )
        _raise(refusal)
        assert wire is not None
        decoded = json.loads(wire)
        if decoded["resume"] == "answered":
            return CallbackResume("answered", execution=_execution(decoded["execution"], self))
        return CallbackResume(
            "unavailable",
            record=int(decoded["record"]),
            refusal=CallbackRefusal(decoded["refusal"]),
        )

    def _check_execution(self, execution: CallbackExecution) -> int:
        if (
            execution._session is not self
            or execution._owner_token is not self._owner_token
            or execution.history_index is None
        ):
            raise CallbackRefusal(
                {
                    "code": "invalid_argument",
                    "stage": "callback",
                    "detail": "provider_callbacks.foreign_execution",
                    "offending": "execution",
                    "remedy": "export and resume the execution in this session",
                }
            )
        index = _count(execution.history_index, "history_index", minimum=0, maximum=2**64 - 1)
        record = self.record(index)
        for name in ("declaration_identity", "request_identity", "output_digest", "state_identity"):
            if record.get(name) != getattr(execution, name):
                raise CallbackRefusal(
                    {
                        "code": "invalid_argument",
                        "stage": "callback",
                        "detail": "provider_callbacks.execution_identity_mismatch",
                        "offending": name,
                        "remedy": "use the execution returned by this session",
                    }
                )
        if values_digest(execution.output.values) != execution.output_digest:
            raise CallbackRefusal(
                {
                    "code": "invalid_argument",
                    "stage": "callback",
                    "detail": "provider_callbacks.execution_output_mismatch",
                    "offending": "output",
                    "remedy": "use the immutable execution returned by this session",
                }
            )
        return index

    def _input_indices(self, inputs: Mapping[str, CallbackExecution] | None) -> dict[str, int]:
        return {
            name: self._check_execution(execution) for name, execution in (inputs or {}).items()
        }

    @bytes_export
    def export(self, result: CallbackExecution | int, artifact_id: str) -> ExportedCallbackResult:
        """Seal one history record as ``callback_result_v1``.

        A declaration and a portable output, not a promise that a callable runs elsewhere.
        """
        index = (
            _count(result, "history_index", minimum=0, maximum=2**64 - 1)
            if isinstance(result, int)
            else self._check_execution(result)
        )
        if index is None:
            raise CausalValueError("the result has no history record to export")
        done, refusal = self._handle.export_record(int(index), artifact_id)
        _raise(refusal)
        assert done is not None
        return ExportedCallbackResult(bytes(done[0]), done[1])


# -- artifacts and closed routes ---------------------------------------------------------------


def consume_callback_result(
    artifact: bytes, expected_identity: str | None = None
) -> CallbackConsumed:
    """Consume a ``callback_result_v1`` artifact: every identity recomputed, every bound checked.

    Nothing is executed and no provider is called. ``expected_identity`` is an identity retained
    independently of the bytes.

    Raises:
        CallbackRefusal: corruption, a bound, a refused declaration or output, a native trust
            claim or an identity mismatch.
    """
    wire, refusal = _native.callback_result_consume(bytes(artifact), expected_identity)
    _raise(refusal)
    assert wire is not None
    d = json.loads(wire)
    return CallbackConsumed(
        identity=d["identity"],
        declaration_identity=d["declaration_identity"],
        request_identity=d["request_identity"],
        stage=d["stage"],
        values=tuple(float(v) for v in d["values"]),
        invocations=int(d["invocations"]),
        state_identity=d["state_identity"],
        reason=d["reason"],
        declaration=d["declaration"],
        request=d["request"],
    )


def inspect_callback_result(artifact: bytes) -> dict[str, Any]:
    """Read the stored metadata without replaying or executing anything."""
    wire, refusal = _native.callback_result_inspect(bytes(artifact))
    _raise(refusal)
    assert wire is not None
    return json.loads(wire)


def values_digest(values: Iterable[float]) -> str:
    """The digest of output values, bit-exact, as used to bind a nested callback's input."""
    return str(_native.callback_values_digest([float(v) for v in values]))


def _closed(route: str) -> None:
    raise CallbackRefusal(json.loads(_native.callback_closed_refusal(route)))


def refuse_interval() -> None:
    """An interval from a callback law: closed (``cell_not_licensed``).

    Raises:
        CallbackRefusal: always.
    """
    _closed("interval")


def refuse_posterior_guarantee() -> None:
    """A posterior or probability guarantee from a callback: closed.

    Raises:
        CallbackRefusal: always.
    """
    _closed("posterior_guarantee")


def refuse_native_trust() -> None:
    """Native trust conferred by callback metadata: closed.

    Raises:
        CallbackRefusal: always.
    """
    _closed("native_trust")


def refuse_identification_licence() -> None:
    """An identification licence from a callback: closed.

    Raises:
        CallbackRefusal: always.
    """
    _closed("identification_licence")


def refuse_joint_law() -> None:
    """A joint law from independently produced callback draws: closed (``joint_law_required``).

    Raises:
        CallbackRefusal: always.
    """
    _closed("joint_law")
