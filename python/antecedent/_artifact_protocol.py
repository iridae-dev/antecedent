"""Typed artifact dispatch. Family consumers retain their scientific authority."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from functools import wraps
from importlib import import_module
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar, overload

from ._freeze import freeze, thaw
from ._scientific import ScientificResult
from .errors import CausalSerializationError, CausalTypeError, CausalValueError
from .execution import Execution

T = TypeVar("T")
# The result type is a read-only label: no result value is accepted or mutated.
ArtifactT_co = TypeVar("ArtifactT_co", covariant=True)
P = ParamSpec("P")

_PERIOD_ARTIFACT_KINDS = frozenset(
    {"ANTECEDENT-PERIOD-REFRESH", "ANTECEDENT-PERIOD-REFRESH-SUMMARY"}
)


if TYPE_CHECKING:
    from .recalc_composite import CompositeSession


@dataclass(repr=False, frozen=True, slots=True)
class ArtifactExpectation(ScientificResult):
    """A family-specific identity binding retained separately from container bytes."""

    kind: str
    identity: Any

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise CausalTypeError("ArtifactExpectation.kind must be a nonempty string")
        ScientificResult.__post_init__(self)


@dataclass(repr=False, frozen=True, slots=True)
class ArtifactAcceptance(ScientificResult):
    status: str
    identity_bound: bool
    replayable: bool
    resumable: bool
    recognized: bool = True
    proof: str | None = None
    unresolved: tuple[str, ...] = ()

    @property
    def verified(self) -> bool:
        """True when the consumer verified the artifact (``status == "verified"``)."""
        return self.status == "verified"

    @property
    def answer_available(self) -> bool:
        """Whether the recorded answer is kept on load (verified or sealed), as on ``Acceptance``."""
        return self.status in ("verified", "sealed")


@dataclass(repr=False, frozen=True, slots=True)
class ExportedArtifact(ScientificResult, Generic[ArtifactT_co]):
    """Container bytes and an independently retainable producer-side expectation."""

    data: bytes
    kind: str
    expectation: ArtifactExpectation | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise CausalTypeError("ExportedArtifact.data must be bytes")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise CausalTypeError("ExportedArtifact.kind must be a nonempty string")
        if not isinstance(self.metadata, Mapping):
            raise CausalTypeError("ExportedArtifact.metadata must be a mapping")
        if self.expectation is not None and not isinstance(self.expectation, ArtifactExpectation):
            raise CausalTypeError(
                "ExportedArtifact.expectation must be ArtifactExpectation or None"
            )
        object.__setattr__(self, "data", bytes(self.data))
        ScientificResult.__post_init__(self)

    def __bytes__(self) -> bytes:
        return self.data

    def export(self) -> ExportedArtifact[ArtifactT_co]:
        return self


@dataclass(frozen=True, slots=True)
class LoadContext:
    """Explicit sources for bundle consumption; never invokes a provider."""

    supplied_sources: Any = None
    actual_claims: Mapping[str, Any] | None = None
    execution: Execution | None = None
    composite_source: CompositeSession | None = None

    def __post_init__(self) -> None:
        from .execution import Execution

        if self.composite_source is not None:
            from .recalc_composite import CompositeSession

            if not isinstance(self.composite_source, CompositeSession):
                raise CausalTypeError(
                    "LoadContext.composite_source must be a CompositeSession or None"
                )
        if self.execution is not None and not isinstance(self.execution, Execution):
            raise CausalTypeError("LoadContext.execution must be an Execution or None")
        if self.actual_claims is not None and not isinstance(self.actual_claims, Mapping):
            raise CausalTypeError("LoadContext.actual_claims must be a mapping or None")
        object.__setattr__(self, "supplied_sources", freeze(self.supplied_sources))
        object.__setattr__(self, "actual_claims", freeze(self.actual_claims))


def _load_execution(context: LoadContext | None, execution: Execution | None) -> Execution | None:
    if context is not None and not isinstance(context, LoadContext):
        raise CausalTypeError("context must be a LoadContext or None")
    if execution is not None and not isinstance(execution, Execution):
        raise CausalTypeError("execution must be an Execution or None")
    retained = None if context is None else context.execution
    if execution is not None and retained is not None and execution is not retained:
        raise CausalTypeError(
            "supply one Execution policy, or the same object in context and execution"
        )
    return execution if execution is not None else retained


@dataclass(repr=False, frozen=True, slots=True)
class ArtifactInspection(ScientificResult):
    kind: str
    manifest: Mapping[str, Any]
    metadata: Mapping[str, Any]
    acceptance: ArtifactAcceptance
    _data: bytes = field(repr=False)

    def export(self) -> ExportedArtifact[Any]:
        return ExportedArtifact(self._data, self.kind, metadata={"manifest": self.manifest})


def container_bytes(value: Any) -> bytes:
    from .outcomes import require_success

    value = require_success(value)
    if isinstance(value, ExportedArtifact):
        return value.data
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise CausalTypeError(
        "an artifact is bytes or ExportedArtifact; read paths with Path.read_bytes()"
    )


@overload
def artifact_export(fn: Callable[P, ExportedArtifact[T]]) -> Callable[P, ExportedArtifact[T]]: ...


@overload
def artifact_export(fn: Callable[P, Any]) -> Callable[P, ExportedArtifact[Any]]: ...


def artifact_export(fn: Callable[P, Any]) -> Callable[P, ExportedArtifact[Any]]:
    """Canonical typed export adapter for family receipts.

    The public return is an ExportedArtifact. Raw bytes require explicit bytes(...).
    """

    @wraps(fn)
    def call(*args: P.args, **kwargs: P.kwargs) -> ExportedArtifact[Any]:
        receipt = fn(*args, **kwargs)
        return artifact_from_receipt(args[0] if args else None, receipt)

    call.__dict__["__signature__"] = inspect.signature(fn).replace(
        return_annotation=ExportedArtifact
    )
    call.__annotations__ = {**fn.__annotations__, "return": ExportedArtifact}
    return call


def artifact_consumer(
    *,
    proof: str,
    replayable: bool = False,
    resumable: bool = False,
    status: str = "verified",
    identity: tuple[str, ...] = ("expected_identity",),
    input: str = "artifact",
    unresolved: tuple[str, ...] = (),
    acceptance: Callable[[Any, bool], ArtifactAcceptance] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Record the proof of one explicitly audited semantic consumer.

    Apply only to a consumer that already verifies the stated contract on every
    successful return. A format discriminator or successful parsing is not proof.
    This adapter executes no extra parse and retains the exact native-backed value.
    """
    if not proof:
        raise ValueError("a semantic consumer must name its actual verification proof")

    def decorate(fn: Callable[P, T]) -> Callable[P, T]:
        signature = inspect.signature(fn)

        @wraps(fn)
        def call(*args: P.args, **kwargs: P.kwargs) -> T:
            arguments = signature.bind(*args, **kwargs).arguments
            result = fn(*args, **kwargs)
            bound = any(arguments.get(name) is not None for name in identity)
            receipt = (
                acceptance(result, bound)
                if acceptance is not None
                else ArtifactAcceptance(
                    status, bound, replayable, resumable, proof=proof, unresolved=unresolved
                )
            )
            raw = arguments.get(input)
            data = raw.encode("utf-8") if isinstance(raw, str) else container_bytes(raw)
            object.__setattr__(result, "acceptance", receipt)
            object.__setattr__(result, "_artifact_bytes", data)
            return result

        return call

    return decorate


# Native source-lineage envelopes with their own bounded framing. Their family
# consumers re-derive every number; recognition here only names the kind.
_SOURCE_ENVELOPE_MAGICS = (
    b"ANTE-SOURCE-EVIDENCE-1",
    b"ANTE-SOURCE-PROJECTION-1",
    b"ANTE-LAW-FUNCTIONAL-1",
    b"ANTE-SENSITIVITY-SOURCE-1",
)


def inspect_artifact(value: Any) -> ArtifactInspection:
    from . import _native

    data = container_bytes(value)
    if data.startswith(b"ANTPOLICYPROG"):
        return ArtifactInspection(
            "policy_program_v1",
            {"artifact_kind": "policy_program_v1"},
            {},
            ArtifactAcceptance("unavailable", False, False, False),
            data,
        )
    for magic in _SOURCE_ENVELOPE_MAGICS:
        if data.startswith(magic):
            kind = magic.decode("ascii")
            return ArtifactInspection(
                kind,
                {"artifact_kind": kind},
                {},
                ArtifactAcceptance("unavailable", False, False, False),
                data,
            )
    if data.startswith(b"ANTECEDENT-TRANSPORT-VIEW\x01"):
        from .transport._wrap import _parse_view_envelope

        payload, _, identification, snapshot, carried_artifacts = _parse_view_envelope(data)
        carried = [item for item in (identification, *carried_artifacts) if item is not None]
        readings = [inspect_artifact(item).to_dict() for item in carried]
        if snapshot is not None:
            # The native failure snapshot is schema-versioned JSON without a
            # format discriminator; its owning consumer rechecks it on load.
            kind = "z_transport_failure_snapshot"
            readings.append(
                ArtifactInspection(
                    kind,
                    {"artifact_kind": kind},
                    {},
                    ArtifactAcceptance("unavailable", False, False, False),
                    snapshot,
                ).to_dict()
            )
        return ArtifactInspection(
            "ANTECEDENT-TRANSPORT-VIEW",
            {"artifact_kind": "ANTECEDENT-TRANSPORT-VIEW"},
            {"envelope": payload, "carried": readings},
            ArtifactAcceptance("unavailable", False, False, False),
            data,
        )
    if data.startswith((b"CAUSAL\0\0", b"ANTECEDENT-", b"ANTBOUNDPRIOR1", b"ANTSTALEBASE1")) or (
        data and data[0] >> 5 == 5
    ):
        raw = json.loads(_native.inspect_artifact_container(data))
        manifest = raw["manifest"]
        kind = manifest["artifact_kind"]
        kind = kind.get("other", "unknown") if isinstance(kind, dict) else kind
        return ArtifactInspection(
            kind,
            manifest,
            raw["metadata"],
            ArtifactAcceptance("unavailable", False, False, False),
            data,
        )
    # The discretized model owner uses its own bounded binary format: an
    # exact length-prefixed tag followed by a layout version. This is only a
    # discriminator; the selected native consumer re-derives every number.
    discretized_tag = b"antecedent.model.discretized_scm_artifact"
    discretized_prefix = len(discretized_tag).to_bytes(8, "little") + discretized_tag
    if data.startswith(discretized_prefix):
        start = len(discretized_prefix)
        if len(data) < start + 4:
            raise CausalSerializationError("truncated discretized composition version")
        kind = discretized_tag.decode("ascii")
        return ArtifactInspection(
            kind,
            {
                "artifact_kind": kind,
                "format_version": int.from_bytes(data[start : start + 4], "little"),
            },
            {},
            ArtifactAcceptance("unavailable", False, False, False),
            data,
        )
    if data.lstrip().startswith(b"{"):
        try:
            body = json.loads(data)
        except (ValueError, UnicodeError) as error:
            raise CausalSerializationError("artifact is not valid JSON") from error
        kind = body.get("format")
        if not isinstance(kind, str):
            raise CausalSerializationError("JSON artifact has no format discriminator")
        if kind in ("matched_case_control_v1", "descriptive_comparison_v1"):
            module_name = "matched" if kind == "matched_case_control_v1" else "descriptive"
            producer = import_module(f"antecedent.{module_name}")
            payload = {key: item for key, item in body.items() if key != "digest"}
            try:
                actual = producer._digest(payload)
            except (ValueError, TypeError) as error:
                raise CausalSerializationError("JSON artifact has invalid digest inputs") from error
            if body.get("digest") != actual:
                raise CausalSerializationError("JSON artifact digest does not match its content")
        acceptance = ArtifactAcceptance("unavailable", False, False, False)
        if kind == "antecedent.repeated_measures_posterior":
            # The historical envelope omits the variance/correlation/assignment
            # declaration needed by the scientific replay owner. Recognition
            # alone cannot verify even a correctly framed retained posterior.
            acceptance = ArtifactAcceptance(
                "unavailable",
                False,
                False,
                False,
                unresolved=("declared_repeated_measures_procedure",),
            )
        return ArtifactInspection(
            kind,
            {"artifact_kind": kind},
            body,
            acceptance,
            data,
        )
    from . import artifacts

    decoded = artifacts.loads(data)  # raises for malformed bytes
    return ArtifactInspection(
        decoded.payload_kind,
        {},
        decoded.to_dict(),
        ArtifactAcceptance("unavailable", False, False, False),
        data,
    )


def exported_artifact(source: Any, *args: Any, **kwargs: Any) -> ExportedArtifact[Any]:
    """Export once, preserving an already typed artifact and family receipt."""
    return artifact_from_receipt(source, source.export(*args, **kwargs))


def artifact_from_receipt(source: Any, receipt: Any) -> ExportedArtifact[Any]:
    """Adapt native bytes or a family receipt at the actual export boundary."""
    if isinstance(receipt, ExportedArtifact):
        return receipt
    data = container_bytes(
        receipt if isinstance(receipt, (bytes, bytearray, memoryview)) else receipt.artifact
    )
    view = inspect_artifact(data)
    metadata = (
        {}
        if isinstance(receipt, (bytes, bytearray, memoryview))
        else {f.name: getattr(receipt, f.name) for f in fields(receipt)}
    )
    metadata.pop("artifact", None)
    identity = metadata.get("identity", metadata.get("premises_digest"))
    if identity is None:
        for name in (
            "identity",
            "premises_digest",
            "contract_identity",
            "claim_id",
            "model_identity",
            "plan_identity",
        ):
            identity = getattr(source, name, None)
            if identity is not None:
                break
    if view.kind == "decision_reliability_v1" and hasattr(source, "digests"):
        draws, candidates = source.digests()
        identity = {"draws_digest": draws, "candidate_set_digest": candidates}
    if view.kind == "sequential_evaluation_v1" and hasattr(source, "identities"):
        identity = {key: source.identities[key] for key in ("model", "policy", "evaluation")}
    if view.kind == "analysis_result":
        identity = {
            name: getattr(source, name, None)
            for name in ("program_id", "claim_id", "data_snapshot_id")
        }
    if view.kind == "causal_posterior" or view.kind in _PERIOD_ARTIFACT_KINDS:
        # These records have no standalone producer identity covering the whole
        # answer. Bind their actual encoding rather than a prior/source digest
        # that identifies only part of the scientific record.
        from hashlib import sha256

        identity = {"expected_encoding_sha256": sha256(data).hexdigest()}
    if callable(identity):
        identity = identity()
    return ExportedArtifact(
        bytes(data),
        view.kind,
        ArtifactExpectation(view.kind, identity) if identity is not None else None,
        metadata,
    )


# No consumer is selected by trying families until one accepts. The native manifest is the discriminator.
_CONSUMERS: dict[str, tuple[str, str]] = {
    "array_fit_replay_v1": ("nuisance_scores", "load_artifact"),
    "staleness_baseline_v1": ("drift", "consume_baseline"),
    "causal_posterior": ("_posterior_artifact", "consume_posterior"),
    "ANTECEDENT-PERIOD-REFRESH": ("transport.advanced", "consume_period_refresh_artifact"),
    "ANTECEDENT-PERIOD-REFRESH-SUMMARY": ("transport.advanced", "PeriodRefreshSummary.consume"),
    "policy_program_v1": ("intervention_programs", "consume_policy_program"),
    "cyclic_equilibrium": ("cyclic", "consume_artifact"),
    "cyclic_family": ("cyclic", "consume_family_artifact"),
    "cyclic_evidence": ("cyclic", "consume_evidence"),
    "scientific_refusal_v1": ("outcomes", "_consume_refusal"),
    "conditional_study_ranking_v1": ("recalc_composite", "ConditionalStudyRanking.consume"),
    "antecedent.model.discretized_scm_artifact": (
        "discretized_composition",
        "consume_discretized_composition",
    ),
    "antecedent.repeated_measures_posterior_bundle": (
        "adaptive_inference",
        "consume_repeated_measures_posterior_artifact",
    ),
    "matched_case_control_v1": ("matched", "replay_matched_case_control"),
    "descriptive_comparison_v1": ("descriptive", "replay_descriptive_comparison"),
    "temporal_counterfactual_v1": (
        "temporal_counterfactual",
        "consume_temporal_counterfactual_artifact",
    ),
    "temporal_counterfactual_general_v1": (
        "temporal_counterfactual",
        "consume_temporal_worlds_artifact",
    ),
    "effect_constancy_v1": ("temporal", "consume_effect_constancy_artifact"),
    "categorical_treatment_v1": ("categorical_treatment", "consume_categorical_effects"),
    "nonlinear_mediation_v1": ("nonlinear_mediation", "consume_mediation_artifact"),
    "adaptive_monitoring_contract_v1": (
        "adaptive_inference",
        "consume_adaptive_monitoring_contract_artifact",
    ),
    "adaptive_inference_result_v1": (
        "adaptive_inference",
        "consume_adaptive_inference_result_artifact",
    ),
    "callback_result_v1": ("provider_callbacks", "consume_callback_result"),
    "cohort_decision_source_v1": ("cohort_decision", "consume_cohort_decision_source"),
    "composition_verification_v1": (
        "composition_verification",
        "consume_composition_verification_artifact",
    ),
    "cross_world_execution_v1": ("cross_world_execution", "consume_cross_world_execution_artifact"),
    "decision_reliability_v1": ("decision_reliability", "consume_decision_reliability_artifact"),
    "evidence_ledger_v1": ("evidence_ledger", "consume_evidence_ledger_artifact"),
    "interface_contract_v1": ("interface_contract", "consume_interface_contract_artifact"),
    "joint_law_propagation_v1": ("joint_law_propagation", "consume_joint_law_propagation_artifact"),
    "measurement_model_v1": ("measurement_models", "consume_measurement_model_artifact"),
    "mechanism_composition_v1": ("mechanism_composition", "consume_mechanism_composition_artifact"),
    "observation_query_v1": ("observation_query", "consume_observation_query_artifact"),
    "predictive_assessment_v1": ("predictive_assessment", "consume_predictive_assessment_artifact"),
    "scientific_bridge_v1": ("scientific_bridge", "consume_scientific_bridge_artifact"),
    "selective_discovery_v1": ("selective_discovery", "consume_selective_discovery_artifact"),
    "sequential_evaluation_v1": ("sequential_evaluation", "consume_artifact"),
    "transported_counterfactual_v1": (
        "transported_counterfactual",
        "consume_transported_counterfactual_artifact",
    ),
    "composition_bundle_v1": ("composition_bundle", "consume_bundle"),
    "composition_bundle": ("composition_bundle", "consume_bundle"),
    "compact_runtime_export_v1": ("compact_export", "CompactExport.consume"),
    "joint_distribution_v1": ("joint_distribution", "JointDistributionArtifact.load"),
}


def acceptance_from_consumer(result: Any, *, identity_bound: bool = False) -> ArtifactAcceptance:
    """Read explicit semantic acceptance; parsing or a family name proves no replay."""
    receipt = getattr(result, "acceptance", None)
    if receipt is None:
        receipt = getattr(result, "receipt", None)
    if isinstance(receipt, ArtifactAcceptance):
        return ArtifactAcceptance(
            receipt.status,
            identity_bound or receipt.identity_bound,
            receipt.replayable,
            receipt.resumable,
            receipt.recognized,
            receipt.proof,
            receipt.unresolved,
        )

    def read(name: str, default: Any = None) -> Any:
        return (
            receipt.get(name, default)
            if isinstance(receipt, Mapping)
            else getattr(receipt, name, default)
        )

    verified = read("verified", read("accepts_as_verified_program"))
    replayed_program = read("accepts_as_verified_program") in (True, "true")
    replayable = read("replayable") is True or replayed_program
    verified = verified in (True, "true") or read("status") == "verified"
    sealed = read("sealed") is True or read("status") == "sealed"
    recognized = read("recognized")
    return ArtifactAcceptance(
        "verified" if verified or replayed_program else "sealed" if sealed else "unavailable",
        identity_bound,
        replayable,
        read("resumable") is True,
        recognized is True or recognized == "true",
    )


def load_scientific(
    value: Any,
    *,
    expected: Any = None,
    context: LoadContext | None = None,
    execution: Execution | None = None,
) -> Any:
    policy = _load_execution(context, execution)
    data = container_bytes(value)
    view = inspect_artifact(data)
    if isinstance(expected, ArtifactExpectation):
        if expected.kind != view.kind:
            raise CausalValueError("expected artifact kind differs from the container kind")
        expected = expected.identity
    selected = _CONSUMERS.get(view.kind)
    if selected is None:
        return view
    module = import_module(f"antecedent.{selected[0]}")
    consumer: Any = module
    for name in selected[1].split("."):
        consumer = getattr(consumer, name)
    parameters = inspect.signature(consumer).parameters
    kwargs: dict[str, Any] = {}
    if "execution" in parameters and policy is not None:
        kwargs["execution"] = policy
    if "expected_identity" in parameters:
        if expected is None and parameters["expected_identity"].default is inspect.Parameter.empty:
            raise CausalValueError(
                f"{view.kind} requires expected=: retain its producer identity separately"
            )
        kwargs["expected_identity"] = (
            expected
            if hasattr(expected, "to_json") or hasattr(expected, "_wire")
            else thaw(expected)
        )
    elif expected is not None:
        if not isinstance(expected, Mapping):
            raise CausalTypeError(f"{view.kind} requires named expected identity fields")
        if view.kind in _PERIOD_ARTIFACT_KINDS:
            from hashlib import sha256

            if set(expected) != {"expected_encoding_sha256"}:
                raise CausalValueError(
                    f"{view.kind} requires expected_encoding_sha256 as its identity expectation"
                )
            digest = expected["expected_encoding_sha256"]
            if not isinstance(digest, str):
                raise CausalTypeError("expected_encoding_sha256 must be a string")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise CausalValueError(
                    "expected_encoding_sha256 must be 64 lowercase hexadecimal characters"
                )
            if sha256(data).hexdigest() != digest:
                raise CausalValueError("artifact encoding differs from expected_encoding_sha256")
        else:
            unknown = set(expected) - set(parameters)
            if unknown:
                raise CausalValueError(f"unknown expectation fields: {sorted(unknown)}")
            kwargs.update(expected)
    if context is not None:
        if context.composite_source is not None:
            if view.kind != "conditional_study_ranking_v1":
                raise CausalValueError(f"{view.kind} does not consume composite source context")
            kwargs["session"] = context.composite_source
        if "supplied" not in parameters and (
            context.supplied_sources is not None or context.actual_claims
        ):
            raise CausalValueError(f"{view.kind} does not consume bundle source context")
        if "supplied" in parameters:
            kwargs["supplied"] = context.supplied_sources
            if "actual_claims" in parameters:
                kwargs["actual_claims"] = context.actual_claims
            elif context.actual_claims:
                raise CausalValueError(f"{view.kind} does not consume claim context")
    from .outcomes import Outcome, require_success

    consumed = (
        consumer(data, **kwargs) if policy is None else policy._call(consumer, data, **kwargs)
    )
    if isinstance(consumed, Outcome) and consumed.refusal is not None:
        return consumed
    result = require_success(consumed)
    acceptance = acceptance_from_consumer(result, identity_bound=expected is not None)
    # Python read models own their forwarding bytes, never a restored callable or live study.
    if isinstance(result, ScientificResult) or not isinstance(result, Mapping):
        object.__setattr__(result, "_artifact_bytes", data)
        object.__setattr__(result, "acceptance", acceptance)
        if view.kind in _PERIOD_ARTIFACT_KINDS:
            object.__setattr__(result, "_inspection_stage", "consumed")
    return result
