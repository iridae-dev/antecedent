"""Bytes-first persistence dispatch. Family consumers retain their scientific authority."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from functools import wraps
from importlib import import_module
from typing import Any, Generic, ParamSpec, TypeVar

from ._freeze import freeze, thaw
from ._scientific import ScientificResult
from .errors import CausalSerializationError, CausalTypeError, CausalValueError

T = TypeVar("T")
P = ParamSpec("P")


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


@dataclass(repr=False, frozen=True, slots=True)
class ExportedArtifact(ScientificResult, Generic[T]):
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

    def export(self) -> bytes:
        return self.data


@dataclass(frozen=True, slots=True)
class LoadContext:
    """Explicit sources for bundle consumption; never invokes a provider."""

    supplied_sources: Any = None
    actual_claims: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.actual_claims is not None and not isinstance(self.actual_claims, Mapping):
            raise CausalTypeError("LoadContext.actual_claims must be a mapping or None")
        object.__setattr__(self, "supplied_sources", freeze(self.supplied_sources))
        object.__setattr__(self, "actual_claims", freeze(self.actual_claims))


@dataclass(repr=False, frozen=True, slots=True)
class ArtifactInspection(ScientificResult):
    kind: str
    manifest: Mapping[str, Any]
    metadata: Mapping[str, Any]
    acceptance: ArtifactAcceptance
    _data: bytes = field(repr=False)

    def export(self) -> bytes:
        return self._data


def container_bytes(value: Any) -> bytes:
    if isinstance(value, ExportedArtifact):
        return value.data
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise CausalTypeError(
        "an artifact is bytes or ExportedArtifact; read paths with Path.read_bytes()"
    )


def bytes_export(fn: Callable[P, Any]) -> Callable[P, bytes]:
    """Retain an export receipt internally while presenting bytes as the public result."""

    @wraps(fn)
    def call(*args: P.args, **kwargs: P.kwargs) -> bytes:
        result = fn(*args, **kwargs)
        if isinstance(result, bytes):
            return result
        return container_bytes(result.artifact)

    setattr(call, "__signature__", inspect.signature(fn).replace(return_annotation=bytes))  # noqa: B010
    call.__annotations__ = {**fn.__annotations__, "return": bytes}
    return call


def inspect_artifact(value: Any) -> ArtifactInspection:
    from . import _native

    data = container_bytes(value)
    if data.startswith(b"ANTECEDENT-TRANSPORT-VIEW\x01"):
        from .transport._wrap import _parse_view_envelope

        payload, _, identification, snapshot, carried_artifacts = _parse_view_envelope(data)
        carried = [
            item for item in (identification, snapshot, *carried_artifacts) if item is not None
        ]
        readings = [inspect_artifact(item).to_dict() for item in carried]
        return ArtifactInspection(
            "ANTECEDENT-TRANSPORT-VIEW",
            {"artifact_kind": "ANTECEDENT-TRANSPORT-VIEW"},
            {"envelope": payload, "carried": readings},
            ArtifactAcceptance("unavailable", False, False, False),
            data,
        )
    if data.startswith((b"CAUSAL\0\0", b"ANTECEDENT-")) or (data and data[0] >> 5 == 5):
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
        return ArtifactInspection(
            kind,
            {"artifact_kind": kind},
            body,
            ArtifactAcceptance("unavailable", False, False, False),
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
    export = source.export
    original = getattr(export, "__wrapped__", None)
    receipt = original(source, *args, **kwargs) if original is not None else export(*args, **kwargs)
    data = receipt if isinstance(receipt, bytes) else receipt.artifact
    view = inspect_artifact(data)
    metadata = (
        {}
        if isinstance(receipt, bytes)
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
    "matched_case_control_v1": ("matched", "replay_matched_case_control"),
    "descriptive_comparison_v1": ("descriptive", "replay_descriptive_comparison"),
    "temporal_counterfactual_v1": (
        "temporal_counterfactual",
        "consume_temporal_counterfactual_artifact",
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


def load_scientific(value: Any, *, expected: Any = None, context: LoadContext | None = None) -> Any:
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
        unknown = set(expected) - set(parameters)
        if unknown:
            raise CausalValueError(f"unknown expectation fields: {sorted(unknown)}")
        kwargs.update(expected)
    if context is not None:
        if "supplied" not in parameters and (
            context.supplied_sources is not None or context.actual_claims
        ):
            raise CausalValueError(f"{view.kind} does not consume bundle source context")
        if "supplied" in parameters:
            kwargs.update(supplied=context.supplied_sources, actual_claims=context.actual_claims)
    result = consumer(data, **kwargs)
    replayable = view.kind != "callback_result_v1"
    if hasattr(result, "nodes") and view.kind.startswith("composition_bundle"):
        replayable = all(node.verified for node in result.nodes)
    acceptance = ArtifactAcceptance(
        "verified" if replayable else "sealed",
        expected is not None,
        replayable,
        view.kind in ("callback_result_v1", "selective_discovery_v1"),
    )
    # Python read models own their forwarding bytes, never a restored callable or live study.
    if isinstance(result, ScientificResult) or not isinstance(result, Mapping):
        object.__setattr__(result, "_artifact_bytes", data)
        object.__setattr__(result, "acceptance", acceptance)
    return result
