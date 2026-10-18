"""Small convenience surface over existing preparation and artifact validation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeVar, overload

from ._api import describe_refusal
from ._artifact_protocol import ExportedArtifact, LoadContext
from ._freeze import freeze, thaw
from .errors import CausalSerializationError, CausalValueError
from .estimation import PreparedAnalysis, _PreparedQuery, _PreparedResult
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, ClassPrior, Frequentist
from .missing import MissingPolicy
from .results import AnalysisResult, CausalResponseView
from .results._execution import Answer, CalibrationInfo, ResultAPI, answer_from_artifact
from .results._report import InspectionReport, as_inspection
from .results._slots import ReasoningSlots
from .transport._impl import (
    ClassicalTransportIdentification,
    ExactTransportDistribution,
    LearnedTrialEstimate,
    StatisticalTransportDistribution,
    TransportResponseGrid,
)
from .transport._restricted import RestrictedTransportExecution


def prepare(
    data: Any,
    *,
    query: _PreparedQuery,
    graph: Any = None,
    discovery: Any = None,
    inference: Frequentist | Bayesian | Any | None = None,
    identifier: str | Identifier | None = None,
    estimator: str | Estimator | Any | None = None,
    estimator_config: Mapping[str, Any] | None = None,
    refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = None,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int | None = None,
    latency: Latency | Literal["interactive", "standard", "report"] | None = None,
    class_prior: ClassPrior | None = None,
    max_completions: int | None = None,
    participant: str | None = None,
    missing: str | MissingPolicy | None = None,
    population_registry: Any | None = None,
    cancel: Any | None = None,
    on_progress: Any | None = None,
    on_stage: Any | None = None,
    validators: Sequence[Any] | Mapping[str, Any] | None = None,
    accept_discovered: bool = True,
    regimes: Sequence[int] | None = None,
    provider: Any | None = None,
    controls: Any | None = None,
) -> PreparedAnalysis[_PreparedResult]:
    """Prepare the same ordinary request as :func:`analyze`, stopping before estimation.

    Every argument :func:`analyze` accepts for a native query is accepted here,
    with the same meaning and the same refusals; ``analyze(...)`` is this call
    followed by ``.estimate()`` (plus ``return_posterior_artifact``). The one
    exception is ``extensibility.ProviderQuery``, which :func:`analyze` dispatches
    directly to its registered provider (it retains no prepared study to stop at).
    """
    return PreparedAnalysis.prepare(
        data,
        query=query,
        graph=graph,
        discovery=discovery,
        inference=inference,
        identifier=identifier,
        estimator=estimator,
        estimator_config=estimator_config,
        refute=refute,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        latency=latency,
        class_prior=class_prior,
        max_completions=max_completions,
        participant=participant,
        missing=missing,
        population_registry=population_registry,
        cancel=cancel,
        on_progress=on_progress,
        on_stage=on_stage,
        validators=validators,
        accept_discovered=accept_discovered,
        regimes=regimes,
        provider=provider,
        controls=controls,
    )


_SEALED_OPERATION_KEY = re.compile(r"^dependencies\.checked_[a-z0-9_]+_operation$")
# Retained data a sealed checked operation carries and the consumer cannot
# re-derive from the exported bytes: the checked AIPW lowering and its row
# binding, the linear-adjustment fit's sufficient statistics, fitted
# counterfactual mechanisms, and functional-effect replay draws and grids.
_SEALED_REPLAY_KEYS = frozenset(
    {
        "checked_aipw.row_binding",
        "dependencies.fitted_counterfactual_mechanisms",
        "dependencies.functional_effect_posterior_draws",
        "dependencies.functional_effect_response_posterior_draws",
        "dependencies.linear_fit_sufficient_statistics",
        "program.checked_aipw_lowering",
        "program.functional_effect_response_grid",
    }
)


def is_sealed_dependency(key: str) -> bool:
    """Whether an unresolved receipt key names a sealed checked operation or its replay data."""
    return key in _SEALED_REPLAY_KEYS or _SEALED_OPERATION_KEY.fullmatch(key) is not None


def sealed_dependencies_only(
    *, recognized: bool, contract_present: bool, unresolved: Iterable[str]
) -> bool:
    """The consumer recognized the artifact and its contract, and every unresolved
    reason is a sealed checked operation (or the replay data one carries) that an
    independent consumer cannot re-execute from bytes alone."""
    keys = tuple(unresolved)
    return bool(recognized and contract_present and keys and all(map(is_sealed_dependency, keys)))


def _split_unresolved(receipt: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(key for key in receipt.get("unresolved", "").split(",") if key)


@dataclass(frozen=True, slots=True)
class Acceptance:
    """What the consumer verified; does not certify assumptions or execution readiness.

    ``verified`` means the consumer replayed a verified program. ``sealed`` means it
    recognized the artifact, verified its contract and identities, and could not
    replay only because every unresolved reason is a sealed checked operation;
    ``unresolved`` names those reasons. Only a verified load is ``replayable``.
    """

    verified: bool
    recognized: bool
    details: Mapping[str, str]
    unresolved: tuple[str, ...] = ()
    identity_bound: bool = False
    resumable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", freeze(self.details))

    def to_dict(self) -> dict[str, Any]:
        return thaw(self)

    @classmethod
    def from_receipt(cls, receipt: Mapping[str, str]) -> Acceptance:
        return cls(
            verified=receipt.get("accepts_as_verified_program") == "true",
            recognized=receipt.get("recognized") == "true",
            details=dict(receipt),
            unresolved=_split_unresolved(receipt),
        )

    @property
    def replayable(self) -> bool:
        """True only when the consumer replayed a verified program."""
        return self.verified

    @property
    def sealed(self) -> bool:
        """Recognized, contract present, and unresolved only by sealed checked operations."""
        return not self.verified and sealed_dependencies_only(
            recognized=self.recognized,
            contract_present="program" in self.details,
            unresolved=self.unresolved,
        )

    @property
    def answer_available(self) -> bool:
        """Whether the recorded answer is kept on load (verified or sealed)."""
        return self.verified or self.sealed

    @property
    def status(self) -> Literal["verified", "sealed", "unavailable"]:
        if self.verified:
            return "verified"
        if self.sealed:
            return "sealed"
        return "unavailable"


@dataclass(frozen=True)
class LoadedResult(ResultAPI):
    """Semantic view of a portable execution: verified, sealed, or explicitly unavailable.

    A verified load replayed the program. A sealed load recognized the artifact,
    verified its contract and identities, kept the recorded answer, and names the
    checked operation it cannot replay in ``acceptance.unresolved``.
    """

    decoded_artifact: Any
    acceptance: Acceptance
    _bytes: bytes = field(repr=False)

    @property
    def program_id(self) -> str | None:
        """Compiled program identity. Distinct from :attr:`claim_id`."""
        return self.inspect().program_id

    @property
    def claim_id(self) -> str | None:
        """Execution claim identity. Distinct from :attr:`program_id`."""
        return self.inspect().claim_id

    @property
    def calibration(self) -> CalibrationInfo:
        return CalibrationInfo.from_contract(self.decoded_artifact.contract)

    @property
    def answer(self) -> Answer:
        """The recorded answer for a verified or sealed load; otherwise unavailable."""
        if not self.acceptance.answer_available:
            return Answer("unavailable", detail="semantic_acceptance_unavailable")
        return answer_from_artifact(self.decoded_artifact.contract, self.decoded_artifact.payload)

    def inspect(self) -> InspectionReport:
        if not self.acceptance.answer_available and not (
            self.acceptance.recognized and isinstance(self.decoded_artifact.contract, Mapping)
        ):
            report = as_inspection(
                replace(
                    ReasoningSlots.from_contract({}),
                    answer=self.answer,
                    calibration=self.calibration,
                )
            )
        else:
            report = as_inspection(
                ReasoningSlots.from_result_section(
                    self.decoded_artifact.contract,
                    self.decoded_artifact.payload,
                    answer=self.answer,
                    calibration=self.calibration,
                )
            )
        return report.model_copy(
            update={
                "stage": "consumed",
                "family": "LoadedResult",
                "operations": ("inspect", "export", "artifact"),
            }
        )

    def export(self, *, artifact_id: str | None = None) -> bytes:
        if artifact_id is not None and artifact_id != self.decoded_artifact.artifact_id:
            raise CausalSerializationError(
                "A loaded execution is immutable; its artifact ID cannot be rewritten by forwarding."
            )
        return self._bytes

    def __repr__(self) -> str:
        return f"<LoadedResult {self.answer.kind} acceptance={self.acceptance.status}>"


@describe_refusal
def _load_analysis(
    data: bytes,
) -> (
    LoadedResult
    | AnalysisResult
    | CausalResponseView
    | ExactTransportDistribution
    | StatisticalTransportDistribution
    | TransportResponseGrid
    | ClassicalTransportIdentification
    | LearnedTrialEstimate
    | RestrictedTransportExecution
):
    """Load a contracted result through the Rust semantic consumer.

    Missing/unknown contracts remain explicitly unavailable and can still be
    forwarded. Loading never creates a live study or upgrades an old artifact.
    """
    from . import artifacts

    prepared: PreparedAnalysis[Any]
    encoded = bytes(data)
    if encoded.startswith(b"ANTECEDENT-TRANSPORT-VIEW\x01"):
        from .transport._wrap import decode_transport_view

        return decode_transport_view(encoded)
    if encoded.startswith(b"ANTECEDENT-LEARNED-TRIAL\x01"):
        from . import _native
        from .transport._impl import _learned_trial

        native = _native.consume_learned_trial(encoded)
        return _learned_trial(native, native.last_result())
    if encoded.startswith(b"ANTECEDENT-EXACT-TRANSPORT\x01"):
        from .transport._impl import _exact_distribution, consume_exact

        prepared = consume_exact(encoded)
        return _exact_distribution(prepared._native, prepared._native.last_result())
    if encoded.startswith(b"ANTECEDENT-STATISTICAL-TRANSPORT\x01"):
        from .transport._impl import _statistical_distribution, consume_statistical

        prepared = consume_statistical(encoded)
        return _statistical_distribution(prepared._native, prepared._native.last_result())
    if encoded.startswith(b"ANTECEDENT-TRANSPORT-GRID\x01"):
        from .transport._impl import _response_grid, consume_response_grid

        prepared = consume_response_grid(encoded)
        return _response_grid(prepared._native, prepared._native.last_result())
    if encoded.startswith(b"ANTECEDENT-Z-TRANSPORT\x01"):
        from .transport._restricted import consume_restricted_artifacts

        return consume_restricted_artifacts([encoded])
    if encoded.startswith(b"ANTECEDENT-TRANSPORT-CERTIFICATE\x01"):
        from .transport.advanced import consume_identification

        return consume_identification(encoded)
    receipt = artifacts.accept(encoded)
    artifact = artifacts.loads(encoded)
    return LoadedResult(artifact, Acceptance.from_receipt(receipt), encoded)


LoadedT = TypeVar("LoadedT")


@overload
def load(
    data: ExportedArtifact[LoadedT], *, expected: Any = None, context: LoadContext | None = None
) -> LoadedT: ...


@overload
def load(
    data: bytes | bytearray | memoryview,
    *,
    expected: Any = None,
    context: LoadContext | None = None,
) -> Any: ...


@describe_refusal
def load(data: Any, *, expected: Any = None, context: Any = None) -> Any:
    """Consume a scientific artifact using its native family consumer.

    Expectations must be supplied separately; envelope metadata never binds an identity implicitly.
    Loading never executes callbacks or restores a live study.
    """
    from ._artifact_protocol import (
        ArtifactAcceptance,
        ArtifactExpectation,
        container_bytes,
        inspect_artifact,
        load_scientific,
    )

    encoded = container_bytes(data)
    legacy_transport = encoded.startswith(
        tuple(
            b"ANTECEDENT-" + marker + b"\x01"
            for marker in (
                b"TRANSPORT-VIEW",
                b"LEARNED-TRIAL",
                b"EXACT-TRANSPORT",
                b"STATISTICAL-TRANSPORT",
                b"TRANSPORT-GRID",
                b"Z-TRANSPORT",
                b"TRANSPORT-CERTIFICATE",
            )
        )
    )
    if not legacy_transport and (
        encoded.startswith((b"CAUSAL\0\0", b"ANTECEDENT-"))
        or encoded.lstrip().startswith(b"{")
        or (encoded and encoded[0] >> 5 == 5)
    ):
        header = inspect_artifact(encoded)
        if header.kind not in (
            "analysis_result",
            "causal_payload",
            "provider_result_v1",
            "external_callback_v1",
        ):
            return load_scientific(encoded, expected=expected, context=context)
    result = _load_analysis(encoded)
    if isinstance(expected, ArtifactExpectation):
        actual_kind = (
            encoded.split(b"\x01", 1)[0].decode("ascii")
            if legacy_transport
            else result.decoded_artifact.payload_kind
            if isinstance(result, LoadedResult)
            else "analysis_result"
        )
        if expected.kind != actual_kind:
            raise CausalSerializationError(
                "expected artifact kind differs from the loaded execution"
            )
        expected = expected.identity
    if context is not None:
        raise CausalValueError("this artifact does not consume bundle source context")
    if expected is not None:
        bindings = expected if isinstance(expected, Mapping) else {"claim_id": expected}
        allowed = {
            "program_id",
            "claim_id",
            "data_snapshot_id",
            "target_id",
            "identification_id",
            "identification_product_id",
            "inference_binding_id",
            "observation_id",
            "execution_id",
        }
        if not bindings or not any(identity is not None for identity in bindings.values()):
            raise CausalValueError(
                "expected must contain at least one independent identity binding"
            )
        if set(bindings) - allowed:
            raise CausalValueError(
                f"unknown expected identity fields: {sorted(set(bindings) - allowed)}"
            )
        inspect_result = getattr(result, "inspect", None)
        if not callable(inspect_result):
            raise CausalValueError(
                "this legacy artifact exposes no independently bindable execution inspection; use its owning specialized consumer"
            )
        report = inspect_result()
        for name, identity in bindings.items():
            if identity is not None and getattr(report, name, None) != identity:
                raise CausalSerializationError(f"expected {name} differs from the loaded execution")
        if isinstance(result, LoadedResult):
            object.__setattr__(
                result, "acceptance", replace(result.acceptance, identity_bound=True)
            )
    if legacy_transport:
        object.__setattr__(
            result, "acceptance", ArtifactAcceptance("verified", expected is not None, True, False)
        )
    return result
