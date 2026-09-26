"""Small convenience surface over existing preparation and artifact validation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ._api import describe_refusal
from .errors import CausalSerializationError
from .estimation import PreparedAnalysis, _PreparedQuery, _PreparedResult
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, ClassPrior, Frequentist
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
    population_registry: Any | None = None,
    cancel: Any | None = None,
    on_progress: Any | None = None,
    on_stage: Any | None = None,
    validators: Sequence[Any] | Mapping[str, Any] | None = None,
    accept_discovered: bool = True,
    regimes: Sequence[int] | None = None,
    running_variable: str | None = None,
    cutoff: float | None = None,
    bandwidth: float | None = None,
    provider: Any | None = None,
    controls: Any | None = None,
) -> PreparedAnalysis[_PreparedResult]:
    """Prepare the same ordinary request as :func:`analyze`, stopping before estimation.

    Every argument :func:`analyze` accepts is accepted here, with the same
    meaning and the same refusals; ``analyze(...)`` is this call followed by
    ``.estimate()`` (plus ``return_posterior_artifact``).
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
        population_registry=population_registry,
        cancel=cancel,
        on_progress=on_progress,
        on_stage=on_stage,
        validators=validators,
        accept_discovered=accept_discovered,
        regimes=regimes,
        running_variable=running_variable,
        cutoff=cutoff,
        bandwidth=bandwidth,
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
    details: dict[str, str]
    unresolved: tuple[str, ...] = ()

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

    artifact: Any
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
        return CalibrationInfo.from_contract(self.artifact.contract)

    @property
    def answer(self) -> Answer:
        """The recorded answer for a verified or sealed load; otherwise unavailable."""
        if not self.acceptance.answer_available:
            return Answer("unavailable", detail="semantic_acceptance_unavailable")
        return answer_from_artifact(self.artifact.contract, self.artifact.payload)

    def inspect(self) -> InspectionReport:
        if not self.acceptance.answer_available and not (
            self.acceptance.recognized and isinstance(self.artifact.contract, Mapping)
        ):
            return as_inspection(
                replace(
                    ReasoningSlots.from_contract({}),
                    answer=self.answer,
                    calibration=self.calibration,
                )
            )
        return as_inspection(
            ReasoningSlots.from_result_section(
                self.artifact.contract,
                self.artifact.payload,
                answer=self.answer,
                calibration=self.calibration,
            )
        )

    def export(self, *, artifact_id: str | None = None) -> bytes:
        if artifact_id is not None and artifact_id != self.artifact.artifact_id:
            raise CausalSerializationError(
                "A loaded execution is immutable; its artifact ID cannot be rewritten by forwarding."
            )
        return self._bytes

    def __repr__(self) -> str:
        return f"<LoadedResult {self.answer.kind} acceptance={self.acceptance.status}>"


@describe_refusal
def load(
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
