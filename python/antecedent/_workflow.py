"""Small convenience surface over existing preparation and artifact validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ._api import describe_refusal
from .errors import CausalSerializationError
from .estimation import PreparedAnalysis, _PreparedQuery
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, ClassPrior, Frequentist
from .results._execution import Answer, CalibrationInfo, ResultAPI
from .results._slots import ReasoningSlots


@describe_refusal
def prepare(
    data: Any,
    *,
    query: _PreparedQuery,
    graph: Any = None,
    discovery: Any = None,
    inference: Frequentist | Bayesian | None = None,
    identifier: str | Identifier | None = None,
    estimator: str | Estimator | Any | None = None,
    estimator_config: Mapping[str, Any] | None = None,
    refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = None,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int = 1,
    latency: Latency | Literal["interactive", "standard", "report"] | None = None,
    class_prior: ClassPrior | None = None,
    max_completions: int | None = None,
    target_population: Any | None = None,
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
) -> PreparedAnalysis:
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
        target_population=target_population,
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
    )


@dataclass(frozen=True, slots=True)
class Acceptance:
    """What the consumer verified; does not certify assumptions or execution readiness."""

    verified: bool
    recognized: bool
    details: dict[str, str]


@dataclass(frozen=True)
class LoadedResult(ResultAPI):
    """Verified or explicitly unavailable semantic view of a portable execution."""

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
        if not self.acceptance.verified:
            return Answer("unavailable", detail="semantic_acceptance_unavailable")
        claim = self.artifact.contract.get("claim") or {}
        kind = claim.get("kind", "unavailable")
        bits = claim.get("value_bits")
        value = (
            struct.unpack("<d", struct.pack("<Q", bits))[0]
            if bits is not None and kind == "point"
            else None
        )
        structural = self.artifact.payload.get("structural_response") or {}
        envelope = structural.get("identified_set") or {}
        bounds = None
        if (
            kind == "bounds"
            and len(envelope.get("lower", [])) == len(envelope.get("upper", [])) == 1
        ):
            bounds = (envelope["lower"][0], envelope["upper"][0])
        return Answer(kind, value=value, bounds=bounds)

    def inspect(self) -> ReasoningSlots:
        if not self.acceptance.verified:
            return replace(
                ReasoningSlots.from_contract({}), answer=self.answer, calibration=self.calibration
            )
        return ReasoningSlots.from_result_section(
            self.artifact.contract,
            self.artifact.payload,
            answer=self.answer,
            calibration=self.calibration,
        )

    def export(self, *, artifact_id: str | None = None) -> bytes:
        if artifact_id is not None and artifact_id != self.artifact.artifact_id:
            raise CausalSerializationError(
                "A loaded execution is immutable; its artifact ID cannot be rewritten by forwarding."
            )
        return self._bytes

    def __repr__(self) -> str:
        return f"<LoadedResult {self.answer.kind} acceptance={'verified' if self.acceptance.verified else 'unavailable'}>"


@describe_refusal
def load(data: bytes) -> LoadedResult:
    """Load a contracted result through the Rust semantic consumer.

    Missing/unknown contracts remain explicitly unavailable and can still be
    forwarded. Loading never creates a live study or upgrades an old artifact.
    """
    from . import artifacts

    encoded = bytes(data)
    receipt = artifacts.accept(encoded)
    artifact = artifacts.loads(encoded)
    return LoadedResult(
        artifact,
        Acceptance(
            verified=receipt["accepts_as_verified_program"] == "true",
            recognized=receipt["recognized"] == "true",
            details=receipt,
        ),
        encoded,
    )
