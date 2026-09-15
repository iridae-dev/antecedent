"""Small convenience surface over existing preparation and artifact validation."""

from __future__ import annotations

import struct
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ._api import describe_refusal
from .errors import CausalSerializationError
from .estimation import PreparedAnalysis, _PreparedQuery
from .ids import Estimator, Identifier, Latency, Refute
from .inference import Bayesian, ClassPrior, Frequentist
from .results._execution import Answer, CalibrationInfo, ResultAPI
from .results._slots import ReasoningSlots, SlotView


@describe_refusal
def prepare(
    data: Any,
    *,
    query: _PreparedQuery,
    graph: Any = None,
    discovery: Any = None,
    inference: Frequentist | Bayesian | None = None,
    identifier: str | Identifier | None = None,
    estimator: str | Estimator | None = None,
    estimator_config: Mapping[str, Any] | None = None,
    refute: bool | Refute | Literal["full", "placebo", "none", "cheap"] | None = None,
    seed: int = 1,
    bootstrap: int | None = None,
    threads: int = 1,
    latency: Latency | Literal["interactive", "standard", "report"] | None = None,
    class_prior: ClassPrior | None = None,
    max_completions: int | None = None,
) -> PreparedAnalysis:
    """Prepare the same ordinary request as analyze, stopping before estimation.

    Unlike the historical PreparedAnalysis.prepare defaults, omitted settings
    follow analyze: standard bootstrap/refutation for scalar effects, and the
    response family's existing uncertainty/validation defaults.
    """
    from .estimation import _resolve_latency_budget

    kind = getattr(query, "kind", "")
    response = kind in ("response_curve", "intervention_response")
    suite: Any = (
        ("none" if response or kind == "counterfactual" else "placebo")
        if refute is None
        else refute
    )
    if refute is None and not response and kind != "counterfactual":
        _, suite = _resolve_latency_budget(latency, bootstrap, True)
        if suite is True:
            suite = "placebo"
    if bootstrap is None and (
        kind == "counterfactual"
        or (
            response
            and (not getattr(query, "is_temporal", False) or isinstance(inference, Bayesian))
        )
    ):
        bootstrap = 0
    return PreparedAnalysis.prepare(
        data,
        query=query,
        graph=graph,
        discovery=discovery,
        inference=inference,
        identifier=identifier,
        estimator=estimator,
        estimator_config=estimator_config,
        refute=suite,
        seed=seed,
        bootstrap=bootstrap,
        threads=threads,
        latency=latency,
        class_prior=class_prior,
        max_completions=max_completions,
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
    def calibration(self) -> CalibrationInfo:
        return CalibrationInfo()

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
        section = self.artifact.contract
        reasoning = section["reasoning"]

        def slot(name: str) -> SlotView:
            raw = reasoning[name]
            value = raw.get("value")
            return SlotView(
                value is not None,
                raw.get("unavailable"),
                (value.get("status", "available") if value is not None else "unavailable"),
                value if value is not None else {},
            )

        identities = section["identities"]
        body = self.artifact.payload
        uncertainty = slot("uncertainty")
        details = dict(uncertainty.payload)
        if body.get("standard_error") is not None:
            details["standard_error"] = body["standard_error"]
        response = body.get("response") or {}
        if response.get("uncertainty") is not None:
            details["response"] = response["uncertainty"]
            if response["uncertainty"] != "none" and not uncertainty.available:
                uncertainty = replace(
                    uncertainty, available=True, reason=None, summary="response_specific"
                )
        structural = body.get("structural_response") or {}
        if structural.get("identified_set_interval") is not None:
            details["identified_set_interval"] = structural["identified_set_interval"]
        support = slot("support")
        if response.get("support") is not None:
            support = replace(
                support, payload={**support.payload, "execution": response["support"]}
            )
        assumptions = slot("assumptions")
        assumptions = replace(
            assumptions, payload={**assumptions.payload, "records": body.get("assumptions", [])}
        )
        return ReasoningSlots(
            identification=slot("identification"),
            support=support,
            uncertainty=replace(uncertainty, payload=details),
            assumptions=assumptions,
            claim_id=bytes(identities["program"]).hex(),
            data_version=bytes(identities["data_snapshot"]).hex(),
            answer=self.answer,
            calibration=self.calibration,
            diagnostics=tuple(
                f"{d.get('code', '')}: {d.get('message', '')}" for d in body.get("diagnostics", [])
            ),
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
