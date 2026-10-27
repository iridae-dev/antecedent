"""One shared workflow for a retained native response and two attested mean callbacks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from . import _native
from ._artifact_protocol import (
    ArtifactAcceptance,
    ArtifactExpectation,
    ExportedArtifact,
    artifact_consumer,
    artifact_export,
    container_bytes,
)
from ._recalc_bounds import _columns
from ._refusal import raise_refusal
from ._scientific import ScientificResult
from .composition import SupportedDecision, _supported_from_wire
from .decision import Contract
from .design import StructuralCandidate, StructuralEntry
from .errors import CausalTypeError, CausalValueError, StructuredRefusal
from .outcomes import Answer, outcome_operation, require_success
from .program_claims import ProgramBinding
from .recalc import RecalcPlan, RecalcReceipt, SessionStanding, _seed, session_standing
from .recalc_external import CallbackProvider, ExternalCallbackRefusal, ExternalCallbackRequest
from .recalc_static import StaticResponseRequest, StaticResponseSession, _response_spec


@dataclass(frozen=True, slots=True)
class CompositeRequest:
    native: StaticResponseRequest
    native_program: ProgramBinding
    externals: tuple[ExternalCallbackRequest, ExternalCallbackRequest]
    decision_contract: Contract
    support_policy: Literal["compare_supported", "require_all"] = "require_all"
    input_order: tuple[str, str, str] = ("native", "external.0", "external.1")


@dataclass(frozen=True, slots=True)
class CompositeResult:
    plan: RecalcPlan
    receipt: RecalcReceipt
    decision: SupportedDecision


def _native_payload(
    request: StaticResponseRequest,
    *,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> dict[str, object]:
    names, columns = _columns(request.data, memory_limit_bytes=memory_limit_bytes, cancel=cancel)
    return {
        "names": names,
        "columns": columns,
        "graph": request.graph,
        "specification": _response_spec(request),
    }


def _payload(
    request: CompositeRequest,
    *,
    memory_limit_bytes: int | None = None,
    cancel: _native.CancellationToken | None = None,
) -> dict[str, object]:
    if len(request.externals) != 2:
        raise CausalValueError(
            "exactly two callback requests are required", reason_code="invalid_argument"
        )
    if not isinstance(request.decision_contract, Contract):
        raise CausalTypeError("decision_contract must be decision.Contract")
    branches = []
    for external in request.externals:
        names, columns = _columns(
            external.data, memory_limit_bytes=memory_limit_bytes, cancel=cancel
        )
        branches.append({"names": names, "columns": columns, "specification": external._wire()})
    return {
        **_native_payload(request.native, memory_limit_bytes=memory_limit_bytes, cancel=cancel),
        "program": json.dumps(request.native_program._wire()),
        "externals": branches,
        "contract": json.dumps(request.decision_contract._wire()),
        "policy": request.support_policy,
        "input_order": request.input_order,
    }


class ConditionalStudyRankingRefusal(StructuredRefusal):
    """A bounded policy, actual-source or independent projection refusal."""

    def __init__(self, wire: Mapping[str, Any]) -> None:
        super().__init__(wire, text=wire["message"])


@dataclass(frozen=True, slots=True)
class ConditionalStudyPolicy:
    """Explicit action→study tables. Supplied sufficiency verdicts remain caller owned."""

    policy_id: str
    branches: Mapping[str, tuple[StructuralCandidate, ...]]

    def _wire(self):
        if not isinstance(self.policy_id, str) or len(self.policy_id.encode()) > 256:
            raise CausalValueError("bounded policy_id required", reason_code="invalid_argument")
        if not isinstance(self.branches, Mapping) or not self.branches:
            raise CausalValueError("an action table is required", reason_code="invalid_argument")
        tables = []
        for action, candidates in self.branches.items():
            if not isinstance(action, str) or len(action.encode()) > 256 or not candidates:
                raise CausalValueError(
                    "a named action with at least one candidate is required",
                    reason_code="invalid_argument",
                )
            rows = []
            for candidate in candidates:
                if not isinstance(candidate, StructuralCandidate):
                    raise CausalTypeError("candidates must be StructuralCandidate")
                if not isinstance(candidate.id, str) or len(candidate.id.encode()) > 256:
                    raise CausalValueError(
                        "bounded candidate identity required", reason_code="invalid_argument"
                    )
                rows.append(
                    {
                        "semantic_id": candidate.id,
                        "verified_sufficient": candidate.verified_sufficient,
                        "cost_units": candidate.cost_units,
                        "sample_budget": candidate.sample_budget,
                    }
                )
            tables.append({"action": action, "candidates": rows})
        return {"policy_id": self.policy_id, "branches": tables}


def _conditional_acceptance(result: Any, bound: bool) -> ArtifactAcceptance:
    return ArtifactAcceptance(
        "verified" if result.source_verified else "sealed",
        bound,
        result.source_verified,
        False,
        proof=(
            "ConditionalStudyRanking::consume: compare actual issued combined sources and rerank"
            if result.source_verified
            else "ConditionalStudyRanking::inspect_retained: validate identity, historical receipt and retained structural ranking"
        ),
        unresolved=() if result.source_verified else ("actual_executed_composite_sources",),
    )


def _conditional_bytes(data: Any) -> bytes:
    raw = container_bytes(data)
    # Keep the owning native consumer's typed artifact/limit refusals for
    # malformed historical bytes and malformed additive envelopes alike.
    if len(raw) > 8 * 1024 * 1024 + 1024:
        return raw
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeError):
        return raw
    if not isinstance(body, dict) or body.get("format") != "conditional_study_ranking_v1":
        return raw
    if body.get("version") != 1 or not isinstance(body.get("payload"), str):
        return raw
    # The native owner applies the original 4 MiB limit before parsing this
    # historical payload, including when the additive wrapper is well formed.
    return body["payload"].encode("utf-8")


@dataclass(frozen=True, slots=True)
class ConditionalStudyRanking(ScientificResult):
    """Source-backed conditional point ranking, with no EVSI or joint-state inference."""

    identity: str
    selected_action: str
    entries: tuple[StructuralEntry, ...]
    source: Mapping[str, object]
    policy: Mapping[str, object]
    _bytes: bytes
    source_verified: bool = False

    @property
    def answer(self) -> Answer:
        return Answer(
            "structured",
            detail="conditional_structural_ranking",
            structured={
                "selected_action": self.selected_action,
                "study_ids": tuple(entry.id for entry in self.entries),
            },
        )

    @artifact_export
    def export(self) -> ExportedArtifact[ConditionalStudyRanking]:
        kind = "conditional_study_ranking_v1"
        data = json.dumps(
            {"format": kind, "version": 1, "payload": self._bytes.decode("utf-8")},
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return ExportedArtifact(data, kind, ArtifactExpectation(kind, self.identity))

    @classmethod
    @outcome_operation
    @artifact_consumer(
        proof="ConditionalStudyRanking owning consumer",
        input="data",
        acceptance=_conditional_acceptance,
    )
    def consume(
        cls,
        data: bytes,
        session: CompositeSession | None = None,
        *,
        expected_identity: str | None = None,
    ) -> ConditionalStudyRanking:
        """Match a historical projection against explicitly replayed actual combined state."""
        data = _conditional_bytes(data)
        if session is None:
            return _conditional_result(
                _native.conditional_study_ranking_inspect(data, expected_identity)
            )
        session = require_success(session)
        if not isinstance(session, CompositeSession):
            raise CausalTypeError("consume requires an actually executed CompositeSession")
        return _conditional_result(
            session._handle.consume_conditional_studies(data, expected_identity)
        )


def _conditional_result(result):
    report, data, error, source_verified = result
    raise_refusal(error, ConditionalStudyRankingRefusal)
    assert report is not None and data is not None
    wire = json.loads(report)
    return ConditionalStudyRanking(
        wire["identity"],
        wire["source"]["selected_action"],
        tuple(
            StructuralEntry(
                row["semantic_id"],
                row["rank"],
                row["verified_sufficient"],
                row["cost_units"],
                row["sample_budget"],
            )
            for row in wire["entries"]
        ),
        wire["source"],
        wire["policy"],
        bytes(data),
        bool(source_verified),
    )


class CompositeSession:
    """Hold actual issued native state; plan and publish branches as one transaction.

    Callback policies remain supplier declarations. Mean responses supply affine
    expectations and point rankings, without joint draws or calibrated intervals.
    """

    __slots__ = ("_handle", "_origin")

    def __init__(self, native: StaticResponseSession, request: StaticResponseRequest):
        if not isinstance(native, StaticResponseSession):
            raise CausalTypeError("native must be an executed StaticResponseSession")
        self._origin = native
        self._handle = _native.CompositeSessionHandle(native._handle, _native_payload(request))

    def plan(
        self,
        request: CompositeRequest,
        *,
        memory_limit_bytes: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> RecalcPlan:
        wire, error = self._handle.plan(
            _payload(request, memory_limit_bytes=memory_limit_bytes, cancel=cancel),
            memory_limit_bytes=memory_limit_bytes,
            cancel=cancel,
        )
        raise_refusal(error, ExternalCallbackRefusal)
        assert wire is not None
        return RecalcPlan.from_wire(json.loads(wire))

    @outcome_operation
    def execute(
        self,
        request: CompositeRequest,
        *,
        providers: Mapping[int, CallbackProvider],
        seed: int = 1,
        threads: int | None = None,
        memory_limit_bytes: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> CompositeResult:
        if len(providers) > 2 or any(
            not isinstance(value, CallbackProvider) for value in providers.values()
        ):
            raise CausalTypeError("providers must map at most two branches to CallbackProvider")
        wire, receipt, error = self._handle.execute(
            _payload(request, memory_limit_bytes=memory_limit_bytes, cancel=cancel),
            {branch: provider._handle for branch, provider in providers.items()},
            seed=_seed(seed),
            threads=threads,
            memory_limit_bytes=memory_limit_bytes,
            cancel=cancel,
        )
        raise_refusal(error, ExternalCallbackRefusal)
        assert wire is not None and receipt is not None
        body = json.loads(wire)
        return CompositeResult(
            RecalcPlan.from_wire(body["plan"]),
            RecalcReceipt._from_wire(body["receipt"], body["plan"], receipt, loaded=False),
            _supported_from_wire(body["decision"]),
        )

    @outcome_operation
    def rank_studies(self, policy: ConditionalStudyPolicy) -> ConditionalStudyRanking:
        """Run original structural ranking for the actual unique selected terminal action.

        The complete explicit rule and actual native/provider source identities are bound.
        This operation executes no callbacks or model fits and supplies no EVSI.
        """
        if not isinstance(policy, ConditionalStudyPolicy):
            raise CausalTypeError("policy must be ConditionalStudyPolicy")
        return _conditional_result(
            self._handle.rank_conditional_studies(json.dumps(policy._wire()))
        )

    @property
    def standing(self) -> SessionStanding:
        """The standing inputs this session family's stage identities declare."""
        return session_standing("composite")

    @property
    def native_response(self):
        from .estimation import _wrap_prepared_response
        from .query import ResponseCurve

        raw = self._handle.native_response()
        if raw is None:
            return None
        basis = json.loads(raw.program_basis_json())
        view = _wrap_prepared_response(
            raw, ResponseCurve(basis["treatment"], basis["outcome"], grid=basis["grid"])
        )
        return view.model_copy(
            update={"data_snapshot_id": basis["snapshot_id"], "program_id": basis["program_id"]}
        )

    def export_native(self) -> bytes:
        """Original retained native result artifact with its actual producing context."""
        return bytes(self._handle.export_native())

    def export_callback(self, branch: int) -> bytes:
        """Original attested callback output envelope, containing no executable closure."""
        return bytes(self._handle.export_callback(branch))


__all__ = [
    "CompositeRequest",
    "CompositeResult",
    "CompositeSession",
    "ConditionalStudyPolicy",
    "ConditionalStudyRanking",
    "ConditionalStudyRankingRefusal",
]
