"""One shared workflow for a retained native response and two attested mean callbacks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from . import _native
from ._recalc_bounds import _columns
from .composition import SupportedDecision, _supported_from_wire
from .decision import Contract
from .design_ranking import StructuralCandidate, StructuralEntry
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .program_claims import ProgramBinding
from .recalc import RecalcPlan, RecalcReceipt, _seed
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


def _native_payload(request: StaticResponseRequest) -> dict[str, object]:
    names, columns = _columns(request.data)
    return {
        "names": names,
        "columns": columns,
        "graph": request.graph,
        "specification": _response_spec(request),
    }


def _payload(request: CompositeRequest) -> dict[str, object]:
    if len(request.externals) != 2:
        raise CausalValueError(
            "exactly two callback requests are required", reason_code="invalid_argument"
        )
    if not isinstance(request.decision_contract, Contract):
        raise CausalTypeError("decision_contract must be decision.Contract")
    branches = []
    for external in request.externals:
        names, columns = _columns(external.data)
        branches.append({"names": names, "columns": columns, "specification": external._wire()})
    return {
        **_native_payload(request.native),
        "program": json.dumps(request.native_program._wire()),
        "externals": branches,
        "contract": json.dumps(request.decision_contract._wire()),
        "policy": request.support_policy,
        "input_order": request.input_order,
    }


def _raise(error: str | None) -> None:
    if error is not None:
        raise ExternalCallbackRefusal(json.loads(error))


class ConditionalStudyRankingRefusal(CausalUnsupportedError):
    """A bounded policy, actual-source or independent projection refusal."""

    def __init__(self, wire):
        super().__init__(wire["message"], reason_code=wire["code"])
        self.detail = wire["detail"]
        self.stage = wire["stage"]


@dataclass(frozen=True, slots=True)
class ConditionalStudyPolicy:
    """Explicit action→study tables. Supplied sufficiency verdicts remain caller owned."""

    policy_id: str
    branches: Mapping[str, tuple[StructuralCandidate, ...]]

    def _wire(self):
        if not isinstance(self.policy_id, str) or len(self.policy_id.encode()) > 256:
            raise CausalValueError("bounded policy_id required", reason_code="invalid_argument")
        if not isinstance(self.branches, Mapping) or not 1 <= len(self.branches) <= 64:
            raise CausalValueError(
                "one to 64 action tables required", reason_code="invalid_argument"
            )
        tables = []
        total = 0
        for action, candidates in self.branches.items():
            if (
                not isinstance(action, str)
                or len(action.encode()) > 256
                or not 1 <= len(candidates) <= 128
            ):
                raise CausalValueError(
                    "bounded action/candidate table required", reason_code="invalid_argument"
                )
            total += len(candidates)
            if total > 512:
                raise CausalValueError(
                    "at most 512 candidates required", reason_code="invalid_argument"
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


@dataclass(frozen=True, slots=True)
class ConditionalStudyRanking:
    """Source-backed conditional point ranking, with no EVSI or joint-state inference."""

    identity: str
    selected_action: str
    entries: tuple[StructuralEntry, ...]
    source: Mapping[str, object]
    policy: Mapping[str, object]
    _bytes: bytes

    def export(self) -> bytes:
        return self._bytes

    @classmethod
    def consume(
        cls, data: bytes, session: CompositeSession, *, expected_identity: str | None = None
    ):
        """Match a historical projection against explicitly replayed actual combined state."""
        if not isinstance(session, CompositeSession):
            raise CausalTypeError("consume requires an actually executed CompositeSession")
        return _conditional_result(
            session._handle.consume_conditional_studies(data, expected_identity)
        )


def _conditional_result(result):
    report, data, error = result
    if error is not None:
        raise ConditionalStudyRankingRefusal(json.loads(error))
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

    def plan(self, request: CompositeRequest) -> RecalcPlan:
        wire, error = self._handle.plan(_payload(request))
        _raise(error)
        assert wire is not None
        return RecalcPlan.from_wire(json.loads(wire))

    def execute(
        self,
        request: CompositeRequest,
        *,
        providers: Mapping[int, CallbackProvider],
        seed: int = 1,
        threads: int | None = None,
        cancel: _native.CancellationToken | None = None,
    ) -> CompositeResult:
        if len(providers) > 2 or any(
            not isinstance(value, CallbackProvider) for value in providers.values()
        ):
            raise CausalTypeError("providers must map at most two branches to CallbackProvider")
        wire, receipt, error = self._handle.execute(
            _payload(request),
            {branch: provider._handle for branch, provider in providers.items()},
            seed=_seed(seed),
            threads=threads,
            cancel=cancel,
        )
        _raise(error)
        assert wire is not None and receipt is not None
        body = json.loads(wire)
        return CompositeResult(
            RecalcPlan.from_wire(body["plan"]),
            RecalcReceipt._from_wire(body["receipt"], body["plan"], receipt, loaded=False),
            _supported_from_wire(body["decision"]),
        )

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
