"""Per-proposal receipts linking identification repair and design ranking (X6).

A failed contract owes evidence (:mod:`antecedent.repair`); a candidate study is
valued for the decision it would inform (:mod:`antecedent.design_ranking`). A
:class:`ProposalBundle` binds the two for every candidate the ranking values, by
identities and digests and never by copy::

    repaired = repair.repair(contract, candidates)
    ranked = design_ranking.rank_designs(decision, ranked_candidates, ...)
    bundle = ProposalBundle.build(repaired.export(), ranked.export(), contract=contract)
    bundle.verify(repaired.export(), ranked.export())
    verdict = bundle.on_arrival(candidate, ArrivedEvidence.law("clinic", [...], ...))

Each :class:`ProposalReceipt` retains the frozen base failure (family, contract,
unresolved obligation ids, the repair premises and data digests), the
hypothetical delta and the derivation the repair family verified on it, the
candidate's declared cost and size, the evidence-lineage and provider snapshots,
the ranking entry (signal request fingerprint, signal identity, EVSI, net value,
rank) and the decision it is a value of. The bundle's identity covers the two
artifact digests, the decision contract and every receipt, in the canonical order
of candidate semantic id, so it does not depend on the order proposals were
supplied in.

The delta and derivation a receipt retains are hypothetical: a receipt is always
:attr:`ProposalReceipt.evidence_state` ``"hypothetical"`` and never available
evidence. :meth:`ProposalBundle.on_arrival` never treats them as evidence either: a
delivery that is itself not available evidence is refused, and otherwise the repair
family's own identification is re-run on the arrived evidence alone, answering
:class:`Verified`, :class:`StillInsufficient` or :class:`Invalidated` (a population
or regime other than the premises the derivation was verified under).

``verify`` recomputes every receipt from the two retained artifacts and refuses a
candidate that appears in one artifact but not the other, a changed delta, a
swapped signal, a changed cost, lineage, value or decision identity, a receipt whose
identity does not follow from its content and a bundle that does not cover exactly
the ranked candidates. It checks the artifacts' seals; by default each artifact is
also consumed (independently replayed) first.

Refusals are :class:`ProposalRefusal`, a
:class:`~antecedent.errors.CausalUnsupportedError` with its registered
``reason_code`` and a stable ``proposal_receipt.*`` ``detail``.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from . import _native, design_ranking, repair
from ._native import proposal_bundle_build as _build
from ._native import proposal_ranked_candidates as _ranked_candidates
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError

EvidenceKindName = Literal["available", "manipulable", "proposed"]


class ProposalRefusal(CausalUnsupportedError):
    """A refused bundle, verification or arrival.

    ``detail`` is the stable ``proposal_receipt.*`` detail (for example
    ``proposal_receipt.delta_mismatch``, ``proposal_receipt.candidate_not_in_ranking``
    or ``proposal_receipt.hypothetical_not_evidence``); ``message`` is the
    explanation. ``reason_code`` is registered.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        detail = str(refusal["detail"])
        message = str(refusal.get("message", ""))
        super().__init__(f"{detail}: {message}", reason_code=refusal["code"])
        self.stage: str = str(refusal.get("stage", "proposal"))
        self.detail: str = detail
        self.message: str = message


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        raise ProposalRefusal(json.loads(refusal))


def _bits(value: int) -> float:
    return float(struct.unpack("<d", struct.pack("<Q", value))[0])


# --------------------------------------------------------------------------- receipts


@dataclass(frozen=True, slots=True)
class BaseFailure:
    """The frozen failed contract a proposal repairs."""

    family: str
    contract: str
    obligation_ids: tuple[str, ...]
    premises_digest: str
    data_digest: str


@dataclass(frozen=True, slots=True)
class Hypothetical:
    """The hypothetical delta and the derivation checked on it.

    Nothing here is evidence: it is what the repair family concluded *if* the
    study delivered exactly its declared evidence.
    """

    classification: str
    delta_digest: str
    delta_regimes: int
    derivation_digest: str | None
    derivation_verified: bool
    addressed: tuple[str, ...]
    unmet: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeclaredCost:
    """The candidate's declared cost and size."""

    units: int
    unit_label: str
    sample_budget: int
    sample_size: int


@dataclass(frozen=True, slots=True)
class SourceLineage:
    """Evidence-lineage and provider snapshots of the valuation, and the digest of
    the ranking's source distributions."""

    snapshots: tuple[str, ...]
    source_digest: str


@dataclass(frozen=True, slots=True)
class Valuation:
    """The ranking entry the proposal was valued by (``basis`` is ``evsi`` or
    ``net_value``); ``rank`` 0 is best."""

    basis: str
    signal_request_fingerprint: str
    signal_identity: str
    evsi: float
    net_value: float | None
    rank: int


@dataclass(frozen=True, slots=True)
class DecisionBinding:
    """Which decision the value is a value of."""

    contract_identity: str
    utility_unit: str
    action_ids_digest: str
    ranking_identity: str


@dataclass(frozen=True, slots=True)
class ProposalReceipt:
    """One proposal's receipt: failure, hypothetical derivation, cost, lineage,
    value and decision, bound by identities and digests."""

    candidate_id: str
    base_failure: BaseFailure
    hypothetical: Hypothetical
    cost: DeclaredCost
    lineage: SourceLineage
    valuation: Valuation
    decision: DecisionBinding
    repair_report_digest: str
    ranking_digest: str
    identity: str

    @property
    def evidence_state(self) -> Literal["hypothetical"]:
        """Always ``hypothetical``: a receipt never holds available evidence."""
        return "hypothetical"

    @property
    def is_available_evidence(self) -> bool:
        """Always ``False``: arrival is answered by :meth:`ProposalBundle.on_arrival`."""
        return False

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> ProposalReceipt:
        base, hyp, cost = wire["base_failure"], wire["hypothetical"], wire["cost"]
        lineage, value, decision = wire["lineage"], wire["valuation"], wire["decision"]
        net = value["net_value_bits"]
        return cls(
            candidate_id=wire["candidate_id"],
            base_failure=BaseFailure(
                base["family"],
                base["contract"],
                tuple(base["obligation_ids"]),
                base["premises_digest"],
                base["data_digest"],
            ),
            hypothetical=Hypothetical(
                hyp["classification"],
                hyp["delta_digest"],
                int(hyp["delta_regimes"]),
                hyp["derivation_digest"],
                bool(hyp["derivation_verified"]),
                tuple(hyp["addressed"]),
                tuple(hyp["unmet"]),
            ),
            cost=DeclaredCost(
                int(cost["units"]),
                cost["unit_label"],
                int(cost["sample_budget"]),
                int(cost["sample_size"]),
            ),
            lineage=SourceLineage(tuple(lineage["snapshots"]), lineage["source_digest"]),
            valuation=Valuation(
                value["basis"],
                value["signal_request_fingerprint"],
                value["signal_identity"],
                _bits(int(value["evsi_bits"])),
                None if net is None else _bits(int(net)),
                int(value["rank"]),
            ),
            decision=DecisionBinding(
                decision["contract_identity"],
                decision["utility_unit"],
                decision["action_ids_digest"],
                decision["ranking_identity"],
            ),
            repair_report_digest=wire["repair_report_digest"],
            ranking_digest=wire["ranking_digest"],
            identity=wire["identity"],
        )


# --------------------------------------------------------------------------- arrival


@dataclass(frozen=True, slots=True)
class ArrivedRegime:
    """One regime that arrived: the population it was collected in, the hard
    interventions (and their levels) and the variables measured.

    ``evidence_kind`` is ``available`` for results that exist. A ``proposed`` or
    ``manipulable`` regime is a hypothesis: delivering one is refused
    (``proposal_receipt.hypothetical_not_evidence``).
    """

    population: str
    measured: Sequence[str]
    interventions: Sequence[str] = ()
    levels: Mapping[str, float] = field(default_factory=dict)
    conditioned_on: Sequence[str] = ()
    joint: bool = True
    evidence_kind: EvidenceKindName = "available"

    def _wire(self) -> dict[str, Any]:
        return {
            "population": self.population,
            "interventions": list(self.interventions),
            "levels": {str(k): float(v) for k, v in self.levels.items()},
            "conditioned_on": list(self.conditioned_on),
            "measured": list(self.measured),
            "joint": bool(self.joint),
            "evidence_kind": self.evidence_kind,
        }


@dataclass(frozen=True, slots=True)
class ArrivedEvidence:
    """What was delivered for a proposed study.

    ``sample_size`` must equal the candidate's planned size and ``snapshot_id``
    names the delivered snapshot. Build with :meth:`law` (an observed law of a
    back-door contract's variables) or :meth:`regimes` (actual regimes added to a
    transport contract's catalog).
    """

    snapshot_id: str
    sample_size: int
    evidence: Mapping[str, Any]

    @classmethod
    def law(
        cls,
        population: str,
        measured: Sequence[str],
        *,
        snapshot_id: str,
        sample_size: int,
        joint: bool = True,
    ) -> ArrivedEvidence:
        """An observed law over ``measured`` in ``population`` (``joint`` or as
        separate marginals); answers a back-door contract only."""
        return cls(
            snapshot_id,
            int(sample_size),
            {
                "kind": "observed_law",
                "population": population,
                "measured": list(measured),
                "joint": bool(joint),
            },
        )

    @classmethod
    def regimes(
        cls, regimes: Sequence[ArrivedRegime], *, snapshot_id: str, sample_size: int
    ) -> ArrivedEvidence:
        """Regimes added to the contract's catalog."""
        items = tuple(regimes)
        if any(not isinstance(r, ArrivedRegime) for r in items):
            raise CausalTypeError("regimes must be ArrivedRegime values")
        return cls(
            snapshot_id,
            int(sample_size),
            {"kind": "regimes", "regimes": [r._wire() for r in items]},
        )

    def _wire(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "sample_size": self.sample_size,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class Verified:
    """The family's identification now succeeds on the arrived evidence alone and its
    derivation was re-verified. ``exact`` says whether the arrived laws equal the
    proposed ones exactly."""

    checker: str
    steps: tuple[str, ...]
    exact: bool
    snapshot_id: str


@dataclass(frozen=True, slots=True)
class StillInsufficient:
    """The arrived evidence leaves the contract unmet or uncertified; ``reasons`` are
    the checker's own."""

    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Invalidated:
    """The arrived evidence is not of the premises the derivation was verified under.

    ``premise`` is ``population`` (a regime of a population the proposed evidence was
    not in: ``proposed_populations`` and ``population`` say which) or ``regime`` (the
    right ``population`` under another intervention set, ``arrived_interventions``).
    """

    premise: Literal["population", "regime"]
    population: str
    proposed_populations: tuple[str, ...] = ()
    arrived_interventions: tuple[str, ...] = ()


ArrivalVerdict = Verified | StillInsufficient | Invalidated


def _verdict(wire: Mapping[str, Any]) -> ArrivalVerdict:
    kind = wire["kind"]
    if kind == "verified":
        return Verified(
            wire["checker"], tuple(wire["steps"]), bool(wire["exact"]), wire["snapshot_id"]
        )
    if kind == "still_insufficient":
        return StillInsufficient(tuple(wire["reasons"]))
    changed = wire["changed"]
    if changed["kind"] == "population":
        return Invalidated(
            "population", changed["arrived"], proposed_populations=tuple(changed["proposed"])
        )
    return Invalidated(
        "regime",
        changed["population"],
        arrived_interventions=tuple(changed["arrived_interventions"]),
    )


# --------------------------------------------------------------------------- bundle


def _bytes(name: str, value: object) -> bytes:
    if not isinstance(value, bytes):
        raise CausalTypeError(f"{name} must be bytes")
    return value


def _replay(repair_artifact: bytes, ranking_artifact: bytes) -> None:
    repair.consume(repair_artifact)
    design_ranking.consume(ranking_artifact)


class ProposalBundle:
    """Every proposal of one repair report and one ranking, in canonical order."""

    __slots__ = (
        "_contract",
        "_native_value",
        "_proposals",
        "_repair_artifact",
        "_ranking_artifact",
    )

    def __init__(self, native: _native.ProposalBundle, contract: Any = None) -> None:
        if not isinstance(native, _native.ProposalBundle):
            raise CausalTypeError("use ProposalBundle.build or ProposalBundle.from_dict")
        self._native_value = native
        self._contract = contract
        self._repair_artifact: bytes | None = None
        self._ranking_artifact: bytes | None = None
        wire = json.loads(native.to_json())
        self._proposals = tuple(ProposalReceipt._from_wire(p) for p in wire["proposals"])

    @classmethod
    def build(
        cls,
        repair_artifact: bytes,
        ranking_artifact: bytes,
        *,
        contract: repair.TransportContract | repair.BackdoorContract | None = None,
        consume: bool = True,
    ) -> ProposalBundle:
        """The bundle of every candidate the ranking values, from the two exported
        artifacts (``repair.RepairResult.export()`` and
        ``design_ranking.DesignRankingResult.export()``).

        With ``consume`` (the default) each artifact is first replayed independently
        by its own consumer. A ranked candidate the repair report does not declare
        and evaluate, a candidate the two artifacts cost or size differently and an
        artifact whose seal does not match its content refuse
        (:class:`ProposalRefusal`). ``contract`` is the failed contract the repair
        was made for; it is needed only by :meth:`on_arrival`.
        """
        repair_artifact = _bytes("repair_artifact", repair_artifact)
        ranking_artifact = _bytes("ranking_artifact", ranking_artifact)
        if consume:
            _replay(repair_artifact, ranking_artifact)
        native, refusal = _build(repair_artifact, ranking_artifact)
        _raise(refusal)
        assert native is not None
        value = cls(native, contract)
        value._repair_artifact = repair_artifact
        value._ranking_artifact = ranking_artifact
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProposalBundle:
        """Read a bundle back from :meth:`to_dict`. Stored identities are kept as
        given, so an edited field is caught by :meth:`verify`, not hidden by it."""
        return cls(_native.ProposalBundle.from_json(json.dumps(dict(data))))

    def to_dict(self) -> dict[str, Any]:
        """The bundle and every receipt as plain data (bit patterns for floats)."""
        return dict(json.loads(self._native_value.to_json()))

    @property
    def identity(self) -> str:
        """Digest of the artifact digests, the decision contract and every receipt."""
        return str(self._native_value.identity)

    @property
    def proposals(self) -> tuple[ProposalReceipt, ...]:
        """Receipts in strictly increasing candidate semantic id."""
        return self._proposals

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(p.candidate_id for p in self._proposals)

    def proposal(self, candidate_id: str) -> ProposalReceipt:
        """The receipt of one candidate."""
        for receipt in self._proposals:
            if receipt.candidate_id == candidate_id:
                return receipt
        raise CausalValueError(f"no proposal {candidate_id!r} in this bundle")

    def verify(
        self, repair_artifact: bytes, ranking_artifact: bytes, *, consume: bool = True
    ) -> None:
        """Cross-check the bundle against the two retained artifacts.

        Returns ``None`` when every receipt is exactly what the artifacts imply and
        the proposals are exactly the ranked candidates, in canonical order; raises
        :class:`ProposalRefusal` naming the first disagreement otherwise (for
        example ``proposal_receipt.cost_mismatch``, ``delta_mismatch``,
        ``signal_mismatch``, ``value_mismatch``, ``candidate_set_mismatch`` or
        ``bundle_identity_mismatch``).
        """
        repair_artifact = _bytes("repair_artifact", repair_artifact)
        ranking_artifact = _bytes("ranking_artifact", ranking_artifact)
        if consume:
            _replay(repair_artifact, ranking_artifact)
        _raise(self._native_value.verify(repair_artifact, ranking_artifact))

    def on_arrival(
        self,
        candidate: repair.StudyCandidate,
        arrived_evidence: ArrivedEvidence,
        *,
        contract: repair.TransportContract | repair.BackdoorContract | None = None,
    ) -> ArrivalVerdict:
        """Re-run identification on evidence that actually arrived for ``candidate``.

        The receipt must be the one the failed contract and the candidate produce
        (same candidate, base failure, obligations and hypothetical delta). The
        arrival is refused unless it is available evidence of the planned sample size
        that fits the family. A regime of another population, or under another
        intervention set, is :class:`Invalidated`; otherwise the family's own
        identification runs on the arrived evidence alone and decides between
        :class:`Verified` and :class:`StillInsufficient`. The hypothetical delta and
        derivation are never consulted as evidence.
        """
        if not isinstance(candidate, repair.StudyCandidate):
            raise CausalTypeError("candidate must be a repair.StudyCandidate")
        if not isinstance(arrived_evidence, ArrivedEvidence):
            raise CausalTypeError("arrived_evidence must be an ArrivedEvidence")
        failed = contract if contract is not None else self._contract
        if not isinstance(failed, (repair.TransportContract, repair.BackdoorContract)):
            raise ProposalRefusal(
                {
                    "code": "invalid_argument",
                    "stage": "proposal",
                    "detail": "proposal_receipt.contract_required",
                    "message": "on_arrival needs the failed repair contract the bundle was made for",
                }
            )
        result, refusal = self._native_value.on_arrival(
            failed._stage,
            json.dumps([candidate._wire()]),
            json.dumps(arrived_evidence._wire()),
        )
        _raise(refusal)
        assert result is not None
        return _verdict(json.loads(result))

    def estimate_arrival(
        self,
        candidate_id: str,
        study: Any,
        *,
        seed: int = 3,
        repair_artifact: bytes | None = None,
        ranking_artifact: bytes | None = None,
    ) -> Any:
        """Evaluate actual finite arrived counts through the checked transport formula.

        Structural ``on_arrival`` remains available separately. This route requires
        original producer artifacts, supplied explicitly when the bundle was loaded
        from a dictionary. Unrepresented sampling designs refuse; point uncertainty
        remains unmeasured.
        """
        from .proposal_arrival import _estimate

        self.proposal(candidate_id)
        original_repair = repair_artifact if repair_artifact is not None else self._repair_artifact
        original_ranking = (
            ranking_artifact if ranking_artifact is not None else self._ranking_artifact
        )
        if original_repair is None or original_ranking is None:
            raise CausalValueError(
                "proposal_arrival.original_artifacts_required", reason_code="invalid_argument"
            )
        return _estimate(
            original_repair, original_ranking, candidate_id, study, self.identity, seed
        )

    def __repr__(self) -> str:
        return f"<ProposalBundle {len(self._proposals)} proposals identity={self.identity[:12]}>"


def ranked_candidate_ids(ranking_artifact: bytes) -> tuple[str, ...]:
    """The semantic ids of the candidates a ranking artifact values, sorted."""
    return tuple(_ranked_candidates(_bytes("ranking_artifact", ranking_artifact)))


__all__ = [
    "ArrivalVerdict",
    "ArrivedEvidence",
    "ArrivedRegime",
    "BaseFailure",
    "DeclaredCost",
    "DecisionBinding",
    "Hypothetical",
    "Invalidated",
    "ProposalBundle",
    "ProposalReceipt",
    "ProposalRefusal",
    "SourceLineage",
    "StillInsufficient",
    "Valuation",
    "Verified",
    "ranked_candidate_ids",
]
