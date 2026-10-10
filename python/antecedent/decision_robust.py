"""Admissibility rules and robustness for decisions under structural uncertainty.

:mod:`antecedent.decision` declares a decision problem and evaluates it on one
set of aligned joint draws. This module adds what a decision needs when its
claims are not one law: admissibility rules (support, declared exclusions and
the uncertainty representation the claims must carry), claim adapters (point
claims, graph-dependent claims, weighted graph-posterior atoms, finite scenarios
and identified sets) and a robustness verdict::

    rules = decision_robust.AdmissibilityRules(
        default_weakest_support="supported", uncertainty="structural_envelope"
    )
    contract = decision_robust.admissible_contract(base_contract, rules)
    result = decision_robust.robust(
        contract,
        [
            decision_robust.Claim.evaluated("graph-1", draws_1),
            decision_robust.Claim.evaluated("graph-2", draws_2),
        ],
        kind="graph_dependent",
    )
    print(result.verdict.kind, result.explain())

Rust owns identity, validation, evaluation, artifacts and refusals; this module
builds declarations and raises each refusal as
:class:`~antecedent.decision.DecisionRefusal` (a
:class:`~antecedent.errors.CausalUnsupportedError`) with its registered
``reason_code``, ``detail`` and ``offending`` intact.

Rules only remove actions in the structure that violates them; none becomes a
penalty. Unidentified and unevaluated mass is reported and never renormalized
away. A structural envelope is not a probability law and completion counts are
not probabilities. A decision whose value came from an external callback keeps
the attested value, exact request fingerprint and trust limit in its artifact
and is never labelled natively verified.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from ._native import admissible_contract_normalize as _normalize
from ._native import evaluate_identified_set_decision as _evaluate_sets
from ._native import evaluate_robust_decision as _evaluate_robust
from ._native import export_admissible_contract as _export_contract
from ._native import export_robust_decision as _export_robust
from ._native import identified_utility_interval as _interval
from ._native import load_admissible_contract as _load_contract
from ._native import replay_robust_decision as _replay_robust
from .decision import Contract, DecisionRefusal
from .errors import CausalValueError
from .external import BoundExternalClaim, LineageLink
from .joint_distribution import JointDistributionArtifact

SupportLabel = Literal[
    "supported", "weak_overlap", "extrapolative", "outside_empirical_support", "missing_evidence"
]
UncertaintyRequirement = Literal["none", "point_only", "structural_envelope", "credible"]
ClaimKind = Literal["point", "graph_dependent", "weighted_graph_posterior", "finite_scenarios"]
ExternalTrust = Literal["externally_attested", "verified_extension"]
#: What a :class:`RobustVerdict` can claim about the robustness of a choice.
RobustVerdictKind = Literal[
    "structurally_robust",
    "support_robust",
    "support_dependent",
    "graph_dependent_choice",
    "unsupported_extrapolation",
    "insufficient_claims",
    "no_admissible_action",
    "worst_case_choice",
    "bayes_choice",
    "report_only",
]
#: What an :class:`IdentifiedVerdict` can claim over identified sets.
IdentifiedVerdictKind = Literal[
    "necessarily_best",
    "no_necessarily_best",
    "worst_case_choice",
    "minimax_regret_choice",
    "tied",
    "report_only",
    "unsupported_extrapolation",
    "insufficient_claims",
    "no_admissible_action",
]
RESULT_LINK_ID = "robust_decision"
_CLAIM_KINDS = ("point", "graph_dependent", "weighted_graph_posterior", "finite_scenarios")


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        raise DecisionRefusal(json.loads(refusal))


def _check_label(value: str, allowed: Sequence[str], what: str) -> None:
    if value not in allowed:
        raise CausalValueError(f"{what} is one of {', '.join(allowed)}; got {value!r}")


_SUPPORT_LABELS = (
    "supported",
    "weak_overlap",
    "extrapolative",
    "outside_empirical_support",
    "missing_evidence",
)
_UNCERTAINTY_LABELS = ("none", "point_only", "structural_envelope", "credible")


# --------------------------------------------------------------------------- rules


@dataclass(frozen=True, slots=True)
class SupportRule:
    """The weakest empirical support one input of one action may have."""

    action_id: str
    input: int
    weakest_allowed: SupportLabel

    def __post_init__(self) -> None:
        _check_label(self.weakest_allowed, _SUPPORT_LABELS, "weakest_allowed")

    def _wire(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "input": self.input,
            "weakest_allowed": self.weakest_allowed,
        }


@dataclass(frozen=True, slots=True)
class DeclaredExclusion:
    """An action removed by declaration (a legal, ethical or logistical rule)."""

    action_id: str
    reason: str

    def _wire(self) -> dict[str, Any]:
        return {"action_id": self.action_id, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class AdmissibilityRules:
    """Admissibility rules beyond hard constraints.

    A rule can only remove an action; it never adds a penalty or a bonus to a
    utility. ``default_weakest_support`` governs every input without its own
    :class:`SupportRule`. ``uncertainty`` names the representation the claims
    must carry (a point is not an envelope and an envelope is not a probability
    law): a claim of another kind makes the decision ``insufficient_claims``.
    """

    default_weakest_support: SupportLabel | None = None
    support_rules: tuple[SupportRule, ...] = ()
    declared_exclusions: tuple[DeclaredExclusion, ...] = ()
    uncertainty: UncertaintyRequirement = "none"

    def __post_init__(self) -> None:
        object.__setattr__(self, "support_rules", tuple(self.support_rules))
        object.__setattr__(self, "declared_exclusions", tuple(self.declared_exclusions))
        if self.default_weakest_support is not None:
            _check_label(self.default_weakest_support, _SUPPORT_LABELS, "default_weakest_support")
        _check_label(self.uncertainty, _UNCERTAINTY_LABELS, "uncertainty")

    def _wire(self) -> dict[str, Any]:
        return {
            "default_weakest_support": self.default_weakest_support,
            "support_rules": [rule._wire() for rule in self.support_rules],
            "declared_exclusions": [item._wire() for item in self.declared_exclusions],
            "uncertainty": self.uncertainty,
        }

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> AdmissibilityRules:
        return cls(
            default_weakest_support=wire["default_weakest_support"],
            support_rules=tuple(
                SupportRule(r["action_id"], r["input"], r["weakest_allowed"])
                for r in wire["support_rules"]
            ),
            declared_exclusions=tuple(
                DeclaredExclusion(e["action_id"], e["reason"]) for e in wire["declared_exclusions"]
            ),
            uncertainty=wire["uncertainty"],
        )


@dataclass(frozen=True, slots=True)
class AdmissibleContract:
    """A decision contract with admissibility rules, under one canonical identity."""

    contract: Contract
    rules: AdmissibilityRules = field(default_factory=AdmissibilityRules)

    def _declaration(self) -> str:
        return json.dumps(
            {"version": 1, "contract": self.contract._wire(), "rules": self.rules._wire()}
        )

    def _normalized(self) -> dict[str, Any]:
        normalized, refusal = _normalize(self._declaration())
        _raise(refusal)
        assert normalized is not None
        return dict(json.loads(normalized))

    @property
    def identity(self) -> str:
        """Covers the base contract and every rule; unchanged by reordering."""
        return str(self._normalized()["identity"])

    @property
    def base_identity(self) -> str:
        """Identity of the base :class:`~antecedent.decision.Contract` alone."""
        return str(self._normalized()["contract"]["identity"])

    def with_admissibility(self, rules: AdmissibilityRules) -> AdmissibleContract:
        """The same decision under different rules."""
        return AdmissibleContract(self.contract, rules)

    def export(self, *, artifact_id: str = "admissible-decision-contract") -> bytes:
        """The contract and its rules as a portable artifact.

        A contract without rules is better exported with
        :meth:`antecedent.decision.Contract.export`, whose bytes are unchanged.
        """
        return bytes(_export_contract(self._declaration(), artifact_id))

    @classmethod
    def load(cls, data: bytes, *, expected_identity: str) -> AdmissibleContract:
        """Load only under the identity the consumer retained independently."""
        wire = json.loads(_load_contract(data, expected_identity))
        return cls(
            Contract._from_wire(wire["contract"]), AdmissibilityRules._from_wire(wire["rules"])
        )


def admissible_contract(
    contract: Contract, rules: AdmissibilityRules | None = None
) -> AdmissibleContract:
    """Attach admissibility rules (none by default) to a decision contract."""
    return AdmissibleContract(contract, rules or AdmissibilityRules())


# --------------------------------------------------------------------------- claims


@dataclass(frozen=True, slots=True)
class Support:
    """Empirical support a structure's evidence has, per input.

    ``per_input`` maps ``(action_id, input_index)`` to a label; every other input
    has ``overall``. An unassessed structure is ``missing_evidence``, so a rule
    that needs support treats it as missing evidence, not as supported.
    """

    overall: SupportLabel = "supported"
    per_input: Mapping[tuple[str, int], SupportLabel] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_label(self.overall, _SUPPORT_LABELS, "overall")
        for label in self.per_input.values():
            _check_label(label, _SUPPORT_LABELS, "per-input support")

    @classmethod
    def unassessed(cls) -> Support:
        return cls("missing_evidence")

    def _wire(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "per_input": [
                {"action_id": action, "input": index, "status": label}
                for (action, index), label in sorted(self.per_input.items())
            ],
        }


@dataclass(frozen=True, slots=True)
class Claim:
    """One structure's claim: draws (or why there are none), probability and support.

    ``probability`` is a genuine probability of the structure (a posterior or a
    declared prior). ``completion_count`` is how many completions produced it and
    is never a probability: Bayes weighting refuses it. Build with
    :meth:`evaluated`, :meth:`unidentified` or :meth:`unevaluated`.
    """

    id: str
    status: Literal["evaluated", "unidentified", "unevaluated"]
    source: JointDistributionArtifact | None = None
    probability: float | None = None
    completion_count: int | None = None
    reason: str | None = None
    support: Support | None = None

    @classmethod
    def evaluated(
        cls,
        id: str,
        source: JointDistributionArtifact,
        *,
        probability: float | None = None,
        completion_count: int | None = None,
        support: Support | None = None,
    ) -> Claim:
        """Aligned joint draws under this structure."""
        return cls(id, "evaluated", source, probability, completion_count, None, support)

    @classmethod
    def unidentified(
        cls,
        id: str,
        *,
        probability: float | None = None,
        completion_count: int | None = None,
        support: Support | None = None,
    ) -> Claim:
        """The quantity is not identified under this structure."""
        return cls(id, "unidentified", None, probability, completion_count, None, support)

    @classmethod
    def unevaluated(
        cls,
        id: str,
        reason: str,
        *,
        probability: float | None = None,
        completion_count: int | None = None,
        support: Support | None = None,
    ) -> Claim:
        """The structure was not evaluated (budget, missing evidence, truncation)."""
        return cls(id, "unevaluated", None, probability, completion_count, reason, support)

    def _wire(self) -> dict[str, Any]:
        if self.probability is not None and self.completion_count is not None:
            raise CausalValueError("a claim has a probability or a completion count, not both")
        probability: dict[str, Any] | None = None
        if self.probability is not None:
            probability = {"kind": "genuine", "value": float(self.probability)}
        elif self.completion_count is not None:
            probability = {"kind": "completion_count", "count": int(self.completion_count)}
        return {
            "id": self.id,
            "probability": probability,
            "status": self.status,
            "reason": self.reason,
            "support": None if self.support is None else self.support._wire(),
        }


@dataclass(frozen=True, slots=True)
class ExternalReceipt:
    """What a decision retains of a value an external callback supplied.

    The attested value, the exact request fingerprint and the trust limit.
    There is no native trust: a decision with a receipt is never labelled
    natively verified, and ``trust="native"`` is refused.
    """

    atom_id: str
    provider_id: str
    snapshot_id: str
    request_fingerprint: str
    attested_value: float
    trust: ExternalTrust = "externally_attested"
    attestor: str | None = None

    @classmethod
    def from_claim(
        cls, claim: BoundExternalClaim, atom_id: str, attested_value: float
    ) -> ExternalReceipt:
        """A caller-attested scalar referencing a bound claim's original request.

        The supplied value has no original quantity mapping. Provider, snapshot and
        request references therefore establish no numerical verification or source
        diagnostic license, even when the original bound response was verified.
        """
        identity = claim.identity_fields
        return cls(
            atom_id=atom_id,
            provider_id=str(identity["provider_id"]),
            snapshot_id=str(identity["snapshot_id"]),
            request_fingerprint=str(identity["provider_fingerprint"]),
            attested_value=float(attested_value),
            trust="externally_attested",
            attestor="caller",
        )

    def _wire(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "provider_id": self.provider_id,
            "snapshot_id": self.snapshot_id,
            "request_fingerprint": self.request_fingerprint,
            "attested_value": float(self.attested_value),
            "trust": {"kind": self.trust, "attestor": self.attestor},
        }


def _claims_payload(claims: Sequence[Claim], kind: str) -> tuple[str, list[Any]]:
    _check_label(kind, _CLAIM_KINDS, "kind")
    sources = [None if c.source is None else c.source._native for c in claims]
    return json.dumps([c._wire() for c in claims]), sources


# --------------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class RobustVerdict:
    """The robustness state of a decision.

    ``kind`` is one of ``structurally_robust`` (uniquely best in every structure),
    ``support_robust`` (uniquely best in every structure that meets the support
    rules, unchanged when they are ignored), ``support_dependent`` (the choice
    changes with the support rules), ``graph_dependent_choice`` (supported
    structures disagree), ``unsupported_extrapolation`` (no structure supports any
    admissible action), ``insufficient_claims``, ``no_admissible_action``,
    ``worst_case_choice``, ``bayes_choice`` and ``report_only``.
    """

    kind: RobustVerdictKind
    action: str | None = None
    unrestricted_choice: str | None = None
    leaders: tuple[tuple[str, tuple[str, ...]], ...] = ()
    reason: str | None = None
    evaluated_mass: float | None = None

    @property
    def selected(self) -> str | None:
        """The chosen action, or ``None`` when the verdict does not choose."""
        return self.action

    @classmethod
    def _from_wire(cls, wire: Any) -> RobustVerdict:
        if isinstance(wire, str):
            return cls(cast("RobustVerdictKind", wire))
        ((kind, value),) = wire.items()
        if kind == "support_dependent":
            return cls(
                kind,
                action=value["supported_choice"],
                unrestricted_choice=value["unrestricted_choice"],
            )
        if kind == "graph_dependent_choice":
            return cls(kind, leaders=tuple((atom, tuple(ids)) for atom, ids in value))
        if kind == "insufficient_claims":
            return cls(kind, reason=value)
        if kind == "bayes_choice":
            return cls(kind, action=value["action"], evaluated_mass=value["evaluated_mass"])
        return cls(kind, action=value)


@dataclass(frozen=True, slots=True)
class ActionRobustness:
    """One action across structures and rules."""

    id: str
    declared_exclusion: str | None
    range: tuple[float, float] | None
    supported_range: tuple[float, float] | None
    unsupported_in: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SupportShortfall:
    """An input that fell short of a support rule under one structure."""

    atom: str
    action: str
    input: int
    status: str
    weakest_allowed: str


@dataclass(frozen=True, slots=True)
class ClaimProfile:
    """What the claims carry beyond their draws: uncertainty kind and support."""

    uncertainty: str
    support: Mapping[str, Support]


def _range(value: Any) -> tuple[float, float] | None:
    return None if value is None else (float(value[0]), float(value[1]))


def _range_list(value: tuple[float, float] | None) -> list[float] | None:
    return None if value is None else [value[0], value[1]]


class RobustDecision:
    """A robustness assessment with the evidence behind it; exportable and replayable."""

    def __init__(
        self,
        contract: AdmissibleContract,
        claims: Sequence[Claim],
        kind: str,
        body: Mapping[str, Any],
        *,
        receipts: Sequence[ExternalReceipt] = (),
        replay: Mapping[str, Any] | None = None,
    ) -> None:
        self._contract = contract
        self._claims = tuple(claims)
        self._kind = kind
        self._receipts_in = tuple(receipts)
        self._body = body
        self._robust: Mapping[str, Any] = body["robust"]
        self._result: Mapping[str, Any] = self._robust["result"]
        self._replay = replay

    @property
    def kind(self) -> str:
        """The claim type that was adapted."""
        return self._kind

    @property
    def contract_identity(self) -> str:
        """Identity of the contract with its rules."""
        return str(self._result["contract_identity"])

    @property
    def verdict(self) -> RobustVerdict:
        return RobustVerdict._from_wire(self._result["verdict"])

    @property
    def selected(self) -> str | None:
        return self.verdict.selected

    @property
    def actions(self) -> tuple[ActionRobustness, ...]:
        return tuple(
            ActionRobustness(
                id=a["id"],
                declared_exclusion=a["declared_exclusion"],
                range=_range(a["range"]),
                supported_range=_range(a["supported_range"]),
                unsupported_in=tuple(a["unsupported_in"]),
            )
            for a in self._result["actions"]
        )

    @property
    def shortfalls(self) -> tuple[SupportShortfall, ...]:
        return tuple(
            SupportShortfall(s["atom"], s["action"], s["input"], s["status"], s["weakest_allowed"])
            for s in self._result["shortfalls"]
        )

    @property
    def unsupported_atoms(self) -> tuple[str, ...]:
        """Evaluated structures that support no admissible action."""
        return tuple(self._result["unsupported_atoms"])

    def _mass(self, name: str) -> float | None:
        value = self._result[name]
        return None if value is None else float(value)

    @property
    def unsupported_mass(self) -> float | None:
        return self._mass("unsupported_mass")

    @property
    def unidentified_mass(self) -> float | None:
        """Probability mass of unidentified structures; never renormalized away."""
        return self._mass("unidentified_mass")

    @property
    def unevaluated_mass(self) -> float | None:
        """Probability mass of unevaluated structures; never renormalized away."""
        return self._mass("unevaluated_mass")

    @property
    def evaluated_mass(self) -> float | None:
        return self._mass("evaluated_mass")

    @property
    def uncertainty_required(self) -> str:
        return str(self._result["uncertainty_required"])

    @property
    def uncertainty_supplied(self) -> str:
        return str(self._result["uncertainty_supplied"])

    @property
    def assumptions(self) -> tuple[str, ...]:
        return tuple(self._result["assumptions"])

    @property
    def profile(self) -> ClaimProfile:
        wire = self._robust["profile"]
        return ClaimProfile(
            uncertainty=str(wire["uncertainty"]),
            support={
                entry["atom_id"]: Support(
                    entry["overall"],
                    {(p["action_id"], p["input"]): p["status"] for p in entry["per_input"]},
                )
                for entry in wire["support"]
            },
        )

    @property
    def structures(self) -> tuple[Mapping[str, Any], ...]:
        """Each structure's own answer: status, probability and per-action values."""
        return tuple(self._body["structural"]["atoms"])

    @property
    def completion_counts(self) -> Mapping[str, int]:
        """Completion counts that were supplied and never used as weights."""
        return {str(atom): int(count) for atom, count in self._body["completion_counts"]}

    @property
    def receipts(self) -> tuple[ExternalReceipt, ...]:
        """External callback receipts: attested value, request fingerprint, trust limit."""
        return tuple(
            ExternalReceipt(
                atom_id=r["atom_id"],
                provider_id=r["provider_id"],
                snapshot_id=r["snapshot_id"],
                request_fingerprint=r["request_fingerprint"],
                attested_value=float(r["attested_value"]),
                trust=r["trust"]["label"],
                attestor=r["trust"]["attestor"],
            )
            for r in self._robust["external_receipts"]
        )

    @property
    def native_verified(self) -> bool:
        """``False`` whenever any structure rests on an external callback."""
        return bool(self._robust["native_verified"])

    @property
    def lineage(self) -> tuple[LineageLink, ...]:
        """Derivation chain: decision contract, each structure's draws (behind its
        external provider and receipt when external), and ``robust_decision``."""
        return tuple(
            LineageLink(
                item["id"],
                item["stage"],
                tuple(item["parents"]),
                item["digest"],
                tuple(item["parent_digests"]),
            )
            for item in self._robust["lineage"]
        )

    @property
    def replayed(self) -> bool:
        """Whether this result was recomputed from its inputs by :func:`replay`."""
        return self._replay is not None and bool(self._replay["recomputed"])

    @property
    def external_atoms(self) -> tuple[str, ...]:
        """Structures a replay could not certify natively because a callback supplied them."""
        return () if self._replay is None else tuple(self._replay["external_atoms"])

    def explain(self) -> str:
        """Why this action, or why none: the choice, its evidence and its limits."""
        verdict = self.verdict
        text = _explain_verdict(verdict, self)
        notes: list[str] = []
        if self.unsupported_atoms:
            notes.append(
                f"structures without support for any admissible action: {self.unsupported_atoms}"
            )
        for name, mass in (
            ("unidentified", self.unidentified_mass),
            ("unevaluated", self.unevaluated_mass),
        ):
            if mass:
                notes.append(
                    f"{mass:.3g} {name} probability mass is reported, not renormalized away"
                )
        excluded = [a for a in self.actions if a.declared_exclusion]
        if excluded:
            notes.append(
                "removed by declaration: "
                + ", ".join(f"{a.id!r} ({a.declared_exclusion})" for a in excluded)
            )
        for receipt in self.receipts:
            notes.append(
                f"values of {receipt.atom_id!r} came from an external callback "
                f"({receipt.provider_id}, {receipt.trust}, request {receipt.request_fingerprint}); "
                "not natively verified"
            )
        return text + "".join(f"; {note}" for note in notes) + "."

    def export(self, *, artifact_id: str = "robust-decision-result") -> bytes:
        """The result bound to its contract, structures, profile and receipts, replayable."""
        payload, sources = _claims_payload(self._claims, self._kind)
        data, refusal = _export_robust(
            self._contract._declaration(),
            self._kind,
            payload,
            sources,
            json.dumps([r._wire() for r in self._receipts_in]),
            artifact_id,
        )
        _raise(refusal)
        assert data is not None
        return bytes(data)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form: verdict, per-action ranges, retained masses and external receipts."""
        verdict = self.verdict
        return {
            "kind": self.kind,
            "contract_identity": self.contract_identity,
            "verdict": {
                "kind": verdict.kind,
                "action": verdict.action,
                "unrestricted_choice": verdict.unrestricted_choice,
                "leaders": [[atom, list(ids)] for atom, ids in verdict.leaders],
                "reason": verdict.reason,
                "evaluated_mass": verdict.evaluated_mass,
            },
            "actions": [
                {
                    "id": a.id,
                    "declared_exclusion": a.declared_exclusion,
                    "range": _range_list(a.range),
                    "supported_range": _range_list(a.supported_range),
                    "unsupported_in": list(a.unsupported_in),
                }
                for a in self.actions
            ],
            "unsupported_atoms": list(self.unsupported_atoms),
            "unsupported_mass": self.unsupported_mass,
            "unidentified_mass": self.unidentified_mass,
            "unevaluated_mass": self.unevaluated_mass,
            "evaluated_mass": self.evaluated_mass,
            "uncertainty_required": self.uncertainty_required,
            "uncertainty_supplied": self.uncertainty_supplied,
            "assumptions": list(self.assumptions),
            "native_verified": self.native_verified,
            "receipts": [
                {
                    "atom_id": r.atom_id,
                    "provider_id": r.provider_id,
                    "snapshot_id": r.snapshot_id,
                    "request_fingerprint": r.request_fingerprint,
                    "attested_value": r.attested_value,
                    "provider_trust": r.trust,
                    "attestor": r.attestor,
                }
                for r in self.receipts
            ],
            "replayed": self.replayed,
            "external_atoms": list(self.external_atoms),
        }

    def __repr__(self) -> str:
        return f"<RobustDecision {self.verdict.kind} {self.selected!r}>"


def _explain_verdict(verdict: RobustVerdict, result: RobustDecision) -> str:
    kind = verdict.kind
    if kind == "structurally_robust":
        return f"{verdict.action!r} is uniquely best in every structure, with every input supported as required"
    if kind == "support_robust":
        return (
            f"{verdict.action!r} is uniquely best in every structure that meets the support "
            "rules, and ignoring the rules does not change the choice"
        )
    if kind == "support_dependent":
        other = verdict.unrestricted_choice
        return (
            f"the supported structures choose {verdict.action!r}, but without the support "
            f"rules {'the choice is ' + repr(other) if other else 'no single action is chosen'}: "
            "the choice depends on the support rules"
        )
    if kind == "graph_dependent_choice":
        parts = ", ".join(f"{atom!r} prefers {list(ids)}" for atom, ids in verdict.leaders)
        return f"the choice depends on the structure: {parts}"
    if kind == "unsupported_extrapolation":
        return "no structure supports any admissible action; the decision would be an extrapolation"
    if kind == "insufficient_claims":
        return f"the claims cannot answer the contract: {verdict.reason}"
    if kind == "no_admissible_action":
        return "hard constraints or declared rules remove every action"
    if kind == "worst_case_choice":
        return f"{verdict.action!r} has the best worst case over structures"
    if kind == "bayes_choice":
        return (
            f"{verdict.action!r} has the best probability-weighted value over "
            f"{verdict.evaluated_mass:.3g} evaluated mass"
        )
    return "per-structure answers only; no action is chosen"


def robust(
    contract: AdmissibleContract,
    claims: Sequence[Claim],
    *,
    kind: ClaimKind = "finite_scenarios",
    receipts: Sequence[ExternalReceipt] = (),
) -> RobustDecision:
    """Evaluate claims under an admissible contract and name the robustness state.

    ``kind`` selects the adapter: ``point`` (exactly one claim),
    ``graph_dependent`` (one answer per graph, no probabilities),
    ``weighted_graph_posterior`` (genuine probabilities make the claims credible)
    or ``finite_scenarios`` (always a structural envelope). The contract's
    structural policy decides how structures combine. ``receipts`` name the
    structures whose values an external callback supplied; the result then keeps
    them and is never natively verified.
    """
    payload, sources = _claims_payload(claims, kind)
    result, refusal = _evaluate_robust(
        contract._declaration(),
        kind,
        payload,
        sources,
        json.dumps([r._wire() for r in receipts]),
    )
    _raise(refusal)
    assert result is not None
    return RobustDecision(contract, claims, kind, json.loads(result), receipts=receipts)


def point_claim(contract: AdmissibleContract, claim: Claim, **kwargs: Any) -> RobustDecision:
    """Adapt one point claim (a single exact law or answer)."""
    return robust(contract, [claim], kind="point", **kwargs)


def graph_dependent(
    contract: AdmissibleContract, claims: Sequence[Claim], **kwargs: Any
) -> RobustDecision:
    """Adapt graph-dependent claims: one answer per graph, no probability."""
    return robust(contract, claims, kind="graph_dependent", **kwargs)


def weighted_atoms(
    contract: AdmissibleContract, claims: Sequence[Claim], **kwargs: Any
) -> RobustDecision:
    """Adapt weighted graph-posterior atoms (genuine probabilities make them credible)."""
    return robust(contract, claims, kind="weighted_graph_posterior", **kwargs)


def finite_scenarios(
    contract: AdmissibleContract, claims: Sequence[Claim], **kwargs: Any
) -> RobustDecision:
    """Adapt a finite set of supplied scenarios (always a structural envelope)."""
    return robust(contract, claims, kind="finite_scenarios", **kwargs)


def replay(
    data: bytes,
    *,
    contract: AdmissibleContract,
    claims: Sequence[Claim],
    kind: ClaimKind = "finite_scenarios",
) -> RobustDecision:
    """Recompute a stored robust decision from its inputs and require an exact match.

    The contract and claims are the consumer's own: a result computed under a
    different contract, structures, draws or claim profile, or one whose numbers,
    lineage or trust label were edited, refuses. A structure that rests on an
    external callback is recomputed over the draws supplied, but the callback is
    not re-run: such a result stays ``native_verified == False`` and
    :attr:`RobustDecision.external_atoms` lists the structures.
    """
    payload, sources = _claims_payload(claims, kind)
    result, refusal = _replay_robust(data, contract._declaration(), kind, payload, sources)
    _raise(refusal)
    assert result is not None
    body = json.loads(result)
    # A replay re-derives the robustness result; per-structure answers and
    # completion counts are not stored in the artifact, so they are empty here.
    shell = {"structural": {"atoms": []}, "completion_counts": [], "robust": body["robust"]}
    stored = tuple(
        ExternalReceipt(
            atom_id=r["atom_id"],
            provider_id=r["provider_id"],
            snapshot_id=r["snapshot_id"],
            request_fingerprint=r["request_fingerprint"],
            attested_value=float(r["attested_value"]),
            trust=r["trust"]["label"],
            attestor=r["trust"]["attestor"],
        )
        for r in body["robust"]["external_receipts"]
    )
    return RobustDecision(contract, claims, kind, shell, receipts=stored, replay=body["replay"])


# --------------------------------------------------------------------- identified sets


@dataclass(frozen=True, slots=True)
class IdentifiedUtility:
    """An identified set (an interval) of one action's utility.

    Every value in ``[lower, upper]`` is compatible with the evidence; the
    interval is not a probability law. ``hard_exclusions`` names the hard
    constraints the action violates.
    """

    action_id: str
    lower: float
    upper: float
    hard_exclusions: tuple[str, ...] = ()
    support: Support | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "hard_exclusions", tuple(self.hard_exclusions))

    @classmethod
    def from_inputs(
        cls,
        contract: AdmissibleContract,
        action_id: str,
        inputs: Sequence[tuple[float, float]],
        *,
        hard_exclusions: Sequence[str] = (),
        support: Support | None = None,
    ) -> IdentifiedUtility:
        """The interval of the action's utility from one interval per input.

        An outer enclosure by interval arithmetic: sharp when each input appears
        once in the utility, and possibly wider when one appears twice.
        """
        bounds, refusal = _interval(
            contract._declaration(), action_id, [(float(lo), float(hi)) for lo, hi in inputs]
        )
        _raise(refusal)
        assert bounds is not None
        return cls(action_id, bounds[0], bounds[1], tuple(hard_exclusions), support)

    def _wire(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "lower": float(self.lower),
            "upper": float(self.upper),
            "hard_exclusions": list(self.hard_exclusions),
            "support": None if self.support is None else self.support._wire(),
        }


@dataclass(frozen=True, slots=True)
class IdentifiedAction:
    """One action's interval and what partial identification says of it."""

    id: str
    utility: tuple[float, float]
    hard_exclusions: tuple[str, ...]
    declared_exclusion: str | None
    support_shortfalls: tuple[tuple[int, str], ...]
    eligible: bool
    dominated_by: tuple[str, ...]
    possibly_optimal: bool
    necessarily_optimal: bool
    max_regret: float | None


@dataclass(frozen=True, slots=True)
class IdentifiedVerdict:
    """What partial identification can claim.

    ``kind`` is ``necessarily_best``, ``no_necessarily_best`` (``actions`` are the
    possibly optimal), ``worst_case_choice``, ``minimax_regret_choice``, ``tied``,
    ``report_only``, ``unsupported_extrapolation``, ``insufficient_claims`` or
    ``no_admissible_action``.
    """

    kind: IdentifiedVerdictKind
    action: str | None = None
    actions: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class IdentifiedDecision:
    """A decision over per-action identified sets."""

    contract_identity: str
    policy: str
    actions: tuple[IdentifiedAction, ...]
    lower_leader: str | None
    upper_leader: str | None
    conflicting_leaders: bool
    verdict: IdentifiedVerdict

    @property
    def selected(self) -> str | None:
        """The chosen action, or ``None`` when the verdict does not choose one."""
        return self.verdict.action

    def explain(self) -> str:
        """The choice (or why none), with the identified interval behind it."""
        verdict = self.verdict
        by_id = {a.id: a for a in self.actions}
        if verdict.kind in {"necessarily_best", "worst_case_choice", "minimax_regret_choice"}:
            chosen = by_id[str(verdict.action)]
            text = (
                f"{verdict.action!r} is chosen by {verdict.kind.replace('_', ' ')} "
                f"(interval [{chosen.utility[0]:.4g}, {chosen.utility[1]:.4g}])"
            )
        elif verdict.kind == "no_necessarily_best":
            text = "no action beats every other everywhere; possibly optimal: " + ", ".join(
                repr(a) for a in verdict.actions
            )
        elif verdict.kind == "insufficient_claims":
            text = f"the claims cannot answer the contract: {verdict.reason}"
        else:
            text = verdict.kind.replace("_", " ")
        if self.conflicting_leaders:
            text += (
                f"; the lower-bound leader {self.lower_leader!r} and the upper-bound leader "
                f"{self.upper_leader!r} differ"
            )
        return text + "."

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form: verdict, leaders and each action's identified interval."""
        verdict = self.verdict
        return {
            "contract_identity": self.contract_identity,
            "policy": self.policy,
            "verdict": {
                "kind": verdict.kind,
                "action": verdict.action,
                "actions": list(verdict.actions),
                "reason": verdict.reason,
            },
            "lower_leader": self.lower_leader,
            "upper_leader": self.upper_leader,
            "conflicting_leaders": self.conflicting_leaders,
            "actions": [
                {
                    "id": a.id,
                    "utility": list(a.utility),
                    "hard_exclusions": list(a.hard_exclusions),
                    "declared_exclusion": a.declared_exclusion,
                    "support_shortfalls": [list(s) for s in a.support_shortfalls],
                    "eligible": a.eligible,
                    "dominated_by": list(a.dominated_by),
                    "possibly_optimal": a.possibly_optimal,
                    "necessarily_optimal": a.necessarily_optimal,
                    "max_regret": a.max_regret,
                }
                for a in self.actions
            ],
        }

    def __repr__(self) -> str:
        return f"<IdentifiedDecision {self.verdict.kind} {self.selected!r}>"


def identified_sets(
    contract: AdmissibleContract, utilities: Sequence[IdentifiedUtility]
) -> IdentifiedDecision:
    """Decide over per-action identified sets (partial identification).

    Under ``require_invariant_best_action`` an action is chosen only if its
    interval beats every other eligible action's entirely; otherwise the
    possibly optimal actions are listed. Under ``maximin`` (or the maximin and
    minimax criteria) the best worst case wins; ``regret`` under ``maximin``
    picks the smallest maximum regret. ``bayes_over_structures`` refuses: an
    identified set has no probability law.
    """
    result, refusal = _evaluate_sets(
        contract._declaration(), json.dumps([u._wire() for u in utilities])
    )
    _raise(refusal)
    assert result is not None
    wire = json.loads(result)
    verdict = wire["verdict"]
    return IdentifiedDecision(
        contract_identity=wire["contract_identity"],
        policy=wire["policy"],
        actions=tuple(
            IdentifiedAction(
                id=a["id"],
                utility=(float(a["utility"][0]), float(a["utility"][1])),
                hard_exclusions=tuple(a["hard_exclusions"]),
                declared_exclusion=a["declared_exclusion"],
                support_shortfalls=tuple(
                    (s["input"], s["status"]) for s in a["support_shortfalls"]
                ),
                eligible=a["eligible"],
                dominated_by=tuple(a["dominated_by"]),
                possibly_optimal=a["possibly_optimal"],
                necessarily_optimal=a["necessarily_optimal"],
                max_regret=a["max_regret"],
            )
            for a in wire["actions"]
        ),
        lower_leader=wire["lower_leader"],
        upper_leader=wire["upper_leader"],
        conflicting_leaders=wire["conflicting_leaders"],
        verdict=IdentifiedVerdict(
            kind=cast("IdentifiedVerdictKind", verdict["kind"]),
            action=verdict.get("action"),
            actions=tuple(verdict.get("actions", ())),
            reason=verdict.get("reason"),
        ),
    )


__all__ = [
    "RESULT_LINK_ID",
    "ActionRobustness",
    "AdmissibilityRules",
    "AdmissibleContract",
    "Claim",
    "ClaimProfile",
    "DeclaredExclusion",
    "ExternalReceipt",
    "IdentifiedAction",
    "IdentifiedDecision",
    "IdentifiedUtility",
    "IdentifiedVerdict",
    "RobustDecision",
    "RobustVerdict",
    "Support",
    "SupportRule",
    "SupportShortfall",
    "admissible_contract",
    "finite_scenarios",
    "graph_dependent",
    "identified_sets",
    "point_claim",
    "replay",
    "robust",
    "weighted_atoms",
]
