"""The composition boundary: what a decision input carries and what may be combined.

A :class:`DecisionInput` wraps one source (a mean grid, an aligned joint law or a
scalar claim) with its provenance (who produced it, under which trust, with which
receipt, calibration and capabilities) and a per-coordinate support map::

    claim = DecisionInput.from_claim("lab", bound_external_claim)
    law = DecisionInput.from_distribution("study", joint_law)
    decided = evaluate_with_support(contract, [claim, law])
    decided.verdict.kind            # compared, or one evaluated, or no supported action
    decided.disposition("treat")    # evaluated, unsupported or unevaluated, with a reason

Trust is never taken from a label. An artifact that calls itself ``native`` or
``exact`` is stored as ``unverified`` unless the library object that produced it
supplies the native execution record or the exact-request verification receipt:
Python can state ``TrustEvidence.none()`` or an external attestation and nothing
stronger, and a requirement of ``"native"`` refuses the label
(:class:`UnverifiedTrustRefusal`).

Support is decided per action. An action whose coordinate is missing or too
weakly supported, or that its source cannot answer (a mean answers only the
expectation of an affine utility over non-outcome inputs), is reported as
``unsupported`` or ``unevaluated`` with a reason and is never scored, while the
supported actions are still compared; ``SupportPolicy.require_all()`` refuses
instead. No supported action at all is a state, not an error.

Statistical pooling, Bayesian borrowing, causal transport and evidence reuse are
separate declared :data:`Operation` values. Independent pooling or paired draws
refuse when two inputs share data, a prior or a fitted model, or when their
dependence is unknown (:class:`DependenceRefusal`), unless a covariance or
joint-law route is declared and licensed by an aligned joint law that is one of
the pair. Conflicting structural atoms never become a silent average.

Rust owns every decision here; this module builds declarations and raises each
refusal as a :class:`CompositionRefusal`, a
:class:`~antecedent.decision.DecisionRefusal` (hence a
:class:`~antecedent.errors.CausalUnsupportedError`) with its registered
``reason_code`` and namespaced ``detail``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from . import _native
from ._native import composition_check as _check
from ._native import composition_check_atoms as _check_atoms
from ._native import composition_check_paired_draws as _check_paired
from ._native import composition_evaluate as _evaluate
from ._native import composition_functional as _functional
from ._native import composition_input_from_distribution as _from_distribution
from ._native import composition_input_from_means as _from_means
from ._native import composition_input_from_scalar as _from_scalar
from .decision import ActionOutcome, ConstraintExclusion, Contract, DecisionRefusal
from .decision_robust import SupportLabel
from .errors import CausalTypeError, CausalValueError
from .external import BoundExternalClaim
from .joint_distribution import JointDistributionArtifact, ScientificQuantity

TrustRequirement = Literal["unrestricted", "native", "exact_request_verified"]
Operation = Literal[
    "statistical_pooling", "bayesian_borrowing", "causal_transport", "evidence_reuse"
]
AtomCombination = Literal["report_each", "worst_case", "weighted_by_declared_probabilities"]
ProviderKind = Literal["native", "external_attested", "external_exact_request_verified"]
OPERATIONS: tuple[Operation, ...] = (
    "statistical_pooling",
    "bayesian_borrowing",
    "causal_transport",
    "evidence_reuse",
)
_REQUIREMENTS = ("unrestricted", "native", "exact_request_verified")
_SUPPORT_NAMES = (
    "supported",
    "weak_overlap",
    "extrapolative",
    "outside_empirical_support",
    "missing_evidence",
)
_TRUST_DETAILS = frozenset(
    {"metadata_only_native_claim", "native_required", "verification_receipt_missing"}
)
_SUPPORT_DETAILS = frozenset({"unsupported_action_not_comparable", "mean_is_not_a_distribution"})
_DEPENDENCE_DETAILS = frozenset(
    {
        "operation_not_declared",
        "shared_evidence_not_independent",
        "unknown_dependence_is_not_independence",
        "dependence_route_not_licensed",
        "paired_draws_across_sources",
        "conflicting_atoms_not_averaged",
        "atom_probabilities_invalid",
    }
)


# --------------------------------------------------------------------------- refusals


class CompositionRefusal(DecisionRefusal):
    """A refused input, evaluation or combination at the composition boundary."""


class UnverifiedTrustRefusal(CompositionRefusal):
    """A native or verified provider was required and only a label supports it.

    ``detail`` is ``composition_boundary.metadata_only_native_claim`` (the artifact
    says ``native`` with no matching native execution record),
    ``composition_boundary.native_required`` or
    ``composition_boundary.verification_receipt_missing``.
    """


class SupportRefusal(CompositionRefusal):
    """Actions that cannot be compared on their sources.

    Raised under ``SupportPolicy.require_all()`` for an unsupported or
    unevaluated action, and for a functional a mean cannot answer.
    """


class DependenceRefusal(CompositionRefusal):
    """Inputs that may not be combined: shared or unknown dependence, an undeclared
    operation, an unlicensed route, draws that cannot be paired, or atoms that
    would be averaged without declared probabilities."""


def _refusal_type(refusal: Mapping[str, Any]) -> type[CompositionRefusal]:
    detail = str(refusal["detail"])
    family, _, slot = detail.partition(".")
    if family == "composition_boundary":
        if slot in _TRUST_DETAILS:
            return UnverifiedTrustRefusal
        if slot in _SUPPORT_DETAILS:
            return SupportRefusal
        if slot in _DEPENDENCE_DETAILS:
            return DependenceRefusal
    return CompositionRefusal


def _raise(refusal: str | None) -> None:
    if refusal is not None:
        wire = json.loads(refusal)
        raise _refusal_type(wire)(wire)


def _label(name: str, value: str, allowed: Sequence[str]) -> str:
    if value not in allowed:
        raise CausalValueError(f"{name} is one of {', '.join(allowed)}; got {value!r}")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise CausalTypeError(f"{name} must be a string")
    if not value.strip():
        raise CausalValueError(f"{name} must be non-empty")
    return value


# --------------------------------------------------------------------------- trust


class TrustEvidence:
    """What backs an input's trust, as far as Python may state it.

    ``TrustEvidence.none()`` (the default) proves nothing and leaves an input
    ``unverified``; :meth:`attested` records an external party's attestation and
    never more. A native execution record or an exact-request verification receipt
    is minted by the library object that produced the input and handed over as a
    ``TrustEvidence``; there is no constructor for either here.
    """

    __slots__ = ("_native_value",)

    def __init__(self, native: _native.TrustEvidence) -> None:
        if not isinstance(native, _native.TrustEvidence):
            raise CausalTypeError("TrustEvidence wraps library-produced evidence only")
        self._native_value = native

    @classmethod
    def none(cls) -> TrustEvidence:
        """No evidence beyond the artifact's own labels."""
        return cls(_native.TrustEvidence.none())

    @classmethod
    def attested(cls, attestor: str) -> TrustEvidence:
        """An external attestation by ``attestor``; never native, never verified."""
        return cls(_native.TrustEvidence.attested(_text("attestor", attestor)))

    @property
    def kind(self) -> str:
        """``none``, ``externally_attested``, ``native_execution`` or
        ``exact_request_verified``."""
        return str(self._native_value.kind)

    def __repr__(self) -> str:
        return f"<TrustEvidence {self.kind}>"


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True, slots=True)
class InputProvenance:
    """Who produced an input, as established by evidence and not by a label.

    ``trust`` is never above what the evidence supports: an artifact labelled
    ``native_licensed`` without a matching native execution record is
    ``unverified`` here, and ``calibration`` is then ``exact_claim_unverified``
    (the ``exact`` claim is not used).
    """

    provider_kind: ProviderKind
    trust: str
    receipt: Mapping[str, Any] | None
    calibration: str
    capabilities: tuple[str, ...]
    provider_id: str
    snapshot_id: str
    lineage_digest: str

    @property
    def native(self) -> bool:
        """Whether Antecedent itself executed the route that produced the input."""
        return self.provider_kind == "native"


@dataclass(frozen=True, slots=True)
class CoordinateSupport:
    """The support status of one source coordinate."""

    coordinate: ScientificQuantity
    status: SupportLabel


class DecisionInput:
    """One decision input: a source, its provenance and its per-coordinate support.

    Build with :meth:`from_distribution`, :meth:`from_means`, :meth:`from_scalar` or
    :meth:`from_claim`. A mean, a scalar and an aligned joint law are different
    objects; none stands in for another.
    """

    __slots__ = ("_native_value", "_provenance", "_source", "_support")

    def __init__(self, native: _native.DecisionInput) -> None:
        if not isinstance(native, _native.DecisionInput):
            raise CausalTypeError("use DecisionInput.from_distribution, from_means or from_scalar")
        summary = json.loads(native.summary_json())
        wire = summary["provenance"]
        self._native_value = native
        self._source: Literal["mean", "joint_law", "scalar"] = summary["source"]
        self._provenance = InputProvenance(
            provider_kind=wire["provider_kind"],
            trust=wire["trust"],
            receipt=wire["receipt"],
            calibration=wire["calibration"],
            capabilities=tuple(wire["capabilities"]),
            provider_id=wire["provider_id"],
            snapshot_id=wire["snapshot_id"],
            lineage_digest=wire["lineage_digest"],
        )
        self._support = tuple(
            CoordinateSupport(ScientificQuantity._from_wire(entry["coordinate"]), entry["status"])
            for entry in summary["support"]
        )

    @classmethod
    def from_distribution(
        cls,
        input_id: str,
        artifact: JointDistributionArtifact,
        *,
        evidence: TrustEvidence | None = None,
        requirement: TrustRequirement = "unrestricted",
    ) -> DecisionInput:
        """An input from an aligned joint-law artifact.

        The artifact's ``trust`` and ``calibration`` are claims, not evidence: the
        provider kind comes from ``evidence`` only. Coordinate support is the
        artifact's mask (absent: every coordinate supported). ``requirement``
        ``"native"`` refuses an artifact that merely says it is native.
        """
        if not isinstance(artifact, JointDistributionArtifact):
            raise CausalTypeError("artifact must be a JointDistributionArtifact")
        native, refusal = _from_distribution(
            _text("input_id", input_id),
            artifact._native,
            _evidence(evidence),
            _label("requirement", requirement, _REQUIREMENTS),
        )
        return cls._built(native, refusal)

    @classmethod
    def from_native_claim(cls, input_id: str, claim: Any, *, contract: Contract) -> DecisionInput:
        """Consume an actual issued native response and its execution record.

        Point summaries and retained aligned posterior rows preserve their own
        support and calibration. Projection metadata cannot supply this authority.
        """
        from .program_claims import NativeClaim

        if not isinstance(claim, NativeClaim):
            raise CausalTypeError("claim must be an issued NativeClaim")
        native, refusal = _native.composition_input_from_native_claim(
            _text("input_id", input_id), claim._native, json.dumps(contract._wire())
        )
        return cls._built(native, refusal)

    @classmethod
    def from_means(
        cls,
        input_id: str,
        quantities: Sequence[ScientificQuantity],
        means: Sequence[float],
        *,
        provider_id: str,
        snapshot_id: str,
        causal_contract_id: str,
        support: SupportLabel | Sequence[SupportLabel] = "supported",
        evidence: TrustEvidence | None = None,
        requirement: TrustRequirement = "unrestricted",
    ) -> DecisionInput:
        """An input from a mean source: one mean per coordinate.

        ``support`` is one label for every coordinate or one label each. A mean
        carries no distribution and no pairing, so it answers only the expectation
        of an affine utility over non-outcome inputs.
        """
        quantities = tuple(quantities)
        values = [float(v) for v in means]
        if len(values) != len(quantities):
            raise CausalValueError("one mean per coordinate")
        labels = _labels(support, len(quantities))
        native, refusal = _from_means(
            _text("input_id", input_id),
            json.dumps([q._wire() for q in quantities]),
            values,
            labels,
            _text("provider_id", provider_id),
            _text("snapshot_id", snapshot_id),
            _text("causal_contract_id", causal_contract_id),
            _evidence(evidence),
            _label("requirement", requirement, _REQUIREMENTS),
        )
        return cls._built(native, refusal)

    @classmethod
    def from_scalar(
        cls,
        input_id: str,
        quantity: ScientificQuantity,
        value: float,
        *,
        provider_id: str,
        snapshot_id: str,
        causal_contract_id: str,
        support: SupportLabel = "supported",
        evidence: TrustEvidence | None = None,
        requirement: TrustRequirement = "unrestricted",
    ) -> DecisionInput:
        """An input from one scalar claim for one coordinate."""
        native, refusal = _from_scalar(
            _text("input_id", input_id),
            json.dumps(quantity._wire()),
            float(value),
            _label("support", support, _SUPPORT_NAMES),
            _text("provider_id", provider_id),
            _text("snapshot_id", snapshot_id),
            _text("causal_contract_id", causal_contract_id),
            _evidence(evidence),
            _label("requirement", requirement, _REQUIREMENTS),
        )
        return cls._built(native, refusal)

    @classmethod
    def from_claim(
        cls,
        input_id: str,
        claim: BoundExternalClaim,
        *,
        requirement: TrustRequirement = "unrestricted",
    ) -> DecisionInput:
        """An input from a bound external claim, as an attested mean source.

        The claim's own per-coordinate support labels are kept. Its trust is carried
        as an external attestation by its provider whatever its label: a
        verification receipt can only come from the library, so a ``native``
        requirement refuses it.
        """
        if not isinstance(claim, BoundExternalClaim):
            raise CausalTypeError("claim must be a BoundExternalClaim")
        identity = claim.identity_fields
        result = cls.from_means(
            input_id,
            claim.quantities,
            [float(v) for v in claim.values],
            provider_id=str(identity["provider_id"]),
            snapshot_id=str(identity["snapshot_id"]),
            causal_contract_id=str(identity["causal_contract_id"]),
            support=cast("tuple[SupportLabel, ...]", claim.support),
            evidence=TrustEvidence.attested(str(identity["provider_id"])),
            requirement=requirement,
        )

        result._native_value._retain_external_source(claim._native)
        return result

    @classmethod
    def _built(cls, native: Any, refusal: str | None) -> DecisionInput:
        _raise(refusal)
        assert native is not None
        return cls(native)

    @property
    def id(self) -> str:
        """The input's identity within one composition."""
        return str(self._native_value.id)

    @property
    def source(self) -> Literal["mean", "joint_law", "scalar"]:
        """What the input supplies."""
        return self._source

    @property
    def source_evidence(self):
        """Original native diagnostics; absent for metadata-only sources."""
        from .source_evidence import SourceEvidence

        if not self._native_value.has_source_evidence:
            return None
        return SourceEvidence._deferred(lambda: self._native_value.source_evidence)

    @property
    def provenance(self) -> InputProvenance:
        """Who produced the input and how far it is trusted, as the evidence supports."""
        return self._provenance

    @property
    def native(self) -> bool:
        """Whether a native execution record backs this input: never a label."""
        return self._provenance.native

    @property
    def support(self) -> tuple[CoordinateSupport, ...]:
        """Every coordinate with its support status, in source order."""
        return self._support

    def explain(self) -> str:
        """What the input supplies, who produced it, and how far it is trusted and supported."""
        p = self._provenance
        who = "Antecedent itself" if p.native else f"an external provider ({p.provider_id})"
        weakest = [c.status for c in self._support]
        worst = max(weakest, key=_SUPPORT_NAMES.index) if weakest else "no coordinates"
        return (
            f"Input {self.id!r} supplies a {self._source.replace('_', ' ')} produced by {who}. "
            f"Provider trust: {p.trust}; calibration: {p.calibration}; natively executed: "
            f"{self.native}. {len(self._support)} coordinate(s), weakest support: {worst}. "
            "Trust is what the evidence supports, never a label."
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe provenance and per-coordinate support."""
        p = self._provenance
        return {
            "id": self.id,
            "source": self._source,
            "provider_kind": p.provider_kind,
            "provider_trust": p.trust,
            "calibration": p.calibration,
            "native": self.native,
            "capabilities": list(p.capabilities),
            "provider_id": p.provider_id,
            "snapshot_id": p.snapshot_id,
            "lineage_digest": p.lineage_digest,
            "support": [c.status for c in self._support],
        }

    def __repr__(self) -> str:
        return (
            f"<DecisionInput {self.id!r} {self._source} "
            f"trust={self._provenance.trust} native={self.native}>"
        )


def _evidence(evidence: TrustEvidence | None) -> _native.TrustEvidence:
    if evidence is None:
        return _native.TrustEvidence.none()
    if not isinstance(evidence, TrustEvidence):
        raise CausalTypeError("evidence must be a TrustEvidence")
    return evidence._native_value


def _labels(support: str | Sequence[str], count: int) -> list[str]:
    if isinstance(support, str):
        return [_label("support", support, _SUPPORT_NAMES)] * count
    labels = [_label("support", s, _SUPPORT_NAMES) for s in support]
    if len(labels) != count:
        raise CausalValueError("one support label per coordinate")
    return labels


def _natives(inputs: Sequence[DecisionInput]) -> list[_native.DecisionInput]:
    items = tuple(inputs)
    if any(not isinstance(i, DecisionInput) for i in items):
        raise CausalTypeError("inputs must be DecisionInput values")
    return [i._native_value for i in items]


# --------------------------------------------------------------------------- evaluation


@dataclass(frozen=True, slots=True)
class SupportPolicy:
    """What to do with actions that cannot be evaluated.

    ``unsupported="compare_supported"`` reports them and compares the rest;
    ``"require_all"`` refuses unless every action is evaluated. ``weakest_support``
    is the weakest status a read coordinate may have.
    """

    unsupported: Literal["compare_supported", "require_all"] = "compare_supported"
    weakest_support: SupportLabel = "supported"

    def __post_init__(self) -> None:
        _label("unsupported", self.unsupported, ("compare_supported", "require_all"))
        _label("weakest_support", self.weakest_support, _SUPPORT_NAMES)

    @classmethod
    def compare_supported(cls, weakest_support: SupportLabel = "supported") -> SupportPolicy:
        """Report unevaluable actions and compare the rest."""
        return cls("compare_supported", weakest_support)

    @classmethod
    def require_all(cls, weakest_support: SupportLabel = "supported") -> SupportPolicy:
        """Refuse unless every action can be evaluated."""
        return cls("require_all", weakest_support)


@dataclass(frozen=True, slots=True)
class UnsupportedReason:
    """One coordinate that blocked an action on one input.

    ``issue`` is ``composition_boundary.coordinate_missing`` (the source has no such
    coordinate) or ``composition_boundary.coordinate_unsupported`` (its ``support``
    is below the policy's weakest allowed status). ``coordinate`` is the position
    of the action's input.
    """

    input_id: str
    coordinate: int
    issue: str
    support: str | None


@dataclass(frozen=True, slots=True)
class ActionDisposition:
    """What became of one action.

    ``status`` is ``evaluated`` (on ``input_id``), ``unsupported`` (a coordinate is
    missing or too weakly supported on every input; see ``reasons``) or
    ``unevaluated`` (coordinates are supported but no input's representation can
    answer it; see ``unevaluated_reason``).
    """

    id: str
    status: Literal["evaluated", "unsupported", "unevaluated"]
    input_id: str | None
    reasons: tuple[UnsupportedReason, ...]
    unevaluated_reason: str | None

    @property
    def evaluated(self) -> bool:
        """Whether this action was scored."""
        return self.status == "evaluated"


@dataclass(frozen=True, slots=True)
class SupportedVerdict:
    """What the support-aware comparison can claim.

    ``kind`` is ``uniquely_optimal`` or ``indistinguishable`` (at least two
    actions evaluated and compared), ``no_admissible_action``,
    ``only_one_evaluated`` (nothing was compared) or ``no_supported_action``
    (a state, not an error).
    """

    kind: Literal[
        "uniquely_optimal",
        "indistinguishable",
        "no_admissible_action",
        "only_one_evaluated",
        "no_supported_action",
    ]
    actions: tuple[str, ...]

    @property
    def compared(self) -> bool:
        """Whether at least two evaluated actions were compared."""
        return self.kind in {"uniquely_optimal", "indistinguishable", "no_admissible_action"}

    @property
    def selected(self) -> str | None:
        """The one best action, when the verdict names exactly one."""
        if self.kind in {"uniquely_optimal", "only_one_evaluated"}:
            return self.actions[0]
        return None


@dataclass(frozen=True, slots=True)
class SupportedDecision:
    """The result of a support-aware evaluation.

    ``outcomes`` are the evaluated actions only, in declaration order; an action in
    another disposition is never scored. ``evpi`` is present only when every
    evaluated action was compared state by state on one aligned source; actions
    answered by different sources are compared by criterion value alone, and their
    regret is withheld.
    """

    contract_identity: str
    dispositions: tuple[ActionDisposition, ...]
    outcomes: tuple[ActionOutcome, ...]
    verdict: SupportedVerdict
    evpi: float | None
    sources: int
    assumptions: tuple[str, ...]
    source_evidence: tuple[Any, ...] = ()

    def disposition(self, action_id: str) -> ActionDisposition:
        """The disposition of one action."""
        for item in self.dispositions:
            if item.id == action_id:
                return item
        raise CausalValueError(f"no action {action_id!r} in this decision")

    @property
    def evaluated_actions(self) -> tuple[str, ...]:
        """Ids of the actions that were scored."""
        return tuple(d.id for d in self.dispositions if d.status == "evaluated")

    @property
    def unsupported_actions(self) -> tuple[str, ...]:
        """Ids of the actions a coordinate's missing or weak support blocked."""
        return tuple(d.id for d in self.dispositions if d.status == "unsupported")

    @property
    def unevaluated_actions(self) -> tuple[str, ...]:
        """Ids of the supported actions no input's representation can answer."""
        return tuple(d.id for d in self.dispositions if d.status == "unevaluated")

    def outcome(self, action_id: str) -> ActionOutcome:
        """The outcome of an evaluated action."""
        for item in self.outcomes:
            if item.id == action_id:
                return item
        raise CausalValueError(f"action {action_id!r} was not evaluated")

    def explain(self) -> str:
        """The verdict, which actions were never scored and why, and what is withheld."""
        verdict = self.verdict
        if verdict.kind == "uniquely_optimal":
            text = f"Among {len(self.outcomes)} compared action(s), {verdict.actions[0]!r} is best."
        elif verdict.kind == "indistinguishable":
            text = "The best actions are indistinguishable: " + ", ".join(
                repr(a) for a in verdict.actions
            )
            text += "."
        elif verdict.kind == "only_one_evaluated":
            text = f"Only {verdict.actions[0]!r} could be evaluated, so nothing was compared."
        elif verdict.kind == "no_admissible_action":
            text = "No evaluated action satisfies the hard constraints."
        else:
            text = "No action is supported by the supplied inputs, so nothing was scored."
        if self.unsupported_actions:
            text += " Unsupported (never scored): " + ", ".join(self.unsupported_actions) + "."
        if self.unevaluated_actions:
            text += (
                " Unevaluated (no input can answer): " + ", ".join(self.unevaluated_actions) + "."
            )
        if self.evpi is None:
            text += " Regret and EVPI are withheld: the actions are not on one aligned source."
        else:
            text += f" EVPI {self.evpi:.4g}."
        if self.assumptions:
            text += " Assumptions: " + "; ".join(self.assumptions) + "."
        return text

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form: verdict, dispositions, outcomes and assumptions."""
        return {
            "contract_identity": self.contract_identity,
            "verdict": {"kind": self.verdict.kind, "actions": list(self.verdict.actions)},
            "dispositions": [
                {
                    "id": d.id,
                    "status": d.status,
                    "input_id": d.input_id,
                    "reasons": [
                        {
                            "input_id": r.input_id,
                            "coordinate": r.coordinate,
                            "issue": r.issue,
                            "support": r.support,
                        }
                        for r in d.reasons
                    ],
                    "unevaluated_reason": d.unevaluated_reason,
                }
                for d in self.dispositions
            ],
            "outcomes": [
                {
                    "id": o.id,
                    "admissible": o.admissible,
                    "expected_utility": o.expected_utility,
                    "value": o.value,
                    "standard_error": o.standard_error,
                    "expected_regret": o.expected_regret,
                    "max_regret": o.max_regret,
                }
                for o in self.outcomes
            ],
            "evpi": self.evpi,
            "sources": self.sources,
            "assumptions": list(self.assumptions),
        }

    def __repr__(self) -> str:
        return (
            f"<SupportedDecision {self.verdict.kind} evaluated={len(self.evaluated_actions)} "
            f"unsupported={len(self.unsupported_actions)} "
            f"unevaluated={len(self.unevaluated_actions)}>"
        )


def _outcome(wire: Mapping[str, Any]) -> ActionOutcome:
    return ActionOutcome(
        id=wire["id"],
        admissible=wire["admissible"],
        exclusions=tuple(
            ConstraintExclusion(e["constraint_id"], e["probability"], e["required"])
            for e in wire["exclusions"]
        ),
        expected_utility=wire["expected_utility"],
        value=wire["value"],
        standard_error=wire["standard_error"],
        expected_regret=wire["expected_regret"],
        max_regret=wire["max_regret"],
    )


def _disposition(wire: Mapping[str, Any]) -> ActionDisposition:
    return ActionDisposition(
        id=wire["id"],
        status=wire["status"],
        input_id=wire["input_id"],
        reasons=tuple(
            UnsupportedReason(r["input_id"], r["coordinate"], r["issue"], r["support"])
            for r in wire["reasons"]
        ),
        unevaluated_reason=wire["unevaluated_reason"],
    )


def evaluate_with_support(
    contract: Contract,
    inputs: Sequence[DecisionInput],
    policy: SupportPolicy | None = None,
) -> SupportedDecision:
    """Evaluate ``contract`` over ``inputs``, deciding support per action.

    Each action is assigned to the first input whose support map covers every
    coordinate it reads at or above ``policy.weakest_support`` and whose
    representation can answer it. The actions assigned to one input are evaluated
    together by the existing evaluator, with hard constraints narrowed to those
    actions. Unsupported and unevaluated actions are reported with a reason and
    left out of the comparison under ``compare_supported``; when every action is
    unsupported the verdict is ``no_supported_action``.

    Raises :class:`SupportRefusal` for an unevaluated action under ``require_all``,
    :class:`DependenceRefusal` for a state-aligned criterion (regret, expected
    regret) over actions answered by different sources, and
    :class:`CompositionRefusal` for any refusal of the underlying evaluator (a mean
    source with a hard constraint, a non-joint draw alignment, a non-outcome
    meaning).
    """
    if not isinstance(contract, Contract):
        raise CausalTypeError("contract must be a decision.Contract")
    policy = SupportPolicy() if policy is None else policy
    result, refusal = _evaluate(
        json.dumps(contract._wire()),
        _natives(inputs),
        policy.unsupported,
        policy.weakest_support,
    )
    _raise(refusal)
    assert result is not None
    wire = json.loads(result)
    return _supported_from_wire(
        wire,
        source_evidence=tuple(
            evidence.project(
                contract,
                [row["id"] for row in wire["dispositions"] if row["input_id"] == source.id],
            )
            for source in inputs
            if (evidence := source.source_evidence) is not None
        ),
    )


def _supported_from_wire(
    wire: Mapping[str, Any], *, source_evidence: Sequence[Any] = ()
) -> SupportedDecision:
    verdict = wire["verdict"]
    return SupportedDecision(
        contract_identity=wire["contract_identity"],
        dispositions=tuple(_disposition(d) for d in wire["dispositions"]),
        outcomes=tuple(_outcome(o) for o in wire["outcomes"]),
        verdict=SupportedVerdict(verdict["kind"], tuple(verdict["actions"])),
        evpi=wire["evpi"],
        sources=int(wire["sources"]),
        assumptions=tuple(wire["assumptions"]),
        source_evidence=tuple(source_evidence),
    )


@dataclass(frozen=True, slots=True)
class Functional:
    """A typed functional of one action's utility.

    A joint law answers every functional. A mean or scalar answers only
    :meth:`expectation` (or :meth:`expected_utility`) of an affine utility over
    non-outcome inputs; a probability, quantile, variance, tail expectation,
    nonlinear utility or outcome-law input refuses
    (``composition_boundary.mean_is_not_a_distribution``), never approximated
    from a mean.
    """

    kind: str
    threshold: float | None = None
    p: float | None = None
    tail: Literal["lower", "upper"] | None = None

    @classmethod
    def expectation(cls) -> Functional:
        """The expectation of the action's utility."""
        return cls("expectation")

    @classmethod
    def expected_utility(cls) -> Functional:
        """The expected utility of the action."""
        return cls("expected_utility")

    @classmethod
    def variance(cls) -> Functional:
        """The variance of the action's utility; needs a joint law."""
        return cls("variance")

    @classmethod
    def probability(cls, threshold: float, tail: Literal["lower", "upper"] = "lower") -> Functional:
        """``P(utility <= threshold)`` (lower) or ``P(utility >= threshold)`` (upper)."""
        return cls("probability", threshold=float(threshold), tail=tail)

    @classmethod
    def quantile(cls, p: float) -> Functional:
        """The ``p`` quantile of the action's utility; needs a joint law."""
        return cls("quantile", p=float(p))

    @classmethod
    def tail_expectation(cls, p: float, tail: Literal["lower", "upper"] = "lower") -> Functional:
        """The mean of the ``tail`` ``p`` fraction of the utility; needs a joint law."""
        return cls("tail_expectation", p=float(p), tail=tail)

    def _wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"kind": self.kind}
        for name in ("threshold", "p", "tail"):
            value = getattr(self, name)
            if value is not None:
                wire[name] = value
        return wire


@dataclass(frozen=True, slots=True)
class FunctionalValue:
    """One functional's value; a mean-source value has no standard error."""

    value: float
    standard_error: float | None
    source_mode: str
    source_evidence: tuple[Any, ...] = ()
    _artifact_factory: Any = field(default=None, repr=False, compare=False)

    @property
    def source_artifact(self) -> Any:
        """Independently consumed original-law or affine-source transformation artifact."""
        return None if self._artifact_factory is None else self._artifact_factory()


def evaluate_functional(
    contract: Contract,
    action_id: str,
    functional: Functional,
    source: DecisionInput,
    *,
    weakest_support: SupportLabel = "supported",
) -> FunctionalValue:
    """One functional of one action's utility from one input."""
    if not isinstance(contract, Contract):
        raise CausalTypeError("contract must be a decision.Contract")
    if not isinstance(source, DecisionInput):
        raise CausalTypeError("source must be a DecisionInput")
    result, refusal = _functional(
        json.dumps(contract._wire()),
        _text("action_id", action_id),
        json.dumps(functional._wire()),
        source._native_value,
        _label("weakest_support", weakest_support, _SUPPORT_NAMES),
    )
    _raise(refusal)
    assert result is not None
    wire = json.loads(result)
    evidence = source.source_evidence

    def build_artifact() -> Any:
        if source.source == "joint_law":
            from .functional_source import LawFunctionalArtifact

            return LawFunctionalArtifact._produce(contract, action_id, functional, source)
        if evidence is not None:
            from .source_projection import SourceProjectionArtifact

            return SourceProjectionArtifact.produce(
                evidence, "transformation", contract=contract, action_id=action_id
            )
        return None

    return FunctionalValue(
        wire["value"],
        wire["standard_error"],
        wire["source_mode"],
        () if evidence is None else (evidence.project(contract, [action_id]),),
        build_artifact if source.source == "joint_law" or evidence is not None else None,
    )


# --------------------------------------------------------------------------- dependence


@dataclass(frozen=True, slots=True)
class EvidenceRelation:
    """How two inputs' evidence relates.

    Build with :meth:`independent`, :meth:`shared_data`, :meth:`shared_prior`,
    :meth:`shared_fitted_model` or :meth:`unknown`. Unknown dependence is never
    independence, and a pair declared independent that carries the same snapshot
    identity is treated as sharing that data.
    """

    kind: Literal[
        "independent_sources",
        "shared_data",
        "shared_prior",
        "shared_fitted_model",
        "unknown_dependence",
    ]
    ids: tuple[str, ...] = ()

    @classmethod
    def independent(cls) -> EvidenceRelation:
        """The two inputs rest on independent sources (declared, not checked)."""
        return cls("independent_sources")

    @classmethod
    def shared_data(cls, *ids: str) -> EvidenceRelation:
        """The two inputs were fitted on shared data named by ``ids``."""
        return cls("shared_data", tuple(ids))

    @classmethod
    def shared_prior(cls, prior_id: str) -> EvidenceRelation:
        """The two inputs share the prior ``prior_id``."""
        return cls("shared_prior", (_text("prior_id", prior_id),))

    @classmethod
    def shared_fitted_model(cls, model_id: str) -> EvidenceRelation:
        """The two inputs come from the one fitted model ``model_id``."""
        return cls("shared_fitted_model", (_text("model_id", model_id),))

    @classmethod
    def unknown(cls) -> EvidenceRelation:
        """Dependence is unknown; this is never independence and refuses every operation."""
        return cls("unknown_dependence")

    def _wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"kind": self.kind}
        if self.kind == "shared_data":
            wire["ids"] = list(self.ids)
        elif self.kind in {"shared_prior", "shared_fitted_model"}:
            wire["id"] = self.ids[0]
        return wire


@dataclass(frozen=True, slots=True)
class DependenceRoute:
    """A declared route that accounts for dependence between two inputs.

    ``source_input`` must be one of the pair and an aligned joint law with the
    matching capability (``covariance`` needs the covariance capability,
    ``joint_law`` the sample capability); anything else refuses with
    ``composition_boundary.dependence_route_not_licensed``.
    """

    kind: Literal["covariance", "joint_law"]
    id: str
    source_input: str

    def _wire(self) -> dict[str, Any]:
        return {"kind": self.kind, "id": self.id, "source_input": self.source_input}


@dataclass(frozen=True, slots=True)
class PairRelation:
    """The declared relation between two inputs, with an optional dependence route."""

    left: str
    right: str
    relation: EvidenceRelation
    route: DependenceRoute | None = None

    def _wire(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "relation": self.relation._wire(),
            "route": None if self.route is None else self.route._wire(),
        }


@dataclass(frozen=True, slots=True)
class CompositionReceipt:
    """What a passed composition check records."""

    operation: Operation
    independence_assumed: bool
    routes: tuple[str, ...]
    shared_evidence: tuple[str, ...]

    def explain(self) -> str:
        """What was checked, whether independence was assumed, and what evidence is shared."""
        text = f"The inputs may be combined under {self.operation!r}."
        if self.independence_assumed:
            text += " Independence of the sources was assumed, not shown."
        else:
            text += " Independence was not assumed."
        if self.shared_evidence:
            text += " Shared evidence: " + ", ".join(self.shared_evidence) + "."
        if self.routes:
            text += " Declared dependence routes: " + ", ".join(self.routes) + "."
        return text

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form of this receipt."""
        return {
            "operation": self.operation,
            "independence_assumed": self.independence_assumed,
            "routes": list(self.routes),
            "shared_evidence": list(self.shared_evidence),
        }

    def __repr__(self) -> str:
        return (
            f"<CompositionReceipt {self.operation} independence_assumed="
            f"{self.independence_assumed} shared={len(self.shared_evidence)}>"
        )


def _receipt(result: str | None) -> CompositionReceipt:
    assert result is not None
    wire = json.loads(result)
    return CompositionReceipt(
        operation=wire["operation"],
        independence_assumed=bool(wire["independence_assumed"]),
        routes=tuple(wire["routes"]),
        shared_evidence=tuple(wire["shared_evidence"]),
    )


def _relations_wire(relations: Sequence[PairRelation]) -> str:
    items = tuple(relations)
    if any(not isinstance(r, PairRelation) for r in items):
        raise CausalTypeError("relations must be PairRelation values")
    return json.dumps([r._wire() for r in items])


def check_composition(
    inputs: Sequence[DecisionInput],
    relations: Sequence[PairRelation],
    operation: Operation | None = None,
) -> CompositionReceipt:
    """Check that ``inputs`` may be combined under the declared ``operation``.

    Every pair of inputs needs a declared :class:`PairRelation`; a pair with none
    is unknown dependence. The four operations are separate and none implies
    another: statistical pooling refuses shared data, priors and fitted models;
    Bayesian borrowing allows a shared prior but refuses shared data and fitted
    models; causal transport and evidence reuse record shared evidence without
    assuming independence. Unknown dependence refuses every operation. A declared
    covariance or joint-law route lifts a refusal only when it is licensed.

    ``operation=None`` refuses with ``composition_boundary.operation_not_declared``.
    Raises :class:`DependenceRefusal`.
    """
    if operation is not None:
        _label("operation", operation, OPERATIONS)
    result, refusal = _check(_natives(inputs), _relations_wire(relations), operation)
    _raise(refusal)
    return _receipt(result)


def check_paired_draws(
    inputs: Sequence[DecisionInput], relations: Sequence[PairRelation]
) -> CompositionReceipt:
    """Check that the draws of ``inputs`` may be paired row by row.

    Pairing draws from separate sources treats them as independent, so this is
    statistical pooling plus a requirement that every input is an aligned joint
    law. Inputs that share data refuse (``shared_evidence_not_independent``).
    """
    result, refusal = _check_paired(_natives(inputs), _relations_wire(relations))
    _raise(refusal)
    return _receipt(result)


@dataclass(frozen=True, slots=True)
class StructuralAtom:
    """One alternative structure, with its declared probability if it has one.

    A completion count is not a probability; leave ``probability`` unset for it.
    """

    id: str
    probability: float | None = None


@dataclass(frozen=True, slots=True)
class AtomPlan:
    """A permitted way to combine structural atoms.

    ``kind`` is ``report_each``, ``worst_case`` or ``weighted``; ``weights`` are the
    declared ``(atom id, probability)`` pairs of a weighted plan, never
    renormalized.
    """

    kind: Literal["report_each", "worst_case", "weighted"]
    weights: tuple[tuple[str, float], ...] = ()


def check_atom_combination(
    atoms: Sequence[StructuralAtom], combination: AtomCombination = "report_each"
) -> AtomPlan:
    """Check that structural ``atoms`` may be combined as requested.

    Two or more atoms are alternative structures; averaging them
    (``weighted_by_declared_probabilities``) is licensed only when every atom carries
    a declared probability and they sum to at most one, so conflicting atoms are
    never silently averaged and missing mass is retained.
    """
    items = tuple(atoms)
    if any(not isinstance(a, StructuralAtom) for a in items):
        raise CausalTypeError("atoms must be StructuralAtom values")
    _label(
        "combination",
        combination,
        ("report_each", "worst_case", "weighted_by_declared_probabilities"),
    )
    result, refusal = _check_atoms(
        json.dumps([{"id": a.id, "probability": a.probability} for a in items]), combination
    )
    _raise(refusal)
    assert result is not None
    wire = json.loads(result)
    return AtomPlan(wire["kind"], tuple((str(i), float(p)) for i, p in wire.get("weights", ())))


__all__ = [
    "OPERATIONS",
    "ActionDisposition",
    "AtomCombination",
    "AtomPlan",
    "CompositionReceipt",
    "CompositionRefusal",
    "CoordinateSupport",
    "DecisionInput",
    "DependenceRefusal",
    "DependenceRoute",
    "EvidenceRelation",
    "Functional",
    "FunctionalValue",
    "InputProvenance",
    "Operation",
    "PairRelation",
    "StructuralAtom",
    "SupportPolicy",
    "SupportRefusal",
    "SupportedDecision",
    "SupportedVerdict",
    "TrustEvidence",
    "UnsupportedReason",
    "UnverifiedTrustRefusal",
    "check_atom_combination",
    "check_composition",
    "check_paired_draws",
    "evaluate_functional",
    "evaluate_with_support",
]
