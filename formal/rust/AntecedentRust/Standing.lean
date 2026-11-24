/-
The production standing checks, as Aeneas translates them from Rust, equal the
specification in `Antecedent.Standing`, so the specification's theorems hold of
the code `compose_claims` runs.

Trusted: Charon and Aeneas translate the Rust faithfully, and the Aeneas model of
the standard library operations the kernels use (enum discriminant reads,
`u8` comparison) is correct.
-/
import Aeneas
import Antecedent.Standing
import AntecedentRust.Extracted.Standing.Funs

open Aeneas Aeneas.Std Result
open AntecedentRust.Extracted.Standing
open Antecedent.Standing

namespace AntecedentRust.Standing

/-- `IdentificationStatus` as the specification names it. -/
def identificationSpec : antecedent_core.identification.IdentificationStatus → Identification
  | .NonparametricallyIdentified => .nonparametric
  | .IdentifiedUnderParametricRestrictions => .parametricRestrictions
  | .IdentifiedUnderPriorRestrictions => .priorRestrictions
  | .PartiallyIdentified => .partiallyIdentified
  | .GraphDependent => .graphDependent
  | .NotIdentified => .notIdentified

/-- `DomainStatus` as the specification names it. -/
def domainSpec : claim.DomainStatus → Domain
  | .Identified => .identified
  | .Supported => .supported
  | .Evaluated => .evaluated
  | .OutsideScope => .outsideScope
  | .Contradicted => .contradicted
  | .Unknown => .unknown
  | .Limited => .limited
  | .NotClaimed => .notClaimed

/-- Both correspondences are bijections: every Rust variant has exactly one specification variant. -/
theorem identificationSpec_bijective :
    (∀ a b, identificationSpec a = identificationSpec b → a = b) ∧
      ∀ s, ∃ a, identificationSpec a = s := by
  refine ⟨fun a b h => ?_, fun s => ?_⟩
  · cases a <;> cases b <;> simp_all [identificationSpec]
  · cases s
    exacts [⟨.NonparametricallyIdentified, rfl⟩, ⟨.IdentifiedUnderParametricRestrictions, rfl⟩,
      ⟨.IdentifiedUnderPriorRestrictions, rfl⟩, ⟨.PartiallyIdentified, rfl⟩,
      ⟨.GraphDependent, rfl⟩, ⟨.NotIdentified, rfl⟩]

theorem domainSpec_bijective :
    (∀ a b, domainSpec a = domainSpec b → a = b) ∧ ∀ s, ∃ a, domainSpec a = s := by
  refine ⟨fun a b h => ?_, fun s => ?_⟩
  · cases a <;> cases b <;> simp_all [domainSpec]
  · cases s
    exacts [⟨.Identified, rfl⟩, ⟨.Supported, rfl⟩, ⟨.Evaluated, rfl⟩, ⟨.OutsideScope, rfl⟩,
      ⟨.Contradicted, rfl⟩, ⟨.Unknown, rfl⟩, ⟨.Limited, rfl⟩, ⟨.NotClaimed, rfl⟩]

/-- `strength_rank` returns the specification's rank and never fails. -/
theorem strength_rank_spec (s : antecedent_core.identification.IdentificationStatus) :
    ∃ r, antecedent_core.identification.IdentificationStatus.strength_rank s = ok r ∧
      r.val = (identificationSpec s).rank := by
  cases s <;> simp [antecedent_core.identification.IdentificationStatus.strength_rank,
    identificationSpec, Identification.rank]

/-- `is_positive` is the specification's `asserts`. -/
theorem is_positive_spec (s : claim.DomainStatus) :
    claim.DomainStatus.is_positive s = ok (domainSpec s).asserts := by
  cases s <;> simp [claim.DomainStatus.is_positive, domainSpec, Domain.asserts]

/-- The derived `PartialEq` on `DomainStatus` is equality of the specification statuses. -/
theorem domain_eq_spec (a b : claim.DomainStatus) :
    claim.DomainStatus.Insts.CoreCmpPartialEqDomainStatus.eq a b =
      ok (domainSpec a == domainSpec b) := by
  cases a <;> cases b <;>
    simp [claim.DomainStatus.Insts.CoreCmpPartialEqDomainStatus.eq,
      claim.DomainStatus.read_discriminant, domainSpec]

/-- The production relay rule `domain_change_allowed` is `domainChangeAllowed`. -/
theorem domain_change_allowed_spec (f t : claim.DomainStatus) :
    claim.domain_change_allowed f t = ok (domainChangeAllowed (domainSpec f) (domainSpec t)) := by
  cases f <;> cases t <;>
    simp [claim.domain_change_allowed, domain_eq_spec, is_positive_spec, domainSpec,
      domainChangeAllowed, Domain.asserts]

/-- The production identification rule is `identificationKeptOrLowered`. -/
theorem identification_status_kept_or_lowered_spec
    (p d : antecedent_core.identification.IdentificationStatus) :
    claim.identification_status_kept_or_lowered p d =
      ok (identificationKeptOrLowered (identificationSpec p) (identificationSpec d)) := by
  cases p <;> cases d <;>
    simp [claim.identification_status_kept_or_lowered,
      antecedent_core.identification.IdentificationStatus.strength_rank, identificationSpec,
      identificationKeptOrLowered, Identification.rank]

/-- Whenever the production relay check accepts a domain change, the derived status is no higher. -/
theorem rust_domain_change_never_raises {f t : claim.DomainStatus}
    (h : claim.domain_change_allowed f t = ok true) : (domainSpec t).le (domainSpec f) := by
  rw [domain_change_allowed_spec] at h
  exact domainChangeAllowed_le (by simpa using h)

/-- Whenever the production identification check accepts, the derived identification is no stronger. -/
theorem rust_identification_never_raised
    {p d : antecedent_core.identification.IdentificationStatus}
    (h : claim.identification_status_kept_or_lowered p d = ok true) :
    (identificationSpec d).le (identificationSpec p) := by
  rw [identification_status_kept_or_lowered_spec] at h
  exact (identificationKeptOrLowered_iff _ _).1 (by simpa using h)

/-! ## Every other derivation and the handoff chain -/

/-- `ClaimKind` as the specification names it. -/
def kindSpec : claim.ClaimKind → Kind
  | .Point => .point
  | .Bounds => .bounds
  | .Mixture => .mixture
  | .Response => .response
  | .PolicyValue => .policyValue
  | .Refusal => .refusal
  | .Incomplete => .incomplete
  | .Associational => .associational

theorem kindSpec_bijective :
    (∀ a b, kindSpec a = kindSpec b → a = b) ∧ ∀ s, ∃ a, kindSpec a = s := by
  refine ⟨fun a b h => ?_, fun s => ?_⟩
  · cases a <;> cases b <;> simp_all [kindSpec]
  · cases s
    exacts [⟨.Point, rfl⟩, ⟨.Bounds, rfl⟩, ⟨.Mixture, rfl⟩, ⟨.Response, rfl⟩,
      ⟨.PolicyValue, rfl⟩, ⟨.Refusal, rfl⟩, ⟨.Incomplete, rfl⟩, ⟨.Associational, rfl⟩]

/-- `domain_upgraded` is `domainUpgraded`. -/
theorem domain_upgraded_spec (p d : claim.DomainStatus) :
    claim.domain_upgraded p d = ok (domainUpgraded (domainSpec p) (domainSpec d)) := by
  cases p <;> cases d <;>
    simp [claim.domain_upgraded, domain_eq_spec, core.cmp.PartialEq.ne.trait_default, core.cmp.PartialEq.ne.default, bne, is_positive_spec, domainSpec, domainUpgraded,
      Domain.asserts]

/-- `domain_departs` is `domainDeparts`. -/
theorem domain_departs_spec (p d : claim.DomainStatus) :
    claim.domain_departs p d = ok (domainDeparts (domainSpec p) (domainSpec d)) := by
  cases p <;> cases d <;>
    simp [claim.domain_departs, domain_eq_spec, core.cmp.PartialEq.ne.trait_default, core.cmp.PartialEq.ne.default, bne, domainSpec, domainDeparts]

/-- `associational_derives_causal` is `associationalDerivesCausal`. -/
theorem associational_derives_causal_spec (p d : claim.ClaimKind) :
    claim.associational_derives_causal p d =
      ok (associationalDerivesCausal (kindSpec p) (kindSpec d)) := by
  cases p <;> cases d <;>
    simp [claim.associational_derives_causal, kindSpec, associationalDerivesCausal, Kind.noValue]

/-- `non_value_derives_value` is `nonValueDerivesValue`. -/
theorem non_value_derives_value_spec (p d : claim.ClaimKind) :
    claim.non_value_derives_value p d = ok (nonValueDerivesValue (kindSpec p) (kindSpec d)) := by
  cases p <;> cases d <;>
    simp [claim.non_value_derives_value, kindSpec, nonValueDerivesValue, Kind.noValue]

/-- `identification_slot_added` is `identificationSlotAdded`. -/
theorem identification_slot_added_spec (every derived : Bool) :
    claim.identification_slot_added every derived = ok (identificationSlotAdded every derived) := by
  cases every <;> cases derived <;> simp [claim.identification_slot_added, identificationSlotAdded]

/-- `chain_retains` is `chainRetains`. -/
theorem chain_retains_spec (c f n r o : Bool) :
    claim.chain_retains c f n r o = ok (chainRetains c f n r o) := by
  cases c <;> cases f <;> cases n <;> cases r <;> cases o <;>
    simp [claim.chain_retains, chainRetains]

/-- Whenever neither production kind rule refuses, the derived kind is no stronger. -/
theorem rust_kind_rules_never_raise {p d : claim.ClaimKind}
    (ha : claim.associational_derives_causal p d = ok false)
    (hv : claim.non_value_derives_value p d = ok false) :
    (kindSpec d).level ≤ (kindSpec p).level := by
  rw [associational_derives_causal_spec] at ha
  rw [non_value_derives_value_spec] at hv
  exact (kind_rules_iff _ _).1 ⟨by simpa using ha, by simpa using hv⟩

/-- Whenever the production departure rule accepts, the derived domain status is no higher. -/
theorem rust_domain_departs_never_raises {p d : claim.DomainStatus}
    (h : claim.domain_departs p d = ok false) : (domainSpec d).le (domainSpec p) := by
  rw [domain_departs_spec] at h
  exact (domainDeparts_iff _ _).1 (by simpa using h)

end AntecedentRust.Standing
