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

end AntecedentRust.Standing
