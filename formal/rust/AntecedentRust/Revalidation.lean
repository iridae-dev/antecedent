/-
The production revalidation rules of the composition graph and the evidence
ledger, as Aeneas translates them from Rust, equal the specification in
`Antecedent.Revalidation`, so the specification's theorems hold of the code
`revalidate` and the ledger's reclassification run.

Trusted: Charon and Aeneas translate the Rust faithfully, and the Aeneas model of
the standard-library operations the kernels use (enum discriminant reads, usize
comparison, boolean operations) is correct.
-/
import Aeneas
import Antecedent.Revalidation
import AntecedentRust.Extracted.Revalidation.Funs

open Aeneas Aeneas.Std Result
open AntecedentRust.Extracted.Revalidation
open Antecedent.Revalidation

namespace AntecedentRust.Revalidation

/-- `Revalidation` as the specification names it. -/
def classSpec : antecedent_core.recalc.Revalidation → Revalidation
  | .StillValid => .stillValid
  | .Recompute => .recompute
  | .Withdrawn => .withdrawn

/-- `TrustStanding` as the specification names it. -/
def trustSpec : composition_verification.graph.TrustStanding → Trust
  | .NotApplicable => .notApplicable
  | .Held => .held
  | .Withdrawn => .withdrawn
  | .Undeclared => .undeclared

/-- `ProviderTrust` as the specification names it. -/
def providerSpec : provider_trust.ProviderTrust → ProviderTrust
  | .NativeSupported => .nativeSupported
  | .ExternallyAttested => .externallyAttested
  | .ExactRequestVerified => .exactRequestVerified
  | .Unverified => .unverified

/-- `StatusStep` as the specification names it. -/
def stepSpec : evidence_ledger.ledger.StatusStep → StatusStep
  | .Withdraw => .withdraw
  | .NeedRecompute => .needRecompute
  | .Keep => .keep

theorem classSpec_injective (a b : antecedent_core.recalc.Revalidation)
    (h : classSpec a = classSpec b) : a = b := by
  cases a <;> cases b <;> simp_all [classSpec]

theorem trustSpec_injective (a b : composition_verification.graph.TrustStanding)
    (h : trustSpec a = trustSpec b) : a = b := by
  cases a <;> cases b <;> simp_all [trustSpec]

/-- `TrustStanding::of_provider` is `Trust.ofProvider`. -/
theorem of_provider_spec (p : provider_trust.ProviderTrust) :
    ∃ t, composition_verification.graph.TrustStanding.of_provider p = ok t ∧
      trustSpec t = Trust.ofProvider (providerSpec p) := by
  cases p <;> simp [composition_verification.graph.TrustStanding.of_provider, trustSpec,
    providerSpec, Trust.ofProvider]

/-- `trust_withdraws` is `trustWithdraws`. -/
theorem trust_withdraws_spec (t : composition_verification.graph.TrustStanding) :
    composition_verification.revalidate.trust_withdraws t = ok (trustWithdraws (trustSpec t)) := by
  cases t <;> simp [composition_verification.revalidate.trust_withdraws, trustSpec, trustWithdraws]

/-- `trust_move_is_numerical` is `trustMoveIsNumerical`. -/
theorem trust_move_is_numerical_spec (b n : composition_verification.graph.TrustStanding) :
    composition_verification.revalidate.trust_move_is_numerical b n =
      ok (trustMoveIsNumerical (trustSpec b) (trustSpec n)) := by
  cases b <;> cases n <;> simp [composition_verification.revalidate.trust_move_is_numerical,
    composition_verification.revalidate.trust_withdraws, trustSpec, trustMoveIsNumerical,
    trustWithdraws, core.cmp.PartialEq.ne.trait_default, core.cmp.PartialEq.ne.default,
    composition_verification.graph.TrustStanding.Insts.CoreCmpPartialEqTrustStanding.eq,
    composition_verification.graph.TrustStanding.read_discriminant]

/-- `revalidation_class` is `revalidationClass`. -/
theorem revalidation_class_spec (w n : Bool) :
    ∃ c, composition_verification.revalidate.revalidation_class w n = ok c ∧
      classSpec c = revalidationClass w n := by
  cases w <;> cases n <;> simp [composition_verification.revalidate.revalidation_class, classSpec,
    revalidationClass]

/-- `loss_class` is `lossClass`. -/
theorem loss_class_spec (any remaining needed : Usize) :
    ∃ c, evidence_ledger.ledger.loss_class any remaining needed = ok c ∧
      classSpec c = lossClass any.val remaining.val needed.val := by
  unfold evidence_ledger.ledger.loss_class lossClass
  by_cases h0 : any.val = 0
  · have : any = 0#usize := by scalar_tac
    simp [this, classSpec]
  · have hne : any ≠ 0#usize := fun h => h0 (by simp [h])
    by_cases hr : remaining.val < needed.val
    · have hlt : remaining < needed := by scalar_tac
      simp [hne, hlt, h0, hr, classSpec]
    · have hlt : ¬ remaining < needed := by scalar_tac
      simp [hne, hlt, h0, hr, classSpec]

/-- `status_step` is `statusStep`. -/
theorem status_step_spec (c r : antecedent_core.recalc.Revalidation) :
    ∃ s, evidence_ledger.ledger.status_step c r = ok s ∧
      stepSpec s = statusStep (classSpec c) (classSpec r) := by
  cases c <;> cases r <;> simp [evidence_ledger.ledger.status_step, stepSpec, statusStep, classSpec]

/-- A claim the production rules classify still valid kept its provider trust standing. -/
theorem rust_trust_move_never_still_valid (b n : composition_verification.graph.TrustStanding)
    (w x : Bool) (h : b ≠ n) :
    claimClass (trustSpec b) (trustSpec n) w x ≠ .stillValid :=
  trust_move_never_still_valid _ _ _ _ (fun e => h (trustSpec_injective _ _ e))

end AntecedentRust.Revalidation
