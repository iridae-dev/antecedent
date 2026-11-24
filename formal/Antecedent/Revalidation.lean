/-
Revalidation, support and provider-trust standing: how what a result rests on
changing moves its class.

Every reuse vocabulary adapts onto one three-state class: still valid,
recompute, withdrawn. This file states the classes, the rules that move a
result between them, and proves the standing properties the code claims:

* combining classes (`worst`) is associative, commutative and idempotent, with
  still valid as identity and withdrawn absorbing;
* a class is still valid only when nothing stands against the result: a moved
  provider trust standing, a recomputed or removed input, a lost source;
* a result never returns to valid by reclassification;
* a provider's trust holds only at a verified level;
* an accepted support summary is never better than any coordinate's label.

Rust counterparts (proved equal to these definitions in `formal/rust`, except
where marked):
* `Revalidation.worst`   ↔ `antecedent_core::recalc::Revalidation::worst`
* `ChangeKind.revalidation` ↔ `antecedent_core::recalc::ChangeKind::revalidation`
* `StageStatus.revalidation` ↔ `antecedent_core::recalc::StageStatus::revalidation`
* `Support.severity`     ↔ `antecedent_core::response::SupportStatus::severity`
* `Trust.ofProvider`     ↔ `antecedent_contract::composition_verification::graph::TrustStanding::of_provider`
* `trustMoveIsNumerical`, `trustWithdraws`, `revalidationClass` ↔
  `antecedent_contract::composition_verification::revalidate::{trust_move_is_numerical,
  trust_withdraws, revalidation_class}`
* `lossClass`, `statusStep` ↔ `antecedent_contract::evidence_ledger::ledger::{loss_class, status_step}`
* `labelsAccepted` models `antecedent_contract::support_labels::check_static_point_labels`
  (not translated; the Kani harnesses in `formal/kani/src/support_labels.rs` check the Rust).
-/

namespace Antecedent.Revalidation

/-- The three-state class, least severe first. -/
inductive Revalidation where
  | stillValid
  | recompute
  | withdrawn
  deriving DecidableEq, Repr

/-- Severity: larger is worse. -/
def Revalidation.rank : Revalidation → Nat
  | .stillValid => 0
  | .recompute => 1
  | .withdrawn => 2

theorem Revalidation.rank_injective {a b : Revalidation} (h : a.rank = b.rank) : a = b := by
  cases a <;> cases b <;> simp_all [Revalidation.rank]

/-- The more severe of two classes, as the derived order's `max` computes it. -/
def Revalidation.worst (a b : Revalidation) : Revalidation :=
  if b.rank < a.rank then a else b

theorem worst_comm (a b : Revalidation) : a.worst b = b.worst a := by
  cases a <;> cases b <;> rfl

theorem worst_assoc (a b c : Revalidation) : (a.worst b).worst c = a.worst (b.worst c) := by
  cases a <;> cases b <;> cases c <;> rfl

theorem worst_idem (a : Revalidation) : a.worst a = a := by
  cases a <;> rfl

theorem worst_stillValid (a : Revalidation) : a.worst .stillValid = a := by
  cases a <;> rfl

theorem worst_withdrawn (a : Revalidation) : a.worst .withdrawn = .withdrawn := by
  cases a <;> rfl

/-- The combined class is at least as severe as either. -/
theorem le_worst (a b : Revalidation) : a.rank ≤ (a.worst b).rank ∧ b.rank ≤ (a.worst b).rank := by
  cases a <;> cases b <;> decide

/-- Combining is still valid only when both are. -/
theorem worst_eq_stillValid {a b : Revalidation} :
    a.worst b = .stillValid ↔ a = .stillValid ∧ b = .stillValid := by
  cases a <;> cases b <;> decide

/-! ## Recalculation stages -/

/-- How a stage's own input or a dependency changed. -/
inductive ChangeKind where
  | added
  | removed
  | modified
  deriving DecidableEq, Repr

/-- An input that is gone withdraws its dependents; any other change asks for recomputation. -/
def ChangeKind.revalidation : ChangeKind → Revalidation
  | .removed => .withdrawn
  | _ => .recompute

/-- A change never leaves its dependents still valid, and withdraws exactly on removal. -/
theorem ChangeKind.revalidation_lowers (c : ChangeKind) :
    c.revalidation ≠ .stillValid ∧ (c.revalidation = .withdrawn ↔ c = .removed) := by
  cases c <;> decide

/-- What forced a recomputation. -/
inductive Because where
  | own (change : ChangeKind)
  | upstream (change : ChangeKind)
  | freshProcess
  deriving DecidableEq, Repr

/-- Per-stage outcome of a plan, with the stage names and refusal details left out. -/
inductive StageStatus where
  | reused
  | recomputed (because : Because)
  | refused
  deriving DecidableEq, Repr

/-- `StageStatus::revalidation`. -/
def StageStatus.revalidation : StageStatus → Revalidation
  | .reused => .stillValid
  | .recomputed (.own c) | .recomputed (.upstream c) => c.revalidation
  | .recomputed .freshProcess => .recompute
  | .refused => .withdrawn

/-- Only reuse is still valid: a recomputed or refused stage never is. -/
theorem StageStatus.stillValid_iff (s : StageStatus) :
    s.revalidation = .stillValid ↔ s = .reused := by
  cases s with
  | reused => decide
  | recomputed b => cases b with
    | own c => cases c <;> decide
    | upstream c => cases c <;> decide
    | freshProcess => decide
  | refused => decide

/-- A withdrawn stage was refused or lost an input. -/
theorem StageStatus.withdrawn_iff (s : StageStatus) :
    s.revalidation = .withdrawn ↔
      s = .refused ∨ s = .recomputed (.own .removed) ∨ s = .recomputed (.upstream .removed) := by
  cases s with
  | reused => decide
  | recomputed b => cases b with
    | own c => cases c <;> decide
    | upstream c => cases c <;> decide
    | freshProcess => decide
  | refused => decide

/-! ## Claims in a composition graph -/

/-- The class from what stands against a claim (`revalidation_class`). -/
def revalidationClass (withdrawn numerical : Bool) : Revalidation :=
  if withdrawn then .withdrawn else if numerical then .recompute else .stillValid

theorem revalidationClass_stillValid (w n : Bool) :
    revalidationClass w n = .stillValid ↔ w = false ∧ n = false := by
  cases w <;> cases n <;> decide

theorem revalidationClass_withdrawn (w n : Bool) :
    revalidationClass w n = .withdrawn ↔ w = true := by
  cases w <;> cases n <;> decide

/-- A provider's verification level (`ProviderTrust`). -/
inductive ProviderTrust where
  | nativeSupported
  | externallyAttested
  | exactRequestVerified
  | unverified
  deriving DecidableEq, Repr

/-- Whether a node's provider trust holds (`TrustStanding`). -/
inductive Trust where
  | notApplicable
  | held
  | withdrawn
  | undeclared
  deriving DecidableEq, Repr

/-- `TrustStanding::of_provider`: an unverified provider holds no trust. -/
def Trust.ofProvider : ProviderTrust → Trust
  | .unverified => .withdrawn
  | _ => .held

/-- A provider's trust is held exactly at a verified level, and is otherwise withdrawn. -/
theorem Trust.ofProvider_held (p : ProviderTrust) :
    (Trust.ofProvider p = .held ↔ p ≠ .unverified) ∧
      (Trust.ofProvider p = .withdrawn ↔ p = .unverified) := by
  cases p <;> decide

/-- `trust_withdraws`: trust no longer holds, or was not stated. -/
def trustWithdraws (now : Trust) : Bool := now == .withdrawn || now == .undeclared

/-- `trust_move_is_numerical`: the standing moved, to one that does not withdraw. -/
def trustMoveIsNumerical (before now : Trust) : Bool := before != now && !trustWithdraws now

/--
The class of one claim as `revalidate` decides it from its provider trust, every
other withdrawal (`w`) and every other numerical reason (`n`).
-/
def claimClass (before now : Trust) (w n : Bool) : Revalidation :=
  revalidationClass (w || trustWithdraws now) (n || trustMoveIsNumerical before now)

/-- A claim whose provider trust moved is never still valid: restored trust asks for recomputation. -/
theorem trust_move_never_still_valid (before now : Trust) (w n : Bool) (h : before ≠ now) :
    claimClass before now w n ≠ .stillValid := by
  cases before <;> cases now <;> cases w <;> cases n <;> simp_all [claimClass,
    revalidationClass, trustWithdraws, trustMoveIsNumerical]

/-- A claim whose trust is withdrawn or undeclared is withdrawn, whatever else holds. -/
theorem trust_withdrawn_withdraws (before now : Trust) (w n : Bool) (h : trustWithdraws now = true) :
    claimClass before now w n = .withdrawn := by
  simp [claimClass, revalidationClass, h]

/-- A still-valid claim kept its trust standing, and that standing does not withdraw it. -/
theorem still_valid_trust (before now : Trust) (w n : Bool)
    (h : claimClass before now w n = .stillValid) :
    before = now ∧ trustWithdraws now = false ∧ w = false ∧ n = false := by
  cases before <;> cases now <;> cases w <;> cases n <;> simp_all [claimClass,
    revalidationClass, trustWithdraws, trustMoveIsNumerical]

/-! ## Evidence-ledger results -/

/-- `loss_class`: nothing lost keeps a result, too few sources left withdraw it. -/
def lossClass (any remaining needed : Nat) : Revalidation :=
  if any = 0 then .stillValid else if remaining < needed then .withdrawn else .recompute

/-- A result keeps its class only when no source was lost. -/
theorem lossClass_stillValid (any remaining needed : Nat) :
    lossClass any remaining needed = .stillValid ↔ any = 0 := by
  unfold lossClass
  split
  · simp_all
  · split <;> simp_all

/-- A result is withdrawn exactly when a source was lost and fewer than it needs remain. -/
theorem lossClass_withdrawn (any remaining needed : Nat) :
    lossClass any remaining needed = .withdrawn ↔ any ≠ 0 ∧ remaining < needed := by
  unfold lossClass
  split
  · simp_all
  · split <;> simp_all

/-- How a reclassification changes a result's status (`StatusStep`). -/
inductive StatusStep where
  | withdraw
  | needRecompute
  | keep
  deriving DecidableEq, Repr

/-- `status_step`. -/
def statusStep (cls current : Revalidation) : StatusStep :=
  match cls, current with
  | .withdrawn, _ => .withdraw
  | .recompute, .stillValid => .needRecompute
  | _, _ => .keep

/-- The class of a result after a step. -/
def StatusStep.apply (current : Revalidation) : StatusStep → Revalidation
  | .withdraw => .withdrawn
  | .needRecompute => .recompute
  | .keep => current

/-- A reclassification combines the new class with the current one, as `worst` does. -/
theorem statusStep_worst (cls current : Revalidation) :
    (statusStep cls current).apply current = current.worst cls := by
  cases cls <;> cases current <;> rfl

/-- A reclassification never lowers a result's severity: no result returns to valid. -/
theorem statusStep_never_restores (cls current : Revalidation) :
    current.rank ≤ ((statusStep cls current).apply current).rank ∧
      cls.rank ≤ ((statusStep cls current).apply current).rank := by
  rw [statusStep_worst]
  exact ⟨(le_worst current cls).1, (le_worst current cls).2⟩

/-- Reclassifications applied in sequence never lower a result's severity. -/
theorem reclassify_monotone (current : Revalidation) (classes : List Revalidation) :
    current.rank ≤ (classes.foldl (fun c k => (statusStep k c).apply c) current).rank := by
  induction classes generalizing current with
  | nil => exact Nat.le_refl _
  | cons k ks ih =>
    exact Nat.le_trans (statusStep_never_restores k current).1 (ih _)

/-! ## Support labels -/

/-- Empirical support classification (`SupportStatus`). -/
inductive Support where
  | supported
  | weakOverlap
  | extrapolative
  | outsideEmpiricalSupport
  | missingEvidence
  deriving DecidableEq, Repr

/-- `SupportStatus::severity`: larger is weaker support. -/
def Support.severity : Support → Nat
  | .supported => 0
  | .weakOverlap => 1
  | .extrapolative => 2
  | .outsideEmpiricalSupport => 3
  | .missingEvidence => 4

theorem Support.severity_injective {a b : Support} (h : a.severity = b.severity) : a = b := by
  cases a <;> cases b <;> simp_all [Support.severity]

/-- The worst label of a list: the most severe, none for an empty list. -/
def worstLabel : List Support → Option Support
  | [] => none
  | l :: ls =>
    match worstLabel ls with
    | none => some l
    | some w => some (if w.severity < l.severity then l else w)

theorem worstLabel_ge : ∀ {ls : List Support} {w : Support}, worstLabel ls = some w →
    (∀ l ∈ ls, l.severity ≤ w.severity) ∧ w ∈ ls
  | [], _, h => by simp [worstLabel] at h
  | l :: ls, w, h => by
    simp only [worstLabel] at h
    cases hw : worstLabel ls with
    | none =>
      rw [hw] at h
      cases h
      have hnil : ls = [] := by
        cases ls with
        | nil => rfl
        | cons _ _ => simp [worstLabel] at hw; split at hw <;> simp at hw
      subst hnil
      simp
    | some w' =>
      rw [hw] at h
      obtain ⟨hge, hmem⟩ := worstLabel_ge hw
      cases h
      refine ⟨fun x hx => ?_, ?_⟩
      · rcases List.mem_cons.1 hx with rfl | hx
        · split <;> omega
        · have := hge x hx
          split <;> omega
      · split
        · exact List.mem_cons_self
        · exact List.mem_cons_of_mem _ hmem

/--
`check_static_point_labels`: one label per value, and a summary equal to the
worst label (any summary is accepted for an empty grid).
-/
def labelsAccepted (labels : List Support) (status : Support) (valueLen : Nat) : Bool :=
  labels.length == valueLen &&
    match worstLabel labels with
    | none => true
    | some w => w == status

/--
An accepted summary is never better than any coordinate's label, and is one of the
labels: a grid with a missing-evidence coordinate is summarized as missing evidence.
-/
theorem labelsAccepted_summary {labels : List Support} {status : Support} {n : Nat}
    (h : labelsAccepted labels status n = true) (hne : labels ≠ []) :
    (∀ l ∈ labels, l.severity ≤ status.severity) ∧ status ∈ labels := by
  simp only [labelsAccepted, Bool.and_eq_true] at h
  obtain ⟨_, h⟩ := h
  cases hw : worstLabel labels with
  | none =>
    cases labels with
    | nil => exact absurd rfl hne
    | cons _ _ => simp [worstLabel] at hw; split at hw <;> simp at hw
  | some w =>
    rw [hw] at h
    have hws : w = status := by simpa using h
    subst hws
    exact worstLabel_ge hw

end Antecedent.Revalidation
