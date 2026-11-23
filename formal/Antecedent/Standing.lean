/-
Claim standing and the rule that a derivation cannot raise it.

A claim reports an identification status and a status for each domain axis
(identification, support, evaluation). Its standing is what those statuses
assert. A derived claim (a relay of one parent, or a checked composition of
several) is accepted only if it asserts nothing its parents did not. This file
states that order independently of the Rust checks, states the checks as the
Rust code makes them, and proves that every accepted derivation, and every
finite chain of them, keeps or lowers standing.

Rust counterparts (proved equal to these definitions in `formal/rust`):
* `Identification.rank`  ↔ `antecedent_core::identification::IdentificationStatus::strength_rank`
* `Domain.asserts`       ↔ `antecedent_contract::claim::DomainStatus::is_positive`
* `domainChangeAllowed`  ↔ `antecedent_contract::claim::domain_change_allowed`
* `identificationKeptOrLowered` ↔ `antecedent_contract::claim::identification_status_kept_or_lowered`
-/

namespace Antecedent.Standing

/-- Identification status of a claim, strongest first. -/
inductive Identification where
  | nonparametric
  | parametricRestrictions
  | priorRestrictions
  | partiallyIdentified
  | graphDependent
  | notIdentified
  deriving DecidableEq, Repr

/-- Evidential strength; larger is stronger. -/
def Identification.rank : Identification → Nat
  | .nonparametric => 5
  | .parametricRestrictions => 4
  | .priorRestrictions => 3
  | .partiallyIdentified => 2
  | .graphDependent => 1
  | .notIdentified => 0

theorem Identification.rank_injective {a b : Identification} (h : a.rank = b.rank) : a = b := by
  cases a <;> cases b <;> simp_all [Identification.rank]

/-- Identification standing order: `a ≼ b` when `a` claims no stronger identification. -/
def Identification.le (a b : Identification) : Prop := a.rank ≤ b.rank

instance : DecidableRel Identification.le := fun a b =>
  inferInstanceAs (Decidable (a.rank ≤ b.rank))

theorem Identification.le_refl (a : Identification) : a.le a := Nat.le_refl _

theorem Identification.le_trans {a b c : Identification} (h₁ : a.le b) (h₂ : b.le c) : a.le c :=
  Nat.le_trans h₁ h₂

theorem Identification.le_antisymm {a b : Identification} (h₁ : a.le b) (h₂ : b.le a) : a = b :=
  rank_injective (Nat.le_antisymm h₁ h₂)

/-- Status of a claim at one domain coordinate. -/
inductive Domain where
  | identified
  | supported
  | evaluated
  | outsideScope
  | contradicted
  | unknown
  | limited
  | notClaimed
  deriving DecidableEq, Repr

/-- Whether the status asserts something: identified, supported or evaluated. -/
def Domain.asserts : Domain → Bool
  | .identified | .supported | .evaluated => true
  | _ => false

/--
Domain standing order: `d ≼ p` when `d` is `p` or is `unknown`. Unknown asserts
nothing and is below every status; any two distinct statuses other than unknown
say different things and are incomparable, so moving between them is never a
lowering.
-/
def Domain.le (d p : Domain) : Prop := d = p ∨ d = .unknown

instance : DecidableRel Domain.le := fun d p =>
  inferInstanceAs (Decidable (d = p ∨ d = .unknown))

theorem Domain.le_refl (d : Domain) : d.le d := Or.inl rfl

theorem Domain.le_trans {a b c : Domain} (h₁ : a.le b) (h₂ : b.le c) : a.le c := by
  unfold Domain.le at *
  rcases h₁ with rfl | rfl <;> rcases h₂ with rfl | h <;> simp_all

theorem Domain.le_antisymm {a b : Domain} (h₁ : a.le b) (h₂ : b.le a) : a = b := by
  unfold Domain.le at *
  rcases h₁ with h | h <;> rcases h₂ with h' | h' <;> simp_all

theorem Domain.unknown_le (d : Domain) : Domain.unknown.le d := Or.inr rfl

/-- The relay rule as the Rust check states it: unchanged, or a positive status weakened to unknown. -/
def domainChangeAllowed (fromStatus toStatus : Domain) : Bool :=
  fromStatus == toStatus || (fromStatus.asserts && toStatus == .unknown)

/-- The identification rule as the Rust check states it. -/
def identificationKeptOrLowered (parent derived : Identification) : Bool :=
  decide (derived.rank ≤ parent.rank)

/-- An allowed domain change never raises standing. -/
theorem domainChangeAllowed_le {f t : Domain} (h : domainChangeAllowed f t = true) : t.le f := by
  cases f <;> cases t <;> simp_all [domainChangeAllowed, Domain.asserts, Domain.le]

/-- An allowed domain change asserts only what the parent asserted, at the same status. -/
theorem domainChangeAllowed_asserts {f t : Domain} (h : domainChangeAllowed f t = true)
    (ht : t.asserts = true) : t = f := by
  cases f <;> cases t <;> simp_all [domainChangeAllowed, Domain.asserts]

/-- The identification check accepts exactly the derivations that do not raise identification. -/
theorem identificationKeptOrLowered_iff (p d : Identification) :
    identificationKeptOrLowered p d = true ↔ d.le p := by
  simp [identificationKeptOrLowered, Identification.le]

/-- The three domain axes a claim reports. -/
inductive Axis where
  | identification
  | support
  | evaluated
  deriving DecidableEq, Repr

/--
What a claim asserts. `Content` stands for everything a relay must carry
unchanged (identities, kind, value, units, reasoning, execution, evidence,
calibration, attestation); its structure is irrelevant to standing.
-/
structure Claim (Content : Type) where
  content : Content
  identification : Identification
  domains : Axis → Domain

/-- Standing order on claims: no stronger identification, and no domain raised. -/
def Claim.le {C : Type} (c p : Claim C) : Prop :=
  c.identification.le p.identification ∧ ∀ a, (c.domains a).le (p.domains a)

theorem Claim.le_refl {C : Type} (c : Claim C) : c.le c :=
  ⟨Identification.le_refl _, fun _ => Domain.le_refl _⟩

theorem Claim.le_trans {C : Type} {a b c : Claim C} (h₁ : a.le b) (h₂ : b.le c) : a.le c :=
  ⟨Identification.le_trans h₁.1 h₂.1, fun x => Domain.le_trans (h₁.2 x) (h₂.2 x)⟩

/--
The relay (`ClaimOperation::Retarget`) acceptance rule of `compose_claims`:
content and identification unchanged, and each domain axis changed only as
`domainChangeAllowed` permits.
-/
def relayAccepted {C : Type} [DecidableEq C] (parent derived : Claim C) : Bool :=
  decide (derived.content = parent.content) &&
  decide (derived.identification = parent.identification) &&
  domainChangeAllowed (parent.domains .identification) (derived.domains .identification) &&
  domainChangeAllowed (parent.domains .support) (derived.domains .support) &&
  domainChangeAllowed (parent.domains .evaluated) (derived.domains .evaluated)

/-- An accepted relay never raises its parent's standing. -/
theorem relay_le {C : Type} [DecidableEq C] {p d : Claim C}
    (h : relayAccepted p d = true) : d.le p := by
  simp only [relayAccepted, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨_, hid⟩, hi⟩, hs⟩, he⟩ := h
  refine ⟨hid ▸ Identification.le_refl _, fun a => ?_⟩
  cases a
  · exact domainChangeAllowed_le hi
  · exact domainChangeAllowed_le hs
  · exact domainChangeAllowed_le he

/--
The identification rule of a checked derivation from several parents
(`identification_strengthened`): the derived status is kept or lowered with
respect to every parent.
-/
def derivationIdentificationAccepted {C : Type} (parents : List (Claim C)) (derived : Claim C) : Bool :=
  parents.all fun p => identificationKeptOrLowered p.identification derived.identification

/-- A checked derivation claims no stronger identification than its weakest parent. -/
theorem derivation_identification_le {C : Type} {ps : List (Claim C)} {d : Claim C}
    (h : derivationIdentificationAccepted ps d = true) :
    ∀ p ∈ ps, d.identification.le p.identification := by
  intro p hp
  simp only [derivationIdentificationAccepted, List.all_eq_true] at h
  exact (identificationKeptOrLowered_iff _ _).1 (h p hp)

/-- A chain of accepted relays, each step relaying the claim before it. -/
inductive RelayChain {C : Type} [DecidableEq C] : Claim C → Claim C → Prop where
  | refl (c : Claim C) : RelayChain c c
  | step {a b c : Claim C} : RelayChain a b → relayAccepted b c = true → RelayChain a c

/-- No finite sequence of accepted relays raises the original claim's standing. -/
theorem relayChain_le {C : Type} [DecidableEq C] {a c : Claim C} (h : RelayChain a c) : c.le a := by
  induction h with
  | refl => exact Claim.le_refl _
  | step _ hbc ih => exact Claim.le_trans (relay_le hbc) ih

/--
No chain of accepted relays turns a status that does not assert anything into
one that does: a relay cannot manufacture a positive domain status.
-/
theorem relayChain_no_new_assertion {C : Type} [DecidableEq C] {a c : Claim C}
    (h : RelayChain a c) (x : Axis) (hc : (c.domains x).asserts = true) :
    c.domains x = a.domains x := by
  rcases (relayChain_le h).2 x with h' | h'
  · exact h'
  · simp [h', Domain.asserts] at hc

end Antecedent.Standing
