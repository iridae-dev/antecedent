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
* `domainUpgraded`, `domainDeparts` ↔ `antecedent_contract::claim::{domain_upgraded, domain_departs}`
* `associationalDerivesCausal`, `nonValueDerivesValue` ↔
  `antecedent_contract::claim::{associational_derives_causal, non_value_derives_value}`
* `identificationSlotAdded` ↔ `antecedent_contract::claim::identification_slot_added`
* `chainRetains`         ↔ `antecedent_contract::claim::chain_retains`

`derivationAccepted` states `derived_departs_from_parents` for every operation other
than a relay, and `Receipt.chain` states `HandoffReceipt::chain`; both are models over
lists that call the translated rules.
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

/-! ## Every other derivation: kind, identification slot and domains -/

/-- Kind of value a claim carries (`ClaimKind`). -/
inductive Kind where
  | point
  | bounds
  | mixture
  | response
  | policyValue
  | refusal
  | incomplete
  | associational
  deriving DecidableEq, Repr

/--
Standing of a kind: a refusal or an incomplete claim carries no value, an
association a noncausal value, and every other kind a causal value.
-/
def Kind.level : Kind → Nat
  | .refusal | .incomplete => 0
  | .associational => 1
  | _ => 2

/-- Whether the kind carries no value: a refusal or an incomplete claim. -/
def Kind.noValue (k : Kind) : Bool := k == .refusal || k == .incomplete

/-- The association rule as the Rust check states it. -/
def associationalDerivesCausal (parent derived : Kind) : Bool :=
  parent == .associational && !(derived == .associational || derived.noValue)

/-- The no-value rule as the Rust check states it. -/
def nonValueDerivesValue (parent derived : Kind) : Bool :=
  parent.noValue && !derived.noValue

/-- The two kind rules refuse exactly the derivations that raise the kind's standing. -/
theorem kind_rules_iff (p d : Kind) :
    (associationalDerivesCausal p d = false ∧ nonValueDerivesValue p d = false) ↔
      d.level ≤ p.level := by
  cases p <;> cases d <;> decide

/-- The upgrade rule as the Rust check states it: a positive status the parent did not hold. -/
def domainUpgraded (parent derived : Domain) : Bool :=
  derived.asserts && parent != derived

/-- The departure rule as the Rust check states it: neither the parent's status nor unknown. -/
def domainDeparts (parent derived : Domain) : Bool :=
  derived != parent && derived != .unknown

/-- The departure rule refuses exactly the domain changes that are not a lowering. -/
theorem domainDeparts_iff (p d : Domain) : domainDeparts p d = false ↔ d.le p := by
  cases p <;> cases d <;> decide

/-- Every upgrade is a departure. -/
theorem domainUpgraded_departs {p d : Domain} (h : domainUpgraded p d = true) :
    domainDeparts p d = true := by
  revert h
  cases p <;> cases d <;> decide

/-- The slot rule as the Rust check states it: a slot some parent did not report. -/
def identificationSlotAdded (everyParentHasSlot derivedHasSlot : Bool) : Bool :=
  derivedHasSlot && !everyParentHasSlot

/-- What a derivation's standing rests on: kind, identification slot and domains. -/
structure Derivable where
  kind : Kind
  slot : Option Identification
  domains : Axis → Domain

/--
Slot order: reporting no identification slot asserts nothing and is below every
slot; a slot is never below no slot (adding one asserts identification no parent
asserted); two slots compare by identification.
-/
def slotLe : Option Identification → Option Identification → Prop
  | none, _ => True
  | some d, some p => d.le p
  | some _, none => False

theorem slotLe_refl : ∀ s, slotLe s s
  | none => trivial
  | some s => Identification.le_refl s

theorem slotLe_trans : ∀ {a b c : Option Identification}, slotLe a b → slotLe b c → slotLe a c
  | none, _, _, _, _ => trivial
  | some _, some _, some _, h₁, h₂ => Identification.le_trans h₁ h₂
  | some _, none, _, h, _ => h.elim
  | some _, some _, none, _, h => h.elim

/-- Standing order on derivations: no stronger kind, slot or domain. -/
def Derivable.le (d p : Derivable) : Prop :=
  d.kind.level ≤ p.kind.level ∧ slotLe d.slot p.slot ∧ ∀ a, (d.domains a).le (p.domains a)

theorem Derivable.le_refl (d : Derivable) : d.le d :=
  ⟨Nat.le_refl _, slotLe_refl _, fun _ => Domain.le_refl _⟩

theorem Derivable.le_trans {a b c : Derivable} (h₁ : a.le b) (h₂ : b.le c) : a.le c :=
  ⟨Nat.le_trans h₁.1 h₂.1, slotLe_trans h₁.2.1 h₂.2.1, fun x => Domain.le_trans (h₁.2.2 x) (h₂.2.2 x)⟩

/-- The three axes, in the order the Rust checks visit them. -/
def axes : List Axis := [.identification, .support, .evaluated]

theorem mem_axes (a : Axis) : a ∈ axes := by
  cases a <;> simp [axes]

/--
The identification-slot rule of `identification_strengthened`: a derived claim
without a slot is refused when a parent reported one, and a derived slot is no
stronger than any parent's.
-/
def slotAccepted (parents : List Derivable) : Option Identification → Bool
  | none => parents.all fun p => p.slot.isNone
  | some s => parents.all fun p =>
      match p.slot with
      | some ps => identificationKeptOrLowered ps s
      | none => true

/--
`derived_departs_from_parents` for an operation other than a relay, in the order
the Rust checks apply. The identified-mass checks are left out: they only refuse
more, so every derivation the Rust function accepts is accepted here.
-/
def derivationAccepted (parents : List Derivable) (d : Derivable) : Bool :=
  (parents.all fun p => !associationalDerivesCausal p.kind d.kind) &&
  (parents.all fun p => !nonValueDerivesValue p.kind d.kind) &&
  !identificationSlotAdded (parents.all fun p => p.slot.isSome) d.slot.isSome &&
  slotAccepted parents d.slot &&
  (parents.all fun p => axes.all fun a => !domainUpgraded (p.domains a) (d.domains a)) &&
  (parents.all fun p => axes.all fun a => !domainDeparts (p.domains a) (d.domains a))

theorem slotAccepted_le {ps : List Derivable} {d : Derivable}
    (hadd : identificationSlotAdded (ps.all fun p => p.slot.isSome) d.slot.isSome = false)
    (hslot : slotAccepted ps d.slot = true) : ∀ p ∈ ps, slotLe d.slot p.slot := by
  intro p hp
  cases hd : d.slot with
  | none =>
    rw [hd] at hslot
    simp only [slotAccepted, List.all_eq_true] at hslot
    have h := hslot p hp
    cases hps : p.slot with
    | none => trivial
    | some _ => simp [hps] at h
  | some s =>
    rw [hd] at hslot hadd
    have hall : (ps.all fun p => p.slot.isSome) = true := by
      revert hadd
      cases (ps.all fun p => p.slot.isSome) <;> simp [identificationSlotAdded]
    rw [List.all_eq_true] at hall
    simp only [slotAccepted, List.all_eq_true] at hslot
    have hs := hslot p hp
    cases hps : p.slot with
    | none =>
      have := hall p hp
      simp [hps] at this
    | some ps =>
      rw [hps] at hs
      exact (identificationKeptOrLowered_iff ps s).1 hs

/-- A derivation the rules accept is no stronger than any of its parents. -/
theorem derivation_le {ps : List Derivable} {d : Derivable}
    (h : derivationAccepted ps d = true) : ∀ p ∈ ps, d.le p := by
  intro p hp
  simp only [derivationAccepted, Bool.and_eq_true, Bool.not_eq_true', List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨hassoc, hval⟩, hadd⟩, hslot⟩, _⟩, hdep⟩ := h
  refine ⟨(kind_rules_iff p.kind d.kind).1 ⟨?_, ?_⟩, slotAccepted_le hadd hslot p hp, fun a => ?_⟩
  · simpa using hassoc p hp
  · simpa using hval p hp
  · exact (domainDeparts_iff _ _).1 (by simpa using hdep p hp a (mem_axes a))

/--
`d` descends from `a` through accepted derivations: it is `a`, or it was derived
from a list of parents one of which descends from `a`.
-/
inductive Descends : Derivable → Derivable → Prop where
  | refl (c : Derivable) : Descends c c
  | step {ps : List Derivable} {d p a : Derivable} :
      derivationAccepted ps d = true → p ∈ ps → Descends p a → Descends d a

/-- No sequence of accepted derivations raises the standing of any claim it started from. -/
theorem descends_le {d a : Derivable} (h : Descends d a) : d.le a := by
  induction h with
  | refl => exact Derivable.le_refl _
  | step hacc hp _ ih => exact Derivable.le_trans (derivation_le hacc _ hp) ih

/-- A derivation never asserts a positive domain status some parent did not hold at that status. -/
theorem derivation_no_new_assertion {ps : List Derivable} {d : Derivable}
    (h : derivationAccepted ps d = true) (x : Axis) (hx : (d.domains x).asserts = true) :
    ∀ p ∈ ps, d.domains x = p.domains x := by
  intro p hp
  rcases (derivation_le h p hp).2.2 x with h' | h'
  · exact h'
  · simp [h', Domain.asserts] at hx

/-! ## The checked composer -/

/--
The parent admission of `compose_checked_study_claims` (facade, not translated): a
point claim, identified nonparametrically or under parametric restrictions, with
identification domain identified or unknown, support domain supported or unknown,
and evaluated domain evaluated.
-/
def checkedParentAdmitted (p : Derivable) : Bool :=
  p.kind == .point &&
  (p.slot == some .nonparametric || p.slot == some .parametricRestrictions) &&
  (p.domains .identification == .identified || p.domains .identification == .unknown) &&
  (p.domains .support == .supported || p.domains .support == .unknown) &&
  p.domains .evaluated == .evaluated

/--
The claim `finish_checked_claim` builds: a point value with no identification slot,
identification and support unknown, evaluated.
-/
def checkedDerived : Derivable where
  kind := .point
  slot := none
  domains
    | .identification => .unknown
    | .support => .unknown
    | .evaluated => .evaluated

/-- The checked composer's claim is no stronger than any parent it admits. -/
theorem checked_composer_le {p : Derivable} (h : checkedParentAdmitted p = true) :
    checkedDerived.le p := by
  simp only [checkedParentAdmitted, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨hk, _⟩, _⟩, _⟩, he⟩ := h
  refine ⟨by simp [checkedDerived, hk, Kind.level], trivial, fun a => ?_⟩
  cases a
  · exact Domain.unknown_le _
  · exact Domain.unknown_le _
  · exact Or.inl he.symm

/-! ## Handoff receipts -/

section Handoff

variable {Id F : Type} [DecidableEq Id] [DecidableEq F]

/-- A handoff receipt (`HandoffReceipt`): which claim went in and out, and what was kept. -/
structure Receipt (Id F : Type) where
  input : Id
  output : Option Id
  retained : List F
  omitted : List F
  unresolved : List F
  unavailable : List F

/-- `merge_unique`: `l` followed by the elements of `r` it does not contain yet. -/
def mergeUnique (l r : List F) : List F :=
  r.foldl (fun acc x => if x ∈ acc then acc else acc ++ [x]) l

theorem mergeUnique_cons (l : List F) (y : F) (r : List F) :
    mergeUnique l (y :: r) = mergeUnique (if y ∈ l then l else l ++ [y]) r := rfl

theorem mem_mergeUnique (l r : List F) (x : F) : x ∈ mergeUnique l r ↔ x ∈ l ∨ x ∈ r := by
  induction r generalizing l with
  | nil => simp [mergeUnique]
  | cons y r ih =>
    rw [mergeUnique_cons, ih]
    by_cases hy : y ∈ l
    · simp only [hy, ↓reduceIte, List.mem_cons]
      constructor
      · rintro (h | h)
        · exact Or.inl h
        · exact Or.inr (Or.inr h)
      · rintro (h | h | h)
        · exact Or.inl h
        · exact Or.inl (h ▸ hy)
        · exact Or.inr h
    · simp only [hy, ↓reduceIte, List.mem_append, List.mem_cons, List.not_mem_nil, or_false]
      exact or_assoc

/-- `equivalent_claim`: nothing omitted or unresolved, the same claim out, every required field kept. -/
def Receipt.equivalent (req : List F) (r : Receipt Id F) : Bool :=
  r.omitted.isEmpty && r.unresolved.isEmpty && r.output == some r.input &&
    req.all fun f => decide (f ∈ r.retained)

/-- The retention rule of a two-hop chain as the Rust check states it. -/
def chainRetains (contiguous firstComplete nextComplete retainedByFirst omitted : Bool) : Bool :=
  contiguous && (firstComplete || nextComplete || retainedByFirst) && !omitted

/--
`HandoffReceipt::chain` with required fields `req` and the discontinuity marker
`disc`: losses merged, fields retained by `chainRetains`.
-/
def Receipt.chain (req : List F) (disc : F) (a b : Receipt Id F) : Receipt Id F :=
  let omitted := mergeUnique a.omitted b.omitted
  let contiguous := a.output == some b.input
  let fc := a.equivalent req
  let nc := b.equivalent req
  let candidates := if nc then a.retained else b.retained
  let unresolved := mergeUnique a.unresolved b.unresolved
  { input := a.input
    output := b.output
    retained := candidates.filter fun f =>
      chainRetains contiguous fc nc (decide (f ∈ a.retained)) (decide (f ∈ omitted))
    omitted := omitted
    unresolved := if contiguous then unresolved else mergeUnique unresolved [disc]
    unavailable := mergeUnique a.unavailable b.unavailable }

variable (req : List F) (disc : F) (a b : Receipt Id F)

/-- A chain omits exactly what either hop omitted. -/
theorem chain_omitted (x : F) :
    x ∈ (a.chain req disc b).omitted ↔ x ∈ a.omitted ∨ x ∈ b.omitted :=
  mem_mergeUnique _ _ _

/-- A chain keeps every unresolved item of either hop. -/
theorem chain_keeps_unresolved (x : F) (hx : x ∈ a.unresolved ∨ x ∈ b.unresolved) :
    x ∈ (a.chain req disc b).unresolved := by
  simp only [Receipt.chain]
  split
  · exact (mem_mergeUnique _ _ _).2 hx
  · exact (mem_mergeUnique _ _ _).2 (Or.inl ((mem_mergeUnique _ _ _).2 hx))

/-- A chain keeps every operation either hop made unavailable. -/
theorem chain_unavailable (x : F) :
    x ∈ (a.chain req disc b).unavailable ↔ x ∈ a.unavailable ∨ x ∈ b.unavailable :=
  mem_mergeUnique _ _ _

/-- A later hop cannot restore a field an earlier hop lost: nothing retained is omitted. -/
theorem chain_retained_not_omitted (x : F) (hx : x ∈ (a.chain req disc b).retained) :
    x ∉ (a.chain req disc b).omitted := by
  simp only [Receipt.chain, List.mem_filter, chainRetains, Bool.and_eq_true, Bool.not_eq_true',
    decide_eq_false_iff_not] at hx ⊢
  exact hx.2.2

/-- A chain invents no field: whatever it retains, one of its hops declared retained. -/
theorem chain_retained_declared (x : F) (hx : x ∈ (a.chain req disc b).retained) :
    x ∈ a.retained ∨ x ∈ b.retained := by
  simp only [Receipt.chain, List.mem_filter] at hx
  split at hx
  · exact Or.inl hx.1
  · exact Or.inr hx.1

/-- Receipts about different claims retain nothing and record the discontinuity. -/
theorem chain_discontinuous (h : a.output ≠ some b.input) :
    (a.chain req disc b).retained = [] ∧ disc ∈ (a.chain req disc b).unresolved := by
  have hc : (a.output == some b.input) = false := by simpa using h
  refine ⟨?_, ?_⟩
  · simp [Receipt.chain, hc, chainRetains]
  · simp only [Receipt.chain, hc]
    exact (mem_mergeUnique _ _ _).2 (Or.inr (List.mem_singleton.2 rfl))

/--
A chain is equivalent to its input claim only if neither hop omitted or left
anything unresolved, the hops describe one path, and it ends at the claim it
started from.
-/
theorem chain_equivalent (h : (a.chain req disc b).equivalent req = true) :
    a.omitted = [] ∧ b.omitted = [] ∧ a.unresolved = [] ∧ b.unresolved = [] ∧
      a.output = some b.input ∧ b.output = some a.input := by
  simp only [Receipt.equivalent, Bool.and_eq_true, List.isEmpty_iff, beq_iff_eq] at h
  obtain ⟨⟨⟨homit, hunres⟩, hout⟩, _⟩ := h
  have hnone : ∀ {l : List F}, (∀ x, x ∉ l) → l = [] := fun {l} hl => by
    cases l with
    | nil => rfl
    | cons y _ => exact absurd (by simp) (hl y)
  have hcont : a.output = some b.input := by
    cases hc : (a.output == some b.input) with
    | true => simpa using hc
    | false =>
      have hne : a.output ≠ some b.input := by simpa using hc
      have := (chain_discontinuous req disc a b hne).2
      rw [hunres] at this
      simp at this
  have hc : (a.output == some b.input) = true := by simpa using hcont
  have hu : mergeUnique a.unresolved b.unresolved = [] := by
    simpa [Receipt.chain, hc] using hunres
  refine ⟨hnone fun x hx => ?_, hnone fun x hx => ?_, hnone fun x hx => ?_,
    hnone fun x hx => ?_, hcont, hout⟩
  · have := (chain_omitted req disc a b x).2 (Or.inl hx); rw [homit] at this; simp at this
  · have := (chain_omitted req disc a b x).2 (Or.inr hx); rw [homit] at this; simp at this
  · have := (mem_mergeUnique a.unresolved b.unresolved x).2 (Or.inl hx); rw [hu] at this; simp at this
  · have := (mem_mergeUnique a.unresolved b.unresolved x).2 (Or.inr hx); rw [hu] at this; simp at this

/-- A chain of hops composed left to right from a first receipt. -/
def chainAll (first : Receipt Id F) (rest : List (Receipt Id F)) : Receipt Id F :=
  rest.foldl (Receipt.chain req disc) first

/-- Along any chain of hops, everything any hop omitted stays omitted. -/
theorem chainAll_keeps_omitted (first : Receipt Id F) (rest : List (Receipt Id F)) (x : F)
    (hx : x ∈ first.omitted ∨ ∃ r ∈ rest, x ∈ r.omitted) :
    x ∈ (chainAll req disc first rest).omitted := by
  induction rest generalizing first with
  | nil =>
    rcases hx with h | ⟨r, hr, _⟩
    · exact h
    · simp at hr
  | cons r rest ih =>
    apply ih
    rcases hx with h | ⟨r', hr', h'⟩
    · exact Or.inl ((chain_omitted req disc first r x).2 (Or.inl h))
    · rcases List.mem_cons.1 hr' with rfl | hmem
      · exact Or.inl ((chain_omitted req disc first r' x).2 (Or.inr h'))
      · exact Or.inr ⟨r', hmem, h'⟩

/-- A chain of at least two hops ends with a two-hop chain. -/
theorem chainAll_cons (first r : Receipt Id F) (rest : List (Receipt Id F)) :
    ∃ a b : Receipt Id F, chainAll req disc first (r :: rest) = a.chain req disc b := by
  induction rest generalizing first r with
  | nil => exact ⟨first, r, rfl⟩
  | cons s rest ih => exact ih (first.chain req disc r) s

/-- No field any hop of a chain omitted is retained by the chain. -/
theorem chainAll_never_restores (first r : Receipt Id F) (rest : List (Receipt Id F)) (x : F)
    (hx : x ∈ first.omitted ∨ ∃ h ∈ r :: rest, x ∈ h.omitted) :
    x ∉ (chainAll req disc first (r :: rest)).retained := by
  intro hret
  have hmem := chainAll_keeps_omitted req disc first (r :: rest) x hx
  obtain ⟨a', b', he⟩ := chainAll_cons req disc first r rest
  rw [he] at hmem hret
  exact chain_retained_not_omitted req disc a' b' x hret hmem

end Handoff

end Antecedent.Standing
