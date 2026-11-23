/-
Artifact acceptance against an independently retained expected identity.

A consumer binding a presented identity (target, program, source, inference or
claim) compares it with the identity it retained itself. The presented identity
may also describe itself, advertising a digest of its own facets; that
advertisement is only ever checked against the consumer's recomputation, never
used in place of the retained expectation.

This file states `bind` as `antecedent_io_core::bound_identity::bind` makes it,
with the facet digest an arbitrary function (no property of the hash is used),
and proves:

* without a retained expectation nothing is ever `bound` (`bind_none_not_bound`);
* `bound` means the slot matched and every facet equals the retained expectation
  (`bind_bound_sound`), whatever the self-description says;
* a presentation whose facets equal the expectation, in the right slot, with no
  false self-description, is bound (`bind_complete`);
* the self-description can only cause a refusal: changing it never turns a
  refusal into `bound` or changes which facets are compared (`bind_self_description_only_refuses`).

Kani checks the same properties on the production `bind` (assurance record
`acceptance.expected_identity`).
-/

namespace Antecedent.Acceptance

/-- The identity layer a consumer binds. -/
inductive Slot where
  | target | program | source | inference | claim
  deriving DecidableEq, Repr

/-- The cyclic-solution facet. -/
inductive Cyclic (D : Type) where
  | absent
  | tag (d : D)
  deriving DecidableEq

/-- The compared facets of an identity. -/
structure Facets (D : Type) where
  payload : D
  coordinates : D
  prior : D
  alignment : D
  cyclic : Cyclic D
  deriving DecidableEq

/-- A presented identity: its slot, facets, and an optional advertised digest of its facets. -/
structure Presented (D : Type) where
  slot : Slot
  facets : Facets D
  selfDescription : Option D

/-- Outcome of a binding. -/
inductive Outcome where
  | bound
  | notRequired
  | slotMismatch
  | falseSelfDescription
  | payloadMismatch
  | coordinatesMismatch
  | priorMismatch
  | alignmentMismatch
  | cyclicMismatch
  deriving DecidableEq, Repr

variable {D : Type} [DecidableEq D]

/-- Whether the advertised digest disagrees with the consumer's recomputation. -/
def falselySelfDescribed (digest : Slot → Facets D → D) (p : Presented D) : Bool :=
  match p.selfDescription with
  | some d => d != digest p.slot p.facets
  | none => false

/-- `bind` as the production code checks it, in the same order. -/
def bind (digest : Slot → Facets D → D) (required : Slot) (p : Presented D)
    (expected : Option (Facets D)) : Outcome :=
  if p.slot ≠ required then .slotMismatch
  else if falselySelfDescribed digest p then .falseSelfDescription
  else match expected with
    | none => .notRequired
    | some e =>
      if p.facets.payload ≠ e.payload then .payloadMismatch
      else if p.facets.coordinates ≠ e.coordinates then .coordinatesMismatch
      else if p.facets.prior ≠ e.prior then .priorMismatch
      else if p.facets.alignment ≠ e.alignment then .alignmentMismatch
      else if p.facets.cyclic ≠ e.cyclic then .cyclicMismatch
      else .bound

/-- `bound` exactly when the slot matches, the self-description is not false, and the
retained expectation is present and equal to the presented facets. -/
theorem bind_eq_bound_iff (digest : Slot → Facets D → D) (r : Slot) (p : Presented D)
    (expected : Option (Facets D)) :
    bind digest r p expected = .bound ↔
      p.slot = r ∧ falselySelfDescribed digest p = false ∧ expected = some p.facets := by
  obtain ⟨slot, ⟨pa, co, pr, al, cy⟩, sd⟩ := p
  by_cases hs : slot = r
  · subst hs
    cases hd : falselySelfDescribed digest ⟨slot, ⟨pa, co, pr, al, cy⟩, sd⟩
    · cases expected with
      | none => simp [bind, hd]
      | some e =>
        obtain ⟨pa', co', pr', al', cy'⟩ := e
        rcases Decidable.em (pa = pa') with rfl | h1 <;> rcases Decidable.em (co = co') with rfl | h2 <;>
          rcases Decidable.em (pr = pr') with rfl | h3 <;> rcases Decidable.em (al = al') with rfl | h4 <;>
          rcases Decidable.em (cy = cy') with rfl | h5 <;>
          simp_all [bind, Ne.symm]
    · simp [bind, hd]
  · simp [bind, hs]

/-- Without a retained expectation, nothing is ever bound. -/
theorem bind_none_not_bound (digest : Slot → Facets D → D) (r : Slot) (p : Presented D) :
    bind digest r p none ≠ .bound := by
  simp [bind_eq_bound_iff]

/-- `bound` means the slot matched and every facet equals the retained expectation. -/
theorem bind_bound_sound {digest : Slot → Facets D → D} {r : Slot} {p : Presented D}
    {expected : Option (Facets D)} (h : bind digest r p expected = .bound) :
    p.slot = r ∧ ∃ e, expected = some e ∧ p.facets = e := by
  obtain ⟨hs, _, he⟩ := (bind_eq_bound_iff digest r p expected).1 h
  exact ⟨hs, p.facets, he, rfl⟩

/-- A presentation that matches the retained expectation and does not lie about itself is bound. -/
theorem bind_complete (digest : Slot → Facets D → D) (r : Slot) (p : Presented D)
    (hs : p.slot = r) (hd : falselySelfDescribed digest p = false) :
    bind digest r p (some p.facets) = .bound := by
  simp [bind, hs, hd]

/--
The self-description can only refuse. If a presentation with some
self-description is bound, the same presentation with none is bound too, and
the expectation is compared the same way.
-/
theorem bind_self_description_only_refuses (digest : Slot → Facets D → D) (r : Slot)
    (p : Presented D) (expected : Option (Facets D))
    (h : bind digest r p expected = .bound) :
    bind digest r { p with selfDescription := none } expected = .bound := by
  obtain ⟨hs, e, rfl, hf⟩ := bind_bound_sound h
  subst hf
  simp [bind, hs, falselySelfDescribed]

/--
No self-description, however chosen, makes a mismatched presentation bound:
binding depends on the retained expectation, which the presented identity
cannot supply.
-/
theorem bind_mismatch_never_bound (digest : Slot → Facets D → D) (r : Slot) (p : Presented D)
    (e : Facets D) (hne : p.facets ≠ e) (sd : Option D) :
    bind digest r { p with selfDescription := sd } (some e) ≠ .bound := by
  intro h
  obtain ⟨_, e', he, hf⟩ := bind_bound_sound h
  cases he
  exact hne hf

end Antecedent.Acceptance
