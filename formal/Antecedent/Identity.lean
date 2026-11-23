/-
Scientific identity under reordering of order-insensitive declarations.

An identity is a hash of a canonical word stream. For a declaration list that
the identity rules call order-insensitive (the sources of a pooling, the
populations and assumptions of a transport), the stream holds the list's
length followed by its elements in sorted order. This file proves, for any
total order on elements:

* reordering such a list never changes the stream (`encodeSet_perm`);
* any other change to it, an element added, removed or replaced, always changes
  the stream (`encodeSet_inj`): the stream determines the list up to order;
* two such lists written one after the other cannot trade elements
  (`encodeTwo_inj`), so moving a source from the transported list to the native
  list of a transport is a change of identity, not a reordering.

The hash is trusted to be injective on the streams it is given (no collisions);
under that assumption the identities agree exactly when the declarations agree
up to order (`identity_eq_iff`).

Rust counterparts: `write_ids` and `sorted_digests` in
`antecedent_contract::evidence_ledger::operation`, and
`TransportIdentity::encode`. Kani checks the sorting kernel on the production
code for bounded lengths (assurance record `identity.source_set_order`).
-/

namespace Antecedent.Identity

/--
A total order given as a Boolean comparison, the shape `sort_unstable` relies on
through `Ord`: total, transitive and antisymmetric.
-/
structure TotalOrder (α : Type) where
  le : α → α → Bool
  total : ∀ a b, le a b || le b a
  trans : ∀ a b c, le a b → le b c → le a c
  antisymm : ∀ a b, le a b → le b a → a = b

variable {α : Type}

/-- One word of the canonical stream: a length or an element. -/
inductive Word (α : Type) where
  | length (n : Nat)
  | item (a : α)
  deriving DecidableEq

/-- Canonical stream of an order-insensitive list: its length, then its elements sorted. -/
def encodeSet (o : TotalOrder α) (xs : List α) : List (Word α) :=
  Word.length xs.length :: (xs.mergeSort o.le).map Word.item

/-- Two sorted lists that are permutations of each other are equal. -/
theorem eq_of_perm_of_sorted (o : TotalOrder α) :
    ∀ {l₁ l₂ : List α}, l₁.Perm l₂ →
      l₁.Pairwise (fun a b => o.le a b) → l₂.Pairwise (fun a b => o.le a b) → l₁ = l₂
  | [], l₂, hp, _, _ => List.Perm.nil_eq hp
  | a :: l₁, [], hp, _, _ => absurd hp.length_eq (by simp)
  | a :: l₁, b :: l₂, hp, h₁, h₂ => by
    have hab : a = b := by
      have ha : a ∈ b :: l₂ := hp.subset (List.mem_cons_self ..)
      have hb : b ∈ a :: l₁ := hp.symm.subset (List.mem_cons_self ..)
      rcases List.mem_cons.1 ha with h | h
      · exact h
      rcases List.mem_cons.1 hb with h' | h'
      · exact h'.symm
      exact o.antisymm a b (List.rel_of_pairwise_cons h₁ h') (List.rel_of_pairwise_cons h₂ h)
    subst hab
    rw [eq_of_perm_of_sorted o ((List.perm_cons a).1 hp) (List.Pairwise.of_cons h₁)
      (List.Pairwise.of_cons h₂)]

theorem sorted_mergeSort' (o : TotalOrder α) (xs : List α) :
    (xs.mergeSort o.le).Pairwise (fun a b => o.le a b) :=
  List.pairwise_mergeSort (le := o.le) (fun a b c h₁ h₂ => o.trans a b c h₁ h₂)
    (fun a b => o.total a b) xs

/-- Sorting a permutation gives the same list. -/
theorem mergeSort_perm_eq (o : TotalOrder α) {xs ys : List α} (h : xs.Perm ys) :
    xs.mergeSort o.le = ys.mergeSort o.le :=
  eq_of_perm_of_sorted o
    (((List.mergeSort_perm xs o.le).trans h).trans (List.mergeSort_perm ys o.le).symm)
    (sorted_mergeSort' o xs) (sorted_mergeSort' o ys)

/-- Reordering an order-insensitive declaration list never changes its stream. -/
theorem encodeSet_perm (o : TotalOrder α) {xs ys : List α} (h : xs.Perm ys) :
    encodeSet o xs = encodeSet o ys := by
  simp [encodeSet, h.length_eq, mergeSort_perm_eq o h]

/-- Equal streams mean the lists hold the same elements with the same multiplicities. -/
theorem encodeSet_inj (o : TotalOrder α) {xs ys : List α}
    (h : encodeSet o xs = encodeSet o ys) : xs.Perm ys := by
  simp only [encodeSet, List.cons.injEq, Word.length.injEq] at h
  have hs : xs.mergeSort o.le = ys.mergeSort o.le :=
    (List.map_inj_right (fun a b h => by cases h; rfl)).1 h.2
  exact ((List.mergeSort_perm xs o.le).symm.trans (hs ▸ List.Perm.refl _)).trans
    (List.mergeSort_perm ys o.le)

/-- The stream of an order-insensitive list determines it up to order. -/
theorem encodeSet_eq_iff (o : TotalOrder α) (xs ys : List α) :
    encodeSet o xs = encodeSet o ys ↔ xs.Perm ys :=
  ⟨encodeSet_inj o, encodeSet_perm o⟩

/-- Every word stream of `encodeSet` starts with the length its elements fill. -/
theorem encodeSet_length (o : TotalOrder α) (xs : List α) :
    (encodeSet o xs).length = xs.length + 1 := by
  simp [encodeSet, List.length_mergeSort]

/-- Two order-insensitive lists, each framed by its length, written one after the other. -/
def encodeTwo (o : TotalOrder α) (xs ys : List α) : List (Word α) :=
  encodeSet o xs ++ encodeSet o ys

/--
Two framed lists cannot trade elements: equal streams mean each list is the
same up to order. Moving a source between the transported and native lists of a
transport changes its identity.
-/
theorem encodeTwo_inj (o : TotalOrder α) {xs ys xs' ys' : List α}
    (h : encodeTwo o xs ys = encodeTwo o xs' ys') : xs.Perm xs' ∧ ys.Perm ys' := by
  unfold encodeTwo at h
  -- The leading length word fixes how many words the first list fills.
  have hlen : xs.length = xs'.length := by
    simp only [encodeSet, List.cons_append, List.cons.injEq, Word.length.injEq] at h
    exact h.1
  have hfirst : (encodeSet o xs).length = (encodeSet o xs').length := by
    simp [encodeSet_length, hlen]
  obtain ⟨h₁, h₂⟩ := List.append_inj h hfirst
  exact ⟨encodeSet_inj o h₁, encodeSet_inj o h₂⟩

/-- Identity of a declaration list under a hash trusted to be collision-free on streams. -/
def identity {D : Type} (hash : List (Word α) → D) (o : TotalOrder α) (xs : List α) : D :=
  hash (encodeSet o xs)

/--
With a collision-free hash, two order-insensitive declaration lists have the
same identity exactly when they are the same up to order: a permitted
reordering keeps the identity and every relevant change breaks it.
-/
theorem identity_eq_iff {D : Type} (hash : List (Word α) → D)
    (hinj : ∀ s t, hash s = hash t → s = t) (o : TotalOrder α) (xs ys : List α) :
    identity hash o xs = identity hash o ys ↔ xs.Perm ys :=
  ⟨fun h => encodeSet_inj o (hinj _ _ h), fun h => congrArg hash (encodeSet_perm o h)⟩

end Antecedent.Identity
