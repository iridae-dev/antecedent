/-
The canonical framing of identities, and why it is uniquely decodable.

Every identity is a hash of a byte stream built from frames: fixed-width words, byte
strings after their length, presence words, count-prefixed lists, and sets and multisets
written as sorted lists. This file states the framing over any byte type and proves:

* each frame is prefix-free: its bytes can be read back from the front of any longer
  stream, so a frame and whatever follows it are both determined (`PrefixFreeOn`);
* sequencing, a tag or presence word choosing how the rest is read, and count-prefixed
  lists keep a stream prefix-free (`PrefixFreeOn.seq`, `PrefixFreeOn.dseq`,
  `prefixFree_option`, `prefixFree_ordered`), so a framed sequence decodes uniquely;
* an ordered list is order-sensitive (`ordered_swap_ne`);
* a set (sorted, adjacent repeats removed) depends exactly on which elements occur:
  reordering or repeating an element keeps the stream, any other change breaks it
  (`set_eq_iff`, `set_perm`, `set_dup`);
* a multiset (sorted, repeats kept) depends exactly on the elements with their
  multiplicities (`multiset_eq_iff`), so a repeat changes it (`multiset_dup_ne`);
* under a hash that is collision-free on the streams, two values share an identity exactly
  when they are equal (`identity_eq_iff`).

The production kernels (`antecedent_core::framing`) are proved to compute these frames in
`formal/rust/AntecedentRust/Framing.lean`.
-/
import Antecedent.Identity

namespace Antecedent.Framing

open Antecedent.Identity (TotalOrder)

variable {α γ β : Type}

/-- `e` is prefix-free on the values satisfying `P`: from `e a ++ r = e b ++ s` both the value
and the rest of the stream are recovered. -/
def PrefixFreeOn (P : α → Prop) (e : α → List β) : Prop :=
  ∀ a b r s, P a → P b → e a ++ r = e b ++ s → a = b ∧ r = s

/-- A prefix-free encoder is injective. -/
theorem PrefixFreeOn.inj {P : α → Prop} {e : α → List β} (h : PrefixFreeOn P e) {a b : α}
    (ha : P a) (hb : P b) (hab : e a = e b) : a = b :=
  (h a b [] [] ha hb (by simpa using hab)).1

/-- A fixed-width injective encoder is prefix-free. -/
theorem prefixFree_of_width {P : α → Prop} {e : α → List β} (n : Nat)
    (hw : ∀ a, P a → (e a).length = n) (hi : ∀ a b, P a → P b → e a = e b → a = b) :
    PrefixFreeOn P e := by
  intro a b r s ha hb h
  obtain ⟨h₁, h₂⟩ := List.append_inj h (by rw [hw a ha, hw b hb])
  exact ⟨hi a b ha hb h₁, h₂⟩

/-- Two frames, one after the other. -/
def seq (e₁ : α → List β) (e₂ : γ → List β) (p : α × γ) : List β := e₁ p.1 ++ e₂ p.2

theorem PrefixFreeOn.seq {P : α → Prop} {Q : γ → Prop} {e₁ : α → List β} {e₂ : γ → List β}
    (h₁ : PrefixFreeOn P e₁) (h₂ : PrefixFreeOn Q e₂) :
    PrefixFreeOn (fun p => P p.1 ∧ Q p.2) (Framing.seq e₁ e₂) := by
  rintro ⟨a, c⟩ ⟨b, d⟩ r s ⟨ha, hc⟩ ⟨hb, hd⟩ h
  simp only [Framing.seq, List.append_assoc] at h
  obtain ⟨rfl, h'⟩ := h₁ a b _ _ ha hb h
  obtain ⟨rfl, rfl⟩ := h₂ c d r s hc hd h'
  exact ⟨rfl, rfl⟩

/-- A frame that decides how the rest is read (a tag, a variant, a presence word), then the
rest. -/
def dseq {γ : α → Type} (e₁ : α → List β) (e₂ : (a : α) → γ a → List β) (p : Σ a, γ a) :
    List β :=
  e₁ p.1 ++ e₂ p.1 p.2

theorem PrefixFreeOn.dseq {γ : α → Type} {P : α → Prop} {Q : (a : α) → γ a → Prop}
    {e₁ : α → List β} {e₂ : (a : α) → γ a → List β}
    (h₁ : PrefixFreeOn P e₁) (h₂ : ∀ a, PrefixFreeOn (Q a) (e₂ a)) :
    PrefixFreeOn (fun p : Σ a, γ a => P p.1 ∧ Q p.1 p.2) (Framing.dseq e₁ e₂) := by
  rintro ⟨a, c⟩ ⟨b, d⟩ r s ⟨ha, hc⟩ ⟨hb, hd⟩ h
  simp only [Framing.dseq, List.append_assoc] at h
  obtain ⟨rfl, h'⟩ := h₁ a b _ _ ha hb h
  obtain ⟨rfl, rfl⟩ := h₂ a c d r s hc hd h'
  exact ⟨rfl, rfl⟩

/-- Two different leading tags never decode as each other, whatever follows. -/
theorem PrefixFreeOn.tag_separates {P : α → Prop} {e : α → List β} (h : PrefixFreeOn P e)
    {a b : α} (ha : P a) (hb : P b) (hab : a ≠ b) (r s : List β) : e a ++ r ≠ e b ++ s :=
  fun he => hab (h a b r s ha hb he).1

/-- An optional value: its presence word, then the value when present. -/
def option (presence : Bool → List β) (e : α → List β) : Option α → List β
  | none => presence false
  | some a => presence true ++ e a

theorem prefixFree_option {P : α → Prop} {presence : Bool → List β} {e : α → List β}
    (hp : PrefixFreeOn (fun _ => True) presence) (he : PrefixFreeOn P e) :
    PrefixFreeOn (fun o => ∀ a, o = some a → P a) (option presence e) := by
  rintro (_ | a) (_ | b) r s ha hb h <;> simp only [option, List.append_assoc] at h
  · exact ⟨rfl, (hp _ _ _ _ trivial trivial h).2⟩
  · exact absurd (hp _ _ _ _ trivial trivial h).1 (by simp)
  · exact absurd (hp _ _ _ _ trivial trivial h).1 (by simp)
  · obtain ⟨-, h'⟩ := hp _ _ _ _ trivial trivial h
    obtain ⟨rfl, rfl⟩ := he a b r s (ha a rfl) (hb b rfl) h'
    exact ⟨rfl, rfl⟩

/-- The frames of a list's items, one after the other. -/
def items (e : α → List β) (xs : List α) : List β := (xs.map e).flatten

/-- Lists of the same length whose items are prefix-free decode item by item. -/
theorem items_decode {P : α → Prop} {e : α → List β} (he : PrefixFreeOn P e) :
    ∀ (xs ys : List α) (r s : List β), xs.length = ys.length → (∀ x ∈ xs, P x) →
      (∀ y ∈ ys, P y) → items e xs ++ r = items e ys ++ s → xs = ys ∧ r = s
  | [], [], _, _, _, _, _, h => ⟨rfl, by simpa [items] using h⟩
  | [], _ :: _, _, _, hl, _, _, _ => absurd hl (by simp)
  | _ :: _, [], _, _, hl, _, _, _ => absurd hl (by simp)
  | x :: xs, y :: ys, r, s, hl, hx, hy, h => by
    simp only [items, List.map_cons, List.flatten_cons, List.append_assoc] at h
    obtain ⟨rfl, h'⟩ := he x y _ _ (hx x (List.mem_cons_self ..)) (hy y (List.mem_cons_self ..)) h
    obtain ⟨rfl, rfl⟩ := items_decode he xs ys r s (by simpa using hl)
      (fun z hz => hx z (List.mem_cons_of_mem _ hz)) (fun z hz => hy z (List.mem_cons_of_mem _ hz)) h'
    exact ⟨rfl, rfl⟩

/-- An ordered list: its count, then each item's frame in the order given. -/
def ordered (count : Nat → List β) (e : α → List β) (xs : List α) : List β :=
  count xs.length ++ items e xs

/-- A count-prefixed list of prefix-free items is prefix-free, for counts the count frame
can hold. -/
theorem prefixFree_ordered {bound : Nat} {P : α → Prop} {count : Nat → List β}
    {e : α → List β} (hc : PrefixFreeOn (· < bound) count) (he : PrefixFreeOn P e) :
    PrefixFreeOn (fun xs : List α => xs.length < bound ∧ ∀ x ∈ xs, P x) (ordered count e) := by
  intro xs ys r s ⟨hxl, hx⟩ ⟨hyl, hy⟩ h
  simp only [ordered, List.append_assoc] at h
  obtain ⟨hl, h'⟩ := hc _ _ _ _ hxl hyl h
  exact items_decode he xs ys r s hl hx hy h'

/-- Ordered framing is order-sensitive: exchanging two different items changes the stream. -/
theorem ordered_swap_ne {bound : Nat} {P : α → Prop} {count : Nat → List β}
    {e : α → List β} (hc : PrefixFreeOn (· < bound) count) (he : PrefixFreeOn P e)
    {a b : α} {xs : List α} (hab : a ≠ b) (hl : xs.length + 2 < bound) (ha : P a) (hb : P b)
    (hx : ∀ x ∈ xs, P x) : ordered count e (a :: b :: xs) ≠ ordered count e (b :: a :: xs) := by
  intro h
  have hP : ∀ (u v : α), P u → P v → ∀ x ∈ u :: v :: xs, P x := by
    intro u v hu hv x hm
    simp only [List.mem_cons] at hm
    rcases hm with rfl | rfl | hm
    · exact hu
    · exact hv
    · exact hx x hm
  have := (prefixFree_ordered hc he).inj ⟨by simpa using hl, hP a b ha hb⟩
    ⟨by simpa using hl, hP b a hb ha⟩ h
  exact hab (List.cons.inj this).1

/-! ## Sets and multisets -/

/-- Remove each element equal to the element just before it, as the production loop does:
on a sorted list, every repeat. -/
def dedupAfter [DecidableEq α] : Option α → List α → List α
  | _, [] => []
  | p, x :: xs => if p = some x then dedupAfter (some x) xs else x :: dedupAfter (some x) xs

/-- A sorted list with its repeats removed. -/
def dedupSorted [DecidableEq α] (xs : List α) : List α := dedupAfter none xs

theorem dedupAfter_cons_same [DecidableEq α] (x : α) (xs : List α) :
    dedupAfter (some x) (x :: xs) = dedupAfter (some x) xs := by
  simp [dedupAfter]

theorem dedupAfter_cons_ne [DecidableEq α] {p : Option α} {x : α} (xs : List α)
    (h : p ≠ some x) : dedupAfter p (x :: xs) = x :: dedupAfter (some x) xs := by
  simp [dedupAfter, h]

/-- On a sorted list whose elements are all at least `q`, removing the repeats after `q`
keeps exactly the elements other than `q`. -/
theorem mem_dedupAfter_some [DecidableEq α] (o : TotalOrder α) :
    ∀ (q : α) (xs : List α), xs.Pairwise (fun a b => o.le a b) → (∀ z ∈ xs, o.le q z) →
      ∀ y, y ∈ dedupAfter (some q) xs ↔ y ∈ xs ∧ y ≠ q
  | _, [], _, _, _ => by simp [dedupAfter]
  | q, x :: t, hs, hq, y => by
    have ht : t.Pairwise (fun a b => o.le a b) := List.Pairwise.of_cons hs
    have hx : ∀ z ∈ t, o.le x z := fun z hz => List.rel_of_pairwise_cons hs hz
    have ih := mem_dedupAfter_some o x t ht hx y
    by_cases hqx : q = x
    · subst hqx
      rw [dedupAfter_cons_same, ih, List.mem_cons]
      constructor
      · rintro ⟨h, hne⟩; exact ⟨Or.inr h, hne⟩
      · rintro ⟨h | h, hne⟩
        · exact absurd h hne
        · exact ⟨h, hne⟩
    · have hne : some q ≠ some x := fun h => hqx (Option.some.inj h)
      rw [dedupAfter_cons_ne t hne, List.mem_cons, ih, List.mem_cons]
      constructor
      · rintro (rfl | ⟨h, -⟩)
        · exact ⟨Or.inl rfl, fun h => hqx h.symm⟩
        · refine ⟨Or.inr h, fun hyq => ?_⟩
          subst hyq
          exact hqx (o.antisymm _ _ (hq x (List.mem_cons_self ..)) (hx _ h))
      · rintro ⟨rfl | h, -⟩
        · exact Or.inl rfl
        · by_cases hyx : y = x
          · exact Or.inl hyx
          · exact Or.inr ⟨h, hyx⟩

/-- Removing the repeats of a sorted list keeps its elements. -/
theorem mem_dedupSorted [DecidableEq α] (o : TotalOrder α) {xs : List α}
    (hs : xs.Pairwise (fun a b => o.le a b)) (y : α) : y ∈ dedupSorted xs ↔ y ∈ xs := by
  cases xs with
  | nil => simp [dedupSorted, dedupAfter]
  | cons x t =>
    have ht : t.Pairwise (fun a b => o.le a b) := List.Pairwise.of_cons hs
    have hx : ∀ z ∈ t, o.le x z := fun z hz => List.rel_of_pairwise_cons hs hz
    rw [dedupSorted, dedupAfter_cons_ne t (by simp), List.mem_cons,
      mem_dedupAfter_some o x t ht hx y, List.mem_cons]
    constructor
    · rintro (h | ⟨h, -⟩)
      · exact Or.inl h
      · exact Or.inr h
    · rintro (h | h)
      · exact Or.inl h
      · by_cases hyx : y = x
        · exact Or.inl hyx
        · exact Or.inr ⟨h, hyx⟩

/-- Strictly ascending. -/
def Strict (o : TotalOrder α) (a b : α) : Prop := o.le a b ∧ a ≠ b

theorem strict_dedupAfter_some [DecidableEq α] (o : TotalOrder α) :
    ∀ (q : α) (xs : List α), xs.Pairwise (fun a b => o.le a b) → (∀ z ∈ xs, o.le q z) →
      (dedupAfter (some q) xs).Pairwise (Strict o)
  | _, [], _, _ => by simp [dedupAfter]
  | q, x :: t, hs, hq => by
    have ht : t.Pairwise (fun a b => o.le a b) := List.Pairwise.of_cons hs
    have hx : ∀ z ∈ t, o.le x z := fun z hz => List.rel_of_pairwise_cons hs hz
    have ih := strict_dedupAfter_some o x t ht hx
    by_cases hqx : q = x
    · subst hqx
      rw [dedupAfter_cons_same]
      exact ih
    · have hne : some q ≠ some x := fun h => hqx (Option.some.inj h)
      rw [dedupAfter_cons_ne t hne]
      refine List.Pairwise.cons (fun y hy => ?_) ih
      obtain ⟨hyt, hyx⟩ := (mem_dedupAfter_some o x t ht hx y).1 hy
      exact ⟨hx y hyt, fun h => hyx h.symm⟩

/-- A sorted list with its repeats removed is strictly ascending. -/
theorem strict_dedupSorted [DecidableEq α] (o : TotalOrder α) {xs : List α}
    (hs : xs.Pairwise (fun a b => o.le a b)) : (dedupSorted xs).Pairwise (Strict o) := by
  cases xs with
  | nil => simp [dedupSorted, dedupAfter]
  | cons x t =>
    have ht : t.Pairwise (fun a b => o.le a b) := List.Pairwise.of_cons hs
    have hx : ∀ z ∈ t, o.le x z := fun z hz => List.rel_of_pairwise_cons hs hz
    rw [dedupSorted, dedupAfter_cons_ne t (by simp)]
    refine List.Pairwise.cons (fun y hy => ?_) (strict_dedupAfter_some o x t ht hx)
    obtain ⟨hyt, hyx⟩ := (mem_dedupAfter_some o x t ht hx y).1 hy
    exact ⟨hx y hyt, fun h => hyx h.symm⟩

theorem length_dedupAfter_le [DecidableEq α] :
    ∀ (p : Option α) (xs : List α), (dedupAfter p xs).length ≤ xs.length
  | _, [] => by simp [dedupAfter]
  | p, x :: t => by
    have := length_dedupAfter_le (some x) t
    simp only [dedupAfter]
    split <;> simp <;> omega

/-- Two strictly ascending lists with the same elements are equal. -/
theorem eq_of_strict_of_mem_iff (o : TotalOrder α) :
    ∀ {l₁ l₂ : List α}, l₁.Pairwise (Strict o) → l₂.Pairwise (Strict o) →
      (∀ y, y ∈ l₁ ↔ y ∈ l₂) → l₁ = l₂
  | [], [], _, _, _ => rfl
  | [], b :: _, _, _, h => absurd ((h b).2 (List.mem_cons_self ..)) (by simp)
  | a :: _, [], _, _, h => absurd ((h a).1 (List.mem_cons_self ..)) (by simp)
  | a :: t₁, b :: t₂, h₁, h₂, h => by
    have hab : a = b := by
      rcases List.mem_cons.1 ((h a).1 (List.mem_cons_self ..)) with hb | hb
      · exact hb
      rcases List.mem_cons.1 ((h b).2 (List.mem_cons_self ..)) with ha | ha
      · exact ha.symm
      exact o.antisymm a b (List.rel_of_pairwise_cons h₁ ha).1 (List.rel_of_pairwise_cons h₂ hb).1
    subst hab
    have ht : ∀ y, y ∈ t₁ ↔ y ∈ t₂ := by
      intro y
      constructor
      · intro hy
        have hne : a ≠ y := (List.rel_of_pairwise_cons h₁ hy).2
        rcases List.mem_cons.1 ((h y).1 (List.mem_cons_of_mem _ hy)) with e | e
        · exact absurd e.symm hne
        · exact e
      · intro hy
        have hne : a ≠ y := (List.rel_of_pairwise_cons h₂ hy).2
        rcases List.mem_cons.1 ((h y).2 (List.mem_cons_of_mem _ hy)) with e | e
        · exact absurd e.symm hne
        · exact e
    rw [eq_of_strict_of_mem_iff o (List.Pairwise.of_cons h₁) (List.Pairwise.of_cons h₂) ht]

/-- A set: sorted, repeats removed, then framed as an ordered list. -/
def set [DecidableEq α] (o : TotalOrder α) (count : Nat → List β) (e : α → List β)
    (xs : List α) : List β :=
  ordered count e (dedupSorted (xs.mergeSort o.le))

/-- A multiset: sorted with repeats kept, then framed as an ordered list. -/
def multiset (o : TotalOrder α) (count : Nat → List β) (e : α → List β) (xs : List α) :
    List β :=
  ordered count e (xs.mergeSort o.le)

section Collections

variable {bound : Nat} {P : α → Prop} {count : Nat → List β} {e : α → List β}

/-- Set framing depends exactly on which elements occur: two lists frame alike if and only if
they hold the same elements, whatever their order and repeats. -/
theorem set_eq_iff [DecidableEq α] (o : TotalOrder α) (hc : PrefixFreeOn (· < bound) count)
    (he : PrefixFreeOn P e) {xs ys : List α} (hxl : xs.length < bound) (hyl : ys.length < bound)
    (hx : ∀ x ∈ xs, P x) (hy : ∀ y ∈ ys, P y) :
    set o count e xs = set o count e ys ↔ ∀ z, z ∈ xs ↔ z ∈ ys := by
  have sx := Antecedent.Identity.sorted_mergeSort' o xs
  have sy := Antecedent.Identity.sorted_mergeSort' o ys
  have memx : ∀ z, z ∈ dedupSorted (xs.mergeSort o.le) ↔ z ∈ xs := fun z =>
    (mem_dedupSorted o sx z).trans List.mem_mergeSort
  have memy : ∀ z, z ∈ dedupSorted (ys.mergeSort o.le) ↔ z ∈ ys := fun z =>
    (mem_dedupSorted o sy z).trans List.mem_mergeSort
  constructor
  · intro h z
    have hlx : (dedupSorted (xs.mergeSort o.le)).length < bound :=
      Nat.lt_of_le_of_lt (length_dedupAfter_le _ _) (by rw [List.length_mergeSort]; exact hxl)
    have hly : (dedupSorted (ys.mergeSort o.le)).length < bound :=
      Nat.lt_of_le_of_lt (length_dedupAfter_le _ _) (by rw [List.length_mergeSort]; exact hyl)
    have heq := (prefixFree_ordered hc he).inj
      ⟨hlx, fun z hz => hx z ((memx z).1 hz)⟩ ⟨hly, fun z hz => hy z ((memy z).1 hz)⟩ h
    rw [← memx, ← memy, heq]
  · intro h
    unfold set
    rw [eq_of_strict_of_mem_iff o (strict_dedupSorted o sx) (strict_dedupSorted o sy)
      (fun z => (memx z).trans ((h z).trans (memy z).symm))]

/-- Reordering a set's elements keeps its frame. -/
theorem set_perm [DecidableEq α] (o : TotalOrder α) {xs ys : List α} (h : xs.Perm ys) :
    set o count e xs = set o count e ys := by
  unfold set
  rw [Antecedent.Identity.mergeSort_perm_eq o h]

/-- Repeating an element of a set keeps its frame. -/
theorem set_dup [DecidableEq α] (o : TotalOrder α) {x : α} {xs : List α} (h : x ∈ xs) :
    set o count e (x :: xs) = set o count e xs := by
  have sx := Antecedent.Identity.sorted_mergeSort' o (x :: xs)
  have sy := Antecedent.Identity.sorted_mergeSort' o xs
  unfold set
  rw [eq_of_strict_of_mem_iff o (strict_dedupSorted o sx) (strict_dedupSorted o sy)]
  intro z
  rw [mem_dedupSorted o sx, mem_dedupSorted o sy, List.mem_mergeSort, List.mem_mergeSort,
    List.mem_cons]
  constructor
  · rintro (rfl | hz)
    · exact h
    · exact hz
  · exact Or.inr

/-- Multiset framing depends exactly on the elements with their multiplicities. -/
theorem multiset_eq_iff (o : TotalOrder α) (hc : PrefixFreeOn (· < bound) count)
    (he : PrefixFreeOn P e) {xs ys : List α} (hxl : xs.length < bound) (hyl : ys.length < bound)
    (hx : ∀ x ∈ xs, P x) (hy : ∀ y ∈ ys, P y) :
    multiset o count e xs = multiset o count e ys ↔ xs.Perm ys := by
  constructor
  · intro h
    have heq := (prefixFree_ordered hc he).inj
      ⟨by simpa [List.length_mergeSort] using hxl, fun z hz => hx z (List.mem_mergeSort.1 hz)⟩
      ⟨by simpa [List.length_mergeSort] using hyl, fun z hz => hy z (List.mem_mergeSort.1 hz)⟩ h
    exact ((List.mergeSort_perm xs o.le).symm.trans (heq ▸ List.Perm.refl _)).trans
      (List.mergeSort_perm ys o.le)
  · intro h
    unfold multiset
    rw [Antecedent.Identity.mergeSort_perm_eq o h]

/-- Multiset framing keeps repeats: adding a copy of an element changes the frame. -/
theorem multiset_dup_ne (o : TotalOrder α) (hc : PrefixFreeOn (· < bound) count)
    (he : PrefixFreeOn P e) {x : α} {xs : List α} (hl : xs.length + 1 < bound) (hpx : P x)
    (hx : ∀ y ∈ xs, P y) : multiset o count e (x :: xs) ≠ multiset o count e xs := by
  intro h
  have hpx' : ∀ y ∈ x :: xs, P y := by
    intro y hy
    rcases List.mem_cons.1 hy with rfl | hy
    · exact hpx
    · exact hx y hy
  have hp := (multiset_eq_iff o hc he (xs := x :: xs) (by simpa using hl) (by omega) hpx' hx).1 h
  have := hp.length_eq
  simp at this

end Collections

/-- Identity of a framed value under a hash. -/
def identity {D : Type} (hash : List β → D) (e : α → List β) (a : α) : D := hash (e a)

/-- Under a hash that is collision-free on the streams it is given, two framed values share an
identity exactly when they are equal. -/
theorem identity_eq_iff {D : Type} {P : α → Prop} (hash : List β → D)
    (hinj : ∀ s t, hash s = hash t → s = t) {e : α → List β} (he : PrefixFreeOn P e) {a b : α}
    (ha : P a) (hb : P b) : identity hash e a = identity hash e b ↔ a = b :=
  ⟨fun h => he.inj ha hb (hinj _ _ h), fun h => h ▸ rfl⟩

end Antecedent.Framing
