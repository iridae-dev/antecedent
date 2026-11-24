/-
The production framing kernels (`antecedent_core::framing`), as Aeneas translates them from
Rust, compute the frames of the specification in `Antecedent.Framing`, so its theorems hold
of the bytes every `CanonicalWriter` hashes.

Each kernel is generic over the sink it appends to (`FrameSink`). The proofs run it on `log`,
the sink that keeps every byte it is given; in production the sink is a BLAKE3 hasher, whose
`update` calls hash the concatenation of their inputs.

Trusted:
* Charon and Aeneas translate the kernels faithfully, and the Aeneas model of the standard
  library operations they use (`u64::to_le_bytes`, integer casts and addition, slice
  indexing, `Vec::push`, array equality) is correct;
* `sort_unstable` sorts ascending under `Ord` (lexicographic for `[u8; 32]`, numeric for
  `u32`): the kernels' two sorts are translated as opaque and modelled in
  `Extracted/Framing/FunsExternal.lean`;
* BLAKE3's incremental `update` hashes the concatenation of its inputs, and BLAKE3 is
  collision-resistant on the streams it is given;
* `f64::to_bits` is the IEEE-754 bit pattern;
* each identity site composes the kernels in the category its meaning needs (ordered, set or
  multiset), which the identity-site inventory (`formal/identity_sites.toml`) records and its
  checks cover; the composition of kernel calls inside `CanonicalWriter` methods and
  `CanonicalEncode` implementations is not translated.
-/
import Aeneas
import Antecedent.Framing
import AntecedentRust.Extracted.Framing.Funs

open Aeneas Aeneas.Std Result
open AntecedentRust.Extracted.Framing
open Antecedent.Framing (PrefixFreeOn items ordered dedupSorted dedupAfter dedupAfter_cons_ne
  dedupAfter_cons_same)

namespace AntecedentRust.Framing

/-! ## Frames as byte lists -/

/-- The sink that keeps every byte it is given, in order. -/
def log : framing.FrameSink (List U8) := { put := fun s b => ok (s ++ b.val) }

/-- The little-endian bytes of a 64-bit word. -/
def bvBytes (b : BitVec 64) : List U8 := b.toLEBytes.map UScalar.mk

/-- The frame of a word. -/
def wordBytes (w : U64) : List U8 := bvBytes w.bv

/-- The frame of a count or length. -/
def countBytes (n : Nat) : List U8 := bvBytes (BitVec.ofNat 64 n)

/-- The frame of a byte string: its length, then the bytes. -/
def bytesSpec (p : List U8) : List U8 := countBytes p.length ++ p

/-- The frame of a presence word. -/
def presenceBytes (present : Bool) : List U8 := wordBytes (if present then 1#u64 else 0#u64)

/-- The frame of a `u32` widened to a word. -/
def u32Bytes (x : U32) : List U8 := wordBytes (core.convert.num.FromU64U32.from x)

/-- The frame of a 32-byte digest: a byte string of length 32. -/
def digestBytes (d : Array U8 32#usize) : List U8 := bytesSpec d.val

theorem bvBytes_length (b : BitVec 64) : (bvBytes b).length = 8 := by
  simp [bvBytes]

theorem bvBytes_inj {a b : BitVec 64} (h : bvBytes a = bvBytes b) : a = b := by
  have hl : a.toLEBytes = b.toLEBytes :=
    (List.map_inj_right (fun x y hxy => by cases hxy; rfl)).1 h
  have h1 := congrArg (fun l => (BitVec.fromLEBytes l).toNat) hl
  simp only [BitVec.fromLEBytes_toLEBytes (by decide : 64 % 8 = 0), BitVec.toNat_cast] at h1
  exact BitVec.eq_of_toNat_eq h1

theorem wordBytes_inj {a b : U64} (h : wordBytes a = wordBytes b) : a = b := by
  have := bvBytes_inj h
  cases a; cases b; cases this; rfl

/-- A word frame is prefix-free: eight bytes that determine the word. -/
theorem word_prefixFree : PrefixFreeOn (fun _ => True) wordBytes :=
  Antecedent.Framing.prefixFree_of_width 8 (fun _ _ => bvBytes_length _)
    (fun _ _ _ _ h => wordBytes_inj h)

/-- A count frame is prefix-free for every count below `2^64`. -/
theorem count_prefixFree : PrefixFreeOn (· < 2 ^ 64) countBytes :=
  Antecedent.Framing.prefixFree_of_width 8 (fun _ _ => bvBytes_length _) (by
    intro m n hm hn h
    have := congrArg BitVec.toNat (bvBytes_inj h)
    simp only [BitVec.toNat_ofNat] at this
    rwa [Nat.mod_eq_of_lt hm, Nat.mod_eq_of_lt hn] at this)

/-- A byte-string frame is prefix-free: the length word says where the bytes end. -/
theorem bytes_prefixFree : PrefixFreeOn (fun p : List U8 => p.length < 2 ^ 64) bytesSpec := by
  intro a b r s ha hb h
  simp only [bytesSpec, List.append_assoc] at h
  obtain ⟨hl, h'⟩ := count_prefixFree _ _ _ _ ha hb h
  exact List.append_inj h' hl

/-- A presence word is prefix-free: absent and present never read as each other. -/
theorem presence_prefixFree : PrefixFreeOn (fun _ => True) presenceBytes :=
  Antecedent.Framing.prefixFree_of_width 8 (fun _ _ => bvBytes_length _) (by
    intro a b _ _ h
    have := wordBytes_inj h
    cases a <;> cases b <;> simp_all)

theorem u32Bytes_inj {a b : U32} (h : u32Bytes a = u32Bytes b) : a = b := by
  have := congrArg UScalar.val (wordBytes_inj h)
  simp only [core.convert.num.FromU64U32.from_val_eq] at this
  exact UScalar.eq_of_val_eq this

theorem u32_prefixFree : PrefixFreeOn (fun _ => True) u32Bytes :=
  Antecedent.Framing.prefixFree_of_width 8 (fun _ _ => bvBytes_length _)
    (fun _ _ _ _ h => u32Bytes_inj h)

theorem digest_prefixFree : PrefixFreeOn (fun _ => True) digestBytes := by
  intro a b r s _ _ h
  obtain ⟨hab, hrs⟩ := bytes_prefixFree a.val b.val r s (by simp) (by simp) h
  exact ⟨(Aeneas.Std.Array.eq_iff a b).2 hab, hrs⟩

/-! ## The real frame -/

/-- The bit pattern of `-0.0`. -/
def negZeroBits : U64 := 9223372036854775808#u64

/-- A quiet NaN and a NaN with another payload. -/
def nanBits : U64 := 9221120237041090560#u64
def nanPayloadBits : U64 := 9221120237041090561#u64

/-- A real frame determines the bit pattern: `-0.0` and `0.0`, and NaNs with different payloads,
frame differently. -/
theorem real_frames_distinguish_bits {a b : U64} (h : a ≠ b) (r s : List U8) :
    wordBytes a ++ r ≠ wordBytes b ++ s :=
  word_prefixFree.tag_separates trivial trivial h r s

theorem neg_zero_frames_apart : wordBytes negZeroBits ≠ wordBytes 0#u64 := by
  intro h
  have := congrArg UScalar.val (wordBytes_inj h)
  simp [negZeroBits] at this

theorem nan_payloads_frame_apart : wordBytes nanBits ≠ wordBytes nanPayloadBits := by
  intro h
  have := congrArg UScalar.val (wordBytes_inj h)
  simp [nanBits, nanPayloadBits] at this

/-! ## The kernels compute the frames -/

theorem frame_word_spec (s : List U8) (w : U64) :
    framing.frame_word log s w ⦃ r => r = s ++ wordBytes w ⦄ := by
  unfold framing.frame_word
  simp [lift, log, wordBytes, bvBytes, core.num.U64.to_le_bytes, Array.to_slice]

theorem cast_bv (c : Usize) : (UScalar.cast .U64 c).bv = BitVec.ofNat 64 c.val := by
  apply BitVec.eq_of_toNat_eq
  simp

theorem frame_count_spec (s : List U8) (c : Usize) :
    framing.frame_count log s c ⦃ r => r = s ++ countBytes c.val ⦄ := by
  unfold framing.frame_count
  simp only [lift, bind_ok]
  have := frame_word_spec s (UScalar.cast .U64 c)
  simpa [wordBytes, countBytes, cast_bv] using this

theorem frame_bytes_spec (s : List U8) (p : Slice U8) :
    framing.frame_bytes log s p ⦃ r => r = s ++ bytesSpec p.val ⦄ := by
  unfold framing.frame_bytes
  apply WP.spec_bind (frame_count_spec s (Slice.len p))
  intro r hr
  simp [log, hr, bytesSpec]

theorem frame_presence_spec (s : List U8) (present : Bool) :
    framing.frame_presence log s present ⦃ r => r = s ++ presenceBytes present ⦄ := by
  unfold framing.frame_presence
  simp only [lift, bind_ok]
  cases present <;> simpa [presenceBytes, core.convert.num.FromU64Bool.from] using frame_word_spec s _

theorem frame_real_bits_spec (s : List U8) (bits : U64) :
    framing.frame_real_bits log s bits ⦃ r => r = s ++ wordBytes bits ⦄ := by
  unfold framing.frame_real_bits
  exact frame_word_spec s bits

/-! ## Ordered lists -/

theorem drop_cons_getElem {α : Type} (l : List α) (j : Nat) (h : j < l.length) :
    l.drop j = l[j] :: l.drop (j + 1) := by
  exact List.drop_eq_getElem_cons h

theorem frame_words_loop_spec (values : Slice U64) (s : List U8) (i : Usize)
    (hi : i.val ≤ values.length) :
    framing.frame_words_loop log s values i
      ⦃ r => r = s ++ items wordBytes (values.val.drop i.val) ⦄ := by
  unfold framing.frame_words_loop
  apply loop.spec_decr_nat (fun p => values.length - p.2.val)
    (fun p => p.2.val ≤ values.length ∧
      p.1 ++ items wordBytes (values.val.drop p.2.val) = s ++ items wordBytes (values.val.drop i.val))
  · rintro ⟨sink, j⟩ ⟨hj, hinv⟩
    simp only [framing.frame_words_loop.body]
    split
    · rename_i hlt
      have hlt' : j.val < values.length := by scalar_tac
      apply WP.spec_bind (Slice.index_usize_spec values j hlt')
      intro x hx
      apply WP.spec_bind (frame_word_spec sink x)
      intro sink1 hs1
      apply WP.spec_bind (Usize.add_spec (x := j) (y := 1#usize) (by scalar_tac))
      intro j1 hj1
      simp only [WP.spec_ok]
      refine ⟨⟨by scalar_tac, ?_⟩, by scalar_tac⟩
      rw [← hinv, hs1, hx, drop_cons_getElem _ _ hlt', hj1]
      simp only [items, List.map_cons, List.flatten_cons, List.append_assoc]
      simp
    · rename_i hge
      simp only [WP.spec_ok]
      rw [← hinv]
      have : values.val.drop j.val = [] := List.drop_eq_nil_of_le (by scalar_tac)
      simp [this, items]
  · exact ⟨hi, rfl⟩

theorem frame_words_spec (s : List U8) (values : Slice U64) :
    framing.frame_words log s values ⦃ r => r = s ++ ordered countBytes wordBytes values.val ⦄ := by
  unfold framing.frame_words
  apply WP.spec_bind (frame_count_spec s (Slice.len values))
  intro r hr
  apply WP.spec_mono (frame_words_loop_spec values r 0#usize (by simp))
  intro r' hr'
  simp [hr', hr, ordered]

theorem allM_zip_val_eq : ∀ (l₁ l₂ : List U8), l₁.length = l₂.length →
    List.allM (fun x : U8 × U8 => ok (decide (x.1.val = x.2.val))) (l₁.zip l₂) =
      ok (decide (l₁ = l₂))
  | [], [], _ => rfl
  | [], _ :: _, h => absurd h (by simp)
  | _ :: _, [], h => absurd h (by simp)
  | x :: xs, y :: ys, h => by
    have ih := allM_zip_val_eq xs ys (by simpa using h)
    by_cases hxy : x = y
    · subst hxy
      simp [ih]
    · have hv : x.val ≠ y.val := fun hv => hxy (UScalar.eq_of_val_eq hv)
      simp [hv, hxy]
      rfl

theorem ne_digest_spec (a b : Array U8 32#usize) :
    core.array.equality.PartialEqArray.ne core.cmp.PartialEqU8 a b ⦃ r => r = decide (a ≠ b) ⦄ := by
  unfold core.array.equality.PartialEqArray.ne core.array.equality.PartialEqArray.eq
  simp
  rw [allM_zip_val_eq _ _ (by simp)]
  simp [Aeneas.Std.Array.eq_iff]

theorem frame_u32s_loop_spec (values : Slice U32) (s : List U8) (i : Usize)
    (hi : i.val ≤ values.length) :
    framing.frame_u32s_loop log s values i
      ⦃ r => r = s ++ items u32Bytes (values.val.drop i.val) ⦄ := by
  unfold framing.frame_u32s_loop
  apply loop.spec_decr_nat (fun p => values.length - p.2.val)
    (fun p => p.2.val ≤ values.length ∧
      p.1 ++ items u32Bytes (values.val.drop p.2.val) = s ++ items u32Bytes (values.val.drop i.val))
  · rintro ⟨sink, j⟩ ⟨hj, hinv⟩
    simp only [framing.frame_u32s_loop.body]
    split
    · rename_i hlt
      have hlt' : j.val < values.length := by scalar_tac
      apply WP.spec_bind (Slice.index_usize_spec values j hlt')
      intro x hx
      simp only [lift, bind_ok]
      apply WP.spec_bind (frame_word_spec sink (core.convert.num.FromU64U32.from x))
      intro sink1 hs1
      apply WP.spec_bind (Usize.add_spec (x := j) (y := 1#usize) (by scalar_tac))
      intro j1 hj1
      simp only [WP.spec_ok]
      refine ⟨⟨by scalar_tac, ?_⟩, by scalar_tac⟩
      rw [← hinv, hs1, hx, drop_cons_getElem _ _ hlt', hj1]
      simp only [items, List.map_cons, List.flatten_cons, List.append_assoc, u32Bytes]
      simp
    · simp only [WP.spec_ok]
      rw [← hinv]
      have : values.val.drop j.val = [] := List.drop_eq_nil_of_le (by scalar_tac)
      simp [this, items]
  · exact ⟨hi, rfl⟩

theorem frame_u32s_spec (s : List U8) (values : Slice U32) :
    framing.frame_u32s log s values ⦃ r => r = s ++ ordered countBytes u32Bytes values.val ⦄ := by
  unfold framing.frame_u32s
  apply WP.spec_bind (frame_count_spec s (Slice.len values))
  intro r hr
  apply WP.spec_mono (frame_u32s_loop_spec values r 0#usize (by simp))
  intro r' hr'
  simp [hr', hr, ordered]

theorem frame_digests_loop_spec (values : Slice (Array U8 32#usize)) (s : List U8) (i : Usize)
    (hi : i.val ≤ values.length) :
    framing.frame_digests_loop log s values i
      ⦃ r => r = s ++ items digestBytes (values.val.drop i.val) ⦄ := by
  unfold framing.frame_digests_loop
  apply loop.spec_decr_nat (fun p => values.length - p.2.val)
    (fun p => p.2.val ≤ values.length ∧
      p.1 ++ items digestBytes (values.val.drop p.2.val) =
        s ++ items digestBytes (values.val.drop i.val))
  · rintro ⟨sink, j⟩ ⟨hj, hinv⟩
    simp only [framing.frame_digests_loop.body]
    split
    · rename_i hlt
      have hlt' : j.val < values.length := by scalar_tac
      apply WP.spec_bind (Slice.index_usize_spec values j hlt')
      intro x hx
      simp only [lift, bind_ok]
      apply WP.spec_bind (frame_bytes_spec sink (Array.to_slice x))
      intro sink1 hs1
      apply WP.spec_bind (Usize.add_spec (x := j) (y := 1#usize) (by scalar_tac))
      intro j1 hj1
      simp only [WP.spec_ok]
      refine ⟨⟨by scalar_tac, ?_⟩, by scalar_tac⟩
      rw [← hinv, hs1, hx, drop_cons_getElem _ _ hlt', hj1]
      simp only [items, List.map_cons, List.flatten_cons, List.append_assoc, digestBytes,
        Array.val_to_slice]
      simp
    · simp only [WP.spec_ok]
      rw [← hinv]
      have : values.val.drop j.val = [] := List.drop_eq_nil_of_le (by scalar_tac)
      simp [this, items]
  · exact ⟨hi, rfl⟩

theorem frame_digests_spec (s : List U8) (values : Slice (Array U8 32#usize)) :
    framing.frame_digests log s values
      ⦃ r => r = s ++ ordered countBytes digestBytes values.val ⦄ := by
  unfold framing.frame_digests
  apply WP.spec_bind (frame_count_spec s (Slice.len values))
  intro r hr
  apply WP.spec_mono (frame_digests_loop_spec values r 0#usize (by simp))
  intro r' hr'
  simp [hr', hr, ordered]

/-! ## Removing repeats -/

/-- A 32-byte digest. -/
abbrev Digest : Type := Array U8 32#usize

/-- The element before position `j`, the one the production loop compares with. -/
def previous (l : List (Array U8 32#usize)) (j : Nat) : Option (Array U8 32#usize) :=
  if j = 0 then none else l[j - 1]?

theorem dedup_sorted_digests_spec (sorted : Slice (Array U8 32#usize)) :
    framing.dedup_sorted_digests sorted ⦃ v => v.val = dedupSorted sorted.val ⦄ := by
  unfold framing.dedup_sorted_digests framing.dedup_sorted_digests_loop
  apply loop.spec_decr_nat (fun p => sorted.length - p.2.val)
    (fun p => p.2.val ≤ sorted.length ∧ p.1.val.length ≤ p.2.val ∧
      p.1.val ++ dedupAfter (previous sorted.val p.2.val) (sorted.val.drop p.2.val) =
        dedupSorted sorted.val)
  · rintro ⟨distinct, j⟩ ⟨hj, hlen, hinv⟩
    simp only [framing.dedup_sorted_digests_loop.body]
    split
    · rename_i hlt
      have hlt' : j.val < sorted.length := by scalar_tac
      have hdrop := drop_cons_getElem _ _ hlt'
      have hp1 : previous sorted.val (j.val + 1) = some sorted.val[j.val] := by
        simp [previous, List.getElem?_eq_getElem hlt']
      -- One step keeps the invariant: what is pushed and what is left to remove repeats from
      -- together still give the whole list without repeats.
      apply WP.spec_bind (Pₘ := fun (d1 : alloc.vec.Vec Digest) =>
        d1.val.length ≤ distinct.val.length + 1 ∧
        d1.val ++ dedupAfter (some (sorted.val[j.val]'hlt')) (sorted.val.drop (j.val + 1)) =
          distinct.val ++ dedupAfter (previous sorted.val j.val) (sorted.val.drop j.val))
      · split
        · rename_i hj0
          have hj0' : j.val = 0 := by scalar_tac
          have hp0 : previous sorted.val j.val = none := by simp [previous, hj0']
          apply WP.spec_bind (Slice.index_usize_spec sorted j hlt')
          intro x hx
          apply WP.spec_mono (alloc.vec.Vec.push_spec distinct x (by scalar_tac))
          intro d1 hd1
          refine ⟨by simp [hd1], ?_⟩
          rw [hp0, hdrop, hd1, hx, dedupAfter_cons_ne _ (by simp)]
          simp
        · rename_i hj0
          have hj0' : j.val ≠ 0 := by scalar_tac
          apply WP.spec_bind (Usize.sub_spec (x := j) (y := 1#usize) (by scalar_tac))
          intro jm hjm
          apply WP.spec_bind (Slice.index_usize_spec sorted jm (by scalar_tac))
          intro y hy
          apply WP.spec_bind (Slice.index_usize_spec sorted j hlt')
          intro x hx
          apply WP.spec_bind (ne_digest_spec y x)
          intro b hb
          have hprev : previous sorted.val j.val = some y := by
            have hjm' : jm.val = j.val - 1 := by scalar_tac
            simp [previous, hj0', hy, hjm',
              List.getElem?_eq_getElem (by scalar_tac : j.val - 1 < sorted.val.length)]
          rw [hprev, hdrop, ← hx]
          split
          · rename_i hbt
            have hne : y ≠ x := by simpa [hb] using hbt
            apply WP.spec_mono (alloc.vec.Vec.push_spec distinct x (by scalar_tac))
            intro d1 hd1
            refine ⟨by simp [hd1], ?_⟩
            rw [hd1, dedupAfter_cons_ne _ (by simpa using hne)]
            simp
          · rename_i hbf
            have heq : y = x := by simpa [hb] using hbf
            simp only [WP.spec_ok]
            refine ⟨by simp, ?_⟩
            rw [heq, dedupAfter_cons_same]
      · rintro d1 ⟨hd1l, hd1⟩
        apply WP.spec_bind (Usize.add_spec (x := j) (y := 1#usize) (by scalar_tac))
        intro j1 hj1
        have hj1' : j1.val = j.val + 1 := by scalar_tac
        simp only [WP.spec_ok]
        refine ⟨⟨by scalar_tac, by scalar_tac, ?_⟩, by scalar_tac⟩
        rw [← hinv, hj1', hp1, hd1]
    · simp only [WP.spec_ok]
      rw [← hinv]
      have : sorted.val.drop j.val = [] := List.drop_eq_nil_of_le (by scalar_tac)
      simp [this, dedupAfter]
  · refine ⟨by simp, by simp, ?_⟩
    simp [previous, dedupSorted]

/-! ## Orders and sorts -/

theorem bytesLe_total : ∀ a b : List Nat, (bytesLe a b || bytesLe b a) = true
  | [], _ => by simp [bytesLe]
  | _ :: _, [] => by simp [bytesLe]
  | x :: xs, y :: ys => by
    have ih := bytesLe_total xs ys
    simp only [bytesLe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at ih ⊢
    rcases Nat.lt_trichotomy x y with h | rfl | h
    · exact Or.inl (Or.inl h)
    · rcases ih with ih | ih
      · exact Or.inl (Or.inr ⟨rfl, ih⟩)
      · exact Or.inr (Or.inr ⟨rfl, ih⟩)
    · exact Or.inr (Or.inl h)

theorem bytesLe_trans : ∀ a b c : List Nat, bytesLe a b = true → bytesLe b c = true →
    bytesLe a c = true
  | [], _, _, _, _ => by simp [bytesLe]
  | _ :: _, [], _, h, _ => by simp [bytesLe] at h
  | _ :: _, _ :: _, [], _, h => by simp [bytesLe] at h
  | x :: xs, y :: ys, z :: zs, h₁, h₂ => by
    have ih := bytesLe_trans xs ys zs
    simp only [bytesLe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h₁ h₂ ⊢
    rcases h₁ with h₁ | ⟨rfl, h₁⟩ <;> rcases h₂ with h₂ | ⟨rfl, h₂⟩
    · exact Or.inl (by omega)
    · exact Or.inl h₁
    · exact Or.inl h₂
    · exact Or.inr ⟨rfl, ih h₁ h₂⟩

theorem bytesLe_antisymm : ∀ a b : List Nat, bytesLe a b = true → bytesLe b a = true → a = b
  | [], [], _, _ => rfl
  | [], _ :: _, _, h => by simp [bytesLe] at h
  | _ :: _, [], h, _ => by simp [bytesLe] at h
  | x :: xs, y :: ys, h₁, h₂ => by
    simp only [bytesLe, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h₁ h₂
    rcases h₁ with h₁ | ⟨rfl, h₁⟩
    · rcases h₂ with h₂ | ⟨h₂, -⟩ <;> omega
    · rcases h₂ with h₂ | ⟨-, h₂⟩
      · omega
      · rw [bytesLe_antisymm xs ys h₁ h₂]

theorem map_val_inj {a b : List U8} (h : a.map (·.val) = b.map (·.val)) : a = b :=
  (List.map_inj_right (fun _ _ hxy => UScalar.eq_of_val_eq hxy)).1 h

/-- `Ord` on `[u8; 32]`: lexicographic on bytes. -/
def digestOrder : Antecedent.Identity.TotalOrder (Array U8 32#usize) where
  le := digestLe
  total _ _ := bytesLe_total _ _
  trans _ _ _ := bytesLe_trans _ _ _
  antisymm a b h₁ h₂ := (Aeneas.Std.Array.eq_iff a b).2 (map_val_inj (bytesLe_antisymm _ _ h₁ h₂))

/-- `Ord` on `u32`: numeric. -/
def u32Order : Antecedent.Identity.TotalOrder U32 where
  le := u32Le
  total _ _ := by simp only [u32Le, Bool.or_eq_true, decide_eq_true_eq]; omega
  trans _ _ _ h₁ h₂ := by simp only [u32Le, decide_eq_true_eq] at h₁ h₂ ⊢; omega
  antisymm a b h₁ h₂ := by
    simp only [u32Le, decide_eq_true_eq] at h₁ h₂
    exact UScalar.eq_of_val_eq (by omega)

/-! ## Sets and multisets -/

theorem frame_digest_set_spec (s : List U8) (digests : alloc.vec.Vec (Array U8 32#usize)) :
    framing.frame_digest_set log s digests
      ⦃ r => r = s ++ Antecedent.Framing.set digestOrder countBytes digestBytes digests.val ⦄ := by
  unfold framing.frame_digest_set
  simp only [lift, bind_ok, framing.sort_digests, alloc.vec.Vec.deref_mut]
  apply WP.spec_bind (dedup_sorted_digests_spec _)
  intro d hd
  apply WP.spec_mono (frame_digests_spec s _)
  intro r hr
  simp only [alloc.vec.Vec.deref, Slice.from_val] at hr
  rw [hr, hd]
  simp [Antecedent.Framing.set, digestOrder, alloc.vec.Vec.val, alloc.vec.Vec.deref]

theorem frame_digest_multiset_spec (s : List U8) (digests : alloc.vec.Vec (Array U8 32#usize)) :
    framing.frame_digest_multiset log s digests
      ⦃ r => r = s ++ Antecedent.Framing.multiset digestOrder countBytes digestBytes digests.val ⦄ := by
  unfold framing.frame_digest_multiset
  simp only [lift, bind_ok, framing.sort_digests, alloc.vec.Vec.deref_mut]
  apply WP.spec_mono (frame_digests_spec s _)
  intro r hr
  rw [hr]
  simp [Antecedent.Framing.multiset, digestOrder, alloc.vec.Vec.val, alloc.vec.Vec.deref]

theorem frame_u32_multiset_spec (s : List U8) (values : alloc.vec.Vec U32) :
    framing.frame_u32_multiset log s values
      ⦃ r => r = s ++ Antecedent.Framing.multiset u32Order countBytes u32Bytes values.val ⦄ := by
  unfold framing.frame_u32_multiset
  simp only [lift, bind_ok, framing.sort_u32s, alloc.vec.Vec.deref_mut]
  apply WP.spec_mono (frame_u32s_spec s _)
  intro r hr
  rw [hr]
  simp [Antecedent.Framing.multiset, u32Order, alloc.vec.Vec.val, alloc.vec.Vec.deref]

/-! ## What the kernels' bytes determine -/

theorem length_lt_two_pow_64 {α : Type} (l : List α) (h : l.length ≤ Usize.max) :
    l.length < 2 ^ 64 := by
  have : Usize.max < 2 ^ 64 := by
    rcases System.Platform.numBits_eq with h' | h' <;> simp [Usize.max, Usize.numBits, h']
  omega

/-- Ordered word lists are uniquely decodable from the front of any stream. -/
theorem ordered_words_prefixFree :
    PrefixFreeOn (fun xs : List U64 => xs.length < 2 ^ 64 ∧ ∀ x ∈ xs, True)
      (ordered countBytes wordBytes) :=
  Antecedent.Framing.prefixFree_ordered count_prefixFree word_prefixFree

/-- Ordered digest lists are uniquely decodable from the front of any stream. -/
theorem ordered_digests_prefixFree :
    PrefixFreeOn (fun xs : List Digest => xs.length < 2 ^ 64 ∧ ∀ x ∈ xs, True)
      (ordered countBytes digestBytes) :=
  Antecedent.Framing.prefixFree_ordered count_prefixFree digest_prefixFree

/-- An optional word (presence word, then the word) is uniquely decodable. -/
theorem option_word_prefixFree :
    PrefixFreeOn (fun o : Option U64 => ∀ a, o = some a → True)
      (Antecedent.Framing.option presenceBytes wordBytes) :=
  Antecedent.Framing.prefixFree_option presence_prefixFree word_prefixFree

/-- Two different domain tags never frame alike, whatever is written after them. -/
theorem tags_separate {t₁ t₂ : List U8} (h₁ : t₁.length < 2 ^ 64) (h₂ : t₂.length < 2 ^ 64)
    (h : t₁ ≠ t₂) (r s : List U8) : bytesSpec t₁ ++ r ≠ bytesSpec t₂ ++ s :=
  bytes_prefixFree.tag_separates h₁ h₂ h r s

/-- `frame_bytes` writes different bytes for different payloads. -/
theorem frame_bytes_eq_iff (s : List U8) (p q : Slice U8) :
    framing.frame_bytes log s p ⦃ r₁ => framing.frame_bytes log s q ⦃ r₂ =>
      (r₁ = r₂ ↔ p.val = q.val) ⦄ ⦄ := by
  apply WP.spec_mono (frame_bytes_spec s p)
  intro r₁ h₁
  apply WP.spec_mono (frame_bytes_spec s q)
  intro r₂ h₂
  rw [h₁, h₂, List.append_cancel_left_eq]
  exact ⟨bytes_prefixFree.inj (length_lt_two_pow_64 _ p.property)
    (length_lt_two_pow_64 _ q.property), fun h => h ▸ rfl⟩

/-- `frame_words` is order-sensitive and injective on sequences. -/
theorem frame_words_eq_iff (s : List U8) (p q : Slice U64) :
    framing.frame_words log s p ⦃ r₁ => framing.frame_words log s q ⦃ r₂ =>
      (r₁ = r₂ ↔ p.val = q.val) ⦄ ⦄ := by
  apply WP.spec_mono (frame_words_spec s p)
  intro r₁ h₁
  apply WP.spec_mono (frame_words_spec s q)
  intro r₂ h₂
  rw [h₁, h₂, List.append_cancel_left_eq]
  exact ⟨ordered_words_prefixFree.inj ⟨length_lt_two_pow_64 _ p.property, fun _ _ => trivial⟩
    ⟨length_lt_two_pow_64 _ q.property, fun _ _ => trivial⟩, fun h => h ▸ rfl⟩

/-- Exchanging two different words of an ordered list changes its frame. -/
theorem words_order_sensitive {a b : U64} {xs : List U64} (hab : a ≠ b)
    (hl : xs.length + 2 < 2 ^ 64) :
    ordered countBytes wordBytes (a :: b :: xs) ≠ ordered countBytes wordBytes (b :: a :: xs) :=
  Antecedent.Framing.ordered_swap_ne count_prefixFree word_prefixFree hab hl trivial trivial
    (fun _ _ => trivial)

/-- `frame_digest_set` writes the same bytes for two inputs exactly when they hold the same
digests, whatever their order and repeats. -/
theorem frame_digest_set_eq_iff (s : List U8) (d₁ d₂ : alloc.vec.Vec Digest) :
    framing.frame_digest_set log s d₁ ⦃ r₁ => framing.frame_digest_set log s d₂ ⦃ r₂ =>
      (r₁ = r₂ ↔ ∀ z, z ∈ d₁.val ↔ z ∈ d₂.val) ⦄ ⦄ := by
  apply WP.spec_mono (frame_digest_set_spec s d₁)
  intro r₁ h₁
  apply WP.spec_mono (frame_digest_set_spec s d₂)
  intro r₂ h₂
  rw [h₁, h₂, List.append_cancel_left_eq]
  exact Antecedent.Framing.set_eq_iff digestOrder count_prefixFree digest_prefixFree
    (length_lt_two_pow_64 _ d₁.property) (length_lt_two_pow_64 _ d₂.property)
    (fun _ _ => trivial) (fun _ _ => trivial)

/-- Reordering the input of `frame_digest_set`, or repeating one of its digests, keeps the
bytes. -/
theorem digest_set_perm {xs ys : List Digest} (h : xs.Perm ys) :
    Antecedent.Framing.set digestOrder countBytes digestBytes xs =
      Antecedent.Framing.set digestOrder countBytes digestBytes ys :=
  Antecedent.Framing.set_perm digestOrder h

theorem digest_set_dup {x : Digest} {xs : List Digest} (h : x ∈ xs) :
    Antecedent.Framing.set digestOrder countBytes digestBytes (x :: xs) =
      Antecedent.Framing.set digestOrder countBytes digestBytes xs :=
  Antecedent.Framing.set_dup digestOrder h

/-- `frame_digest_multiset` writes the same bytes for two inputs exactly when one is a
reordering of the other: repeats count. -/
theorem frame_digest_multiset_eq_iff (s : List U8) (d₁ d₂ : alloc.vec.Vec Digest) :
    framing.frame_digest_multiset log s d₁ ⦃ r₁ => framing.frame_digest_multiset log s d₂ ⦃ r₂ =>
      (r₁ = r₂ ↔ d₁.val.Perm d₂.val) ⦄ ⦄ := by
  apply WP.spec_mono (frame_digest_multiset_spec s d₁)
  intro r₁ h₁
  apply WP.spec_mono (frame_digest_multiset_spec s d₂)
  intro r₂ h₂
  rw [h₁, h₂, List.append_cancel_left_eq]
  exact Antecedent.Framing.multiset_eq_iff digestOrder count_prefixFree digest_prefixFree
    (length_lt_two_pow_64 _ d₁.property) (length_lt_two_pow_64 _ d₂.property)
    (fun _ _ => trivial) (fun _ _ => trivial)

theorem digest_multiset_dup_ne {x : Digest} {xs : List Digest} (hl : xs.length + 1 < 2 ^ 64) :
    Antecedent.Framing.multiset digestOrder countBytes digestBytes (x :: xs) ≠
      Antecedent.Framing.multiset digestOrder countBytes digestBytes xs :=
  Antecedent.Framing.multiset_dup_ne digestOrder count_prefixFree digest_prefixFree hl trivial
    (fun _ _ => trivial)

/-- `frame_u32_multiset` writes the same bytes for two inputs exactly when one is a reordering
of the other. -/
theorem frame_u32_multiset_eq_iff (s : List U8) (v₁ v₂ : alloc.vec.Vec U32) :
    framing.frame_u32_multiset log s v₁ ⦃ r₁ => framing.frame_u32_multiset log s v₂ ⦃ r₂ =>
      (r₁ = r₂ ↔ v₁.val.Perm v₂.val) ⦄ ⦄ := by
  apply WP.spec_mono (frame_u32_multiset_spec s v₁)
  intro r₁ h₁
  apply WP.spec_mono (frame_u32_multiset_spec s v₂)
  intro r₂ h₂
  rw [h₁, h₂, List.append_cancel_left_eq]
  exact Antecedent.Framing.multiset_eq_iff u32Order count_prefixFree u32_prefixFree
    (length_lt_two_pow_64 _ v₁.property) (length_lt_two_pow_64 _ v₂.property)
    (fun _ _ => trivial) (fun _ _ => trivial)

theorem u32_multiset_dup_ne {x : U32} {xs : List U32} (hl : xs.length + 1 < 2 ^ 64) :
    Antecedent.Framing.multiset u32Order countBytes u32Bytes (x :: xs) ≠
      Antecedent.Framing.multiset u32Order countBytes u32Bytes xs :=
  Antecedent.Framing.multiset_dup_ne u32Order count_prefixFree u32_prefixFree hl trivial
    (fun _ _ => trivial)

/-- `frame_real_bits` writes different bytes for different bit patterns; in particular for
`-0.0` and `0.0`, and for NaNs with different payloads. -/
theorem frame_real_bits_eq_iff (s : List U8) (a b : U64) :
    framing.frame_real_bits log s a ⦃ r₁ => framing.frame_real_bits log s b ⦃ r₂ =>
      (r₁ = r₂ ↔ a = b) ⦄ ⦄ := by
  apply WP.spec_mono (frame_real_bits_spec s a)
  intro r₁ h₁
  apply WP.spec_mono (frame_real_bits_spec s b)
  intro r₂ h₂
  rw [h₁, h₂, List.append_cancel_left_eq]
  exact ⟨wordBytes_inj, fun h => h ▸ rfl⟩

/-- `frame_presence` writes different bytes for present and absent. -/
theorem frame_presence_eq_iff (s : List U8) (a b : Bool) :
    framing.frame_presence log s a ⦃ r₁ => framing.frame_presence log s b ⦃ r₂ =>
      (r₁ = r₂ ↔ a = b) ⦄ ⦄ := by
  apply WP.spec_mono (frame_presence_spec s a)
  intro r₁ h₁
  apply WP.spec_mono (frame_presence_spec s b)
  intro r₂ h₂
  rw [h₁, h₂, List.append_cancel_left_eq]
  exact ⟨presence_prefixFree.inj trivial trivial, fun h => h ▸ rfl⟩

end AntecedentRust.Framing
