-- The model of the items extract.sh translates as opaque (`--opaque`). Written by hand and
-- kept across retranslation; the proofs in AntecedentRust/Framing.lean rely on it.
--
-- `sort_digests` and `sort_u32s` call the standard library's `sort_unstable`, which sorts
-- ascending under `Ord`: lexicographic on bytes for `[u8; 32]`, numeric for `u32`. Under
-- those orders equal elements are identical, so every ascending sort of a slice returns the
-- same slice; the model returns it through `List.mergeSort`. Trusted: `sort_unstable` sorts.
module
public import Aeneas
public import AntecedentRust.Extracted.Framing.Types
@[expose] public section
open Aeneas Aeneas.Std Result

namespace AntecedentRust.Extracted.Framing

/-- Lexicographic `≤` on byte values: the order of `Ord` on `[u8; N]`. -/
def bytesLe : List Nat → List Nat → Bool
  | [], _ => true
  | _ :: _, [] => false
  | a :: xs, b :: ys => decide (a < b) || (decide (a = b) && bytesLe xs ys)

/-- `Ord` on `[u8; 32]` as a Boolean `≤`. -/
def digestLe (a b : Array Std.U8 32#usize) : Bool :=
  bytesLe (a.val.map (·.val)) (b.val.map (·.val))

/-- `Ord` on `u32` as a Boolean `≤`. -/
def u32Le (a b : Std.U32) : Bool := decide (a.val ≤ b.val)

/-- [antecedent_core::framing::sort_digests]: `<[[u8; 32]]>::sort_unstable`. -/
def framing.sort_digests (s : Slice (Array Std.U8 32#usize)) :
    Result (Slice (Array Std.U8 32#usize)) :=
  ok (Slice.from (s.val.mergeSort digestLe) (by simp [List.length_mergeSort]))

/-- [antecedent_core::framing::sort_u32s]: `<[u32]>::sort_unstable`. -/
def framing.sort_u32s (s : Slice Std.U32) : Result (Slice Std.U32) :=
  ok (Slice.from (s.val.mergeSort u32Le) (by simp [List.length_mergeSort]))

end AntecedentRust.Extracted.Framing
