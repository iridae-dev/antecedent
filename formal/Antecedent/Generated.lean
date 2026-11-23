/-
Generated cases against the specifications, checked with Plausible when this file
builds. Each property compares a specification with an independent statement of the
same rule, so a mistake in either shows as a counterexample and fails the build.
These are tests, not proofs: the theorems in the specification files are the proofs.
-/
import Plausible
import Antecedent.Standing
import Antecedent.Identity
import Antecedent.Acceptance

open Plausible

namespace Antecedent.Generated

open Antecedent.Standing

deriving instance Plausible.Arbitrary for Domain
deriving instance Plausible.Arbitrary for Identification

instance : Plausible.Shrinkable Domain := ⟨fun _ => []⟩
instance : Plausible.Shrinkable Identification := ⟨fun _ => []⟩
instance : Repr Domain := inferInstance

/-- Natural numbers under `≤`, the element order of the identity cases. -/
def natOrder : Antecedent.Identity.TotalOrder Nat where
  le a b := decide (a ≤ b)
  total a b := by simp [Nat.le_total]
  trans a b c h₁ h₂ := by simp_all; omega
  antisymm a b h₁ h₂ := by simp_all; omega

/-- The relay domain rule, stated independently as a table of what may change. -/
def relayTable : Domain → Domain → Bool
  | .identified, .identified | .supported, .supported | .evaluated, .evaluated
  | .outsideScope, .outsideScope | .contradicted, .contradicted | .unknown, .unknown
  | .limited, .limited | .notClaimed, .notClaimed => true
  | .identified, .unknown | .supported, .unknown | .evaluated, .unknown => true
  | _, _ => false

-- Meaning-preserving and invalid domain changes: the rule matches the table.
#eval Testable.check (∀ f t : Domain, domainChangeAllowed f t = relayTable f t)

-- A derivation may never report a stronger identification than its parent.
#eval Testable.check (∀ p d : Identification,
  identificationKeptOrLowered p d = true → d.rank ≤ p.rank)

-- Reordering a declaration list keeps its stream; a different multiset changes it.
#eval Testable.check (∀ xs : List Nat,
  Antecedent.Identity.encodeSet natOrder xs.reverse = Antecedent.Identity.encodeSet natOrder xs)
#eval Testable.check (∀ xs ys : List Nat,
  (Antecedent.Identity.encodeSet natOrder xs = Antecedent.Identity.encodeSet natOrder ys) =
    xs.isPerm ys)

-- Moving the last element of one framed list to the front of the other changes the stream.
#eval Testable.check (∀ (xs ys : List Nat) (x : Nat),
  Antecedent.Identity.encodeTwo natOrder (xs ++ [x]) ys ≠
    Antecedent.Identity.encodeTwo natOrder xs (x :: ys))

end Antecedent.Generated
