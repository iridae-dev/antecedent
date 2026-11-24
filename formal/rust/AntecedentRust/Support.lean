/-
The production support severity order, as Aeneas translates it from Rust, is the
specification's in `Antecedent.Revalidation`, so the summary rule proved there
orders labels as the code does.

Trusted: Charon and Aeneas translate the Rust faithfully, and the Aeneas model of
enum discriminant reads and u8 values is correct.
-/
import Aeneas
import Antecedent.Revalidation
import AntecedentRust.Extracted.Support.Funs

open Aeneas Aeneas.Std Result
open AntecedentRust.Extracted.Support
open Antecedent.Revalidation

namespace AntecedentRust.Support

/-- `SupportStatus` as the specification names it. -/
def supportSpec : response.SupportStatus → Support
  | .Supported => .supported
  | .WeakOverlap => .weakOverlap
  | .Extrapolative => .extrapolative
  | .OutsideEmpiricalSupport => .outsideEmpiricalSupport
  | .MissingEvidence => .missingEvidence

/-- `SupportStatus::severity` is the specification's severity and never fails. -/
theorem severity_spec (s : response.SupportStatus) :
    ∃ r, response.SupportStatus.severity s = ok r ∧ r.val = (supportSpec s).severity := by
  cases s <;> simp [response.SupportStatus.severity, supportSpec, Support.severity]

end AntecedentRust.Support
