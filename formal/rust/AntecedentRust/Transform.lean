/-
The production obligation decisions of `TransformationReport::compose`, as Aeneas
translates them from Rust, equal the specification in `Antecedent.Contract`, so
the obligation rule the composition theorems are proved about is the one the
code runs.

Trusted: Charon and Aeneas translate the Rust faithfully, and the Aeneas model of
the standard-library operations the kernels use (enum discriminant reads, boolean
operations) is correct. The loop that applies the decisions to the record list
is modeled (`addObligationRust`), not translated.
-/
import Aeneas
import Antecedent.Contract
import AntecedentRust.Extracted.Transform.Funs

open Aeneas Aeneas.Std Result
open AntecedentRust.Extracted.Transform
open Antecedent.Contract

namespace AntecedentRust.Transform

/-- `ObligationKind` as the specification names it. -/
def kindSpec : obligation.ObligationKind → ObligationKind
  | .UserAssertion => .userAssertion
  | .GraphicalImplication => .graphicalImplication
  | .EmpiricalDiagnostic => .empiricalDiagnostic
  | .FailedCheck => .failedCheck
  | .Uncheckable => .uncheckable
  | .CheckNotRun => .checkNotRun

/-- `AssumptionStatus` as the specification names it. -/
def statusSpec : antecedent_core.assumption.AssumptionStatus → AssumptionStatus
  | .Declared => .declared
  | .Supported => .supported
  | .Contradicted => .contradicted
  | .Untestable => .untestable

/-- `ObligationMerge` as the specification names it. -/
def mergeSpec : transform.ObligationMerge → Merge
  | .Skip => .skip
  | .ReplaceResolved => .replaceResolved
  | .Append => .append

/-- `obligation_merge` is the specification's decision and never fails. -/
theorem obligation_merge_spec (identical sameId unresolved : Bool) :
    ∃ m, transform.obligation_merge identical sameId unresolved = ok m ∧
      mergeSpec m = obligationMerge identical sameId unresolved := by
  cases identical <;> cases sameId <;> cases unresolved <;>
    simp [transform.obligation_merge, obligationMerge, mergeSpec]

/-- `obligation_kept_on_replace` is the specification's rule. -/
theorem obligation_kept_on_replace_spec (sameId unresolved : Bool) :
    transform.obligation_kept_on_replace sameId unresolved = ok (keptOnReplace sameId unresolved) := by
  cases sameId <;> cases unresolved <;> simp [transform.obligation_kept_on_replace, keptOnReplace]

/-- `obligation_unresolved` is the specification's unresolved rule. -/
theorem obligation_unresolved_spec (k : obligation.ObligationKind)
    (s : antecedent_core.assumption.AssumptionStatus) :
    obligation.obligation_unresolved k s = ok (obligationUnresolved (kindSpec k) (statusSpec s)) := by
  cases k <;> cases s <;>
    simp [obligation.obligation_unresolved, obligation.ObligationKind.is_unresolved, kindSpec,
      statusSpec, obligationUnresolved, ObligationKind.unresolved]

/-- Keep the elements a fallible predicate accepts, in order. -/
def filterR (p : Obligation → Result Bool) : List Obligation → Result (List Obligation)
  | [] => ok []
  | e :: es => do
    let b ← p e
    let rest ← filterR p es
    ok (if b then e :: rest else rest)

theorem filterR_ok (f : Obligation → Bool) :
    ∀ l, filterR (fun e => ok (f e)) l = ok (l.filter f)
  | [] => rfl
  | e :: es => by
    cases h : f e <;> simp [filterR, filterR_ok f es, h]

/-- One later obligation added with the decisions the production kernels return. -/
def addObligationRust (acc : List Obligation) (o : Obligation) : Result (List Obligation) := do
  let m ← transform.obligation_merge (acc.contains o) (acc.any fun e => e.id == o.id) o.unresolved
  match m with
  | .Skip => ok acc
  | .ReplaceResolved => do
    let kept ← filterR (fun e => transform.obligation_kept_on_replace (e.id == o.id) e.unresolved) acc
    ok (kept ++ [o])
  | .Append => ok (acc ++ [o])

/-- The production decisions add an obligation exactly as the specification does. -/
theorem addObligationRust_eq (acc : List Obligation) (o : Obligation) :
    addObligationRust acc o = ok (addObligation acc o) := by
  rw [← addObligationBy_eq]
  obtain ⟨m, hm, hspec⟩ :=
    obligation_merge_spec (acc.contains o) (acc.any fun e => e.id == o.id) o.unresolved
  have hkept : (fun e : Obligation =>
      transform.obligation_kept_on_replace (e.id == o.id) e.unresolved) =
      fun e => ok (keptOnReplace (e.id == o.id) e.unresolved) := by
    funext e
    exact obligation_kept_on_replace_spec _ _
  simp only [addObligationRust, addObligationBy, hm, ← hspec, hkept, filterR_ok]
  cases m <;> simp [mergeSpec]

/--
With the production decisions, adding a later obligation keeps every unresolved
earlier one and keeps the later one when it is unresolved.
-/
theorem rust_compose_keeps_unresolved (acc : List Obligation) (o e : Obligation)
    (he : e ∈ acc ∨ e = o) (hu : e.unresolved = true) :
    ∃ out, addObligationRust acc o = ok out ∧ e ∈ out := by
  refine ⟨_, addObligationRust_eq acc o, ?_⟩
  rcases he with h | rfl
  · exact addObligation_keeps_unresolved acc o e h hu
  · exact addObligation_adds_unresolved acc e hu

end AntecedentRust.Transform
