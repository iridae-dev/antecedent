/-
Scientific quantities, assumptions, obligations, what a contract means, and how
transformation reports compose.

A contract states a claim about a quantity that holds whenever its premises hold.
Its obligations record which premises are not yet discharged. A transformation of
a contract is described by a report: per semantic layer, what the transformation
did (preserved, weakened, invalidated, refused, ...), and the obligations it
introduced or kept. This file proves that the meaning of a contract depends on
every premise (dropping one is not sound in general), and that composing
transformation reports never drops an unresolved obligation, never erases an
effect recorded on a layer, and never clears a refusal: a chain of
transformations cannot end with a stronger standing than any step left it.

Rust counterparts:
* `Quantity`           ↔ `antecedent_contract::quantity::ScientificQuantity`
* `AssumptionStatus`   ↔ `antecedent_core::assumption::AssumptionStatus`
* `ObligationKind`     ↔ `antecedent_contract::obligation::ObligationKind`
* `Obligation.unresolved` ↔ `ObligationRecord::is_unresolved`
* `Report.compose`     ↔ `antecedent_contract::transform::TransformationReport::compose`
* `obligationMerge`, `keptOnReplace`, `obligationUnresolved` ↔
  `antecedent_contract::transform::{obligation_merge, obligation_kept_on_replace}` and
  `antecedent_contract::obligation::obligation_unresolved` (proved equal in `formal/rust`)
The composition is exercised on the production code by the adversarial campaign
(`parity/adversarial_campaign.toml`, `contract.*` cases).
-/

namespace Antecedent.Contract

/-- What a number is about. Every field is part of its meaning: a change of population,
regime, horizon, functional, conditioning or transform is a different quantity. -/
structure Quantity where
  variableId : String
  units : String
  population : String
  regime : String
  horizon : Nat
  functional : String
  conditioning : List (String × String)
  transform : String
  deriving DecidableEq

/-- Status of an assumption. -/
inductive AssumptionStatus where
  | declared
  | supported
  | contradicted
  | untestable
  deriving DecidableEq, Repr

/-- How an obligation came about. -/
inductive ObligationKind where
  | userAssertion
  | graphicalImplication
  | empiricalDiagnostic
  | failedCheck
  | uncheckable
  | checkNotRun
  deriving DecidableEq, Repr

def ObligationKind.unresolved : ObligationKind → Bool
  | .checkNotRun | .failedCheck => true
  | _ => false

/-- One obligation a contract or claim carries. -/
structure Obligation where
  id : String
  kind : ObligationKind
  status : AssumptionStatus
  deriving DecidableEq

/-- Unresolved: its check did not run or failed, or its assumption is only declared or is
contradicted. -/
def Obligation.unresolved (o : Obligation) : Bool :=
  o.kind.unresolved || o.status == .declared || o.status == .contradicted

/-- A contract: a claim about a quantity, under premises. -/
structure Contract (World : Type) where
  quantity : Quantity
  premises : List (World → Prop)
  claim : World → Prop

/-- What a contract means: in every world where all its premises hold, its claim holds. -/
def Contract.Means {W : Type} (c : Contract W) : Prop :=
  ∀ w, (∀ p ∈ c.premises, p w) → c.claim w

/-- Adding a premise keeps a contract's meaning. -/
theorem Contract.means_add_premise {W : Type} {c : Contract W} (h : c.Means) (p : W → Prop) :
    ({ c with premises := p :: c.premises } : Contract W).Means := by
  intro w hw
  exact h w (fun q hq => hw q (List.mem_cons_of_mem _ hq))

/--
Dropping a premise is not sound in general: there is a contract that means what it
says whose claim fails once a premise is removed. A transformation that loses a
required premise must refuse, or carry it as an unresolved obligation.
-/
theorem Contract.drop_premise_unsound :
    ∃ (c : Contract Bool) (p : Bool → Prop), c.premises = [p] ∧ c.Means ∧
      ¬ ({ c with premises := [] } : Contract Bool).Means := by
  let q : Quantity := ⟨"", "", "", "", 0, "", [], ""⟩
  refine ⟨⟨q, [fun w => w = true], fun w => w = true⟩, fun w => w = true, rfl, ?_, ?_⟩
  · intro w hw
    exact hw _ (List.mem_singleton.2 rfl)
  · intro h
    have := h false (by simp)
    simp at this

/-- The semantic layers a transformation can affect. -/
inductive Layer where
  | target | identification | program | inference | support | data | execution | results
  deriving DecidableEq, Repr

/-- What a transformation did to a layer. Several effects may apply to one layer. -/
inductive Effect where
  | preserves | weakens | requiresReidentification | requiresReestimation | invalidates | refused
  deriving DecidableEq, Repr

/-- A transformation report: per-layer effects and the obligations it carries. -/
structure Report where
  effects : List (Layer × Effect)
  obligations : List Obligation

def Report.effectsOn (r : Report) (l : Layer) : List Effect :=
  (r.effects.filter (fun e => e.1 == l)).map (·.2)

def Report.refused (r : Report) : Bool :=
  r.effects.any (fun e => e.2 == .refused)

/-- Adding one later obligation, as `TransformationReport::compose` does. -/
def addObligation (acc : List Obligation) (o : Obligation) : List Obligation :=
  if acc.contains o then acc
  else if !o.unresolved && acc.any (fun e => e.id == o.id) then acc
  else if o.unresolved then (acc.filter (fun e => e.id != o.id || e.unresolved)) ++ [o]
  else acc ++ [o]

/-- `r` then `s`: effects are unioned per layer and obligations added in order. -/
def Report.compose (r s : Report) : Report where
  effects := r.effects ++ s.effects
  obligations := s.obligations.foldl addObligation r.obligations

theorem addObligation_keeps_unresolved (acc : List Obligation) (o e : Obligation)
    (he : e ∈ acc) (hu : e.unresolved = true) : e ∈ addObligation acc o := by
  unfold addObligation
  split
  · exact he
  split
  · exact he
  split
  · exact List.mem_append_left _ (List.mem_filter.2 ⟨he, by simp [hu]⟩)
  · exact List.mem_append_left _ he

theorem addObligation_adds_unresolved (acc : List Obligation) (o : Obligation)
    (hu : o.unresolved = true) : o ∈ addObligation acc o := by
  by_cases h : acc.contains o
  · simp only [addObligation, h, ↓reduceIte]
    exact List.contains_iff_mem.1 h
  · have hm : o ∉ acc := fun hm => h (List.contains_iff_mem.2 hm)
    simp [addObligation, hu, hm]

theorem foldl_keeps_unresolved (os acc : List Obligation) (e : Obligation)
    (he : e ∈ acc) (hu : e.unresolved = true) : e ∈ os.foldl addObligation acc := by
  induction os generalizing acc with
  | nil => exact he
  | cons o os ih => exact ih _ (addObligation_keeps_unresolved acc o e he hu)

theorem foldl_adds_unresolved (os acc : List Obligation) (e : Obligation)
    (he : e ∈ os) (hu : e.unresolved = true) : e ∈ os.foldl addObligation acc := by
  induction os generalizing acc with
  | nil => simp at he
  | cons o os ih =>
    rcases List.mem_cons.1 he with rfl | h
    · exact foldl_keeps_unresolved os _ e (addObligation_adds_unresolved acc e hu) hu
    · exact ih _ h

/-- The unresolved rule as the Rust check states it (`obligation_unresolved`). -/
def obligationUnresolved (k : ObligationKind) (s : AssumptionStatus) : Bool :=
  k.unresolved || s == .declared || s == .contradicted

theorem Obligation.unresolved_eq (o : Obligation) :
    o.unresolved = obligationUnresolved o.kind o.status := rfl

/-- What composition does with one later obligation (`ObligationMerge`). -/
inductive Merge where
  | skip
  | replaceResolved
  | append
  deriving DecidableEq, Repr

/-- The decision as the Rust check states it (`obligation_merge`). -/
def obligationMerge (identical sameId unresolved : Bool) : Merge :=
  if identical || (!unresolved && sameId) then .skip
  else if unresolved then .replaceResolved
  else .append

/-- Which earlier record survives a replacement, as the Rust check states it. -/
def keptOnReplace (sameId unresolved : Bool) : Bool := !sameId || unresolved

/-- One later obligation added as the production loop adds it, from the two decisions. -/
def addObligationBy (acc : List Obligation) (o : Obligation) : List Obligation :=
  match obligationMerge (acc.contains o) (acc.any fun e => e.id == o.id) o.unresolved with
  | .skip => acc
  | .replaceResolved => acc.filter (fun e => keptOnReplace (e.id == o.id) e.unresolved) ++ [o]
  | .append => acc ++ [o]

/-- The production decisions add an obligation exactly as `addObligation` does. -/
theorem addObligationBy_eq (acc : List Obligation) (o : Obligation) :
    addObligationBy acc o = addObligation acc o := by
  unfold addObligationBy addObligation obligationMerge keptOnReplace
  cases hc : acc.contains o <;> cases hu : o.unresolved <;>
    cases ha : (acc.any fun e => e.id == o.id) <;> simp [bne, *]

/-- Composition never drops an unresolved obligation of either report. -/
theorem compose_keeps_unresolved (r s : Report) (e : Obligation)
    (he : e ∈ r.obligations ∨ e ∈ s.obligations) (hu : e.unresolved = true) :
    e ∈ (r.compose s).obligations := by
  rcases he with h | h
  · exact foldl_keeps_unresolved _ _ e h hu
  · exact foldl_adds_unresolved _ _ e h hu

/-- Composition records exactly the effects either report recorded on each layer. -/
theorem compose_effects (r s : Report) (l : Layer) (x : Effect) :
    x ∈ (r.compose s).effectsOn l ↔ x ∈ r.effectsOn l ∨ x ∈ s.effectsOn l := by
  simp [Report.compose, Report.effectsOn, List.filter_append]

/-- A refusal anywhere in a composition refuses it. -/
theorem compose_refused (r s : Report) (h : r.refused = true ∨ s.refused = true) :
    (r.compose s).refused = true := by
  simp only [Report.refused, Report.compose, List.any_append, Bool.or_eq_true] at *
  exact h

/-- A chain of transformations, composed left to right from a first report. -/
def chain (first : Report) (rest : List Report) : Report :=
  rest.foldl Report.compose first

/-- No unresolved obligation of any step of a chain is dropped. -/
theorem chain_keeps_unresolved (first : Report) (rest : List Report) (e : Obligation)
    (he : e ∈ first.obligations ∨ ∃ r ∈ rest, e ∈ r.obligations) (hu : e.unresolved = true) :
    e ∈ (chain first rest).obligations := by
  induction rest generalizing first with
  | nil =>
    rcases he with h | ⟨r, hr, _⟩
    · exact h
    · simp at hr
  | cons r rest ih =>
    apply ih
    rcases he with h | ⟨r', hr', h'⟩
    · exact Or.inl (compose_keeps_unresolved _ _ e (Or.inl h) hu)
    · rcases List.mem_cons.1 hr' with rfl | hmem
      · exact Or.inl (compose_keeps_unresolved _ _ e (Or.inr h') hu)
      · exact Or.inr ⟨r', hmem, h'⟩

/-- An effect recorded by any step of a chain is recorded by the chain. -/
theorem chain_keeps_effects (first : Report) (rest : List Report) (l : Layer) (x : Effect)
    (hx : x ∈ first.effectsOn l ∨ ∃ r ∈ rest, x ∈ r.effectsOn l) :
    x ∈ (chain first rest).effectsOn l := by
  induction rest generalizing first with
  | nil =>
    rcases hx with h | ⟨r, hr, _⟩
    · exact h
    · simp at hr
  | cons r rest ih =>
    apply ih
    rcases hx with h | ⟨r', hr', h'⟩
    · exact Or.inl ((compose_effects _ _ _ _).2 (Or.inl h))
    · rcases List.mem_cons.1 hr' with rfl | hmem
      · exact Or.inl ((compose_effects _ _ _ _).2 (Or.inr h'))
      · exact Or.inr ⟨r', hmem, h'⟩

end Antecedent.Contract
