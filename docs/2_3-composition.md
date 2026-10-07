# Composing sources, recalculating and exporting a decision (2.3, milestone C)

Milestone C joins the pieces of [milestone B](2_3-decisions-breadth.md) and the
external claims of [external scientific objects](2_3-external-science.md) into
one decision, recalculates it selectively, and exports it as one portable
bundle. Every route here is **point-only**. Nothing on this page is an interval,
a coverage statement or a probability license. Any standard error, p-value,
interval or Monte Carlo error carried inside a composed artifact is a
diagnostic with calibration unmeasured, and the composition never changes that
label.

For each route: the question it answers, its scope bounds, claim label, Python
entry point and refusal namespace.

## Program binding

- **Question.** Is an external claim bound to the actual identified program and
  request it answers, rather than to a graph alone?
- **Scope.** A program names graph, checked contract, treatment, outcome, target
  population, intervention kind, horizon, a strictly increasing dose grid with
  units, outcome units, functional and transform. One BLAKE3 identity covers all
  of them. A substituted treatment or outcome, another population, a changed
  dose grid, an incompatible quantities override, a changed graph or contract,
  and a graph-only identity are refused. Nothing is converted or rescaled, and
  the bound claim keeps its external trust.
- **Claim.** Point-only.
- **Python.** `antecedent.program_claims`: `bind_to_program`,
  `check_external_program`, `ProgramBinding`.
- **Refusals.** `program_binding.*`; the underlying result binding keeps
  `external_response_binding.*`.

## Native response claims

- **Question.** Can a native mean-curve response enter a decision as a typed
  claim whose coordinates, support, trust and calibration travel with it,
  without a mean ever becoming an outcome law?
- **Scope.** One coordinate per dose with a support label, native provider trust
  and an unmeasured calibration. A coordinate outside empirical support or
  lacking evidence is withheld and reported, not used. A caller cannot assert a
  calibrated status. The joint law a Rust response can supply is a posterior
  over the mean curve, not an interventional outcome law.
- **Claim.** Point-only.
- **Python.** `antecedent.program_claims`: `native_claim`, `NativeClaim`,
  `NativeDecisionSource`.
- **Refusals.** `native_claims.*`.
- **Limit.** A Python native claim is mean-only because the Python response view
  keeps no draws.

## Composition boundary

- **Question.** Can a decision input carry its support, trust and capabilities
  from evidence only, can each action be judged on its own coordinates, and can
  sources be combined only under a declared operation that refuses shared or
  unknown dependence?
- **Scope.** Trust comes only from evidence: an artifact labelled native or exact
  is stored as unverified unless a matching native execution record or an
  exact-request receipt is supplied. An action that reads an unsupported
  coordinate is reported as unsupported with its reason while the others are
  compared; all actions unsupported is a state, not an error. Independent
  pooling and paired draws are refused when inputs share data, a prior or a
  fitted model, when dependence is unknown, or when inputs declared independent
  carry one snapshot, unless a licensed covariance or joint-law route names one
  of the pair. Conflicting atoms are not averaged.
- **Claim.** Point-only. Relations and dependence routes are caller declarations
  the checks read; nothing is estimated or pooled.
- **Python.** `antecedent.composition`: `DecisionInput`, `Functional`,
  `evaluate_functional`, `DependenceRoute`, `EvidenceRelation`.
- **Refusals.** `composition_boundary.*`; support evaluation passes through
  `coordinate_support.*` and the decision evaluator's `decision_evaluation.*`.

## Recalculation plan and receipt

- **Question.** After a change, which stages can be reused and which must be
  recomputed, and can the reuse be proven by counted work?
- **Scope.** The stage model covers data, identification, fold fits, score table,
  law, target population, utility and decision, plus plan-level prior and
  provider stages. The plan names for each stage its status and the dependency
  that determined it. Reuse is proven by counted cross-fitted fold fits: a first
  run does 10 fold fits (5 folds, 2 nuisance sets), a utility-only change does
  0, and a compatible target-weight change under a declared retarget does 0.
  Without a declared retarget licence the change refits. In a fresh process no
  derived stage is reused unless the caller supplies a portable artifact. The
  per-call cache is not a persistent fit cache, and a second session fits
  again. Only the cross-fitted AIPW route with a net-benefit rule is driven.
- **Claim.** Point-only: the plan table, the counted work, and the point law and
  net benefit of that route. Its law standard error is a diagnostic with
  calibration unmeasured.
- **Python.** `antecedent.recalc`: `plan_recalculation`, `RecalcSession`,
  `RecalcRequest`, `Declaration`, `Capabilities`.
- **Refusals.** `recalc.*`.

The receipt of a run is a separate portable artifact (`recalc_receipt_v1`). A
consumer recomputes the plan from the stored declarations and requires the
stored status, counts, totals and both identities to match; an edit is refused
even when the container is resealed. A loaded receipt never claims reuse: any
derived stage stored as reused under a fresh-process boundary is refused.
Counts are the producer's and cannot be re-observed. The receipt is a record,
not a certificate that a cache exists.

- **Python.** `antecedent.recalc`: `consume`, `RecalcReceipt`.
- **Refusals.** `recalc_receipt.*`.

## Composition bundle and verifiers

- **Question.** Can a composed decision be exported as one portable bundle that
  an independent consumer verifies node by node?
- **Scope.** A typed directed acyclic graph of nodes, with each edge carrying the
  Merkle digest of its upstream node, so a changed upstream identity changes
  every dependent digest and the bundle identity. Declaration order does not
  change the identity. Each embedded node is verified through its artifact's own
  consumer, with binding facts read from the decoded fields and never from text
  a producer wrote beside it. A reference node is verified only as matched a
  supplied source and is never replayed. A kind with no verifier (causal
  contract, attestation, quantity coordinates, transformation) fails and must
  travel as a reference. A consumer that retains the bundle identity
  independently refuses any other; without one it checks only internal
  consistency.
- **Claim labels.** A decision over an aligned joint law is labelled `joint_draw`.
  A decision resting on an external mean is labelled `point_only_attested`, its
  declared mean is shown but is not a verified value, and a decision that needs
  a joint law over a mean-only claim fails `unsupported_law`.
- **Claim.** Point-only: the bundle certifies decoding, identities and bindings.
- **Python.** `antecedent.composition_bundle`: `BundleBuilder`, `Bundle`,
  `consume_bundle`, `describe_artifact`.
- **Refusals.** `composition_bundle.*` (shared by the bundle and the verifiers).

## The multi-source acceptance story and its limits

The acceptance record `2.3C.C4.multi_source_acceptance` composes only the routes
above and the B routes they depend on. It adds no estimator and no new
artifact.

**Setup.** Graph `x -> a`, `x -> y`, `a -> y`; a program for `y` under `do(a)`
over doses 0, 1 and 2. Three actions read the mean at each dose: wait reads
m0, treat reads m1 minus 1, extend reads m2 minus 3.

- A native claim N has means 2, 4 and 9; dose 2 is unsupported.
- External E1 is attested, with means 1, 3 and 5.5.
- External E2 is bound to the program, with means 2, 3.5 and 5, and a joint law
  of two rows, (2, 6, 3) with weight 0.25 and (2, 1, 7) with weight 0.75.
- E1 and E2 are declared independent. N and E2 share data (registry 7), so they
  are not.

**Hand-derived values.** These are the numbers the tests assert.

| Source | wait | treat | extend |
| --- | --- | --- | --- |
| N alone | 2 | 3 | unsupported |
| N then E1 | 2 | 3 | 2.5 |
| E2 means | 2 | 2.5 | 2 |
| E2 law, E[U] | 2 | 2.5 | 2 |

With N alone, treat is best among the supported actions; with E1 added, treat
remains optimal. Over the joint law, the utilities are (2, 5, 0) and (2, 0, 4),
so E[U] is (2, 2.5, 2), EVPI is 2 (1 with extend masked), the expected regrets
are (2.5, 2, 2.5) and P(U_treat >= 5) is 0.5. A probability or quantile is
answered only by the joint law and is refused over a mean. The design ranking
gives EVPI 1/8, EVSI 1/16 and net values 0.0425 and 0.0125. The composed bundle
labels the law decision `joint_draw` and the mean-only decision
`point_only_attested`, and a fresh interpreter consumes it and reads the same
values.

**Mutation table.** Each change to a binding fact is either refused at its named
boundary or recalculates only the stages that depend on it.

| Mutation | Outcome |
| --- | --- |
| Quantity | Refused at the program binding (`program_binding.*`, `external_response_binding.*`); a tampered quantity in a bundle fails the dependent decision. |
| Law | A mean, a marginal or another meaning never stands in for the joint law; a bundle joint decision over a mean-only claim alone fails `unsupported_law`. |
| Structure | A changed graph or premise is refused at the program binding and re-identifies. |
| Support | Support withdrawn from a study excludes only the affected action. |
| Snapshot | Dependents recompute, the data refits once and equals an independent rerun, and the external branches are kept. |
| Provider operation | An unknown, changed or unavailable operation is refused. |
| Evidence overlap | Overlap is never pooled or counted twice; a bundle that declares independence over overlapping evidence fails. |
| Utility | Only the decision recomputes, with zero fit work. |

**Limits found by the acceptance.**

- Python cannot state a native execution record, so a Python native claim enters
  composition as unverified.
- A native response retains no draws, so a probability or quantile comes only
  from an external joint law.
- The recalculation stage model has no graph or query edge into external-study
  branches. Only the binding refusals protect them; a changed external study
  leaves unrelated branches valid but is not tracked by the stage model.
- The design ranking is bound to the joint law only by its source digest.

**Registry note.** The acceptance record declares two details of the
`composition_bundle` namespace (`unsupported_law` and
`expected_identity_mismatch`) because it adds no refusal of its own. The other
refusals the story observes keep their owning records.
