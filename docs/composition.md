# Compose sources into one decision

Combine native results, external claims and independently fitted mechanisms into one decision, recalculate only what a change affects, and export the result as one portable bundle. The routes join [decisions, design and repair](decisions.md#routes-at-a-glance) with the
external claims of [external scientific objects](guides/build/scientific-interfaces.md). Every route here is **point-only**. Nothing on this page is an interval,
a coverage statement or a probability license. Uncertainty carried inside a composed artifact keeps its source standing.
Composition supplies no additional calibration or interval guarantee.

Each route below gives the question it answers, its scope bounds, claim label, Python
entry point and refusal namespace. The measured-spread routes are in
[composed uncertainty](guides/composed-uncertainty.md); to check which downstream claims survive a change see [composition verification](composition-verification.md).

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
- **Limit.** A mean-only response remains mean-only. Actual native producers
  may retain aligned credible rows; these describe uncertainty over the mean
  response, rather than an interventional outcome law.

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
  again. The `RecalcSession` example uses cross-fitted AIPW with a net-benefit
  rule. Separate checked sessions support adjusted regression, doubly robust
  effects, static transport, Bayesian summaries, temporal analysis,
  design-specific effects and bounded external mean callbacks; see
  [recalculation capabilities](recalculation-capabilities.md).
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

Build, export and consume a bundle:

```python
from antecedent import composition_bundle as cb

result_bytes = cb.mean_decision(contract, claim)     # point-only attested result
builder = cb.Bundle.builder()
builder.add_artifact("external_claim", claim, node_id="claim")
builder.add_artifact("decision_contract", contract.export(artifact_id="contract"), node_id="contract")
builder.add_artifact("decision_result", result_bytes, node_id="result")
bundle = builder.connect("claim", "result").connect("contract", "result").build()

data, identity = bundle.export(artifact_id="my-bundle"), bundle.identity

# In another process, under the identity retained independently of the bytes:
consumed = cb.consume_bundle(data, expected_identity=identity)
consumed.require_verified()
consumed.claim_label                      # "point_only_attested"
consumed.value("result", "treat.expected_utility")   # 9.0
```

`contract` and `claim` are the decision contract and bound external claim of
[evaluating actions with a decision contract](decisions.md). `add_artifact(artifact)`
infers the node kind from the container when you do not name it, and an artifact
object whose `export(artifact_id=...)` returns container bytes may be passed instead
of bytes. A bundle never claims that a serialized result recreates an executable study.

## A multi-source example

This example combines native and external responses, a joint-law decision and
study ranking. Each source keeps its own scientific standing.

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

**Limits of this example.**

- Its mean-only native response supplies no probability or quantile. A producer
  that retains aligned credible rows can supply a posterior over its mean
  response; those rows still do not supply a predictive outcome law.
- `native_claim` requires actual producer-issued native execution authority.
  Handcrafted response views and caller-supplied trust labels cannot supply it.
- The declarative planner compares identities; executing family sessions bind
  their own checked inputs and measure actual work. A changed external branch
  must be rebound and executed under its provider's replay policy.
- A ranking and a joint-law decision retain their source identities. Binding
  them together does not validate the utility or the generating model.

## Compose independently fitted mechanisms

Individually valid conditional mechanisms do not imply a valid joint model. Composition compiles *declared* mechanisms into one executable joint law and refuses, with a structured finding, whenever the declarations do not establish a coherent one. Nothing is inferred from names or fits.

### Boundary

| Item | Supported | Refused |
| --- | --- | --- |
| Structure | explicitly mapped acyclic graphs | any directed cycle (before execution) |
| Family | additive-noise linear-Gaussian with declared parents and declared unit-variance Gaussian shared latents; or finite conditional tables | mixed families, nonlinear or non-additive mechanisms |
| Execution | closed-form joint mean/covariance and `do()` law; exact enumeration and truncated factorization; structural sampling | conditioning on observations (a separate query) |
| Uncertainty | first-order delta-method variance of a linear contrast of interventional means, from a declared parameter law | intervals, posteriors, variance functionals, any claim over several mechanisms with unknown dependence |

Residual `sigma` of a linear-Gaussian mechanism is the *idiosyncratic* part;
a declared latent adds `loading^2` to the total residual variance and
`loading_i * loading_j` to the covariance of the mechanisms that share it.

### What is checked

Each defect is a finding with a closed kind, a `mechanism_composition.<slot>`
detail, the offending subject, expected and supplied semantics, and the
declaration that would license the composition.

- **Variable identity**: one mechanism per variable; every parent has a mechanism.
- **Causal parents / conditioning set**: a mechanism's parents equal the declared
  graph's parents, and the fit conditioned on exactly its parents.
- **Units / transforms**: a consumer's parent units and transform equal the
  producer's output; no implicit conversion.
- **Population / regime**: a mechanism fitted elsewhere is admitted only through a
  declared selection diagram with no arrow into its variable, or an explicit
  invariance assumption (regime: invariance only). The transport theory itself is
  owned elsewhere; here only the declaration is represented.
- **Support / overlap**: the producer's output support lies inside the domain
  the consumer was fitted on (finite tables: level sets and complete rows);
  interventions must lie inside every consumer's domain.
- **Shared latents**: a latent shared with a mechanism that is absent,
  unreciprocated, undeclared by its peers or without a loading is refused. It is
  never silently treated as independent.
- **Provenance**: owning fit or binding identity and evidence identity; a
  mechanism that was declared rather than estimated needs acknowledgement.
- **Uncertainty alignment**: one meaning (posterior or sampling), matching
  parameter sets, symmetric positive semidefinite covariances.
- **Posterior dependence**: separately fitted mechanisms are *not* independent by
  default. An uncertainty claim over more than one mechanism needs an aligned
  joint law, an explicit cross-mechanism covariance block for every pair (a zero
  block is a declared independence for that pair), or independence by design with
  a justification and distinct evidence identities. Otherwise the refusal is
  `joint_law_required`. Point execution is unaffected.

### Claim labels

Composition is exact given the declared mechanisms. The delta-method variance is
a parameter-covariance variance only: it omits second-order terms, holds residual
scales fixed and carries no coverage claim. The interval route is closed with
`cell_not_licensed` until whole-method coverage is measured.

### Prepare, refresh and export

In Rust, `StudyBuilder::mechanism_composition` takes the semantic declarations and the
numbers together: `check_mechanism_composition` checks the coordinates, the
plan is bound to the numbers (`lower_plan`) and the result is compiled under
the coherence checker, so every defect of a request is returned with its
code, detail and remedy before anything executes. A mechanism built from a
fitted registry slot is labelled `fitted`; coefficients or tables written down
by the analyst are labelled `declared`, need the acknowledgement above and are
listed by `declared_mechanisms()`. The facade sets the label from the source
of the numbers. The prepared composition answers an intervention law, a mean
contrast with its delta-method variance and explicit cross-covariance label,
structural samples, or an exact finite law, and exports one query as
`mechanism_composition_v1`.

`refresh_mechanisms` takes mechanisms refitted on new outcome-only data:

| | |
|---|---|
| keeps | the declared graph, units, transforms, populations, regimes, admissions, latent declarations, fitted domains and supports, the kind of declared dependence |
| recomputes | the refitted numbers, their fit and evidence identities (so the plan identity), the reported marginal covariances, the joint model |
| refuses | a refit of an unknown variable, of another family or with other parents; a joint law or cross-covariance declared for the earlier fit (it is `unknown` again until redeclared); any prepare refusal on the new numbers |

The call takes `&self` and returns a new value; a refusal leaves the prepared
composition unchanged.

### Python surface

`antecedent.mechanism_composition` exposes the same route with typed
declarations: `Quantity`, `MechanismDeclaration`, `Mechanism.declared_linear`,
`Mechanism.declared_table` and `Mechanism.fitted_linear` (which reads the plain
linear-Gaussian slot of a fitted `FittedGcm`; written-down numbers are labelled
declared and need `admit_declared_mechanisms=True`), `Dependence` and
`Admission`, then `compose_mechanisms`. The prepared object answers
`intervention_law`, `mean_contrast`, `mean_contrast_uncertainty`, `sample` and
`finite_law`, refreshes with `refresh_mechanisms`, exports one query with
`export` and is consumed by `consume_mechanism_composition_artifact`. A refusal
is a `MechanismCompositionRefusal` listing every defect (`refusals`,
`details`); `interval()` raises `cell_not_licensed`.

The `composed_model` producer is a row of
[the producer and consumer matrix](producer-consumer-matrix.md): a composed model
is a point mean source for the declared-adapter consumers and is refused by the
rest with typed errors. The design composition-bundle has no node kind for a composed model.

#### Checked Python handoffs

`prepared.as_mean_source(quantity=..., contrast=...)` selects a checked interventional
mean contrast and requires its compatible scientific coordinate. The canonical
`results.MeanSource` carries the point, identity and lineage into decision and inverse
queries. It does not carry the model's law or independent marginal variances as joint
decision uncertainty.

`prepared.export(query, ...)` returns a typed `ExportedArtifact` with the query-selected
bytes, expectation and immutable export metadata. Shared
`ant.load(...)` invokes `consume_mechanism_composition_artifact`, preserving its checks
and typed consumed result.
