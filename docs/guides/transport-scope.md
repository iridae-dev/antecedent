# Transport theorem scope and limits

This page cites registries. It does not copy them.

Stage transport routes are licensed in `transport_stages.toml`; the analyze Cartesian cell remains trial-IPW `TransportQuery` on an explicit `Admg`.

## Theorem families

`TheoremScope` constructors live in
`crates/antecedent-core/src/query/transport_contract.rs`. Completeness is
per family:

| Family | Guarantee | Pin |
| --- | --- | --- |
| Classical single-source sID | complete in the paper experimental-information family | `antecedent.transport.advanced.identify_classical` → [arXiv:1312.7485v1](https://arxiv.org/abs/1312.7485v1) |
| Classical meta-transport | complete in the multi-source experimental family | `antecedent.transport.advanced.identify_meta` → [Bareinboim 2013](https://proceedings.mlr.press/v31/bareinboim13a.pdf) |
| Finite catalog search | sound and incomplete | closed as `transport.finite_catalog_search` |
| Limited / z-experiment | sound and incomplete within 12 observed and 4 controllable variables; positives use cited joints; a line-11 obstruction is structural in the declared controllable set; two sources are searched separately, and complementary factors combine under two registered factorizations (disconnected graph components and, on one connected graph, intervention-separated outcome groups); a single connected c-factor that would need one fabricated joint over both sources' interventions is refused with reason code `transport_not_certified` whose message names `z_transport.multi_source_combination_not_searched` | `antecedent.transport.advanced.identify_z_transport` and `antecedent_identify.decide_two_source_z_transport` → Bareinboim and Pearl, *Causal Transportability with Limited Experiments* (AAAI 2013, [R-408](https://ftp.cs.ucla.edu/pub/stat_ser/r408.pdf)) and *Transportability from Multiple Environments with Limited Experiments* (mz-transportability, NeurIPS 2014); the c-factor step and the completeness result are Lee and Honavar's [arXiv:1309.6842](https://arxiv.org/abs/1309.6842), which this implementation does not claim |
| Finite scenario envelope (X2) | no new theorem: each scenario of an explicitly supplied list of one to 64 fixed selection ADMGs (at most 12 observed variables) is decided independently by the classical catalog route; the envelope is a report over those decisions, not an identification claim | `antecedent.transport.advanced.prepare_transport_scenarios` → see [Finite scenario envelopes](#finite-scenario-envelopes) |
| Two-step temporal sequence (X5) | classical sID on the explicit two-slice unrolled selection diagram, deciding `do(A_1=a_1, A_2=a_2)` as one joint intervention; sound and incomplete; exact and point-only | `antecedent.transport.advanced.prepare_temporal_transport_sequence` → see [Two-step temporal sequences](#two-step-temporal-sequences-22a-x5) |
| ADMG conditional transport (2.2B X2) | sound and incomplete: rule 2 moves every conditioned variable it can into the intervention set (the IDC reduction), classical sID decides the reduced joint over the complete source family, and the conditional is that joint normalized at the remaining conditioned levels; when sID does not certify the reduced joint, an exactly verified two-model witness makes the query `proven_non_transportable` (every three-node case has one), and otherwise it is `not_certified`, never proven; at most 6 observed variables; exact and point-only | `antecedent.transport.advanced.identify_admg_conditional_transport` → see [ADMG conditional transport](#admg-conditional-transport-22b-x2) |

Classical sID completeness applies only to the paper experimental-information family, not to every catalog the API can represent. The static checker in
`scripts/check_transport_stages.py` refuses a completeness guarantee that
drops those pins.

## Support split

- Analyze Cartesian ([`docs/support-matrix.md`](../support-matrix.md), generated
  from `parity/support_licensed.toml`): licensed trial-IPW
  `TransportQuery` × `Admg` × explicit × Frequentist × none.
- Stage routes (`parity/transport_stages.toml`, default closed): identify,
  prepare, evaluate, uncertainty, consume for exact, statistical, grid,
  catalog, and learned-trial paths.

Do not add stage routes as fake analyze cells.

## Finite scenario envelopes

`prepare_transport_scenarios` takes a `TransportScenarioSet`: a finite list you
supply of named fixed graphs over the same variables, each with its own
selection targets and optionally a declared weight, plus one shared coordinate
schema (`VariableCoordinate` names, domains and cardinalities, units). A
scenario may restate its coordinates; any disagreement between scenarios, or a
catalog, law, sample or request outside the schema, refuses with reason code
`schema_mismatch` (detail `scenarios.coordinate_mismatch`).

What the schema check does and does not cover, exactly:

- Names, domains, cardinalities and units must agree across scenarios and cover
  each graph exactly.
- Values (law axes and interventions, sample rows and interventions, the
  request) must be finite numbers inside the declared domain. A non-finite or
  non-numeric value belongs to no domain, so it refuses even where the declared
  domain is `unspecified`; an `unspecified` domain otherwise makes no claim about
  which finite numbers occur.
- Units are compared only between the schema declarations and the catalog's
  environments, and only when both declare one; a catalog domain that is
  `unspecified` cannot disagree. Laws, samples and the request carry no units,
  so a value's unit is never checked.

Graph scope: each scenario is decided by the classical catalog sID row with that
row's scope inherited unchanged. The scenario code adds one gate, at most 12
observed variables; it admits the same selection ADMGs (directed and bidirected
edges over static observed variables, any number of outcomes and treatments) as
that row. The row's limit that multi-outcome and larger general graphs are not
enumerated against a latent-SCM oracle narrows its conformance coverage, not
admission: every scenario result is a checked derivation or an independently
verified s-hedge. A scenario's `not_certified` status is a defensive status:
the scenario path reaches it only when the catalog search finds no derivation
at any stage and the classical identifier finds no verified s-hedge. The
catalog search's first stage is the classical search itself, so a derivation
there means `identified` or `missing_evidence`, and a failed classical search
always leaves an obstruction from which a checkable s-hedge is built
(`structurally_unidentified`). A tight pretreatment-subset budget only adds an
inconclusive obligation to a `missing_evidence` or hedge result; it cannot
produce `not_certified`. No random ADMG in the tests (4 to 11 variables, several
outcomes and treatments, random selections; 2.15 million in a one-off run, about
913 thousand s-hedges) reached it, so the status is exercised in the tests by a
supplied decision and is retained for routes that can return an undecided
scenario, such as the conditional route below.

Conditional questions (2.2B B1, promoted into the envelope). With
`conditioned_on` (Rust: `StudyBuilder::conditional_transport_scenarios`,
`decide_conditional_transport_scenarios`) every scenario asks the conditional
question `P*(y | do(x), w)` and is decided by the
[ADMG conditional transport](#admg-conditional-transport-22b-x2) route, with
that route's scope unchanged: at most 6 observed variables, 0-3 treatments and
1-3 conditioned variables, refused for the whole set with
`route_not_supported` (`admg_transport.bounds_exceeded`) otherwise. Every stage
of each scenario's decision (rule-2 reduction and its independent re-check,
reduced-joint sID decision, catalog binding) is charged to the set's one shared
budget, so entry charges, cumulative memory, receipts and `unevaluated`
scenarios behave exactly as above. An identified scenario compiles through the
route's exact prepared path (the reduced joint at `do(x, w')`, normalized at the
requested `w''`); a zero-mass conditioning event is that scenario's
`support_failure`. Here `not_certified` is reachable: a reduced joint with a
verified s-hedge is reported `not_certified` with the inspection-only
`ConditionalObstructionCandidate` in its detail, because lifting the s-hedge to
the conditional is paper-inherited; the scenario is `structurally_unidentified`
only when the route proves it with an exactly verified two-model witness, which
the artifact carries. The request (`at`) binds exactly the treatments and the
conditioned variables (`admg_transport.invalid_request`). Exact laws only:
counted laws or `StatisticalTransportData` refuse with `cell_not_licensed`
(`admg_transport.interval_withheld`). A conditional set exports artifact format
version 2 (the question's `conditioned_on`, each identified scenario's checked
conditional record, each not-certified scenario's candidate and each proven
scenario's witness); a classical set still exports version 1 byte for byte, and
a version-1-only reader refuses a version-2 artifact by its version.

What it is:

- A per-scenario decision. Each scenario binds its own evidence; nothing
  certified for one scenario satisfies another. Every scenario is reported
  whatever its status: `identified`, `structurally_unidentified` (a verified
  s-hedge for that scenario only), `missing_evidence`, `not_certified`,
  `unsupported_provider`, `support_failure` or `unevaluated`.
- A structural envelope: the minimum and maximum of the target response over
  the scenarios that identified, naming the scenarios that attain them. It is
  not a confidence interval and not a sharp bound, and it says nothing about
  scenarios that did not identify.
- With declared weights, a report that never renormalizes over the identified
  scenarios: every other scenario's weight and the undeclared residual stay
  unaccounted mass, placed anywhere in the outcome's declared domain.
- Points from supplied exact laws or from empirical plug-in tables fitted once
  from `StatisticalTransportData` samples (once for the whole set, not per
  scenario; each scenario then compiles against them, so provider coverage is
  per scenario); never an interval: the bootstrap is fixed at zero replicates.
  Sample rows and interventions are checked against the schema first.
- Bounded by one search budget (`max_steps`, `max_depth`, `memory_bytes`,
  `cancel`) shared by every scenario entered and every step of its search,
  pretreatment separation tests, s-hedge check (witness construction and its
  independent verification, one operation each) and proof replay. Memory is
  cumulative: a decided scenario keeps holding its engine's peak live state (an
  upper bound on what its derivation retains), which every later charge sits on
  top of. A stop leaves the scenario being decided and every later one
  `unevaluated` (detail `scenarios.unevaluated_budget: search.<stop>`) with one
  receipt; none is dropped. The receipt's `explored` lists the scenarios fully
  decided and `unevaluated` the one being decided plus the later ones, so the
  two are disjoint. A 65th scenario refuses with `route_not_supported`
  (`scenarios.count`).
- Weighted totals are order independent: identified, unaccounted and per-status
  masses are summed in sorted order with compensation, and mass within 1e-12 of
  one is fully accounted for (the tolerance weights are accepted under), so
  renaming scenarios never changes a report.

What it is not:

- Not an equivalence-class solver: a `Cpdag` or `Pag` refuses with
  `route_not_supported` (`scenarios.equivalence_class_input`). Enumerating the
  members of a class is the caller's choice and responsibility.
- Not cross-scenario inference: `aggregate_interval()` always refuses with
  `scenario_aggregate_not_licensed` (`scenarios.shared_data_aggregate`).

`consume_transport_scenarios_artifact` replays the whole set under the
producer's recorded budget and requires an identical report. Two digests guard
it. The scientific premises digest binds the scenarios, weights, coordinate
schema (and so the variable names the report is read with), question, request
and budgets (including the memory limit). A separate data-identity digest binds
the evidence catalog (regimes, environments, snapshot bindings), every stored
law, the provider identity and the fitted-sample summaries, so a snapshot or
provenance label the report never shows cannot be edited unnoticed; a mismatch
is a typed `DataIdentityMismatch`. The two are separate so refreshed data
replaces only the second.

A report truncated by an operation, depth or memory bound exports: the limits
are recorded and the consumer replays the identical prefix. A report truncated
by cancellation is never exported (typed `CancelledNotReplayable`), because
nothing recorded lets a consumer reproduce where the interruption fell.

## Decision inspection on the z route

Every bounded single-source decision (`ZTransportStage.decide`) exposes
structured inspection, not only a reason string. `identified` and
`combined_identified` carry the checked proof graph and its per-factor binding
obligations; `missing_evidence` names the unbound cited factor as a typed
object; `proven_non_transportable` carries the reduced line-11 terminal record.
`not_certified` carries a typed reason kind and the recursive rules the search
explored — the search built no expression, so its `proof_graph` is null and the
explored region is reported instead of a fabricated graph. A budget or
cancellation is surfaced as `outcome = "exhausted"` with a limits receipt: the
step and depth limits in force, which budget tripped, and the steps consumed and
depth reached when the search stopped (both absent when the budget tripped
before the search was entered). `ZTransportStage.not_certified_inspection`
exposes the same explored-region record for a stage the bounded identifier did
not certify, and `failure_snapshot` carries the limits receipt on its
`exhausted_computation` status.

## Mechanism sensitivity on the z route

`PreparedZTransportStage.mechanism_sensitivity` and `export_sensitivity`
apply only to the registered surrogate formula (an outcome conditional times
its shared parent marginal from one cited `do(z)` joint). Every other in-bound
graph, including an admissible selection diagram whose recursive derivation
executes, refuses sensitivity with `IncompatibleFormula`; the point result is
unaffected. The sensitivity range is an exact assumption range, not a sampling
interval.

### Joint mechanism deviations (2.2B X3)

`antecedent.transport.advanced.joint_mechanism_sensitivity(stage, JointDeviation(...))`
perturbs the surrogate formula's outcome kernel and shared parent marginal together,
each within its own fraction bound (a box; a coupled total budget refuses with
`joint_sensitivity.budget_coupling`). It reports the exact joint assumption range,
the axis tipping points (equal to the 2.1 one-factor values) and a tipping frontier
of bisection brackets (certified against the floating-point evaluation of the range)
under one shared search budget. Fixed-graph parent
or conditional-mechanism perturbations, the treatment mechanism, more than two
factors and source-target discrepancy diagnostics refuse with typed
`joint_sensitivity.*` details. Declared bounds: at most 2 factors declared (the outcome kernel and the shared parent marginal), 64 parent levels, 32 outcome categories, a frontier grid of 1 to 33
points, at most 100000 search operations, a bisection depth of 64 iterations per
frontier line and a declared memory cap of at most 512 MiB; the uncertainty
record's internal estimator has a bootstrap request cap of 2000. The range is
never a confidence interval, and its guarantee (exact within the declared contamination
class) covers the range only. The one declared sampling composition (conservative endpoint
percentile bootstrap, paper-inherited) is record `2.2B.X3.joint_sensitivity_uncertainty`,
in progress: wired, not measured. Its routes refuse with `cell_not_licensed` until its two
coverage records (a gated zero-box record and a `one_sided` positive-box record) are
measured at the cut. Export and consume use a
separate version 3 artifact. Derivation and scope: [joint-mechanism-sensitivity.md](joint-mechanism-sensitivity.md).

## Learned continuous-outcome transport (2.2A X4)

One cell: a randomized binary source treatment with known probabilities, a
continuous outcome, complete baseline covariates equal to the certified
standardizers (a direct or baseline-standardization certificate, the
`prepare_trial` graph contract), and an overlap-supported target. The estimand is
`E_target[E(Y | X, A=1, S=1) - E(Y | X, A=0, S=1)]`.

- **Target and design.** Nested cohort: the target is the nonparticipants of one
  IID cohort. Independent samples: a separately sampled representative IID target
  with fixed sample sizes. The two are distinct declared designs; sampling is IID
  only (`learned_transport.non_iid_design` refuses anything else).
- **Nuisances.** Outcome regressions per arm and source membership are
  `antecedent.learners` specs, cross-fitted on one shared stratified fold
  assignment. Each nuisance for a fold is fitted on the other folds alone; the fit
  applies no preprocessing beyond an intercept column, so "preprocessing inside the
  folds" holds trivially today (the lifecycle test refits each fold independently and
  would catch a future full-sample transform for learners it changes). The fold
  assignment depends on the source and arm flags only, never on outcomes or
  covariates. Provider implementation and version and the fold assignment are recorded.
- **Claim.** Model double robustness of the point: consistent when either the
  outcome regressions or the participation model is. No efficiency, rate or CATE
  claim; a heterogeneous or simultaneous target refuses
  (`learned_transport.cate_requested`).
- **Overlap.** Source-membership and treatment overlap are separate diagnostics
  with separate thresholds; insufficient overlap refuses
  (`learned_transport.membership_overlap`, `learned_transport.treatment_overlap`)
  rather than extrapolating.
- **Estimator menu.** `advanced.estimator_menu(query, outcome=..., membership=...)`
  inspects, without fitting, which estimators are eligible for this graph, query and
  learners, and for each the required laws, graph conditions, nuisance tasks,
  support, sampling design, uncertainty status and refusal. For the two trial
  estimators these are computed: laws and graph conditions from the certificate (the
  standardizers, rule and premises), nuisance tasks from the learner specs and fold
  count, support from the thresholds in force, sampling design from the declared design
  and uncertainty status from the actual route status (requested replicates, floor,
  `cell_not_licensed`). The prepared study computes its menu from its own options; the
  standalone call uses the release defaults and marks nuisance tasks as default. Each
  entry lists its `static_fields`: descriptions that are fixed text rather than derived
  (all requirement fields of the refused `dr_learner_cate`,
  `exact_finite_law_evaluator` and `smoothed_dose_transport_aipw`, and three of the
  supplied-probability IPW entry). A direct certificate admits no covariates, so its
  membership and randomization laws are listed as marginal.
  Selection stays manual; nothing is recommended.
- **Bounds.** At most 20 cross-fitting folds and a bootstrap cap of 2000 replicates
  (floor 199); a consumer admits at most 1,000,000 rows and 256 features.
- **Interval.** The original joint outer refit percentile bootstrap (replicate
  floor 199) overcovered at the smallest sample size and undercovered for the
  largest nested cohort. A design-specific analytic influence interval passed
  the six-point 95% grid at 2,000 datasets per point. The public route remains
  closed (`cell_not_licensed`) pending promotion: estimates report the interval
  withheld and artifacts carry no interval field. The calibration harness is in
  `crates/antecedent/tests/learned_continuous_calibration.rs`; the candidate
  interval is compiled only under the `calibration-internal`
  feature of `antecedent-estimate` (enabled by the facade's dev-dependencies alone), so
  no ordinary dependent can obtain the interval around the `cell_not_licensed` refusal.
  The sample-size grid `n/2, n, 2n` is swept by `scripts/gate_calibration.sh` for the two
  coverage records only; the weak-overlap and misspecified-nuisance tests print
  measurements at the base point and emit no record.
- **Bootstrap replicates.** Replicates refit every nuisance on their own resample and
  reuse the point run's fold label of each resampled row (duplicates of one row stay in
  one fold, so a row never trains and tests its own copy). Two behaviors are documented
  rather than changed, and their effect on coverage is unmeasured beyond what the
  calibration records themselves measure: replicate fold sizes are not rebalanced (a
  replicate whose fold lacks a role fails and counts as a failed replicate, which
  withholds the interval under the strict replicate policy), and only the point run is
  gated on membership overlap (a replicate whose resampled out-of-fold membership falls
  below the threshold still contributes its value; the shared estimator also serves the
  2.1 learned-trial cells, whose seeded results must not change).
- **Artifact consumer.** An independent consumer replays the certificate, the folds
  and the score; it does **not** replay the nuisance fits. Stored out-of-fold
  predictions and provider versions are producer evidence: they cannot be verified
  without refitting. A single edited prediction, provenance entry or fold fails
  (evidence digest, then score, diagnostics, overlap and provenance-versus-spec checks);
  a forger who rewrites the predictions, the point, diagnostics and overlap consistently
  and re-digests produces an artifact that consumes. The digests are integrity checks,
  not authentication, and provider versions are not compared with the consuming build's.

## Two-step temporal sequences (2.2A X5)

One cell: a fixed two-step intervention sequence `[a1, a2]` over one discrete
action alphabet (at most 8 actions), on an explicit two-slice unrolled selection
diagram of at most 12 coordinates and 4096 complete covariate histories. The
estimand is the target distribution and mean of the outcome under
`do(A_1=a1, A_2=a2)`. The claim is exact and point-only; temporal sampling
intervals, initial-state uncertainty and new-period refresh are 2.3A, and an
interval request refuses (`temporal_transport.interval_requested`,
`estimator_inference_mismatch`).

- **Unrolling.** `TemporalSequenceSpec` places every coordinate in time: baseline
  covariates, the covariates observed before each action, the two actions and the
  outcome. Directed edges may not run backward in time; latent confounding is a
  bidirected edge. The Rust `unroll_two_slice` builds the diagram from a lagged
  `TemporalDag` template through the ADR 0021 unfolding (parents before the first
  slice are dropped, so the initial state is fixed, not modelled). This template
  route is Rust-only (Python supplies the unrolled diagram directly); it is
  exercised end to end (identify, prepare, evaluate, export, consume) against the
  enumerated truth of a known model whose period-1 outcome is a step-2 covariate.
- **One longitudinal intervention.** Both actions are the treatment set of one
  classical question on the unrolled diagram. Time step 1 and time step 2 are never
  transported separately and multiplied; a time-varying confounder (a step-2
  covariate downstream of the first action that affects the second action and the
  outcome) is reported explicitly and handled by the joint identification.
- **Mechanism differences.** A selection target on a coordinate is a time-indexed
  difference between source and target at that coordinate's slice. Every other
  non-action coordinate is assumed invariant at its slice; the report lists each
  assumption per slice and whether the derived formula borrows it from the source.
  An action cannot be a selection target: the sequence sets it.
- **Sequence semantics.** The sequence is hard: one fixed action per step, never a
  policy reading the history. `[a, b]` and `[b, a]` are different interventions with
  different identities and answers.
- **Evidence per step.** The evidence catalog states which regimes exist; the report
  lists, per slice, the regimes measuring or intervening there and marks each
  `cited_by_derivation` only when a bound leaf of the identified proof reads it. What
  a diagram requires is whatever its derivation cites, read from the derivation's own
  leaves and never assumed: for the fixture diagram (selections at the baseline and a
  step-2 covariate) it is the source's outcome law under each complete history over
  the coordinates that reach the outcome (`do` on baseline, covariates and both
  actions); with a selection only at the baseline and an action experiment it is the
  outcome law under `do(a1, a2)` conditioned on the baseline. A catalog regime no leaf
  reads is not evidence the sequence needs.
- **Support.** A history/horizon-local report lists every initial state (step 1) and
  complete history (step 2) with its target mass and status. A complete history is
  outside support unless the supplied laws serve every source leaf the identified
  proof cites at that history and the sequence (population, regime, the leaf's
  intervention coordinates at the history's values, its axes, and positive
  conditioning mass): a source law with no interventions, or one intervening only on
  the actions, serves none of the fixture diagram's complete histories. The exact
  evaluator enumerates the whole lattice, so this holds whether or not the target
  reaches the history (an unreached, served history is reported `unreached`). A first
  action the target never takes at a reached initial state also refuses
  (`temporal_transport.history_outside_support`) instead of extrapolating, but only
  when the proof reads the target's law over that action (as the fixture diagram's
  does); a proof reading only the target's initial-state law needs no such mass, and
  this too is read from the proof's leaves. The report is never more optimistic than
  the evaluator.
- **Budget.** One `SearchBudget` covers the identification search, its verification
  replays and the growth of the history lattice (charged per state with the live bytes
  retained). A stop is a receipt (`temporal_transport.history_budget`), never a
  non-identification verdict, and its `explored` list names every finished stage
  (identification, then history step 1). Operations, memory and cancellation can stop
  inside either history step; history growth is charged at depth 1 and 2, below the
  identification search's own depth, so a depth stop is always an identification stop.
  Caps refuse as `temporal_transport.bounds_exceeded`.
- **Fixed initial state.** The initial-state law (baseline and step-1 covariates) is a
  point: it is not declared separately but read from the target's observational law.
  What is enforced is that a same-window refresh may not move the target's
  initial-state masses (beyond 1e-9); a moved initial state is initial-state
  uncertainty or a new period (2.3A) and refuses as a changed window.
- **Lifecycle.** A same-window evidence refresh re-estimates under the same proof and
  identity. The licensed horizon is 2 steps. A longer horizon, a third period, laws over other coordinates or another
  measurement window refuse (`temporal_transport.horizon`) and need a new
  preparation. The artifact binds the graph, selections, slots, variable names,
  question, ordered sequence, horizon and limits; a consumer re-decides, recompiles
  and recomputes the point bit for bit. The catalog and law snapshots are data
  identity outside that digest, bound by a separate data digest, so a swapped catalog
  or law snapshot is refused even when the replayed point does not change; exporting a
  report that another preparation evaluated is refused.

## Multi-source limited experiments (2.2A X1)

`identify_multi_source_z_transport` (Python) and `decide_mz_transport` (Rust) decide
`TR^mz` (Bareinboim, Lee, Honavar and Pearl, NeurIPS 2013): a target effect answered
from two to four source populations, each with its own selection targets,
controllable set and experiment levels. The result is sound and incomplete within the
bounds (12 observed variables, 4 controllables per source, 64 candidate regimes,
4096 operations, depth 24) and never mixes sources inside one c-factor. The checked
line-11 obstruction rests on R-443 (Bareinboim and Pearl, NeurIPS 2014; Theorems 4-5
and Corollary 1, over the power-set information family of Def. 2) and is a strict
subset of the paper's FAILs: it is certified only at a forced line-11 terminal with no
active experiment where no source can exchange, and is replayed by an independent
checker. Where this search departs from the paper's `TR^mz`, each departure is pinned
by an executed test (`crates/antecedent-identify/tests/mz_transport_search.rs`,
`mz_transport_paper_reference.rs`, and `crates/antecedent-estimate/tests/mz_transport_execution.rs`):

- R-443 Fig. 1(e,f), an experiment on `Z2` in one source and on `Z1` in another, is not
  transportable in the paper; here it is `not_certified` with `mz_transport.fabricated_joint`
  and never an obstruction, because the failure is reached after an exchange.
- The paper's own non-transportable example, an `X`-experiment in one source and a
  `Z`-experiment in the other on R-443 Fig. 2(a,b) (`X -> Y <- Z`, `X <-> Y`, `Z <-> Y`,
  `S_a` into `Z`, `S_b` into `Y`), is a checked obstruction, and the paper's two models
  `M1`, `M2` (its Eqs. 3-4) are executed: they agree on the target observational law and
  every supplied experiment but differ on `P*(y | do(x))`. A three-node analogue
  (`Z -> X -> Y`, `X <-> Y`) is kept as a second obstruction with its own two-model witness.
- A paper FAIL reached after an exchange whose every candidate line-10 branch fails is
  `not_certified` with `mz_transport.search_incomplete`, never an obstruction.
- Fig. 3 line 10 fires only with no active experiment (one exchange per branch); the search
  also lets the active source exchange its remaining controllables. A differential test
  against a literal one-exchange reference of `TR^mz` over R-443 Fig. 1 variants and 6000
  seeded two-source selection ADMGs finds the extension never eligible (after an exchange
  the non-treatment vertices are still one c-component inside `An(Y)` in `D_Xbar`, so no
  controllable of the active source can enter `X`), the search never identifies where the
  reference fails, and the regime cited for an exchange of several controllables is their
  one joint `do(Z_i ∩ X)`.
- Fig. 3 line 11 returns a weighted combination of all certifying sources; the search
  returns the first in canonical source order and does not retain the others, so a missing
  regime of that source is reported even when another source could have supplied it.
  Either source gives the enumerated truth and the same number.
- The line-10 separation is tested in the graph with edges into `X` removed and needs the
  source's controllable set to meet `X` (selection separated only after the mutilation, into
  a non-ancestor of `Y`, not separated, and separated with nothing to exchange are each
  pinned).
- Target experiments are supported by the paper and refused here by design:
  `invalid_argument` with `mz_transport.invalid_catalog`, before any search.

Every stage of one decision (target only, each source's `TR^z`, the combined search
and the obstruction replay) charges one `SearchBudget`: operations and depth are
totals, and the memory bound is never absent. It is the smaller of the context's hard
limit and a 512 MiB default cap, and the live-state estimate accumulates across
stages. An identified decision carries a success-path search receipt (limits and
memory cap in force, operations consumed, depth reached, stages explored and left
unevaluated). The exported artifact stores it with the proof; a consumer replays the
decision under the stored limits (refusing limits above its own maxima) and accepts
only an identical receipt.

Refusals pair a reason code with an `mz_transport.*` detail: `route_not_supported`
(`bounds_exceeded`), `invalid_argument` (`invalid_query`, `invalid_catalog`,
`functional_graph_mismatch`), `transport_not_certified` (`search_incomplete`,
`fabricated_joint`, `invalid_derivation`, `invalid_obstruction`),
`transport_missing_evidence` (`missing_joint_regime`),
`transport_proven_non_transportable` (`checked_obstruction`),
`transport_budget_cancel` (`budget`) and `transport_missing_provider`
(`empirical_counts_required`).

Points are exact or the empirical plug-in of counted laws. The joint-bootstrap
interval route is closed (`cell_not_licensed`, `mz_transport.interval_withheld`) until
its coverage records are measured: no public constructor yields an interval and every
public consumer (io, facade, Python) refuses an artifact that carries one. The internal
bootstrap estimator the calibration harness measures (`mz_transport_bootstrap_interval`,
its draw machinery and interval types) is compiled only under the `calibration-internal`
feature of `antecedent-estimate`, enabled by dev-dependencies alone, so it is absent from
normal and Python builds; the declared-sampling classification that reports why an
interval is withheld (`mz_sampling_dependence`, `mz_interval_withheld_reason`) stays
public. The data
identity digest binds the whole evidence catalog (sampling designs, weights,
dependence declarations, snapshot and dataset ids) and each law's snapshot.

## Mixed-source proof search (2.2A X9)

`identify_mixed_source_transport` (Python) and `decide_mixed_source` (Rust) answer
`P(y | do(x))` in one population from several studies of it, each measuring a
different set of variables or experimenting on different ones. The theorem-scoped
routes run first (target-first sID, then the declared z or mz route, then the
classical meta route when two or more declared sources can each experiment on every
variable); a solved query returns `named_route` (`mixed_search.named_route`) and is never searched.
Otherwise a generic search chains a frozen rule set (`x9.rules.v1`: input,
marginalize, condition, product and do-calculus rules 1-3, each with at most two
moved variables and at most three intervened variables) over the target
population's whole-population, available, unconditioned joint laws. Every
attempted rule application charges one `SearchBudget` shared with the named
stages. Every step records its premises and the study distribution it uses, and
an independent proof checker (its own m-separation criterion, its own set logic,
expressions rebuilt in a fresh arena) replays each one; a deletion never drops a
variable the expression still mentions. The two deletion rules are in the frozen
set and the checker accepts them, but the search does not attempt them: within
this rule set they only undo an insertion, so they never add a quantity (none did
over 1,479 random closures), and attempting them would only spend budget.

The checks are layered, and the docs do not claim more than each does. The shared
expression layer compiles the proof's root and binds its leaf set (syntactic and
structural; it does not replay numbers). The independent oracle checks every
do-calculus side condition, and the expression semantics are verified by
enumeration: every accepted rule instance is compared with an enumerated
structural model (including Pearl's `W(Z)` exception for rule 3), and every
identified query in a random and an exhaustive three-node sweep, with one or two
treatments, equals the enumerated truth.

How incomplete. "Sound, incomplete" is measured, not just declared. Against the
complete Shpitser-Pearl ID algorithm, on a single observational study of the whole
joint with the named routes bypassed (the generic rule search run alone; every
query with non-empty disjoint outcome and treatment sets), the search never
identified a query ID refutes (230,468 queries) and identifies:

| ADMG nodes | queries | ID-identifiable | search identifies |
|---|---|---|---|
| 3, every ADMG | 768 | 612 | 612 (100%) |
| 4, every ADMG | 204,800 | 142,827 | 142,826 (99.9993%) |
| 5, 8,000 random ADMGs x 3 queries | 24,000 | 16,676 | 16,238 (97.4%) |
| 6, 300 random ADMGs x 3 queries (outside the claim) | 900 | 534 | 190 (35.6%) |

Known gap classes, each pinned as a test that fails when it is deliberately fixed:
(1) the napkin family, where ID's formula is a ratio of a marginalised C-factor
(the four-node napkin is the only four-node miss; 108 of 109 four-node ratio-form
queries are found); (2) the intervention bound, at most three treatments per
derived quantity, so every four-treatment query is `not_certified` even when its
effect is trivially `P(y)`; (3) the operation budget, which at six nodes ends about
two thirds of the identifiable queries as `exhausted`. Up to five nodes and three
treatments the only misses are napkin embeddings (1 of 16,239). A `not_certified` or
`exhausted` answer on a single fully observed study is therefore never evidence of
non-identification; the named sID route runs first and decides those exactly. There is
no complete oracle for the several-study case in this repository (generalized
identification of Lee, Correa and Bareinboim is not implemented), so the
multi-study claim remains soundness against enumerated structural models, with no
completeness figure.

Bounds: at most 10 observed variables, 16 usable distributions and 4 sources, under a
search budget of 20000 operations at depth 16.

Outcomes: `identified` (with alternative derivations only when actually found),
`named_route`, `missing_evidence` (`mixed_search.missing_joint`: the exact joint a
study holds only as separate marginals), `not_certified` (the rule set reached its
fixpoint: not a non-identification claim) and `exhausted` (limits receipt). Regimes
of other populations, selected samples, model artifacts, conditioned projections,
unavailable evidence and value-restricted (per-level) experiments are excluded and
listed with their reason: a trial of `do(X = true)` only supplies one level, and the
search states identification for every level; an experimental model artifact refuses
(`mixed_search.posterior_as_law`). The route is point-only: exact laws are
evaluated, counted laws are `cell_not_licensed`, and the artifact
(`checked_mixed_source_point_v1`) is re-decided, re-checked and recomputed by an
independent consumer.

The catalog descriptor (`CatalogDistribution`) is the single owner of how the
search classifies inputs: origin, selection, kind, measured set, interventions and
levels, joint versus marginals, identity, study and snapshot are read from it. The
pairwise shared-data relation is not consulted here (exact laws claim a point); only
the mz statistical path uses it.

External parity (Ananke GID/AID 0.5.0, dosearch 1.0.12) is recorded from executed
oracle runs (verbatim outputs, pinned versions, harness hashes; the temporary
environments and harness scripts were deleted afterwards, so the hashes are
provenance only and the outputs are not reproducible from the repository). The
parity test parses each verbatim output: the verdict must equal the recorded one
and the distributions the oracle's formula cites must equal the ones our proof
cites. Where Antecedent returns a theorem-scoped named route instead of a proof
(both single-distribution dosearch cases) the agreement is `agree_named_route`, a
weaker class than agreement of proofs; see
`conformance/identify/mixed_source_external/expected.json`.

## ADMG conditional transport (2.2B X2)

`identify_admg_conditional_transport` (Python) and
`decide_admg_conditional_transport` (Rust) answer the target population's
`P*(y | do(x), w)` over a selection ADMG (directed and bidirected edges, at most 6
observed variables, 0-3 treatments, 1-3 conditioned variables) from one source
that can run every experiment plus the target's observational law. The query
wraps the classical query (`ConditionalTransportQuery { base, conditioned_on }`);
it is not a second transport engine.

- **Reduction.** Rule 2 of the do-calculus, applied in the target population,
  moves a conditioned `w` into the intervention set when `Y` is m-separated from
  `w` given `X` and the other conditioned variables in the graph with edges into
  `X` and out of `w` removed (IDC line 1). The moved set equals the one the repo's
  `IdcIdentifier` moves on every three-node ADMG under every selection pattern and
  on a stride of four-node ADMGs with a single selection; it does not depend on
  the selection targets at all (selection nodes are parentless and conditioned on
  in the target, so they neither open nor keep open a path), which a paired sweep
  with and without selection pins. The classical sID engine then
  decides the reduced joint `P*(y, w'' | do(x, w'))`, and the answer is that joint
  normalized over `y` at the requested `w''`. A zero-mass conditioning event is
  `transport_support_failure` (`admg_transport.support_failure`).
- **Guarantee.** Sound and incomplete. Every decision re-checks its own reduction
  before it returns: each move's rule-2 premise and the maximality of the moved
  set are replayed by an independent augmented-graph separation test (distinct
  code from the search's path test), charged to the same budget; the reduced joint
  is built only through the sID engine's verified-derivation path. A stored
  derivation re-runs both checks when it is prepared (`StudyBuilder`, Python
  `prepare_exact`) and when an artifact is consumed. The conditional values are
  compared with enumerated latent SCMs on every three-node ADMG and a seeded
  four-node sample.
- **Obstruction.** When sID does not certify the reduced joint (an s-hedge, or
  no derivation), a bounded search looks for a *two-model witness*: two finite
  latent models compatible with the selection diagram (binary observed
  variables, one discrete latent per bidirected edge; source and target share
  every latent law and every mechanism except the selection targets'; every
  parameter an exact rational strictly inside (0, 1), so every law is positive)
  that agree on every source experimental law `P(v \ z | do(z))` and on the
  target observational law `P*(v)`, yet give different `P*(y | do(x), w)` at a
  recorded level. `verify_conditional_witness` checks all of this by exact
  enumeration and trusts no theorem; such a pair refutes every formula over those
  laws, which is non-transportability by definition (Lee, Correa and Bareinboim,
  *General Transportability*, AAAI 2020, Definition 3 and Lemma 2). A verified
  pair is `proven_non_transportable` (`transport_proven_non_transportable`,
  `admg_transport.proven_non_transportable`) with a
  `ConditionalNonTransportabilityProof` carrying the witness, the moves and the
  reduced joint's s-hedge (`"proof": true` in Python, with a `witness`). The
  search perturbs one mechanism block of a seeded positive base model inside the
  null space of that block's linear law map (found modulo a prime, lifted by
  rational reconstruction); only the exact verifier certifies. It is incomplete:
  with no verified pair, or above its work bound, the decision stays
  `not_certified` (`admg_transport.not_certified`, stage `conditional_witness`:
  `no_witness` or `out_of_scope`) with the inspection-only
  `ConditionalObstructionCandidate` (`"proof": false`). On every three-node
  selection ADMG and query, all 984 reduced-joint s-hedges carry a verified
  witness, each re-checked by an independent verifier in the tests. The paper's
  Theorem 1 (under conditional minimality the conditional is transportable iff
  the joint is) is the completeness statement for this class; its proof was read
  but is not used, so the row stays sound and incomplete.
- **Budget.** One `SearchBudget` (at most 4096 operations, depth 24, a memory cap
  that is never absent) is charged by every rule-2 separation test of the search
  and of its re-check, every sID step, the catalog binding and each witness-search
  block and null-space direction tried, charged with the attempt's own live-state estimate (a stop inside the
  witness search is an `exhausted` receipt naming that stage, never `not_certified`). Live-state memory
  is cumulative: each finished stage's peak is retained by every later charge. A
  stop is `exhausted` with a receipt of explored and unevaluated stages, never a
  verdict.
- **Artifact.** `checked_admg_conditional_point_v1`: the consumer refuses stored
  limits above its own, and a graph or query above the route's size bounds,
  before any work (before either digest is hashed), checks a premises digest and
  a separate data-identity digest, re-checks the proof and re-decides the query
  under the producer's stored limits, re-binds the leaves and recomputes every
  point bit for bit. The consumer is independent of the artifact, not of the
  implementation: it re-runs the producer's search and evaluator, and only the
  proof check is distinct code, so a bug shared by producer and consumer replays
  identically. It does not authenticate the laws (the data digest names
  snapshots), and it cannot detect a wrong causal graph or a consistently
  re-sealed forgery (the digests are integrity checks, not signatures).
  A proven obstruction exports as `checked_admg_conditional_obstruction_v1`
  (Python `export_obstruction()`, `consume_admg_conditional_obstruction_artifact`;
  Rust `export_admg_conditional_obstruction`): the graph, selections, query, the
  proof record and the names under a premises digest, with no catalog or law. The
  consumer refuses an oversized graph, query or witness before hashing, then
  re-verifies the witness exactly and re-checks the moves and the s-hedge; any
  edit is refused (`premises_mismatch` unsealed, `invalid_derivation` re-sealed).
- **Scenario envelope.** The row also runs inside the finite scenario envelope:
  see [Finite scenario envelopes](#finite-scenario-envelopes) (conditional
  questions).
- **Not licensed.** Counted laws, an empirical plug-in and any interval
  (`cell_not_licensed`, `admg_transport.interval_withheld`); selection-bias
  (`S = 1` sampling) semantics; soft interventions; gID / g-transportability with
  surrogate or heterogeneous experiments (deferred to 2.3A).
- **Witness extension note.** `ConditionalObstructionCandidate` is additive: it
  pairs the unchanged `SHedgeCertificate` of the reduced joint with the rule-2 moves
  and the non-movable remainder (`ConditionalObstructionRecord`). The hedge and
  s-hedge certificate shapes are unchanged. The proof is the separate two-model
  witness (`ConditionalWitnessRecord`), not an upgrade of the candidate.

## Smoothed dose-response transport (2.2B X4)

One cell: a randomized continuous source dose with a **known** conditional density
`pi(a | x)` on a declared support, transported to an overlap-supported target, on a grid
of at most 16 doses at one declared bandwidth `h` with the Epanechnikov kernel. The
estimand is `psi_h(a) = E_target[ integral K_h(a - t) E(Y | X, A=t, S=1) dt ]`; the
bandwidth and kernel are part of it, never tuned. The derivation of the score, its model
double robustness and the executed numerical checks are in
[smoothed-dose-response-transport.md](../smoothed-dose-response-transport.md); record
`2.2B.X4.smoothed_dose_response_transport`, refusal namespace `dose_response`.

- **Surface.** Rust `StudyBuilder::smoothed_dose_transport` / `PreparedSmoothedDose`
  (estimate, refresh, interval, estimator menu) and `consume_smoothed_dose_artifact`;
  Python `advanced.prepare_smoothed_dose`, `advanced.consume_smoothed_dose`,
  `advanced.smoothed_dose_estimator_menu`.
- **Support.** Every window `[a - h, a + h]` must lie in the dose support (no boundary
  kernels; `dose_response.grid_outside_dose_support`); known densities must be valid and
  above a floor; the local kernel-weight effective sample size and the number of
  distinct doses per window are gated; membership overlap as in the learned-trial cell.
  An estimated (generalized-propensity) density, point-curve, stochastic, derivative,
  conditional or simultaneous targets are refused.
- **Nuisances.** `mu(t, x)` (Regression over a row-wise dose-by-covariate basis) and
  membership (BinaryProbability) through `LearnerSpec`, cross-fitted on one shared fold
  assignment; every fold model is kept as a portable predictor, so a learner without one
  is refused (`dose_response.learner_not_portable`).
- **Errors kept apart.** Every window is split at the basis knots inside it, so a
  linear-family fit is integrated exactly; for any other fitted curve (a tree learner)
  each grid dose records the estimated quadrature error (the largest target-row
  `|nu_Q - nu_2Q|`, an estimate rather than a bound, refused above the declared
  tolerance, `dose_response.quadrature_tolerance`). The smoothing-bias diagnostic
  `psi_h - psi_{h/2}` (plug-in, from the fitted curve only, so zero for a fit linear in
  the dose) is reported with its local-quadratic extrapolation; neither is added to the
  estimate. The analytic influence-function standard error is a diagnostic only.
- **Bounds.** At most 16 grid doses, 20 cross-fitting folds, 200,000 rows, 256 covariates,
  basis degree 1 to 3, at most 8 knots, and a bootstrap request of 199 to 2000 replicates
  (or none); a request above a cap refuses as `dose_response.bounds_exceeded`, while 1 to
  198 replicates is below the floor: the point is kept and the interval withheld
  (`dose_response.bootstrap_below_floor`). A mandatory 512 MiB cap on the estimated workspace (lowered by a context hard memory limit) refuses before any fit or
  replay as a resource refusal.
- **Interval.** The pointwise joint outer refit percentile bootstrap of the whole
  composed estimator (`smoothed_dose_interval_internal`, `calibration-internal` only)
  is wired in `crates/antecedent/tests/smoothed_dose_calibration.rs`; its route is closed
  (`cell_not_licensed`) until its two coverage records are measured.
- **Artifact.** `checked_smoothed_dose_transport_v1`: a consumer re-derives the
  certificate, re-predicts every nuisance from the stored fold models and re-integrates
  every quadrature bit for bit, without fitting, under its own row, covariate, memory and
  cancellation limits; it refuses stored bounds looser than its own (tighter ones are
  accepted). It cannot establish that the stored models were fitted as recorded, and a
  re-sealed seed, sampling design or support threshold the rows still pass consumes,
  since none of them enters the replayed point.

## Exact binary observation recovery (2.2B X10)

Graph-licensed recovery, not MAR/IPCW. A separate recovery stage recovers the full
law `P(X(1), O)` of item-missing binary variables from one named observed pattern
law `P(R, X*, O)` (`identify_observation_recovery`, `StudyBuilder::observation_recovery`),
then feeds it to the ordinary target ID stage of sID for a causal effect.

- **Class.** A binary DAG m-graph: at most 3 partially observed variables `X`, each
  with a response indicator `R` (parents among `X` and `O` only) and a deterministic
  proxy `X*` with levels `0`, `1`, `?`; at most 2 fully observed `O`; at most 864
  observed cells. Selection nodes, bidirected edges, non-binary variables,
  `R -> R` edges and noisy proxies refuse as `recovery.unsupported_mechanism`.
- **Decision.** Recoverable exactly when no `X_i -> R_i` edge exists. The formula
  `P(R = 1, X* = x, O = o) / prod_i p(R_i = 1 | pa(R_i))`, each propensity a ratio
  of observed pattern margins, is derived in the module docs, lowered to the
  expression IR with every leaf bound to a margin of the named catalog distribution,
  and checked by an independent checker. A self-censoring edge is refused as not
  recoverable for every model Markov to the m-graph, shown by an exact-integer
  witness (`recovery.nonrecoverable_witness`, reason code
  `transport_proven_non_transportable`). The witness models are degenerate (every
  other mechanism an independent fair coin), so nothing is claimed under faithful
  or generic parameters: there, some self-censoring graphs are generically
  identified (for example a shadow-variable graph `Z -> X -> R_X` with `Z` not a
  parent of `R_X`), an assumption this route does not make. In this class both
  directions are proved in the repo over all models Markov to the m-graph; the
  general Nabi, Bhattacharya and Shpitser (2020) criterion (with colluders) is
  paper-inherited.
- **Evidence.** One joint law over exactly `R`, `X*`, `O`; separate marginals refuse as
  `recovery.missing_margin`. No complete-case fallback and no MAR or IPCW
  substitution: the missingness assumption lives in the graph and is checked.
- **Recovered law.** A derived law. Its catalog descriptor carries
  `LawOrigin::Recovered` provenance, so it never supplies a population law
  (`supplies_population_law` requires a measured origin) and satisfies no
  catalog-route factor; the X9 mixed-source search excludes it as `recovered_law`.
  The catalog wire carries it in an additive `recovered_derivation` field (absent
  for every other origin, so 2.1 catalogs are byte-identical; a 2.1 reader refuses
  it) and round-trips its identity exactly. The exact table's snapshot identity is
  `recovered:<derivation identity>`, never the observed snapshot, so it is refused
  if offered back as the observed law; the table itself keeps the observed regime
  id it was derived from and the `supplied_exact` provider tag. The effect handoff
  requires the same population, the variables `X ∪ O`, the m-graph's causal
  restriction as the effect graph (`recovery.handoff_mismatch` otherwise) and
  strictly positive support.
- **Budget.** One `SearchBudget` (at most 50000 operations, depth 32) bounds every stage;
  a stop is `recovery.budget` with a receipt, never a verdict. The receipt's
  `explored` lists the stages completed and `unevaluated` the stage the stop reached
  (`class_check`, `formula`, `witness` or `downstream_effect`) and those left. The
  standalone witness verifier is uncharged; the role bounds cap it at 8 non-proxy
  nodes.
- **Artifact.** `checked_observation_recovery_point_v1`: a consumer re-decides under
  the stored limits, re-checks the formula and recomputes the recovered law and every
  effect point bit for bit. The data digest binds the whole catalog, the snapshot
  bindings and every observed cell. It cannot establish that the observed table describes real
  data, nor the m-graph's untestable premises. Exact laws only: point-only, and counted
  laws refuse as `cell_not_licensed`.

## Study planning over the X1 and X9 catalogs (2.2B X6)

Structural plus declared-cost planning, not a success probability. Given a query the
multi-source (`mz`) or mixed-source (`mixed`) route cannot identify from its catalog,
and a declared universe of candidate studies, `plan_studies`
(`antecedent::design::plan_studies`, Python `plan_studies`) finds the cheapest subset
that would make the same route identify the query if the studies delivered exactly
the declared regimes.

- **Candidates.** A study is a population (the target or a declared source; new
  populations are not planned), an intervention set with its feasible level
  combinations (or every level), the jointly measured margin, a recruitment
  declaration, a positive integer cost, a sample budget (a tie-breaker only) and
  `requires`/`conflicts` constraints. Each compiles to a typed hypothetical
  `EvidenceCatalogDelta`. Source experiments stay inside the source's controllable
  set; the target never experiments on the mz route; value-restricted studies refuse
  on the mixed route; everything else refuses as `study_plan.invalid_candidate`.
- **Search.** At most 16 candidates, 8 regimes each; every subset of at most 3 is
  evaluated in cost order (cost units, sample budget, size, sorted ids;
  `x6.ranking.v1`) by re-running the route's decision on the preview catalog. One
  `SearchBudget` (at most 200000 operations, depth 24, 16 on the mixed route, and
  512 MiB; at most 32 proposals retained) bounds the base decision and every subset decision; a stop keeps the
  proposals verified so far with a receipt of the unevaluated subsets
  (`study_plan.budget`), never a verdict. A decision above the route's own cap (4096
  operations on mz, 20000 on mixed) is inconclusive for that subset.
- **Sufficiency.** Only a re-run decision that identifies, whose derivation re-checks
  against the preview catalog (an X9 derivation through the independent checker, an
  X9 named route re-run and bound), and whose formula cites a proposed regime. The
  proposal names each repaired factor or X9 proof step and the margin the proof reads
  from each proposed regime; a smaller margin is verified only as its own declared
  candidate (no margin power-set search).
- **Claims.** No sufficient subset is `study_plan.none_certified` within the universe,
  never an impossibility claim. The only theorem-limited refusal is a base X1 decision
  that is a replayed checked obstruction over the declared controllable sets
  (`study_plan.theorem_limited`); a new population or a wider controllable set is
  outside this universe. The top proposal is marked minimal only when every strictly
  cheaper subset was decided without a stop or an over-cap decision: cost-minimal among
  the verified-derivable subsets of at most three candidates of this declared universe,
  under this search and rule set. A cheaper subset of four or more candidates is never
  examined, and a cheaper subset the route refuses (for example over the X9
  usable-distribution bound) keeps the claim: it is only not sufficient under this
  search. Supersets of a sufficient subset are listed `dominated`, for ranking only.
- **Arrival.** `receive` requires the frozen base unchanged and every proposed regime
  present as available evidence of exactly the proposed shape, bound to the provider's
  snapshot (`study_plan.arrival_mismatch`; a `hypothetical:` snapshot is a planning
  preview and is refused), then decides the real catalog through the public route; a
  stop or cancellation there is `study_plan.budget`, never "not identified".
  Positivity is a premise: a law without mass at a level the formula reads passes
  `receive` and is refused by the ordinary evaluator.
- **Artifact.** `study_plan_v1` carries the premises, the base catalog lineage and the
  whole plan under premises, data and plan digests. A consumer refuses stored limits
  (operations, depth, memory cap) above its own or its context's hard memory limit
  before any work and accepts only a plan its full replay reproduces
  (`study_plan.invalid_artifact`); a cancelled replay is `study_plan.budget`, and a
  cancelled plan cannot be exported. It cannot tell whether a producer stated another
  base catalog or candidate universe and re-sealed honestly, nor whether a study will
  deliver its declared regimes. The base catalog's bindings and snapshots (lineage) are
  stored and digested, and a plan refuses a base whose available regime has no
  provider binding outside the `hypothetical:` namespace (`study_plan.invalid_query`),
  so a re-sealed artifact with its base bindings cleared or replaced by placeholders
  no longer replays. Replay cannot tell a binding renamed to another real snapshot;
  `receive` requires the arriving catalog to keep every stored base binding exactly.

## Path-specific edge intervention (2.2A X8)

One cell on the two-world cross-world contrast: how much of the treatment effect flows
along a chosen set of edges of an explicit Markovian DAG while every other edge sees the
baseline value. The graph has at most 8 variables, a query declares at most 8 worlds, and
a consumer admits at most 100,000 rows; the claim is a point, never an interval. Out of
contract shapes refuse with `cross_world.*` details (for example
`cross_world.graph_outside_contract`). Latent confounding is handled by the 2.2B X8
cell below.

## ADMG counterfactual identification (2.2B X8)

One cell: the effect of treatment on the treated, `P(Y_x = y | X = x')` with `x != x'`,
on a supplied explicit ADMG (or DAG) of at most 6 finite-discrete variables with at
most 4 levels each, from the observational joint (exact law or empirical counts). ID*
on the conjunction `{Y_x = y, X = x'}` identifies each district term by ID under one
`SearchBudget` (at most 100000 operations, depth 64); Python
`antecedent.counterfactual_id.prepare_effect_on_treated`, record
`2.2B.X8.admg_counterfactual_id`. When ID* stops and the treatment is binary, the
consistency identity `P(Y_x = y, X = x') = P_x(y) - P(y, X = x)` answers wherever ID
identifies `P_x(y)`. A query ID* does not identify (conflicting
subscripts, or an ID hedge on a district term) and the binary complement does not
answer refuses with `route_not_supported` and a checkable obstruction: it is not
identified by ID*, which is NOT a proof of non-identifiability (ID* composed with ID
is not complete from `P(V)`; a graph whose ETT is identified through `P(y | do(x))`
was refused by ID* alone). The claim is a point and never carries an interval;
path-specific queries on ADMGs are deferred to 2.3. The consumer refuses stored limits
above its own before any work; replay does not protect against a producer sealing a
wrong graph, query or law, nor against a bug shared by the engine and the consumer
(the consumer re-derives with the same engine). The recorded y0 0.2.11 differential
agrees on the verdict on 237 of 257 graphs, and on the 171 both identify y0's recorded
expression evaluates to the same numerator.

## Estimation assumptions

`EmpiricalTable` is the conservative default. `LearnedCategorical` and
`TrialAipw` change the assumption set and must be passed explicitly. Exact
laws make no sampling-coverage claim.

## Calibration scope

Bound records only:

- four `ClassicalTransport` percentile-bootstrap rows
- two `TransportQuery` / `transport.trial_ipw` analytic-SE rows

Learned-trial intervals stay uncalibrated (`calibration_reason`).
`transport.multi_source_calibrated_coverage` and
`transport.simultaneous_grid_bands` stay closed. T7/T8 full coverage
remeasurement is not claimed here.

## Migration and failures

- Day-1 verbs: [`migrations/2.0-transport-day1.md`](../migrations/2.0-transport-day1.md)
- Outcome map: [`transport-failure.md`](transport-failure.md)
- Architecture: [`architecture/transport-exact.md`](../architecture/transport-exact.md),
  [`architecture/transport-meta-grid.md`](../architecture/transport-meta-grid.md)
