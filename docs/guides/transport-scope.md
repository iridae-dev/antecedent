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
scenario, such as the later 2.2B ADMG rows. The bounded
ADMG transport class of a later row (2.2B B1) is separate and not part of this
row.

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
  (all requirement fields of the refused `dr_learner_cate` and
  `exact_finite_law_evaluator`, and three of the supplied-probability IPW entry).
  Selection stays manual; nothing is recommended.
- **Interval.** The single method is the joint outer refit percentile bootstrap of the
  whole cross-fitted estimator, grouped per design, replicate floor 199. Its route is
  closed (`cell_not_licensed`) until its two coverage records are measured: estimates
  report the interval withheld and artifacts carry no interval field. The calibration
  harness is wired in `crates/antecedent/tests/learned_continuous_calibration.rs`; the
  interval estimator it measures is compiled only under the `calibration-internal`
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
  (`temporal_transport.history_outside_support`) instead of extrapolating. The report
  is never more optimistic than the evaluator.
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
  identity. A longer horizon, a third period, laws over other coordinates or another
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
- The paper's own non-transportable example type, an `X`-experiment in one source and a
  `Z`-experiment in the other (R-443 Fig. 2), is a checked obstruction; the exact paper
  graph was not available offline, so the test uses the closest analogue (`Z -> X -> Y`,
  `X <-> Y`) and exhibits two SCMs that agree on the target observational law and every
  supplied experiment but differ on `P*(y | do(x))`.
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
