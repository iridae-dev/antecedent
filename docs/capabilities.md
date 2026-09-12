# Capabilities

This page is a readable tour of what exists in Antecedent. The parity manifests
are the maintained implementation inventory; the [support matrix](support-matrix.md)
is the public **license** for analysis cells. Presence here does not mean every
query × graph class × structure × inference × validation combination runs.
For selection guidance and product boundaries, see [Comparison](comparison.md).

## How to read capability claims

The matrix has three active runtime states:

* **licensed** — the staged path runs under the row's recorded evidence
  contract;
* **n/a** — the coordinate does not denote and is a typed impossibility;
* **refused** — the coordinate is meaningful, but this release does not
  license it.

The historical `allowed_unlicensed` wire value remains decodable for
compatibility, but this release has no active allowlist entries and the release
gate rejects new ones. Evidence kinds are scoped: a known-truth fixture may pin
only identification or an effect point, while an internal cross-check may establish
prepared-vs-fresh consistency without pinning the scientific target. Read each
row's `limitations`; a shared method name is not a parity claim.

A licensed row means the staged runtime path, refusal boundary, and recorded evidence
contract are exercised for that coordinate. It does not mean causal assumptions were
verified from the data, intervals are universally calibrated, identification is complete
beyond the named subset, or parametric restrictions disappeared. In particular, priors
cannot convert a nonidentified estimand into an identified one.

At analysis level, the support matrix is the license. The 1.6 matrix keeps
the 1.5 licensed cells and adds temporal policy cells: per-horizon
`TemporalMediationEffect`, multi-step and joint `Sequence` overlays,
observation-adjusted temporal curves (Frequentist IPCW pairs and the
parametric Bayesian observed-data CAR route), DBN-posterior mixtures on
the contrasts the handle already runs, and bounded prior transfer on
named Pulse / Sustained / ResponseCurve cells. The 1.5 additions remain:
retargetable
prepared AIPW scores (AllObserved iid AIPW and cell-AIPW only; `analyze()`
does not always return scores), exceedance functionals, cell-saturated joint AIPW,
`TieredBackground` as a fast path over ADMG / PAG adjustment, and joint
influence-function standard errors on static Cpdag / Pag effect and response
aggregates. Unknown tiers retain distinct canonical scenario effects.
Completions stay envelope atoms; the runtime class is not collapsed.
Static Cpdag / Pag Frequentist aggregates publish joint-IF standard errors.
Frequentist DBN Pulse/Sustained mixtures publish shared outer-block
bootstrap uncertainty; TemporalCpdag/TemporalPag class envelopes still
disclose that envelope SE omits between-atom variance pending 1.9
calibration. Temporal PAG results retain MAG completions
and disclose finite-window audit caps. Those families are not licensed on every
coordinate: Bayesian
incomplete-class temporal cells, Bayesian
and partial-graph derivatives, accepted / Bayesian / nested counterfactuals,
and cheap/full counterfactual validation remain refused. Importability is not
a license. The [support matrix](support-matrix.md) is the public license.

## Graph primitives

Implemented graph representations:

* DAG;
* ADMG;
* CPDAG;
* PAG;
* temporal DAG;
* temporal CPDAG;
* temporal PAG.

Graph operations:

* d-separation;
* m-separation;
* districts;
* latent projection;
* Markov-equivalence completions;
* definite-status separation;
* temporal unfolding;
* intervention overlays.

Static and temporal graphs have separate semantics. A static graph is not
interpreted as temporal by default. `AverageEffect` on a supplied `Cpdag` is
licensed (explicit/accepted, Frequentist and Bayesian) via a MEC envelope;
runtime class stays `Cpdag`. Completing the graph yourself is still the `Dag`
cell. Pulse and single-step Sustained on incomplete `TemporalCpdag` /
`TemporalPag` are licensed (explicit/accepted, Frequentist) via a completion
envelope. A fully oriented supplied class stays that class. Completing those
graphs yourself is still the `TemporalDag` coordinate. Bayesian
incomplete-class temporal cells stay refused.

Graph interchange is available through NetworkX, DOT, JSON, GML, and versioned
CBOR artifacts.

## Discovery

### Static

* PC
* FCI
* RFCI
* GES
* DirectLiNGAM
* NOTEARS

### Temporal and multi-context

* PCMCI
* PCMCI+
* LPCMCI
* J-PCMCI+
* regime-specific RPCMCI workflows

### Bayesian structure learning

* exact DAG posterior;
* order MCMC;
* structure MCMC;
* CI-screened graph posterior;
* DBN posterior.

Selected posterior graph samples can be propagated into licensed Bayesian
or Frequentist effect envelopes. Static graph-posterior analysis is limited to
`AverageEffect` and `ResponseCurve` / one-coordinate `InterventionResponse`
with DAG atoms. Temporal graph-posterior analysis is limited to pulse and
single- or multi-step sustained effects with `TemporalDag` atoms under
Bayesian or Frequentist inference. DBN-posterior response surfaces,
TemporalCpdag/Pag posterior mixing, and ADMG/CPDAG/PAG posterior ATE atoms
are refused.

### Conditional independence tests

* partial correlation;
* weighted and robust partial correlation;
* regression CI;
* k-nearest-neighbour CI;
* mixed k-nearest-neighbour CI;
* symbolic conditional mutual information;
* GPDC;
* G²;
* oracle tests;
* Bayesian CI tests.

Multiplicity corrections include BH, BY, Bonferroni, and Holm.

Discovery stability tools include block bootstrap, lag and threshold
sensitivity, orientation stability, environment holdout, synthetic-null
checks, and permutation or phase-randomized surrogates.

## Identification

Antecedent reports whether a query is:

* nonparametrically identified;
* partially identified;
* graph-dependent;
* not identified.

Implemented identification strategies:

* backdoor adjustment;
* efficient backdoor adjustment;
* front-door identification;
* instrumental variables;
* sharp regression discontinuity;
* an explicitly scoped, incomplete ID/IDC implementation for DAGs and ADMGs;
* line-5 hedge node-set diagnostics (not fully validated C-forest certificates);
* bounded path-specific identification by selected-edge graph reduction;
* generalized adjustment for partial graphs;
* unfolded temporal backdoor;
* temporal mediation;
* pairwise backdoor identification for continuous-response functionals;
* sharp binary-IV Balke–Pearl ATE bounds by response-type enumeration;
* single-source selection-diagram transport on a sound sID subset (direct,
  S-admissible / exogenous standardization, singleton c-components), with
  `NotCertified` outside that subset.

`AutoIdentifier` reports applicable strategies. It does not silently choose an
estimator.

For PAGs, Antecedent uses generalized adjustment, identification envelopes, or
explicit graph completions. Licensed PAG analysis is `AverageEffect`,
`ResponseCurve` / `InterventionResponse`, and `ConditionalEffect` after a MAG
visibility check; this is a sufficient adjustment criterion, not a licensed
complete identification theory for MAG or PAG response functionals, and not a
path-specific, distribution, or mediation surface. Full PAG-native ID and IDC
are outside the supported scope.
General multi-node sID recursion and definitive non-transportability
certificates are outside the 0.9 transport contract.

## Estimation

### Frequentist

* linear and generalized-linear outcome regression;
* g-computation;
* inverse probability weighting;
* propensity matching;
* covariate-distance matching;
* stratification;
* AIPW;
* front-door two-stage estimation;
* Wald estimation;
* 2SLS;
* sharp local-linear regression discontinuity;
* linear conditional effect models;
* temporal adjustment;
* temporal mediation;
* functional plug-in estimation;
* continuous causal-response curves (Kennedy-style cross-fitted doubly robust
  local polynomial);
* observed-law Riesz average derivatives;
* additive-GAM plug-in Jacobians and directional derivatives (at most two
  treatment dimensions);
* additive-GAM g-computation for numeric hard, shift, and stochastic
  intervention responses;
* selected-outcome IPW and cross-fitted AIPW, plus marginal right/left Kaplan–Meier
  IPCW and conditional right/left Cox IPCW, composed into point-only response
  curves under explicit observation assumptions. The same selected / KM /
  Cox pairs ride Frequentist `TemporalDag` curves at validation `none`;
  unlicensed non-Complete pairs refuse at compile.

Response results keep structural identification, empirical support, and
uncertainty kind as separate axes. Pointwise and simultaneous bands are not aliases. Frequentist temporal observation curves use nuisance-refitting outer block-bootstrap bands. Bayesian temporal observations use the Gaussian observed-data SEM with latent-trajectory Gibbs sampling under declared ignorable trajectory coarsening and distinct priors. Neither substitutes complete-data intervals. Interval censoring
and truncation remain Gaussian-likelihood stages, not a causal-response MLE.
One-shot `discovery=` on response queries fails closed; discover and accept
the structure before estimating a response.
The list above is inventory. Derivative cells are licensed on explicit or
accepted Frequentist DAGs at validation `none`; Bayesian and partial-graph
derivatives remain refused. `ResponseCurve` and `InterventionResponse` are
licensed on `Dag` and `TemporalDag` under Frequentist and Bayesian inference
with validation `none`, and on `Cpdag` / `Pag` under Frequentist and Bayesian
inference with validation `none` via the same generalized-adjustment envelope
as ATE (see the [support matrix](support-matrix.md)). Bayesian
responses require the documented Gaussian additive models and AllObserved population, with pointwise posterior intervals. Static Bayesian response uses complete observations; temporal Bayesian response also supports the five licensed observation pairs through its observed-data SEM backend. Bayesian Cpdag/Pag response
mixes identified-mass means only. `Admg` response remains refused. Frequentist TemporalCpdag/Pag responses retain completion identified sets; DAG-posterior responses retain atom probabilities and unidentified mass.
`ConditionalEffect` is licensed on `Dag`, `Cpdag`, and `Pag`. The public
license is that matrix, not this page.

Three of these carry parametric scope conditions that the estimator cannot check
at runtime:

* **Front-door two-stage estimation** is the linear-SEM product-of-coefficients
  estimator. It assumes linear structural equations and no direct treatment to
  outcome edge. The general nonparametric front-door formula is reached through
  the ID path and functional plug-in estimation, not through this estimator.
* **Sharp regression discontinuity** uses a caller-supplied bandwidth with a
  uniform kernel and reports a conventional, not bias-corrected, interval. There
  is no data-driven bandwidth selector and no Calonico–Cattaneo–Titiunik robust
  correction, so the estimate is only as defensible as the chosen bandwidth.
* **Propensity-matching standard errors** use a pooled homoskedastic variance
  proxy rather than the full Abadie–Imbens conditional variance estimator. Under
  heteroskedastic outcome variance the reported standard error is biased.

Applying the first two outside their assumed regime produces a biased estimate
with no runtime signal.

`response.kennedy_dr` is also a least-squares construction (additive GAMs plus
a local-quadratic of the doubly robust pseudo-outcome) and needs finite
outcome moments. Unlike the three cases above, it reports
`response.outcome_tail_ratio` at runtime and warns
`response.heavy_tailed_outcome` when the ratio exceeds 20. That warning does
not demote `evidence_status` or `support.status`. See
[causal-responses.md](causal-responses.md#least-squares-kennedy-dr-regularity).

### Bayesian

* Bayesian g-computation;
* temporal Bayesian g-computation;
* Gaussian conditional effects with linear treatment–modifier interactions;
* Gaussian temporal mediation with observed baseline-parent adjustment and
  direct/mediated/total posterior decomposition;
* static and temporal Gaussian response estimation;
* multi-step sustained sequential g-computation with shared stationary
  mechanism draws;
* conjugate Gaussian models;
* Laplace GLM approximation;
* HMC GLMs;
* graph-by-effect posterior envelopes on the exact licensed DAG and
  `TemporalDag` query families described above;
* same-design prior transfer (including licensed Bayesian Pulse,
  single-step Sustained, and temporal `ResponseCurve` on explicit
  `TemporalDag`, when a fixture names source cell, target cell, and
  `PriorCatalog.filter_compatible`);
* effect-level and mapped prior transfer;
* prior catalogs and compatibility filtering;
* power-prior mixtures;
* conflict-sensitive prior weighting;
* transport policies across compatible designs.

Unidentified graph-posterior mass is retained rather than silently
renormalized away. Static DAG-posterior ATE (Bayesian and Frequentist) and
temporal DBN-posterior pulse / single- and multi-step sustained paths consume frozen
known-truth mixture fixtures: the identified atoms pin the conditional effect,
unidentified mass stays visible, and priors do not upgrade structural
identification. Prepared-vs-fresh equality remains an additional execution
invariant rather than the license.

Conditional effects, temporal mediation and DBN-posterior pulse/sustained effects license query-native `cheap` and `full` validation. Multi-step Sustained refuters re-estimate the complete sequential model. Bayesian checks retain each child mechanism for PPC; full prior sensitivity refits the composed effect. Single-regression sensitivity formulas are inapplicable to composed effects, while sequential unobserved-confounder perturbations remain available. Composed mediation and
multi-step sustained posteriors support conjugate and Laplace backends; HMC
composition remains refused. See the [1.2 evidence ledger](v1.2-evidence.md).

## Observation, transport, and interference

These are stage modules. They change what identifies the estimand and are not
hidden behind an ordinary `target_population` flag.

* **Observation** (`antecedent.observation`): complete, right/left/interval-
  censored, truncated, and selected mechanisms. Assumptions are declared
  separately from the recorded columns; MAR / independent censoring is never
  inferred from column presence.
* **Structural transport** (`antecedent.transport`): single-source selection
  diagrams and trial-to-target IPW/AIPW with separate selection and treatment
  overlap diagnostics. Distinct from Bayesian prior/evidence transfer in
  `antecedent.priors`.
* **Randomized interference** (`antecedent.interference`): assignment design,
  exposure mapping, and exposure-contrast estimands with Horvitz–Thompson and
  Hájek estimates. The network is fixed and supplied by the caller.

Multi-source meta-transport, cyclic/equilibrium models, and observational
network interference remain outside the current contract.

## Interventions and counterfactuals

Antecedent includes a structural causal model layer.

Supported mechanisms:

* linear-Gaussian models;
* constant mechanisms;
* discrete mechanisms;
* hierarchical linear and generalized-linear models;
* Minnesota BVAR;
* linear Gaussian state-space models;
* Gaussian-process mechanisms.

Supported interventions:

* hard interventions;
* soft interventions;
* stochastic interventions;
* sequenced interventions;
* temporal policies;
* dynamic policies;
* mechanism overrides.

Do-sampling methods include weighting, KDE, and MCMC.

Counterfactual primitives exist:

* abduction–action–prediction;
* nested counterfactuals;
* temporal trajectories;
* unit-level counterfactual analysis.

`analyze` licenses `Counterfactual` on an explicit Frequentist DAG at
validation `none` as a two-world GCM ITE. Nested counterfactuals, temporal
trajectories, accepted or graph-posterior structure, Bayesian inference, and
cheap/full validation remain refused. The public license is the
[support matrix](support-matrix.md).

## Attribution and diagnostics

Antecedent can analyze:

* anomalous outcomes;
* distribution shifts;
* structural changes;
* mechanism changes;
* change points;
* unit-level change;
* path contributions;
* arrow strength;
* feature relevance;
* root-cause rankings.

Implemented techniques:

* likelihood-ratio tests;
* mean-difference tests;
* classifier-based tests;
* MMD;
* Gaussian KL divergence;
* CUSUM-style scans;
* Shapley attribution;
* coalition caching.

## Validation and sensitivity

Estimate validation:

* placebo refuters;
* random common-cause refuters;
* unobserved common-cause refuters;
* bootstrap refuters;
* data-subset refuters;
* dummy-outcome refuters;
* overlap diagnostics;
* E-values;
* graph refutation.

The 1.2 functional suites refit path-specific effects on row subsets and compare
entire interventional-distribution tables, including conditional strata.
Temporal mediation uses mediator-placebo and contrast-specific stability checks.
Static `MediationEffect` cheap/full uses a mediation-native suite: placebo
mediator (indirect-effect target), random common cause on the requested contrast,
binary mediator-range overlap when applicable, and an 80% subset on `full`.
Continuous-treatment conditional overlap reports unsupported mass under a
descriptive residual-support model; lower `comparison` values mean better
support. Passing any of these checks does not establish causal identification
or validate the structural assumptions.

Sensitivity methods:

* linear sensitivity;
* partial-linear sensitivity;
* nonparametric sensitivity;
* Riesz sensitivity.

Bayesian validation:

* prior predictive checks;
* posterior predictive checks;
* prior sensitivity;
* MCMC diagnostics;
* simulation-based calibration hooks.

Resampling support:

* IID bootstrap;
* Bayesian bootstrap;
* moving-block bootstrap;
* circular-block bootstrap;
* column permutation;
* phase-randomized surrogates.

### "Not applicable" means three different things

The words "not applicable" surface in three unrelated places. A caller who
only sees the bare phrase cannot tell which claim is being made — each is a
different strength of statement, and only one of them is permanent:

* **The support matrix's `not_applicable`** (`SupportRefusal::NotApplicable`,
  wire id `not_applicable`). This is the strongest claim in the system: the
  coordinate — a fixed (query, graph class, structure, inference, validation)
  cell — does not denote, permanently, independent of any run's data. See the
  [support matrix](support-matrix.md).
* **`antecedent-validate`'s `NotApplicable`** (`ValidationOutcome::NotApplicable`
  / `ValidationError::NotApplicable`). This is a per-run, data-dependent skip:
  a requested validator is incompatible with *this run's* problem — an
  E-value on a non-binary treatment, an MCMC diagnostic on a non-MCMC
  posterior, a refuter outside its applicable regime. The same validator can
  run cleanly on a different dataset against the same licensed cell. Callers
  that only read `result.refutations` cannot see this skip — the produced
  `RefutationReport`s and the skips are two disjoint outcomes, and
  `result.refutations` carries only the former. Every execute path now emits
  one `refute.validator.not_applicable` diagnostic per skipped validator into
  `result.diagnostics`, naming the validator and the reason, so the skip is
  visible instead of silently dropped. Its message states explicitly that
  the skip is per-run and data-dependent, not a permanent support-matrix
  refusal, so the two senses of "not applicable" are never mistaken for each
  other at the point a caller actually reads them.
* **The response path's NaN scalar summary**
  (`estimate.response.no_scalar_summary`). A function-valued or
  not-point-identified response has no single-number effect summary;
  `result.effect` is `NaN` and a diagnostic states the scalar reading is "not
  applicable" — the caller must read `result.response` instead. This is not
  an error and not a refusal; it says the wrong field was checked, not that
  anything failed.

None of the three imply each other. A licensed cell can still emit a
per-run validator skip or a NaN scalar summary; a matrix `not_applicable`
cell never reaches either of the other two because `analyze` refuses it
before validation or estimation runs.

## Experimental design

Antecedent can rank candidate actions such as:

* measuring a variable;
* intervening on a variable;
* observing an environment;
* changing a sampling plan.

Ranking criteria:

* expected information gain;
* probability of identification;
* expected effect-interval width;
* decision utility.

The design layer supports batched Monte Carlo evaluation, common random
numbers, and early stopping.

## Incremental state

`CausalState` supports stateful and online workflows.

Available components:

* explicit invalidation;
* incremental OLS;
* streaming covariance;
* particle-filter state-space models;
* local score caches;
* rolling mechanism diagnostics;
* configurable cache budgets;
* prepared analyses;
* progressive and cancellable execution;
* adaptive resampling.

Invalidation does not automatically rerun an analysis.

`PreparedStudy` (`Study::prepare`) caches identification across the licensed
prepared paths. For `AverageEffect`, an estimate click reuses prepare-time
identification (`exec.identify.cached`) on `Dag`, `Cpdag`, `Pag`, and `Admg`,
including the CPDAG MEC envelope, the generalized-adjustment PAG envelope, and
the general-ID bidirected ADMG functional. Static graph
posteriors freeze each weighted atom's
identified/unidentified status, result, and estimand; temporal DBN posteriors
also freeze each identified atom's finite-unfolding indexer. Unidentified
atoms remain unidentified and keep their original weight. Temporal response
and mediation paths retain their existing query-native caches. This is an
execution property, not a license: refused coordinates remain refused.

The sharp-RD estimator remains the deliberate identify-per-click exception.
Progress sinks receive an `identify.compute` label exactly when identification
is computed, so a prepared click's reuse is observable rather than merely
flagged. Prepared graph and query identity are immutable; a changed graph or query
requires a new prepare cycle, so no cache can cross coordinate boundaries. A
same-schema data refresh may reuse structural identification but still
re-estimates from the replacement data.

## Data support

Antecedent supports:

* tabular data;
* time series;
* panel data;
* multi-environment data;
* event data converted into temporal frames.

Python interfaces support NumPy, pandas, and Arrow CDI. Rust uses `TableView`.

## Artifacts

Durable artifact format **0.4** is the 1.0 wire freeze. Versioned artifacts
include:

* graphs;
* graph posteriors;
* model bundles;
* analysis traces;
* causal state.

Artifacts use schema-versioned CBOR containers with optional
Zstandard-compressed sections, selective reads, and memory-mapped access.
