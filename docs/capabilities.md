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
compatibility, but 0.9 has no active allowlist entries and the release gate
rejects new ones. Evidence kinds are scoped: a known-truth fixture may pin only
identification or an effect point, while an internal cross-check may establish
prepared-vs-fresh consistency without pinning the scientific target. Read each
row's `limitations`; a shared method name is not a parity claim.

A licensed row means the staged runtime path, refusal boundary, and recorded evidence
contract are exercised for that coordinate. It does not mean causal assumptions were
verified from the data, intervals are universally calibrated, identification is complete
beyond the named subset, or parametric restrictions disappeared. In particular, priors
cannot convert a nonidentified estimand into an identified one.

At analysis level, the licensed query families are `AverageEffect`,
`ConditionalEffect`, `PathSpecificEffect`, `InterventionalDistribution`,
`ResponseCurve`, `InterventionResponse`, `PulseEffect`, `SustainedEffect`, and
`TemporalMediationEffect`, only on the exact graph / structure / inference /
validation rows in the matrix. Root query types outside that list —
`Counterfactual`, static `MediationEffect`, and all six derivative query types —
have no licensed `analyze` cell in 0.9. Importability is not a license.

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
interpreted as temporal by default. `Cpdag`, `TemporalCpdag`, and `TemporalPag`
are implemented graph/interchange types, but 0.9 licenses no analysis cell on
them. In particular, successful completion to a DAG does not turn an
incomplete-class cell into a licensed one.

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
effect envelopes. Static graph-posterior analysis is limited to
`AverageEffect` with DAG atoms. Temporal graph-posterior analysis is limited to
pulse and single-step sustained effects with `TemporalDag` atoms and validation
`none`. Frequentist mixtures, response mixtures, and ADMG/CPDAG/PAG posterior
atoms are refused.

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
explicit graph completions. Licensed PAG analysis is `AverageEffect` only; this
is not a licensed `ResponseCurve`, path-specific, distribution, or mediation
surface. Full PAG-native ID and IDC are outside the supported scope.
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
* selected-outcome IPW and cross-fitted AIPW, plus marginal right/left-censoring
  IPCW, composed into point-only response curves under explicit observation
  assumptions.

Response results keep structural identification, empirical support, and
uncertainty kind as separate axes. Pointwise and simultaneous bands are not
aliases. Observation-adjusted curves omit joint observation/curve uncertainty
bands rather than reusing invalid complete-data intervals. Interval censoring
and truncation remain Gaussian-likelihood stages, not a causal-response MLE.
Bayesian inference and one-shot `discovery=` on response queries fail closed.
The list above is inventory. Derivative cells are refused by `analyze`.
`ResponseCurve` and `InterventionResponse` are licensed on `Dag` and
`TemporalDag` under Frequentist inference (see the
[support matrix](support-matrix.md)); Bayesian response and TemporalCPDAG/PAG
response remain unlicensed. The public license is that matrix, not this page.

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
* conjugate Gaussian models;
* Laplace GLM approximation;
* HMC GLMs;
* graph-by-effect posterior envelopes on the exact licensed DAG and
  `TemporalDag` query families described above;
* same-design prior transfer;
* effect-level and mapped prior transfer;
* prior catalogs and compatibility filtering;
* power-prior mixtures;
* conflict-sensitive prior weighting;
* transport policies across compatible designs.

Unidentified graph-posterior mass is retained rather than silently
renormalized away. Current graph-posterior evidence is an internal
prepared-vs-fresh cross-check, not a known-truth pin for the mixture effect.

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

`analyze` refuses `Counterfactual`; it is not a licensed staged cell. The
public license is the [support matrix](support-matrix.md).

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

Sensitivity methods:

* linear sensitivity;
* partial-linear sensitivity;
* nonparametric sensitivity;
* Riesz sensitivity.

Bayesian validation:

* prior predictive checks;
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

`PreparedStudy` (`Study::prepare`) does not cache identification uniformly
across graph classes. For `AverageEffect`, an estimate click reuses
prepare-time identification (`exec.identify.cached`) on `Dag`, `Cpdag`, and
`Admg` without bidirected edges — this is the engine's caching behaviour, not a
license: `AverageEffect` on a `Cpdag` is refused, so only the `Dag` and `Admg`
arms are reachable from a licensed cell. `Pag`, bidirected `Admg`, `graph_posterior`
structures, and the sharp-RD estimator are not cached: prepare() still
accepts and freezes them, but each estimate click re-runs identification
against the frozen graph rather than reusing a stored result. The frozen
graph, query, and identifier never change across clicks either way — only
whether the identification step itself is skipped or repeated.

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
