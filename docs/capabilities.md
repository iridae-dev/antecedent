# Capabilities

The full inventory of what Antecedent implements, area by area. The
[README](https://github.com/iridae-dev/antecedent#readme) carries the
highlights; this page is the reference list. For what is deliberately *not*
implemented, see [Comparison](comparison.md).

## Graphs

Supported graph classes:

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
interpreted as temporal by default.

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

Posterior graph samples can be propagated into downstream effect analyses.

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
* ID and IDC for DAGs and ADMGs;
* hedge certificates;
* nonparametric path-specific identification;
* generalized adjustment for partial graphs;
* unfolded temporal backdoor;
* temporal mediation.

`AutoIdentifier` reports applicable strategies. It does not silently choose an
estimator.

For PAGs, Antecedent uses identification envelopes or explicit graph
completions. Full PAG-native ID and IDC are outside the supported scope.

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
* functional plug-in estimation.

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

### Bayesian

* Bayesian g-computation;
* temporal Bayesian g-computation;
* conjugate Gaussian models;
* Laplace GLM approximation;
* HMC GLMs;
* graph-by-effect posterior envelopes;
* same-design prior transfer;
* effect-level and mapped prior transfer;
* prior catalogs and compatibility filtering;
* power-prior mixtures;
* conflict-sensitive prior weighting;
* transport policies across compatible designs.

Unidentified graph-posterior mass is retained rather than silently
renormalized away.

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

Counterfactual support:

* abduction–action–prediction;
* nested counterfactuals;
* temporal trajectories;
* unit-level counterfactual analysis.

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
* Reisz sensitivity.

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

## Data support

Antecedent supports:

* tabular data;
* time series;
* panel data;
* multi-environment data;
* event data converted into temporal frames.

Python interfaces support NumPy, pandas, and Arrow CDI. Rust uses `TableView`.

## Artifacts

Versioned artifacts:

* graphs;
* graph posteriors;
* model bundles;
* analysis traces;
* causal state.

Artifacts use schema-versioned CBOR containers with optional
Zstandard-compressed sections, selective reads, and memory-mapped access.
