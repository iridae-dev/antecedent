# Changelog

## 2.1.0

Development branch for Bayesian parity. Added model-scoped Bayesian support for attribution, trial transport, finite-network interference, and CPDAG/PAG class-posterior intervention response; expanded conditional and response posterior evaluation. New learned and specialist estimator contracts remain scoped to their implemented models and evidence.

The evidence-to-decision work in this branch adds a portable restricted-experiment failure snapshot, checked hypothetical study proposals, and arrival validation against actual catalog evidence and provider identity. The registered surrogate route accepts a sufficient joint source margin and executes exact or empirical point estimates through Rust and Python; a separate versioned artifact lets an independent consumer recheck the proof, catalog, laws, and point result. A direct joint source-exchange case carries both intervention values. The bounded single-source TRz decision now follows recursive source exchanges through the remaining controllable set and records a replayable line-11 obstruction only with the complete source experimental family and target observational joint law. Independent binary SCM fixtures check both a restricted-experiment obstruction and a positive formula requiring two successive exchanges. Missing laws and computation exhaustion remain distinct from obstruction. Compatible outcome-kernel contamination reports an assumption range, witnesses, and tipping fraction with independent replay. No interval is claimed for these point routes. The 2.1 release candidate remains pending the compiler and evidence gates.

## 2.0.0

Antecedent 2.0.0 keeps the established `identify → estimate → inspect → refresh → export → consume` lifecycle across discovery, graph uncertainty, identification, estimation, validation, temporal and response analysis, Bayesian inference, attribution, design, state, and artifacts.

### What's New?

#### Learners and heterogeneous effects

- Adds a Rust-native prediction layer with stable learner specifications, capability checks, zero-copy design views, row-indexed folds, and fold-local preprocessing. The causal estimators remain independent of a particular ML provider.
- Adds controlled linear, ridge, logistic, elastic-net, gradient-boosted-tree, random-forest, extra-tree, and CPU-native neural nuisance routes, plus restrained automatic nuisance selection where licensed. The standard Python wheel includes the neural route.
- Adds reusable out-of-fold predictions, fold assignments, nuisance diagnostics, overlap/trimming disclosure, and implementation provenance.
- Adds DML, cross-fitted AIPW, DR-Learner, and honest causal-forest paths. CATE uncertainty remains deliberately limited: a forest's leaf dispersion is a diagnostic, not a pointwise standard error or confidence interval.

#### Structural transport

- Adds explicit population, environment, evidence-regime, sampling, and dependence contracts. Separate experimental marginals never silently become a joint experiment, and a proposed intervention never becomes observed evidence.
- Adds population- and regime-aware functionals with checked derivation DAGs, theorem-scoped single-source identification, target-only identification, checked non-transportability witnesses, and bounded catalog search.
- Adds exact finite-discrete evaluation and prepared statistical transport with factor-level provider bindings, support diagnostics, joint bootstrap uncertainty, complementary-source synthesis, and target response grids.
- Adds durable transport artifacts and migrations that preserve evidence and source lineage across Rust, Python, and independent consumers.
- Distinguishes an unavailable joint/provider from invalid input, theorem-stage non-certification, structural non-transportability, support failure, numerical failure, and budget/cancellation.

#### Claim integrity and release boundaries

- Strengthens typed support, evidence, provenance, calibration, artifact, and refusal contracts across the expanded workflow.
- Makes calibration scope part of the result contract: `calibrated` requires a coverage record that applies to and attests the executing code.


### Breaking changes from 1.11

Python

- The theorem-stage transport names moved from `antecedent.transport` to
  `antecedent.transport.advanced` and are no longer re-exported: `DirectFormula`,
  `NonTransportableCertificate`, `PopulationFactor`, `RecursiveFactorizationFormula`,
  `SelectionDiagram`, `StandardizationFormula`, `TransportCertificate`,
  `TransportIdentification`, `TransportQuery`, `TrialTransportEstimate`,
  `estimate_trial_effect` and `identify`. `OverlapDiagnostic` and
  `TransportOverlapReport` are no longer exported anywhere; they are the type of
  `TrialTransportEstimate.overlap`. `antecedent.TransportQuery` is gone from the
  package root (use `antecedent.transport.advanced.TransportQuery`, or the
  ordinary-question wrapper `antecedent.transport.Transport`). Each old spelling
  raises an `AttributeError` naming its new home; see
  [the migration page](docs/migrations/2.0-transport-day1.md).
- `AcceptedGraph.asserted` and `AcceptedGraph.accepted` are removed: one spelling
  per object, `AcceptedGraph.from_graph` and `AcceptedGraph.from_discovery`.
- Graphs built from a discovery result carry every analysed variable, isolated
  ones included, taken from the new `PcmciDiscoveryResult.variable_names`; a
  result without names raises instead of defaulting to `x`/`y`.
- A refusal's `reason_code` is attached by the native layer; the `reason=<code>:`
  message prefix is no longer parsed into one.
- `PosteriorArtifact.draws` is a getter returning a float64 array.
- Without the native extension and without installed package metadata,
  `antecedent.__version__` is `"unknown"` (it used to be a stale literal).

Rust crates and features

- The Cargo feature `ml-gpu` is renamed `ml-neural` on `antecedent`,
  `antecedent-estimate` and `antecedent-learn`: it is a CPU-only Burn
  `NdArray<f32>` network and there is no GPU backend. The empty features
  `ml-faer` (`antecedent-learn`) and `hmc`, `smc` (`antecedent-prob`) are removed;
  the code they named is always compiled.
- `antecedent-identify::oracle_dot` (the frozen-oracle DOT parser, which panics on
  malformed input) is available only with the new `test-util` feature.
- `IdentityDomain` gains `LearnedTrial` and `TransportCertificate`
  (`IdentityDomain::ALL` is now 13 entries); learned-trial and transport-certificate
  artifact ids are derived in their own domains and differ from any 1.11 digest.
- The graph, posterior and analysis-result conversion functions re-exported from
  `antecedent::io` (`dag_from_dot`, `cpdag_to_json`, `encode_causal_posterior`, ...)
  return `IoError` instead of `CausalError`.

Wire formats and readers

- Readers reject duplicate section ids in a container, trailing bytes after a
  CBOR document, and unknown keys in query and response-query documents
  (`deny_unknown_fields`); a payload that carried a misspelled key used to load
  with the default and now fails.
- A posterior artifact's stored summaries must describe its embedded draws (the
  quantiles to 1e-9 relative, the mean and SD within eight standard errors); one
  whose summaries do not is rejected. External-estimate receipts report
  `evaluated_domain` as `"evaluated"` (it was `"unknown"`), and the analysis
  artifact gains an optional `HedgeCertificateWire`.
- Numerical behaviour changed where a result would have looked stronger than its
  evidence (for example Monte Carlo Shapley returns `Cancelled` unless every
  permutation finished, and `PriorSensitivity::evaluate` refuses when the estimator
  already carries a prior); the per-crate fixes are recorded in the commit history.
