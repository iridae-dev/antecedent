# Changelog

## 2.0.0 — draft

This is a release-preparation record. Workspace and Python package metadata are 2.0.0; the tag and publish are still pending, so PyPI still serves the last published release. `calibrated` still requires a coverage record that attests this commit. Historical per-release prose has been removed from the working tree and remains available in Git history.

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

### Cohesive causal workflow

- Keeps the established identify → estimate → inspect → refresh → export →
  consume lifecycle across discovery, graph uncertainty, identification,
  estimation, validation, temporal and response analysis, Bayesian inference,
  attribution, design, state, and artifacts.
- Preserves fail-closed support boundaries, typed assumptions, diagnostics,
  provenance, and durable claims instead of strengthening a result at an API
  or serialization seam.

### Learners and heterogeneous effects

- Adds the Rust-native learner substrate and learner specifications for
  fold-local prediction.
- Adds DML, DR-Learner, and honest causal-forest paths with held-out nuisance
  diagnostics, overlap/trimming disclosure, learner provenance, and explicit
  limits on CATE uncertainty.

### Structural transport

- Adds population and evidence-regime catalogs, typed transport outcomes,
  population-aware expression evaluation, and durable derivations.
- Distinguishes an unavailable joint/provider from an invalid request,
  theorem-stage non-certification, support failure, or numerical failure.

### Calibration status

- Does not claim a new calibration attestation. The long-running measurement
  program remains a release-roadmap item; existing records retain only their
  recorded scope and do not attest this tree.
- Runtime `calibrated` requires a covering record that still attests the
  current code; matching but stale or non-attesting records are
  `scope_not_assessed` (`coverage_record_not_attesting`), not calibrated.
