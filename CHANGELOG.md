# Changelog

## 2.1.0

Antecedent 2.1.0 keeps the `identify → estimate → inspect → refresh → export → consume` lifecycle of 2.0.0 and makes every licensed route execute from a checked operation retained at preparation.

### What's New?

#### Checked execution on every licensed route

- Preparation seals the identification proof, class envelope or posterior atoms, procedure, inference settings, and validation suite into the prepared plan. Estimates and refreshes run from that plan after the builder is released, a refresh refuses a schema change, and a one-shot run of a sealed route prepares and executes the same operation and reports the reused identification.
- Static CPDAG and PAG effects, graph-posterior effects, temporal class effects, temporal mediation, static and temporal response, Bayesian specialist procedures, trimmed and untrimmed AIPW, and trial transport execute this way, as do the public transport stages. Sealed static routes keep their projection and tier-replicate diagnostics.
- Each licensed cell in `parity/support_licensed.toml` cites the builder-independent evidence for every estimator it licenses, and the release gate executes that evidence. An independent consumer of an exported sealed result verifies the artifact and reports the checked operation it cannot replay as an unresolved dependency.

#### Bayesian parity

- Adds model-scoped Bayesian support for attribution, trial transport, finite-network interference, and CPDAG/PAG class-posterior intervention response, and expands conditional and response posterior evaluation. Learned and specialist estimator contracts stay scoped to their implemented models and evidence.
- Bayesian transport intervals draw each cited dataset from its own random stream, and a cited law's probabilities must reconcile with its empirical counts before a point and its interval are published together.

#### z-transportability and restricted experiments

- Identifies a target effect from a single source's experiments on a bounded controllable set. The registered surrogate route accepts a sufficient joint source margin; the bounded TRz decision follows recursive source exchanges through the remaining controllable set, takes the source c-factor at each exchange, binds exchanged coordinates to the summation variable or the requested treatment level, and refuses a request whose level no cited regime supports. Independent binary SCM fixtures pin the two-exchange formula at both treatment levels.
- A structural line-11 obstruction is certified from the selection diagram and independently rechecked on replay. A two-source decision identifies from whichever source suffices and combines two complementary factors under either of two factorizations: two disconnected outcome components, or one connected graph whose intervened outcomes m-separate into two groups given the treatments. Each factor is one source's checked derivation bound only to its own regimes; a single connected c-factor whose outcomes stay m-connected given the treatments would need a fabricated joint over both sources' interventions and is refused by name. Two single-family obstructions never combine into a proven obstruction. The component laws execute independently and combine as a point-only product distribution, checked against an independently enumerated SCM.
- Refusals are typed end to end: invalid input, missing evidence, budget exhaustion, cancellation, structural non-transportability, support failure, and numerical failure carry distinct kinds and registered reason codes from identification through estimation, artifacts, and the Python bindings.
- A portable failure snapshot, checked hypothetical study proposals with a budget receipt of evaluated and unevaluated candidates, and arrival validation against actual catalog evidence and provider identity. The point artifact (version 2) embeds the proof, catalog binding, laws, program, and a premises digest; a consumer rechecks all of them under its own evaluation limits and recomputes the point.
- Every bounded decision outcome exposes structured inspection rather than a bare reason string. A not-certified decision carries a typed reason kind and the recursive rules the search explored; because the search built no expression, its proof graph is null and the explored region is reported instead of a fabricated one. A budget or cancellation surfaces through `decide` and the failure snapshot as a limits receipt naming the step and depth limits in force, which budget tripped, and the steps consumed and depth reached when the search stopped, rather than only an exhausted-computation status.
- Compatible outcome-kernel contamination reports an assumption range, witnesses, and a tipping fraction; fixed-graph sensitivity can perturb both response arms together or one declared treatment-level slice while holding the other arm fixed. A named conditional-mechanism route also supports one categorical factor among the outcome's non-treatment parents when all of its graph parents are present in the bound joint parent law and that law factorizes over the fixed DAG. It propagates the changed factor through downstream outcome parents and reports row-wise extremal witnesses. Unsupported external-parent factors and joint deviations remain refused. These are assumption ranges, not sampling intervals.
- Empirical cited tables publish a nominal percentile bootstrap interval, and Bayesian providers publish a posterior equal-tail interval, both labelled with an unmeasured coverage status; exported point artifacts stay point-only.
- The restricted-experiment route is a first-class prepared study in Python: it exports, loads, inspects, previews, and refreshes with a new snapshot identity like every other route.

#### Nested counterfactuals

- The fixed-DAG natural-direct-effect route replays both treatment worlds from one abduced exogenous table. It also fits a non-separable outcome basis, under which the frozen mediator carries its abduced disturbance into a non-additive outcome so the direct effect reads the shared draw. A discriminating fixture confirms the route matches the shared-abduction truth and differs from the disturbance-free value a non-shared draw produces, a separable outcome or a disturbance-free mediator collapses the two, and an underdetermined outcome basis is refused with a typed error. The effect stays point-only.

### Breaking changes from 2.0.0

Python

- Loading an exported result whose route executes from a sealed checked operation keeps the recorded answer and identities, but `loaded.acceptance.verified` is false and `acceptance.status` is `"sealed"`; `acceptance.unresolved` names the checked operation an independent consumer cannot replay. Code that treated `verified` as the only accepted state must accept `sealed`.
- Transport bindings raise the typed `Causal*` exception classes (`CausalCancelledError`, `CausalResourceError`, `CausalUnsupportedError` with a registered reason code, `CausalSerializationError`, `CausalValueError`) instead of `ValueError`.
- `PreparedZTransportStage.estimate` reports the interval `method` and `reason` as separate keys, and its refresh on a handle prepared with empirical counts refuses count-free laws.

Rust

- `IdentificationError` gains typed variants for cancellation, budgets, invalid input, invalid catalogs, missing evidence, and invalid derivations; `identify_z_transport`, `verify_z_transport_derivation`, the failure snapshot, the planner, and proposal replay take `SidLimits` and an `ExecutionContext`.
- `ZTransportArtifactWire` and the sensitivity artifact are version 2; earlier bytes are refused with `IoError::UnsupportedVersion`. Consumption takes `ZTransportConsumeLimits`.
- `EstimationError::Refused` carries the transport reason codes; `IoError` gains `Refused` and `ZTransport` variants.

## 2.0.0

Antecedent 2.0.0 keeps the established `identify → estimate → inspect → refresh → export → consume` lifecycle across discovery, graph uncertainty, identification, estimation, validation, temporal and response analysis, Bayesian inference, attribution, design, state, and artifacts.

### What's New?

#### Learners and heterogeneous effects

- Adds a Rust-native prediction layer with stable learner specifications, capability checks, zero-copy design views, row-indexed folds, and fold-local preprocessing. The causal estimators remain independent of a particular ML provider.
- Adds controlled linear, ridge, logistic, elastic-net, gradient-boosted-tree, random-forest, extra-tree, and CPU-native neural nuisance routes, plus restrained automatic nuisance selection where licensed. The standard Python wheel includes the neural route.
- Adds reusable out-of-fold predictions, fold assignments, nuisance diagnostics, overlap/trimming disclosure, and implementation provenance.
- Adds DML, cross-fitted AIPW, DR-Learner, and honest causal-forest paths. Forest leaf dispersion remains a diagnostic, not inferential uncertainty. The DR-Learner also supports pointwise CATE inference at prespecified profiles that exactly match retained covariate tuples in both arms and pass the propensity-overlap check; it uses cross-fitted DR scores and linear-final-stage HC0 covariance. Penalized final stages withhold inferential standard errors. The pointwise normal bounds are uncalibrated and are not simultaneous confidence bands; synthetic known-truth checks do not establish interval coverage.

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
