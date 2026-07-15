# TODO

Items are marked DONE with notes until independently verified and removed. Do not remove an item that you have just finished fixing.

---

## P1 — Graph-layer soundness (remaining)

### P1.12 Discrete conditional mechanism: linear-probability fits used as softmax logits — incomplete
`crates/causal-model/src/registry.rs:283-297`, `crates/causal-model/src/mechanism.rs:304-336`
One-vs-rest least squares on category indicators produces predicted probabilities in [0,1]; these
are stored as `logit_coeffs` and passed through softmax at evaluation. softmax(π) ≠ π — true
(0.9, 0.1) becomes ≈ (0.69, 0.31); all parent-conditional discrete sampling and `log_prob_column`
values are biased toward uniform.
**Done (interim):** evaluation applies `ln(clip(π))` before softmax so recovered probs ≈π.
**Still open:** fit true multinomial-logit coefficients via IRLS (blocked on P5 multinomial GLM).
The interim recover-π trick is numerically close but is not a proper MLE and does not unblock
likelihood-based model comparison for discrete conditionals.

**Related leftover from P1.7 (otherwise fixed):** cycle/collider conflicts are recorded out-of-band
(`conflict_edges` + `orientation.conflicts` diagnostics) and runs continue, but Tigramite-style
`x-x` Endpoint marks are still deferred pending an `Endpoint` enum extension.

---

## P4 — Algorithm parity upgrades (bring implementations up to their published names)

### P4.3 LPCMCI: from FCI-lite to LPCMCI
`crates/causal-discovery/src/lpcmci.rs:78-97` runs the PC1+MCI engine plus rules
{collider, R1, R2, R3, disc-path}. R1/R2/R4 and lagged `o→` init are fixed; close the remaining
algorithmic gap: middle marks, weakly-minimal separating sets, interleaved ancestral
edge-removal/orientation phases, and rules R8, R9, R10 (uncovered potentially directed paths) —
required for FCI-style completeness.

### P4.4 J-PCMCI+ per Günther et al.
`crates/causal-discovery/src/jpcmci_plus.rs:127-183` runs PCMCI independently per environment,
pools links by intersection with `p = max` (`pool_scored_links`, lines 226-258 — whose doc promises
*union* semantics; fix doc or code), and context variables never enter any CI test
(`attach_context_nodes`, lines 260-294, is decoration). The published algorithm augments the
variable set with observed context + dataset/time dummy variables and runs PCMCI+ once on pooled
data with link assumptions.
**Immediate bug regardless of redesign:** line 145 keeps only the **last environment's** sepsets
(`last_sepsets = engine_result.sepsets`) for collider orientation of the pooled graph.
Also: the `MultiEnvSamplePlan` built and validated at lines 105-143 is discarded (each env rebuilds
its own frame) while its byte counts are reported in diagnostics — wire it in or drop it.

### P4.5 RPCMCI: masking, not row-splicing
`crates/causal-discovery/src/rpcmci.rs:283-309` (`subset_series`) gathers regime rows by index and
re-declares them a contiguous series, so lagged pairs span regime gaps — statistically wrong CI
tests for interleaved regimes. Saggioro et al. mask samples instead and alternate between regime
assignment and per-regime discovery; the alternating optimization is entirely absent
(`run_median_split` is a stand-in heuristic).
**Fix:** implement masked CI evaluation (only use effective rows whose full lag window lies within
one regime), then the alternating assignment loop.

### P4.7 Generalized/PAG identification beyond the empty set
`crates/causal-identify/src/generalized.rs:98-121` tests only `Z = ∅` per MAG completion; any
confounded-but-adjustable completion reports NotIdentified. Implement generalized adjustment-set
search per completion (candidate sets from possible ancestors, m-separation on legal MAGs),
and document the current limitation loudly in the module docs until then (frontdoor.rs:3-16 is the
model for honest limitation docs). MAG completion filter is in place (`is_mag_completion`). The full ID/IDC algorithm
is roadmap — see P5.3.

### P4.8 GCM attribution parity (DoWhy-GCM)
- `attribute_unit_change` (`crates/causal-attribution/src/unit_change.rs:80-83,154-183`): abduction
  runs and is discarded (`let _ = exo;`); the payoff is the linear surrogate `Σβᵢ(xᵢ−refᵢ)` — for
  an additive game the Shapley loop is a tautology (φᵢ = βᵢ(xᵢ−refᵢ) exactly), and non-LinearGaussian
  mechanisms silently get `betas = vec![1.0; n]`. Implement the real payoff: evaluate the outcome
  mechanism on coalition-mixed parent values with the abduced noise (Budhathoki-style factual vs
  counterfactual output decomposition). Also: per-unit MC stderrs are averaged as if they were a
  mean stderr (lines 124-139) — aggregate with 1/√n.
- Anomaly attribution (`crates/causal-attribution/src/anomaly.rs:33-97`): implement Janzing et al.
  2020 — IT/outlier score of the target distributed over ancestor **noise terms via Shapley**
  (replace noise coordinates with reference draws). The current per-node −log p(y|parents) +
  |residual| conflates "node is anomalous" with "node received anomalous input", yet the facade
  exports it as `anomaly_attribution` (`crates/causal/src/gcm.rs:123-132`).
- `feature_relevance` (`crates/causal-attribution/src/feature_relevance.rs:12-69`): currently a
  one-at-a-time finite-difference do-contrast |E[Y|do(X=μ+δ/2)] − E[Y|do(X=μ−δ/2)]| — no
  interactions, no efficiency property. Implement Shapley feature relevance with
  marginal/conditional randomization (the Shapley engine in `shapley.rs` is verified correct;
  reuse it).
- `distribution_change` (`crates/causal-attribution/src/distribution_change.rs:30-35`): structure
  is correct Budhathoki 2021; add the KL-based target functional (DoWhy's default; `gaussian_kl`
  is fixed), and
  use common random numbers across coalition payoffs (seed is currently `seed + mask`, line 267 —
  extra MC variance; exact-mode efficiency is unaffected but sampled modes pay for it).

### P4.9 do-samplers: bias and dead code
`crates/causal-model/src/do_sampler.rs`
- `WeightingDoSampler` (lines 128-151): the IPW numerator was never implemented (`lp_do` computed
  as zeros then `let _ = lp_do[i]; let _ = t_do;`); the kernel bandwidth is the mechanism residual
  SD σ — a fixed bandwidth giving O(σ²) smoothing bias that never shrinks with n, plus a `min(1e6)`
  weight cap. The conformance test passes only because its data is noiseless. Use a shrinking
  bandwidth (e.g. Silverman on the treatment margin) and finish or remove the numerator. The
  non-Gaussian branch (lines 143-149) degenerates to exact matching — error for genuinely
  continuous treatments.
- `McmcDoSampler` (lines 291-349): the chain targets a Silverman-KDE of ≥64 pilot draws, not the
  interventional law, and the docstring's "exact when the proposal is the target" describes
  independent MH, not the random-walk implemented. MH mechanics are correct; fix the docs and
  consider targeting the mechanism density directly.

### P4.10 Matching: variance and bias correction
`crates/causal-estimate/src/propensity/stratification.rs:334-337` treats matched differences as
i.i.d. (`sample_std/√n`); with-replacement donor reuse makes them correlated → understated SE.
Implement the Abadie–Imbens (2006) variance with donor-usage counts K_i, add the regression bias
adjustment, and document that the bootstrap is invalid for NN matching (Abadie–Imbens 2008). This
is DoWhy-parity-level today but — unlike the library's other simplifications — undocumented.

---

## P5 — Roadmap features from DESIGN.md not yet built

DESIGN.md is the roadmap; these are its unimplemented chapters, listed so nothing is lost. Ordered
roughly by how much current claims/outputs depend on them.

1. **Static (non-temporal) discovery** (DESIGN.md:1211-1221): PC, FCI, RFCI, GES, LiNGAM,
   score-based search / NOTEARS. `causal-discovery` is temporal-only today. The Meek-rule and
   CI-test infrastructure already exists and is verified correct — PC is the natural first target,
   as DESIGN says.
2. **Bayesian graph discovery** (DESIGN.md:1281-1305): `GraphPosteriorEngine`, MCMC/enumeration/DBN
   structure search. Requires adding the documented `causal-discovery → causal-prob` dependency.
3. **Deep identification** (DESIGN.md:868-882, 903, 925): ID algorithm for semi-Markovian models,
   IDC, hedge certificates, `AutoIdentifier`, memoized recursion; maximal adjustment sets; the two
   missing `IdentificationStatus` variants (`IdentifiedUnderParametricRestrictions`,
   `IdentifiedUnderPriorRestrictions` — `crates/causal-core/src/identification.rs:11-20`).
4. **Statistical layer** (DESIGN.md:1000-1061): multinomial logistic (unblocks P1.12), negative
   binomial, probit IRLS (unblocks P4.12), robust M-estimation, ridge/lasso (optional
   separation fallback; hard-fail already shipped); **robust covariance §11.3** — HC0–HC3, cluster, multiway, HAC/Newey-West (zero hits
   repo-wide today; SEs are homoskedastic-analytic or bootstrap); shared resampling engine §11.4
   additions — cluster and stationary-block bootstrap, permutation resampling.
5. **Mechanism families** (DESIGN.md:1422-1429): BVAR, state-space, GP, hierarchical (only
   conjugate Gaussian + Laplace GLM exist). Counterfactual trajectories (line 1637).
   Simulation-based calibration (line 1801). ESS/R-hat diagnostics
   (`crates/causal-prob/src/diagnostics.rs:3` explicitly defers). Bayes-factor CI and posterior
   dependence probability (DESIGN.md:1152-1157).
6. **Performance infrastructure** (DESIGN.md:983, 2112-2139, 2883-2903): runtime-dispatched SIMD
   kernels (nothing today: dispatch is compile-time `cfg!` to autovectorized loops,
   `crates/causal-kernels/src/dispatch.rs:21-26`); the missing kernels from the §21 list
   (covariance, standardization, pairwise distance, contingency, bootstrap weights); the documented
   feature-flag surface (`rayon`, `simd-runtime`, `blas`, `polars`, `serde-json`,
   `gaussian-process`, `hmc`, `smc`, `python`, `networkx-io`, `plot-data` — none exist; actual
   flags are `arrow`, `faer`, and undocumented `portable-optimized`). Note: `rayon` appears nowhere;
   parallelism is hand-rolled `std::thread::scope` (`engine.rs:412,499`) — decide whether the
   roadmap keeps rayon or blesses the current approach.
7. **Serialization** (DESIGN.md:185, 2273, 2289): zstd section compression (fields always `None`,
   `crates/causal-io/src/container.rs:163-165`); real version migrations (only identity 0.1→0.1
   exists, `migrate.rs:16-37`); GML and NetworkX-compatible exchange; model bundles.
8. **Data model** (DESIGN.md:310, 348, 458-508, 2348): `EventData`; `SampleRequest` as specified;
   the five missing split strategies (random-IID, grouped/cluster, blocked-temporal,
   rolling-origin — only discovery/estimation-gap, environment-holdout, regime-holdout exist,
   `split.rs:41,143,176`); Arrow C Data Interface zero-copy (today Arrow enters via in-process
   `RecordBatch` and is copied, `arrow_adapter.rs:31-35` — the copy is at least diagnosed).
9. **Graph algorithms** (DESIGN.md:623-641, 671): Markov blankets; intervention/mutilation via
   overlays instead of cloning (`mutilate` returns a full new Dag,
   `crates/causal-graph/src/ancestry.rs:73`).
10. **causal-expr** completions: simplification, LaTeX rendering, compiled evaluators.
11. **Core query model** (DESIGN.md:727-739): `CausalQuery::Distribution` and `PathSpecific`
    variants (code has undocumented `MechanismChange`/`UnitChange` instead — reconcile the roadmap
    with what emerged).
12. **Python packaging** (DESIGN.md:2321-2338): wheel matrix verification and explicit
    `catch_unwind` at the FFI boundary rather than relying on PyO3's PanicException
    (`py.typed` + stubs landed with P2.5).

---

## P6 — Code quality: DRY / SOLID / idiomatic

### P6.9 API hygiene — remaining
Done already: `FdrControl`/`DiscoveryAccept`/`MissingPolicy`/`ParCorrMode`; `series_len` checked;
planner validates DAG vars; overlay `active` in `is_empty`; ptrs `pub(crate)`; gcm uses
transparent `AnalysisError`.
Still open:
- `KnnCmiWorkspace.distances` never used while the hot loop allocates fresh vecs
  (`crates/causal-stats/src/ci/types.rs:31` / `advanced.rs`).
- `IdentificationWorkspace { _private: () }` threaded through a trait whose impls also all ignore
  `assumptions` (`crates/causal-identify/src/identifier.rs:19-22`).
- `PreparedCiTest` never used beyond a shape check — DESIGN §12 prepare-once contract
  (`crates/causal-discovery/src/engine.rs:331-338`).

---

## P7 — DESIGN.md maintenance (roadmap stays; fix internal inconsistencies and stale facts)

Per project convention, DESIGN.md leads the code — do **not** delete unbuilt sections. But the
document contradicts itself and reality in places that aren't roadmap:

1. Two different Python layouts described (§3 lines 96-98: `python/src/causal/` + `rust/`; §25.1
   lines 2321-2338: flat `causal/` + `_native.*`); code matches neither exactly. Pick one.
2. §3.2 (lines 222-227) requires validate/design/state to depend on "all analysis crates" while
   §3.1's own responsibility statements (lines 171-181) imply far fewer; code followed §3.1.
   Reconcile.
3. Parity status vocabulary (lines 2466-2473: `not_planned/planned/implemented/conformant/
   deviates/blocked`) vs actual manifests using `pending/in_progress/done/intentional_deviation`
   (`parity/dowhy.toml:2`). Standardize one vocabulary and use it in both.
4. Dependency diagram (lines 191-227) stale in both directions (e.g. discovery lacks the documented
   causal-prob edge; undocumented data→kernels, prob→kernels, identify→data, model→kernels,
   counterfactual→data+graph, attribution→data+graph+stats, io→estimate+identify). All real edges
   point downward — no layering violations — so this is purely a diagram refresh.
5. Stale facts: root dir named `causal-rs/` (line 73); conformance layout
   (`paper_examples|generated|reference_outputs`, lines 103-106) vs actual per-domain dirs;
   `CausalError` described as a core type (it's a facade alias of `AnalysisError`); "graph overlays
   instead of cloning" (line 671) vs cloning `mutilate`.
6. Document the systems the repo grew that DESIGN doesn't know about: the deviation-governance
   flow (`parity/*_deviations.md`, `parity/release.toml`, `scripts/gate_*.sh`), ADRs 0012-0017,
   the facade surface (`RefuteSuite`, `gcm` module, `strategy_table`, `discovery_defaults`,
   `analyze_ate`/`analyze` Python entry points, weights on `discover_pcmci`), the
   `portable-optimized` feature, and `CausalQuery::MechanismChange`/`UnitChange`.
7. Add a status marker per DESIGN section (built / partial / planned) so the roadmap-vs-done
   distinction is explicit for readers who don't have this file.
