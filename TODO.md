# TODO

Prioritized backlog from the 2026-07-22 full-repo review (math correctness and DoWhy/Tigramite
parity, DESIGN.md conformance, code quality). Ranked by order to address: P0 first. DESIGN.md is
the conceptual roadmap — items in P5 are planned features not yet built, not documentation errors.

Review baseline: 423 tests passing, clippy clean (5 trivial warnings, see P6.14), zero
`todo!`/`unimplemented!`/TODO markers. All findings below were verified against the code by the
reviewing agents; P0.1, P0.2, and P1.1 were additionally confirmed by direct numerical simulation
or execution of a counterexample.

---

## P0 — Confirmed wrong math (small diffs, fix immediately)

### P0.1 `gaussian_kl` log term inverted — DONE
`crates/causal-stats/src/divergence.rs:18`
```rust
Ok(0.5 * ((var0 / var1).ln() + (var0 + (mu0 - mu1).powi(2)) / var1 - 1.0))
```
KL(N(μ0,σ0²)‖N(μ1,σ1²)) = ½[ln(**σ1²/σ0²**) + (σ0²+(μ0−μ1)²)/σ1² − 1]; the code has
`ln(var0/var1)`. Confirmed by simulation: μ0=μ1=0, var0=1, var1=2 returns −0.597 (KL must be ≥ 0);
truth is 0.0966. Negative KL propagates into `residual_likelihood_ratio` (same file, line 79) where
`sqrt(2·KL)` becomes NaN, and thence into `MechanismChangeMethod::LikelihoodRatio`
(`crates/causal-attribution/src/mechanism_change.rs:70`).
**Fix:** flip to `(var1 / var0).ln()`. Add a unit test with unequal variances (the existing test
only covers identical distributions, where the flipped term vanishes) and a non-negativity
property test.
**Status:** Fixed — log term flipped; unequal-variance + non-negativity tests added.

### P0.2 `sample_inv_gamma` inverts the scale parameter — DONE
`crates/causal-prob/src/conjugate.rs:333-337`
```rust
let g = sample_gamma(shape, 1.0 / scale, rng);
1.0 / g.max(f64::MIN_POSITIVE)
```
If G ~ Gamma(α, rate=β) then 1/G ~ InvGamma(α, β); the code passes `rate = 1/β`, so σ² draws are
off by a factor β². Confirmed by simulation: α=5, β=4 (true mean 1.0) gives sampler mean 0.0626 =
(1/β)/(α−1). Every NIG posterior draw in `draw_nig` (conjugate.rs:295-324) has wrong σ² and wrong
coefficient spread (`sigma * scale_chol`); posterior means are unaffected (computed separately),
which is why `nig_draws_include_sigma2` (positivity-only assertion) passes.
**Fix:** pass `scale` as the gamma rate directly. Add a moment test (empirical mean ≈ β/(α−1)).
**Status:** Fixed — InvGamma now uses rate=β; moment test `α=5,β=4` → mean ≈ 1.0.

### P0.3 `StreamingCovariance::append` computes wrong cross-covariances — DONE
`crates/causal-state/src/suff_stats.rs:166-171`
```rust
for j in 0..self.dim { self.m2[i * self.dim + j] += delta[i] * di; }  // j never used
```
Every entry of row i gets `delta[i]*di`; correct Welford cross-covariance update is
`delta[i] * (x[j] − mean_new[j])`. Off-diagonals are wrong and the matrix is not symmetric.
Re-exported by the facade as `causal::StreamingCovariance`. The test at suff_stats.rs:241-252 uses
perfectly correlated equal-variance data for which the bug is invisible.
**Fix:** use `delta[i] * d[j]` where `d[j] = x[j] − mean_new[j]`. Replace the test data with
distinct-variance, partially correlated columns and assert against a batch covariance computation.
**Status:** Fixed — Welford uses `delta[i]*d[j]`; batch-reference test with partial correlation.

### P0.4 Block-shuffle permutation p-values anti-conservatively biased — DONE
`crates/causal-stats/src/ci/block_shuffle.rs:61-68`
Failed null replicates are counted via `partial_correlation(...).unwrap_or(0.0)` — i.e. as
"never extreme" — deflating p-values. Mass replicate failure still reports significance.
**Fix:** propagate replicate errors (the weighted variant at
`crates/causal-stats/src/ci/parcorr_variants.rs:291` already does this correctly — mirror it), or
at minimum count failures and error above a threshold.
**Status:** Fixed — `block_shuffle_pvalue` returns `Result`; replicate `None` → `Err`; wired through `parcorr.rs`.

### P0.5 E-value validator is vacuous under the default config — DONE
`crates/causal-validate/src/evalue.rs:35,59,77-80`
The E-value formula `rr + sqrt(rr*(rr−1))` is mathematically ≥ 1.0 for every input, and the default
`threshold: 1.0` with `passed = e_value >= threshold` means every estimate — including exactly zero
effect — passes. `ValidationSuite::full_effect()` uses this default, so the suite's E-value verdict
is unconditionally "passed". (The E-value formula itself is correct — exact VanderWeele–Ding with
the `exp(0.91·d)` continuous-outcome conversion.)
**Fix:** require an explicit threshold (no default), or default to a meaningful value (commonly
E ≥ 2 is read as "moderate robustness") and document the interpretation. Add a test that a zero
effect fails.
**Status:** Fixed — default threshold 2.0 (moderate robustness); documented that DoWhy has no
pass/fail gate; zero-effect test asserts failure under the default.

---

## P1 — Graph-layer soundness (wrong separations / unsound orientations)

These make the latent-variable stack (LPCMCI, PAG identification, refuters that consult
m-separation) return wrong answers, not just weaker ones.

### P1.1 ADMG m-separation misses collider-connected augmentation — DONE
`crates/causal-graph/src/msep.rs:126-195` (`build_moral_ancestral`)
Moralization only converts directed/bidirected edges to undirected and marries direct co-parents.
Correct m-separation (Richardson) requires the augmented graph: for each bidirected-connected
district C in the ancestral subgraph, `C ∪ pa(C)` must form a clique (equivalently: marry any two
nodes joined by a collider-connecting path through bidirected chains). Confirmed by execution:
`X → A ↔ B ← Y`, query `X ⟂ Y | {A,B}` wrongly returns separated.
**Fix:** after restricting to the ancestral set, compute bidirected-connected districts and clique
`C ∪ pa(C)` per district before the co-parent marriage step. Add the counterexample above as a test.
**Status:** Fixed — ancestral districts clique `C ∪ pa(C)`; counterexample test added.

### P1.2 PAG m-separation drops the collider-descendant rule; truncation returns "separated" — DONE
`crates/causal-graph/src/pag.rs:340-365` (`path_active_given`), `msep.rs:309-361`
A collider on a path is treated as open only if the collider itself is in Z; the m-connection
criterion opens a collider if it *or any descendant* is in Z (e.g. `X → C ← Y`, `C → D`, Z = {D}
is connected; code says separated). Separately, the `max_paths`/`max_len` budget truncation in
`is_m_separated` returns `Ok(true)` ("separated") with no flag when the only active path was
dropped.
**Fix:** precompute "possible descendants in Z" per node (using definite-status descent for PAGs)
and open colliders accordingly; on budget exhaustion return an explicit indeterminate/error rather
than "separated".
**Status:** Fixed — colliders open via definite directed descendants; truncated searches return
`GraphError::SearchBudgetExhausted` instead of claiming separation.

### P1.3 Discriminating-path rule does not implement discriminating paths — DONE
`crates/causal-discovery/src/discriminating_paths.rs:20-94`,
`crates/causal-discovery/src/rule_scheduling.rs:346-407`
Three compounding bugs vs FCI R4 / LPCMCI:
1. The path finder only requires a chain of definite directed edges ending in a circle at c; the
   definition requires a path ⟨a, d₁,…,d_k, c, b⟩ where **a and b are non-adjacent** and **every
   intermediate d_i is a collider on the path AND a parent of b**. None of this is checked, so the
   rule fires on non-discriminating triples.
2. The sepset test is inverted: `rule_scheduling.rs:373-377` asks whether `b ∈ Sep(a,c)`; R4 asks
   whether **c ∈ Sep(a,b)** (a, b are the non-adjacent pair). `discriminating_paths.rs:98-102`
   hardcodes the inverted convention.
3. The collider branch should orient both edges at c (`d_k *→ c ←* b`); the code only sets an
   arrow at b on the c–b edge.
**Fix:** implement the full path definition (non-adjacency + collider-and-parent-of-b intermediates),
correct the sepset lookup to `Sep(a,b)`, and orient both endpoints in the collider branch. Add
fixture tests from the FCI literature (e.g. Zhang 2008 examples).
**Status:** Fixed — Zhang discriminating-path definition; `c ∈ Sep(a,b)` gate; collider orients
`dₖ *→ c ←* b`; non-collider orients `c → b`. Zhang minimal fixture + R4 rule tests added.

### P1.4 LPCMCI R1 under-orients and requires too-strong premises — DONE
`crates/causal-discovery/src/rule_scheduling.rs:120-178`
FCI R1: if `a *→ b o–* c` (any mark at a, arrow at b) and a,c non-adjacent, orient `b → c` (tail at
b **and** arrow at c). The code (i) requires the a–b edge to be fully directed tail→arrow, missing
`a o→ b` / `a ↔ b` premises, and (ii) sets only the tail at b (line 163 keeps `mark_at_c`
unchanged), producing `b —o c` where the literature derives `b → c`.
**Fix:** relax the premise to arrow-at-b, and set both endpoint marks in the conclusion.
**Status:** Fixed — premise is arrow-at-b; conclusion sets Tail at b and Arrow at c.

### P1.5 LPCMCI R2 checks the circle on the wrong endpoint and can overwrite tails — DONE
`crates/causal-discovery/src/rule_scheduling.rs:211-231`
FCI R2: if `a → b *→ c` or `a *→ b → c` and `a *–o c` (circle **at c**), orient arrow at c. The
code requires `at_a_ac == Circle` (circle at a), and only skips when at_c is already Arrow — so it
can overwrite a Tail at c with an Arrow (unsound) and fails to fire when at_c is Circle but at_a is
Tail. Its chain cases also require specific circle marks at b, missing e.g. `a → b → c`.
**Fix:** fire only when the mark at c on the a–c edge is Circle; never overwrite a Tail; cover both
chain forms per the rule.
**Status:** Fixed — circle-at-c gate; both chain forms (`a→b *→c`, `a*→b →c` incl. fully directed);
Tail at c is never overwritten.

### P1.6 LPCMCI inserts lagged links as tail–arrow directed edges — DONE
`crates/causal-discovery/src/evidence.rs:154-193` (`pag_from_scored_links`)
Contemporaneous pairs get o–o (correct) but lagged links get `insert_directed` (tail at source). In
a PAG a tail asserts ancestorship; a dependent lagged pair may be `X_{t−τ} ↔ Y_t` (lagged latent
confounding). LPCMCI (Gerhardus & Runge 2020) initializes lagged links as `o→` (arrow at the later
node by time order, circle at the earlier).
**Fix:** insert lagged links with circle-at-source, arrow-at-target; let the rules upgrade circles
to tails.
**Status:** Fixed — lagged inserts use `insert_circle_arrow(src, tgt)`.

### P1.7 Orientation conflicts abort the run or are silently first-writer-wins
`crates/causal-discovery/src/orientation.rs:406-475`, `crates/causal-graph/src/cpdag.rs:168-188`
`orient_undirected` returns `GraphError::Cycle` when an orientation would create a directed cycle,
which propagates as a hard error and fails the whole PCMCI+ run; conflicting collider orientations
are silently skipped. Tigramite records conflicts (`x-x` marks) and continues.
`OrientationState.conflicts` exists but is never incremented by any rule
(`orientation.rs:98` — `RuleDelta.conflicts` is never written).
**Fix:** convert cycle/collider conflicts into recorded conflict marks (populate the existing
`conflicts` machinery), keep the run alive, and surface the conflicts in the result diagnostics.

### P1.8 `TemporalPag::set_marks` skips the acyclicity check
`crates/causal-graph/src/temporal_pag.rs:191-204`
Unlike `Pag::set_marks`, the temporal variant does not run the acyclicity check, so LPCMCI rules
can introduce (almost-)cycles silently.
**Fix:** apply the same validation as `Pag::set_marks`.

### P1.9 `CompletionSampler` produces non-MAG "MAG" completions
`crates/causal-graph/src/completion.rs:75-126`
Enumerates all 2^k circle assignments rejecting only directed cycles; ancestral-graph legality
(no almost-directed cycles; tails implying ancestorship) is not checked, so downstream consumers
(generalized identification, P4.7) iterate over illegal completions. Also contains a committed
stream-of-consciousness comment ("Actually invalid ones shouldn't count toward max? Plan says…") —
clean up while in there. Related: `crates/causal-graph/src/pag.rs:84-95` `is_pag_legal` is
`const fn(_) -> bool { true }` and its call site discards the result — implement or remove.
**Fix:** add MAG legality checks (ancestrality + no almost-directed cycles) to the completion
filter; implement `is_pag_legal` for the invariants it names.

### P1.10 Counterfactual `predict` silently ignores Soft and Stochastic interventions
`crates/causal-counterfactual/src/engine.rs:204-221`
The world loop consults only `overlay.hard_set` and `overlay.shifts`; `overlay.soft` and
`overlay.stochastic` are populated without error by `InterventionOverlay::from_interventions`
(`crates/causal-model/src/overlay.rs:123-132`) but never read, so a counterfactual world requesting
one silently uses the factual mechanism.
**Fix:** either implement them (mirror `sample.rs::sample_with_overlay` /
`sample_structural_with_overlay`) or reject them with an explicit error in `predict`. Note the
related semantics inconsistency: `soft_to_slot` gives soft interventions different noise semantics
on the two sampling paths (`crates/causal-model/src/sample.rs:159-167`) — unify while fixing.

### P1.11 Model falsification tests residual independence against descendants
`crates/causal-model/src/evaluate.rs:201-219` (`residual_independence_tests`)
Each node's inferred noise is tested against every non-parent, including the node's own
descendants. In a true ANM, Y's noise causes Y's descendants, so this dependence is expected — the
test sets `falsified = true` (evaluate.rs:86-92) for correctly specified models with any chain
structure (X→Y: X's residual vs Y correlates perfectly).
**Fix:** restrict the test set to non-descendants (compute descendants per node; the topo-order
predecessor logic in `local_markov_tests`, evaluate.rs:236-241, is the model to follow).

### P1.12 Discrete conditional mechanism: linear-probability fits used as softmax logits
`crates/causal-model/src/registry.rs:283-297`, `crates/causal-model/src/mechanism.rs:304-336`
One-vs-rest least squares on category indicators produces predicted probabilities in [0,1]; these
are stored as `logit_coeffs` and passed through softmax at evaluation. softmax(π) ≠ π — true
(0.9, 0.1) becomes ≈ (0.69, 0.31); all parent-conditional discrete sampling and `log_prob_column`
values are biased toward uniform.
**Fix:** fit multinomial-logit coefficients via IRLS (blocked on P5 multinomial GLM), or as an
interim fix apply `ln(clip(π, ε, 1−ε))` before storing so softmax recovers ≈π.

---

## P2 — Honest reporting / silent failures

### P2.1 Hard-coded `se_analytic: 0.0` in two estimators
- `crates/causal-estimate/src/conditional.rs:165-173` (`ConditionalLinearAdjustment`): zero SE with
  `se_bootstrap: None` is a claim of exact certainty. Implement the delta-method SE
  `sqrt(gᵀ σ²(XᵀX)⁻¹ g)` with `g = e_T + w̄·e_{T×W}` — `invert_square` and `form_xtx` are already
  imported in this file.
- `crates/causal-estimate/src/temporal_mediation.rs:124-132`: same; reuse the Sobel SE
  implementation that already exists in `frontdoor.rs:234-254`.
If an SE genuinely can't be computed, report `None`/NaN — never 0.0.

### P2.2 Bayesian facade SE is wrong by ~√n_draws; Laplace fixes σ²=1
- `crates/causal/src/analysis.rs:1021` reports `sd/√n_draws` (Monte-Carlo error of the posterior
  mean, ~31× too small at 1000 draws) where every other estimator reports a sampling SE. Report the
  posterior sd itself.
- `crates/causal-prob/src/laplace.rs:327-330` fixes the working variance σ²=1 for GaussianIdentity,
  so the posterior scale is wrong unless residual variance ≈ 1. Estimate the residual variance and
  scale the curvature.

### P2.3 GLM convergence flag is never checked (separation goes undiagnosed)
`crates/causal-stats/src/glm.rs:209-272` has no separation detection; under complete separation the
fit returns `converged: false` — but no caller checks it: `fit_propensity`
(`crates/causal-stats/src/propensity.rs:64-73`), `PropensityModel::fit`
(`crates/causal-estimate/src/prepare.rs:57-72`), `GlmAdjustmentAte::fit`
(`crates/causal-estimate/src/glm_adjustment.rs:218-230`). A separated propensity model yields scores
pinned at the 1e-9 clamp and IPW/AIPW run on degenerate weights silently. DoWhy inherits sklearn's
L2-regularized logistic here, so behavior diverges from DoWhy exactly in the separation regime.
**Fix:** propagate `converged: false` as an error or a loud overlap-report diagnostic at every call
site; consider an optional ridge penalty as a separation fallback (ties into P5 GLM work).

### P2.4 Systemic silent bootstrap replicate-dropping in causal-estimate
Failed replicates are skipped uncounted, so `se_bootstrap` is a `sample_std` over a
selection-biased survivor set, and <2 survivors yields `Some(NaN)` — callers can't distinguish
"bootstrap failed" from "ran". Sites:
`propensity/weighting.rs:146`, `propensity/stratification.rs:154`, `propensity/matching.rs:153`,
`propensity/distance.rs:159-176`, `aipw.rs:254`, `iv.rs:266,466-476`, `frontdoor.rs:319-321`,
`rd.rs:273`, `util.rs:79-82` (all under `crates/causal-estimate/src/`).
**Fix:** count failures, expose `replicates_failed` in the result, and error (or return `None` with
a diagnostic) above a failure threshold. Best done together with P6.3 (consolidate the 8 copy-pasted
bootstrap loops onto one shared helper so the fix lands once).

### P2.5 Python bindings: suppressed refuters, swallowed errors, test execution context
`python/src/lib.rs`
- Line 307: Bayesian mode overwrites the user's `refute=True` with `RefuteSuite::None`; lines
  327-328 then compute `refutation_passed = result.refutations.is_empty() || …` → reports
  `refutation_passed=True` for checks that never ran. **Fix:** run the refuters, or error, or
  report `refutation_ran: false` explicitly.
- Lines 344, 351: posterior encode/probability errors are `.ok()`-swallowed to `None`,
  indistinguishable from "not requested". Raise instead.
- Every binding runs under `ExecutionContext::for_tests` (line 297 et al.): serial, scalar-only
  kernels, cache disabled — causal-kernels and parallel discovery are unreachable from Python.
  **Fix:** construct a real `ExecutionContext` and expose a `threads=` kwarg.
- Results are lossy scalars: per-unit ITEs, interventional draws, and the oriented CPDAG/PAG are
  computed then discarded — return them.
- `python/causal/` is missing `py.typed` and `.pyi` stubs (DESIGN.md:2337 claims py.typed);
  `identification.py`/`query.py`/`validation.py` are empty `__all__ = []` placeholders; a stale
  in-tree `_native.so` makes `import causal` fail on a fresh checkout — remove from tree.
- `load_float64_columns` (lib.rs:186-199) loads, counts bytes, and drops the data; exists only to
  satisfy the copy-gate test. Make it feed the real ingestion path or fold the gate into it.
- All errors collapse to `ValueError` — map error categories to distinct exception types.

### P2.6 `causal-design` ranker objectives are fabricated
`crates/causal-design/src/ranker.rs:409,459,507-511`
"Expected information gain" = `0.35 + 0.15 * rng.next_f64()`; identifiability decided by
`graph_keys[i] % 2 == 0` (parity of an opaque key); hardcoded SE reductions `se0 * 0.15` / `* 0.1`.
Exported as `causal::rank_designs` "(DESIGN.md §19)". The Monte-Carlo/CRN scaffolding around these
is real; the payoffs it averages are placeholders.
**Fix:** implement the real objectives (EIG via posterior simulation; identifiability via the
actual identifier on the candidate graph; SE reduction via simulated design analysis). Do not ship silent placeholder numbers.

### P2.7 Bayesian validators are permanent no-ops in the suite
`crates/causal-validate/src/suite.rs:316-332`
`PriorPredictive | PosteriorPredictive | PriorSensitivity` always return `NotApplicable` from
`run_one`, yet `ValidationSuite::bayesian_diagnostics()` constructs a suite of exactly those three —
a public constructor guaranteed to do nothing. Real implementations exist in `bayesian_checks.rs`
but are never wired in. Also `PriorSensitivity::to_report` (`bayesian_checks.rs:252-262`) passes
whenever the range is finite (can only fail on NaN).
**Fix:** wire the `bayesian_checks.rs` implementations into `run_one`; give PriorSensitivity a real
threshold on the sensitivity range.

### P2.8 Dead-end facade builder APIs
`crates/causal/src/analysis.rs:179-197` vs `compile()` at 391-453: `discover_jpcmci_plus`,
`discover_rpcmci`, and `GraphInput::Pag`/`TemporalPag`/`DiscoverLpcmci` always hit
`Err(Unsupported)` in `compile()`. `reject_dag_only_on_pag` (`planner.rs:299`) is never called
outside a test.
**Fix:** wire them through (the underlying crates exist) or remove the builder methods until the
paths are backed; a builder method that can never succeed is worse than absence.

### P2.9 Fixed seeds / test contexts in production paths
`crates/causal-model/src/evaluate.rs:200,229`, `crates/causal-attribution/src/distribution_change.rs:273`,
`crates/causal-attribution/src/feature_relevance.rs:39`, `crates/causal-attribution/src/anomaly.rs:167-169`
use hardcoded seeds and/or `ExecutionContext::for_tests(1)` internally, defeating the library's
own no-global-RNG discipline and hiding Monte-Carlo variability from callers.
**Fix:** thread the caller's `ExecutionContext`/RNG through these paths.

### P2.10 Mislabeled statistics
- `KnnCmi` (`crates/causal-stats/src/ci/advanced.rs:215-222`) is not CMI/KSG: negative mean k-th NN
  distance in the joint space; no marginal counts, no digamma, no conditioning correction; `df: n`
  is meaningless. Valid only as a generic permutation dependence statistic. **Fix:** implement
  KSG/FP-style CMI (Runge 2018 CMIknn) or rename honestly.
- `classifier_two_sample` (`crates/causal-stats/src/divergence.rs:57-59`) is a one-line alias of
  `mean_diff_two_sample`; `mechanism_change.rs:82-83` stamps `"classifier_two_sample"` into result
  artifacts — consumers see a classifier test that is blind to variance/shape changes. **Fix:**
  implement a real C2ST or rename the method id.
- "Mixed regression CI" (`crates/causal-stats/src/ci/gsquared.rs:243-266`) is a trivial wrapper
  around partial correlation; `weighted_parcorr` in the factory binds a unit-weight variant
  (`crates/causal-stats/src/ci/factory.rs:48`). Implement or relabel.

### P2.11 CI tests ignore the requested significance/confidence methods
`crates/causal-stats/src/ci/gsquared.rs:44-63`, `advanced.rs:110,361,481`
`GSquared`, `KnnCmi`, `SymbolicCmi`, `MixedKnnCmi`, `Gpdc` never read
`request.significance`/`confidence`; permutation counts are hardcoded at 49 and CI level at 0.95. A
caller requesting `BlockShuffle` silently gets analytic χ².
**Fix:** honor the request or return an explicit unsupported-method error; hoist permutation count
and level from the request.

### P2.12 Silent-fallback sweep (each small; fix opportunistically, loudest first)
- `crates/causal-data/src/transforms.rs`: `equal_width_bin` maps NaN → bin 0; `ordinal_patterns`
  treats NaN as tied (`unwrap_or(Equal)`). The file carries
  `#![allow(clippy::all, clippy::pedantic, clippy::restriction)]` — remove the blanket allow and
  handle NaN explicitly (error or dedicated missing bin).
- `crates/causal-data/src/resample.rs:83-98`: block bootstrap clamps indices to n−1 when
  `block_size > n`, padding replicates with the last row; reachable via
  `BlockBootstrapStability::new()` default `block_size: 20`
  (`crates/causal-validate/src/stability.rs:971`) on short series. Error when `block_size >= n`.
- `crates/causal-attribution/src/anomaly.rs:81`: noise-inference error discarded →
  `residual_abs = 0.0` for every unit (indistinguishable from perfect fit). Propagate.
- `crates/causal-attribution/src/path.rs:48-51,100-102`: non-linear edges get strength
  `unwrap_or(0.0)` → all-zero path shares with no diagnostic; also uses |β| path products, which
  cannot represent cancelling paths, and `total_change` sums absolute shares. Use signed products;
  error on unsupported mechanisms.
- `crates/causal-attribution/src/robust.rs:193-197`: all-zero prediction column triggers a silent
  substitution of the last node's predictions (a legitimate all-zero prediction is clobbered).
  Replace the sentinel with an explicit flag. Document that the payoff equals the interventional
  mean only for linear models.
- `crates/causal-model/src/registry.rs:133-137`: per-family fit errors swallowed during model
  selection — record which candidates failed and why.
- `crates/causal-model/src/do_sampler.rs:235-241`: KDE bandwidth recovered by parsing a note
  string; parse failure → silent 1.0 feeding MCMC. Store bandwidth in a typed field.
- `crates/causal-counterfactual/src/engine.rs:104-115`: `abduct(allow_missing=true)` zero-fills a
  whole missing column and treats the zeros as observed when inferring children's noise, flagged
  only by a global `AssumedNoise` kind (doc claims per-cell granularity). Track missingness
  per-cell or restrict the flag's scope honestly.
- `crates/causal-io/src/wire.rs:268-274`: version parse failure → `0.0.0` written into durable
  artifacts. Error instead.
- `crates/causal-io/src/graph_json.rs:575-617`: encoder silently drops names on length mismatch
  while the decoder rejects the same mismatch. Make the encoder error.
- `crates/causal-io/src/container.rs:891-903`: up-to-4-GiB allocations from untrusted length
  prefixes before plausibility checks — memory-DoS in an interchange format. Validate lengths
  against remaining input size before allocating.
- `crates/causal-estimate/src/overlap.rs:124-127`: missing weights → ESS reported as n. Report
  `None`.
- `crates/causal-discovery/src/evidence.rs:82`: `let _ = graph.insert_directed(...)` drops
  cycle-forming links from `evidence.graph` while they remain in `evidence.links`, though the doc
  says the two are aligned. Similar swallows at `jpcmci_plus.rs:288`,
  `crates/causal-graph/src/projection.rs:53,66`. Record dropped links in diagnostics.
- `crates/causal-discovery/src/engine.rs:272`: public `mci_test` discards the conditioning-set
  truncation count that the batch path reports as a diagnostic. Return it.
- `crates/causal-model/src/evaluate.rs:78,118-147`: `held_out_loglik` is computed in-sample — no
  split exists. Implement a real holdout or rename.
- `crates/causal-attribution/src/anomaly.rs:148-193`: `intrinsic_influence` is a population
  do-contrast with a hardcoded seed, not intrinsic (noise-based) influence. Rename or implement
  (see P4.8).

---

## P3 — Conformance/test strengthening

Do this before (or alongside) the P4 parity work — the current fixtures are why every P0/P1 bug
shipped green.

### P3.1 Strengthen DoWhy conformance fixtures
`conformance/dowhy/linear_gaussian_ate` is real (pinned DoWhy 0.14, black-box estimate) but the SCM
is noiseless, so any consistent estimator matches to 1e-14 — it proves plumbing, not numerics.
**Add:** noisy SCM fixtures recording DoWhy's point estimates **and SEs** for linear regression,
IPW (ATE/ATT, with clipping), AIPW, 2SLS, and frontdoor; assert against `val`/`se` with tolerances.

### P3.2 Strengthen Tigramite conformance fixtures
`tests/tigramite_pcmci_lag1.rs` is real (tigramite 5.2.1.30) but trivial (2 vars, one lag-1 edge)
and compares edge sets only. `tests/tigramite_pcmci_plus_lag0.rs` is self-referential clean-room
with a subset (not equality) assertion — it passes under substantial over-connection.
**Add:** fixtures with ≥4 variables, contemporaneous + lagged links, comparing `val_matrix` and
`p_matrix` (not just edge sets) and FDR-adjusted p-values; a real tigramite PCMCI+ fixture with
edge-set **equality**; any fixture at all for LPCMCI, J-PCMCI+, RPCMCI, and the CI-test statistics
(GPDC, CMIknn, G²) against tigramite outputs.

### P3.3 Fix the three vacuous tests
- `crates/causal-state/src/suff_stats.rs:241-252` — test data (perfectly correlated, equal
  variance) cannot see the P0.3 bug; use distinct-variance partially correlated columns.
- `crates/causal-graph/src/unfold.rs:421` — `let _ = ...is_d_separated(...)` in
  `unfold_dsep_on_chain`: the test named for d-separation asserts nothing about it.
- `crates/causal-stats/src/ci/calibration.rs:223-226` — `assert!(within_two_se || rate < 0.12)`:
  the escape hatch (2.4× nominal) means the calibration claim is never enforced.

### P3.4 Small tigramite-alignment deltas (decide and document, or align)
- Alpha boundary: engine removes at `p >= alpha` (`engine.rs:218`) / retains at `p < alpha`
  (`evidence.rs:35`); tigramite keeps `p <= alpha`. Measure-zero; align for fixture exactness.
- ParCorr residualization includes an intercept (`crates/causal-kernels/src/parcorr.rs:115-123`)
  while tigramite does plain `lstsq` with no intercept, but df (`n − 2 − |Z|`,
  `ci/parcorr.rs:106`) doesn't count it — statistics differ slightly on non-centered data. Either
  drop the intercept (tigramite parity) or count it in df, and document the choice.
- PC-phase frame built at `max_lag` (`engine.rs:308-309`, T−τ_max samples) vs tigramite's default
  `cut_off='2xtau_max'`; MCI already matches. Align or document.
- Meek R4 is applied in PCMCI+ orientation (`orientation.rs:332-393`); the logic is sound but
  tigramite applies only R1–R3 — harmless extra orientation; document as a deviation.
- `parity/dowhy.toml:9` cites "DESIGN.md §35.9", which doesn't exist (34 sections). Fix the
  reference.

### P3.5 Fuzzing coverage (DESIGN.md:2816-2827 lists 8 areas; 3 targets exist)
Missing targets: artifact/container deserialization (highest value — see the P2.12 4-GiB
allocation), expression parsing, temporal sample requests, Python boundary, Arrow metadata.

---

## P4 — Algorithm parity upgrades (bring implementations up to their published names)

### P4.1 PCMCI: full-family MCI phase
`crates/causal-discovery/src/engine.rs:547-565` only computes MCI statistics for PC-surviving
parents. Runge et al. 2019 / tigramite `run_mci` test **all** N²·τ_max pairs `(X_{t−τ}, Y_t)`
conditioning on `pa(Y_t)` and time-shifted `pa(X_{t−τ})`, with significance/FDR over the full
p-matrix. Consequences today: PC1 false negatives can never be recovered, and the BH family in
`Pcmci::run` (`pcmci.rs:81-85`) is the survivor family despite the doc comment claiming the full
MCI family.
**Fix:** iterate all pairs in the MCI phase; keep the (correct) conditioning-set construction from
`mci_batch_for_target`. Update the BH family accordingly.

### P4.2 PCMCI+ per Runge 2020
`crates/causal-discovery/src/pcmci_plus.rs:82-102` is currently one PC1 pass with `min_lag=0` plus
sepset colliders and Meek. Implement the published structure:
1. Lagged-only PC1 skeleton `B̂⁻(X_t^j)`.
2. Contemporaneous phase testing all pairs (τ = 0…τ_max) with conditioning sets S drawn from
   contemporaneous adjacencies plus `B̂(X_t^j)` and `B̂(X_{t−τ}^i)`.
3. Collider orientation with majority/conservative rules (tigramite default
   `contemp_collider_rule='majority'`, re-testing neighbor subsets) and conflict marks (needs P1.7).
4. Meek R1–R3 restricted to contemporaneous links.
Also fix the direction-asymmetry: X→Y and Y→X at lag 0 are tested as separate links with different
conditioning and whichever survives inserts one undirected edge
(`evidence.rs:114-147` `cpdag_from_scored_links`); tigramite symmetrizes.

### P4.3 LPCMCI: from FCI-lite to LPCMCI
`crates/causal-discovery/src/lpcmci.rs:78-97` runs the PC1+MCI engine plus rules
{collider, R1, R2, R3, disc-path}. After the P1 rule fixes, close the algorithmic gap: o–o
initialization (P1.6), middle marks, weakly-minimal separating sets, interleaved ancestral
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

### P4.6 FDR options
`crates/causal-stats/src/fdr.rs` implements BH only (correctly). Add Benjamini–Yekutieli and
Bonferroni/Holm (DESIGN.md:1061), plus tigramite's `exclude_contemporaneous` family handling in
`get_corrected_pvalues`.

### P4.7 Generalized/PAG identification beyond the empty set
`crates/causal-identify/src/generalized.rs:98-121` tests only `Z = ∅` per MAG completion; any
confounded-but-adjustable completion reports NotIdentified. Implement generalized adjustment-set
search per completion (candidate sets from possible ancestors, m-separation check after P1.1/P1.2),
and document the current limitation loudly in the module docs until then (frontdoor.rs:3-16 is the
model for honest limitation docs). Depends on P1.9 (legal completions). The full ID/IDC algorithm
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
  is correct Budhathoki 2021; add the KL-based target functional (DoWhy's default) after P0.1, and
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

### P4.11 Backdoor `minimal_only` semantics
`crates/causal-identify/src/backdoor.rs:183-186`: the loop breaks after the first size class with a
valid set, returning minimum-cardinality sets only; the subset-filter at 167-175 is dead code
(within one size class no proper subsets exist). Docstring says "inclusion-minimal".
**Fix:** either continue enumerating sizes with the subset filter live (true inclusion-minimal), or
change the doc to "minimum cardinality". Roadmap also lists maximal adjustment sets (P5.3).

### P4.12 Smaller correctness parity items
- Wald IV per-arm variances divide by n, not n−1 (`crates/causal-estimate/src/iv.rs:319-320`).
- Probit Fisher weight in the g-computation delta method is wrong
  (`crates/causal-estimate/src/glm_adjustment.rs:365-378` uses μ′(η) for all families; probit needs
  φ(η)²/(μ(1−μ))). Unreachable today only because `fit_glm` rejects probit
  (`crates/causal-stats/src/glm.rs:114-116`) — and `GlmAdjustmentAte::prepare` accepts probit then
  fails late in `fit`; validate early. Fix the weight when probit IRLS lands (P5.4).
- Placebo refuter offers only random-data replacement (`placebo.rs:16`, `common.rs:191`), matching
  DoWhy's default, but the "permute" mode (preserves the treatment marginal — relevant for binary
  treatments) is missing.
- `nested_hard_counterfactual` (`crates/causal-counterfactual/src/engine.rs:352-368`) concatenates
  outer+inner interventions into one simultaneous world (later duplicates override earlier) — not
  nested-counterfactual semantics; correct only for disjoint hard sets. Rename or implement.
- `residual_likelihood_ratio` p-value (`divergence.rs:80`) is `erfc(sqrt(2·KL)/√2)` — an ad-hoc
  calibration with no sample-size dependence, not an LR test. Derive a real test (e.g. asymptotic
  χ² on 2n·KL) or label it a heuristic score.
- Conjugate known-σ² prior scaling (`crates/causal-prob/src/conjugate.rs:174-186`): the supplied
  prior variance is silently reinterpreted as σ²·V0, contradicting `GaussianCoefficientPrior`'s own
  docs. Pick one convention and align code + docs.
- `sequential_allocate` "interaction" terms (`crates/causal-attribution/src/shapley.rs:258-276`)
  are just the next marginal along the path (plus a dead first loop) — rename or compute real
  interaction residuals.
- Modulo bias: `rng.next_u64() % n` in Fisher–Yates (`shapley.rs:303-308`) and every bootstrap
  index draw. Negligible in practice; fix once in a shared sampling helper (P6.4).
- Two `.unwrap()` calls in library code: `crates/causal-counterfactual/src/engine.rs:271,366` —
  the only non-test unwraps in the workspace; replace with proper errors.

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
   binomial, probit IRLS (unblocks P4.12), robust M-estimation, ridge/lasso (helps P2.3
   separation); **robust covariance §11.3** — HC0–HC3, cluster, multiway, HAC/Newey-West (zero hits
   repo-wide today; SEs are homoskedastic-analytic or bootstrap); shared resampling engine §11.4
   additions — cluster and stationary-block bootstrap, permutation resampling; multiple testing
   beyond BH (P4.6).
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
12. **Python packaging** (DESIGN.md:2321-2338): `py.typed` + stubs (also in P2.5), wheel matrix
    verification, explicit `catch_unwind` at the FFI boundary rather than relying on PyO3's
    PanicException.

---

## P6 — Code quality: DRY / SOLID / idiomatic

### P6.1 Consolidate special functions
Abramowitz–Stegun erf (7.1.26) is hand-rolled with identical coefficients 4×:
`crates/causal-stats/src/glm.rs:37`, `crates/causal-stats/src/divergence.rs:85`,
`crates/causal-prob/src/laplace.rs:457`, `crates/causal-validate/src/common.rs:262`; a fifth family
(normal_ppf, ln_gamma, incomplete beta, gamma P/Q) lives in `crates/causal-stats/src/ci/analytic.rs`
and `gsquared.rs`. Create one special-functions module (natural home: causal-stats or
causal-kernels) and route everything through it.

### P6.2 One dense-solver path
`FaerBackend` (column-pivoted QR behind `DenseLinearAlgebra`, `faer_backend.rs:37` — "not normal
equations — DESIGN.md §11.6") is the sanctioned path, yet:
`crates/causal-estimate/src/util.rs:86` `ols_colmajor` does normal equations + Gauss–Jordan and is
duplicated inline in `conditional.rs:141-157` and `prediction.rs:70-86` (each with a
`let _ = FaerBackend;` fig leaf); `crates/causal-prob/src/linalg.rs` is an independent
Cholesky/SPD stack exported `pub` with no external users (crate ADR says backends shouldn't be
exposed — make it `pub(crate)`); `crates/causal-stats/src/ci/parcorr_variants.rs:468`
`invert_symmetric` is line-for-line `gram.rs:55` `invert_square`. Route estimator OLS through the
backend trait; delete the copies.

### P6.3 One bootstrap engine
`crates/causal-data/src/resample.rs` is the canonical engine; `causal-estimate` (which depends on
causal-data) hand-rolls index draws in `util.rs:70-77` and copy-pastes the
resample-gather-refit loop 8× (weighting, stratification, matching, distance, aipw, iv, frontdoor,
rd — only `adjustment.rs:202` uses the shared `bootstrap_se`);
`crates/causal-validate/src/bootstrap_refute.rs:85-95` hand-rolls again. Every copy re-invites the
P2.4 silent-drop bug. Consolidate onto one helper that also carries the failure accounting from
P2.4 and unbiased index sampling (P6.4).

### P6.4 Shared sampling primitives
Box–Muller exists 5× in prod (`crates/causal-kernels/src/rng.rs:9` canonical;
`laplace.rs:499`, `conjugate.rs:326`, `causal-validate/src/common.rs:374`,
`bayesian_checks.rs:291`) plus 4 test copies; Fisher–Yates 2×; categorical sampling 3×;
sample-sd/mean-var 3×; three rank/quantile-binning implementations. Move to causal-kernels' rng and
a small stats-util module; fix `% n` modulo bias once there.

### P6.5 Graph plumbing dedupe
Five near-identical BFS reachability implementations (`dag.rs:144`, `admg.rs:195`,
`temporal.rs:131` — this one allocating a workspace **per edge insertion**, making bulk
construction O(E·(V+E)); `marked_storage.rs:70`; projection walkers); two Kahn's-algorithm copies;
duplicated moralization (`dsep.rs` vs `msep.rs:126-195` — merge when fixing P1.1); the 2^m
enumeration duplicated between `backdoor.rs:114-186` and `efficient.rs:99-164`.

### P6.6 Replace stringly-typed dispatch with enums
Estimand/estimator ids matched as strings across causal-validate (`suite.rs:172`, `rcc.rs:63`,
`graph_refute.rs:451`, `unobserved_common_cause.rs:588`) and the facade planner
(`&*method == "backdoor.adjustment"`): a typo silently becomes permanent `NotApplicable`. Introduce
a closed enum (with an `Other(String)` escape) so the applicability matrix is compile-checked. Same
disease elsewhere: KDE bandwidth in a note string (P2.12), and `Assumption` wire-encoded via
`Debug` formatting into durable artifacts (`crates/causal-io/src/trace.rs:737`) — give it a stable
serialization.

### P6.7 VariableId ↔ dense-index assumption
`crates/causal-identify/src/backdoor.rs:304-310` assumes `VariableId.raw() == dense id` — correct
only for `Dag::with_variables` graphs; otherwise identification silently targets wrong nodes. Same
in `generalized.rs:67-68`; `temporal_backdoor.rs:176-177` launders dense ids through
`VariableId::from_raw`. Thread a proper id↔index mapping (the workspace types already exist).

### P6.8 Untangle the `include!`-assembled propensity module
`crates/causal-estimate/src/propensity/mod.rs:43-47` `include!`s files into one ~1400-line
effective module; file names no longer match contents (weighting.rs holds stratification math and
vice versa). Convert to real `mod` items with correct visibility and re-sort code to match file
names.

### P6.9 API hygiene
- Boolean parameters → enums/options: `discover_pcmci(max_lag, alpha, fdr: bool, accept: bool)`
  and siblings (`crates/causal/src/analysis.rs:154-197`), `abduct(data, allow_missing: bool)`,
  `partial_correlation_batch(..., portable: bool)`.
- Dead fields/params: `crates/causal/src/review.rs:85,183` (`series_len` unused);
  `planner.rs:341` (`let _ = input.graph;` — logical compile never validates query variables
  against the DAG, so errors surface late at run time via `identify_static`; validate at compile);
  `crates/causal-model/src/overlay.rs:30` (`active` never read);
  `KnnCmiWorkspace.distances` never used while the hot loop allocates fresh vecs
  (`crates/causal-stats/src/ci/types.rs:31`);
  `IdentificationWorkspace { _private: () }` threaded through a trait whose impls also all ignore
  `assumptions` (`crates/causal-identify/src/identifier.rs:19-22`);
  `PreparedCiTest` never used — the DESIGN §12 prepare-once contract is currently a shape check
  (`crates/causal-discovery/src/engine.rs:324-331`).
- Visibility: test-only raw-pointer exports `series_columnar_ptr`/`columnar_ptr` are `pub`
  (`crates/causal-data/src/multi_env_plan.rs:142`); `causal-prob::linalg` (P6.2).
- Error-type consistency: causal-validate alone uses thiserror; the facade stringifies typed errors
  into `AnalysisError::Compile` despite transparent variants existing
  (`crates/causal/src/gcm.rs:247-259`).

### P6.10 Trivial clippy
3× doc-list-indentation in causal-estimate; 2× `push_str(" ")` → `push(' ')` in
`crates/causal-io/src/graph_dot.rs:129,148`.

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

---

## Verified correct (for confidence — no action needed)

PC1 condition selection (faithful to tigramite `run_pc_stable`, `max_combinations=1`); MCI
conditioning sets and 2τ_max frame; lag alignment in `lagged_frame.rs`; ParCorr statistic/p-value
(df, t-transform, Fisher-z CI); Benjamini–Hochberg; Meek R1–R3 and the collider rule; DAG
d-separation; PAG definite-status classification; FCI R3; marked-edge endpoint bookkeeping; AIPW
influence function and SE; 2SLS (full first-stage instrument set, structural residuals for SEs —
with regression tests pinning both); IPW/ATT/ATC weights with Hajek normalization and linearized
SEs; GLM IRLS (logit/Poisson) and deviances; GLM g-computation with delta-method SEs; sharp RD
local-linear; frontdoor product-of-coefficients with Sobel SE; Wald IV; E-value formula; backdoor
criterion; Henckel O-set; frontdoor and IV graphical criteria; refuter replicate p-values; exact
and permutation-MC Shapley; coalition cache; SCM ancestral sampling and do-mutilation;
abduction–action–prediction (exact for additive mechanisms, refuses non-invertible);
distribution-change attribution skeleton (Budhathoki 2021); NIG/known-σ² posterior moments; Laplace
approximation (all four links, damped Newton, curvature diagnostics); container format (CBOR
manifest, Arrow IPC sections, BLAKE3, version fields); workspace layering (all dependency edges
point downward); `#![forbid(unsafe_code)]` in semantic crates; no-panic discipline (two unwraps
noted in P4.12); pinned DoWhy/Tigramite provenance.
