# Backlog

Actionable implementation items. Prefer small PRs with conformance / Python tests.
Parity inventories (`parity/*.toml`) stay the capability contract; this file tracks
**composition and Python-facade gaps** that inventories mark done in Rust but that
are thin or missing on the polished `causal.analyze` path.

Legend: `[ ]` open · `[x]` done (check when merged + gated).

---

## P4 — External prior bank (cross-design transfer)

**Goal.** Given a catalog of previously fit posteriors (tagged by caller-chosen
context) and a new analysis, answer: which posteriors are usable as priors, how
to map them when designs differ, how much to trust each (α / weights), and how
to record transport assumptions — without pretending priors create
nonparametric ID.

**Motivating use case (not the API shape).** Fielded surveys / studies tagged by
product and context, then transferring into a new survey with a related but
non-identical design. Same machinery applies to experiments, observational
batches, manufacturing runs, etc. — the library speaks in *sources*, *targets*,
and *designs*, not “studies.”

**Depends on.** Coefficient-subspace sequential Bayes (`prior_from` / `PriorSet`
hydration) and Shared Bayesian UX (prior sensitivity + PPC on the Python path)
— both shipped. Identical-design sequential Bayes is the special case of this
pipeline.

**Out of scope for the library.** Domain similarity models (embeddings, product
rules, business taxonomy), governance of the prior bank, and max-trust policy
defaults beyond a documented `ConflictPolicy`. Callers supply similarity scores
/ tags; the library owns compatibility, mapping, discounting, mixture,
transport assumptions, and diagnostics.

**Design invariants (match existing Bayesian rules).**

- Priors are recorded as `PriorRestriction` assumptions; they never upgrade
`IdentificationStatus`.
- Heterogeneous designs transfer at the **effect-functional** (or explicitly
mapped parameter) level by default — not silent `coef_i → coef_i`.
- Unmapped / incompatible mass goes to a weakly informative baseline (same
spirit as graph-envelope `unidentified_mass`), never silently renormalized.
- Conflict can only shrink external weight (α → 0), never invent identification.

**Target workflow.**

```text
catalog.filter(target) → rank(similarity)
  → map(effect|named params) → power-prior / mixture (α_k, w_k)
  → apply transport policy → PriorSet + assumptions
  → analyze(..., inference=Bayesian(prior_from=…))
  → prior PPC + prior sensitivity gate α / inclusion
```

Add `parity/bayesian.toml` rows as each slice ships (`python_facade = thin|full`). Prefer `conformance/bayesian/prior_bank_*` + `gate_bayesian.sh`.

---

### A. Posterior catalog + compatibility

Metadata wrapper around existing `causal_posterior` artifacts — not a new draw
format.

**Rust**

- [ ] `**PriorSourceMeta` (causal-prob or causal-io)** — Fields: `artifact_id`,
  estimand fingerprint (query kind + treatment/outcome ids or names), optional
  caller tags (`tags: Arc<[Arc<str>]>` or structured `BTreeMap` — product /
  context / population are caller conventions, not library enums), design
  schema summary (variable names + roles), identification status at fit time,
  contrast coding if any, optional free-form `provenance` map. CBOR section or
  sidecar manifest; draws stay in existing `posterior.draws`.
- [ ] **Enrich posterior schema labels** — Require durable quantity names on
  effect columns (`Effect { name }`) and optional semantic names on coefficients
  (`coef_treatment`, not only `coef_0`) when encoding artifacts used in the bank.
  Migration: unnamed coefs remain valid for P1-C same-design hydrate only.
- [ ] `**PriorCatalog**` — `Vec<PriorSourceRef>` (meta + bytes or path).
  Methods: `filter_compatible(&TargetDesign) -> Vec<CompatibilityReport>`,
  `rank(scores: &[(id, f64)])`. Compatibility checks: estimand match or
  declared map exists; required variables present; reject if old fit was
  unidentified unless caller opts into `AllowUnidentifiedAsPrior`.
- [ ] `**CompatibilityReport**` — `Compatible | Partial { missing, mappable } |
  Rejected { reason }`; never panics on mismatch.

**Python**

- [ ] `**PriorSource` / `PriorCatalog**` — Construct from artifact bytes +
  meta dict; `catalog.compatible_with(query=..., variables=..., tags=...)`.
- [ ] **Smoke: catalog filter** — Three synthetic artifacts (matching estimand,
  wrong estimand, unnamed-coef-only); assert accept / reject / partial reasons.

**Conformance / tests**

- [ ] `**conformance/bayesian/prior_bank_catalog**` — Fixed meta fixtures;
  expected JSON lists accepted ids + rejection reasons.
- [ ] **Rust unit** — CBOR round-trip of `PriorSourceMeta`; filter table-driven.

---

### B. Effect-level / mapped priors (heterogeneous designs)

Coefficient-subspace hydrate (P1-C) stays for identical designs. Cross-design
transfer needs an effect-functional bridge (surveys are the usual example).

**Rust**

- [ ] `**PriorMapping**` — Enum: `IdenticalCoefficientSubspace` (P1-C),
  `EffectFunctional { source_quantity, target: EffectPrior }`,
  `NamedParameters(Vec<(source_name, target_name)>)`. Validate lengths /
  unknown names with clear `ProbError` / `AnalysisError`.
- [ ] `**EffectPrior**` — Gaussian (or mixture-of-Gaussians) prior on a scalar
  or low-dim functional: mean/sd from source effect draws (or stored summary).
  Helper: `EffectPrior::from_effect_draws(draws, quantity_idx)`.
- [ ] `**hydrate_prior(mapping, artifact, baseline: PriorSet) -> PriorSet**` —
  Builds coefficient prior only for mapped dims; unmapped dims keep baseline
  weakly informative entries. Record each mapped source as a
  `PriorAssumption` (`id`: `external_effect_prior` / `external_named_prior`).
- [ ] **Hard error paths** — Dimension mismatch under
  `IdenticalCoefficientSubspace`; missing effect column under `EffectFunctional`;
  unknown name under `NamedParameters`.

**Python**

- [ ] `**PriorMapping` + `Bayesian(prior_from=..., mapping=...)**` — Default
  mapping for banked sources: `EffectFunctional` when estimands match; require
  explicit mapping otherwise.
- [ ] **Smoke: effect-prior transfer** — Source A (Z confounder) → artifact;
  target B (same T/Y, extra covariate). With `EffectFunctional`, new posterior
  mean moves toward A’s effect vs weakly informative baseline; with identical-
  subspace mapping, raise.

**Conformance / tests**

- [ ] `**conformance/bayesian/prior_bank_effect_map**` — Known ATE from source A;
  target B recovers shrunk posterior mean within tolerance; assumption ids
  present in result.
- [ ] **Rust unit** — `from_effect_draws` moments; mapping validation errors.

---

### C. Power-prior / mixture weighting + conflict → α

**Rust**

- [ ] `**ExternalPriorWeight**` — Per source: `alpha ∈ [0, 1]` (power-prior
  exponent on old likelihood / precision scaling on Gaussian approx) and
  optional `mixture_weight w_k` with `∑ w_k ≤ 1`; leftover mass on baseline
  `PriorSet::weakly_informative`.
- [ ] `**compose_external_priors(sources, weights, baseline) -> ComposedPrior**` —
  Gaussian approx path first (match conjugate / Laplace): precision-add under
  power prior `Λ ← Λ₀ + α Λ_old`, mean accordingly; mixture path reuses envelope
  spirit (draw mixture or moment-match). Return `PriorSet` + structured
  assumption payload (`sources`, `alphas`, `weights`).
- [ ] `**ConflictPolicy**` — Inputs: prior-PPC p-value and/or Gaussian KL /
  residual LR between prior predictive and new data (reuse
  `causal-stats` divergence + `PriorPredictiveCheck`). Output: multiply α by
  `shrink(conflict)` (e.g. α' = α · 1{p > p_min} · exp(−β · kl)); never
  increase α. Attach conflict summary beside prior sensitivity.
- [ ] **Wire into `BayesianConfig`** — `prior_from: Option<ComposedPrior |
  PriorSet | artifact>`; facade` analyze` records assumptions + conflict
  diagnostics on validation/diagnostics (not only silent scale).

**Python**

- [ ] `**compose_external_priors([...], weights=..., baseline=..., conflict=...)**`
  — Public helper returning object usable as `Bayesian(prior_from=...)`.
- [ ] `**ConflictPolicy(p_min=..., kl_scale=...)**` — Documented defaults;
  surface applied α' on `result.diagnostics` or validation report.
- [ ] **Smoke: weight + conflict** — Two sources, caller weights `(0.7, 0.3)`;
  conflicting source (mean far from new DGP) gets α shrunk under policy;
  non-conflicting retains weight; assumption record lists both.

**Conformance / tests**

- [ ] `**conformance/bayesian/prior_bank_power_mixture**` — Analytic Gaussian
  toy: composed prior precision = baseline + α·old; expected mean/sd in JSON.
- [ ] `**conformance/bayesian/prior_bank_conflict_shrink**` — Conflict fixture
  forces α' < α; no-conflict fixture leaves α unchanged (tolerance).
- [ ] **Rust unit** — ∑w > 1 rejected; α outside [0,1] rejected; leftover
  baseline mass preserved.

---

### D. Transport / population shift

**Rust**

- [ ] `**TransportPolicy**` — Explicit invariance claims, e.g.
  `InvariantConditionalOutcome` (`P(Y|do(T),X)`), `InvariantEffectModifiers`,
  `InvariantPropensity` — stored as assumptions, never inferred silently.
- [ ] **Prior build under transport** — When target population differs, reweight
  or adjust source likelihood / effect summary toward
  `TargetPopulation` / custom distribution weights (reuse population registry
  patterns from propensity path). If required adjustment is unidentified or
  data for transport are missing → reject source or force α = 0 with reason.
- [ ] **Assumption recording** — Each applied transport claim →
  `PriorAssumption` / dedicated assumption variant with policy id + source
  artifact id.

**Python**

- [ ] `**TransportPolicy(...)` on compose / `Bayesian**` — Required when
  catalog meta population ≠ target population (or when caller passes
  `target_population=` that differs).
- [ ] **Smoke: missing transport** — Differing population tags without policy
  → clear error; with policy + synthetic shift, prior composes and assumptions
  list transport id.

**Conformance / tests**

- [ ] `**conformance/bayesian/prior_bank_transport**` — Two populations; with
  policy, composed prior finite and assumptions non-empty; without policy,
  structured error code stable in expected JSON.
- [ ] **Parity note** — `bayes.prior_bank.transport`: document supported
  policies; mark unsupported shifts `not claimed`.

---

### E. Facade composition + docs / gates

**Rust**

- [ ] `**CausalAnalysis` / execute path** — Accept composed external prior;
  run conflict check after binding new data (needs likelihood context);
  attach prior sensitivity grid optionally over α as well as isotropic scale.
- [ ] **Inventory rows** — `bayes.prior_bank.catalog`,
  `.effect_map`, `.power_mixture`, `.conflict`, `.transport` in
  `parity/bayesian.toml`.

**Python**

- [ ] **Example: survey prior bank** — `python/examples/prior_bank_surveys.py`
  (illustrative domain): two fake survey artifacts (product/context tags) →
  catalog → compose with similarity-derived weights → analyze new target →
  print accepted sources, α', prior PPC, effect posterior.
- [ ] **First-class types** — `PriorSource`, `PriorCatalog`, `PriorMapping`,
  `ComposedPrior`, `ConflictPolicy`, `TransportPolicy` in public API (not only
  `_native`).

**Shared gates**

- [ ] **Extend `gate_bayesian.sh`** — Run `prior_bank_*` conformance + Python
  smokes (`test_prior_bank.py`).
- [ ] **Docs** — Short `docs/prior_bank.md`: workflow, invariants, what callers
  must supply (similarity), survey example as one use case; link from
  `docs/README.md` / backlog pointer.

---

### Suggested P4 slice order

1. Catalog meta + compatibility filter (A) — unblocks ranking UX.
2. Effect-functional `PriorMapping` + hydrate (B) — unblocks heterogeneous designs.
3. Power-prior / mixture compose (C, without conflict) — controlled trust knobs.
4. ConflictPolicy → α shrink (C) — data-dependent trust.
5. TransportPolicy (D) — population shift; can ship “reject if mismatched” first.
6. Facade example + gates (E).

---

## P5 — Interactive / online performance

**Goal.** Button-click latency for business questions (“did the campaign move
revenue?”, “what pulse hit defect rate?”, “why this unit?”) without weakening
identification or inventing numbers. Hot paths, workspaces, Arrow CDI, GIL
detach, `CausalState`, and design-ranking adaptive MC already exist; this
section productizes **latency tiers**, **progressive results**, and **session
reuse** so interactive UX is a first-class contract (ADR 0011), not a silent
default change.

**Design invariants**

- Same estimand, ID status, and assumption recording across tiers; only sample
size / backend / validator depth may change — and must be visible on the
result (`MonteCarloBudget`, actual bootstrap replicates, backend id).
- Discovery is evidence and is **not** on the estimate click path (invariant 6).
- Priors / approximations never upgrade nonparametric ID.
- Optimizations must not silently change statistical semantics; gates + benches
required for new hot contracts (`docs/hot_paths.md`).

**UX spine (spreadsheet / dashboard)**

```text
discover once (seconds) → review / accept graph artifact
  → many estimate / validate / attribute clicks on that artifact
  → re-discover only on explicit refresh or regime change
```

One-shot `analyze(..., discovery=…)` stays a script convenience; interactive
products should surface “structure ready” vs “effect ready.”

---

### A. Latency tiers + progressive / cancellable execute

- [ ] `**LatencyMode` / compute budget** — `Interactive | Standard | Report` (or
  explicit `wall_ms` / `bootstrap` / `n_draws` / `validators`) on
  `CausalAnalysis` / Python `analyze`. Map to known-equivalent configs; do not
  change science defaults silently.
  - Interactive: analytic SE or conjugate/Laplace + few draws; `bootstrap=0`;
  refute off or cheap-only (overlap, E-value); no HMC.
  - Standard: current defaults (`bootstrap=50`, `n_draws=1000`, `refute=True`).
  - Report: more replicates / draws / full suite; optional HMC.
- [ ] **Progressive result stages** — Compile once; stream: (1) identification
  fail-fast, (2) point + analytic/Laplace summary, (3) bootstrap / posterior
  fills, (4) refuters / PPC. Same logical plan throughout.
- [ ] **Wire `CancellationToken` + `ProgressSink`** into estimate bootstrap,
  discovery CI loops, Shapley, graph×effect envelopes (hooks exist on
  `ExecutionContext`; estimate path does not poll today). Python: cancel +
  progress callback through `_native`.
- [ ] **Surface effort on results** — Actual replicates, draws, early-stop flag,
  and stage timings on `PerformanceView` / diagnostics (mirror design-ranking
  `MonteCarloBudget` honesty).

**Conformance / tests**

- [ ] **Smoke: interactive vs standard** — Same synthetic ATE; interactive
  returns finite point + ID; standard SE/quantiles within tolerance of full
  run; result records mode / effort.
- [ ] **Cancel mid-bootstrap** — Token cancel yields structured partial or
  cancelled outcome; no panic; no silent full result.

---

### B. Session reuse: prepared analysis + CausalState online path

- [ ] **Durable prepared handle** — Expose compile-once / re-estimate-many
  (fixed schema, graph, query, estimator; swap or append data) in Rust facade +
  Python OO API (`PreparedAnalysis` / `result.refresh(data)`).
- [ ] `**CausalState` as primary online path** — Document + example: append
  batch → invalidate → `refresh_results` under `CacheBudget`; version results
  so UI never mixes stale ID with new estimates (ADR 0016 already matches;
  demos still prefer fresh `analyze()`).
- [ ] **Python session objects** — Hold `_native` analysis / `CausalState` /
  fitted GCM across clicks; avoid clone-model+data-per-method on attribution
  hot paths.
- [ ] **Enable `CachePolicy` on Python production contexts** — Match attribution
  benches; coalition / sample caches for repeated “attribute” clicks.

**Conformance / tests**

- [ ] **Prepared re-estimate match** — Two-shot refresh equals fresh `analyze`
  on same data (tolerance policy); second shot cheaper on bench smoke.
- [ ] **State append OLS** — Already gated; add Python dual for refresh UX.

---

### C. Adaptive Monte Carlo (bootstrap, draws, graph envelopes)

- [ ] **Adaptive bootstrap** — Stop when SE relative change < ε with min/max
  replicates; report actual count (same spirit as design `early_stopped`).
- [ ] **Adaptive Bayesian draws** — Cap by quantile-width / ESS target under
  Laplace path first; HMC stays report-tier.
- [ ] **Graph×effect interactive subsample** — Stratified graph subset +
  renormalized weights for UI; leftover mass stays `unidentified_mass` (never
  silent renormalize to 1). Full mixture for Report.
- [ ] **Shared workspaces across refute + bootstrap** — Refitters reuse warmed
  `EstimationWorkspace` / propensity buffers from the point estimate.

**Conformance / tests**

- [ ] **Adaptive bootstrap pin** — Fixed seed; early-stop replicate count stable;
  SE within tolerance of max-replicate run on toy.
- [ ] **Envelope subsample honesty** — Interactive envelope mean/quantiles +
  `unidentified_mass` match full mixture within documented tolerance *or*
  explicitly flagged as approximate with mass accounting.

---

### D. Discovery off the estimate click path

- [ ] **Artifact-first UX docs + example** — Discover once → serialize /
  hold graph evidence → many `analyze(..., graph=artifact)` clicks; re-discover
  only on explicit refresh. Contrast with one-shot `discovery=` script path.
- [ ] **Cache accepted graph / completion** — Versioned accepted CPDAG/PAG/
  temporal completion; estimate-only clicks never re-run PCMCI/FCI.
- [ ] **Stability / rediscover policy** — Optional scheduled or user-triggered
  rediscovery; never implicit on prior-scale / bootstrap / treatment tweaks.
- [ ] **Python: clear errors** when `discovery=` + interactive profile conflict
  (or auto-split into discover then estimate with a warning).

**Conformance / tests**

- [ ] **Example: spreadsheet discover-then-estimate** — Extend or dual
  `python/examples/` manufacturing / sales path; assert discovery not invoked
  on second estimate.
- [ ] **Parity note** — Document interactive discovery UX in relevant
  `parity/discovery*.toml` / docs pointer.

---

### E. Data plane + result payload hygiene (Python)

- [ ] **Arrow CDI as documented interactive path** — Prefer CDI / contiguous
  float64 over pandas column-at-a-time copy; keep copy-gate honest.
- [ ] **Column projection after ID** — Gather only treatment / outcome /
  adjustment (and needed lags) before kernel work on wide sheets.
- [ ] **No default full posterior artifact to Python** — Summaries for UI;
  artifact bytes on explicit download / sequential-prior hydrate (P4).
- [ ] **Batch multi-query** — One `TableView`, N queries (shared prepared sample
  / lag frame); Python helper or facade batch API.
- [ ] **Refute as second click / background** — Interactive profile: cheap
  validators inline; placebo/bootstrap refute suite opt-in or async.

**Conformance / tests**

- [ ] **Arrow interactive smoke** — CDI path reports zero-copy (or bounded copy)
  and returns estimate; pandas path remains correct but not the latency default
  in docs.
- [ ] **Projection bench** — Wide synthetic table; projected gather matches
  full-column estimate; allocation/copy under budget.

---

### Suggested P5 slice order

1. Latency tiers + wire cancel/progress (A) — unblocks honest interactive UX.
2. Discovery-off-click docs + artifact-first example (D) — product narrative.
3. Prepared analysis / CausalState Python session (B) — online reuse.
4. Adaptive bootstrap + effort reporting (C) — correct early stop.
5. Arrow / projection / no-default-artifact (E) — binding-plane latency.
6. Graph-envelope interactive subsample (C) — graph×effect envelope already shipped.

---

## Docs / gates (keep backlog honest)

- [ ] **Link this backlog from `docs/README.md`** — “Open composition work” pointer.
- [x] **Update conformance notes** — Cleared “Facade/Python wiring deferred” on `bayesian__dag_posterior` (graph×effect facade shipped).
- [ ] **Parity note convention** — When a capability is Rust-done but Python-thin, add `python_facade = thin|full` (or equivalent) in the relevant `parity/*.toml` row.
- [ ] **Example: sales spreadsheet E2E** — Discover → Bayesian ATE → path decompose → ITE; plus temporal pulse Bayesian (`python/examples/`).
- [ ] **Hot-path index** — When P5 ships tiers / progressive execute / prepared
  refresh, add rows to `docs/hot_paths.md` + baseline smokes (cancel, adaptive
  bootstrap, prepared re-estimate).

---

## Suggested implementation order

1. ~~P0 temporal Bayesian~~ / ~~P1 E2E Bayesian~~ — shipped (PPC, sequential
  `prior_from`, graph×effect + DBN envelopes, backend/sensitivity UX).
2. ~~P3 science/product limits~~ — shipped or closed in inventory (PAG ID scope,
  temporal PAG→DAG, RPCMCI OOS, J-PCMCI+ MV space-dummy, population propensity,
  nonparametric path-specific ID, review UX, GCM discover-compose).
3. ~~P2 facade composition~~ — shipped (path/interventional+discovery, panel
  Bayesian, EventFrame discovery, pooled panel PCMCI-family).
4. P4 external prior bank: catalog → effect map → power mixture → conflict → transport.
5. P5 interactive performance (tiers / progressive execute; graph-envelope
  subsample; prior-artifact payload hygiene aligns with P4).

