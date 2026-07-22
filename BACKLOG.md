# Backlog

Interactive / online performance. Prefer small PRs with conformance / Python
tests. Parity inventories (`parity/*.toml`) stay the capability contract; this
file tracks **latency tiers**, **progressive results**, and **session reuse** so
interactive UX is a first-class contract (ADR 0011), not a silent default
change.

Legend: `[ ]` open · `[x]` done (check when merged + gated).

**Goal.** Button-click latency for business questions (“did the campaign move
revenue?”, “what pulse hit defect rate?”, “why this unit?”) without weakening
identification or inventing numbers. Hot paths, workspaces, Arrow CDI, GIL
detach, `CausalState`, and design-ranking adaptive MC already exist.

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

## A. Latency tiers + progressive / cancellable execute

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

## B. Session reuse: prepared analysis + CausalState online path

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

## C. Adaptive Monte Carlo (bootstrap, draws, graph envelopes)

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

## D. Discovery off the estimate click path

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

## E. Data plane + result payload hygiene (Python)

- [ ] **Arrow CDI as documented interactive path** — Prefer CDI / contiguous
  float64 over pandas column-at-a-time copy; keep copy-gate honest.
- [ ] **Column projection after ID** — Gather only treatment / outcome /
  adjustment (and needed lags) before kernel work on wide sheets.
- [ ] **No default full posterior artifact to Python** — Summaries for UI;
  artifact bytes on explicit download / sequential-prior hydrate.
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

## Docs / gates

- [ ] **Parity note convention** — When a capability is Rust-done but Python-thin,
  add `python_facade = thin|full` (or equivalent) in the relevant `parity/*.toml`
  row.
- [ ] **Example: sales spreadsheet E2E** — Discover → Bayesian ATE → path
  decompose → ITE; plus temporal pulse Bayesian (`python/examples/`).
- [ ] **Hot-path index** — When tiers / progressive execute / prepared refresh
  ship, add rows to `docs/hot_paths.md` + baseline smokes (cancel, adaptive
  bootstrap, prepared re-estimate).

---

## Suggested slice order

1. Latency tiers + wire cancel/progress (A) — unblocks honest interactive UX.
2. Discovery-off-click docs + artifact-first example (D) — product narrative.
3. Prepared analysis / CausalState Python session (B) — online reuse.
4. Adaptive bootstrap + effort reporting (C) — correct early stop.
5. Arrow / projection / no-default-artifact (E) — binding-plane latency.
6. Graph-envelope interactive subsample (C).
