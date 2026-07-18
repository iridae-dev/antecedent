# Remaining Work

Items are marked DONE with notes until independently verified and removed. Do not remove an item that you have just finished fixing. DESIGN.md is the roadmap; these are its unimplemented chapters, listed so nothing is lost.

Parity inventories use `pending` / `in_progress` / `done` only. Unfinished DESIGN chapters stay `pending` here and in `parity/*.toml` until shipped.

Ordered by remaining difficulty/scope (simplest → hardest).

1. **Thread `ExecutionContext.kernel_policy` through stats/CI call sites**: G², distance correlation, weighted ParCorr, `standardize_columns`, and similar still hardcode `KernelPolicy::default_policy()`, so `force_scalar` / disallow-portable via context cannot reach them. Small plumbing pass; keeps §23.2 differential-test escape hatch honest.

2. **Posterior draw reduction kernels** (DESIGN.md §23.2 deferred): Shared scalar+portable reductions over posterior draw batches when Bayesian estimation / PPC paths need them. Soft-coupled to mechanism / Bayes work in item 7; skip until a concrete caller exists.

3. **Distribution & path-specific pipelines** (DESIGN.md §8): Identify + estimate for shipped `CausalQuery::Distribution` / `PathSpecific` (types, Unsupported plumbing, GCM sampling / path-contribution wrappers, and minimal query wire are done). Interventional-distribution identification is **IDC** — implement only with/after deep identification (**item 6**); do not fork a second `AutoIdentifier`. Path-specific *natural effects* need path-restricted ID (same ID/IDC family); nonparametric path-specific remains `pending` on `context.mediation.nonparametric`. Path *contribution* attribution already exists (`path_decompose` / `attribute_path_specific`). Full `CausalQuery` CBOR model-bundle embedding ships with `causal-io` model bundles / `query_wire`. Medium once IDC exists.

4. **Bayesian graph discovery** (DESIGN.md §13.7): `GraphPosteriorEngine`, MCMC / enumeration / DBN structure search. Requires the documented `causal-discovery → causal-prob` dependency (absent today). High mixing/convergence and test burden; tiny-DAG enumeration is a medium-risk MVP. Softly helped by PC screening (item 5) and Bayes-factor / ESS work in item 7. Graph-weighted effect envelopes over supplied `WeightedGraphSamples` already ship.

5. **Static (non-temporal) discovery** (DESIGN.md §13.3–13.6): PC, FCI, RFCI, GES, LiNGAM, score-based search / NOTEARS. `causal-discovery` is temporal-only today (PCMCI family; LPCMCI is the PAG surface). Meek-rule and CI-test infrastructure already exists and is verified — PC is the natural first target. Full list is high surface; PC-only MVP is medium risk because CI/Meek reuse.

6. **Deep identification** (DESIGN.md §10.1–10.3): ID algorithm for semi-Markovian models, IDC, hedge certificates, `AutoIdentifier`, memoized recursion; maximal adjustment sets; missing `IdentificationStatus` variants (`IdentifiedUnderParametricRestrictions`, `IdentifiedUnderPriorRestrictions` — `crates/causal-core/src/identification.rs`). Backdoor / frontdoor / IV / RD and generalized adjustment shipped; not full ID/IDC. Highest scientific-correctness risk; status-variant plumbing alone is a smaller first slice. Softly helped by expr simplification (shipped in `causal-expr`).

7. **Mechanism families and Bayesian gaps** (DESIGN.md §14.4, §16, §18.4, §12): Largest multi-backend bundle.
    - Mechanism families: BVAR, state-space, GP, hierarchical (only conjugate Gaussian + Laplace GLM exist today).
    - Counterfactual trajectories; simulation-based calibration (SBC).
    - ESS / R-hat diagnostics (`crates/causal-prob/src/diagnostics.rs` explicitly defers).
    - Bayes-factor CI and posterior dependence probability.
    ESS/SBC wait on HMC/SMC backends; overlaps optional feature flags in DESIGN §30. External Stan/PyMC adapters are **not** required for completion (native Laplace is canonical; see DESIGN §14.5).
