# Strict Rust follow-through

Owner: integration reviewer, with disjoint source review by the three reviewers.
The full workspace library Clippy command uses `--locked --workspace --lib -- -D warnings`.
Findings are fixed at their owning layer rather than suppressed globally. Executed
results, including follow-up tests, are recorded in [validation](validation.md).

| Owning paths | Change and semantic boundary |
| --- | --- |
| `core/{identity,physical_choice,route_memory}.rs` | Merge identical identity-axis arms, use checked arithmetic directly, represent implemented physical capabilities as a closed bit set, preserve the previous four-boolean hash inputs/order, replace a fully populated Option array and expect with a fixed Result array. Public capability methods remain const and retain their signatures. |
| `data/file_source/ipc_scan.rs` | Iterate actual Flatbuffer containers; retain checked bounds and overflow refusal. |
| `state/{disk_retention,weighted_refresh}.rs` | Mark generation-sensitive removal result must-use; borrow moment coordinates in identical row-major arithmetic order. |
| `contract/bridge/{claim,propagation,refusal}.rs` | Separate source-point validation from claim propagation; preserve licensing-first validation, formulas, refusal details and provenance. Add must-use to existing pure refusal constructors. See [contract review](contract_lint.md). |
| `identify/{generalized,path_specific,sid/conditional_witness,temporal_pag_path_specific}.rs` | Separate projection/ranking, district contrasts, exact witness certification and incomplete-window publication. Enumeration, cancellation, expression construction and validation order stay unchanged. |
| `model/{history_operators,observation_uncertainty}.rs` | Use field auto-deref and direct mutable iteration; preserve error priority and gamma draw/summation order. |
| `attribution/latent_blocks.rs` | Borrow the matching player/group slot together instead of carrying an independent vector index; retain overlap and uncovered-player refusals. Refit branch preserves baseline-before-comparison order. |
| `io-core/{analysis_result_artifact/association,temporal_design_query_wire,contract_section/verify}.rs` | Separate term validation and assignment conversion at the original check positions; preserve wire fields, row order and typed failures. |
| `io-core/{bridge_distribution,observation_query_artifact}.rs` | Preserve public typed-error and versioned wire signatures with narrowly scoped compatibility dispositions; private mapping helpers box errors, digest formatting borrows. See [IO-core review](io_core_lint.md). |
| `estimate/joint_design/` | Explicit imports, separate likelihood preflight, exact category/design comparisons retained. See [joint design review](joint_design_lint.md). |
| `estimate/{association,joint_coefficients,propensity/matching,streamed_iv,temporal_adjustment,synthetic_rollout}.rs` | Separate meaningful validation/publication and design-assembly responsibilities; preserve resampling streams, rank/tie order, admission and numerical order. Conformal required periods remain an integer-valued finite number without a saturating usize cast. |
| `estimate/response/frontdoor_derivative.rs` | Separate uncertainty realization from point preparation and publication; retain public signature, exact streams, bootstrap drop policy and Bayesian/frequentist publication differences. Check narrowing conversions instead of truncating. |
| `estimate/joint_panel/{decomposition,likelihood,inference,time_varying,selection_uncertainty,streaming}.rs` | Keep moment arithmetic ordered. Replace nested Option with explicit independent/correlated spatial-prior states, retaining a separate factorization failure; both callers preserve draw order. Name the private streaming tile-consumer role and selection groups without changing public signatures or serialization. |
| `estimate/{effect_estimate,iv_bounds,streamed_adjustment,streamed_aipw,temporal_slope_design,temporal_response/bayes,temporal_design/regime}.rs` | Move declarations before statements, check RowId narrowing, preserve SE branch arithmetic, reuse vector capacity, share the coefficient-draw type internally, borrow validated regime weights. An impossible short internal weight axis now refuses rather than silently truncating or indexing past its end. |

These are implementation checks, not new calibration or performance evidence.
Cold private bridge errors gain one box allocation; successful draw processing
and structured public refusal payloads are unchanged. The serde zero predicate
keeps its required borrowed argument with an item-specific reason. Public
front-door arguments and structured bridge errors remain compatibility contracts;
no blanket lint allowance or invented public abstraction is used to hide them.

The methods reviewer independently inspected integration's panel, streaming,
RowId, receipt and temporal changes against their surrounding callers. No material
issue was found: valid-input arithmetic/RNG order and public serialization stay
unchanged, spatial factorization failure remains distinct, and trailing regime
weights are still ignored. This is source review; executed checks are separate.

A final caller-contract trace found that the serial-dependence memo ignored
disabled/bounded caching and finite admission. This is fixed at the production
Bayesian fit boundary by an explicit reuse enum, bypassing lookup and insertion
for restricted contexts. Memory limits derive from that same admission pool;
there is no separate finite limit hidden behind an unlimited pool. Opt-in
unbounded reuse retains the bounded numerical memo. Production-fit tests verify
policy, warm-entry bypass, source/output retention and exact numerical equality.

Binding follow-through preserves the exact private four-part study reply through
a named alias, boxes cold private transport errors while retaining their typed
structured conversion, and uses ordinary Result/Option borrowing/clone helpers.
The evidence-plan family property retains its required PyO3 instance getter
shape with a getter-scoped unused-self explanation. Native row ingestion keeps
its detached ownership/copy counters and slice order; no NumPy borrow escapes.
