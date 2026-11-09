# Thread inventory

Every parallel region in the library runs under the study's `ExecutionContext`:
it leases its workers, and their scratch, from the context's admission pool
before any thread starts. This page lists where threads start and how each site
is bounded.

Two tests hold the rule:

- `crates/antecedent-core/tests/unleased_parallelism.rs`
  (`production_code_starts_no_thread_outside_the_admission_pool`) scans the
  production sources of every workspace crate and of the Python extension for
  thread creation and parallel-iterator calls (`thread::spawn`, `thread::scope`,
  `thread::Builder`, `spawn_scoped`, `rayon::`, `Par::rayon`, `par_iter` and
  related). Any file that is not on its allowlist fails. Each allowlisted file
  carries a reason and a marker the file must keep, so a stale reason fails too.
- `crates/antecedent/tests/workspace_invariants.rs` (`no_direct_rayon_dependency`)
  rejects a direct `rayon` dependency in any workspace `Cargo.toml`. Transitive
  `rayon` comes from faer and the forust-ml learner.

Test code (calibration harnesses, stack-size spawns in tests, race tests) is not
a product pool and is not scanned.

## The study budget

| Entry | Location | Bound |
|---|---|---|
| Default study width | `default_user_threads`, `DEFAULT_USER_THREAD_CAP` in `crates/antecedent-core/src/execution/budget.rs` | Available parallelism clamped to `[1, 16]`. An explicit thread count of at least 1 is a pin and may exceed the cap. |
| Python `threads=` | `resolve_user_threads` in `python/src/context.rs` | Omitted takes the default width; an explicit count is passed through as a pin. |
| `ExecutionContext::production`, `production_default` | `crates/antecedent-core/src/execution/context.rs` | Store the budget on the context; they start no pool. |
| `ExecutionContext::serial_inner` | same | A serial clone for work inside an outer map that already uses the budget. |

## Sites that start threads

These are the allowlisted production files.

| Site | Location | Why it is bounded |
|---|---|---|
| The parallel map (`map_indexed`, `try_map_indexed` and their scratch variants) | `crates/antecedent-core/src/execution/pipeline.rs` | Takes the worker lease before it spawns and holds it until the workers join. Every top-level map leases, including serial execution; a nested map reuses its enclosing worker's reservation and stays serial. Cancellation and admission refusals stop scheduling and return typed errors. |
| Batch resampling | `crates/antecedent-data/src/resample/replicates.rs` | Leases its workers and per-worker scratch with `admit_scratch` before the scope starts, and the scope uses exactly the granted worker count. |
| Python call watcher | `python/src/interrupt.rs` | One worker (`WORKER_STACK_BYTES`, 64 MiB stack) so the calling thread can poll Python signals and release the GIL; the work itself admits itself. |
| Large-stack route and replay threads | `crates/antecedent/src/analysis/flagship_mixed.rs`, `python/src/mixed_decision_api.rs` | One scoped thread that only provides a larger stack; the caller blocks on it, so it adds no parallelism. |

Everything else that runs in parallel goes through `map_indexed`: bootstrap
replicates, cross-fitting folds, batch queries, CI-test batches, PCMCI targets,
exact DAG enumeration, HMC and structure-MCMC chains, posterior functionals and
graph-posterior atoms. Each worker receives the inner context the map hands it.

## Third-party pools

| Library | How it is bounded |
|---|---|
| faer | `FaerScope` (`crates/antecedent-stats/src/faer_scope.rs`) passes each kernel an explicit degree: `Par::Seq` for one worker, or `Par::Rayon` with the caller's exact worker count, never zero. The library never calls `faer::Par::rayon`, `get_global_parallelism` or `set_global_parallelism`, and does not use the convenience factorizations that read the process-global pool (`ColPivQr::new`, `thin_svd`, `self_adjoint_eigenvalues`, `self_adjoint_eigen`). Scratch is admitted against the same lease before it is allocated. |
| forust-ml (gradient-boosted trees) | `crates/antecedent-learn/src/gbt.rs` always calls `set_parallel(false)`: the crate exposes only a global-pool boolean, so the outer folds own the parallelism budget. |
| Burn (neural learner) | Initial weights are drawn on the host from the learner seed and no training step is stochastic, so concurrent fits share no state and need no process-wide lock (`crates/antecedent-learn-burn/src/lib.rs`). |
| Polars | Not linked. Python ingest reads a Polars frame through the Arrow C data interface and never sizes a Polars pool; the decision record is `parity/polars_decision/`. |

## Outside the study budget

- **Python providers.** A provider callable (`python/src/provider_callbacks_api.rs`)
  or a registered provider (`ProviderRegistry.execute` in
  `python/antecedent/extensibility.py`) runs on the caller's Python thread and
  receives no thread count. A provider that starts joblib, a BLAS pool or its own
  threads is outside the study budget.
- **BLAS and OpenMP.** Product code never sets `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` or `n_jobs`, and no OpenBLAS or MKL
  crate is linked; faer's matrix products use the `gemm` crate under faer's
  degree. The documentation notebook kernel (`scripts/render_notebooks.py`) pins
  those variables and `RAYON_NUM_THREADS` to 1 for itself only.
