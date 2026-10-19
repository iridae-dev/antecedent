# Thread inventory (R01)

What exists today. R02 enforces the dispositions. No redesign.

Dispositions: `already_bounded` (this call clamps to the study `ExecutionContext`), `bound_per_operation` (the API can take a per-call width; the call site does not), `startup_only` (process lifetime, not per study), `isolate_or_refuse` (a study thread budget cannot be applied).

Searches: `rayon`, `ThreadPool`, `std::thread`, `available_parallelism`, `faer::Par` / `get_global_parallelism`, `polars`, `burn`, `openmp` / `OMP_NUM_THREADS` / `OPENBLAS` / `MKL_NUM_THREADS`, `joblib`, `n_jobs`, `ExecutionContext`, `map_indexed`. Direct `rayon` in a workspace `Cargo.toml` is rejected (`crates/antecedent/tests/workspace_invariants.rs:12`). Transitive `rayon` is faer 0.24.4 (`features = ["rayon"]`, `Cargo.toml:74`) and forust-ml.

Test-only `std::thread` (calibration harness, stack-size spawns, `dml.rs` cache race, `fit_counts` / `content_identity` / `execution_attempt` tests) is not a product pool.

## Core

| Entry | Location | Parallelizes | Scope | Thread count | Allocation | 3.0 | Evidence |
|---|---|---|---|---|---|---|---|
| Default study width | `crates/antecedent-core/src/execution.rs:48` | Nothing; sets the budget | Per context, not process-global | `available_parallelism`, clamp `[1, 16]` (`DEFAULT_USER_THREAD_CAP`, line 43). Explicit `threads >= 1` may exceed the cap (`python/src/lib.rs:2645`) | None | `already_bounded` | `default_user_threads` clamps; `resolve_user_threads` pins an explicit count per call |
| `ExecutionContext::map_indexed` | `crates/antecedent-core/src/execution.rs:750` | Index-aligned independent work | Per operation | `(max_threads).clamp(1, n)` (line 759). `n <= 1` or serial context stays on the caller. Workers receive `serial_inner` (line 729, 763) | `Vec<Option<Result<…>>>` of length `n` (line 764). No standing pool | `already_bounded` | `std::thread::scope` (line 770); nested `map_indexed` sees `max_threads == 1` |
| `BatchStudy::estimate_many` | `crates/antecedent/src/analysis/batch.rs:568` | Batch queries, chunked | Per operation | `max_threads`, one spawn per query in the chunk (line 573) | One result slot per query. Shared nuisance cache | `already_bounded` | Scope width is `ctx.parallelism`, but workers are given that same `ctx`, not `serial_inner` (line 578). See conflicts |
| `PreparedBatch::estimate` | `crates/antecedent/src/analysis/batch.rs:913` | Prepared plans | Per operation | Same as `estimate_many` (line 915) | Same | `already_bounded` | Same full-`ctx` handoff (line 921) |
| Python GIL watcher | `python/src/interrupt.rs:190` | The native body of one Python call | Per call | Exactly one worker | 64 MiB stack (`WORKER_STACK_BYTES`, line 36) | `already_bounded` | Caller waits; the worker is not sized from `max_threads` |
| Forust GBT | `crates/antecedent-learn/src/gbt.rs:77` and `:127` | Trees / prediction, if enabled | Per fit, flag only | Hardcoded `parallel = false` | Dense col-major copy of the design before fit | `already_bounded` | `set_parallel` is a bool (`forust-ml` `gradientbooster.rs:1158`). `true` uses `rayon` `par_iter` (`gradientbooster.rs:811`), which is the process pool, not a lease. `parallelism_control: true` (line 54) means the flag is forced off |

`production` / `production_default` (`execution.rs:722`, `:814`) store that budget on the context. They do not start a pool.

## faer

faer 0.24.4 `GLOBAL_PARALLELISM` defaults to `2` when `rayon` is on (`faer-0.24.4/src/lib.rs:1112`), which `get_global_parallelism` decodes as `Par::rayon(0)` (line 1140), i.e. `rayon::current_num_threads()` (line 939). Antecedent never calls `set_global_parallelism` or passes `faer::Par` (repo search: no matches).

| Entry | Location | Parallelizes | Scope | Thread count | Allocation | 3.0 | Evidence |
|---|---|---|---|---|---|---|---|
| Process `Par` | faer `lib.rs:1127` (`set_`), `:1140` (`get_`) | Every convenience linalg call | Process-global | Rayon global pool (default: available parallelism) | Rayon pool is lazy and process-wide | `bound_per_operation` | Call sites can take an explicit `Par`; these do not |
| Column-pivoted QR least squares | `crates/antecedent-stats/src/faer_backend.rs:52`, solve `:84` | QR factor and least-squares solve | Process-global (via faer) | `ColPivQr::new_imp` reads `get_global_parallelism` (faer `solvers.rs:1216`); `solve_lstsq_in_place_with_conj` reads it again (`solvers.rs:2522`) | Column scales `Vec`; `Mat::from_fn` copy; RHS `Mat`; faer owns QR, column perms, `Q_coeff`, `MemBuffer` scratch. Residuals reuse `LeastSquaresWorkspace` (line 36, 91) | `bound_per_operation` | No `Par` argument at the call |
| Thin SVD min-norm | `faer_backend.rs:141` | SVD of a possibly rank-deficient design | Process-global | `Svd::new_imp` reads `get_global_parallelism` (faer `solvers.rs:1344`) | Owned `Mat` copy; faer allocates thin `U`, `V`, `S` (`solvers.rs:1347`) | `bound_per_operation` | `a.thin_svd()` takes no `Par` |
| Meat eigenvalues | `crates/antecedent-stats/src/covariance.rs:547` | Symmetric eigenvalues of the multiway meat | Process-global | `self_adjoint_eigenvalues` reads `get_global_parallelism` (faer `solvers.rs:430`) and sizes scratch from that `par` | `Mat::from_fn` copy of the `ncols × ncols` meat | `bound_per_operation` | Convenience method, no `Par` |
| Symmetric eigen | `covariance.rs:577` | Eigenpairs | Process-global | `SelfAdjointEigen::new_imp` reads `get_global_parallelism` (faer `solvers.rs:1444`) | Copied `Mat`; faer zeros `U` and `S`; caller allocates a `d × d` eigenvector buffer (line 582) | `bound_per_operation` | `self_adjoint_eigen` takes no `Par` |

In-tree Cholesky (`antecedent-kernels`) is serial. It is not a pool.

## Polars candidates

Not present. Search: `polars` in `*.toml`, `*.in`, `*.txt` — no matches. `Cargo.toml` has no Polars crate. Python ingest accepts a Polars frame only as an Arrow capsule (`python/antecedent/_data.py`); it does not call `thread_pool_size`.

| Entry | Location | Parallelizes | Scope | Thread count | Allocation | 3.0 | Evidence |
|---|---|---|---|---|---|---|---|
| Polars pool (candidate, not linked) | No in-tree call | Lazy preparation, if adopted | Process-global | Set once for the process. A later study `threads=` does not resize it | Not in tree | `startup_only` | No pool in this repo. Do not add Polars here. A study budget that differs from the process pool cannot be applied by resizing |

## Burn

Not a thread pool. Search in `crates/antecedent-learn-burn`: no `rayon`, `thread::`, or `Par`. Burn 0.16 is `NdArray<f32>` CPU only (`lib.rs:18`).

| Entry | Location | Parallelizes | Scope | Thread count | Allocation | 3.0 | Evidence |
|---|---|---|---|---|---|---|---|
| `TRAIN_LOCK` | `crates/antecedent-learn-burn/src/lib.rs:23`, held at `:167` | Training is serialized, not parallel | Process-global mutex | Ignored. `NeuralNetLearner` sets `parallelism_control: false` (`crates/antecedent-learn/src/neural.rs:42`) | Col-major to row-major `f32` copy, target `Vec<f32>`, Burn tensors; weights copied out so the artifact is `Send` | `isolate_or_refuse` | Comment and lock: Burn's backend RNG is global (`InferBackend::seed` at line 168). Concurrent folds cannot hold distinct seeds without the lock |

## Python providers

No `n_jobs`, `joblib`, or thread-pool construction under `python/antecedent` or `python/src` (search those tokens: no product matches).

| Entry | Location | Parallelizes | Scope | Thread count | Allocation | 3.0 | Evidence |
|---|---|---|---|---|---|---|---|
| Study context for a Python call | `python/src/lib.rs:2645`, context at `:2673` | Delegates to the core budget | Per call | Omitted/`0` → `default_user_threads`; explicit `>= 1` is a pin | Cache cap `PY_DEFAULT_CACHE_MAX_BYTES` (4_000_000, line 2641) | `already_bounded` | `ExecutionContext::production(seed, threads)` |
| `PyProvider::invoke` | `python/src/provider_callbacks_api.rs:682` | The user callable | Caller thread, GIL held | Not passed | JSON context; answers over 64 MiB refused (line 708) | `isolate_or_refuse` | Context keys are `stream_seed`, `attempt`, `idempotency_key`, `inputs` (line 689). No thread count. `Python::attach` (line 699) |
| `ProviderRegistry.execute` | `python/antecedent/extensibility.py:1025` | Whatever the registered object does | The caller’s Python thread | Not passed | Result arrays the provider returns | `isolate_or_refuse` | `execute` forwards `request` only. Host sklearn / EconML `n_jobs` is outside this call |

## BLAS / joblib

Not present in the product. Search `OMP_NUM_THREADS`, `OPENBLAS`, `MKL_NUM_THREADS`, `threadpoolctl`, `n_jobs`, `joblib` in `*.rs` / `*.py`: the only assignments are the notebook kernel below. `joblib` appears as a fixture package name (`crates/antecedent-discovery/tests/jpcmci_plus_oracle_matrix.rs:177`), not a pool. No OpenBLAS/MKL crate in `Cargo.lock`. faer’s matmul is the `gemm` crate, still under faer’s `Par`, not a BLAS library.

| Entry | Location | Parallelizes | Scope | Thread count | Allocation | 3.0 | Evidence |
|---|---|---|---|---|---|---|---|
| Notebook kernel env | `scripts/render_notebooks.py:66` | Docs kernel only | Process env of that kernel | `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `RAYON_NUM_THREADS` all `"1"` | None | `startup_only` | Set in `kernel.json` `env` before the kernel starts. Product runs do not set these |

## Bootstrap

| Entry | Location | Parallelizes | Scope | Thread count | Allocation | 3.0 | Evidence |
|---|---|---|---|---|---|---|---|
| Index batch | `crates/antecedent-data/src/resample.rs:585` | Replicate index fills | Per operation | `max_threads`, skipped when `1` or `n_replicates < 2` (line 592) | Caller-owned `out` (`n * n_replicates` `u32`). Each worker has a length-`n` scratch (line 613). Per-replicate RNG is `stream_for(Resample, stream_base ^ r)` (line 543) | `already_bounded` | `std::thread::scope` (line 603) |
| IID and cluster SE | `crates/antecedent-estimate/src/util.rs:346` and `:470`; spawn `:604` | Replicate refits | Per operation | `(max_threads).clamp(1, n_rep)` (line 401, 615). Adaptive early-stop stays serial (line 402) | One `u32` index matrix, `n * replicates` or `groups * replicates`, `try_reserve_exact` (line 377, 500) before any spawn. Each worker owns scratch `S` and an index `Vec` (line 628) | `already_bounded` | Own `thread::scope`, not `map_indexed`. Fit closure is not given `serial_inner` |
| Smoothed-dose replicates | `crates/antecedent-estimate/src/smoothed_dose.rs:1915` | Bootstrap replicates of the dose fit | Per operation | `clamp(1, bootstrap)` (line 1908) | Memory check is `workspace_bytes * concurrent` against the hard limit (line 1909) before `map_indexed` | `already_bounded` | `map_indexed` passes `inner` (serial) into each replicate |

## Graph search

`crates/antecedent-graph/src`: not present (`std::thread`, `map_indexed`, `rayon`, `parallelism` — no matches). Completion search there is serial.

| Entry | Location | Parallelizes | Scope | Thread count | Allocation | 3.0 | Evidence |
|---|---|---|---|---|---|---|---|
| CI batch | `crates/antecedent-discovery/src/ci.rs:29` | Independent CI queries | Per operation | `max_threads.min(n / 4)`; below 2 queries/thread stays serial (`MIN_QUERIES_PER_THREAD`, line 16, 37) | Fresh `CiWorkspace` per chunk (line 47) | `already_bounded` | `ctx.map_indexed` (line 43); inner context is serial |
| PCMCI parents and MCI | `crates/antecedent-discovery/src/engine.rs:666` and `:731` | One target per task | Per operation | `max_threads.max(1)` (line 570); serial when that is 1 or there is one target (line 638, 710) | Each task clones `prepared_ci` and builds a local `DiscoveryWorkspace` (line 671, 733) | `already_bounded` | `map_indexed` over targets |
| PCMCI+ / J-PCMCI+ | `pcmci_plus.rs:160`, `jpcmci_plus.rs:213` | Same engine maps | Per operation | Same `max_threads` read, passed into `lagged_pc1_parents` / `select_parents_all` | Same as the engine | `already_bounded` | No second spawn; they call the engine |
| Exact DAG enumeration | `crates/antecedent-discovery/src/exact_enumeration.rs:107` | Score of each enumerated DAG | Per operation | `map_indexed` width | Mask list for `n <= 6`; hard memory limit checked before enumerate (line 97) | `already_bounded` | `map_indexed(masks.len(), …, inner)` |

`ChunkSchedule` (`ci.rs:66`) only widens a serial scan. It does not spawn.

## Posterior evaluation

| Entry | Location | Parallelizes | Scope | Thread count | Allocation | 3.0 | Evidence |
|---|---|---|---|---|---|---|---|
| HMC chains | `crates/antecedent-prob/src/hmc.rs:409` | Chains (`n_chains` default 4, line 66) | Per fit | `map_indexed(n_chains)` | One `LaplaceWorkspace` per chain (line 411). Samples concatenated after join (line 420). `ctx == None` reuses one workspace serially (line 416) | `already_bounded` | Pool is `map_indexed`. The closure discards the serial context (`\|chain, _\|`) |
| Structure MCMC chains | `crates/antecedent-discovery/src/graph_mcmc.rs:107` | One mask chain per index | Per operation | `map_indexed(n_chains)` | Merge allocates concatenated traces (`merge_chain_outputs`) | `already_bounded` | Worker receives `inner` (serial). Comment at line 105 |
| Posterior functional | `crates/antecedent-estimate/src/bayesian/evaluator.rs:109` | Non-identity g-comp draws | Per operation | `map_indexed(n_draws)` only when `n_draws * nrows * ncols >= 1 << 16` (line 83, 109) | One `beta: Vec<f64>` per draw (line 111). Serial path reuses `workspace.row` | `already_bounded` | `map_indexed`; closure discards inner (`\|d, _\|`) |
| Atom / graph fits | `crates/antecedent/src/analysis/execute/bayesian_path.rs:1448` (also `:1519`, `:1757`, `:2121`); `class_posterior.rs:70`; `admg_posterior.rs:174`; `class_posterior_response.rs:201`; `admg_posterior_response.rs:259`; `temporal_posterior_response.rs:280`; `temporal_class_posterior.rs:277`; `temporal_class_posterior_response.rs:166`; `temporal_class_mediation_posterior.rs:200`; `bayesian_graph_posterior_atoms.rs:109`; `crates/antecedent/src/analysis/identification_cache/mod.rs:79` | One identified atom or posterior graph per task | Per operation | `map_indexed` over the atom or graph count | Per-task fit state inside the closure | `already_bounded` | All go through `map_indexed`, which serializes the inner context |

## Conflicts

A study budget is `ExecutionContext.parallelism`. These ignore it:

1. **faer convenience calls.** `ColPivQr::new`, `solve_lstsq_in_place_with_conj`, `thin_svd`, `self_adjoint_eigenvalues`, and `self_adjoint_eigen` read `get_global_parallelism()`. `serial_inner` does not change that atomic. Bootstrap workers, cross-fit folds, and batch queries therefore run at the study width and each least-squares or eigen call may also occupy the process rayon pool (default available parallelism, not `threads=` and not the cap of 16).

2. **Batch scopes keep the full context.** `estimate_many` (`batch.rs:578`) and `PreparedBatch::estimate` (`batch.rs:921`) spawn `max_threads` workers with the outer `ctx`. Each worker may enter `map_indexed` or `evaluate_bootstrap_parallel` at that same width. `map_indexed` itself does not do this; it passes `serial_inner`.

3. **Polars candidate.** Process-global pool, not resizable per study. Do not treat a study `threads=` as `polars.thread_pool_size`.

4. **Burn training lock.** One process mutex. Fold-level `map_indexed` cannot give each fold a share of the study threads; the second fold waits, and seeding is global.

5. **Python providers and BLAS/joblib.** Callback and registry executions receive no thread count. Product code never sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, or `n_jobs`. The notebook kernel pins those to 1 for itself only. A provider that starts joblib or a BLAS pool is outside the study budget.

6. **Forust, if `parallel` were true.** Fit and predict force `false` (`gbt.rs:77`, `:127`). The underlying control is a global rayon boolean, not a per-call width.
