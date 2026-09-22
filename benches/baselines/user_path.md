# User-path jobs (1.11 P1)

Established: 2026-09-19 (the date this file was first committed; the measurement date was not written down)
Machine class: not recorded; docs/hot_paths.md describes these baselines as Apple M1 class references
Commit: 6a1a2463 (the commit that added this file; the measured commit was not written down)

Not a `hot_paths.md` merge blocker. Not in `scripts/gate_release.sh` Criterion smoke.

Harness: `crates/antecedent/benches/user_path.rs`.

```text
# CI / compile smoke (toy n)
cargo bench -p antecedent --bench user_path -- --test

# Local 10⁴ / 10⁵ (this M-series laptop)
USER_PATH_N=10000 cargo bench -p antecedent --bench user_path -- --quick
USER_PATH_N=100000 cargo bench -p antecedent --bench user_path -- --quick
```

Jobs (default user path: `ExecutionContext::production_default`, published interval once):

| Job | What it hits |
| --- | --- |
| `dag_ate_frequentist` | Dag ATE, default 199-replicate bootstrap at full n |
| `dag_ate_bayesian_laplace` | Laplace GLM + certified draws |
| `intervention_level` | InterventionResponse level |
| `kennedy_curve` | Kennedy-DR mean curve |
| `graph_posterior_mixture` | Few-atom GP mixture under `ctx.parallelism` |
| `temporal_pulse` | Pulse on a keepable series (block-length / Politis–White when B>0) |

Recorded on each run (stderr): wall time, `copy_count`, `bytes_borrowed`, `exec.identify.cached` hits, `max_threads`. Peak RSS is local (`/usr/bin/time -l`).

Full 10⁴ / 10⁵ numbers set 1.11 / 2.0 sitting-time budgets. They do not replace `hot_paths.md`.

## Profile poles (local)

Kennedy-DR + local-quadratic, Laplace GLM, default 199 bootstrap, Laplace draws, GP per-atom refit, GAC search, temporal block-length, pandas ingest vs Arrow CDI.

P4 hunts only what those profiles name. Known leftovers after this cut:

- GAC candidate enumeration is still per-call; ordinary ID already memos `SubproblemKey`. Inspect/capability must not search (pinned). User graphs that never hit the cap of 40 do not pay a memo miss.
- `PY_DEFAULT_CACHE_MAX_BYTES` (4 MB) is on; `CacheBudget` refuses (`StateError::CacheBudget`) rather than growing into OOM.
- `arch_simd_available()` stays false. No `simd-runtime` unless a profiled path matches scalar tests bit-for-bit.
- Arrow CDI borrow is the ingest win; pandas/dict `as_columns` stays the explicit copy.
