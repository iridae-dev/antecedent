**Validation evidence for the full 1.x review**

Environment: macOS arm64, pinned Rust 1.85, CPython 3.13. Commands ran against the shared `1.11` working tree. The working tree was already dirty and received additional concurrent changes during the review. No passing result below is an exact-commit release certificate.

| Check | Result |
| --- | --- |
| `cargo test --workspace` | Passed: 2,789 tests, 303 ignored, 174 result blocks including doctests. This build preceded the late contract-reporting edits. |
| Late `cargo test -p antecedent --test v110_contract --test practitioner_facade` | Passed: 132 tests, covering the concurrently modified contract test and practitioner facade. |
| Initial `python/.venv/bin/python -m pytest python/tests -q` | Passed: 1,986 tests, 127 warnings, 73.82 seconds, using the existing optimized extension. |
| Optimized extension build and install | Passed: `maturin develop --release --offline` confirmed installation of editable antecedent 1.11.0 after a 2m57s release build. |
| Python suite after confirmed extension installation | Passed: 2,001 tests, 127 warnings, 42.16 seconds, including the 15 concurrent matrix-edge regressions. |
| `cargo clippy --workspace --all-targets -- -D warnings` | Failed: three existing errors in `antecedent-core/src/execution.rs`: truncating cast at line 50, missing `# Panics` and `# Errors` documentation for `try_map_indexed` at line 517. |
| `gate_rustfmt.sh` | Failed: 28 files outside the three explicitly exempted measured pin suites require formatting. Filename list retained in `quality-gates.txt`. No formatting changes were applied. |
| Ruff check / format, run from `python/` with project configuration | Passed initially (194 files) and after concurrent test additions (195 files). |
| mypy, run from `python/` | Passed: no issues in 45 source files. |
| `gate_provenance_schema.sh` | Passed: 168 records, 422 path references. Does not verify historical authorship or all bibliographic metadata. |
| `gate_metadata_consistency.sh` | Passed: version 1.11.0 and 168 provenance records. |
| `gate_support_matrix.sh` | Passed: 3,402 cells; 463 licensed, 1,999 not applicable, 940 reasoned refusals, zero unexplained refusals. |
| `gate_docs_support_matrix.sh` | Passed. |
| `gate_parity_schema.sh` | Passed after routing uv's cache to a writable directory. |
| `gate_hot_path_baselines.sh` | Passed metadata checks: 24 linked baseline files, 15 parseable numeric gates. This was not a fresh measured timing-regression campaign. |
| `gate_calibration_attestation.sh` | Passed: 258 matching + 329 replay-attested = 587 records. Subject to R1's shared-dependency gap. |
| `gate_release.sh` | All domain/composition gates and designated benchmark smokes passed. Overall script exited 1 at final `cargo deny check`: default advisory-cache lock was on a read-only path. Retrying with a temporary config and writable `/tmp` database path reached the fetch, but GitHub DNS resolution failed. Dependency advisory audit remains unverified; this is an environment failure, not a reported dependency vulnerability. |
| Public-API review probes | Executed and reproduced R2–R8; dependency probe reproduced R1. Recorded in `probe-output.txt`. Public Python padding probe reproduced R11 in `sample-scope-output.txt`. |
| Arrow and mmap safety analysis | Static source/API analysis only; no UB/concurrent-write exploit was executed. |

The normal Rust test build also warned that `TRUTH_GIVEN_IDENTIFIED` at `crates/antecedent/tests/v19_static_mixture_calibration.rs:57` is unused. CI sets `RUSTFLAGS=-D warnings`, so the ordinary local test pass does not imply CI compilation would pass.

The initial uv-based lint/build/schema commands encountered a sandbox-denied default cache directory. The tools themselves were available. Ruff and mypy were run directly from the existing virtual environment with the correct working directory. Later uv commands used `UV_CACHE_DIR=/tmp/antecedent-review-uv`, `UV_OFFLINE=1`, and, for gates, `UV_NO_SYNC=1`. An exploratory Ruff invocation from the repository root used different configuration for examples; its diagnostics were discarded after the documented invocation from `python/` passed.

An intermediate `maturin develop --skip-install --release --offline` built a wheel but did not establish installation of that wheel. A subsequent Python run during that transition collected the 15 newly added matrix-edge tests and reported 1,998 passed / 3 failed (`KeyError: components` in the new Bayesian uncertainty assertions). After confirmed installation of the current extension, all 2,001 tests passed. The transition run is not used to claim a new production defect.

The eleven review findings are independently reproducible or established by safety-contract inspection. R1–R10 occur in files unchanged from committed HEAD; R11's relevant code region also existed in HEAD despite concurrent edits elsewhere in its file. Their validity does not depend on the late concurrent test additions.

Not performed: fresh full statistical calibration, independent sibling practitioner acceptance, fresh advisory-database audit, CodeQL, cross-platform wheels, publishing dry-runs, isolated installed-wheel acceptance outside this checkout, or release-candidate verification with a successful CI run on an exact clean commit. The 303 ignored Rust tests are explicitly excluded from the passing-test count.

Logs were written under `/tmp/antecedent-review-*.log` during execution. Durable summaries and the numerical probe output are retained here; temporary logs are not assumed to survive machine cleanup.
