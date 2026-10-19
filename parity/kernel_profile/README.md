# Kernel profile (N15)

`plan.toml` names four kernels. It is not a measurement. Every kernel has status `registered_not_measured`. No timings are stored here. Explicit SIMD stays `unmade` until that kernel has a portable-layout measurement. A layout contract for each kernel is `absent`, so that measurement has not started.

N16's scalar reference and tolerance must exist before promotion. An upstream change to an owning source requires a remeasure. Matching, CI, and family work do not supply this record. `benches/baselines/kernel_reductions.md` is a different record.

## Kernels

| Id | Role | Owner | Symbols |
| --- | --- | --- | --- |
| `fused_reductions` | fused reductions | `crates/antecedent-kernels/src/portable.rs` | `masked_sum`, `masked_mean`, `masked_variance`, `masked_covariance` |
| `bitmap_mask` | bitmap/mask handling | `crates/antecedent-kernels/src/scalar.rs` | same four names, when `mask` is `Some`; bitmap type is `BitMaskView` in `crates/antecedent-kernels/src/view.rs` |
| `score_gram` | score/Gram accumulation | `crates/antecedent-stats/src/gram.rs` | `form_xtx`, `form_xty`, `accumulate_xtx`, `accumulate_xtx_xty_row` |
| `column_scaling` | column scaling | `crates/antecedent-kernels/src/scalar.rs` | `standardize_inplace` |

`pearson_fused` in `parcorr.rs` calls `pearson`. `portable.rs` `standardize_inplace` calls the scalar function. `form_gram` and `scale_column` in `parcorr.rs`, and `symmetric_scales` / `equilibrate_into` in `crates/antecedent-kernels/src/gram.rs`, are different operations. `form_xtwx` and `accumulate_xx` in `covariance.rs` are the weighted score sandwich. `scan_against_validity` in `column.rs` is a column invariant scan.

## Commands

No independent portable-layout bench exists for these four kernels. No repository script inspects LLVM vectorization or assembly. `cargo asm` is not a project command.

These existing benches time something else. They are not an N15 measurement:

```bash
cargo bench -p antecedent-kernels --bench reductions -- --sample-size 30
cargo bench -p antecedent-stats --bench linalg -- gram_xtx_n400_p12
cargo bench -p antecedent-kernels --bench partial_correlation
```

`reductions` times `masked_covariance_n10k`, `standardize_inplace_n10k`, `weighted_sum_n10k`, `pairwise_l1_n256`, and `accumulate_contingency_n10k` in one target. `linalg` times `form_xtx` as `gram_xtx_n400_p12` among other linear-algebra functions (`crossprod_xty_n400_p12`, `weighted_xtwx_n400_p12`). `partial_correlation` times a batch that calls private partial-correlation helpers. None of the three isolates the four kernels, and none is a portable-layout measurement under a layout contract.

`python3 scripts/bench_regression.py [CRITERION_DIR] [RESOURCE_DIR]` exists. It checks Criterion relative change for the release lanes. It does not profile these kernels.

There is no command that profiles `masked_sum`, `masked_mean`, or `masked_variance` alone, no command that profiles the mask loops, no command that profiles `accumulate_xtx_xty_row`, and no command that profiles `standardize_inplace` apart from the mixed `reductions` target.

After a layout contract exists for a kernel, LLVM remarks and assembly for its crate are a toolchain invocation, not a repository script:

```bash
RUSTFLAGS="-C remark=loop-vectorize" cargo rustc -p antecedent-kernels --release --lib -- --emit asm
RUSTFLAGS="-C remark=loop-vectorize" cargo rustc -p antecedent-stats --release --lib -- --emit asm
```

`--emit asm` writes assembly under `target/release/deps`. Nothing in the repository extracts those remarks per kernel. Do not treat either invocation as a completed profile, and do not record a SIMD decision from it until the portable-layout measurement exists.
