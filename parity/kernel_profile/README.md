# Kernel profile (N15)

This is the historical measurement at `8f69473d8`. The current diagnostic harness
uses the underlying borrowed views directly; experiment-only layout wrappers
and the isolated N16 reduction API have been retired. This record does not
measure the subsequently promoted Gram backend.

`plan.toml` names four kernels. `measurement.toml` is the portable-layout measurement. Explicit SIMD is `not_promoted`: the release bodies are scalar `f64`, and this record does not add an architecture kernel.

Any future SIMD promotion needs an independent numerical contract and fresh measurements. An upstream change to an owning source requires a remeasure. Matching, CI, and family work do not supply this record. `benches/baselines/kernel_reductions.md` is a different record.

## Layout contract

| Id | Contract |
| --- | --- |
| `fused_reductions` | contiguous `F64VectorView` in `crates/antecedent-kernels/src/view.rs` |
| `bitmap_mask` | `BitMaskView` in the same module. `1` = included, least-significant bit first. |
| `score_gram` | `ColumnMajorDesign` and `DesignRow` in `crates/antecedent-stats/src/gram.rs` |
| `column_scaling` | a mutable contiguous `f64` slice |

## Measurement

```bash
cargo test --release -p antecedent-stats --bench portable_layout --offline -- --nocapture --test-threads=1
```

The bench times each kernel on that contract. `form_xtx` is the public measurement of `accumulate_xtx`. The numbers and the assembly notes are in `measurement.toml`.

## Kernels

| Id | Role | Owner | Symbols |
| --- | --- | --- | --- |
| `fused_reductions` | fused reductions | `crates/antecedent-kernels/src/portable.rs` | `masked_sum`, `masked_mean`, `masked_variance`, `masked_covariance` |
| `bitmap_mask` | bitmap/mask handling | `crates/antecedent-kernels/src/scalar.rs` | same four names, when `mask` is `Some`; bitmap type is `BitMaskView` in `crates/antecedent-kernels/src/view.rs` |
| `score_gram` | score/Gram accumulation | `crates/antecedent-stats/src/gram.rs` | `form_xtx`, `form_xty`, `accumulate_xtx`, `accumulate_xtx_xty_row` |
| `column_scaling` | column scaling | `crates/antecedent-kernels/src/scalar.rs` | `standardize_inplace` |

`pearson_fused` in `parcorr.rs` calls `pearson`. `portable.rs` `standardize_inplace` calls the scalar function. `form_gram` and `scale_column` in `parcorr.rs`, and `symmetric_scales` / `equilibrate_into` in `crates/antecedent-kernels/src/gram.rs`, are different operations. `form_xtwx` and `accumulate_xx` in `covariance.rs` are the weighted score sandwich. `scan_against_validity` in `column.rs` is a column invariant scan.

## Other benches

These time something else. They are not the N15 measurement:

```bash
cargo bench -p antecedent-kernels --bench reductions -- --sample-size 30
cargo bench -p antecedent-stats --bench linalg -- gram_xtx_n400_p12
cargo bench -p antecedent-kernels --bench partial_correlation
```

`reductions` times `masked_covariance_n10k`, `standardize_inplace_n10k`, `weighted_sum_n10k`, `pairwise_l1_n256`, and `accumulate_contingency_n10k` in one target. `linalg` times `form_xtx` as `gram_xtx_n400_p12` among other linear-algebra functions (`crossprod_xty_n400_p12`, `weighted_xtwx_n400_p12`). `partial_correlation` times a batch that calls private partial-correlation helpers. None of the three is the portable-layout measurement.

`python3 scripts/bench_regression.py [CRITERION_DIR] [RESOURCE_DIR]` exists. It checks Criterion relative change for the release lanes. It does not profile these kernels.

LLVM remarks for this record: `RUSTFLAGS="-C remark=loop-vectorize -C debuginfo=1" cargo rustc -p antecedent-kernels --release --lib -- --emit asm` and the same command for `antecedent-stats` printed no loop-vectorize remark. The assembly inspection is in `measurement.toml`.

## Promotion (N18)

`decision.toml` defers explicit SIMD. N17 was not built, so there is no AVX2 or NEON kernel to compare. `ARCH_SIMD_COMPILED` stays `false`. The portable bodies measured in `measurement.toml` remain the implementation.
