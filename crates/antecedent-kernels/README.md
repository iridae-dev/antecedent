# antecedent-kernels

Borrowed matrix and vector views and numerical kernels. The crate has no causal semantics.

Dispatch selects once per batch through `KernelPolicy` (`force_scalar`,
`allow_portable_optimized`, `allow_arch_simd`). Scalar kernels are the correctness reference;
portable kernels are always compiled. No architecture-specific kernel is compiled in, so
`allow_arch_simd` never changes the selection.

Kernels include masked reductions and covariance, gather, standardization, pairwise L1,
contingency accumulation, bootstrap weighted sums and means, Cholesky and Gram products, and
partial correlation.

Most users want the [`antecedent`](https://crates.io/crates/antecedent) facade, which uses these kernels internally and exposes none of them as causal API. Depend on this crate directly only if you are extending the workspace's numerical code.

Main types: `F64MatrixView`, `F64VectorView`, `KernelPolicy`. API reference: <https://docs.rs/antecedent-kernels>.

## License

MIT OR Apache-2.0.
