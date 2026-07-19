# causal-kernels

Borrowed matrix/vector views and scalar, portable-optimized, and (optional)
architecture-specific kernels. Contains no causal semantics. 1
and §23.2.

Public dispatch selects once per batch via `KernelPolicy` (`force_scalar`,
`allow_portable_optimized`, `allow_arch_simd`). Scalar is the correctness gold
standard; portable is always compiled. Arch SIMD is selected only when
`arch_simd_available()` is true (requires a justified `simd-runtime` feature).

Kernels include masked reductions/covariance, gather/copy, standardization,
pairwise L1, contingency accumulation, bootstrap weighted sum/mean/dot, and
partial correlation.
