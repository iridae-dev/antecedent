# antecedent-kernels

Borrowed matrix/vector views and scalar, portable-optimized, and (optional)
architecture-specific kernels. Contains no causal semantics.

Public dispatch selects once per batch via `KernelPolicy` (`force_scalar`,
`allow_portable_optimized`, `allow_arch_simd`). Scalar is the correctness gold
standard; portable is always compiled. No architecture-specific kernel is
compiled in, so `allow_arch_simd` never changes the selection.

Kernels include masked reductions/covariance, gather, standardization,
pairwise L1, contingency accumulation, bootstrap weighted sum/mean, and
partial correlation.
