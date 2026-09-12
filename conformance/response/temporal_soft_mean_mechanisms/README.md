# Population-mean-targeting Soft policies

`multiplicative(factor)` replaces a structural assignment `f` with `factor*f`.
`truncated_shift(delta, lower, upper)` instead defines

`f_policy = f + clip(mu + delta, lower, upper) - mu`,

where `mu = E[f]` under preceding interventions. This distribution-dependent
policy preserves parent effects and innovations. Its population mean satisfies
the bounds; individual outcomes need not. The sequential linear engine computes
the resulting causal mean exactly under its linear mechanism restrictions.
Bayesian draws evaluate the policy using each draw's implied population mean.

This is not stochastic clipping. For `X ~ N(0,1)` and bounds `[0,1]`, zero shift
leaves the population-mean-targeting mechanism unchanged with mean zero, whereas
`E[clip(X,0,1)] = phi(0)-phi(1)+1-Phi(1) = 0.3156268098137464`.
The frozen counterexample and consuming regression distinguish these estimands.
