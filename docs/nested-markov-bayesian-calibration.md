# Continuous binary Verma Bayesian pilot: frozen acceptance design

Implementation and acceptance wiring are **unmeasured**. This document records the
scientific design; it contains no measured coverage result, waiver, or public license.
The normal `antecedent.transport.binary_nested_markov` route remains frozen.

The numerical candidate fits the original eleven-dimensional binary Verma nested-Markov
law: `X1 -> X2 -> X3 -> X4`, `X2 <-> X4`, parameter order
`a,c0,c1,q20,q21,q40,q41,g00,g01,g10,g11`. Its probability factorization is
`P(X1) P(X3|X2) Q(X2,X4|X1,X3)`, with the four Q probabilities
`g,q2-g,q4-g,1-q2-q4+g`. Strictly positive feasible raw parameters are required.
These are directly specified full-model laws, not a binary latent-U DAG restriction.
The raw-coordinate product Beta-kernel prior is truncated to this positive feasible
polytope. Conditional g-interval widths must not divide that density. Shapes are at
least one, making each coordinate log density concave. The three singleton coordinates
are updated by exact Beta conjugacy; the remaining eight use full-bracket slice shrinkage.

Primary scientific sources: [Evans and Richardson, original smooth identifiable model](https://arxiv.org/abs/1511.06813),
[Neal, slice sampling](https://arxiv.org/abs/physics/0009028), and
[Vehtari et al., rank-normalized/folded convergence diagnostics](https://arxiv.org/abs/1903.08008).

## Independent deterministic reference

`crates/antecedent-learn/tests/nested_markov_bayesian_reference.rs` integrates all four
association coordinates by polynomial-exact conditional Gauss-Legendre quadrature,
then numerically integrates the four margins at orders 16 and 24. Exact independent
Beta moments handle the three singleton coordinates. The low-count table has one
observation in every cell. Uniform raw priors and changed q4 Beta(4,1) kernels exercise
actual prior sensitivity. Checks cover all fourteen posterior means, all 196 covariance
entries, 95% effect credible endpoints, marginal q4 endpoints, and integration refinement.
No production sampler or production summary is used in this oracle. This finite reference
does not establish repeated-sampling coverage.

## Ignored repeated-sampling measurement

`crates/antecedent-io/tests/nested_markov_bayesian_calibration.rs` freezes two coordinates:

| Coordinate | Raw parameters in the order above | Prior on every raw coordinate |
| --- | --- | --- |
| uniform | `.43,.68,.31,.37,.61,.29,.64,.18,.28,.24,.46` | Beta(1,1) kernel |
| beta2 | `.36,.59,.24,.28,.66,.38,.73,.17,.22,.27,.50` | Beta(2,2) kernel |

Both use the shared sample-size grid based at 4000: 2000, 4000 and 8000 independent
multinomial observations. Independent enumeration and an independent generator produce
the tables. Exact intervention truths use `c_t(1-q40)+(1-c_t)(1-q41)`; contrast is
mean1 minus mean0. Every table executes the real graph/checked-ID/interior point-fit
prerequisite, continuous posterior producer, export and fresh independently bounded
consumer. The original point artifact is an eligibility restriction used to preserve
checked identification and fit replay; posterior existence does not require an MLE.

The frozen sampler uses four chains, 2048 warmup sweeps, 4096 retained draws per chain,
5,000,000 total proposal bound, and 95% equal-tailed posterior candidates. Declared
seeds, raw prior, counts, coordinate order and method/RNG identity are bound into the
artifact. No adaptation silently increases work, discards a chain, or drops a draw.
Every parameter and derived estimand must pass rank-normalized/folded split Rhat <=1.01
and bulk and five/ninety-five tail ESS >=400. These are modern diagnostics; they do not
prove convergence or the precision of the 2.5/97.5 endpoints.

The additional **measurement acceptance gate**, separate from the posterior estimator,
uses ESS of the actual 2.5/97.5 endpoint-CDF indicators. Their rank normalization is
affine on two-valued draws. Local bulk ESS must be >=400. An estimated 95% Monte Carlo
probability bracket `p +/- 1.959963984540054 sqrt(p(1-p)/ESS)` is mapped through the
empirical quantile function. Its width must be <=20% of the original interval width
at both endpoints of all three effects. This ESS-based MC bracket assumes sufficiently
mixed stationary chains. It is an estimated precision check, not a rigorous endpoint
confidence guarantee. A failed precision check is a whole-replicate failure, never
permission to discard draws or change the frozen estimator.

Extension rechecks restore a bounded auxiliary JSON sidecar adjacent to the exact
original tally path, keyed by test and sample size. Its fixture digest binds the original
law, raw prior family, sample grid/seed, sampler/RNG, precision gates and measured commit. The recorded
replicate prefix must equal the requested extension start; missing, mismatched, oversized
or inconsistent state is refused. Successful counts, sums, cross-products, posterior
covariance sums, failure categories and diagnostic extrema resume in replicate order.
Thus covariance, bias and diagnostics aggregate the original 400 plus the next 1600,
matching a whole 2000-replicate run. Worker results retain only independently replayed
summary statistics; full posterior draw arrays are released inside each worker, so their
peak memory follows the worker count rather than all measurement replicates. Atomic synchronized writes occur before coverage
assertions so a deferred recheck can recover the same prefix. A nonignored synthetic
partition/roundtrip test checks exact aggregation and incompatibility without simulation.

The harness uses shared `n_sim()` (default 400), the existing 1000-replicate precision
layer and 2000-replicate recheck rule without altering their acceptance bands or the 1%
failure cap. All zero-cell/scope/MLE/chain/numerical/budget/replay/MC-precision failures
count in the unconditional coverage denominator for each of mean0, mean1 and contrast.
It reports failure categories, convergence and endpoint diagnostics, mean bias, average
posterior joint covariance and repeated-sampling covariance. Covariance and bias summaries
are conditional on successful whole-method replicates; they cannot erase any failures
from coverage. Frozen auxiliary gates require relative Frobenius covariance error <=25%
and mean bias <=25% of the empirical sampling standard deviation in every effect.

Any eventual measured frequentist coverage applies only to these declared interior laws,
sample-size grid, priors and numerical settings. It is neither a generic Bayesian
coverage guarantee nor evidence for other graphs, boundary cells, fractional counts,
other priors, arbitrary levels, or other sampler settings. Whole-method calibration and
explicit public activation remain later work; this harness must not be run prematurely.
