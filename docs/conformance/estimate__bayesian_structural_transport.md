# Finite-discrete Bayesian transport law oracle

**Suite path:** `conformance/estimate/bayesian_structural_transport`

The fixed table has counts `(X=0,Y=0)=20`, `(0,1)=0`, `(1,0)=10`, and
`(1,1)=1`. Under the identified `X → Y` intervention query, the target
probability of `Y=1` at `do(X=1)` is the conditional cell ratio.

The empirical-support Bayesian bootstrap gives the two `X=1` cells
Dirichlet parameters `(10,1)`, so its posterior mean is `1/11`. The declared
binary state-space Dirichlet provider adds one to each cell, giving `(11,2)`
and posterior mean `2/13`. The `X=0,Y=1` zero cell remains impossible only
under empirical support. These are exact Dirichlet moment identities; the
Rust test compares 2,048 joint draws with the frozen values and round trips
each provider through the prepared-study artifact. This fixture checks
posterior computation and provider distinction, not interval calibration.

## Expected summary

Top-level keys: `counts_x0y0_x0y1_x1y0_x1y1, draws, empirical_support_p_y1_given_do_x1, monte_carlo_tolerance, schema_version, state_space_p_y1_given_do_x1` (6 fields).
