# Structural transport and randomized interference

Antecedent 0.5 adds two causal settings that should not be hidden behind an
ordinary target-population flag. Both use specialized stage APIs because they
change what information identifies the estimand.

## Structural transport is not prior transfer

`antecedent.transport` describes a source population, a target population, and
the variables whose causal mechanisms may differ. This is graphical
transportability: it asks whether source experiments and target observations
identify a response in the target. It is separate from `antecedent.priors`,
which transports statistical evidence after the causal quantity is already
defined.

The 0.5 contract is single-source. A selection diagram contains:

- one source and one target population key;
- a causal ADMG;
- an explicit set of mechanism-selection targets;
- the variables experimentally available in the source.

The identifier emits population-labelled factors and a positive certificate
when one of its implemented sound rules applies. The implemented subset covers
direct transport, exhaustive pre-treatment S-admissible standardization (up to
20 candidate covariates), and recursive singleton-c-component factorization.
S-admissibility is checked by adding the selection nodes explicitly and testing
m-separation in the treatment-mutilated selection diagram. Above the bounded
subset search, Antecedent fails closed rather than substituting a heuristic.

When a general multi-node c-component requires recursion outside that subset,
the result is `NotCertified`. That means “this implementation has not certified
a formula,” not “the effect is proven non-transportable.”

Trial-to-target statistical estimation is a separate, complementary operation.
The binary randomized-trial estimator reports IPW and optional augmented IPW,
plus separate diagnostics for trial-selection overlap and within-trial treatment
overlap. A single combined overlap number would conceal which assumption is
failing.

`transport_augmented_response_grid` evaluates a caller-specified augmented
target-population equation at every intervention value. Callers supply target-row
outcome regressions, observed-treatment regressions, and the grid-local source
weights; the API deliberately does not conceal treatment-density estimation or
bandwidth choice. It returns the transported mean and grid-specific source
effective sample size, while keeping selection overlap separate. This composes
the algebraic forms of the Dahabreh et al. trial-generalization correction and
Kennedy et al. local continuous-response weights, including signed local-linear
equivalent weights. Neither paper derives the joint estimator, so this primitive
does not claim double robustness, efficiency, or simultaneous-band inference.

## Interference starts with the assignment design

Under interference, an outcome may depend on more than its own treatment.
`antecedent.interference` therefore requires three explicit objects:

1. an assignment design;
2. an exposure mapping from the global assignment vector to a unit exposure;
3. a contrast between two exposure levels.

Supported designs are Bernoulli, complete, and cluster randomization. Built-in
exposures include own treatment, treated-neighbor count/fraction, and weighted
neighbor exposure on a fixed validated network.

Exposure probabilities are enumerated exactly through the configured small-
design limit and estimated with deterministic seeded Monte Carlo above it. The
result retains both Horvitz–Thompson and Hájek estimates; they are not aliases.
Positivity is checked at the unit/exposure level.

The variance diagnostic in 0.5 is a conservative covariance-free Young bound.
It is intentionally not labelled as the exact Aronow–Samii joint-exposure
variance estimator. The `conservative_variance` field, documentation, and
provenance record preserve that distinction; no coverage theorem is claimed.

## Scope boundaries

- Multi-source meta-transport is not part of 0.5.
- `NotCertified` is not a non-transportability theorem.
- The network is treated as fixed and supplied by the caller.
- Observational network treatment and graph semantics for contagion or
  allocational interference remain future work.
- Cyclic/equilibrium causal systems remain outside this release.
