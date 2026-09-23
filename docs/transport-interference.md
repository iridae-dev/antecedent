# Structural transport and randomized interference

Transport is not a second product or a population flag. It extends the same
causal contract to a richer evidence environment: the target population,
source regimes, experiments, measurements, sampling, and dependencies remain
explicit. Read [refusal and partial knowledge](refusal-and-partial-knowledge.md)
for why missing evidence, non-certification, support failure, and budget
exhaustion have different meanings.

Transport and interference are licensed causal settings that should not be
hidden behind an ordinary target-population flag: both change what information
identifies the estimand, so their design facts are explicit fields of their
queries. One cell of each is licensed at validation `none`:
`TransportQuery` × `Admg` × explicit × Frequentist (Direct / S-admissible sID
plus binary trial-to-target IPW), and `InterferenceQuery` × `Dag` × explicit ×
Frequentist (NeighborCount under Bernoulli assignment, HT/Hájek, Young
variance). The [support matrix](support-matrix.md) is the license.

Both run on the ordinary lifecycle and retain a study like every other
licensed cell:

```python
import antecedent as ant
from antecedent import interference, transport

query = transport.advanced.TransportQuery(
    ant.ResponseCurve("a", "y", grid=[0.0, 1.0]),
    transport.advanced.SelectionDiagram("trial", "target", ["x"]),
    source_experiments=["a"],
    trial="trial",                    # source-trial membership column
    selection_probability="s",        # P(S=1 | X) on every row
    treatment_probability="e",        # P(A=1 | X, S=1) on trial rows
)
result = ant.analyze(data, graph=admg, query=query)   # admg: an Admg over data's columns
study = result.study
updated = study.refresh(new_data)
report = result.inspect().to_dict()
loaded = ant.load(result.export())

query = ant.InterferenceQuery(
    interference.BernoulliAssignment(0.5),
    interference.NeighborCount(),
    interference.ExposureContrast(
        "y", interference.ExposureLevel(0.0), interference.ExposureLevel(1.0)
    ),
    network=edges,                    # fixed (from, to[, weight]) unit-row edges
    realized_assignment=assignment,   # binary assignment in unit-row order
)
result = ant.analyze(units, graph=[], query=query)
```

The selection diagram and trial column bindings, and the network and realized
assignment, freeze at prepare. The data is what a refresh replaces: a
transport refresh reads the trial columns from the new rows, and an
interference refresh executes on new outcomes under the frozen design. The
transported IPW is `result.estimate.ate` with its selection and treatment
overlap on `result.transport_overlap`; the Horvitz–Thompson contrast is
`result.estimate.ate` with the Hájek estimate, conservative variance and
exposure-probability methods on `result.interference`.

A design-defined estimand has no score table to reweight, so
`study.retarget(...)` refuses (`reason_code="population_not_estimable"`); there
is no refuter suite for it, so a second-click `study.refute(...)` and a positive
`bootstrap=` refuse (`option_not_applicable`). On `analyze` a construction
outside the licensed cell — a transported derivative or other inner functional,
RecursiveFactorization, complete or cluster randomization, or another exposure
mapping — refuses with `construction_not_licensed`, because a licensed claim
fails closed.

## Unlicensed utilities

`transport.advanced.estimate_trial_effect` and `interference.estimate` are unlicensed
utilities that return bare numbers: augmented IPW via `mu0` /
`mu1`, Bernoulli, complete and cluster randomization, every built-in exposure
mapping, and `seed` as the exposure-probability Monte Carlo seed. They call the
same Rust estimators as `analyze` (`trial_to_target_effect`,
`estimate_interference`) and return bare numbers, with no study, contract,
export or calibration slot, and they never claim a license.
`analyze(query=TransportQuery(...))` / `analyze(query=InterferenceQuery(...))`
is the licensed, study-retaining path. On a licensed construction the two agree
on the point numbers; the `analyze` path derives its Monte Carlo stream from the
analysis seed.

The calibration slot of both cells is keyed by the construction the coverage
harness measures (`crates/antecedent/tests/v110_calibration_design.rs`):
query, graph class, `fixed` structure, `tabular` modality, `Frequentist`,
estimator (`transport.trial_ipw` / `interference.ht_hajek`), `analytic_se`,
`iid` dependence, `point` identification, at the reported 0.95 level. A
record measured under that key binds to the claims of an `analyze` result; the
interference record is a named conservative boundary, reported as
`scope_not_assessed` with its observed coverage.

## Structural transport is not prior transfer

`antecedent.transport` describes a source population, a target population, and
the variables whose causal mechanisms may differ. This is graphical
transportability: it asks whether source experiments and target observations
identify a response in the target. It is separate from `antecedent.priors`,
which transports statistical evidence after the causal quantity is already
defined.

The licensed transport contract is single-source. A selection diagram contains:

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

`antecedent.transport.advanced.identify` stages identification alone: it decides
*whether* a formula is sound and returns the formula and certificate. The Rust
`trial_to_target_effect` and `transport_augmented_response_grid` primitives take
the identification result as a required argument and refuse to run when it is
`NotCertified`, returning an error that carries the certificate's reason and
message rather than an estimate. They also refuse a `RecursiveFactorization`
certificate: the Dahabreh-style IPW/AIPW algebra evaluates direct transport and
standardization, not the truncated product of population-labelled factors.

The binary randomized-trial estimator reports IPW and optional augmented IPW
(the licensed `TransportQuery` cell publishes IPW; augmented IPW is available
from the unlicensed `transport.advanced.estimate_trial_effect` utility), plus separate
diagnostics for trial-selection overlap and within-trial treatment overlap. A single combined overlap number would conceal which assumption is
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
neighbor exposure on a fixed validated network; the unlicensed
`interference.estimate` utility accepts all of them. The licensed
`InterferenceQuery` cell on `analyze` is NeighborCount under Bernoulli
assignment.

Exposure probabilities are enumerated exactly through the configured small-
design limit and estimated with deterministic seeded Monte Carlo above it. The
result retains both Horvitz–Thompson and Hájek estimates; they are not aliases.
Positivity is checked at the unit/exposure level.

The licensed variance diagnostic is a conservative covariance-free Young bound.
It is intentionally not labelled as the exact Aronow–Samii joint-exposure
variance estimator. The `conservative_variance` field, documentation, and
provenance record preserve that distinction; no coverage theorem is claimed.

## Scope boundaries

- The licensed cells are the implemented sID subset plus Dahabreh IPW
  (transport) and NeighborCount under Bernoulli assignment (interference).
  RecursiveFactorization, NotCertified, Aronow–Samii variance, cheap/full,
  Bayesian, accepted, and graph-posterior are refused.
- Multi-source meta-transport is not licensed.
- `NotCertified` is not a non-transportability theorem.
- The network is treated as fixed and supplied by the caller.
- Observational network treatment and graph semantics for contagion or
  allocational interference are outside this contract.
- Cyclic/equilibrium causal systems remain outside this release.
