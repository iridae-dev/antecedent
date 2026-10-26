# Cross-world selective execution

Cross-world quantities (counterfactuals, natural and path-specific contrasts,
temporal counterfactuals) are computed by one staged pipeline whose stages are
kept separate so a change of one input rebuilds only the stages that depend on
it.

| Stage | What it is | Canonical dependencies |
|---|---|---|
| mechanism fitting | the mechanisms the worlds are evaluated under (declared or supplied, not fitted, in the delivered rows) | population, history, graph, mechanism parameters |
| factual abduction | the exogenous posterior given each unit's factual evidence | mechanisms, factual evidence, noise law, inference method |
| exogenous draws | the shared exogenous states every world reads | abduction, stream coordinates `(unit, node, purpose)` |
| action-specific prediction | each named world evaluated under the shared states | draws, world definitions, routes |
| summary / decision | the requested quantity from the coupled worlds | prediction, quantity, declared coupling |

An action-only change (new worlds, routes or quantity) reuses abduction and the
shared draws. Changed factual evidence, a changed mechanism parameter at the
same shape, a changed graph or noise law, or a different population or history
is a different abduction identity and abduces again. Observed work counters
(`mechanism_fits`, `abductions`, `abductions_reused`, `shared_draws`,
`world_predictions`, `summaries`) count work at the actual boundary; a reused
abduction is never counted as work. Counters of a cancelled or refused call are
not committed, and a failed call leaves retained abductions unchanged.

## Capability matrix

The matrix is checked data (`parity/cross_world_capability.toml`, compared with
`antecedent_counterfactual::selective::matrix::ROWS` by a test). A request
outside an executed row is a scientific refusal naming the unmet dimension and
the declaration that would open it; it is not a cache miss.

| Row | Status |
|---|---|
| counterfactual marginal, finite Markovian SCM, with or without factual evidence | executed |
| counterfactual joint event over worlds, finite Markovian SCM | executed (shared exogenous coupling) |
| natural / path-specific contrast, finite Markovian SCM | executed (shared exogenous coupling) |
| the same three quantities, fully declared finite SCM with an exogenous term shared by several mechanisms | executed (one joint exogenous state per unit) |
| fixed-population two-slice temporal counterfactual | executed, staged with abduction reuse |
| fixed-population temporal counterfactual, any horizon and any number of named action histories | executed, staged with abduction reuse (`GeneralTemporalExecutor`); sealed as `temporal_counterfactual_general_v1`. The two-slice, two-history route keeps its stable `temporal_counterfactual_v1` layout |
| transported static additive linear path-specific | keeps its own theorem-scoped route |
| transported cross-world execution | closed: `cell_not_licensed`, prerequisites named |
| latent confounding without declared mechanisms, nonparametric ADMG | closed: `cross_world_not_identified` |
| declared model whose joint exogenous space exceeds the enumeration bound | executed by exact variable elimination (`exogenous_method = variable_elimination`); `route_not_supported` (`cross_world_execution.elimination_bound_exceeded`) only when the elimination order needs a table above its bound |
| incompatible history, soft interventions, fitting a finite SCM from data | closed |

## The finite SCM rows

Variables are finite and listed topologically. Every mechanism table and every
finite exogenous law is declared. Each variable names one exogenous term; a term
may be named by one variable (a Markovian model, rows `finite_markovian_scm`) or
by several (rows `finite_shared_exogenous_scm`). Each unit's abduction enumerates
the joint exogenous states exactly, keeps those consistent with the evidence and
normalizes; every world evaluates the same equations with its hard interventions
and routes under the same posterior state, so worlds are coupled by construction
(Pearl 2009, chapter 7; Balke and Pearl 1994). Nothing is sampled: `shared_draws`
is zero.

Past 65536 joint exogenous states the joint space is never formed. The same
quantity is computed exactly by variable elimination over the factor graph of the
exogenous terms and the per-world variables: one indicator factor per mechanism
and world (reading the same exogenous variable in every world, which is the
coupling), the factual evidence and the interventions as constants, every
variable summed out except the ones the quantity reads, smallest induced table
first. Its cost is exponential in the width of the elimination order, not in the
product of the term sizes, so a model with many weakly connected terms is cheap.
The receipt records `exogenous_method = variable_elimination`, the units, the
largest table, the multiply-accumulate count and a digest of every unit's
elimination order; nothing is retained for reuse, because the elimination depends
on the worlds and the quantity as well as the evidence.

A shared exogenous term is latent confounding in the structural sense. When the
model is fully declared there is no identification question: the term takes one
value per joint state, every mechanism that reads it sees that value, in the
factual world and in every counterfactual world. A patient observed treated is
thereby known to carry the hidden state that also shaped the outcome, and the
counterfactual outcome inherits it; replacing the shared term by independent
noise gives a different answer (see `examples/python/shared_exogenous_counterfactual.py`).

Three situations are kept apart in refusals:

- A model whose mechanisms or exogenous law are not declared (a nonparametric
  ADMG, or latent confounding to be fitted from data) does not identify a
  cross-world law: `cross_world_not_identified`
  (`cross_world_execution.mechanisms_not_declared`,
  `cross_world_execution.latent_confounding_not_identified`).
- A declared model whose elimination order needs a table above the documented
  bound (`cross_world_execution.elimination_bound_exceeded`), or a request outside
  an executed row, is unsupported execution, not non-identification:
  `route_not_supported`.
- Intervals are not extended to shared terms: the measured whole-path coverage
  record covers one exogenous term per variable, so a shared-term model reads a
  point only.

Whether draws may be paired is decided only by the joint-law rule
(`check_propagation`): a quantity over several worlds needs the aligned
exogenous states (`SharedAbducedExogenous`); independently drawn world noise or
an undeclared coupling is refused with `joint_law_required`. A mean contrast of
single-world marginals needs no coupling.

Bounds (registered in `parity/limits.toml`):

- Variables: 4096
- Worlds: 256
- Units: 4000000
- Joint exogenous states: 65536 per unit (the product of every exogenous term's states, shared or not); larger models are eliminated, not refused
- Elimination table entries: 4194304 per table
- Elimination variables: 8192 per unit
- Elimination work: 4294967296 multiply-accumulate operations per request

## Readings and calibration

Every result is `point_only` from an exact computation (enumeration or variable elimination) on a declared model; model
adequacy is a premise, not a check. A posterior reading and a sampled posterior
(`ExogenousInference::PosteriorSampling`) are closed with `cell_not_licensed`:
the mechanisms are declared and the exogenous posterior is enumerated exactly, so
there is no parameter uncertainty a posterior could describe, and sampling only
adds Monte Carlo error to an exact value. The unqualified `interval()` is closed.

`measured_interval(unit_sampling)` reads an interval through a declared unit
sampling from the derived whole-path spread:

| Unit sampling | Interval for | Variance |
| --- | --- | --- |
| `independent_sample` | the population mean of the unit counterfactual probabilities | `s^2 / n` of the unit values |
| `fixed_units` | the mean of the realized counterfactual outcomes of these units | `sum v_i / n^2` |

Both are normal intervals at 0.95, are conditional on the declared mechanisms
being the truth, and are licensed only by a currently attesting whole-path
coverage record for the model class (at most three variables of two to four
levels, exact enumeration, the mechanisms' own population), the quantity, the unit
sampling and the unit count; otherwise they refuse with `cell_not_licensed`.

Measured design (`cross_world_interval_calibration.rs`): response-type finite
Markovian SCMs with unit-level evidence of the observed treatment and outcome, the
quantity `P(Y_{do(X=0)} = 0 | evidence)`, truth by exact population probability
(independent sample) and by the realized counterfactual outcome (fixed units).

## Export and resume

`cross_world_execution_v1` seals the mechanisms, factual conditioning, world
definitions with routes, the quantity, the declared coupling, the shared
exogenous states (with semantic stream-coordinate digests) and the numbers. A
consumer replays everything with a fresh executor and refuses on any
difference; the verified executor is returned so an action-only change resumes
from the verified abduction. The artifact claims a point only.

## Provenance

Abduction-action-prediction: Pearl, *Causality* (2nd ed., 2009), chapter 7.
Balke and Pearl, "Counterfactual Probabilities: Computational Methods, Bounds
and Applications", UAI 1994. Pearl, "Direct and Indirect Effects", UAI 2001.
