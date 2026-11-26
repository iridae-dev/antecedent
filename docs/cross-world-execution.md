# Compute cross-world quantities in stages

Compute a quantity that compares one unit across several worlds: a counterfactual given what
was observed (`P(Y_{do(X=0)} = 1 | X = 1)`), a joint event over worlds, or a natural or
path-specific contrast. Use this route when you can declare the whole structural model: finite
variables, every mechanism table and the law of every exogenous (noise) term. The answer is then
an exact computation under that model. Model adequacy is a premise, not a check, and a model
whose mechanisms are not declared is refused as not identified.

## Ask a counterfactual of a declared model

A hidden frailty `u` (frail with probability 0.3) decides both who is treated, `x = u`, and how
the outcome responds, `y = x xor u`. For a patient observed treated, what is the probability of
`y = 1` had they not been treated?

```python
from antecedent import cross_world_execution as cw
from antecedent.outcomes import require_success

model = cw.FiniteSCM(
    population="clinic",
    history="current_regime",
    exogenous=(cw.ExogenousTerm("u", (0.7, 0.3)),),
    variables=(
        cw.FiniteVariable("x", 2, (), "u", (0, 1)),
        # table[u * 2 + x]: u = 0 gives y = x, u = 1 gives y = 1 - x.
        cw.FiniteVariable("y", 2, ("x",), "u", (0, 1, 1, 0)),
    ),
)
request = cw.CrossWorldRequest(
    scm=model,
    units=(cw.FactualUnit("patient", {"x": 1}),),
    worlds=(cw.World("untreated", {"x": 0}), cw.World("treated", {"x": 1})),
    quantity=cw.probability("untreated", "y", 1),
)
executor = cw.CrossWorldExecutor()
result = require_success(executor.execute(request))
print(result.value, result.label, result.matrix_row)   # 1.0, a point-only label, the matrix row
print(result.work)                                     # abductions, abductions_reused, ...
```

`result.value` is the mean over units of the requested quantity; `result.units` holds each
unit's value and the probability of its evidence. The value here is 1: observing `x = 1` reveals
`u = 1`, and the same `u` is read in the untreated world. `label` states that the result is a
point with no interval. Quantities are built with `cw.probability`, `cw.joint_probability` and
`cw.mean_contrast`; a `cw.World` may carry `cw.WorldRoute`s for natural and path-specific
contrasts. `cw.seal(request)` executes and returns an exportable `CrossWorldExecution`
([export and resume](#export-and-resume)). The complete script is
[`examples/python/shared_exogenous_counterfactual.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/shared_exogenous_counterfactual.py)
([`examples/rust/shared_exogenous_counterfactual.rs`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/shared_exogenous_counterfactual.rs) in Rust).

## Stages and reuse

Cross-world quantities (counterfactuals, natural and path-specific contrasts,
temporal counterfactuals) are computed by one staged pipeline. The stages are
kept separate so that a change of one input rebuilds only the stages that depend on
it: calling the same executor again with new worlds or a new quantity reuses the
abduction, and `work["abductions"]` is then 0.

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
the declaration that would open it; it is not a cache miss. In Python,
`cw.capability_matrix()` returns the rows and `cw.decide(...)` returns the row a
request falls in or raises its refusal.

| Row | Status |
|---|---|
| counterfactual marginal, finite Markovian SCM, with or without factual evidence | executed |
| counterfactual joint event over worlds, finite Markovian SCM | executed (shared exogenous coupling) |
| natural / path-specific contrast, finite Markovian SCM | executed (shared exogenous coupling) |
| the same three quantities, fully declared finite SCM with an exogenous term shared by several mechanisms | executed (one joint exogenous state per unit) |
| fixed-population two-slice temporal counterfactual | executed, staged with abduction reuse |
| fixed-population temporal counterfactual, any horizon and any number of named action histories | executed, staged with abduction reuse (`GeneralTemporalExecutor`); sealed as `temporal_counterfactual_general_v1`. The two-slice, two-history route is sealed as `temporal_counterfactual_v1` |
| transported static additive linear path-specific | keeps its own theorem-scoped route |
| transported cross-world execution | closed: `route_not_supported` (`cross_world_execution.no_transported_row_derivation`), prerequisites named |
| latent confounding without declared mechanisms, nonparametric ADMG | closed: `cross_world_not_identified` |
| declared model whose joint exogenous space exceeds the enumeration bound | executed by exact variable elimination (`exogenous_method = variable_elimination`); `route_not_supported` (`cross_world_execution.elimination_bound_exceeded`) only when the elimination order needs a table above its bound |
| incompatible history, soft interventions, fitting a finite SCM from data | closed: `route_not_supported` |

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
noise gives a different answer: 0.3 instead of 1 in the example above.

Two situations are kept apart in refusals:

- A model whose mechanisms or exogenous law are not declared (a nonparametric
  ADMG, or latent confounding to be fitted from data) does not identify a
  cross-world law: `cross_world_not_identified`
  (`cross_world_execution.mechanisms_not_declared`,
  `cross_world_execution.latent_confounding_not_identified`).
- A declared model whose elimination order needs a table above the documented
  bound (`cross_world_execution.elimination_bound_exceeded`), or a request outside
  an executed row, is unsupported execution, not non-identification:
  `route_not_supported`.

The whole-path coverage design measured one exogenous term per variable. A
shared-term model is therefore outside the measured class: its
[measured interval](#readings-and-calibration) is reported with a calibration
status that says so.

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
adequacy is a premise, not a check. Three readings are closed, each with its own code:

| Reading | Code | Why |
| --- | --- | --- |
| the unqualified `interval()` | `sampling_dependence_unknown` | an interval needs a declared unit sampling |
| a posterior (`posterior()`) | `uncertainty_representation_insufficient` | the mechanisms are declared and the exogenous posterior is enumerated exactly, so there is no parameter uncertainty a posterior could describe |
| a sampled abduction (`ExogenousInference::PosteriorSampling`) | `estimator_inference_mismatch` | sampling only adds Monte Carlo error to an exactly computable value |

`measured_interval(unit_sampling)` on a sealed execution reads an interval through a
declared unit sampling from the derived whole-path spread. With `n` units, `s^2` the
sample variance of the unit values and `v_i` the posterior predictive variance of unit `i`:

| Unit sampling | Interval for | Variance | Quantile |
| --- | --- | --- | --- |
| `independent_sample` | the population mean of the unit counterfactual probabilities | `s^2 / n` | Student t, `n - 1` degrees of freedom |
| `fixed_units` | the mean of the realized counterfactual outcomes of these units | `sum v_i / n^2` | normal |

Both are 0.95 intervals conditional on the declared mechanisms being the truth. The
interval is always reported together with a `calibration` entry and `record_id`: the
entry says whether a current whole-path coverage record measured this model class,
quantity, unit sampling and unit count, or the request lies outside what was measured.
The measured class is at most three variables of at most four levels with one
exogenous term per variable; a larger model or a shared term is reported with that
status, not withheld. The interval is refused for a sampled abduction
(`estimator_inference_mismatch`), a transported target population
(`transport_not_certified`) and a sampling that determines no variance, such as a
single unit under `independent_sample` (`uncertainty_representation_insufficient`).
`whole_path_uncertainty(unit_sampling)` returns the spread itself, labelled
`uncalibrated_spread`, with every source of uncertainty listed.

Measured design (`cross_world_interval_calibration.rs`): response-type finite
Markovian SCMs with unit-level evidence of the observed treatment and outcome, the
quantity `P(Y_{do(X=0)} = 0 | evidence)`, truth by exact population probability
(independent sample) and by the realized counterfactual outcome (fixed units).

## Export and resume

`cross_world_execution_v1` seals the mechanisms, factual conditioning, world
definitions with routes, the quantity, the declared coupling, the shared
exogenous states (with semantic stream-coordinate digests) and the numbers. A
consumer replays everything with a fresh executor and refuses on any
difference; the replayed executor is returned so an action-only change resumes
from its abduction. The artifact claims a point only.

```python
sealed = require_success(cw.seal(request))
replayed = require_success(
    cw.consume_cross_world_execution_artifact(sealed.to_bytes(), expected_identity=sealed.identity)
)
assert replayed.result.value == sealed.value
resumed = cw.CrossWorldExecutor.from_artifact(sealed.to_bytes(), expected_identity=sealed.identity)
```

## Discretize a Gaussian composition into a finite model

The finite counterfactual routes (exact enumeration, finite counterfactual transport) need variables with finitely many levels. A composed linear-Gaussian model has a closed form, and the discretization is the bounded bridge between them, with a computed error bound against the closed form. In Python, call `PreparedMechanismComposition.discretize(levels)` or `antecedent.discretized_composition.discretize`, then build a `DiscreteContrast` (`DiscreteContrast.ate` or `mean_at`) and pass it to `mean_contrast`. In Rust the surface is `antecedent_model::discretized_scm::DiscretizedLinearGaussian` and the facade `PreparedMechanismComposition::discretize` with `mean_contrast`, `refresh`, `export` and `consume_discretized_composition`.

### Method

You declare a number of levels per variable (2 to 64; never defaulted). Each
variable is cut at its `i/n` quantiles under the composition's observational
regime into equal-probability cells; the outer cells are open and the support
is not truncated. A cell's value is its conditional mean. A child's table row
for a parent configuration is the Gaussian mechanism evaluated at the parents'
cell values and integrated over the child's cells. `do(V = level)` fixes the
variable at that level's value; the closed form is computed at the same value.

### Error bound

For each variable the bound `|err_v| <= e_v + sum_p |b_vp| |err_p|` is evaluated
in topological order from the mean error `e_v` of each table. It is a computed
guarantee for the mean of every variable and for a linear contrast of means. It
says nothing about variances, probabilities or counterfactual quantities of the
discretized law.

### Refresh, export and consume

`refresh` discretizes a refit composition again under the declared plan: the
plan is kept; the cell edges, values, tables and identity are recomputed; a
refit that adds or removes a variable is refused (`levels_undeclared`) and the
handle is left as it was. `export` writes the source premises, the plan and the
claimed numbers. The consumer recomputes them: it
re-runs the discretization from the carried premises and plan and requires the
identity and every claimed edge, value and mean-error bound to equal the
re-derived ones bit for bit, so an edited premise or an edited claimed number
is refused. A consumer that retained the identity independently refuses a
self-consistent artifact of another composition. Only a linear-Gaussian
composition is exportable.

### Label

The discretized value is point-only: no interval, posterior or calibration, and the parameter uncertainty declared on the source mechanisms is not propagated through the discretization. A route refuses a statistical paradigm it does not support.

### What is refused, and the remedy

These details are registered under
[finite-SCM discretization](reference/refusal-codes.md#finite-scm-discretization).

| Refused request | Detail | Remedy |
| --- | --- | --- |
| Levels not declared or outside 2..=64 | `discretized_scm.levels_undeclared`, `levels_out_of_range` | Declare levels per variable |
| Joint space too large | `discretized_scm.state_space_too_large` | Fewer levels or fewer variables |
| Non-Gaussian, finite, mixed or bounded-support mechanism | `discretized_scm.non_gaussian_mechanism`, `support_not_real_line` | Use the route native to that family |
| Mechanism loading on a shared latent cause | `discretized_scm.latent_confounding` | Use a Markovian composition |
| Interval or posterior on the discretized value | `discretized_scm.point_only` | Use the point and its bound |
| A paradigm the route is closed to | `paradigm_parity.paradigm_not_supported` | Use the paradigm the parity row names |
| Bytes that are truncated, of another version, or whose premises or claimed numbers no longer re-derive | `discretized_scm_artifact.malformed`, `version_unsupported`, `rederivation_refused`, `claimed_numbers_differ`, `identity_mismatch` | Export again from the producing composition |
| A finite-table or non-linear-Gaussian composition exported | `discretized_scm_artifact.not_exportable` | Export from a linear-Gaussian composition |

## Refusals of the finite routes

Refusals raise `cw.CrossWorldExecutionRefusal`, a
[structured refusal](reference/refusal-codes.md#structured-refusals-and-their-remedies):
branch on `reason_code` and `detail`, and read `remedy`.

| Code | When |
| --- | --- |
| `cross_world_not_identified` | mechanisms or the exogenous law are not declared (`cross_world_execution.mechanisms_not_declared`, `cross_world_execution.latent_confounding_not_identified`) |
| `joint_law_required` | a quantity over several worlds with `independent_world_noise` or an undeclared coupling |
| `route_not_supported` | a request outside an executed row (transported execution, incompatible history, soft interventions, fitting from data) or an elimination table above its bound |
| `sampling_dependence_unknown`, `uncertainty_representation_insufficient`, `estimator_inference_mismatch`, `transport_not_certified` | a closed reading; see [readings and calibration](#readings-and-calibration) |

The whole-path uncertainty details (`cross_world_uncertainty.*`) are listed under
[composed spread, whole path, evidence overlap and interface contracts](reference/refusal-codes.md#composed-spread-whole-path-evidence-overlap-and-interface-contracts).

## Related

- [Interventions and worlds](concepts/interventions-and-worlds.md)
- [Fit and query a structural causal model](guides/gcm-estimands.md)
- [Refusals](concepts/refusals.md)

## References

Abduction-action-prediction: Pearl, *Causality* (2nd ed., 2009), chapter 7.
Balke and Pearl, "Counterfactual Probabilities: Computational Methods, Bounds
and Applications", UAI 1994. Pearl, "Direct and Indirect Effects", UAI 2001.
