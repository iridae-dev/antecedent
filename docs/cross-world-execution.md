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
| fixed-population two-slice temporal counterfactual | executed, staged with abduction reuse |
| transported static additive linear path-specific | keeps its own theorem-scoped route |
| transported cross-world execution | closed: `cell_not_licensed`, prerequisites named |
| semi-Markovian / nonparametric ADMG | closed: `cross_world_not_identified` |
| incompatible history, soft interventions, fitting a finite SCM from data | closed |

## The finite Markovian row

Variables are finite and listed topologically, each with its own independent
finite exogenous term (a term shared by two variables is latent confounding and
is refused). Each unit's abduction enumerates the joint exogenous states
exactly, keeps those consistent with the evidence and normalizes; every world
evaluates the same equations with its hard interventions and routes under the
same posterior state, so worlds are coupled by construction (Pearl 2009,
chapter 7; Balke and Pearl 1994). Nothing is sampled: `shared_draws` is zero.

Whether draws may be paired is decided only by the joint-law rule
(`check_propagation`): a quantity over several worlds needs the aligned
exogenous states (`SharedAbducedExogenous`); independently drawn world noise or
an undeclared coupling is refused with `joint_law_required`. A mean contrast of
single-world marginals needs no coupling.

Bounds (registered in `parity/limits.toml`):

- Variables: 12
- Worlds: 8
- Units: 1024
- Joint exogenous states: 65536 per unit

## Readings and calibration

Every result is `point_only` from exact enumeration of a declared model; model
adequacy is a premise, not a check. Interval and posterior readings are closed
with `cell_not_licensed`, and so is a sampled posterior
(`ExogenousInference::PosteriorSampling`).

Calibration design recorded for opening a sampled or interval route (not
measured): the whole abduction-action-prediction path must be measured, not the
components. Method: seeded posterior draws of the exogenous terms given factual
evidence, world prediction per draw, interval over the contrast. Data-generating
process: finite Markovian SCMs with response-type exogenous terms and unit-level
evidence. Nominal level 0.95. Support range: 2 to 4 levels, up to 3 variables,
posterior masses from 0.05 to 0.95. Probes: weak and strong evidence, rare
posterior states, near-deterministic mechanisms. Until a measured record exists
the route stays closed.

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
