# ADR 0012: Native compiled SCM with overlay sampling

## Status

Accepted.

## Context

GCM fits mechanisms on a graph and samples under interventions and
counterfactual worlds. A natural port clones the SCM per intervention or walks
the semantic graph per draw. forbid both on the hot path.

## Decision

- Compile each DAG once into a `CompiledCausalModel` (topo order + parent gather
 plans + mechanism store).
- Apply interventions as immutable `InterventionOverlay` views; never clone the
 SCM per world or draw.
- Dispatch mechanisms via monomorphized / enum kernels, not trait-object-per-scalar.
- Counterfactuals abduce exogenous noise once, then reuse it across action
 overlays (AAP), recording `NoiseInferenceKind` on results.
- Exogenous noise is retained in a `NoiseTable` and replayed through the same compiled
  model under an overlay; the per-node precedence is shared with forward sampling, so
  replay never clones the SCM per draw or coalition.
- Nonlinear mechanism families are typed `MechanismSlot` variants. Tree ensembles fit
 through the learner contract and are stored as a provider-independent tree map, so the
 hot path stays an enum kernel and a saved model evaluates without any ML provider. An
 additive-noise family keeps exact residual abduction; a categorical family uses the
 `U(0,1)` inverse CDF with posterior abduction. Candidate sets, per-node constraints,
 declared validation splits and the resulting scores are part of the assignment record.
- Programmable interventions are typed value expressions (`ValueExpr`) attached to a node
 as a transform of its generated value or as an assignment replacing its mechanism. They
 compile to postfix column programs stored on the overlay and run by one executor inside the
 per-node precedence, so forward sampling, conditional sampling, the do-samplers, noise
 replay and counterfactual evaluation share their meaning. Reads of other variables join the
 graph's parent edges to fix the evaluation order; a cycle, an unknown or undeclared read, a
 type that does not fit the node and conflicting directives are coded refusals.
- Shapley / distribution-change attribution live in the attribution inventory
 (`parity/attribution.toml`).

## Consequences

Batch simulation and CF benches exercise overlay reuse. Streaming CF summaries
must match retained-draw aggregates (tested by comparing an outcomes-only result with the full-retention column).
