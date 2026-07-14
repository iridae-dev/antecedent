# ADR 0012: Native compiled SCM with overlay sampling

## Status

Accepted (Phase 7).

## Context

DoWhy-GCM fits mechanisms on a graph and samples under interventions and
counterfactual worlds. A natural port clones the SCM per intervention or walks
the semantic graph per draw. DESIGN.md §15.6 / §16.1 forbid both on the hot path.

## Decision

- Compile each DAG once into a `CompiledCausalModel` (topo order + parent gather
  plans + mechanism store).
- Apply interventions as immutable `InterventionOverlay` views; never clone the
  SCM per world or draw.
- Dispatch mechanisms via monomorphized / enum kernels, not trait-object-per-scalar.
- Counterfactuals abduce exogenous noise once, then reuse it across action
  overlays (AAP), recording `NoiseInferenceKind` on results.
- Shapley / distribution-change attribution remain Phase 10
  (`parity/phase7_deviations.md`).

## Consequences

Batch simulation and CF benches exercise overlay reuse. Streaming CF summaries
must match retained-draw aggregates (`streaming_matches_retained`).
