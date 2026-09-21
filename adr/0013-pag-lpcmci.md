# ADR 0013: PAG / LPCMCI typed graph classes

## Status

Accepted.

## Context

Latent confounding requires ADMGs and PAGs. Silent coercion of PAGs into DAGs
breaks identification soundness.

## Decision

- Distinct `Admg`, `Pag`, `TemporalPag` types with endpoint validation.
- m-separation via ancestral moralization (ADMG). On a PAG the answer is about
 every member MAG: definite-status paths (Zhang 2008, including unshielded
 circle-circle non-colliders) decide what they can, the enumerated completions
 decide the rest, and a class that cannot be settled is `Undetermined`, never
 separated.
- Completions streamed under `max_completions` (no unbounded retain).
- Identification over PAGs uses `GeneralizedAdjustmentIdentifier` +
 `IdentificationEnvelope` with explicit unidentified mass. Single-treatment
 responses additionally run Shpitser–Pearl ID on each valid MAG completion with
 its invisible directed edges read as latent-confounded; this is sound, not
 complete (IDP is not implemented). Every input, circle-free or not, must
 complete to a maximal ancestral graph.
- LPCMCI is its own type returning `TemporalPag`; discriminating paths and rule
 scheduling are separate modules.
- Planner rejects DAG-only identifiers on PAG inputs.

## Consequences

Exit criteria gated by `scripts/gate_pag.sh`. Shapley attribution remains
Attribution inventory; J/RPCMCI live under context / pinned baseline.
