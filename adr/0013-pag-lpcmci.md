# ADR 0013: PAG / LPCMCI typed graph classes

## Status

Accepted.

## Context

Latent confounding requires ADMGs and PAGs. Silent coercion of PAGs into DAGs
breaks identification soundness (DESIGN.md §2 / §21.2).

## Decision

- Distinct `Admg`, `Pag`, `TemporalPag` types with endpoint validation.
- m-separation via ancestral moralization (ADMG) and definite-status paths (PAG).
- Completions streamed under `max_completions` (no unbounded retain).
- Identification over PAGs uses `GeneralizedAdjustmentIdentifier` +
 `IdentificationEnvelope` with explicit unidentified mass.
- LPCMCI is its own type returning `TemporalPag`; discriminating paths and rule
 scheduling are separate modules.
- Planner rejects DAG-only identifiers on PAG inputs.

## Consequences

Exit criteria gated by `scripts/gate_pag.sh`. Shapley attribution remains
Attribution inventory; J/RPCMCI live under context / pinned baseline.
