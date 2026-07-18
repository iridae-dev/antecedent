# ADR 0015 — attribution and change explanation

- Status: Accepted
- Date: 2026-07-21

## Context

DESIGN.md requires distribution/unit/mechanism-change attribution,
Shapley and path decompositions, robust variants, posterior contribution
blocks, and graph-sensitive root-cause ranking with GCM parity.

## Decision

- Expand `causal-attribution` (no new crate); keep anomaly APIs.
- Shapley exact methods enforce `max_exact_components`; MC modes always report
  budget + stderr; coalition evaluations use a semantic cache under
  `ExecutionContext.cache_policy`.
- Mechanism-change *detection* is a separate API from attribution.
- Robust attribution uses regression hybrids (pinned baseline `distribution_change_robust`
  parity) rather than full density estimation.
- Facade + Python surfaces bind the same Rust entry points as GCM.

## Consequences

`scripts/gate_attribution.sh` gates inventory honesty, conformance pins, and
Shapley criterion smokes. `parity/gcm.toml` Shapley / distribution-change rows
are `done`.
