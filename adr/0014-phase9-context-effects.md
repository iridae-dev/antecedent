# ADR 0014 — Phase 9 context, regimes, effects, and mediation

- Status: Accepted
- Date: 2026-07-21

## Context

DESIGN.md Phase 9 requires J-PCMCI+, RPCMCI, Tigramite effects parity, linear
temporal mediation, and panel/multi-env refinements without stubs.

## Decision

- Separate `JpcmciPlus` / `Rpcmci` types (not PCMCI+ flags).
- Multi-env sample plans share column geometry and lag maps; no sibling series clones.
- Linear temporal mediation is the Phase 9 natural-effect surface; exotic
  nonparametric path-specific ID is deferred.
- Shapley / mechanism-change across regimes → Phase 10; design/state → Phase 11.

## Consequences

`scripts/gate_phase9.sh` gates inventory honesty, conformance pins, and
regime/mediation criterion smokes.
