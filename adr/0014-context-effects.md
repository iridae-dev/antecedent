# ADR 0014 — context, regimes, effects, and mediation

- Status: Accepted
- Date: 2026-07-21

## Context

DESIGN.md requires J-PCMCI+, RPCMCI, pinned baseline effects parity, linear
temporal mediation, and panel/multi-env refinements without stubs.

## Decision

- Separate `JpcmciPlus` / `Rpcmci` types (not PCMCI+ flags).
- Multi-env sample plans share column geometry and lag maps; no sibling series clones.
- Linear temporal mediation is the natural-effect surface; exotic
  nonparametric path-specific ID is deferred.
- Shapley / mechanism-change across regimes → ; design/state → .

## Consequences

`scripts/gate_context.sh` gates inventory honesty, conformance pins, and
regime/mediation criterion smokes.
