# ADR 0014 — context, regimes, effects, and mediation

- Status: Accepted
- Date: 2026-07-21
- Updated: 2026-07-23

## Context

 requires J-PCMCI+, RPCMCI, pinned baseline effects parity, linear
temporal mediation, and panel/multi-env refinements without stubs.

## Decision

- Separate `JpcmciPlus` / `Rpcmci` types (not PCMCI+ flags).
- Multi-env sample plans share column geometry and lag maps; no sibling series clones.
- Linear temporal mediation remains the natural-effect *estimator* surface.
- Nonparametric path-specific **identification** ships via
  `PathSpecificIdentifier` (recanting + surgery + general ID) plus
  `functional.effect`; GCM `path_decompose` contribution stays linear-Gaussian.
- Unsupervised RPCMCI regime search is out of scope; callers supply
  `RegimeAssignment` (optional alternating refinement of those labels).
- Shapley / mechanism-change across regimes → ; design/state → .

## Consequences

`scripts/gate_context.sh` gates inventory honesty, conformance pins, and
regime/mediation criterion smokes.
