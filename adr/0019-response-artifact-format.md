# ADR 0019 — causal-response artifact format

- Status: Accepted
- Date: 2026-08-14
- Supersedes: [0018](0018-release-version-history.md) (format-freeze clause only)

## Context

Antecedent 0.5 introduces response curves, derivatives, Jacobians, explicit
observation mechanisms, empirical-support reports, and uncertainty bands. The
0.2 artifact schema cannot durably encode these query and result types. Package
and artifact versions are independent; adding these wire types does not itself
set the package release version.

## Decision

- Advance `antecedent_io::STABLE_FORMAT` from 0.2 to 0.3.
- Encode response queries and results only through explicit versioned wire types,
  including grid dimensions, observation assumptions, structural identification,
  empirical support, uncertainty kind, assumptions, and provenance.
- Continue accepting artifact formats 0.1 and 0.2 as migration sources. The 0.1
  schema upgrade remains in place; 0.2 section payloads pass through unchanged.
- Validate decoded response queries and reject integer-size overflows rather than
  silently truncating them.
- Treat a response envelope as the payload of a partially identified response and
  of nothing else, and let it be *coordinate-free* (`dimension = 0`, empty grid)
  so that the identified set of a scalar or vector functional — a binary-IV ATE
  bound, for instance — is representable. A Jacobian envelope is deliberately
  excluded: the envelope wire carries no outcome/treatment extents, so its rows
  could only be validated against a row-major convention this format has never
  stated. Adding one needs its own wire shape.

## Consequences

Readers targeting format 0.3 can persist the full response contract without
conflating structural identification, empirical support, and statistical
uncertainty. Older artifacts remain readable through the migration registry;
readers frozen at 0.2 must reject new 0.3 artifacts according to the manifest's
minimum-reader version.
