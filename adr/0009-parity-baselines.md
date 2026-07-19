# ADR 0009: Parity baselines

- Status: Accepted
- Date: 2026-07-21

## Decision

Pinned reference baselines:

| Project | Pin |
|---------|-----|
| DoWhy | v0.14 at commit `178ecc9c690a02f2801c1f70da2695f5744186cc` |
| Tigramite | tag `5.2.1.25` at commit `5a8768754e6103755b006e9357e21c1a58534927` |
| Tigramite extended | commit `ff3ff13e1481073b8c5833a6fde1c304627a208e` for post-release features |

Parity is capability parity, not Python API parity. Manifests live under
`parity/`.

## Consequences

- Conformance fixtures record the pin, command, and environment used to
 generate reference outputs.
- Upstream pin changes require manifest updates and regression review.
