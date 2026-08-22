# ADR 0009: Parity baselines

- Status: Accepted
- Date: 2026-07-21

## Decision

Pinned reference baselines:

| Project | Pin |
|---------|-----|
| DoWhy | v0.14 at commit `178ecc9c690a02f2801c1f70da2695f5744186cc` |
| Tigramite core fixtures | package `5.2.1.30`, reference commit `5a8768754e6103755b006e9357e21c1a58534927` |
| Tigramite J-PCMCI+/RPCMCI fixtures | package `5.2.9.7`, reference commit `5a8768754e6103755b006e9357e21c1a58534927` |

Parity is capability parity, not Python API parity. Manifests live under
`parity/`.

The package versions above are authoritative because they are the versions
recorded in the frozen fixture environments. The reference commit is retained
as provenance metadata; it is not a claim that both PyPI packages map to that
single source revision.

## Consequences

- Conformance fixtures record the pin, command, and environment used to
 generate reference outputs.
- Upstream pin changes require manifest updates and regression review.
