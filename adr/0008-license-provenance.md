# ADR 0008: License and provenance

- Status: Accepted
- Date: 2026-07-21
- Design: DESIGN.md §35.8, §27

## Decision

- Dual license: **MIT OR Apache-2.0**.
- Contributions require Developer Certificate of Origin (DCO) sign-off.
- No CLA initially.
- Machine-readable algorithm provenance under `provenance/`.
- Clean-implementation rules: no source translation from pinned baseline or
 line-by-line pinned baseline ports.

## Consequences

- SPDX headers on source files.
- CI enforces DCO trailers.
- Substantive algorithms ship with provenance TOML records.
