# ADR 0002: Artifact encoding

- Status: Accepted
- Date: 2026-07-21
- Design: DESIGN.md §35.2, §24

## Decision

Canonical durable artifacts use:

- CBOR for semantic metadata;
- Arrow IPC for large arrays;
- a sectioned, versioned container with BLAKE3 checksums;
- optional Zstandard compression.

JSON is for debugging and interchange, not the canonical durable encoding.
Internal Rust structs are not serialized directly; versioned wire types mediate.

## Consequences

- `causal-io` owns the container format and wire types.
- Schema migrations are explicit and versioned.
