# ADR 0007: Supported Rust and Python versions

- Status: Accepted
- Date: 2026-07-21

## Decision

- Rust **1.85**, edition **2024** (pinned in `rust-toolchain.toml`).
- First public Python release targets CPython **3.11 through 3.14**.

## Consequences

- CI and wheel builds use these version floors.
- MSRV bumps require an ADR update and release notes.
