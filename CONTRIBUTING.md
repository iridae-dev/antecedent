# Contributing

## License

By contributing, you agree that your contributions will be licensed under
**MIT OR Apache-2.0** (dual license). Source files should include:

```text
SPDX-License-Identifier: MIT OR Apache-2.0
```

## Developer Certificate of Origin (DCO)

Every commit must be signed off with a DCO trailer:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Use `git commit -s` (or add the trailer manually). The trailer certifies the
statements in [DCO](DCO). A CLA is not required.

CI rejects commits that lack a valid `Signed-off-by` line.

## Clean implementation

This project is independently implemented from published papers, specifications,
and public behavior. Do **not**:

- copy or translate source, comments, docstrings, tests, or notebooks from
 pinned external baselines;
- translate pinned external baselines code line by line;
- commit upstream GPL source, translated GPL tests, or fixtures with unclear
 redistribution status.

Reference libraries may be executed as black-box comparators in isolated
conformance tooling. Every substantive algorithm needs a machine-readable
provenance record under [`provenance/`](provenance/).

In project docs, "port" means capability parity, not source translation.

## Pull requests

- Implementation PRs cite papers, standards, or independent design notes.
- Feature PRs that touch designated hot paths include benchmarks, allocation
 assertions, and scalar-versus-optimized differential tests where applicable.
- Changing an accepted ADR requires a superseding ADR and migration analysis.

## Toolchain

- Rust **1.85**, edition **2024** (see `rust-toolchain.toml`).
- Format with `cargo fmt`; lint with `cargo clippy -- -D warnings`.
