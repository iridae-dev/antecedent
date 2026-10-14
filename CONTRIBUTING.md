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

## Public API changes

The public surface of every published crate is snapshotted in
`parity/public_api/<crate>.txt` and checked by `scripts/gate_public_api.sh`. A
change that adds, removes or re-types a public item fails the gate until you
regenerate the snapshots with `bash scripts/gate_public_api.sh --write` and
commit the diff, so API changes are always visible in review. Prefer
`pub(crate)` for items only the crate itself uses.

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
- Format with `cargo fmt --all` from the repo root, or `cargo fmt -p <crate>` for one crate. Do not invoke `rustfmt` directly and do not pass `--edition`: `rustfmt.toml` sets edition 2024, and `--edition` overrides it and rewrites imports. Format only crates you changed. Lint with `cargo clippy -- -D warnings`.

For focused Rust edits, use `bash scripts/test.sh dev -p <crate> --test <target>`
or `--lib`. Python-only edits can use `bash scripts/test.sh python-test` against
the current native extension. Run optimized numerical tests with
`bash scripts/test.sh`; use
`bash scripts/test.sh all` for Rust tests, doctests, optional feature tests and
the Python suite. See [test performance](docs/guides/test-performance.md) for
focused runs, build settings and timing reports.
