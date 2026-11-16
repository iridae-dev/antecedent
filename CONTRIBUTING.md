# Contributing

This page takes you from a clone to a pull request that CI accepts: what to install, how to
build, which checks to run for which kind of change, and how commits are signed.
[Development](docs/development.md) is the full reference for every gate, the record procedures and
the release procedure; [test performance](docs/guides/test-performance.md) covers focused test runs.

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

CI rejects a pull request in which any non-merge commit lacks a valid `Signed-off-by` line. To
refuse such a commit locally, install the repository's commit-message hook:

```bash
git config core.hooksPath scripts/githooks
```

## Prerequisites

| Tool | Needed for |
| --- | --- |
| Rust **1.97**, edition **2024** | Everything. `rust-toolchain.toml` pins the channel with `rustfmt` and `clippy`, so `rustup` installs it on first use. |
| [`cargo-nextest`](https://nexte.st/docs/installation/) | Every Rust test lane of `scripts/test.sh`. |
| [`uv`](https://docs.astral.sh/uv/) and CPython 3.11 to 3.14 | The Python package, its tests and lint, and the script tests. |
| [Nix](https://nixos.org/download/) | Only for earning scientific and calibration records (see [below](#scientific-and-calibration-records)). |

## Build and test

From the repository root:

```bash
cargo check --workspace                                  # compile the Rust workspace
bash scripts/test.sh dev -p <crate> --test <target>      # one Rust test target (or --lib), cheap compilation
bash scripts/test.sh python                              # sync the locked environment, build the extension, run pytest
bash scripts/test.sh python-test tests/<file>.py -q      # pytest against the extension already built
bash scripts/test.sh engineering                         # every engineering test
```

`bash scripts/test.sh engineering` runs the script tests, the Rust tests and doctests, the
optional-feature tests and the Python suite. It runs no scientific assertion and measures nothing.
`bash scripts/test.sh --help` lists every lane.

The Python extension always builds with Cargo's release profile; see
[native extension builds](docs/development.md#native-extension-builds).

## Which checks to run

Run the narrowest check that covers your change, then let CI run the rest. Each row names what CI
would otherwise fail on.

| You changed | Run locally |
| --- | --- |
| Rust code | `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, then `bash scripts/test.sh dev -p <crate> --test <target>` (or `--lib`) |
| A public item of a published crate | `bash scripts/gate_public_api.sh --write`, and commit the snapshot diff (see [Public API changes](#public-api-changes)) |
| Python code or stubs | `bash scripts/gate_python_lint.sh` (Ruff, Ruff format, mypy), then `bash scripts/test.sh python-test <tests>`; after a Rust change, `bash scripts/test.sh python` rebuilds the extension first |
| A script under `scripts/` | `bash scripts/test.sh scripts` |
| A registry under `parity/` or `provenance/`, or a generated page | `bash scripts/gate_release.sh --section static` (source reading only, no build) |
| Documentation or examples | The commands in [Documentation and examples](#documentation-and-examples) |
| Anything a scientific or calibration record depends on | `bash scripts/test.sh` (see [Scientific and calibration records](#scientific-and-calibration-records)) |

`bash scripts/gate_release.sh` runs the whole pull-request gate in the order CI does: static
checks, builds, engineering tests, then record currency. A local full run is optional; CI runs
each section as its own job.

### Formatting and lint

Format with `cargo fmt --all` from the repo root, or `cargo fmt -p <crate>` for one crate. Do not
invoke `rustfmt` directly and do not pass `--edition`: `rustfmt.toml` sets edition 2024, and
`--edition` overrides it and rewrites imports. Format only crates you changed. CI checks
`cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings`.

## Documentation and examples

The documentation site is built from `docs/` with MkDocs. With the matching Antecedent installed
in your environment:

```bash
python -m pip install -r requirements-docs.txt
python scripts/generate_example_catalog.py --check    # the example catalog is current
python scripts/generate_doc_release_links.py --check  # repository links are pinned to the release tag
python scripts/check_docs_experience.py
python scripts/example_coverage.py --check            # every catalogued example source is selected by a runner
python scripts/run_doc_examples.py                    # executes documented snippets and embedded examples
python scripts/render_notebooks.py                    # executes the notebooks in isolated kernels
mkdocs build --strict
```

`python3 scripts/check_doc_links.py` (part of the static gate) checks that relative links and
pinned repository links resolve. Several pages are generated or hold generated blocks; never edit
those by hand. [Documentation sources](docs/README.md) lists them and the editorial conventions.

## Scientific and calibration records

A supported method is backed by recorded passes of its scientific assertions
(`parity/scientific_records.toml`) and, where it reports an interval, by measured coverage
records (`parity/coverage_records.toml`). Scientific assertions and calibration measurements run
on development machines, never in CI. A change to what a recorded assertion or coverage record
depends on owes that record, and CI fails until it is current again.

Before you push, run `bash scripts/test.sh` on a clean checkout of your commit: it runs the
engineering tests, then re-earns only the owed scientific and calibration records. Commit the
records it rewrites with the change. To see what a change owes without running anything:

```bash
python3 scripts/scientific_records.py status
python3 scripts/calibration_status.py status
```

CI and the release gate only check that every record is current
(`python3 scripts/scientific_records.py status --check`,
`python3 scripts/calibration_status.py --check`) and never re-run either lane. Records are earned
in audited Nix builds on a reviewed platform; `python3 scripts/evidence_platforms.py status`
reports whether your machine can earn. See
[Recorded scientific tests](docs/development.md#recorded-scientific-tests),
[Earning platforms](docs/development.md#earning-platforms) and
[Coverage records](docs/development.md#coverage-records) for the procedures.

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
- Feature PRs that touch designated [hot paths](docs/hot_paths.md) include benchmarks, allocation
  assertions, and scalar-versus-optimized differential tests where applicable.
- Changing an accepted [ADR](adr/README.md) requires a superseding ADR and migration analysis.
- Every commit carries the DCO sign-off, and the records the change owes are committed with it.
