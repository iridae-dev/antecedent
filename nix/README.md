# Nix evidence build boundary

Scientific Rust evidence is earned by `scientific_records.py run`, whose default
`--rust-build-backend nix` builds each Cargo target from the clean committed
`HEAD` in the local Nix sandbox and runs that exact binary. The point of the
boundary: a change to code a claim executed (or to anything that can affect
its build) makes the claim owed again, while an unrelated change leaves it
current, so CI only checks that records are up to date.

## What is pinned and checked

- `flake.lock` pins nixpkgs; `evidence-toolchain.nix` pins Rust 1.97.1, its LLVM
  tools and `rustc-dev` (for the compiler-inventory driver) by SHA-256 for
  `aarch64-darwin`. Cargo comes from the locked nixpkgs. Crates come from
  `importCargoLock`, checksum-verified against `Cargo.lock`; this production
  vendor is the one whose unit graph is validated.
- `nix_evidence_source.py` copies the committed tree, imports it into the store
  and compares NAR hashes; the checkout must stay clean except for this run's
  own content-verified ledger and manifests.
- `evidence-target.nix` builds every owed target in one derivation. Each target
  runs its own `cargo test --no-run`, exactly as it would alone, in sequence over
  one shared target directory, so dependencies compile once per feature set; a
  single Cargo invocation over several targets would unify their features and is
  never used. Each target keeps its own executable, metadata and unit graph under
  `targets/<crate>/<kind>/<name>`.
- The build uses a private scratch tree, the pinned `rustc`,
  a stable `--remap-path-prefix` (so `file!()` and panic locations do not depend
  on the store hash), coverage instrumentation, and the analyzer driver as
  workspace wrapper to inventory every workspace item (one shared inventory).
- `nix_evidence_policy.validate_build` requires the daemon's `sandbox = true`
  and `sandbox-fallback = false` with only the reviewed sandbox paths, then checks
  every build-script and procedural-macro unit (and their dependencies) in the
  emitted graph against `compile-time-reviewed.json`, which is bound to the
  `Cargo.lock` hash. A new package, version, feature set or unit kind refuses.
- The manifest records source commit and NAR hash, derivation output, binary
  SHA-256, toolchain versions and the selected audit rows. Publication rechecks
  all of them after execution.

## Compile-time review and item evidence

`compile-time-reviewed.json` covers the union of compile-time units of every
scientific Rust target and of the Python extension. Each row has a
`source_projection` state:

- `reviewed-safe`: reads nothing that selects repository text. Procedural
  macros read only their input tokens and spans.
- `complete-source`: the unit has assessed host inputs, such as the Python
  interpreter and linker that PyO3 and `python/build.rs` find. Evidence that
  depends on it keeps complete source.
- `unresolved`: not assessed. The Nix backend refuses it.

When every selected unit is `reviewed-safe`, the evidence reads the items an
assertion depends on rather than whole sources (`scripts/evidence_items.py`). The
analyzer driver runs as the workspace compiler wrapper and records each item's
dependencies, name lookups and method resolutions; with execution coverage, earning
selects the executed functions whole and follows those dependencies. Positions are
not inputs: they reach a binary through panic locations and debug information,
which a passing assertion does not read, except in an assertion whose executed code
observes them, which keeps its files whole. Bodies inside a function-like macro's
input are part of the invocation, which is recorded whole; an item under a custom
attribute or a derive that may read documentation is hashed with its doc comments.

## Python and calibration

Scientific claims are Rust tests; `evidence_audit` refuses a Python scientific
row. Python tests check that the binding returns the Rust result and run in the
engineering lane. Calibration snapshots build on the host: the same registry is
applied to the host unit graph (`host_compiler_graph`), with complete source
retained.

## Trust assumptions

The local Nix daemon and its configuration, signed binary-cache substitutes for
the tool closure, the macOS system runtime and frameworks, and the reviewed
registry and toolchain sources are trusted. Under the Darwin sandbox,
`/private/tmp` stays readable; the source review in `compile-time-audit.md`,
`compile-time-suite01-review.md` and `compile-time-helpers-review.md` accounts for
that exposure. Disposable fixtures (`nix_evidence_engineering_fixture.py`,
`nix_python_engineering_fixture.py`, `probe_nix_sandbox.py`) exercise the
mechanics without earning records.
