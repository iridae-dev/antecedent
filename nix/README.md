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
  workspace wrapper to inventory ordinary function bodies (one shared inventory).
- `nix_evidence_policy.validate_build` requires the daemon's `sandbox = true`
  and `sandbox-fallback = false` with only the reviewed sandbox paths, then checks
  every build-script and procedural-macro unit (and their dependencies) in the
  emitted graph against `compile-time-reviewed.json`, which is bound to the
  `Cargo.lock` hash. A new package, version, feature set or unit kind refuses.
- The manifest records source commit and NAR hash, derivation output, binary
  SHA-256, toolchain versions and the selected audit rows. Publication rechecks
  all of them after execution.

## Compile-time review and precise evidence

`compile-time-reviewed.json` covers the union of compile-time units of every
scientific Rust target and of the Python extension. Each row has a
`source_projection` state:

- `reviewed-safe`: reads nothing that selects repository text. Procedural
  macros read only their input tokens and spans.
- `complete-source`: the unit has assessed host inputs, such as the Python
  interpreter and linker that PyO3 and `python/build.rs` find. Evidence that
  depends on it keeps complete source.
- `unresolved`: not assessed. The Nix backend refuses it.

When every selected unit is `reviewed-safe`, unexecuted function bodies become
holes in the evidence projection (`scripts/evidence_projection.py`). Such a hole
keeps the exact line count, last-line bytes and last-line characters of the body,
so every later line and column is unchanged. Rust embeds source locations in
generated code far beyond `line!()`. Examples are `#[track_caller]` calls such as
`unwrap` and indexing, overflow and bounds panics, and macros that read spans.
Lexical analysis cannot rule these out, so no hole may move later source. Edits
that change only the inside of an unexecuted body keep a claim current. Edits
elsewhere in the file, or ones that change its layout, make it owed. A body is
never a hole when it:

- contains a macro, item or attribute,
- lies inside a function-like macro's input,
- is under a custom attribute macro.

A custom derive scopes only to its own data item and helper attributes.

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
