# Compile-time source review for the Linux evidence platforms

This is the review behind `nix/compile-time-reviewed.x86_64-unknown-linux-gnu.json` and
`nix/compile-time-reviewed.aarch64-unknown-linux-gnu.json`. Like the Apple-silicon review
(`../compile-time-audit.md`, `../compile-time-helpers-review.md`,
`../compile-time-suite01-review.md`), it is a scoped source review of locked package
versions, not a proof of compiler, operating-system or subprocess behaviour. Package paths are
relative to the registry archive `Cargo.lock` names.

## What the review has to establish on Linux

The policy `nix-linux-sandbox-audit-v1` requires the daemon to build in the Linux sandbox
(`sandbox = true`, no fallback, no host path but `/bin/sh` mapped into the store). A build
then sees only its derivation's inputs and its own scratch directory, in private mount,
network and process namespaces, so the ambient host files the Apple-silicon review accounts
for (`/private/tmp`) do not exist in it. Two questions remain for each build-script and
procedural-macro unit, and for everything such a unit depends on:

1. Does it read repository text or source positions other than the tokens it is given? A
   unit that does not is `reviewed-safe`; evidence may then bind the items an assertion ran.
2. Does it read the builder's environment in a way the recipes do not fix? The recipes
   (`evidence-common.nix`) set the compiler, scratch directories and Cargo configuration;
   the standard environment sets `CC`, `AR` and `PATH` to store paths.

## Units and how they were resolved

The unit graphs are the ones Cargo resolves for a Linux host, for every scientific and
calibration target and for the extension (`scripts/evidence_platforms.py draft-registry
--platform <target>`). On `aarch64-unknown-linux-gnu` every compile-time unit has the same
package, version, checksum, unit kind, source path and features as a unit of the
Apple-silicon registry. `x86_64-unknown-linux-gnu` has those and seven more, from four
packages that Apple silicon does not build.

A unit shared with Apple silicon is the same source. Its Apple-silicon finding was taken as
the starting point and the unit's sources (the build script and what it includes, or the
whole `src/` of a library or procedural macro) were searched for code conditional on the
host or target operating system, C library, vendor or architecture, and for absolute host
paths. Packages with no such code keep their finding unchanged. The packages with such code
are listed below with what the Linux branch does. None of them reads repository text.

- `cc` 1.4.7: on a Linux host it resolves `CC`/`CXX`/`AR`, or `cc`/`c++`/`ar` on `PATH`,
  and runs them; there is no SDK discovery (`xcrun` is the Apple branch). Its family probe
  and flag checks are written under `OUT_DIR`. The Linux branches in `src/lib.rs` choose
  compiler flags from the target triple. No absolute host path is named outside tests.
- `jobserver` 0.1.35 (`src/unix.rs:75-80,184-197`): creates its pipe with `pipe2`, and
  reopens the descriptors Cargo passes through `/dev/fd/<n>`. A `fifo:` path comes only
  from `CARGO_MAKEFLAGS`, `MAKEFLAGS` or `MFLAGS`; the builder sets none of the last two
  and Cargo sets the first to its own jobserver.
- `libc` 0.2.189 (`build.rs:89-200,243-256`): reads `CARGO_CFG_TARGET_*`, and on
  `linux`/`gnu` the `RUST_LIBC_UNSTABLE_GNU_*` and `CARGO_CFG_LIBC_UNSTABLE_GNU_*`
  switches, which the builder does not set; runs `$RUSTC --version`. `freebsd-version`
  and `emcc` are run only for those targets.
- `proc-macro2` 1.0.107 (`build.rs:229-241`) and `thiserror` 2.0.20 (`build.rs:160-172`):
  on Linux a probe-directory cleanup that fails with `ENOTEMPTY` is tolerated. Nothing
  else differs.
- `blake3` 1.8.7: on `x86_64` it assembles its own `c/blake3_{sse2,sse41,avx2,avx512}_x86-64_unix.S`
  through `cc`; on `aarch64` it compiles its own `c/blake3_neon.c`, as on Apple silicon.
  The Windows and MSVC branches are not taken.
- `pulp` 0.22.3 (`build.rs:193-`): on `x86_64` it writes `OUT_DIR/x86_64_asm.rs`,
  generated from constants in the script and its own `CARGO_PKG_VERSION_*`; on `aarch64`
  it returns before writing.
- `portable-atomic` 1.15.0, `crossbeam-utils` 0.8.23, `crossbeam-deque` 0.8.8,
  `crossbeam-epoch` 0.9.21, `getrandom` 0.3.4, `cubecl-common` 0.4.0, `zstd-safe` 7.4.0 and
  8.2.0, `ahash` 0.8.12, `matrixmultiply` 0.3.11, `serde_json` 1.0.151: the conditional
  code reads `TARGET` and `CARGO_CFG_TARGET_*` and prints Cargo directives.
- `zstd-sys` 2.1.2+zstd.1.5.7: compiles the bundled zstd sources of its own package through
  `cc`. `pkg-config` 0.3.34 is called only with the `pkg-config` feature or
  `ZSTD_SYS_USE_PKG_CONFIG`; neither is set, so its host search paths are never consulted.
- `find-msvc-tools` 0.1.13: the lookup is `cfg(windows)`-gated and non-MSVC targets return
  before it.
- `paste` 1.0.15: its `env!` modifier reads a variable named by its input. No package in
  either Linux graph, and no workspace crate, invokes it; the only uses are in `paste`'s own
  tests.
- `target-lexicon` 0.13.5: the matches are its table of target triples. The build script
  includes its own sources, writes `OUT_DIR/host.rs` from `TARGET` and runs
  `$RUSTC --version`.
- `pyo3-build-config`, `pyo3-ffi`, `pyo3`, `numpy` and the workspace `python/build.rs` find
  and run the host Python and a linker (through `cc` or `which` on Linux). They stay
  `complete-source`, as on Apple silicon: evidence that selects them keeps complete source,
  and no scientific target selects them.

The units `x86_64-unknown-linux-gnu` adds:

- `private-gemm-x86` 0.1.22 (`build.rs`, read for every environment, file and process
  call): reads `OUT_DIR` only and writes `OUT_DIR/asm.s` and `OUT_DIR/asm.rs`, generated
  from constants in the script.
- `defer` 0.2.1 (`src/lib.rs`, whole file): a `no_std` scope guard used by that script.
- `interpol` 0.2.1 (`src/lib.rs`, whole file): format-string interpolation macros that
  parse their input tokens with `syn` and re-emit `std::format!`-family calls. No file,
  environment, process, time or randomness input.
- `syn` 1.0.109, used by `interpol`: `build.rs` runs only `$RUSTC --version`; in `src/` the
  filesystem and environment matches are the `syn::File` syntax-tree type and an `include!`
  of its own `src/await.rs`.

## What this review does not cover

It does not show that the units' output is independent of time, entropy or the kernel's
`/proc`, `/sys` and `/dev`, which the sandbox leaves visible. A new recipe or lock is rebuilt
once on Linux and compared bit for bit before its build is used, which detects such a
dependence in the binaries without excluding it. The local Nix daemon, the pinned nixpkgs
and its signed substitutes, the toolchain archives and the registry sources are trusted, as
the registries' `scope` states.

A package, version, feature set or unit kind without a row refuses, as does a row left
`unresolved`. After a `Cargo.lock` change, `scripts/evidence_platforms.py draft-registry
--all` adds the new units of every platform as `unresolved`; each is then read as above
before its state is set.
