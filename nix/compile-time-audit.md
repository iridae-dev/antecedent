# Partial compile-time source audit

This is a focused source review, **not a completed audit of the scientific
build boundary**. It records what selected high-impact packages can read or
execute, so those inputs can be constrained and checked before evidence is
earned. Package source references below are relative to the corresponding
Cargo registry archive named in `Cargo.lock`; line numbers refer to the locked
versions reviewed. No scientific or calibration tests were run in this review.

## Scope and active features

The local `aarch64-apple-darwin` default-feature `antecedent` graph was
inspected with `cargo tree --locked --offline --target aarch64-apple-darwin`.
It includes `blake3` 1.8.7, `cc` 1.4.7 and `zstd-sys` 2.1.0+zstd.1.5.7.
The observed `blake3` features were `default` and `std`; `zstd-sys` enabled
`legacy`, `std`, and `zdict_builder`, but not `pkg-config`, `bindgen`, or
`cmake`. The `antecedent-py` graph includes PyO3 and `pyo3-build-config`
0.29.2; the default Rust facade graph does not include PyO3. These are
observations for those commands, not proof of the feature graph for every
scientific target or Python build. Each actual target's selected features,
build and development dependencies still need verification.

## Reviewed paths and conditions

* **blake3 1.8.7:** `build.rs:281–287,366–385` compiles the vendored NEON C
  source for the default aarch64 configuration and enumerates the package's
  `c` directory. It delegates native compilation to `cc`. The reviewed
  build script has no literal `/private/tmp` path; its source directory,
  target configuration, compiler and environment remain inputs.
* **zstd-sys 2.1.0+zstd.1.5.7:** `build.rs:245–279` enumerates and sorts C
  files from vendored zstd directories for its bundled build. Despite the
  observed features above, `build.rs:433–461` switches to host `pkg-config`
  whenever `ZSTD_SYS_USE_PKG_CONFIG` is present. That switch can select
  external libraries or header directories, including paths under a host
  temporary directory. It must be absent in the actual Nix build environment,
  or its full effects must be explicitly assessed and bound. The host earning
  preflight separately rejects this variable; it does not attest the Nix
  builder's environment.
* **cc 1.4.7:** `src/tool.rs:204–243` uses `OUT_DIR`, or the process temporary
  directory as a fallback, for a compiler-family probe; `src/tempfile.rs`
  creates randomized names with exclusive creation and mode 0600 on Unix.
  `src/lib.rs:1499–1504` reuses an existing `flag_check.c` in `OUT_DIR` rather
  than rewriting it. A fresh private build target and `OUT_DIR` are therefore
  necessary, and file ownership and isolation must be verified. The helper
  invokes external compilers and may resolve tools, flags, and SDKs using
  `PATH`, compiler-related variables, and Apple SDK discovery. It also reads
  target-specific `HOST_`/`TARGET_` variants (`src/lib.rs:4296–4327`). Its
  temporary-name randomness is observed; this review does not prove that all
  downstream compiler output is independent of paths or entropy.
* **PyO3 0.29.2:** `pyo3-build-config/src/impl_.rs:675–703` can read an
  arbitrary absolute `PYO3_CONFIG_FILE`; `:2003–2165` parses and searches
  cross-configuration directories; `:2493–2515,2573–2600` executes a
  configured, virtual-environment, or `PATH`-selected Python interpreter.
  The workspace's `python/build.rs` uses that interpreter configuration and
  locates a native linker to construct the Python test link and rpath. A pinned
  interpreter, its libraries/configuration, linker, `PATH`, and allowed
  `PYO3_*` and Python variables must be established for Python evidence.
  Searching the PyO3 macro and build sources found no literal `/private/tmp`
  path; caller-selected paths and interpreter behavior are not covered by
  that search.

## Required boundary and remaining work

The prototype `evidence-target.nix` places `HOME`, `CARGO_HOME` and
`CARGO_TARGET_DIR` below the Nix build's `TMPDIR`, and configures Cargo to use
vendored sources offline. Its intended clean Nix builder environment must be
measured and verified: the host preflight checks the caller's environment and
cannot establish what the Nix builder actually receives. In particular, prove
that unapproved compiler, SDK, pkg-config, Python and feature overrides cannot
enter the build. Verify the identity of the vendored packages, compiler,
linker, SDK, system libraries and interpreter, and ensure that the build's
scratch directories, including `OUT_DIR`, are newly created and private.

The macOS sandbox probe showed that an unrelated host file under `/private/tmp`
was readable, even though the workspace sentinel and loopback were denied.
Source review of these selected packages does not remove that exposure: other
build scripts, procedural macros, transitive helper crates, compilers,
interpreters, dynamically selected paths, time and entropy remain unreviewed
or unbound. The full per-target graph and all remaining compile-time packages
and macro invocation sites have **not** been audited. Evidence earning must
remain fail-closed until the complete boundary, its verification, and the
binding between the resulting artifact and its manifest are implemented and
tested.
