# Build-helper review for the Darwin `suite_01` audit

This records the practical `/private/tmp` review of the 14 additional locked
package versions reachable from the 39 build-script/proc-macro package versions
in the `suite_01` compiler-unit graph. It is a scoped source review, not a proof
of all compiler, OS, subprocess, or macro-expansion behavior. The local Nix
daemon, pinned/signed-cache toolchain and dependencies, and macOS runtime are
trusted as described in `compile-time-reviewed.json`.

The Cargo 1.97 Nix metadata-only graph, generated using the verified offline
`cargo vendor` subset, has the same 86 distinct package/version, unit
kind/mode, source path, feature and platform signatures in the compile-time
closure as the reviewed local Cargo 1.85 graph. This does not prove byte-for-byte
identity of that diagnostic vendor and the production Nix `importCargoLock`
vendor. Production builds must use locked checksum-verified source inputs and
validate their emitted unit signatures against the registry.

- `autocfg` 1.5.1 (`src/lib.rs`, `src/rustc.rs`) uses `OUT_DIR` and a Rust
  compiler subprocess; it consults wrapper/target/flag variables and creates
  compiler probes. Its direct paths and nondeterministic probe names are
  described in `compile-time-suite01-review.md`.
- `cc` 1.4.7 (`src/lib.rs:1499-1553`, `src/tool.rs:204-243`,
  `src/tempfile.rs`) selects compiler tools and writes probe files under
  `OUT_DIR`, or the process temporary directory if unavailable. It can reuse
  `flag_check.c`; fresh private build directories and sanitized compiler and
  temporary-directory variables are required. See `compile-time-audit.md`.
- `find-msvc-tools` 0.1.13 (`src/find_tools.rs:150-159,322+`) is Windows-tool
  discovery: non-MSVC target strings return before that lookup, and its Windows
  implementation is cfg-gated. Its environment/process/file paths were searched;
  this conclusion is specific to the audited Darwin target.
- `jobserver` 0.1.35 (`src/lib.rs:254-275`, `src/unix.rs:102-127`) can open
  a FIFO at a path specified by `CARGO_MAKEFLAGS`, `MAKEFLAGS`, or `MFLAGS`.
  The Nix build must clear inherited forms before starting Cargo, so Cargo's
  own jobserver descriptors are used; otherwise arbitrary host temporary files
  could be named. Its `/dev/fd` reopening branch is Linux-only.
- `pkg-config` 0.3.34 (`src/lib.rs:629-740`) selects a program and search paths
  from target-specific or generic environment variables and executes it. The
  audited Darwin zstd path uses bundled sources when `ZSTD_SYS_USE_PKG_CONFIG`
  is absent; Nix must keep this and the `PKG_CONFIG_*` overrides out of the
  build environment. Executing a different active pkg-config caller requires
  separate review of its allowed files and program.
- `rustc_version` 0.4.1 (`src/lib.rs:200-212`) and `version_check` 0.9.5
  (`src/lib.rs:159-164,302-342`) execute the configured Rust compiler and can
  use wrapper/flag information. The Nix build must pin `RUSTC` and exclude
  unapproved wrappers and flags.
- `nano-gemm-codegen` 0.2.1 (`src/lib.rs`), `heck` 0.5.0, `semver` 1.0.28,
  `shlex` 2.0.1, `syn` 2.0.119 and 3.0.6, and `unicode-ident` 1.0.26: searched
  their Rust source trees for direct temporary-path, environment, filesystem,
  and subprocess APIs. No active direct matching site was found (the `syn`
  matches were documentation examples). This is a scoped source search, not an
  audit of generated tokens, compiler internals, or arbitrary indirect code.

All 53 package versions in this closure, including the preceding 39 direct
build/proc-macro versions, are bound by exact Cargo.lock checksums and reviewed
unit mode, target source and features in `compile-time-reviewed.json`. New
versions, features, helper units and compiler-unit kinds refuse the scoped policy
until separately reviewed. Python-extension units and other target-specific
closures are not approved by this registry merely because `suite_01` passed.
