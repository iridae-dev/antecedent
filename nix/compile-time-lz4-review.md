# Darwin review of the vendored LZ4 build

This is a metadata and source review of two Apple Silicon compiler units.
The Arrow IPC feature change adds the locked runtime package LZ4 and the
featureless build and run-custom-build units of `lz4-sys`
`1.11.1+lz4-1.10.0`. The registry binds the exact package checksum. The
locked archive hash and the locally resolved package's build script and
vendored source files were compared before review.

- `build.rs:16–25` compiles four explicit package-relative C translation
  units with the previously reviewed `cc` helper. The four translation
  units and their package headers select vendored includes and the compiler
  SDK's standard headers under the reviewed Apple ARM C toolchain.
- `build.rs:27–52` reads Cargo's `TARGET`. Windows-only CRT overrides and
  Wasm-only shim inclusion are inactive for this reviewed target.
- `build.rs:54–74` enumerates the package's `liblz4/lib` directory and
  copies its headers under Cargo's `OUT_DIR`. The directory
  enumeration creates separate destination files and does not concatenate
  them in iteration order. The script emits the output directory and its
  include subdirectory to Cargo. It contains no package discovery via
  pkg-config and no caller-selected source path.

This review inherits the pinned environment, private output directory and
compiler-helper assumptions in the existing Darwin audit. It covers the
specific checksum, target and featureless unit signatures in the registry.
Offline, locked unit-graph metadata established that the discovery library
selects no new compile-time unit and that the composition integration
targets and native Python extension select these two units. No compilation
or scientific assertion ran as part of this review.

## Linux

The same two units, with the same checksum and featureless signatures, are
selected for `x86_64-unknown-linux-gnu` and `aarch64-unknown-linux-gnu`.
`build.rs` was read again from the locked package. Neither Linux `TARGET`
contains `windows` or matches a Wasm target, so the script takes the same
path as on Darwin: the four vendored translation units are compiled by the
reviewed `cc` helper under the Linux audit's pinned toolchain, and the
`liblz4/lib` headers are copied under `OUT_DIR`. No repository function text
or source position is read. This review inherits the assumptions of
`nix/linux/compile-time-audit.md`.
