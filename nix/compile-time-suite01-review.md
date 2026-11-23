# Selected `suite_01` compile-time source review

This is a **partial, non-approving source review**, focused on whether code can
reach ambient temporary files. It is not proof that the build is hermetic or
that the Nix builder uses the local graph. No package below is marked reviewed
for earning: subprocesses, proc-macro dependencies, caller-generated tokens,
system libraries and the actual Nix environment still require assessment.

The metadata-only inventory described in `compile-time-inventory.md` was run
for `antecedent --test suite_01`, `aarch64-apple-darwin`, with the
`calibration-internal` feature, Cargo/rustc 1.85.1 and Cargo.lock SHA-256
`15df5baadf12136cc92d9ae9dac41e44c2f8d6f000748db4a26e41e053243d5f`.
It reported 234 compiler units and 39 versioned packages containing active
build-script or procedural-macro units. All observations here are for the
listed locked versions and the local graph, not a verified Nix unit graph.
Package-relative filenames below refer to the corresponding registry source.

## Build-script source inspected

* `crossbeam-deque` 0.8.8 and `crossbeam-epoch` 0.9.21: read each complete
  `build.rs`. These files read `TARGET` and `CARGO_CFG_SANITIZE` and print
  Cargo configuration; they contain no explicit filesystem or subprocess calls.
  `crossbeam-utils` 0.8.23 additionally includes `build-common.rs` and
  `no_atomic.rs`; both are small source-defined target helpers/tables. This
  does not establish the behavior of compiler or operating-system inputs.
* `quote` 1.0.47: read `build.rs`. It obtains `RUSTC` and invokes that program
  with `--version`; that executable and its transitive runtime are outside the
  build script's source review. `serde_json` 1.0.151 `build.rs` reads Cargo's
  target architecture and pointer-width variables, with no direct file access.
  `rayon-core` 1.13.0 `build.rs` prints a rerun directive only.
* `snap` 1.1.2: read its complete `build.rs`. It requires `OUT_DIR` and writes
  generated `tag.rs` and `crc32_table.rs` there using `File::create`. A fresh,
  private, verified output directory and safe path ownership remain necessary;
  the code does not itself establish that `OUT_DIR` is isolated.
* `nano-gemm-c32`, `nano-gemm-c64`, `nano-gemm-f32` and `nano-gemm-f64`
  0.2.1: read their four `build.rs` files. Each obtains `OUT_DIR`, generates
  both x86 and AArch64 code by calling `nano-gemm-codegen` 0.2.1, and writes
  `codegen.rs`. A lexical search of that helper's 2,726-line `src/lib.rs` for
  common file, environment, process, temp and nondeterministic APIs found no
  matching site; the helper was **not fully reviewed**, and absence of those
  strings is not a transitive safety proof.
* `flatbuffers` 24.12.23 `build.rs` calls `rustc_version::version_meta`;
  inspected `rustc_version` 0.4.1 `src/lib.rs:200–212`, which selects `RUSTC`
  and may prepend `RUSTC_WRAPPER`. `ahash` 0.8.12 `build.rs` calls
  `version_check::supports_feature` and reads target architecture. Inspected
  `version_check` 0.9.5 `src/lib.rs:159–164,302–342`: it runs `RUSTC` and can
  consult encoded or ordinary Rust flags. The identity and environment of
  these subprocesses remain inputs; the helper review is not exhaustive.
* `num-traits` 0.2.19 `build.rs` calls `autocfg::new` and asks it to compile a
  probe expression. Inspected `autocfg` 1.5.1 `src/lib.rs:170–210,295–342,
  622–633` and `src/rustc.rs:15–84`: it checks `OUT_DIR`, invokes rustc and
  optional wrappers, supplies target/flags, and creates probe names using a
  randomly seeded `HashSet` iteration. The naming randomness is observable;
  its effect on probe outcomes or final artifact identity has not been proven
  absent. `OUT_DIR` isolation and all subprocess inputs remain unresolved.

## Additional active build-script paths

The feature lists below are those in the local metadata-only `suite_01` unit
graph. “Output directory” means Cargo's `OUT_DIR`; its practical containment
assumes the Nix build uses a fresh private target directory under its builder
temporary directory and does not admit caller-selected output paths. That
assumption and the builder environment still need to be checked in the actual
derivation. The review trusts the locked package sources and the pinned tools
to behave as inspected; it is not an audit of operating-system implementation
or a proof about arbitrary transitive subprocesses.

* **Direct configuration:** `getrandom` 0.3.4 (features: none), `build.rs`,
  reads `CARGO_CFG_SANITIZE` and `CARGO_CFG_TARGET_FAMILY`; its `RUSTC` /
  `RUSTC_WRAPPER` version subprocess runs only when target family is Windows.
  Under the recorded Darwin graph it has no explicit temporary-file read.
  `libm` 0.2.16 (`arch,default`), `build.rs` and `configure.rs`, read Cargo
  target, feature, optimization and directory variables, enumerate environment
  names prefixed `CARGO_FEATURE_`, and print configuration. The inspected files
  do not directly open or execute a path under `/private/tmp`. `zstd-safe`
  7.3.0 (`arrays,legacy,std,zdict_builder`), `build.rs`, reads target architecture
  and OS and emits a cfg; its dependency `zstd-sys` is reviewed separately in
  `compile-time-audit.md`.
* **Controlled output plus compiler:** `serde` 1.0.229
  (`alloc,default,derive,rc,serde_derive,std`) and `serde_core` 1.0.229
  (`alloc,rc,result,std`), their complete `build.rs` files, write
  `OUT_DIR/private.rs` from the Cargo package patch version and execute the
  program named by `RUSTC --version`; serde_core additionally reads `TARGET`.
  They do not name `/private/tmp` directly, but depend on the output directory
  and the selected compiler. `zerocopy` 0.8.57 (`derive,simd,zerocopy-derive`),
  `build.rs`, reads a relative `Cargo.toml` and executes `RUSTC --version`;
  the package working directory, vendored manifest and compiler must match the
  intended locked inputs. `paste` 1.0.15's `build.rs` and `zmij` 1.0.23's
  `build.rs` execute `RUSTC --version`; `zmij` also reads `OPT_LEVEL`.
* **Compiler probes and outputs:** `proc-macro2` 1.0.107
  (`default,proc-macro`), `build.rs:145–237`, and `thiserror` 1.0.69 (no
  features) and 2.0.20 (`default,std`), `build.rs` probe routines, compile
  package-relative probe sources using `RUSTC`, optional `RUSTC_WRAPPER` and
  `RUSTC_WORKSPACE_WRAPPER`, `TARGET` and `CARGO_ENCODED_RUSTFLAGS`; they create
  and remove an `OUT_DIR/probe` directory. Thiserror 2 also writes
  `OUT_DIR/private.rs`. These paths are practically controlled only if the
  actual builder keeps the output directory private and prevents unapproved
  compiler wrappers/flags; subprocess runtime reads are outside this source
  inspection.
* **Conditional generated output:** `pulp` 0.22.3 (`std,x86-v3`), `build.rs`,
  first calls `version_check::is_min_version` (which invokes `RUSTC`) and reads
  `CARGO_CFG_TARGET_ARCH`; it returns before its `OUT_DIR/x86_64_asm.rs` write
  when that architecture is AArch64. Its path thus still depends on the
  observed target and Rust executable, and the generated-code branch is active
  for x86_64 builds.
* **Dynamic executable resolution:** `libc` 0.2.189 (`default,std`), `build.rs`,
  reads target variables, `LIBC_CI`, `LIBC_BUILD_VERBOSE` and
  `RUST_LIBC_UNSTABLE_*`; it invokes `RUSTC` and may invoke `RUSTC_WRAPPER`.
  Its unconditional `emcc_version_code()` attempts to resolve `emcc` from
  `PATH` even on Darwin; with `LIBC_CI` set it can also attempt
  `freebsd-version`. No direct `/private/tmp` path is specified by this script,
  but a PATH-selected tool can perform arbitrary reads. The host preflight
  rejects these `LIBC_*` and `RUST_LIBC_UNSTABLE_*` overrides; that does not
  establish the separate Nix builder environment. The builder PATH and these
  variables need concrete control before this unit is accepted.

## Procedural-macro source searches

For `dyn-stack-macros` 0.1.3, `enum-as-inner` 0.6.1, `equator-macro` 0.2.1,
0.4.2 and 0.6.0, and `seq-macro` 0.3.6, searched their complete `src/*.rs`
trees for common direct environment, file, process, temp and randomness APIs
and inspected imports and macro entrypoints. No matching direct source site was
found. This is a lexical search, **not a full call-graph audit**; `syn`,
`quote`, `proc-macro2`, `heck`, Rust's proc-macro runtime, macro input and
generated code are not established safe by this search.

`paste` 1.0.15 `src/segment.rs:154–166` calls `std::env::var` using the name
from an `env!(...)` macro-input segment. The unit graph includes several
callers, including `pulp`, `parquet` and `gemm` packages. A complete audit must
inspect actual expanded caller tokens and bound environment values; searching
for literal `paste!` and `env!` in source is not sufficient to establish this.

### Scoped `paste` caller inspection

The local `suite_01` Cargo tree identifies eight direct package parents:
`gemm`, `gemm-c32`, `gemm-c64`, `gemm-common`, `gemm-f32`, and `gemm-f64`
0.19.0; `parquet` 54.3.1; and `pulp` 0.22.3. The installed source trees for
those eight packages and `paste` 1.0.15 were checked against their Cargo.lock
archive checksums and contents before this inspection. The exact source and
feature graph must still be compared with the Nix builder's unit graph.

* Literal `paste!` syntax appears in `gemm-common/src/gemm.rs:924`,
  `parquet/src/arrow/arrow_reader/statistics.rs:323,797`, and `pulp/src/lib.rs`,
  `src/aarch64.rs`, `src/wasm.rs`, `src/x86.rs`, and `src/x86/v2.rs`, `v3.rs`,
  `v4.rs`. `gemm-common` uses paste inside an exported `macro_rules!` body;
  Parquet uses it inside statistics macros called with `Min` or `Max`; Pulp
  includes macro definitions forwarding identifiers and token fragments.
* Searched all eight parents' `src/**/*.rs` for literal `env ! (...)`, including
  whitespace and `r#env` spelling. The only matches are
  `parquet/src/file/properties.rs:51`, which uses `CARGO_PKG_VERSION` outside
  the two identified paste bodies, and `pulp/src/x86.rs:347`, which includes an
  `OUT_DIR` file under an x86_64 cfg. The latter is excluded by the recorded
  AArch64 target; this conclusion does not apply to an x86_64 build. The search
  does not show a literal caller supplying `TMPDIR` or another environment name
  to paste's `[<...>]` segment syntax.
* In `paste/src/lib.rs:195–260` the macro recursively expands token groups;
  `src/segment.rs:21–84,154–166` recognizes `env!("NAME")` within parsed
  segments. `src/lib.rs` uses it for pasted identifiers, and
  `src/attr.rs:105–135` also uses it for name-value attributes such as docs.
  This reads the environment **value**; it is not itself a filesystem read.
  A different caller or macro-generated token
  could still cause an environment lookup, and a downstream compiler or
  build-script subprocess could independently read a path named by such a
  value. The source search does not expand macros, cover generated input, or
  prove that all actual caller tokens were seen. No observed literal call here
  establishes an ambient `/private/tmp` read; dynamic/generated cases and the
  builder's actual environment remain unresolved.

For `serde_derive` 1.0.229 (`default`), `thiserror-impl` 1.0.69 (no features)
and 2.0.20 (no features), and `zerocopy-derive` 0.8.57 (no features), searched
their `src/**/*.rs` files for common direct filesystem, subprocess,
environment and temporary-directory APIs and inspected the macro root modules.
`serde_derive` and `thiserror-impl` 2.0.20 use the compile-time
`CARGO_PKG_VERSION_PATCH` in an identifier; a `zerocopy-derive` test-only module
contains `ZEROCOPY_BLESS` environment and file writes and is declared under
`#[cfg(test)]` in its root. Those searches do not inspect every call or emitted
token. The shared `syn`, `quote`, `proc-macro2`, standard-library and
compiler/runtime behavior remains a practical trust assumption to assess, not
an established absence of indirect reads.

## Remaining review and practical trust boundary

The 39 locally observed build-script/proc-macro package versions now have an
entrypoint inspection or scoped macro-source search in this file or the
companion `compile-time-audit.md`; this is **not** a full audit of their
transitive helper crates, all conditional branches, caller-generated tokens or
native compilers. The 134 other packages in the local package graph include
helpers callable during build; they have not all been reviewed. In particular,
Nix builder environment and `PATH`, private `OUT_DIR`/`TMPDIR`, vendored source
identity, compiler/SDK/runtime and `paste` callers must be documented and
checked; `libc`'s `emcc` lookup and `cc`'s temp fallback are dynamic paths that
need concrete handling. Python-extension build units are **unreviewed**. The
Nix Cargo 1.97 unit graph for this target still needs to be captured and
compared against the local 1.85 graph before this source review can be applied
to it. Under the practical source-audit approach, trusted compiler, package and
OS components are assumptions to name and bind, not a mathematical proof of
all operating-system reads. The existing earning path remains fail-closed.
