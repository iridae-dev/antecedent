# Evidence compiler inventory

`driver.rs` is a `rustc` driver used as the workspace compiler wrapper when Rust
scientific assertions are earned. It reports each ordinary function body's span
and text; with execution coverage, `scripts/evidence_earning.py` turns bodies an
assertion never ran into holes of a guarded source template. It needs the
compiler libraries of the pinned compiler (`pin.toml`):

```sh
rustup component add rustc-dev
```

Without them, earning keeps complete source. Body projection applies only to a
compilation with no procedural macro or build script anywhere in its resolved
dependency graph; when no owed assertion qualifies, the driver is not built. Run without
`ANTECEDENT_ANALYZER_OUTPUT`, the driver also prints every item and the query
dependency edges; `scripts/evidence_analyzer_probe.py` uses that on a disposable
fixture to check the compiler interface.

Keeping full source alone does not establish the data read by a procedural macro
or build script. Earning therefore also requires an assessed `compile_time` input
declaration on the consumer's Cargo.toml in `parity/fingerprint_inputs.toml` for
every compile-time package in its dependency graph. Unknown boundaries or missing
Cargo metadata refuse earning; declared data, environment inputs and missing paths
remain dependencies. See the declaration example in that registry. Policy
`conservative-source-v2` invalidates manifests earned before these safeguards.

### Compile-time boundary assessment (3.0.0 freeze)

The locked offline metadata inventory currently finds 130 versioned compile-time
packages (107 distinct names) across the 24 workspace consumers. This is the full
resolved graph used by `compile_time_code`, including target-specific packages;
it is not a list of packages executed on the current host. A package-name row
therefore needs to assess every resolved version of that name.

The initial source review establishes these concrete boundaries:

| Package | Reviewed reads and behavior |
| --- | --- |
| `blake3` 1.8.7 | `build.rs` enumerates its registry `c/` directory and compiles bundled C/assembly through `cc`. Reads target/feature metadata, `BLAKE3_CI`, `CC`, and `CFLAGS`; `cc` additionally selects the compiler, flags, archiver and SDK from the host environment. An empty data boundary would omit the bundled sources and external toolchain. |
| `proc-macro2` 1.0.107 | Compiles bundled `src/probe/*.rs` with the selected Rust compiler. Reads `RUSTC`, wrappers, `RUSTC_BOOTSTRAP`, `RUSTC_STAGE`, `TARGET`, `OUT_DIR` and encoded Rust flags. Probe outputs are generated files, not independent inputs. |
| `quote` 1.0.47 | Build script executes `RUSTC --version`; no additional data-file read in that script. |
| `thiserror` 1.0.69 / 2.0.20 | Compiles bundled `build/probe.rs`, using Rust compiler/wrappers, bootstrap/stage, target and encoded Rust flags. Version 2 also generates `private.rs` using Cargo's package patch version. |
| `getrandom` 0.3.4 / 0.4.3 | Reads Cargo's sanitizer metadata; 0.3.4 additionally reads target family and probes the Rust compiler through its wrapper on Windows. |
| `libc` 0.2.189 | Reads target metadata and its unstable ABI overrides, probes the Rust compiler/wrapper, and has target-dependent `freebsd-version`, Emscripten compiler and VxWorks environment inputs. |
| `serde` / `serde_core` 1.0.229 | Generate `private.rs` from Cargo's package patch version and probe `RUSTC --version`; `serde_core` also reads `TARGET`. |
| `serde_json` 1.0.151 | Build script reads Cargo target architecture and pointer width. |
| `zmij` 1.0.23 | Probes `RUSTC --version` and reads `OPT_LEVEL`. |
| Workspace `antecedent-py` | `python/build.rs` reads target OS and `OUT_DIR`, obtains interpreter configuration through `pyo3_build_config`, and on macOS discovers the linker through `xcrun`, `cc` or `which`. The selected Python installation and native linker are external inputs. |

These are review findings, **not completed declarations**. In particular, the
`cc` and PyO3 helpers need their own transitive read assessment, as do the
remaining macro/build packages (including CUDA/HIP and zstd tooling).

The current manifest format only records repository-relative file inputs.
`Fingerprinter.declared` bounds paths to the repository; an external path is
treated as a missing declaration, not hashed. `Cargo.lock` is hashed and the
Rust compiler release/commit are checked, but earning does not verify installed
registry file checksums or record the native compiler, SDK or Python interpreter
identity. Lockfile checksums alone do not detect local edits to extracted package
files. Before declaring registry/native-tool reads covered, earning needs either
a verified immutable package/tool environment or explicit external-input support
with equivalent checking. Listing outside paths or filling `inputs = []` would
not supply that protection.

Cargo-generated metadata (`OUT_DIR`, package versions, target/feature variables)
must also be distinguished from user overrides: declaring a generated variable
as a host environment input currently only refuses a pre-existing host value;
it does not record the value Cargo passes to a build script. Do not waive these
reads without establishing which locked build configuration determines them.

What a manifest covers is described in
[Earning and checking the records](../../docs/guides/scientific-verification.md#earning-and-checking-the-records).
Engineering checks:

```sh
PYTHONPATH=scripts python3 -m unittest test_evidence_projection test_evidence_manifests test_evidence_earning test_scientific_records
PYTHONPATH=scripts python3 scripts/test_evidence_pipeline.py
python3 scripts/evidence_analyzer_probe.py
python3 scripts/benchmark_evidence_manifests.py
```
