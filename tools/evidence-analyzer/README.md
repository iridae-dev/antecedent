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

The metadata graph is a conservative inventory, not the exact set executed on a
particular build: it includes target-specific and optional packages. On the
reviewed macOS host, the offline default-feature workspace tree contains 58
versioned compile-time packages, while the full metadata inventory above has 130.
Neither count substitutes for the graph of the actual test target and its enabled
features. Scientific Rust earning builds each selected test target separately;
Python earning uses its existing native extension. Those build configurations
and their dependencies must be bound to the input assessment before narrowing
the conservative inventory.

Two packages illustrate why the graph and invocation sites matter:

* `const-random-macro` 0.1.16 derives macro output from `CONST_RANDOM_SEED` when
  set at macro compilation, and otherwise reads operating-system randomness.
  It appears in the full metadata graph but not the reviewed host default graph.
  An assessment must prove it inactive for the actual build or control and bind
  its seed; an empty input list cannot represent an entropy read.
* `paste` 1.0.15 can evaluate an `env!(...)` string supplied inside macro input.
  The environment names are therefore selected at its caller sites, including
  callers in dependencies, rather than fixed by `paste` itself. A review needs
  to cover those sites and expansions under the actual build graph.

To finish this review, first derive and test the host, target, feature and
build/dev dependency graph for each earning command; then assess every active
package's transitive helper calls and caller-selected inputs. Record package
source identity, compiler and native tool/SDK identities, interpreter inputs,
environment and any remaining data reads in a checkable boundary. Refuse an
unknown graph, uncontrolled entropy, or unrepresentable input rather than
claiming an empty boundary. Build scripts such as `num-traits`, `ahash` and
`flatbuffers` call helper crates that probe the compiler, so searching only
their own `build.rs` for environment and process APIs is insufficient.

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

`python3 scripts/registry_integrity.py` is an optional diagnostic: after a
conservative ambient-environment check, it compares offline resolved registry
archives to `Cargo.lock` and extracted package files and membership to those
archives. It executes no scientific or calibration tests. It is not yet
enforced by earning and does not establish native tool, SDK, interpreter or
other compile-time input boundaries; a successful diagnostic is not an
authorization to declare those boundaries complete.

Earning does reject known ambient Cargo configuration and build overrides
before creating a snapshot, and checks them again before publishing. Scientific
run entry points also check before discovering rows or Cargo metadata. This
fail-closed preflight does not assess arbitrary macro reads or establish that
the installed native toolchain and SDK are immutable.

What a manifest covers is described in
[Earning and checking the records](../../docs/guides/scientific-verification.md#earning-and-checking-the-records).
Engineering checks:

```sh
PYTHONPATH=scripts python3 -m unittest test_evidence_projection test_evidence_manifests test_evidence_earning test_scientific_records
PYTHONPATH=scripts python3 scripts/test_evidence_pipeline.py
python3 scripts/evidence_analyzer_probe.py
python3 scripts/benchmark_evidence_manifests.py
```
