# Cargo compile-time inventory (review aid)

Generate a target-scoped offline inventory without compiling or executing any
test:

```sh
env -u CONDA_PREFIX python3 scripts/nix_compile_time_inventory.py --crate antecedent \
  --test suite_01 --target-triple aarch64-apple-darwin > /tmp/nix-audit.json
```

The command uses the existing evidence preflight and refuses known ambient
compiler, Cargo, or Python overrides. On this machine, `CONDA_PREFIX` must be
unset for the command; this does not change or erase that environment.

The command checks the pinned compiler and native target, and uses the selected
scientific target's declared features. It emits package versions, Cargo.lock
checksums, feature unions, parent relationships, and build-script or
procedural-macro target kinds. The target-filtered Cargo tree is conservative:
it can include dependencies that are not compiled for that particular test.

It also asks pinned Cargo 1.85.1 for `--unit-graph --no-run` and includes the
compiler-unit modes, features, platforms, target source paths and dependency
indices. That metadata query uses `RUSTC_BOOTSTRAP=1` only to enable Cargo's
unstable graph option; it does not build or execute tests. The resulting graph
is specific to the **local command and host environment**. It has not been
generated and compared inside the actual Nix derivation and is not proof of
the Nix build's units or of what a unit reads. This command describes Rust
scientific test targets, not the separate Python native-extension build.

When a Nix target derivation has emitted `cargo-metadata.json` and
`cargo-unit-graph.json`, pass both paths with `--builder-metadata` and
`--builder-unit-graph` to normalize and compare local and builder graphs. The
comparison includes unit profiles, features, target/mode and dependency edges
without depending on Cargo's unit-index ordering. Supplied JSON is diagnostic
data, not proof that the daemon built or sandboxed the artifact; verify its
relationship to the intended derivation separately. Cargo 1.85 legacy IDs and
newer canonical package IDs are supported.

For the current lockfile and the declared `calibration-internal` feature, the
example reports 173 packages and 234 local compiler units, including 39
versioned packages with build-script or procedural-macro units. These counts
change with the lock, source, target or features. Every package and unit is
marked `unreviewed`. The report records Cargo and rustc identities, workspace
manifest and root configuration hashes, and the diagnostic graph environment;
these hashes are not an attestation of a Nix build or a complete dependency
fingerprint.

`--hints` scans each package's Rust sources for candidate temporary-directory,
environment, filesystem, subprocess, and nondeterministic input sites. It
returns locations, not decisions. The patterns include false positives and
miss indirect calls, macro expansions and native code; a lack of matches is
not evidence of safety. The tool records the Cargo.lock SHA-256 but does not
verify the Nix sandbox, Nix store or extracted package sources. Source audit
findings and outstanding boundary work are in `compile-time-audit.md`.
