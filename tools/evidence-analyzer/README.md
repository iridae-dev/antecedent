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

What a manifest covers is described in
[Earning and checking the records](../../docs/guides/scientific-verification.md#earning-and-checking-the-records).
Engineering checks:

```sh
PYTHONPATH=scripts python3 -m unittest test_evidence_projection test_evidence_manifests test_evidence_earning test_scientific_records
PYTHONPATH=scripts python3 scripts/test_evidence_pipeline.py
python3 scripts/evidence_analyzer_probe.py
python3 scripts/benchmark_evidence_manifests.py
```
