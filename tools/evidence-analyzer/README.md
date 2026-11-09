# Evidence compiler inventory

`driver.rs` is a `rustc` driver used as the workspace compiler wrapper when Rust
scientific assertions are earned. It builds with the compiler libraries of the
pinned compiler (`pin.toml`; `rustup component add rustc-dev`, or the Nix
toolchain in `nix/evidence-toolchain.nix`). With `ANTECEDENT_ANALYZER_OUTPUT`
set, it writes one JSON line per workspace module, glob import and item of each
crate it compiles: an item's key (its file and the byte offset of its name), what
its signature and body refer to, the module scopes its names are looked up in,
the method resolutions another declaration could change, and whether it reads
source positions. Keys and files are workspace-relative; references outside the
workspace are not recorded.

`scripts/evidence_items.py` combines this inventory with execution coverage to
select the items an assertion depends on, and checks them later from source alone
(`scripts/rust_items.py` finds and hashes items). What a manifest covers is
described in
[Earning and checking the records](../../docs/guides/scientific-verification.md#earning-and-checking-the-records);
the compile-time boundary under which item evidence is allowed is in
[`nix/README.md`](../../nix/README.md).

Host builds (calibration measurements) keep complete source whenever compile-time
code runs anywhere in the compilation, and require an assessed `compile_time`
input declaration in `parity/fingerprint_inputs.toml` for every compile-time
package in the consumer's dependency graph; unknown boundaries refuse earning.

Engineering checks:

```sh
PYTHONPATH=scripts python3 -m unittest test_evidence_items test_evidence_manifests test_evidence_earning test_scientific_records
python3 scripts/evidence_analyzer_probe.py   # the real driver on a disposable fixture; needs rustc-dev
python3 scripts/benchmark_evidence_manifests.py
```
