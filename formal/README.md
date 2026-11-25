# Formal checks

Machine-checked properties of Antecedent operations and the adversarial cases run against
them. Nothing here is needed to build or use the library. Each property is described by
an assurance record in [`parity/assurance_records.toml`](../parity/assurance_records.toml),
and the required adversarial cases by
[`parity/adversarial_campaign.toml`](../parity/adversarial_campaign.toml). The choice of
tools is [ADR 0027](../adr/0027-rust-verification-route.md).

| Directory | What it holds | Checked by |
| --- | --- | --- |
| `Antecedent/` | Lean specifications and proofs; Plausible-generated cases (`Generated.lean`) | `lake build` here |
| `rust/` | The production Rust kernels translated by Charon and Aeneas (`AntecedentRust/Extracted/`), and proofs that they equal the specifications | `lake build` in `rust/` |
| `kani/` | Kani proof harnesses on the production crates' public API (its own workspace) | `cargo kani` in `kani/` |
| `adversarial/` | proptest cases against the production crates (its own workspace) | `cargo test` in `adversarial/` |

Tool versions are pinned in [`toolchain.toml`](toolchain.toml).

## Running the checks

```bash
cd formal && lake build                      # specifications, proofs, generated cases
cd formal/rust && lake exe cache get && lake build
cd formal/kani && cargo kani                 # Kani 0.68.0
cd formal/adversarial && cargo test
```

After changing a covered Rust item, a proof or a pin, rerun the record's checks and
rebind it. `AENEAS_DIR` is an unpacked Aeneas release at the pinned version:

```bash
AENEAS_DIR=/path/to/aeneas python3 scripts/assurance.py run
python3 scripts/assurance.py check
```

The full adversarial campaign, including every deliberate defect:

```bash
AENEAS_DIR=/path/to/aeneas python3 scripts/adversarial.py run --mutations
```

## Operation records

Every operation an assurance record lists in `operations` is described once, in
[`parity/operation_records.toml`](../parity/operation_records.toml), with the fields
inputs, output, premises, transformation, justification, implementation, obligations and
limitations ([ADR 0029](../adr/0029-operation-records.md)). A record is keyed by the
operation's defining Rust path and attached to its code by `file` and `item`.
`justification` and `implementation` are generated from the assurance records, identity
sites, public-API snapshots, compiled-path classification and route registry; the other
fields are written by hand, and each premise names the refusal token the checker finds in
the code.

```bash
python3 scripts/operation_records.py --write   # regenerate; add a skeleton for a new operation
python3 scripts/operation_records.py check     # static gate
```

## Adding a translated kernel

Keep the kernel a named function over enumerations, integers or fixed-size arrays, and
have the larger function call it. Add an `extract` line to [`rust/extract.sh`](rust/extract.sh),
run it, write the equality proof in `rust/AntecedentRust/`, and add a
`verified_implementation` check, naming the Rust items it covers, to the record. List the
kernel in the record's `operations`, run `python3 scripts/operation_records.py --write`, and
fill in the hand-written fields of its new operation record.
