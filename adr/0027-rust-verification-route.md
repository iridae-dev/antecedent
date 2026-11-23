# ADR 0027: Rust verification route

- Status: Accepted
- Date: 2026-11-23

## Context

Machine-checked properties of Antecedent are worth most when they hold of the Rust that
runs, not only of a model of it. Four open-source Rust verifiers were candidates: Aeneas
with Charon, Verus, Creusot and Kani. Each was tried on the same production kernels:

- the claim-standing checks of `compose_claims` (`domain_change_allowed`,
  `identification_status_kept_or_lowered`, `IdentificationStatus::strength_rank`): pure
  functions over fieldless enums;
- `antecedent_io_core::bound_identity::bind`: comparisons of 32-byte digests, with a
  BLAKE3 facet digest;
- the order-insensitive source-list encoding of the evidence ledger (`sorted_digests`,
  `write_ids`): `Vec` and `sort_unstable`.

Versions tried: Aeneas `nightly-2026.11.23-2e3d448` with Charon 0.1.322 (nightly Rust
2026-09-17), Verus 0.2026.11.22.1ce6a59 (Rust 1.100.0), Creusot v0.15.0 (nightly Rust
2026-08-03), Kani 0.68.0 with CBMC 6.11.0.

## Findings

| | Aeneas + Charon | Kani | Verus | Creusot |
| --- | --- | --- | --- | --- |
| Runs on unmodified production code | yes | yes, through the public API | no | no |
| Standing kernels | translated with no errors; equality with the Lean specification proved | not tried (Aeneas sufficed) | verified a copy in 3 s, only inside `verus!` with a `Structural` derive | not run |
| `bind` | not tried (BLAKE3 would be opaque) | 3 harnesses, 17 s, for every digest function (stubbed) | not tried | not run |
| `sorted_digests` | `sort_unstable` has no model | needs `cfg(kani)` declared in the crate | not tried | not run |
| What it trusts | Charon and Aeneas translation; the Aeneas standard-library model; Lean | Kani's MIR-to-CBMC translation; CBMC; bounded unwinding | Verus's own rustc and `vstd`; Z3 | Why3, its solvers, the Coma translation |
| Proof language | Lean, shared with the specifications | Rust harnesses | Rust with Verus specification syntax | Rust attributes, Why3 obligations |
| Cost | Lean library on Mathlib: 7.7 GB, 90 s from the Mathlib build cache | about 1 GB toolchain; minutes to compile dependencies once | its own Rust toolchain | opam switch, Why3, SMT solvers, a nightly Rust, and GTK3, cairo and gtksourceview from the system package manager for the Why3 IDE; two install attempts failed |
| Deliberate defect caught | relay rule weakened: proof fails | payload comparison removed: harness fails | not tried | not run |

Verus and Creusot verify code written for them: functions inside `verus!` with Verus
derives, or code annotated with `creusot-contracts`. Using either on Antecedent would put
verification syntax, a dependency and a second compiler into published crates, which the
roadmap's rule against proof overhead for users rules out. Creusot's result rests on its
installation cost and that requirement; it was not run on the kernels.

## Decision

- Aeneas with Charon is the primary route. Kernels that are pure functions over
  enumerations, integers and fixed-size arrays are translated from the production Rust
  (`formal/rust/extract.sh`), and Lean proves the translation equal to the specification
  in `formal/`, so the specification's theorems hold of the code. When a kernel's
  decision is inline in a larger function, it is extracted into a named function the
  larger function calls, with unchanged behaviour.
- Kani is the second route, for properties of code with standard-library collections,
  byte arrays or hashing that Aeneas cannot translate. Harnesses live in `formal/kani`,
  a separate workspace that calls the public API, so no published crate gains a `cfg`,
  dependency or build script. Hashes are stubbed with arbitrary values, so a proof uses
  no property of the hash.
- Verus and Creusot are not adopted.
- Each assurance record (`parity/assurance_records.toml`) names which route each check
  used. A Lean proof about a specification, a Lean proof about the translated Rust and a
  Kani proof are separate kinds of check and are never reported as one.

## Consequences

- The proofs about translated Rust need the Aeneas Lean library, which builds on Mathlib.
  CI fetches the Mathlib build cache. The proofs are checked from it, and can be rebuilt
  from source without it.
- Private kernels without a public entry point can be translated by Aeneas, but not
  checked by Kani without declaring `cfg(kani)` in the crate. Declaring it in the
  workspace manifest would change every evidence fingerprint, so a Kani check of a
  private kernel waits until that is worth a bookkeeping rebind of all records.
- `formal/toolchain.toml` pins every tool version; changing one makes every assurance
  record stale until its checks run again.
