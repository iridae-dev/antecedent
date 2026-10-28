# Security, licensing, unsafe-code, and dependency review

Date: 2026-10-18
<!-- generated:release:reviewed-version:begin -->
Scope: workspace crates + `python` extension (package version **3.0.0**)
<!-- generated:release:reviewed-version:end -->

Focused source review for 2.4.1 covers the new experimental transport proof consumer,
exact-evaluation budget changes, Gaussian study pooling, weighted source-proof boundary,
and calibration decision/configuration readers. This is a source review; candidate cargo-deny and CI security
checks remain required and no new advisory database or CodeQL scan is asserted here.

Compared with `v2.4.0`, the reviewed working-tree Rust diff introduces no `unsafe` line.
An independent comparison of external Cargo.lock package name/version/source/checksum
tuples finds no added, removed or changed external dependency entries. Workspace package
version changes do not change that external dependency set.

The new trial-compatible transport verifier checks a 16 MiB wrapper limit before CBOR
decoding and caps original proof members at 4,096. Checked operation arithmetic bounds
source-row and directed-path replay before original report hashing, Fisher testing and
member decisions. Each retained member grid keeps its caller operation/depth limit and original stored
execution policy; this is a per-grid bound, not a global aggregate operation budget.
Total worst-case work is the proof bound plus at most 4,096 bounded grid replays.
An embedded member-count mismatch refuses before those grid consumers execute. Exact
binary source rows, their identity, the original report with directed/bidirected/confounded
edges and every retention decision are replayed, rather than trusting a supplied p-value
or an exclusion list. Producer trial input is capped at one million rows. These changes
close a new wrapper amplification path; they do not change the preexisting ordinary
transport artifact consumer into a new general bounded-CBOR decoder. The regression
fixtures cover oversized wrappers, excessive member counts, zero operation budgets and
semantic mutations; release gates must execute them on the final source.

Exact empirical evaluation now charges actual cached computation under caller operation
and depth limits instead of refusing from a pessimistic uncached preflight bound. Caches
reuse causal derivations, then rebind empirical laws on every bootstrap draw; they cannot
substitute the original sample's values for a resample. Actual supplied experiment catalogs
bound admissible intervention factors. The independent fresh-versus-cached resample
oracle and unavailable-regime refusal are part of this review's evidence.

Equal-study Gaussian pooling adds a distinct wire method and recomputes coefficients,
variance and Student degrees of freedom on artifact consumption. It retains registered
source independence and original ledger identity checks; equal numerical values do not
justify overlapping-source reuse. The ledger container retains its 128 MiB byte bound
and synthesis-count cap. Finite inputs, bounded study counts and nonfinite arithmetic
refusals precede publication. Common-mean Gaussian assumptions are a scientific scope,
not authority granted by importing an artifact. Weighted Gaussian working-model methods
similarly require an original native source proof; caller-supplied metadata cannot create
that proof or upgrade generic robust nuisance claims.

The optional suite coverage-decision configuration is validated against one complete
source-fixed gate roster, audited constructors and labels, scientific source hashes and
an exact fresh validation domain. Unknown configuration, unbudgeted looks, conflicting
cell counts, cap overflow and replay-domain/source mismatches refuse. Log collection
rechecks decision arithmetic and binds source, grid, record identity and denominators;
it retains required successor looks and recordless decisions. The source audit is a
review boundary, not automated dataflow analysis. Its 256-cell cap is a guard on the
source-possible family, not evidence of nonadaptive hypothesis choice. Family risk
controls coverage compatibility rejections only, conditional on valid binomial cells;
it grants neither a positive coverage certificate nor a budget for arbitrary method
revisions. Frozen scientific coverage floors, actual scored replicate minima, reporting,
precision, skip and named-boundary obligations remain separate. Old collected records
retain their original decision and RNG semantics.

External callbacks remain caller code rather than sandboxed execution. No new source
review statement grants native authority to a detached receipt or imported provider
output. Candidate dependency-policy, workflow and installed-wheel gates still need their
own successful receipts before publication.

## Callback execution

A callback is a provider function the library does not own. The protocol lives in
`antecedent-contract` (`callback_protocol`: declaration, reuse plan, output validation,
standing) and the executor in the analysis facade (`CallbackSession`).

| Concern | Control |
|---------|---------|
| Planning and verification | `CallbackSession::plan` and `plan_reuse` read declarations and stored records only and take no provider, so they cannot execute one. `verify_against_callback`, artifact `from_bytes` and `inspect` never invoke a provider. |
| Declarations | A `CallbackDeclaration` is data (an instance of the one interface-contract schema). Text, input and dependency counts are bounded (`MAX_CALLBACK_TEXT_BYTES`, `MAX_CALLBACK_INPUTS`, `MAX_CALLBACK_DEPENDENCIES`). Opaque or undeclared dependencies make reuse ineligible. |
| Provider identity | Execution checks the supplied provider is the declared object before calling; another declaration's input is refused. |
| Side effects | `CallbackExecutionPolicy::allow_side_effects` defaults to `false`; a side-effecting replay contract is refused with `side_effects_not_allowed` unless allowed, and is repeated only under a declared idempotent retry contract with an idempotency key within its attempt bound. |
| Resource bounds | Invocation count per request, `MAX_CALLBACK_DRAWS` and `MAX_CALLBACK_POINTS`, and `max_output_values` bound the output; cancellation is observed before the call, passed to the callback and observed after it. The wall-clock bound is applied after return (late results are discarded); there is no preemption and no sandbox. |
| Output | `validate_output` accepts only a well-formed value of the declared quantity and support; non-finite or malformed laws are refused. |
| Ledger | `CallbackSession` keeps an append-only history, per-request attempt counts, the invocation count and the evidence ledger in which nested callbacks' overlap is computed. A resumed process without a compatible provider records the stage `unavailable`; the historical result stays inspectable and is not silently re-executed. |
| Standing | Results are `diagnostic` or an external claim (`standing_of`); metadata does not mint native trust, a joint law, an identification licence, an interval or a posterior guarantee. |

Residual risk: a provider is arbitrary in-process code with the caller's privileges. The
library bounds and records the call; it does not isolate it. Treat provider code as trusted
code the caller chose to run.

## Artifact deserializers

Every artifact has its own version constant and a fresh-consumer path that re-derives the
stored digests, identities and derived numbers from the stored fields. A stored digest is
never trusted on its own, and a semantic edit that reseals the container is refused.

| Artifact decoder | Byte cap | Gates |
|------------------|----------|-------|
| Transport handoff (`transport_handoff_artifact`) | 8 MiB | Major-version and required-feature gate; callback identity re-derived; handoff re-run against the declaration with `verify_against_callback` |
| Joint-law propagation (`joint_law_propagation_artifact`) | 8 MiB (`MAX_JOINT_LAW_PROPAGATION_ARTIFACT_BYTES`); at most 256 inputs, 4096 shared-evidence entries, 4096 bytes per text field | Length checked before decode; version gate; non-finite results refused |
| Composition verification (`composition_verification_artifact`) | 32 MiB (`MAX_COMPOSITION_VERIFICATION_ARTIFACT_BYTES`); at most 65,536 shared-data relations | Version gate; ledger re-verification (`verify_ledger`) |
| Interface contract (`interface_contract_artifact`) | 16 MiB | Version gate; declaration bounds; identity recomputed |
| Callback result (`callback_result_artifact`) | 24 MiB | Version gate; declaration and output re-validated; output digest recomputed |
| Evidence ledger export (`evidence_ledger_artifact`) | 128 MiB | Version gate; source graph and overlap re-derived |
| Counterfactual transport, transported counterfactual, sampled transport | 64 MiB (transported) | Version gate; stored expectations and identity compared on consume |
| Cohort decision source (`cohort_decision`) | 64 MiB | Support, predictions, mean and identity recomputed from rows and model; caller supplies expectations |
| Measurement model (`measurement_model_wire`) | 8 MiB | Expected model identity required; digest recomputed |
| Inverse query result | 16 MiB (`MAX_INVERSE_QUERY_ARTIFACT_BYTES`) | Version gate; contract and source digests required from the consumer |
| Design ranking, decision and robust decision results | 8 MiB each (`MAX_DESIGN_RANKING_ARTIFACT_BYTES`, `MAX_DECISION_ARTIFACT_BYTES`); at most 65,536 robust items | Version gate; contract and source digests required from the consumer |
| Structural decision result | no separate byte constant; decoded through the artifact container | Version gate; contract and source digests required from the consumer |
| Adaptive result and monitoring contract | 256 MiB each (`MAX_ADAPTIVE_INFERENCE_RESULT_BYTES`, `MAX_ADAPTIVE_MONITORING_CONTRACT_BYTES`) | Checked decode needs the execution context; unchecked decode returns metadata only |

Decoders return typed errors and do not panic on malformed input. They are exercised by the
`fuzz/` targets (no-panic and bounded-allocation) and by per-artifact semantic-mutation tests.
Version gates refuse newer and unknown versions rather than reading them best-effort.

## Unsafe code policy

| Crate | Policy | Notes |
|-------|--------|-------|
| Most semantic crates (`antecedent-*` except below / kernels) | `#![forbid(unsafe_code)]` | Verified locally by `scripts/gate_release.sh` |
| `antecedent-data` | `#![deny(unsafe_code)]` + scoped `allow` | Foreign buffers (`buffer.rs`) and Arrow CDI (`arrow_ffi.rs`) |
| `antecedent-io-core` | `#![deny(unsafe_code)]` + scoped `allow` | Thin mmap (`mmap_file.rs`); `unsafe` mapped constructors on `MappedArtifactReader` |
| `antecedent-io` | `#![forbid(unsafe_code)]` | Estimator-bound artifacts; re-exports `antecedent-io-core` |
| `antecedent-kernels` | `#![allow(unsafe_code)]` | Only reviewed SIMD / aliasing kernels |
| `python` / `antecedent-py` | `#![allow(unsafe_code)]` | Required by PyO3 |

Gate fails if a forbid-crate loses `forbid(unsafe_code)`, or if data/io-core lose `deny` / their scoped escape modules.

## Licensing

- Project: `MIT OR Apache-2.0` (see `LICENSE-MIT`, `LICENSE-APACHE`, ADR 0008).
- Dependencies audited with **cargo-deny** (`deny.toml` license allow-list and
  advisories): the CI `deny` job runs `cargo deny check` on every PR and push, and
  the release-candidate gate refuses to run without cargo-deny installed.
- Default features must remain wheel-distributable without system BLAS
  ([ADR 0001](https://github.com/iridae-dev/antecedent/blob/main/adr/0001-linear-algebra-backend.md)).

## Dependency notes

| Component | Role | Review |
|-----------|------|--------|
| `faer` | Default linear algebra | Pure Rust; no system BLAS in default wheels |
| `paste` (transitive via `gemm`) | faer build-time macro | Unmaintained (`RUSTSEC-2024-0436`); ignored in `deny.toml` with reason — no runtime use; revisit when faer drops it |
| `bincode` (transitive via `burn-core`) | Record serialization in the CPU-native neural learner, included in standard Python wheels and optional for Rust consumers | Unmaintained (`RUSTSEC-2025-0141`), no vulnerability; ignored in `deny.toml` with a documented exception; revisit with the Burn upgrade |
| `arrow-array` / `arrow-schema` / `arrow-buffer` | Tabular / IPC sections | Feature-gated where needed; no algorithm duplication in Python |
| `pyo3` 0.29 / `numpy` 0.29 | Python boundary | Current Python bindings; the lockfile passes the advisory policy in `deny.toml` |
| `blake3` / `ciborium` / `serde` | Artifact container | CBOR + checksums under the format-0.5 artifact contract |
| `thiserror` | Error types | No runtime concerns |

`cargo deny check` passed on 2026-09-07: advisories, bans, licenses, and
sources were all `ok`. Its configured warning-level duplicate dependency and
unused license-allowance reports remain non-failing maintenance signals;
`yanked = "warn"` is unchanged in `deny.toml`.

## Wheel purity

Default maturin wheels use the `faer` path and must not link system BLAS.
Optional `blas` features (if added later) are non-default.

## CodeQL

- **CI:** `.github/workflows/codeql.yml` runs on every push to `main`, every pull
  request, and weekly (Mon 04:00 UTC). Uses the same
  `.github/codeql/codeql-config.yml` and `security-and-quality` suites as below.
- **Local strict gate** (requires `codeql` on `PATH`): `bash scripts/gate_codeql.sh`
  — not wired into Actions; fails unless rust / python / actions SARIF reports have
  **0** findings (same config and query filters as CI). Run before release or when
  touching security-sensitive surfaces.
- Third-party Actions in workflows are pinned to commit SHAs; workflows set explicit `permissions`.

## Published surface

| Surface | Destination | Notes |
|---------|-------------|--------|
| Rust facade `antecedent` + `antecedent-*` library crates | crates.io | Tag workflow `publish-release.yml` (jobs `crates-dry-run`, `publish-crates`); see `scripts/publish_crates.sh` |
| Python package `antecedent` (PyO3 crate `antecedent-py`) | PyPI / GitHub Release assets | **Not** on crates.io (`publish = false`) |

## Evidence commands

```bash
# Local slow path: unsafe forbid scan + inventory + benches (+ optional deny)
bash scripts/gate_release.sh

# License / advisory / source policy (optional local)
cargo deny check

# CodeQL (strict local gate — 0 findings; CI uses .github/workflows/codeql.yml)
bash scripts/gate_codeql.sh
```

## Callback and artifact review

The callback protocol, executor and the artifact families above add no `unsafe` block and no
workflow permission. The only manifest change on the branch is an in-workspace dev-dependency
(`antecedent-model` -> `antecedent-identify`, tests only); the lockfile gained that one
internal edge and no external crate, so no new license or `deny.toml` entry is needed.
