# T7–T9 implementation and verification

The implementation adds classical multi-source μsID, retained exact/empirical
finite response grids, and independently consumable structural/grid claims.
The theorem and finite-catalog scopes are deliberately separate. See the
[architecture and branch mapping](../architecture/transport-meta-grid.md),
[example](../../examples/python/transport_meta_grid.py), and
[stage contracts](../../parity/transport_stages.toml).

## Mathematical and epistemic checks

- Target-only identification runs first. Source substitution carries the selected
  population and its independently checked selection premise. Negative claims
  require a common checked μs-hedge against the complete source collection.
- Independent finite SCM enumeration covers all 64 ordered three-node ADMGs,
  seven source-selection pairs, two parameterizations, scalar/joint outcomes,
  intervention values, means and contrasts. Every positive recursive rule is
  consumed. An additional 20,480 four-node graph/source cases admit only checked
  positive proofs or checked obstructions.
- The complementary-source construction yields means 0.26 and 0.74 and contrast
  0.48. Each source alone has a checked obstruction. Source permutations,
  ablations, irrelevant sources, catalog alternatives and selection mutations
  are exercised.
- Missing joint experiments remain missing. Relevant zeros yield located support
  failures; an undefined conditional is skipped only under the existing checked
  irrelevant-zero rule. Every requested grid point survives in order.
- Dataset aliases share a fitted joint and resampling stream; conflicting aliases
  fail. Unknown dependence withholds IID uncertainty. Grids retain paired draws
  and pointwise labels. No formula averaging or source pooling is introduced.

## Durable claims and lifecycle

New certificate and grid payloads use explicit schema versions and required
features in the existing artifact container. Independent consumers check graph,
query, selections, premises, leaves, bindings, tables, sample provenance,
reasoning slots and embedded numerical claims. Tampering is tested even after
recomputing the container checksum. Missing raw samples prevent estimator replay,
not proof or embedded-table numerical verification. Loading does not fit, fetch
or resample.

Preparation retains native authority; metadata inspection reports factors,
bindings, source assumptions and execution capabilities. Snapshot replacement
invalidates execution. Refresh publishes only a fully validated candidate, so a
failed refresh retains the old state. Scalar projections carry loss receipts.
The frozen Rust-generated `NotCertified` fixture also round-trips through Python
without becoming an impossibility claim. Historical exact and statistical-v2
regressions remain passing.

## Verification record

The broad affected-crate run passed **1,715 Rust tests**, with 288 ignored tests
left ignored. A subsequent library regression run passed **795 tests**, overlapping
that broad run. The full Python regression run passed **2,084 tests**, with two
skips. The final focused Python run passed **112 tests**, with the long calibration
test skipped. It covers the frozen artifact, finite-provider schema validation,
resource-bound binding, and existing causal artifact compatibility. Final native
proof/grid tampering checks and strict Clippy also pass.

Commands used include:

```sh
cargo check --workspace --all-targets --features ml-full
cargo clippy -p antecedent -p antecedent-identify -p antecedent-io \
  -p antecedent-expr -p antecedent-estimate -p antecedent-py \
  --all-targets --features ml-full -- -D warnings
cargo test -p antecedent-identify -p antecedent-expr -p antecedent-estimate \
  -p antecedent-io -p antecedent --lib --tests --features ml-full
cargo test -p antecedent-identify --test meta_transport_scm
cargo test -p antecedent --lib analysis::transport_grid
cargo test -p antecedent-io --lib transport_certificate
ANTECEDENT_ALLOW_DEBUG_NATIVE=1 python/.venv/bin/python -m pytest python/tests -q
```

Python Ruff and mypy checks, formatting, provenance, artifact compatibility,
transport-stage and support-matrix gates accompany these tests. The main support
matrix retains 463 licensed cells; the new stage registry separately records
transport identification, execution and consumption contracts. No new calibrated
coverage cell is licensed.

## Explicitly deferred work

Hours-long multi-source and grid calibration has **not** been run. Reproducible
unequal-size sampling fixtures and bounded deterministic checks are implemented;
those checks measure reproducibility and inference bookkeeping, not coverage.
The opt-in calibration test is skipped by default and emits candidate records
requiring review and registry binding. T7 and T8 calibration-dependent completion
gates remain open. Simultaneous bands, quantile inference, derivatives and general
limited-experiment completeness remain unlicensed.
