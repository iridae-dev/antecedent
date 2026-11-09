# Parity manifests

Domain capability inventories and pinned external baseline oracles. See ADR 0009
and [docs/development.md](../docs/development.md).

[How Antecedent checks its scientific methods](../docs/guides/scientific-verification.md)
follows these evidence distinctions through AIPW and arrow-strength examples,
from independent references to executing assertions and calibration requirements.

**Inventories** (status: `pending` | `done`):

- [estimate.toml](estimate.toml) — identify / estimate / refute
- [discovery.toml](discovery.toml) — temporal discovery / CI / graphs / effects
- [bayesian.toml](bayesian.toml) — Bayesian core
- [gcm.toml](gcm.toml) — GCM / counterfactual
- [pag.toml](pag.toml) — PAG / LPCMCI
- [context.toml](context.toml) — Context / regime / effects
- [attribution.toml](attribution.toml) — Attribution
- [design_state.toml](design_state.toml) — Design / incremental-state
- [release.toml](release.toml) — Release-prep / parity-closure
- [response.toml](response.toml) — response / observation / transport / interference
- [scenario_matrix.toml](scenario_matrix.toml) — curated cells of the method-family scenario matrix (categorical, dependence, degeneracy, unsupported inputs, held-out regimes, weights/missingness); `scripts/scenario_matrix.py` joins them with `counterparts.toml` and `missing_policy_support.toml`, and `scripts/evidence_audit.py report` lists the gaps (not an inventory)
- [compiler.toml](compiler.toml) — inspect / claim / reuse / handoff compositions (not analysis-matrix cells)

**Supported units** (one notion of support, one count; see ADR 0020):

A supported unit is one scientific combination, from exactly one source: a
support-matrix cell (`support_cells.toml`), a graphless row
(`support_graphless.toml`), or a capability with no matrix axes described by a
capability record (`capability_records.toml`: transport stages, adaptive
procedures, decisions, bridges). One rule decides support for every unit: a
provenance record plus an executed independent scientific check (known truth, an
analytic or enumerated value, or an external reference implementation; never
internal consistency or a frozen output of this library). A capability record
whose combination is already a matrix cell or graphless row names it with
`matrix_units` or `graphless_units` and is not counted again.

API and stage routes listed under a record are **entry points**. They have no
support status of their own: a route is open only as an entry point of a
supported unit and otherwise refuses with `not_supported`, so a route count is
never the support count. `python3 scripts/gate_support.py` applies the rule to
every unit, prints the one headline (`Supported units: N (matrix cells a,
graphless rows b, capabilities outside the matrix c); not supported: M`) and
lists each unit that lacks a provenance record or an independent check;
`--report` prints the same and exits 0.

**Support matrix** (support, not an inventory; see ADR 0020):

- [support_axes.toml](support_axes.toml)
- [support_n_a.toml](support_n_a.toml)
- [support_refusal_reasons.toml](support_refusal_reasons.toml)
- [support_cells.toml](support_cells.toml)
- [support_graphless.toml](support_graphless.toml) — exact graphless design,
  method, and inference supports with assignment-unit support gates. It is
  generated into a separate Rust lookup and
  [graphless support page](../docs/graphless-support-matrix.md); absent keys
  default to refusal and cannot inherit a geometric cell. Run
  `python3 scripts/gate_support.py` to check generated output, and add `--cited-tests` to check that
  each cited known-truth and retained-route test is still in its harness.

Cross-language names: [docs/api_naming.md](../docs/api_naming.md).

**Baseline pins** (oracle metadata only — not inventories):

- [baselines/dowhy.toml](baselines/dowhy.toml)
- [baselines/tigramite.toml](baselines/tigramite.toml)
- [baselines/causal-learn.toml](baselines/causal-learn.toml)
- [baselines/lingam.toml](baselines/lingam.toml)
- [baselines/statsmodels.toml](baselines/statsmodels.toml)
- [baselines/bpbounds.toml](baselines/bpbounds.toml)
- [baselines/causaleffect.toml](baselines/causaleffect.toml)
- [baselines/riskRegression.toml](baselines/riskRegression.toml)

## Capability row schema

Each `[[capabilities]]` row uses:

| Field | Required | Values |
|-------|----------|--------|
| `id` | yes | Dotted capability id, unique within the manifest |
| `status` | yes | `pending` \| `done` |
| `group` | see below | Grouping key |
| `description` | see below | One-line capability description |
| `owner` | see below | Owning domain |
| `notes` | no | Free-form evidence / gate pointers |
| `python_facade` | no | `full` \| `thin` |
| `evidence_kind` | on `done` rows | See table below |
| `external_oracle` | with external kinds | `"<project> <pin>"` |
| `known_truth_fixture` | no | Strongest exercised fixture path |
| `oracle_components` | external kinds only | Components the row's oracle test exercises: `estimator:<id>`, `identifier:<id>`, `discovery:<GraphClass>` |
| `evidence_test` | on support-matrix cross-check rows | Repository-relative Rust/Python test file |
| `evidence_assertion` | on support-matrix cross-check rows | Executing test-function name |
| `limitations` | no | What the evidence does **not** demonstrate |
| `known_truth_reference` | inline analytic/enumerated truth | Origin `kind`, exact `source` (`path::assertion`), `proposition`, `tolerance` and `limitations`; source identity is checked, independence still requires review |
| `evidence_activity` | no | `statistical_smoke` distinguishes an ordinary stochastic check from scientific conformance and measured calibration |
| `calibration_evidence` | no | Explicit measurement test/assertion, group, phase and status; ordinary point conformance cannot stand in for this measurement |

**`evidence_kind`** states exactly what proposition the row's evidence
demonstrates, so wording in `notes` cannot outrun it. Required on every `done`
row except in `release.toml` (infrastructure rows; evidence map lives in
`gate_release.sh`). Values (different propositions, not a scientific sufficiency ranking):

| Value | Demonstrated proposition |
|-------|--------------------------|
| `implementation_exists` | Code + ordinary unit tests; no numerical truth |
| `internal_known_truth` | Matches a closed-form / analytic / clean-room fixture |
| `internal_cross_check` | Agrees with another Antecedent estimator only |
| `regression_pin` | Equals a frozen output of this library (a seeded run or a copied value): a change detector, not truth. Names its test, never a `known_truth_fixture` |
| `frozen_external_oracle` | Matches a frozen, pinned upstream-package run |
| `behavioral_parity` | Agrees with an upstream package across a range of inputs |
| `contract_equivalence` | Theorem-level / method-contract argument |

For the two external kinds, `external_oracle` and `known_truth_fixture` are
required. The gate also requires a matching project/version record under
`parity/baselines`, valid frozen `expected.json`, and an executing Rust/Python
conformance test that names the fixture. It enforces the fixture-authoritative
rule too: the named project must literally appear in the frozen fixture. A
clean-room enumeration is `internal_known_truth` no matter what the row's prose
says.

Supported support-matrix rows use a stricter truth condition. Every row names the
executing test function that carries its evidence, and that function itself
(with the helpers it calls) must build the row's axis values; known-truth and
external-oracle rows must also name a fixture that same function parses and
asserts on. An `internal_cross_check` row must not name a
`known_truth_fixture`: agreement between two Antecedent paths is consistency
evidence, not independent truth evidence. The same holds for a `regression_pin`
row, and a fixture whose `oracle.kind` is `regression_pin` (frozen output of this
library) may back known truth only for the inference modes its oracle lists in
`independent_inferences`. Every conformance `oracle` block carries a `kind` from
one closed vocabulary (`external_package`, `closed_form`, `enumeration`,
`independent_reimplementation`, `regression_pin`), enforced by
`gate_evidence_reachability.sh`; the same gate requires that block on every
fixture a registry cites as `known_truth_fixture`, so a known-truth fixture
always states where its truth comes from. A pinned upstream run is
`external_package` (the oracle kind; `frozen_external_oracle` is a row's
`evidence_kind`, not an oracle kind). A `process_attested` row exists only in
`release.toml`, for a release-process fact no test can execute; its notes name the
gate or workflow that enforces it.

A fixture mixing independently generated and product-generated fields names its
`oracle.independent_fields`. A known-truth row must explicitly select a nonempty
subset as `evidence_fields`; regression outputs cannot borrow another field's
independence. Retained `oracle.reference_review` records bind the generator,
transitive comparison helper, environment lock, assertion source and fixture
payload by digest. Gates reject stale reviews. These receipts preserve the
scope and limitations of an actual reference review; hashes and source scanning
alone cannot establish scientific sufficiency, historical exposure or theorem
correspondence.

Fixtures serving only software checks may declare `evidence_roles = ["software"]`
with `evidence_kind`, exact `evidence_source` (`path::assertion`),
`evidence_proposition` and `limitations`. They need no invented numerical oracle.
Known-truth claims cannot cite such a fixture, even if it also carries an
external-origin tag. The audit lists these software propositions separately.

An explicit provenance `method_review` records bounded primary-source
correspondence, assumptions, resolved findings, exposure uncertainty and limits.
It binds the provenance content and implementation/assertion/reference artifacts
by digest. Stale reviews fail the provenance gate. Structural provenance/API
links, bounded primary reviews and measured calibration remain separate report
counts; none substitutes for another.

A supported row labelled `frozen_external_oracle` compares **the cell's own
output** with a pinned upstream run. Most supported units instead rest on
external oracles at the **component** level: the identifier and estimator each
supported cell ran are recorded in `supported_routes.toml` (generated by the
supported compiler test from executed plans), and a capability row with an
external kind names the components its oracle test exercises in
`oracle_components`. `scripts/generate_support_matrix_docs.py` joins the two
into the support matrix's *external oracles* column, and
`scripts/external_evidence.py` (run by `gate_parity_schema.sh`) keeps the join
honest: components only on external rows, and only components a supported cell
runs. A component link never means the cell's own output was compared.

What these fields do **not** mean, stated once so readers do not infer it:
`evidence_kind` names the *strongest demonstrated* proposition, not the only
evidence a row has, and says nothing about whether that evidence is
scientifically sufficient — `frozen_external_oracle` means "matched a pinned
upstream run on the frozen cases", not "correct", and an upstream package can
be wrong. `limitations` is a curated list of the caveats a reader most needs,
not an exhaustive one; its absence is not a claim of none. A scipy/numpy pin
recording the *computational substrate* of a clean-room harness does not make
the evidence external — external means an upstream implementation of the
capability under test produced the compared values.

**`group` / `description` / `owner`** are required in
[estimate.toml](estimate.toml), [discovery.toml](discovery.toml),
[context.toml](context.toml), and [compiler.toml](compiler.toml), and unused
elsewhere.

The required-key contract is enforced by
[`scripts/gate_parity_schema.sh`](../scripts/gate_parity_schema.sh), which every
feature gate and `gate_release.sh` invoke. It exists because the gates' inline
`caps()` parser reads keys with a regex default — an absent `status` yields
`None`, matches no honesty check, and would otherwise pass forever without the
row ever being marked. The schema gate also pins that regex against a real TOML
parse, and enrolls new manifests automatically: any `parity/*.toml` carrying
`[[capabilities]]` rows that is not listed in the gate fails it.

**`python_facade`:** When a capability is Rust-done but the Python surface is
incomplete or `_native`-only without a typed facade, set `python_facade =
"thin"`. Use `"full"` when the public `antecedent` package exposes the capability
end-to-end (analyze kwargs, typed wrappers, or dedicated helpers). Older rows
may still embed `python_facade=full` inside `notes`; prefer the dedicated key
for new / updated rows.

Recorded black-box outputs live under `conformance/**/expected.json` in a
`reference` block (`project`, pin, command, `outputs`). Runtime and ordinary CI never install upstream packages. Retain independently
authored black-box generation scripts and locked comparator environments; execute
them in isolated tooling, outside the runtime environment. Keep the frozen
`reference.command` and environment pins as the audit trail. Historical deleted
generators remain reproducibility gaps until their replacements are checked.

`python3 scripts/evidence_audit.py report` joins existing claims, test sources,
fixtures, method specifications and provenance. `--json` retains the complete
links, source hashes and review limitations. This is a source inventory, not
proof that tests ran or that citations/algorithms are scientifically sufficient.
`python3 scripts/verify_provenance_citations.py --check` checks DOI/title identity
offline against retained Crossref/DataCite metadata; `--fetch` explicitly refreshes
unresolved public metadata. Exact theorem/algorithm correspondence is a separate
review. Coverage measurement is the separate calibration workflow, run on
development machines (`docs/development.md`).

Permanent product contracts are marked `done` with an inline note. There is no
waiver / deviation status.

Do not mark a capability `done` without conformance fixtures under
`conformance/` **or** a named calibration/unit harness recorded in the
corresponding feature gate script, plus a recorded reference-output generation
command where black-box comparison applies.

## Exit gates

```bash
bash scripts/gate_parity_schema.sh
python3 scripts/check_parity_inventories.py
bash scripts/gate_composition.sh
bash scripts/gate_transport.sh
bash scripts/gate_gcm_conformance.sh
bash scripts/gate_upstream_names.sh
bash scripts/gate_metadata_consistency.sh
bash scripts/gate_evidence_reachability.sh
python3 scripts/calibration_status.py --check
bash scripts/gate_release.sh
```

`scripts/measure_calibration_groups.sh` is the statistical calibration suite (SE
coverage, CI Type I / permutation uniformity, discovery null FPR). It never runs
in CI, but CI fails while any coverage record is owed
(`python3 scripts/calibration_status.py --check`); calibration status never
changes support. A change to statistical code is measured on
a development machine with `bash scripts/measure_calibration.sh`, which runs only
the groups behind records that are not current and collects them. A record is
current while the dependency fingerprint of its test (`scripts/evidence_fingerprint.py`,
stored as the record's `fingerprint`) equals the tree's, or while a passing probe
(`scripts/calibration_probe.py`: re-running the record's first replicates
reproduces its stored digests bit for bit, recorded in
`calibration_probe_attestations.toml`) was made at a tree with that fingerprint.
`python3 scripts/calibration_status.py status` prints the result (see
`docs/development.md`).

PAG: inventory (`pag.toml`), LPCMCI / latent-projection / envelope /
DAG-only-reject conformance; static FCI/RFCI `done`. Permanent: PAG-native
ID is generalized adjustment only; full ID/IDC needs MAG/ADMG completion.

Context: J/RPCMCI, effects, conditional ATE gated by `check_parity_inventories.py`.
RPCMCI uses caller-supplied regime labels; unsupervised regime search is out of scope.

Release: artifact format freeze, wheel matrix, conformance docs, hot-path
baselines, security review (`release.toml`, ADR 0017). Package version is
whatever `[workspace.package].version` in `Cargo.toml` says — do not restate it
here.

Capability: `capability_records.toml` states each capability outside the matrix
(a supported unit; `scripts/gate_support.py` counts it once; theorem, graph class,
population semantics, evidence family, provider, estimand, inference claim,
bounds, refusal codes, identity inputs, wire changes, fixture ids, optional
coverage-record ids). A record is `supported` or `not_supported`. A supported
record cites executed positive, negative and artifact fixtures (a budget fixture
for a search, an independent construction fixture for an `interval` claim); to be
a supported unit it also needs a provenance record and a `truth` or `parity`
fixture (an independent check). Its routes are entry points that run, and every route it keeps closed cites an executed runtime-refusal
test. A `not_supported` record may cite planned fixtures, and every one of its
routes stays refused in release builds; planned fixture ids do not count as
executed evidence. `inference_claim` is `none`, `point_only`, `interval`,
`assumption_range` or `structural_envelope`. Calibration is an annotation, not a
gate: listed coverage-record ids must exist, but support never waits on a
measurement. `gate_capability_records.sh` enforces this by reading source
(`check_capability_records.py` resolves every cited test, then
`run_evidence_rows.py --static` resolves each cited Rust test to its exact,
non-ignored test and each Python test to its exact node); it executes nothing. The cited tests run in the
engineering lane (CI and locally) or, for scientific assertions, in the scientific
lane on development machines; CI's `cited-tests` shards check that each one
compiles and is not ignored. Every new bounded
search runs under `antecedent_core::SearchBudget` (operation and depth limits,
plus the context's cancellation and hard memory limit on every charge), charged
inside a loop or recursion that live code reaches, and a supported search's
budget fixture must exercise it. The capability and release gates validate the
registry.
