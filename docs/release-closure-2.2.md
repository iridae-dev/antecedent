# 2.2 release closure (B8 / X11)

This page is the maintainer procedure for cutting 2.2.0. It is process, not product:
B8 adds no cell, only the scripts below and the order in which to run them. It is not
linked from the published navigation.

## What is carried forward

- **X7 (GPU lane): carried forward to 2.3 or later, with no code in 2.2.** The 2.1 neural
  cross-fit baseline ([`benches/baselines/neural_crossfit.md`](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/neural_crossfit.md))
  is CPU-only (Burn `NdArray<f32>`), records no accelerator measurement, and says it does not
  establish a speedup. The condition for starting the lane (a measured whole-analysis gain) is not
  met, so B7 ships nothing. There is no X7 promotion record.
- **The old calibration backlog does not block the cut.** `parity/calibration_backlog.md` lists
  44 unmeasured cells across 22 coordinates, the same figure as the 2.1.0 notes, and
  `parity/reason_codes.toml` ratchets `estimator_grid_not_measured` (`max_uses`) so it can only
  shrink. Work it down when cheap (fixed-graph coordinates before graph-posterior ones; the five
  fixed-graph coordinates need a measurement DGP and are not worked in 2.2), but an unrelated old
  coordinate never holds back otherwise complete 2.2 work. What must hold is that no *new* 2.2
  interval ships under `estimator_grid_not_measured` (see the checklist).
- Calibration is measured **once**, at the cut, after B. No B package measures it earlier.

## Release-closure tooling

| Script | What it enforces | Run by |
| --- | --- | --- |
| `scripts/check_limits_agreement.py` | Each numeric `bounds` entry of every 2.2 record equals the Rust constant, the `docs/guides/transport-scope.md` statement, the Python docstring or default, and the number quoted in the record's `bounds_exceeded` refusal, anchored per bound (`refusal=r"regex"`, one capture group: a number found elsewhere in the text proves nothing). A numeric bound with no row in its `MAPPING` table fails (a `frozen` record gets a gap until `in_progress`); a non-integer bound that states numbers (`quadrature_nodes`, `tolerance`, `per_decision_operations`, `operation_limit`) is compared through a `NONINT` row or excluded there with a recorded reason. Works for any record status, `carried_forward` included. Extend `MAPPING` with additive `Bound(...)` rows for each B record. The self-test runs on a pinned synthetic tree, never the live records. `--strict` turns documentation gaps into failures. | `gate_release.sh` |
| `scripts/check_release_claims.py` | `docs/release-notes/v2.2.0.md` defines the six terms (point only, nominal, calibrated, structural envelope, assumption range, statistical interval), has one entry per record with exactly its claim word (prose forms `structural envelope` / `assumption range` and a plain `none` after "claim" count as claim words), restates the record's `Bounds`, status and guarantee (stale in draft mode, an error otherwise and in `--final`), says the interval is withheld for any record with a closed uncertainty route and that an assumption range or structural envelope is never a confidence interval, never prints a `complete` guarantee without its qualifier (`exact_range_complete_within_declared_contamination_class` carries `within the declared contamination class; sampling interval not offered`), never describes a `calibrated` record as calibrated while its coverage records are absent, and states the X7 carry-forward. `--write-draft` regenerates the skeleton from the records; `--final` fails on any DRAFT or TODO. | `gate_release.sh` (draft mode) |
| `scripts/check_interval_coordinates.py` | Zero newly introduced unmeasured interval coordinates over **every** 2.2 record (A and B): no licensed route carries `estimator_grid_not_measured`, every uncertainty route is closed with an executed refusal test or licensed with all its coverage records, every allocated coverage id (any claim or status, `carried_forward` included) is a string literal in the body of an `#[ignore]` test (a comment does not count) and a registered `run_*` group name in `scripts/gate_calibration.sh`; a `cov.*` id named in a record's text but absent from its `coverage_records` is a hidden reservation and fails; a `== 2.2...` calibration registration whose test emits a coverage record no 2.2 record owns fails. Reports per-record calibration state for every record that allocates ids. | `gate_b_exit.sh` |
| `scripts/gate_b_exit.sh` (+ `b_exit_report.py`) | Per-package table PASS / FAIL / PENDING_IMPLEMENTATION / PENDING_CALIBRATION / CARRIED_FORWARD. The story-test registry is `B_PACKAGES` in `scripts/b_exit_report.py` (empty until package owners add stories); a package reads PASS only with registered stories and an `evidence` list of route-evidence test names, each defined in a story file (and passed in its run log) or in an already-cited record fixture, at least one in a story file. An allocated coverage id never passes without a calibration reading (a carried record's unmeasured ids get a CARRIED_FORWARD calibration row). See the story list below. `--require-calibrated`, `--require-implemented`, `--release` (both). | `gate_release.sh`; the cut with `--release` |
| `scripts/gate_a_exit.sh` | The A stories and the X1/X4 calibration state. `--require-calibrated` at the cut. | `gate_release.sh`; the cut with the flag |

All of these have `--self-test`, run by `bash scripts/gate_selftests.sh`.

### B exit stories (recommended; owners register them in `B_PACKAGES`)

Each story must take its cell from clean preparation through independent artifact consumption and
its typed refusal (mirror `crates/antecedent/tests/a_exit_gate.rs`).

| Package | Story |
| --- | --- |
| B1 | prepare latent-confounded selection ADMG + conditional query, point vs enumerated truth, export, independent consume; non-transportable diagram gives `transport_not_certified`; counted laws refused |
| B2 | prepare trial + dose + grid, smoothed psi_h, export/consume; tampered artifact fails; `.interval()` refuses `cell_not_licensed` |
| B3 | z baseline, joint 2-factor deviation with tipping frontier, export/consume; re-sealed mutated threshold/factor set fails replay; union never labelled a CI |
| B4 | failed X1 (and X9) decision, plan studies, export/replay; bounds-exceeded catalog inconclusive, never "sufficient"; arrival flips the decision |
| B5 | confounded ADMG ETT vs Y0 fixture, evaluate, export/consume; non-identified event refuses; uncertainty refuses |
| B6 | binary m-graph with item-missing variables, recover joint law, feed downstream ID, export/consume; out-of-class mechanism refuses; `prepare_empirical` refuses `cell_not_licensed` |

## Cut procedure, in order

1. **Finish B.** Every B record is `in_progress` or `promoted`, each B package has registered its
   story tests in `B_PACKAGES`, and any B interval has its coverage ids allocated and its
   calibration harness wired (the route stays closed until step 4). Regenerate the draft notes
   with `python3 scripts/check_release_claims.py --write-draft` to pick up new records, then
   hand-edit the prose.
2. **Measure calibration once**, on a clean checkout of the commit being cut:
   `bash scripts/measure_calibration.sh --dry-run` to size it, then
   `bash scripts/measure_calibration.sh`. Only owed groups run: the X1 and X4 coverage records
   (`crates/antecedent/tests/mz_transport_calibration.rs`, `learned_continuous_calibration.rs`),
   any B interval records, and any old coordinates taken on.
3. **Regenerate and commit** `parity/coverage_records.toml` and
   `crates/antecedent-io/src/coverage_records_data.rs` (the collector writes both).
4. **Open the measured routes and promote.** For X1, X4 and each B interval record, flip the
   uncertainty route from `closed` (`cell_not_licensed`) to `licensed` together with its row in
   the owning registry (`parity/transport_stages.toml`, `parity/support_licensed.toml`), move the
   record to `promoted`, and remove the `calibration-internal` gating only where the record says
   it is lifted. Hand-edit `parity/transport_coverage.md` (flip the X1/X4/X5 "Closed" rows) and
   `parity/counterfactual_coverage.md` (new title, add B5); neither has a generator.
   Then `python3 scripts/check_promotion_records.py` and `bash scripts/gate_promotion.sh`.
5. **Release notes.** Write `docs/release-notes/preparation.toml` with
   `target_version = "2.2.0"`, finish `docs/release-notes/v2.2.0.md` (remove the DRAFT banner and
   every TODO; add the `## Explicit refusals` section and the licensed-block markers as in
   v2.1.1, and the nav entry in `mkdocs.yml`), then run
   `python3 scripts/generate_support_matrix_docs.py` (it freezes the v2.1.1 licensed block) and
   `python3 scripts/check_release_claims.py --final`.
6. **Exit gates.** `bash scripts/gate_a_exit.sh --require-calibrated` and
   `bash scripts/gate_b_exit.sh --release`, plus `python3 scripts/check_limits_agreement.py --strict`
   (every bound also stated in the docs), and `bash scripts/gate_release.sh`.
7. **Version.** `bash scripts/set_version.sh 2.2.0`, commit the bump as its own commit, and get a
   green CI run on that HEAD (including the `python-wheels` matrix).
8. **Tag.** `CI_RUN_ID=<run> bash scripts/tag_release.sh`. It runs
   `scripts/gate_release_candidate.sh` (clean tree required; fresh-venv install of the local
   wheel with the full pytest suite), tags `v2.2.0` and leaves the push to you.

## TODO.md B8 checklist mapped to gates

| B8 item | Mechanized by | Status |
| --- | --- | --- |
| Regenerate support, transport-stage, counterfactual, graphless, calibration and reason-code inventories | Support matrix, reason codes and coverage data: `generate_support_matrix_docs.py` re-run and `git diff` in `gate_release.sh`. Graphless: `gate_graphless_support.sh`. Transport stages: `check_transport_stages.py` via `gate_transport.sh`. Calibration backlog and readiness: `generate_calibration_backlog.py --check` and `calibration_readiness.py --check` in `gate_release.sh`. | Mechanized, except `parity/transport_coverage.md` and `parity/counterfactual_coverage.md`, which stay manual edits (step 4); needs work only if they drift unnoticed |
| Every 2.2 cell with an interval has a matching coverage record | `check_interval_coordinates.py` (all records) and `gate_a_exit.sh --require-calibrated`, `gate_b_exit.sh --release`; attestation by `gate_calibration_attestation.sh` | Mechanized |
| No newly added 2.2 interval under `estimator_grid_not_measured` | `check_promotion_records.py` rule plus `check_interval_coordinates.py` (a); the `max_uses` ratchet only stops growth of the old backlog | Mechanized |
| Old backlog reduction, fixed-graph before graph-posterior, never blocking | `parity/calibration_backlog.md` and `calibration_readiness.md` generators; policy above | Policy only, by design |
| Every promoted row's positive and negative fixtures executed | `gate_promotion.sh` | Mechanized |
| Every search path has cancellation and budget tests | `check_promotion_records.py` (`budget` fixture, `.charge(` in `search_impl`) | Mechanized per record; verify each B search declares `search = true` |
| Every durable new type has artifact compatibility / migration tests | Record `compatibility` and `wire_changes` are prose; `release.artifact_schema` covers the format chain | **Needs work**: a per-record artifact or compatibility fixture role, or a manual checklist line per B package |
| Every artifact mutation of premises, source identity, data identity or result fails verification | Per-record mutation fixtures (for example `x1.mutated_artifact.negative`) | **Needs work**: convention only, no generic rule for B records |
| `inspect()` exposes unsuccessful scenarios and searches | Per-record fixtures (X2, X9) | **Needs work**: a fixture per B cell or a manual line |
| Python and Rust docs agree on limits | `check_limits_agreement.py` | Mechanized for the mapped bounds; documentation gaps are listed (X4 folds, X8 caps, X9 caps) and close under `--strict` once the statements are written |
| Release notes distinguish point-only, nominal, calibrated, structural envelope, assumption range and statistical interval | `check_release_claims.py` (`--final` at the cut) | Mechanized |
| Generated-doc cleanliness | `gate_release.sh` (conformance, support matrix, historical notes untouched) | Mechanized |
| Fresh-install Python package tests | `gate_release_candidate.sh` step 4 and the CI `python-wheels` matrix (`CI_RUN_ID`) | Mechanized |
| Cut the version only after B | Step 7 above | Procedural |
| X7 carried forward | This page, the draft notes, `b_exit_report.py` row "B7 X7" | Recorded, no gate needed |
