# Practitioner acceptance for 1.11

Workstream S ships with 1.11. The independent suite lives in the sibling
`../antecedent-practitioner-scenarios` repository. It exercises installed Python
and public `antecedent` Rust APIs; its runner does not import library test helpers
or read the parity inventory. The library's consuming regression tests remain
in this repository.

## Run against a candidate

Build a fresh release wheel, then run the sibling suite with explicit prerelease
overrides. Substitute the wheel filename produced by maturin:

```bash
cd python
uv run maturin build --release --out /tmp/antecedent-s-wheel
cd ../../antecedent-practitioner-scenarios
python3 run.py \
  --wheel /tmp/antecedent-s-wheel/<antecedent-1.11.0-platform-wheel>.whl \
  --rust-checkout ../causal-library \
  --scale
```

Without overrides, `python3 run.py --scale` installs `antecedent==1.11.0` and
builds against the same exact crates.io version. This mode requires the release
to have been published. The local override never makes an editable Python
installation: the runner creates its own environment and checks that imports
resolve inside it. Python artifacts are loaded in a fresh isolated interpreter;
Rust artifacts are consumed by a fresh invocation of the consumer binary.

The suite owns its dependency locks, scenario inventory, known-truth generators,
runner self-tests and `leftovers.json`. `reports/acceptance.json` records the
candidate commit, dirty-tree state and diff digest, wheel digest, Rust lock
digest, and individual Python/Rust outcomes. `reports/python.json` records the
installed dependency versions and per-job details. A smoke run without `--scale`
is useful during development but cannot accept S.

## Acceptance rule

Before cutting 1.11, require all of the following:

- Every inventoried Python and Rust job passes, including the separately
  selected 10,000- and 100,000-row jobs. Expected refusals pass only when the
  requested public route rejects for the expected reason. Missing results,
  unexpected success and mismatched artifacts fail acceptance.
- Every suite failure has a ledger record naming the public route, owner,
  scenario and supporting claim. Classify it as `bug`, `silent_refuse`,
  `inspect_execute_drift`, `docs_lie`, `missing_pin`, `unusable_default`, or
  `too_slow_to_be_true`. Close it only with a tested fix or an evidenced reason;
  do not defer an in-scope defect to 2.0 or use skips to hide it.
- Library fixes have consuming in-repo regression tests. Support changes retain
  the existing matrix and calibration obligations. A known-truth scenario is
  not a substitute for interval-coverage measurement.
- The independent report matches the actual candidate source and wheel. After
  any further library change, rerun the suite; working-tree evidence is useful
  during implementation but does not certify a later committed release.
- The existing release-candidate gate also passes against a successful CI run
  on the exact candidate commit, including the platform wheel matrix.

The suite stays outside `scripts/gate_release.sh`. Neither a green library gate
alone nor a green independent suite alone authorizes the cut. Keep both sets of
evidence. P owns timing budgets and benchmarks; S owns the scientific assertions
on practitioner-sized inputs.

## 2.0 inheritance

Transport remains a 2.0 obligation. It inherits this suite and adds public-API
transport jobs when the relevant transport milestones land. New graph types,
PAG-native full ID and other research extensions do not enter the 1.11 leftover
list merely because a practitioner might find them useful.

## Matrix edge-case suite

The sibling repository also contains a second inventory and runner in
`matrix_edges/` and `run_matrix_edges.py`. After installing the candidate wheel
with the main runner, run:

```bash
python3 run_matrix_edges.py --wheel /absolute/path/to/candidate.whl \
  --rust-checkout ../causal-library
```

These installed-Python public routes cross explicit/accepted graph classes,
inference and validation modes; test derivative, observation and likelihood
boundaries; distinguish structural disagreement from unidentified mass; and
verify artifact disclosures and integrity. Every positive cell must report the
exact expected execution coordinate. The report lists untested licensed cells;
it is not a claim of exhaustive matrix execution or interval calibration.
The second suite has its own frozen inventory, public-license snapshot,
leftover ledger, per-case subprocess isolation, and acceptance accounting tests.

## Current candidate

This section records a past candidate and does not describe the current
tree. Coverage claims for that library candidate attached to remesure
`6a41568f` plus replay waiver `facade-reexports-02c06614` (waiver `to` =
`02c06614`), and its 587 records were remesured after the finding repairs. The
2026-09-19 count below (258 matching plus 329 replay-attested) is the
pre-repair working-tree snapshot. On the current tree the coverage records owe
re-measurement (see `result.calibration`), so none of these figures is a
current coverage claim.

The sibling suite and `scripts/gate_release.sh` on 2026-09-19 were
working-tree checks. The acceptance rule above still requires a rerun of
this suite against the exact committed candidate and wheel, and
`scripts/gate_release_candidate.sh` against a successful CI run on that
commit, including the platform wheel matrix.

## Implementation evidence — 2026-09-19 working tree

The local 1.11 working-tree implementation passed 32 Python jobs, including ten
10,000/100,000-row jobs, and 11 Rust jobs. All seven runner integrity checks
passed; all eight recorded leftovers were closed with evidence. The report is
in the sibling repository at `reports/acceptance.json`.

Library validation passed the full source Python suite (1,986 tests, 91.95%
coverage), the final installed-wheel suite (1,986 tests), the consuming Rust
facade regression, Python lint/format/type checks, and `scripts/gate_release.sh`
(the inventory/composition gate, not release-candidate acceptance). Calibration
status at that snapshot reported 258 matching and 329 attested-by-replay
records, with no remesure owed under the then-current facet rule (the R1
`core` gap). Those facade-export changes did not change numerical algorithms
or extend supported scientific cells.

A separate strict Clippy check of the facade regression failed on three existing
warnings in unchanged `antecedent-core/src/execution.rs`: one potentially
truncating cast and missing error/panic documentation. Those warnings are
baseline code, not a regression from the new facade exports.

The matrix follow-up also ran strict facade-only Clippy (`--no-deps`), exposing
24 existing warnings in unchanged Rust code. Together with the three warnings
above, strict Clippy is a baseline check failure; the new contract
implementation itself produced no Clippy diagnostic.

The second inventory contains 210 cases targeting 116 distinct licensed
coordinates out of 463. The remaining 347 are explicitly listed as untested
by this suite. Ten matrix-runner accounting tests and the original seven
runner integrity tests protect acceptance. The consuming regressions add
23 Python cases, bringing the installed-wheel collection to 2,009 tests;
affected Rust contract/response and facade tests cover 153 tests. The reports
record the final outcomes, including a rerun of the original practitioner
suite and scale jobs.
