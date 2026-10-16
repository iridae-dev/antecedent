#!/usr/bin/env bash
# Coverage citation gate.
#
# A licensed row that quotes a measured coverage cites the test that produced
# it, in the form `0.895 (static_calibration::path_specific_two_path_*)` or
# `(crates/antecedent/tests/derivative_calibration.rs::ade_..._coverage)`.
# The 2026-09-15 traceability pass found quoted numbers that no test had
# produced on the current tree; a citation that names a test which does not
# exist is the cheapest such drift to catch mechanically. This gate resolves
# every `<module>::<fn>` citation in parity/support_licensed.toml against the
# test functions actually defined in crates/antecedent/tests/<module>.rs
# (`fn name` and the `name => (...)` entries of the calibration macros),
# expanding `{a,b}` alternatives and `*` globs, and fails on any citation that
# matches no test. It extends scripts/gate_evidence_reachability.sh, which
# checks the evidence_test / evidence_assertion columns but not the prose.
#
# It then requires every coverage figure in the prose to be attributed
# (scripts/coverage_citations.py): a figure either cites the coverage record whose
# observed value it states, or says it is not a registry value (a probe or an
# earlier measurement the registry does not carry). Known-truth values, standard
# errors and pinned values are not coverage figures and are never rejected.
# Two-decimal rates and percentages are scanned too (nominal levels exempt), and a
# disclosure names its source, `(probe <module>::<test_fn>)`, resolved to a compiled
# test; the figures and bare disclosures that predate this are ratcheted in
# parity/_coverage_citation_backlog.txt, which may only shrink.
#
# Run directly, or via scripts/gate_release.sh (CI's `gates` job, every PR).
# --self-test: broken citations must fail this gate (scripts/selftest_cases.py).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--self-test" ]]; then
  exec python3 "$ROOT/scripts/selftest_cases.py" citations
fi

python3 "$ROOT/scripts/gate_coverage_citations.py"

python3 scripts/coverage_citations.py check

# The level the runtime reports each interval at, the records that measure it and the cells that
# cite them agree (scripts/check_interval_levels.py); a missing record that a registered design
# will produce on the next measurement is reported as pending, not a failure.
python3 scripts/check_interval_levels.py --self-test
python3 scripts/check_interval_levels.py
