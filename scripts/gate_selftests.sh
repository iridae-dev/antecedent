#!/usr/bin/env bash
# Gate self-tests: every gate that decides a release must fail on deliberately
# broken input, or its green result proves nothing.
#
# Each case builds an overlay of the repo and runs a whole gate against it, so this
# takes tens of minutes. It is not part of scripts/gate_release.sh and CI does not
# run it. Run it after changing a gate, its self-test cases
# (scripts/selftest_cases.py), or the registries the gates read:
#   bash scripts/gate_selftests.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== gate self-tests (broken inputs must fail) =="
bash scripts/gate_parity_schema.sh --self-test
bash scripts/gate_docs_support_matrix.sh --self-test
bash scripts/gate_composition.sh --self-test
bash scripts/gate_transport.sh --self-test
bash scripts/gate_promotion.sh --self-test
python3 scripts/promotion_selftest.py --release 2.3
python3 scripts/release_evidence_report.py --self-test
python3 scripts/b_exit_report_cli_selftest.py
python3 scripts/check_2_3_prerequisites.py --self-test
python3 scripts/run_evidence_rows_selftest.py
python3 scripts/test_calibration_groups.py
python3 scripts/verify_wheel_matrix_selftest.py
bash scripts/gate_release_candidate.sh --self-test
bash scripts/gate_calibration_attestation.sh --self-test
bash scripts/gate_coverage_citations.sh --self-test
bash scripts/gate_evidence_reachability.sh --self-test
bash scripts/gate_metadata_consistency.sh --self-test
bash scripts/gate_support_matrix.sh --self-test
bash scripts/gate_a_exit.sh --self-test
bash scripts/gate_b_exit.sh --self-test
python3 scripts/check_limits_agreement.py --self-test
python3 scripts/check_limits_agreement.py --release 2.3 --self-test
python3 scripts/check_release_claims.py --self-test
bash scripts/gate_graphless_support.sh --self-test
bash scripts/gate_named_tests.sh --self-test
echo "gate self-tests: ok"
