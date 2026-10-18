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
bash scripts/gate_gcm_conformance.sh --self-test
bash scripts/gate_promotion.sh --self-test
python3 scripts/release_evidence_report.py --self-test
python3 scripts/decision_exit_report_cli_selftest.py
python3 scripts/check_prerequisites.py --self-test
python3 scripts/check_acceptance_manifest.py --self-test
python3 scripts/test_check_acceptance_manifest.py
python3 scripts/test_check_release_closure.py
python3 scripts/test_calibration_recipe_receipts.py
python3 scripts/test_generate_calibration_backlog.py
python3 scripts/test_ci_workflow.py
bash scripts/gate_scale_cases.sh --self-test
python3 scripts/run_evidence_rows_selftest.py
python3 scripts/test_calibration_groups.py
python3 scripts/test_calibration_decision.py
python3 scripts/test_calibration_family.py
python3 scripts/test_calibration_decision_logs.py
python3 scripts/test_calibration_key_aliases.py
python3 scripts/test_collect_coverage_records.py
python3 scripts/test_coverage_snapshot_attestation.py
python3 scripts/test_promotion_refusal_helpers.py
python3 scripts/test_publish_crates.py
python3 scripts/test_release_support_delta.py
python3 scripts/test_rust_source.py
python3 scripts/test_rust_equivalence.py
python3 scripts/test_calibration_dependency_facets.py
python3 scripts/test_transport_replay_contract.py
python3 scripts/verify_release_selftest.py
python3 scripts/verify_wheel_matrix_selftest.py
bash scripts/gate_release_candidate.sh --self-test
bash scripts/gate_calibration_attestation.sh --self-test
bash scripts/gate_coverage_citations.sh --self-test
bash scripts/gate_evidence_reachability.sh --self-test
bash scripts/gate_metadata_consistency.sh --self-test
bash scripts/gate_support_matrix.sh --self-test
bash scripts/gate_composition_exit.sh --self-test
bash scripts/gate_decision_exit.sh --self-test
python3 scripts/check_limits_agreement.py --self-test
python3 scripts/check_release_claims.py --self-test
python3 scripts/check_python_surface.py --self-test
bash scripts/gate_graphless_support.sh --self-test
bash scripts/gate_named_tests.sh --self-test
echo "gate self-tests: ok"
