#!/usr/bin/env bash
# Gate self-tests: every gate that decides a release must fail on deliberately
# broken input, or its green result proves nothing.
#
# Each case builds an overlay of the repo and runs a whole gate against it, so this
# takes tens of minutes. It is not part of the tiers of scripts/gate_release.sh and CI does not
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
bash scripts/gate_capability_records.sh --self-test
python3 scripts/release_evidence_report.py --self-test
python3 scripts/decision_exit_report_cli_selftest.py
python3 scripts/check_prerequisites.py --self-test
python3 scripts/check_release_inventory.py --self-test
python3 scripts/check_test_registry.py --self-test
python3 scripts/test_check_acceptance_manifest.py
python3 scripts/test_check_release_closure.py
python3 scripts/test_evidence_audit.py
python3 scripts/test_ignored_test_lanes.py
python3 scripts/test_check_counterparts.py
python3 scripts/test_joint_panel_reference.py
python3 scripts/test_closure_audit.py
python3 scripts/test_closure_report_rows.py
python3 scripts/test_route_manifest_diff.py
python3 scripts/test_scenario_matrix_pending.py
python3 scripts/test_gate_compiler_bypass.py
python3 scripts/test_penalty_resource_guards.py
python3 scripts/test_check_architecture.py
python3 scripts/test_gate_retained_routes.py
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
python3 scripts/test_capability_refusal_helpers.py
python3 scripts/test_publish_crates.py
python3 scripts/test_gate_support.py
python3 scripts/test_rust_source.py
python3 scripts/test_rust_equivalence.py
python3 scripts/test_calibration_build_cache.py
python3 scripts/test_evidence_fingerprint.py
python3 scripts/test_calibration_status.py
python3 scripts/test_calibration_readiness_parsing.py
python3 scripts/selftest_cases.py calibration
python3 scripts/test_temporal_capture_transfer.py
python3 scripts/test_calibration_probe.py
python3 scripts/test_transport_replay_contract.py
python3 scripts/verify_release_selftest.py
python3 scripts/verify_wheel_matrix_selftest.py
bash scripts/gate_release_candidate.sh --self-test
bash scripts/gate_evidence_reachability.sh --self-test
bash scripts/gate_metadata_consistency.sh --self-test
python3 scripts/test_evidence_selftest.py
bash scripts/gate_composition_exit.sh --self-test
bash scripts/gate_decision_exit.sh --self-test
python3 scripts/check_limits_agreement.py --self-test
python3 scripts/check_release_claims.py --self-test
python3 scripts/check_python_surface.py --self-test
python3 scripts/gate_support.py --self-test
echo "gate self-tests: ok"
