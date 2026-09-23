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
bash scripts/gate_release_candidate.sh --self-test
bash scripts/gate_calibration_attestation.sh --self-test
bash scripts/gate_coverage_citations.sh --self-test
bash scripts/gate_evidence_reachability.sh --self-test
bash scripts/gate_metadata_consistency.sh --self-test
bash scripts/gate_support_matrix.sh --self-test
echo "gate self-tests: ok"
