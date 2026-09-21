#!/usr/bin/env bash
# Attribution gate: inventory honesty + fixtures + benches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash scripts/gate_parity_schema.sh

python3 - <<'PY'
import sys

sys.path.insert(0, "scripts")
import parity_rows as pr

EVIDENCE = {
    "attribution.shapley": "crates/antecedent-attribution/src/shapley.rs",
    "attribution.coalition_cache": "crates/antecedent-attribution/src/coalition.rs",
    "attribution.distribution_change": "crates/antecedent-attribution/src/distribution_change.rs",
    "attribution.distribution_change_robust": "crates/antecedent-attribution/src/robust.rs",
    "attribution.mechanism_change_detection": "crates/antecedent-attribution/src/mechanism_change.rs",
    "attribution.mechanism_change_kernel": "crates/antecedent-stats/src/divergence.rs",
    "attribution.mechanism_change_change_point": "crates/antecedent-stats/src/divergence.rs",
    "attribution.unit_change": "crates/antecedent-attribution/src/unit_change.rs",
    "attribution.path_decompose": "crates/antecedent-attribution/src/path.rs",
    "attribution.feature_relevance": "crates/antecedent-attribution/src/feature_relevance.rs",
    "attribution.root_cause": "crates/antecedent-attribution/src/root_cause.rs",
    "attribution.structure": "crates/antecedent-attribution/src/structure_change.rs",
    "attribution.facade": "crates/antecedent/tests/attribution.rs",
}

EXIT_ARTIFACTS = [
    "conformance/attribution/path_allocation/expected.json",
    "conformance/attribution/distribution_change_grid/expected.json",
    "conformance/attribution/structure_change_grid/expected.json",
    "conformance/attribution/mechanism_unit_change/expected.json",
    "conformance/attribution/anomaly_root_cause/expected.json",
    "conformance/attribution/arrow_strength/expected.json",
    "conformance/validate/refuters/expected.json",
    "conformance/validate/confounding_sensitivity/expected.json",
    "conformance/validate/riesz_sensitivity/expected.json",
    "conformance/validate/overlap_graph_refutation/expected.json",
    "conformance/attribution/distribution_change_y_shift/expected.json",
    "conformance/attribution/structure_change_parent_swap/expected.json",
    "conformance/attribution/mechanism_change_detect/expected.json",
    "conformance/attribution/mechanism_change_kernel_shift/expected.json",
    "conformance/attribution/mechanism_change_change_point/expected.json",
    "crates/antecedent-attribution/benches/shapley.rs",
    "benches/baselines/shapley.md",
    "parity/attribution.toml",
]

problems = pr.honesty_problems("parity/attribution.toml", EVIDENCE)
problems += pr.exit_artifact_problems(EXIT_ARTIFACTS)
problems += pr.require_done("parity/gcm.toml", (
    "gcm.attribution.shapley",
    "gcm.attribution.distribution_change",
    "gcm.attribution.robust",
    "gcm.attribution.unit_change",
    "gcm.attribution.feature_relevance",
    "gcm.attribution.structure",
), "Attribution")
pr.finish("Attribution", problems, "Attribution inventory evidence map OK")
PY

echo "== cargo test attribution / facade attribution =="
bash scripts/counted_cargo.sh test -p antecedent-attribution --lib
bash scripts/counted_cargo.sh test -p antecedent --test attribution

echo "== criterion smoke (shapley) =="
cargo bench -p antecedent-attribution --bench shapley -- --test

echo "Attribution gate PASSED"
