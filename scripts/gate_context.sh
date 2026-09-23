#!/usr/bin/env bash
# Context / regime / effects gate: inventory honesty + fixtures + benches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source scripts/python_smoke.sh

bash scripts/gate_parity_schema.sh

python3 - <<'PY'
import sys

sys.path.insert(0, "scripts")
import parity_rows as pr

EVIDENCE = {
    "validate.functional": "crates/antecedent-validate/src/functional.rs",
    "validate.temporal_mediation": "crates/antecedent/tests/prepared_analysis.rs",

    "context.panel_data": "crates/antecedent-data/src/panel.rs",
    "context.context_graph": "crates/antecedent-graph/src/cpdag.rs",
    "context.jpcmci_plus": "crates/antecedent/tests/context_effects.rs",
    "context.rpcmci": "crates/antecedent/tests/context_effects.rs",
    "context.mediation": "crates/antecedent/tests/context_effects.rs",
    "context.mediation.nonparametric": "crates/antecedent-identify/src/path_specific.rs",
    "context.conditional": "crates/antecedent/tests/context_effects.rs",
    "context.prediction": "crates/antecedent/tests/context_effects.rs",
    "context.query_model_planned_variants": "crates/antecedent-io/src/query_wire.rs",
}

EXIT_ARTIFACTS = [
    "conformance/discovery/jpcmci_plus_two_env/expected.json",
    "conformance/discovery/jpcmci_plus_two_env_space_dummy_mv/expected.json",
    "conformance/discovery/rpcmci_two_regime/expected.json",
    "conformance/context/temporal_mediation/expected.json",
    "conformance/context/conditional_effect/expected.json",
    "conformance/context/prediction_smoke/expected.json",
    "crates/antecedent-discovery/benches/rpcmci.rs",
    "crates/antecedent-estimate/benches/temporal_mediation.rs",
    "benches/baselines/regime_mediation.md",
    "parity/context.toml",
]

problems = pr.honesty_problems("parity/context.toml", EVIDENCE)
problems += pr.exit_artifact_problems(EXIT_ARTIFACTS)
problems += pr.require_done(
    "parity/discovery.toml",
    ("discovery.jpcmci_plus", "discovery.rpcmci", "discovery.effects"),
    "Context",
)
problems += pr.require_done("parity/estimate.toml", ("estimate.conditional",), "Context")
pr.finish("Context", problems, "Context inventory evidence map OK")
PY

echo "== cargo test data / discovery / estimate / identify / facade context =="
bash scripts/counted_cargo.sh test -p antecedent-data --lib
bash scripts/counted_cargo.sh test -p antecedent-discovery --lib
bash scripts/counted_cargo.sh test -p antecedent-discovery --test jpcmci_plus_oracle_matrix
bash scripts/counted_cargo.sh test -p antecedent-discovery --test rpcmci_fixed_regime_oracle
bash scripts/counted_cargo.sh test -p antecedent-estimate --lib
bash scripts/counted_cargo.sh test -p antecedent-identify --lib temporal_mediation::
bash scripts/counted_cargo.sh test -p antecedent-validate --lib functional::
bash scripts/counted_cargo.sh test -p antecedent --test context_effects

echo "== criterion smoke (regime + mediation) =="
cargo bench -p antecedent-discovery --bench rpcmci -- --test
cargo bench -p antecedent-estimate --bench temporal_mediation -- --test

echo "== Python EventFrame / panel pooled discovery facade smoke =="
python_smoke tests/test_eventframe_discovery.py tests/test_panel_pooled_discovery.py

echo "Context gate PASSED"
