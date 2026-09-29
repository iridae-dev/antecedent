#!/usr/bin/env bash
# Design / incremental-state gate: inventory honesty. Rust fixtures run in the Rust job.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash scripts/gate_parity_schema.sh

python3 - <<'PY'
import sys

sys.path.insert(0, "scripts")
import parity_rows as pr

EVIDENCE = {
    "design_state.candidate_types": "crates/antecedent-design/src/candidate.rs",
    "design_state.eig": "crates/antecedent-design/src/ranker.rs",
    "design_state.id_probability": "crates/antecedent-design/src/ranker.rs",
    "design_state.effect_width": "crates/antecedent-design/src/ranker.rs",
    "design_state.decision_utility": "crates/antecedent-design/src/decision.rs",
    "design_state.design_ranker": "crates/antecedent-design/src/ranker.rs",
    "design_state.antecedent_state": "crates/antecedent-state/src/state.rs",
    "design_state.incremental_ols": "crates/antecedent-state/src/suff_stats.rs",
    "design_state.streaming_cov": "crates/antecedent-state/src/suff_stats.rs",
    "design_state.cache_budget": "crates/antecedent-state/src/store.rs",
    "design_state.facade": "crates/antecedent/tests/design_state.rs",
    "design_state.incremental.particle_graph_score": "crates/antecedent-state/src/graph_score.rs",
    "design_state.rolling_mechanism_diagnostics": "crates/antecedent-state/src/mechanism_diag.rs",
}

EXIT_ARTIFACTS = [
    "conformance/design_state/rank_candidates_eig/expected.json",
    "conformance/design_state/incremental_ols_match/expected.json",
    "conformance/design_state/incremental_graph_score_match/expected.json",
    "conformance/design_state/incremental_particle_filter_match/expected.json",
    "conformance/design_state/rolling_mechanism_diag_match/expected.json",
    "crates/antecedent-design/benches/design_rank.rs",
    "crates/antecedent-state/benches/state_append.rs",
    "benches/baselines/design_state.md",
    "parity/design_state.toml",
    "adr/0016-design-state.md",
    "provenance/design.eig.toml",
    "provenance/state.incremental_ols.toml",
    "provenance/state.incremental.particle_graph_score.toml",
]

problems = pr.honesty_problems("parity/design_state.toml", EVIDENCE)
problems += pr.exit_artifact_problems(EXIT_ARTIFACTS)
pr.finish("Design state", problems, "Design state inventory evidence map OK")
PY

# Rust suites and the Criterion smoke run in the Rust job and gate_release.sh.
echo "Design state gate PASSED"
