#!/usr/bin/env bash
# Design / incremental-state gate: inventory honesty + fixtures + benches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path(".")
text = (root / "parity/design_state.toml").read_text()

def caps(text: str):
    blocks = re.split(r"\n\[\[capabilities\]\]\n", text)[1:]
    out = []
    for b in blocks:
        def g(k, default=None):
            m = re.search(rf'^{k}\s*=\s*"([^"]*)"', b, re.M)
            if m:
                return m.group(1)
            m = re.search(rf'^{k}\s*=\s*(\d+)', b, re.M)
            return m.group(1) if m else default
        out.append({
            "id": g("id"),
            "status": g("status"),
        })
    return out

EVIDENCE = {
    "design_state.candidate_types": "crates/causal-design/src/candidate.rs",
    "design_state.eig": "crates/causal-design/src/ranker.rs",
    "design_state.id_probability": "crates/causal-design/src/ranker.rs",
    "design_state.effect_width": "crates/causal-design/src/ranker.rs",
    "design_state.decision_utility": "crates/causal-design/src/decision.rs",
    "design_state.design_ranker": "crates/causal-design/src/ranker.rs",
    "design_state.causal_state": "crates/causal-state/src/state.rs",
    "design_state.incremental_ols": "crates/causal-state/src/suff_stats.rs",
    "design_state.streaming_cov": "crates/causal-state/src/suff_stats.rs",
    "design_state.cache_budget": "crates/causal-state/src/store.rs",
    "design_state.facade": "crates/causal/tests/design_state.rs",
    "design_state.incremental.particle_graph_score": "crates/causal-state/src/graph_score.rs",
}

missing = []
for c in caps(text):
    if c["status"] == "intentional_deviation":
        missing.append(f"{c['id']}: intentional_deviation is retired; use pending (TODO.md) or done")
        continue
    if c["status"] != "done":
        continue
    ev = EVIDENCE.get(c["id"])
    if not ev:
        missing.append(f"{c['id']} (status={c['status']}) has no evidence mapping")
        continue
    if not (root / ev).exists():
        missing.append(f"{c['id']} evidence missing: {ev}")

for path in [
    "conformance/design_state/rank_candidates_eig/expected.json",
    "conformance/design_state/incremental_ols_match/expected.json",
    "conformance/design_state/incremental_graph_score_match/expected.json",
    "conformance/design_state/incremental_particle_filter_match/expected.json",
    "crates/causal-design/benches/design_rank.rs",
    "crates/causal-state/benches/state_append.rs",
    "benches/baselines/design_state.md",
    "parity/design_state.toml",
    "adr/0016-design-state.md",
    "provenance/design.eig.toml",
    "provenance/state.incremental_ols.toml",
    "provenance/state.incremental.particle_graph_score.toml",
    "TODO.md",
]:
    if not (root / path).exists():
        missing.append(f"required exit artifact missing: {path}")

if missing:
    print("Design state gate FAILED:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

print("Design state inventory evidence map OK")
PY

echo "== cargo test design / state / facade design_state =="
cargo test -p causal-design --lib
cargo test -p causal-state --lib
cargo test -p causal --test design_state

echo "== criterion smoke (design + state) =="
cargo bench -p causal-design --bench design_rank -- --test
cargo bench -p causal-state --bench state_append -- --test

echo "Design state gate PASSED"
