#!/usr/bin/env bash
# Phase 11 design / state gate: inventory honesty + fixtures + benches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path(".")
text = (root / "parity/phase11.toml").read_text()

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
            "phase": int(g("phase", "0") or 0),
        })
    return out

EVIDENCE = {
    "phase11.candidate_types": "crates/causal-design/src/candidate.rs",
    "phase11.eig": "crates/causal-design/src/ranker.rs",
    "phase11.id_probability": "crates/causal-design/src/ranker.rs",
    "phase11.effect_width": "crates/causal-design/src/ranker.rs",
    "phase11.decision_utility": "crates/causal-design/src/decision.rs",
    "phase11.design_ranker": "crates/causal-design/src/ranker.rs",
    "phase11.causal_state": "crates/causal-state/src/state.rs",
    "phase11.incremental_ols": "crates/causal-state/src/suff_stats.rs",
    "phase11.streaming_cov": "crates/causal-state/src/suff_stats.rs",
    "phase11.cache_budget": "crates/causal-state/src/store.rs",
    "phase11.facade": "crates/causal/tests/phase11_design_state.rs",
}

missing = []
for c in caps(text):
    if c["phase"] != 11:
        continue
    if c["status"] not in ("done", "intentional_deviation"):
        continue
    ev = EVIDENCE.get(c["id"])
    if not ev:
        missing.append(f"{c['id']} (status={c['status']}) has no evidence mapping")
        continue
    if not (root / ev).exists():
        missing.append(f"{c['id']} evidence missing: {ev}")

for path in [
    "conformance/phase11/rank_candidates_eig/expected.json",
    "conformance/phase11/incremental_ols_match/expected.json",
    "crates/causal-design/benches/design_rank.rs",
    "crates/causal-state/benches/state_append.rs",
    "benches/baselines/phase11_design_state.md",
    "parity/phase11_deviations.md",
    "parity/phase11.toml",
    "adr/0016-phase11-design-state.md",
    "provenance/design.eig.toml",
    "provenance/state.incremental_ols.toml",
]:
    if not (root / path).exists():
        missing.append(f"required exit artifact missing: {path}")

if missing:
    print("Phase 11 gate FAILED:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

print("Phase 11 inventory evidence map OK")
PY

echo "== cargo test design / state / facade phase11 =="
cargo test -p causal-design --lib
cargo test -p causal-state --lib
cargo test -p causal --test phase11_design_state

echo "== criterion smoke (design + state) =="
cargo bench -p causal-design --bench design_rank -- --test
cargo bench -p causal-state --bench state_append -- --test

echo "Phase 11 gate PASSED"
