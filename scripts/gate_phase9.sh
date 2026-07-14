#!/usr/bin/env bash
# Phase 9 context/regime/effects gate: inventory honesty + fixtures + benches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path(".")
text = (root / "parity/phase9.toml").read_text()

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
    "phase9.panel_data": "crates/causal-data/src/panel.rs",
    "phase9.context_graph": "crates/causal-graph/src/cpdag.rs",
    "phase9.jpcmci_plus": "conformance/phase9/jpcmci_plus_two_env",
    "phase9.rpcmci": "conformance/phase9/rpcmci_two_regime",
    "phase9.mediation": "conformance/phase9/temporal_mediation",
    "phase9.conditional": "conformance/phase9/conditional_effect",
    "phase9.prediction": "conformance/phase9/prediction_smoke",
}

missing = []
for c in caps(text):
    if c["phase"] != 9:
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
    "conformance/phase9/jpcmci_plus_two_env/expected.json",
    "conformance/phase9/rpcmci_two_regime/expected.json",
    "conformance/phase9/temporal_mediation/expected.json",
    "conformance/phase9/conditional_effect/expected.json",
    "conformance/phase9/prediction_smoke/expected.json",
    "crates/causal-discovery/benches/rpcmci.rs",
    "crates/causal-estimate/benches/temporal_mediation.rs",
    "benches/baselines/phase9_regime_mediation.md",
    "parity/phase9_deviations.md",
    "parity/phase9.toml",
]:
    if not (root / path).exists():
        missing.append(f"required exit artifact missing: {path}")

# Coarse inventory flips
def require_done(path, cid):
    text = (root / path).read_text()
    block = None
    for b in re.split(r"\n\[\[capabilities\]\]\n", text)[1:]:
        if re.search(rf'^id\s*=\s*"{cid}"', b, re.M):
            block = b
            break
    if not block:
        missing.append(f"{cid} missing from {path}")
        return
    m = re.search(r'^status\s*=\s*"([^"]*)"', block, re.M)
    if not m or m.group(1) != "done":
        missing.append(f"{cid} must be status=done when Phase 9 gate passes")

for cid in (
    "tigramite.discovery.jpcmci_plus",
    "tigramite.discovery.rpcmci",
    "tigramite.effects",
):
    require_done("parity/tigramite.toml", cid)
require_done("parity/dowhy.toml", "dowhy.estimate.conditional")

if missing:
    print("Phase 9 gate FAILED:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

print("Phase 9 inventory evidence map OK")
PY

echo "== cargo test data / discovery / estimate / identify / facade phase9 =="
cargo test -p causal-data --lib
cargo test -p causal-discovery --lib
cargo test -p causal-estimate --lib
cargo test -p causal-identify --lib temporal_mediation::
cargo test -p causal --test phase9_context_effects

echo "== criterion smoke (regime + mediation) =="
cargo bench -p causal-discovery --bench rpcmci -- --test
cargo bench -p causal-estimate --bench temporal_mediation -- --test

echo "Phase 9 gate PASSED"
