#!/usr/bin/env bash
# Context / regime / effects gate: inventory honesty + fixtures + benches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path(".")
text = (root / "parity/context.toml").read_text()

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
    "context.panel_data": "crates/causal-data/src/panel.rs",
    "context.context_graph": "crates/causal-graph/src/cpdag.rs",
    "context.jpcmci_plus": "crates/causal/tests/context_effects.rs",
    "context.rpcmci": "crates/causal/tests/context_effects.rs",
    "context.mediation": "crates/causal/tests/context_effects.rs",
    "context.conditional": "crates/causal/tests/context_effects.rs",
    "context.prediction": "crates/causal/tests/context_effects.rs",
    "context.mediation.nonparametric": "parity/context_deviations.md",
}

missing = []
for c in caps(text):
    if c["status"] not in ("done", "intentional_deviation"):
        continue
    ev = EVIDENCE.get(c["id"])
    if not ev:
        missing.append(f"{c['id']} (status={c['status']}) has no evidence mapping")
        continue
    if not (root / ev).exists():
        missing.append(f"{c['id']} evidence missing: {ev}")

for path in [
    "conformance/tigramite/jpcmci_plus_two_env/expected.json",
    "conformance/tigramite/rpcmci_two_regime/expected.json",
    "conformance/context/temporal_mediation/expected.json",
    "conformance/context/conditional_effect/expected.json",
    "conformance/context/prediction_smoke/expected.json",
    "crates/causal-discovery/benches/rpcmci.rs",
    "crates/causal-estimate/benches/temporal_mediation.rs",
    "benches/baselines/regime_mediation.md",
    "parity/context_deviations.md",
    "parity/context.toml",
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
        missing.append(f"{cid} must be status=done when Context gate passes")

for cid in (
    "tigramite.discovery.jpcmci_plus",
    "tigramite.discovery.rpcmci",
    "tigramite.effects",
):
    require_done("parity/tigramite.toml", cid)
require_done("parity/dowhy.toml", "dowhy.estimate.conditional")

if missing:
    print("Context gate FAILED:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

print("Context inventory evidence map OK")
PY

echo "== cargo test data / discovery / estimate / identify / facade context =="
cargo test -p causal-data --lib
cargo test -p causal-discovery --lib
cargo test -p causal-estimate --lib
cargo test -p causal-identify --lib temporal_mediation::
cargo test -p causal --test context_effects

echo "== criterion smoke (regime + mediation) =="
cargo bench -p causal-discovery --bench rpcmci -- --test
cargo bench -p causal-estimate --bench temporal_mediation -- --test

echo "Context gate PASSED"
