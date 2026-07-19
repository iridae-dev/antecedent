#!/usr/bin/env bash
# Attribution gate: inventory honesty + fixtures + benches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path(".")
text = (root / "parity/attribution.toml").read_text()

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
    "attribution.shapley": "crates/causal-attribution/src/shapley.rs",
    "attribution.coalition_cache": "crates/causal-attribution/src/coalition.rs",
    "attribution.distribution_change": "crates/causal-attribution/src/distribution_change.rs",
    "attribution.distribution_change_robust": "crates/causal-attribution/src/robust.rs",
    "attribution.mechanism_change_detection": "crates/causal-attribution/src/mechanism_change.rs",
    "attribution.mechanism_change_kernel": "crates/causal-stats/src/divergence.rs",
    "attribution.mechanism_change_change_point": "crates/causal-stats/src/divergence.rs",
    "attribution.unit_change": "crates/causal-attribution/src/unit_change.rs",
    "attribution.path_decompose": "crates/causal-attribution/src/path.rs",
    "attribution.feature_relevance": "crates/causal-attribution/src/feature_relevance.rs",
    "attribution.root_cause": "crates/causal-attribution/src/root_cause.rs",
    "attribution.structure": "crates/causal-attribution/src/structure_change.rs",
    "attribution.facade": "crates/causal/tests/attribution.rs",
}

missing = []
for c in caps(text):
    if c["status"] == "intentional_deviation":
        missing.append(f"{c['id']}: intentional_deviation is retired; use pending or done")
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
    "conformance/attribution/distribution_change_y_shift/expected.json",
    "conformance/attribution/structure_change_parent_swap/expected.json",
    "conformance/attribution/mechanism_change_detect/expected.json",
    "conformance/attribution/mechanism_change_kernel_shift/expected.json",
    "conformance/attribution/mechanism_change_change_point/expected.json",
    "crates/causal-attribution/benches/shapley.rs",
    "benches/baselines/shapley.md",
    "parity/attribution.toml",
]:
    if not (root / path).exists():
        missing.append(f"required exit artifact missing: {path}")

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
        missing.append(f"{cid} must be status=done when Attribution gate passes")

for cid in (
    "gcm.attribution.shapley",
    "gcm.attribution.distribution_change",
    "gcm.attribution.robust",
    "gcm.attribution.unit_change",
    "gcm.attribution.feature_relevance",
    "gcm.attribution.structure",
):
    require_done("parity/gcm.toml", cid)

if missing:
    print("Attribution gate FAILED:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

print("Attribution inventory evidence map OK")
PY

echo "== cargo test attribution / facade attribution =="
cargo test -p causal-attribution --lib
cargo test -p causal --test attribution

echo "== criterion smoke (shapley) =="
cargo bench -p causal-attribution --bench shapley -- --test

echo "Attribution gate PASSED"
