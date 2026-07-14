#!/usr/bin/env bash
# Phase 10 attribution gate: inventory honesty + fixtures + benches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path(".")
text = (root / "parity/phase10.toml").read_text()

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
    "phase10.shapley": "crates/causal-attribution/src/shapley.rs",
    "phase10.coalition_cache": "crates/causal-attribution/src/coalition.rs",
    "phase10.distribution_change": "crates/causal-attribution/src/distribution_change.rs",
    "phase10.distribution_change_robust": "crates/causal-attribution/src/robust.rs",
    "phase10.mechanism_change_detection": "crates/causal-attribution/src/mechanism_change.rs",
    "phase10.unit_change": "crates/causal-attribution/src/unit_change.rs",
    "phase10.path_decompose": "crates/causal-attribution/src/path.rs",
    "phase10.feature_relevance": "crates/causal-attribution/src/feature_relevance.rs",
    "phase10.root_cause": "crates/causal-attribution/src/root_cause.rs",
    "phase10.facade": "crates/causal/tests/phase10_attribution.rs",
}

missing = []
for c in caps(text):
    if c["phase"] != 10:
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
    "conformance/phase10/distribution_change_y_shift/expected.json",
    "conformance/phase10/mechanism_change_detect/expected.json",
    "crates/causal-attribution/benches/shapley.rs",
    "benches/baselines/phase10_shapley.md",
    "parity/phase10_deviations.md",
    "parity/phase10.toml",
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
        missing.append(f"{cid} must be status=done when Phase 10 gate passes")

for cid in (
    "gcm.attribution.shapley",
    "gcm.attribution.distribution_change",
    "gcm.attribution.robust",
    "gcm.attribution.unit_change",
    "gcm.attribution.feature_relevance",
):
    require_done("parity/gcm.toml", cid)

if missing:
    print("Phase 10 gate FAILED:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

print("Phase 10 inventory evidence map OK")
PY

echo "== cargo test attribution / facade phase10 =="
cargo test -p causal-attribution --lib
cargo test -p causal --test phase10_attribution

echo "== criterion smoke (shapley) =="
cargo bench -p causal-attribution --bench shapley -- --test

echo "Phase 10 gate PASSED"
