#!/usr/bin/env bash
# PAG / LPCMCI gate: inventory honesty + fixtures + benches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path(".")
text = (root / "parity/pag.toml").read_text()

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
    "pag.graph.admg": "crates/causal-graph/src/admg.rs",
    "pag.graph.pag_temporal": "crates/causal-graph/src/pag.rs",
    "pag.graph.m_separation": "crates/causal-graph/src/msep.rs",
    "pag.graph.latent_projection": "crates/causal/tests/pag.rs",
    "pag.graph.completions_streamed": "crates/causal-graph/src/completion.rs",
    "pag.identify.generalized_adjustment": "crates/causal/tests/pag.rs",
    "pag.discovery.lpcmci": "crates/causal/tests/pag.rs",
    "pag.facade.dag_only_reject": "crates/causal/tests/pag.rs",
    "pag.discovery.fci_rfci": "parity/pag_deviations.md",
    "pag.identify.full_id_idc": "parity/pag_deviations.md",
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
    "conformance/pag/lpcmci_chain/expected.json",
    "conformance/pag/latent_projection_msep/expected.json",
    "conformance/pag/envelope_unidentified_mass/expected.json",
    "conformance/pag/dag_only_pag_reject/expected.json",
    "crates/causal-graph/benches/mseparation.rs",
    "crates/causal-discovery/benches/pag_orientation.rs",
    "benches/baselines/pag.md",
    "parity/pag_deviations.md",
    "parity/pag.toml",
]:
    if not (root / path).exists():
        missing.append(f"required exit artifact missing: {path}")

# Coarse tigramite rows
tigramite = (root / "parity/tigramite.toml").read_text()
for cid in ("tigramite.discovery.lpcmci", "tigramite.graphs.separation"):
    block = None
    for b in re.split(r"\n\[\[capabilities\]\]\n", tigramite)[1:]:
        if re.search(rf'^id\s*=\s*"{cid}"', b, re.M):
            block = b
            break
    if not block:
        missing.append(f"{cid} missing from tigramite.toml")
        continue
    m = re.search(r'^status\s*=\s*"([^"]*)"', block, re.M)
    if not m or m.group(1) != "done":
        missing.append(f"{cid} must be status=done when PAG gate passes")

if missing:
    print("PAG gate FAILED:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

print("PAG inventory evidence map OK")
PY

echo "== cargo test graph / discovery LPCMCI / identify / facade pag =="
cargo test -p causal-graph --lib
cargo test -p causal-discovery --lib
cargo test -p causal-identify --lib
cargo test -p causal --test pag
cargo test -p causal --lib refuses_dag_only

echo "== criterion smoke (m-sep + PAG orientation) =="
cargo bench -p causal-graph --bench mseparation -- --test
cargo bench -p causal-discovery --bench pag_orientation -- --test

echo "PAG gate PASSED"
