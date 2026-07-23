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
    "pag.graph.latent_projection": "crates/antecedent/tests/pag.rs",
    "pag.graph.completions_streamed": "crates/causal-graph/src/completion.rs",
    "pag.graph.cpdag_mec_completions": "crates/causal-graph/src/cpdag_completion.rs",
    "pag.identify.generalized_adjustment": "crates/antecedent/tests/pag.rs",
    "pag.identify.full_id_idc": "crates/causal-identify/src/id.rs",
    "pag.discovery.lpcmci": "crates/antecedent/tests/pag.rs",
    "pag.discovery.fci_rfci": "crates/causal-discovery/src/fci.rs",
    "pag.facade.dag_only_reject": "crates/antecedent/tests/pag.rs",
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
    "conformance/pag/lpcmci_chain/expected.json",
    "conformance/pag/latent_projection_msep/expected.json",
    "conformance/pag/envelope_unidentified_mass/expected.json",
    "conformance/pag/dag_only_pag_reject/expected.json",
    "crates/causal-graph/benches/mseparation.rs",
    "crates/causal-discovery/benches/pag_orientation.rs",
    "benches/baselines/pag.md",
    "parity/pag.toml",
]:
    if not (root / path).exists():
        missing.append(f"required exit artifact missing: {path}")

# Domain inventory rows
discovery_inv = (root / "parity/discovery.toml").read_text()
for cid in ("discovery.lpcmci", "discovery.graphs.separation"):
    block = None
    for b in re.split(r"\n\[\[capabilities\]\]\n", discovery_inv)[1:]:
        if re.search(rf'^id\s*=\s*"{cid}"', b, re.M):
            block = b
            break
    if not block:
        missing.append(f"{cid} missing from discovery.toml")
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
cargo test -p antecedent --test pag
cargo test -p antecedent --lib refuses_dag_only

echo "== criterion smoke (m-sep + PAG orientation) =="
cargo bench -p causal-graph --bench mseparation -- --test
cargo bench -p causal-discovery --bench pag_orientation -- --test

echo "PAG gate PASSED"
