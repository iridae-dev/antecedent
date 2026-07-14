#!/usr/bin/env bash
# Phase 12 release gate: inventory honesty, docs, artifacts, security, benches.
# Invokes prior phase gates unless SKIP_PRIOR_GATES=1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${SKIP_PRIOR_GATES:-0}" != "1" ]]; then
  echo "== prior phase gates =="
  bash scripts/gate_phase45_parity.sh
  bash scripts/gate_phase6.sh
  bash scripts/gate_phase7.sh
  bash scripts/gate_phase8.sh
  bash scripts/gate_phase9.sh
  bash scripts/gate_phase10.sh
  bash scripts/gate_phase11.sh
fi

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path(".")

def caps(path: Path):
    text = path.read_text()
    blocks = re.split(r"\n\[\[capabilities\]\]\n", text)[1:]
    out = []
    for b in blocks:
        def g(k, default=None):
            m = re.search(rf'^{k}\s*=\s*"([^"]*)"', b, re.M)
            if m:
                return m.group(1)
            m = re.search(rf'^{k}\s*=\s*(\d+)', b, re.M)
            return m.group(1) if m else default
        out.append({"id": g("id"), "status": g("status"), "phase": int(g("phase", "0") or 0)})
    return out

missing = []

# No required pending/planned/blocked in DoWhy / Tigramite capability tables.
for manifest in ["parity/dowhy.toml", "parity/tigramite.toml"]:
    for c in caps(Path(manifest)):
        if c["status"] in ("pending", "planned", "blocked", "in_progress"):
            missing.append(f"{manifest}: {c['id']} still {c['status']}")

EVIDENCE = {
    "phase12.parity_closure": "parity/phase12_deviations.md",
    "phase12.graph_dot_json": "crates/causal-io/src/graph_dot.rs",
    "phase12.artifact_schema": "crates/causal-io/src/migrate.rs",
    "phase12.wheel_matrix": ".github/workflows/ci.yml",
    "phase12.conformance_docs": "docs/conformance/README.md",
    "phase12.hot_path_baselines": "docs/hot_paths.md",
    "phase12.security_review": "docs/security_review.md",
}

for c in caps(Path("parity/phase12.toml")):
    if c["phase"] != 12:
        continue
    if c["status"] not in ("done", "intentional_deviation"):
        missing.append(f"phase12.toml {c['id']} status={c['status']}")
        continue
    ev = EVIDENCE.get(c["id"])
    if not ev:
        missing.append(f"{c['id']} has no evidence mapping")
    elif not (root / ev).exists():
        missing.append(f"{c['id']} evidence missing: {ev}")

for path in [
    "adr/0017-phase12-1.0-prep.md",
    "parity/phase12.toml",
    "parity/phase12_deviations.md",
    "docs/artifacts.md",
    "docs/hot_paths.md",
    "docs/security_review.md",
    "docs/conformance/README.md",
    "deny.toml",
    "conformance/phase12/graph_dot_json/expected.json",
    "conformance/phase12/artifact_migrate/expected.json",
    "crates/causal-io/src/graph_dot.rs",
    "crates/causal-io/src/graph_json.rs",
    "crates/causal-io/src/migrate.rs",
    "scripts/generate_conformance_docs.py",
]:
    if not (root / path).exists():
        missing.append(f"required exit artifact missing: {path}")

# Semantic crates must forbid unsafe_code.
semantic = [
    "crates/causal-core",
    "crates/causal-data",
    "crates/causal-graph",
    "crates/causal-expr",
    "crates/causal-io",
    "crates/causal-identify",
    "crates/causal-stats",
    "crates/causal-prob",
    "crates/causal-estimate",
    "crates/causal-validate",
    "crates/causal-model",
    "crates/causal-counterfactual",
    "crates/causal-attribution",
    "crates/causal-design",
    "crates/causal-state",
    "crates/causal-discovery",
    "crates/causal",
]
for crate in semantic:
    lib = root / crate / "src" / "lib.rs"
    text = lib.read_text()
    if "#![forbid(unsafe_code)]" not in text:
        missing.append(f"{crate} missing #![forbid(unsafe_code)]")

# Baseline files referenced by hot_paths index.
hot = (root / "docs/hot_paths.md").read_text()
for base in (root / "benches/baselines").glob("*.md"):
    if base.name not in hot and "baselines/" + base.name not in hot:
        # Allow baselines not linked if mentioned via relative link path
        if f"baselines/{base.name}" not in hot and base.name.replace(".md", "") not in hot:
            pass  # soft: index is curated; require at least the phase12-listed set below

required_baselines = [
    "gather.md",
    "graph_traversal.md",
    "dseparation.md",
    "adjustment.md",
    "partial_correlation.md",
    "pcmci.md",
    "phase5_ci_orientation.md",
    "propensity.md",
    "matching.md",
    "phase8_pag.md",
    "phase9_regime_mediation.md",
    "phase10_shapley.md",
    "phase11_design_state.md",
]
for name in required_baselines:
    if not (root / "benches/baselines" / name).exists():
        missing.append(f"missing baseline {name}")
    if name not in hot:
        missing.append(f"docs/hot_paths.md does not reference {name}")

if missing:
    print("Phase 12 gate FAILED:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

print("Phase 12 inventory / artifact evidence map OK")
PY

echo "== regenerate conformance docs (must be clean) =="
python3 scripts/generate_conformance_docs.py
if ! git diff --exit-code -- docs/conformance >/dev/null; then
  echo "docs/conformance is stale; commit regenerated output"
  git diff --stat -- docs/conformance
  exit 1
fi

echo "== cargo test phase12 surfaces =="
cargo test -p causal-io --lib
cargo test -p causal --test phase12_graph_interchange
cargo test -p causal --test phase12_artifact_migrate

echo "== criterion smoke (designated hot paths) =="
cargo bench -p causal-kernels --bench gather -- --test
cargo bench -p causal-graph --bench traversal -- --test
cargo bench -p causal-graph --bench dseparation -- --test
cargo bench -p causal-identify --bench adjustment -- --test
cargo bench -p causal-discovery --bench pcmci -- --test
cargo bench -p causal-design --bench design_rank -- --test
cargo bench -p causal-state --bench state_append -- --test

if command -v cargo-deny >/dev/null 2>&1; then
  echo "== cargo deny check =="
  cargo deny check
else
  echo "WARN: cargo-deny not installed; CI installs it. Skipping local deny check."
fi

echo "Phase 12 gate PASSED"
