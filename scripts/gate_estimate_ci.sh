#!/usr/bin/env bash
# Estimate/CI parity gate: inventory honesty + conformance + calibration.
# black-box / Exact pins diverge.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path(".")
estimate_inv = (root / "parity/estimate.toml").read_text()
discovery_inv = (root / "parity/discovery.toml").read_text()

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
    # Estimate inventory
    "estimate.identify.efficient_backdoor": "conformance/estimate/efficient_backdoor",
    "estimate.glm": "conformance/estimate/glm_adjustment",
    "estimate.propensity": "conformance/estimate/propensity_ipw",
    "estimate.matching": "conformance/estimate/distance_matching",
    "estimate.doubly_robust": "conformance/estimate/aipw",
    "estimate.iv": "conformance/estimate/iv_2sls",
    "estimate.rd": "conformance/estimate/rd_sharp",
    "estimate.two_stage": "conformance/estimate/frontdoor",
    "estimate.refute.unobserved_common_cause": "conformance/estimate/refuters",
    "estimate.refute.overlap": "conformance/estimate/refuters",
    "estimate.refute.data_subset": "conformance/estimate/refuters",
    "estimate.refute.dummy_outcome": "conformance/estimate/refuters",
    "estimate.refute.evalue": "conformance/estimate/refuters",
    "estimate.refute.graph": "conformance/estimate/refuters",
    "estimate.refute.sensitivity": "conformance/estimate/refuters",
    # Discovery / CI inventory
    "discovery.data.transforms": "crates/causal-data/src/transforms.rs",
    "discovery.ci.multivariate_partial_corr": "crates/causal-stats/src/ci/calibration.rs",
    "discovery.ci.weighted_partial_corr": "crates/causal-stats/src/ci/calibration.rs",
    "discovery.ci.robust_partial_corr": "crates/causal-stats/src/ci/calibration.rs",
    "discovery.ci.regression": "crates/causal-stats/src/ci/calibration.rs",
    "discovery.ci.knn_dependence": "crates/causal-stats/src/ci/calibration.rs",
    "discovery.ci.mixed_knn_dependence": "crates/causal-stats/src/ci/calibration.rs",
    "discovery.ci.symbolic_cmi": "crates/causal-stats/src/ci/calibration.rs",
    "discovery.ci.gpdc": "crates/causal-stats/src/ci/advanced.rs",
    "discovery.ci.gsquared": "crates/causal-stats/src/ci/calibration.rs",
    "discovery.ci.oracle": "crates/causal-discovery/src/engine_tests.rs",
    "discovery.pcmci_plus": "conformance/discovery/pcmci_plus_lag0",
    "discovery.graphs.endpoints": "crates/causal-graph/src/cpdag.rs",
    "discovery.data.masks": "conformance/discovery/masked_mci_lag1",
    "discovery.data.vector_variables": "conformance/discovery/vector_vars_pcmci",
}

missing = []
# Only gate the estimate/CI evidence set (not every inventory row).
by_id = {c["id"]: c for c in caps(estimate_inv) + caps(discovery_inv)}
for cid, ev in EVIDENCE.items():
    c = by_id.get(cid)
    if c is None:
        missing.append(f"{cid} missing from parity manifests")
        continue
    if c["status"] != "done":
        missing.append(f"{cid} status={c['status']} (expected done)")
        continue
    p = root / ev
    if not p.exists():
        missing.append(f"{cid} evidence path missing: {ev}")

if missing:
    print("parity inventory gaps:")
    for m in missing:
        print(" -", m)
    sys.exit(1)
print(f"parity inventory evidence map: ok ({len(EVIDENCE)} estimate/CI rows)")
PY

echo "== conformance / calibration =="
cargo test -p antecedent --test estimate_conformance --test estimate_linear_gaussian_ate
cargo test -p antecedent-validate --test refuters
cargo test -p antecedent-discovery --test discovery_pcmci_lag1 --test discovery_pcmci_plus_lag0 --test discovery_masked_mci_lag1 --test discovery_vector_vars_pcmci --test discovery_notears_chain
cargo test -p antecedent-stats --lib ci::calibration
bash scripts/gate_estimate_reuse.sh
echo "estimate_ci parity gate: ok"
