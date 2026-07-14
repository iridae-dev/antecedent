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
dowhy = (root / "parity/dowhy.toml").read_text()
tig = (root / "parity/tigramite.toml").read_text()

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
    # DoWhy estimate
    "dowhy.identify.efficient_backdoor": "conformance/estimate/efficient_backdoor",
    "dowhy.estimate.glm": "conformance/estimate/glm_adjustment",
    "dowhy.estimate.propensity": "conformance/estimate/propensity_ipw",
    "dowhy.estimate.matching": "conformance/estimate/distance_matching",
    "dowhy.estimate.doubly_robust": "conformance/estimate/aipw",
    "dowhy.estimate.iv": "conformance/estimate/iv_2sls",
    "dowhy.estimate.rd": "conformance/estimate/rd_sharp",
    "dowhy.estimate.two_stage": "conformance/estimate/frontdoor",
    "dowhy.refute.unobserved_common_cause": "conformance/estimate/refuters",
    "dowhy.refute.overlap": "conformance/estimate/refuters",
    "dowhy.refute.data_subset": "conformance/estimate/refuters",
    "dowhy.refute.dummy_outcome": "conformance/estimate/refuters",
    "dowhy.refute.evalue": "conformance/estimate/refuters",
    "dowhy.refute.graph": "conformance/estimate/refuters",
    "dowhy.refute.sensitivity": "conformance/estimate/refuters",
    # Tigramite CI / discovery
    "tigramite.data.transforms": "crates/causal-data/src/transforms.rs",
    "tigramite.ci.multivariate_partial_corr": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.weighted_partial_corr": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.robust_partial_corr": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.regression": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.cmi_knn": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.mixed_cmi_knn": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.symbolic_cmi": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.gpdc": "parity/ci_deviations.md",
    "tigramite.ci.gsquared": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.oracle": "crates/causal-discovery/src/engine_tests.rs",
    "tigramite.discovery.pcmci_plus": "conformance/tigramite/pcmci_plus_lag0",
    "tigramite.graphs.endpoints": "crates/causal-graph/src/cpdag.rs",
}

missing = []
# Only gate the estimate/CI evidence set (not every DoWhy/Tigramite row).
by_id = {c["id"]: c for c in caps(dowhy) + caps(tig)}
for cid, ev in EVIDENCE.items():
    c = by_id.get(cid)
    if c is None:
        missing.append(f"{cid} missing from parity manifests")
        continue
    if c["status"] not in ("done", "intentional_deviation"):
        missing.append(f"{cid} status={c['status']} (expected done or intentional_deviation)")
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
cargo test -p causal --test estimate_conformance --test dowhy_linear_gaussian_ate
cargo test -p causal-validate --test refuters
cargo test -p causal-discovery --test tigramite_pcmci_lag1 --test tigramite_pcmci_plus_lag0
cargo test -p causal-stats --lib ci::calibration
bash scripts/gate_estimate_reuse.sh
echo "estimate_ci parity gate: ok"
