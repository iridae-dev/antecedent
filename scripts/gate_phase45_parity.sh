#!/usr/bin/env bash
# Phase 4/5 parity gate: inventory honesty + conformance + calibration.
# Fails if a phase=4|5 status=done capability lacks known evidence, or if
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
            "phase": int(g("phase", "0") or 0),
        })
    return out

# Evidence map for phase 4/5 done (and intentional_deviation) rows.
EVIDENCE = {
    # DoWhy phase 4
    "dowhy.identify.efficient_backdoor": "conformance/phase4/efficient_backdoor",
    "dowhy.estimate.glm": "conformance/phase4/glm_adjustment",
    "dowhy.estimate.propensity": "conformance/phase4/propensity_ipw",
    "dowhy.estimate.matching": "conformance/phase4/distance_matching",
    "dowhy.estimate.doubly_robust": "conformance/phase4/aipw",
    "dowhy.estimate.iv": "conformance/phase4/iv_2sls",
    "dowhy.estimate.rd": "conformance/phase4/rd_sharp",
    "dowhy.estimate.two_stage": "conformance/phase4/frontdoor",
    "dowhy.refute.unobserved_common_cause": "conformance/phase4/refuters",
    "dowhy.refute.overlap": "conformance/phase4/refuters",
    "dowhy.refute.data_subset": "conformance/phase4/refuters",
    "dowhy.refute.dummy_outcome": "conformance/phase4/refuters",
    "dowhy.refute.evalue": "conformance/phase4/refuters",
    "dowhy.refute.graph": "conformance/phase4/refuters",
    "dowhy.refute.sensitivity": "conformance/phase4/refuters",
    # Tigramite phase 5
    "tigramite.data.transforms": "crates/causal-data/src/transforms.rs",
    "tigramite.ci.multivariate_partial_corr": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.weighted_partial_corr": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.robust_partial_corr": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.regression": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.cmi_knn": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.mixed_cmi_knn": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.symbolic_cmi": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.gpdc": "parity/phase5_deviations.md",
    "tigramite.ci.gsquared": "crates/causal-stats/src/ci/calibration.rs",
    "tigramite.ci.oracle": "crates/causal-discovery/src/engine_tests.rs",
    "tigramite.discovery.pcmci_plus": "conformance/tigramite/pcmci_plus_lag0",
    "tigramite.graphs.endpoints": "crates/causal-graph/src/cpdag.rs",
}

missing = []
for c in caps(dowhy) + caps(tig):
    if c["phase"] not in (4, 5):
        continue
    if c["status"] not in ("done", "intentional_deviation"):
        continue
    ev = EVIDENCE.get(c["id"])
    if not ev:
        missing.append(f"{c['id']} (status={c['status']}) has no evidence mapping")
        continue
    p = root / ev
    if not p.exists():
        missing.append(f"{c['id']} evidence path missing: {ev}")

if missing:
    print("parity inventory gaps:")
    for m in missing:
        print(" -", m)
    sys.exit(1)
print(f"parity inventory evidence map: ok ({len(EVIDENCE)} phase 4/5 rows)")
PY

echo "== conformance / calibration =="
cargo test -p causal-analysis --test phase4_conformance --test dowhy_linear_gaussian_ate
cargo test -p causal-validate --test phase4_refuters
cargo test -p causal-discovery --test tigramite_pcmci_lag1 --test tigramite_pcmci_plus_lag0
cargo test -p causal-stats --lib ci::calibration
bash scripts/gate_phase4_reuse.sh
echo "phase4/5 parity gate: ok"
