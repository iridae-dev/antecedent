#!/usr/bin/env bash
# GCM gate: inventory honesty + fixtures + benches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path(".")
text = (root / "parity/gcm.toml").read_text()

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
    "gcm.model.compiled_plans": "crates/causal-model/src/compile.rs",
    "gcm.model.mechanisms": "crates/causal-model/src/mechanism.rs",
    "gcm.model.registry_fit": "crates/causal-model/src/registry.rs",
    "gcm.model.sampling": "crates/causal-model/src/sample.rs",
    "gcm.do_sampling": "crates/antecedent/tests/gcm.rs",
    "gcm.model.falsification": "crates/causal-model/src/evaluate.rs",
    "gcm.counterfactual.aap": "crates/antecedent/tests/gcm.rs",
    "gcm.attribution.basic": "crates/antecedent/tests/gcm.rs",
}

missing = []
# Only gate the GCM/CF evidence set (attribution inventory is gated separately).
by_id = {c["id"]: c for c in caps(text)}
for cid, ev in EVIDENCE.items():
    c = by_id.get(cid)
    if c is None:
        missing.append(f"{cid} missing from gcm.toml")
        continue
    if c["status"] != "done":
        missing.append(f"{cid} status={c['status']}")
        continue
    p = root / ev
    if not p.exists():
        missing.append(f"{cid} evidence missing: {ev}")

for path in [
    "conformance/gcm/gcm_fit_intervene/expected.json",
    "conformance/gcm/gcm_anomaly/expected.json",
    "conformance/gcm/gcm_cf_ite/expected.json",
    "conformance/gcm/do_sampling_weighting/expected.json",
    "conformance/gcm/do_sampling_kde/expected.json",
    "conformance/gcm/do_sampling_mcmc/expected.json",
    "crates/causal-model/benches/sample_overlay.rs",
    "crates/causal-counterfactual/benches/counterfactual_batch.rs",
    "parity/gcm.toml",
]:
    if not (root / path).exists():
        missing.append(f"required exit artifact missing: {path}")

# Domain inventory rows
estimate_inv = (root / "parity/estimate.toml").read_text()
for cid in ("gcm.surface", "gcm.do_sampling"):
    block = None
    for b in re.split(r"\n\[\[capabilities\]\]\n", estimate_inv)[1:]:
        if re.search(rf'^id\s*=\s*"{cid}"', b, re.M):
            block = b
            break
    if not block:
        missing.append(f"{cid} missing from estimate.toml")
        continue
    m = re.search(r'^status\s*=\s*"([^"]*)"', block, re.M)
    if not m or m.group(1) != "done":
        missing.append(f"{cid} must be status=done when GCM gate passes")

bayes = (root / "parity/bayesian.toml").read_text()
for b in re.split(r"\n\[\[capabilities\]\]\n", bayes)[1:]:
    if re.search(r'^id\s*=\s*"bayes.model.pcm_scm_registry"', b, re.M):
        m = re.search(r'^status\s*=\s*"([^"]*)"', b, re.M)
        if not m or m.group(1) != "done":
            missing.append("bayes.model.pcm_scm_registry must be status=done")
        break
else:
    missing.append("bayes.model.pcm_scm_registry missing from bayesian.toml")

if missing:
    print("GCM gate FAILED:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

print("GCM inventory evidence map OK")
PY

echo "== cargo test causal-model / counterfactual / attribution / facade GCM =="
cargo test -p antecedent-model --lib
cargo test -p antecedent-counterfactual --lib
cargo test -p antecedent-attribution --lib
cargo test -p antecedent --test gcm
cargo test -p antecedent --lib

echo "== criterion smoke (overlay + CF batch) =="
cargo bench -p antecedent-model --bench sample_overlay -- --test
cargo bench -p antecedent-counterfactual --bench counterfactual_batch -- --test

echo "GCM gate PASSED"
