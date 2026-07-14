#!/usr/bin/env bash
# Phase 6 Bayesian gate: inventory honesty + fixtures + benches.
# Fails if a phase=6 status=done capability lacks known evidence.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
from pathlib import Path
import re
import sys

root = Path(".")
text = (root / "parity/bayesian.toml").read_text()

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
    "bayes.prob.columnar_posteriors": "crates/causal-prob/src/posterior.rs",
    "bayes.prob.priors": "crates/causal-prob/src/prior.rs",
    "bayes.backend.conjugate_gaussian": "crates/causal/tests/phase6_bayesian.rs",
    "bayes.backend.laplace_glm": "crates/causal/tests/phase6_bayesian.rs",
    "bayes.estimate.gcomp": "crates/causal/tests/phase6_bayesian.rs",
    "bayes.estimate.graph_envelopes": "crates/causal/tests/phase6_bayesian.rs",
    "bayes.validate.ppc": "crates/causal/tests/phase6_bayesian.rs",
    "bayes.validate.prior_sensitivity": "crates/causal/tests/phase6_bayesian.rs",
    "bayes.data.bayesian_bootstrap": "provenance/data.bayesian_bootstrap.toml",
    "bayes.io.posterior_artifact": "crates/causal-io/src/posterior.rs",
    "bayes.facade.inference_mode": "crates/causal/src/inference.rs",
    # intentional_deviation rows need the deviations doc as evidence
    "bayes.model.pcm_scm_registry": "parity/phase6_deviations.md",
    "bayes.discovery.dag_posterior": "parity/phase6_deviations.md",
    "bayes.backend.stan_pymc": "parity/phase6_deviations.md",
    "bayes.backend.hierarchical_bvar_gp": "parity/phase6_deviations.md",
    "bayes.validate.mcmc_diagnostics": "parity/phase6_deviations.md",
}

missing = []
for c in caps(text):
    if c["phase"] != 6:
        continue
    if c["status"] not in ("done", "intentional_deviation"):
        continue
    ev = EVIDENCE.get(c["id"])
    if not ev:
        missing.append(f"{c['id']} (status={c['status']}) has no evidence mapping")
        continue
    p = root / ev
    if not p.exists():
        missing.append(f"{c['id']} evidence missing: {ev}")

# Exit-criterion fixtures
for path in [
    "conformance/phase6/shared_functional_ate/expected.json",
    "conformance/phase6/nonidentified_prior/expected.json",
    "conformance/phase6/laplace_glm/expected.json",
    "crates/causal-prob/benches/laplace_glm.rs",
    "crates/causal-estimate/benches/posterior_functional.rs",
    "parity/phase6_deviations.md",
]:
    if not (root / path).exists():
        missing.append(f"required exit artifact missing: {path}")

if missing:
    print("Phase 6 gate FAILED:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

print("Phase 6 inventory evidence map OK")
PY

echo "== cargo test causal-prob / estimate bayesian / io posterior / phase6 conformance =="
cargo test -p causal-prob --lib
cargo test -p causal-estimate --lib bayesian
cargo test -p causal-estimate --lib envelope
cargo test -p causal-validate --lib bayesian_checks
cargo test -p causal-io --lib posterior
cargo test -p causal-data --lib resample
cargo test -p causal --test phase6_bayesian

echo "== criterion smoke (reuse gates) =="
cargo bench -p causal-prob --bench laplace_glm -- --test
cargo bench -p causal-estimate --bench posterior_functional -- --test

echo "Phase 6 gate PASSED"
