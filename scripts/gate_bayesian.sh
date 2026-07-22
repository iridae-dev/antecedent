#!/usr/bin/env bash
# Bayesian gate: inventory honesty + fixtures + benches.
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
        })
    return out

EVIDENCE = {
    "bayes.prob.columnar_posteriors": "crates/causal-prob/src/posterior.rs",
    "bayes.prob.priors": "crates/causal-prob/src/prior.rs",
    "bayes.backend.conjugate_gaussian": "crates/causal/tests/bayesian.rs",
    "bayes.backend.laplace_glm": "crates/causal/tests/bayesian.rs",
    "bayes.estimate.gcomp": "crates/causal/tests/bayesian.rs",
    "bayes.estimate.temporal_gcomp": "crates/causal/tests/bayesian.rs",
    "bayes.estimate.graph_envelopes": "crates/causal/tests/bayesian.rs",
    "bayes.validate.ppc": "crates/causal/tests/bayesian.rs",
    "bayes.validate.prior_sensitivity": "crates/causal/tests/bayesian.rs",
    "bayes.data.bayesian_bootstrap": "provenance/data.bayesian_bootstrap.toml",
    "bayes.io.posterior_artifact": "crates/causal-io/src/posterior.rs",
    "bayes.facade.inference_mode": "crates/causal/src/inference.rs",
    "bayes.model.pcm_scm_registry": "crates/causal-model/src/lib.rs",
    "bayes.discovery.dag_posterior": "crates/causal-discovery/src/exact_enumeration.rs",
    "bayes.backend.hierarchical_bvar_gp": "crates/causal-model/src/registry.rs",
    "bayes.validate.mcmc_diagnostics": "crates/causal-validate/src/bayesian_checks.rs",
    "bayes.ci.tests": "crates/causal-stats/src/ci/bayes.rs",
    "bayes.prior_bank.catalog": "crates/causal-io/src/prior_bank.rs",
    "bayes.prior_bank.effect_map": "crates/causal-estimate/src/bayesian.rs",
    "bayes.prior_bank.power_mixture": "crates/causal-prob/src/external_prior.rs",
    "bayes.prior_bank.conflict": "crates/causal-validate/src/conflict.rs",
    "bayes.prior_bank.transport": "crates/causal-prob/src/transport.rs",
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
    p = root / ev
    if not p.exists():
        missing.append(f"{c['id']} evidence missing: {ev}")

# Exit-criterion fixtures
for path in [
    "conformance/bayesian/shared_functional_ate/expected.json",
    "conformance/bayesian/nonidentified_prior/expected.json",
    "conformance/bayesian/laplace_glm/expected.json",
    "conformance/bayesian/dag_posterior/expected.json",
    "conformance/bayesian/temporal_pulse/expected.json",
    "conformance/bayesian/prior_bank_catalog/expected.json",
    "conformance/bayesian/prior_bank_effect_map/expected.json",
    "conformance/bayesian/prior_bank_power_mixture/expected.json",
    "conformance/bayesian/prior_bank_conflict_shrink/expected.json",
    "conformance/bayesian/prior_bank_transport/expected.json",
    "conformance/bayesian/prior_bank_alpha_sensitivity/expected.json",
    "crates/causal-prob/benches/laplace_glm.rs",
    "crates/causal-estimate/benches/posterior_functional.rs",
]:
    if not (root / path).exists():
        missing.append(f"required exit artifact missing: {path}")

if missing:
    print("Bayesian gate FAILED:")
    for m in missing:
        print(" -", m)
    sys.exit(1)

print("Bayesian inventory evidence map OK")
PY

echo "== cargo test causal-prob / estimate bayesian / io posterior / bayesian conformance =="
cargo test -p causal-prob --lib
cargo test -p causal-discovery --lib graph_posterior::
cargo test -p causal-discovery --lib exact_enumeration::
cargo test -p causal-discovery --lib structure_mcmc::
cargo test -p causal-discovery --lib order_mcmc::
cargo test -p causal-discovery --lib ci_screened_posterior::
cargo test -p causal-discovery --lib dbn_posterior::
cargo test -p causal-estimate --lib bayesian
cargo test -p causal-estimate --lib envelope
cargo test -p causal-validate --lib bayesian_checks
cargo test -p causal-io --lib posterior
cargo test -p causal-io --lib prior_bank
cargo test -p causal-data --lib resample
cargo test -p causal --test bayesian
cargo test -p causal --test manufacturing_temporal

echo "== criterion smoke (reuse gates) =="
cargo bench -p causal-prob --bench laplace_glm -- --test
cargo bench -p causal-estimate --bench posterior_functional -- --test

echo "== Python panel Bayesian facade smoke =="
(
  cd python
  unset CONDA_PREFIX || true
  uv run pytest tests/test_panel_bayesian.py tests/test_temporal_bayesian_pulse.py tests/test_prior_bank.py -q
)

echo "Bayesian gate PASSED"
