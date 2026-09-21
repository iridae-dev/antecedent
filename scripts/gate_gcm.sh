#!/usr/bin/env bash
# GCM gate: inventory honesty + fixtures + benches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash scripts/gate_parity_schema.sh

python3 - <<'PY'
import sys

sys.path.insert(0, "scripts")
import parity_rows as pr

EVIDENCE = {
    "gcm.model.compiled_plans": "crates/antecedent-model/src/compile.rs",
    "gcm.model.mechanisms": "crates/antecedent-model/src/mechanism.rs",
    "gcm.model.registry_fit": "crates/antecedent-model/src/registry.rs",
    "gcm.model.sampling": "crates/antecedent-model/src/sample.rs",
    "gcm.do_sampling": "crates/antecedent/tests/gcm.rs",
    # Additive shift do(X := X + delta): overlay accumulation is the primary
    # implementation. Harness: crates/antecedent-model/src/overlay.rs
    # (`overlay::tests::shift_overlay_accumulates_and_is_independent_of_hard_set`,
    # via `cargo test -p antecedent-model --lib` below) plus
    # crates/antecedent/tests/gcm.rs::gcm_shift_intervention_differs_from_hard_set
    # (via `cargo test -p antecedent --test gcm` below), which shows shift and hard
    # set are observably different on a linear-Gaussian fixture.
    "gcm.do_sampling.shift": "crates/antecedent-model/src/overlay.rs",
    "gcm.model.falsification": "crates/antecedent-model/src/evaluate.rs",
    "gcm.counterfactual.aap": "crates/antecedent/tests/gcm.rs",
    "gcm.attribution.basic": "crates/antecedent/tests/gcm.rs",
}

EXIT_ARTIFACTS = [
    "conformance/gcm/gcm_fit_intervene/expected.json",
    "conformance/gcm/gcm_anomaly/expected.json",
    "conformance/gcm/gcm_cf_ite/expected.json",
    "conformance/gcm/do_sampling_weighting/expected.json",
    "conformance/gcm/do_sampling_kde/expected.json",
    "conformance/gcm/do_sampling_mcmc/expected.json",
    "crates/antecedent-model/benches/sample_overlay.rs",
    "crates/antecedent-counterfactual/benches/counterfactual_batch.rs",
    "parity/gcm.toml",
]

# Only the GCM/CF evidence set is gated here (the attribution inventory is gated separately).
problems = pr.evidence_map_problems(EVIDENCE, ["parity/gcm.toml"])
problems += pr.exit_artifact_problems(EXIT_ARTIFACTS)
problems += pr.require_done("parity/estimate.toml", ("gcm.surface", "gcm.do_sampling"), "GCM")
problems += pr.require_done("parity/bayesian.toml", ("bayes.model.pcm_scm_registry",), "GCM")
pr.finish("GCM", problems, "GCM inventory evidence map OK")
PY

echo "== cargo test antecedent-model / counterfactual / attribution / facade GCM =="
bash scripts/counted_cargo.sh test -p antecedent-model --lib
bash scripts/counted_cargo.sh test -p antecedent-model --test scm_oracle
bash scripts/counted_cargo.sh test -p antecedent-model --features gaussian-process --lib \
  gaussian_process_matches_exact_logdet_oracle
bash scripts/counted_cargo.sh test -p antecedent-counterfactual --lib
bash scripts/counted_cargo.sh test -p antecedent-attribution --lib
bash scripts/counted_cargo.sh test -p antecedent --test gcm
bash scripts/counted_cargo.sh test -p antecedent --lib

echo "== criterion smoke (overlay + CF batch) =="
cargo bench -p antecedent-model --bench sample_overlay -- --test
cargo bench -p antecedent-counterfactual --bench counterfactual_batch -- --test

echo "GCM gate PASSED"
