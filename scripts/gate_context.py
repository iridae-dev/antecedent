#!/usr/bin/env python3

import gate_hot_path_baselines as hp
import parity_rows as pr

EVIDENCE = {
    "validate.functional": "crates/antecedent-validate/src/functional.rs",
    "validate.temporal_mediation": "crates/antecedent/tests/prepared_analysis.rs",

    "context.panel_data": "crates/antecedent-data/src/panel.rs",
    "context.context_graph": "crates/antecedent-graph/src/cpdag.rs",
    "context.jpcmci_plus": "crates/antecedent/tests/context_effects.rs",
    "context.rpcmci": "crates/antecedent/tests/context_effects.rs",
    "context.mediation": "crates/antecedent/tests/context_effects.rs",
    "context.mediation.nonparametric": "crates/antecedent-identify/src/path_specific.rs",
    "context.conditional": "crates/antecedent/tests/context_effects.rs",
    "context.prediction": "crates/antecedent/tests/context_effects.rs",
    "context.query_model_planned_variants": "crates/antecedent-io-core/src/query_wire.rs",
}

EXIT_ARTIFACTS = [
    "conformance/discovery/jpcmci_plus_two_env/expected.json",
    "conformance/discovery/jpcmci_plus_two_env_space_dummy_mv/expected.json",
    "conformance/discovery/rpcmci_two_regime/expected.json",
    "conformance/context/temporal_mediation/expected.json",
    "conformance/context/conditional_effect/expected.json",
    "conformance/context/prediction_smoke/expected.json",
    "crates/antecedent-discovery/benches/rpcmci.rs",
    "crates/antecedent-estimate/benches/temporal_mediation.rs",
    "benches/baselines/baselines.toml",
    "parity/context.toml",
]

problems = pr.honesty_problems("parity/context.toml", EVIDENCE)
problems += pr.exit_artifact_problems(EXIT_ARTIFACTS)
problems += hp.missing_ids(("regime_mediation",))
problems += pr.require_done(
    "parity/discovery.toml",
    ("discovery.jpcmci_plus", "discovery.rpcmci", "discovery.effects"),
    "Context",
)
problems += pr.require_done("parity/estimate.toml", ("estimate.conditional",), "Context")
pr.finish("Context", problems, "Context inventory evidence map OK")
