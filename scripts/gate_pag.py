#!/usr/bin/env python3

import gate_hot_path_baselines as hp
import parity_rows as pr

EVIDENCE = {
    "pag.graph.admg": "crates/antecedent-graph/src/admg.rs",
    "pag.graph.pag_temporal": "crates/antecedent-graph/src/pag.rs",
    "pag.graph.m_separation": "crates/antecedent-graph/src/msep.rs",
    "pag.graph.latent_projection": "crates/antecedent/tests/pag.rs",
    "pag.graph.completions_streamed": "crates/antecedent-graph/src/completion.rs",
    "pag.graph.cpdag_mec_completions": "crates/antecedent-graph/src/cpdag_completion.rs",
    "pag.identify.generalized_adjustment": "crates/antecedent/tests/pag.rs",
    "pag.identify.full_id_idc": "crates/antecedent-identify/src/id.rs",
    "pag.identify.response_visibility_id": "crates/antecedent-identify/src/mag_id_bruteforce.rs",
    "pag.discovery.lpcmci": "crates/antecedent/tests/pag.rs",
    "pag.discovery.fci_rfci": "crates/antecedent-discovery/src/fci.rs",
    "pag.facade.dag_only_reject": "crates/antecedent/tests/pag.rs",
}

EXIT_ARTIFACTS = [
    "conformance/graph/cpdag_operations/expected.json",
    "conformance/graph/pag_operations/expected.json",
    "conformance/graph/definite_status_separation/expected.json",
    "conformance/graph/latent_projection/expected.json",
    "conformance/graph/pag_mag_completion/expected.json",
    "conformance/identify/efficient_adjustment/expected.json",
    "conformance/identify/id_hedge/expected.json",
    "conformance/identify/idc/expected.json",
    "conformance/identify/generalized_adjustment/expected.json",
    "conformance/identify/mag_visibility_id/expected.json",
    "conformance/identify/path_specific/expected.json",
    "conformance/identify/auto_envelopes/expected.json",
    "conformance/discovery/rfci/expected.json",
    "conformance/pag/lpcmci_chain/expected.json",
    "conformance/pag/latent_projection_msep/expected.json",
    "conformance/pag/envelope_unidentified_mass/expected.json",
    "conformance/pag/dag_only_pag_reject/expected.json",
    "crates/antecedent-graph/benches/mseparation.rs",
    "crates/antecedent-discovery/benches/pag_orientation.rs",
    "benches/baselines/baselines.toml",
    "parity/pag.toml",
]

problems = pr.honesty_problems("parity/pag.toml", EVIDENCE)
problems += pr.exit_artifact_problems(EXIT_ARTIFACTS)
problems += hp.missing_ids(("pag",))
pr.finish("PAG", problems, "PAG inventory evidence map OK")
