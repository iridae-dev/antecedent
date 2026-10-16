"""PyO3 signature ``seed`` defaults stay on the published values.

``seed=0`` is the default for law enumeration, cross-world contrasts, and
identification bounds, plus the other entry points pinned in ``SEED_ZERO``
(attribution and mechanism samplers that were published with that default).
``seed=1`` is the default for sampling estimators. Both exist on purpose.
Changing either one changes published numerical results, so a new numeric
default or a silent flip between 0 and 1 fails this test.

``seed=None`` is not a numeric default. Four prepare-then-evaluate stages use
it as an optional override of the seed frozen at prepare, and the count-fitted
observation build uses it as an optional seed that defaults inside the engine. Those names are
pinned in ``SEED_OPTIONAL`` so that optional form cannot grow, shrink, or
turn into a third integer unnoticed.

The scanner reads ``python/src`` as text and does not import the extension.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"

SEED_ZERO = (
    "decision_reliability_api.rs::decision_reliability_assess_producer",
    "decision_reliability_api.rs::decision_reliability_consume_derived",
    "decision_reliability_api.rs::decision_reliability_derive",
    "decision_reliability_api.rs::decision_reliability_derive_producer",
    "decision_reliability_api.rs::decision_reliability_verify_derived",
    "attribution_api.rs::attribute_distribution_change",
    "attribution_api.rs::attribute_distribution_change_robust",
    "attribution_api.rs::attribute_feature_relevance",
    "attribution_api.rs::attribute_path_specific",
    "attribution_api.rs::attribute_structure_change",
    "attribution_api.rs::attribute_unit_change",
    "attribution_api.rs::counterfactual_ite",
    "attribution_api.rs::mechanism_change_detection",
    "attribution_api.rs::sample_do_py",
    "attribution_api.rs::sample_interventional_distribution",
    "counterfactual_id_api.rs::consume_counterfactual_id_artifact_native",
    "counterfactual_id_api.rs::evaluate",
    "counterfactual_id_api.rs::prepare_counterfactual_id_native",
    "cross_world_api.rs::consume_cross_world_edge_contrast_artifact",
    "cross_world_api.rs::evaluate_cross_world_edge_contrast",
    "decision_reliability_api.rs::decision_reliability_assess",
    "decision_reliability_api.rs::decision_reliability_consume",
    "decision_reliability_api.rs::decision_reliability_rank",
    "design_api.rs::rank_designs",
    "discovered_transport_api.rs::verify_discovered_transport_estimate_artifact",
    "gcm_api.rs::attribute_distribution_change",
    "gcm_api.rs::attribute_distribution_change_robust",
    "gcm_api.rs::attribute_feature_relevance",
    "gcm_api.rs::attribute_path_specific",
    "gcm_api.rs::attribute_paths",
    "gcm_api.rs::attribute_paths",
    "gcm_api.rs::attribute_structure_change",
    "gcm_api.rs::attribute_unit_change",
    "gcm_api.rs::counterfactual_ite",
    "gcm_api.rs::mechanism_change_detection",
    "gcm_api.rs::rank_root_causes",
    "gcm_api.rs::rank_root_causes",
    "gcm_api.rs::sample_do",
    "predictive_assessment_api.rs::predictive_assessment_assess",
    "recovery_api.rs::prepare_empirical",
    "recovery_api.rs::prepare_exact",
    "recovery_chain_api.rs::consume_recovery_chain_artifact",
    "recovery_chain_api.rs::recover_recovery_chain",
    "temporal_counterfactual_api.rs::consume_temporal_counterfactual_artifact",
    "temporal_counterfactual_api.rs::evaluate_temporal_counterfactual",
    "transport_admg_conditional_api.rs::prepare_empirical",
    "transport_admg_conditional_api.rs::prepare_exact",
    "transport_mixed_api.rs::prepare_empirical",
    "transport_mixed_api.rs::prepare_exact",
    "transport_mz_api.rs::prepare_empirical",
    "transport_mz_api.rs::prepare_exact",
    "transport_z_api.rs::prepare_empirical",
    "transport_z_api.rs::prepare_exact",
    "transport_z_api.rs::receive",
)

SEED_OPTIONAL = (
    "observation_query_api.rs::count_fitted_observation_build",
    "transport_admg_conditional_api.rs::estimate",
    "transport_mixed_api.rs::estimate",
    "transport_mz_api.rs::estimate",
    "transport_z_api.rs::estimate",
)


def _attributes(text: str):
    index = 0
    length = len(text)
    while True:
        start = text.find("#[", index)
        if start < 0:
            return
        cursor = start + 2
        depth = 1
        quote = None
        while cursor < length and depth:
            char = text[cursor]
            if quote:
                if char == "\\":
                    cursor += 2
                    continue
                if char == quote:
                    quote = None
                cursor += 1
                continue
            if char in ("'", '"'):
                quote = char
                cursor += 1
                continue
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
            cursor += 1
        yield text[start:cursor], cursor
        index = cursor


def _signature_body(attr: str) -> str | None:
    match = re.search(r"signature\s*=\s*\(", attr)
    if match is None:
        return None
    cursor = match.end()
    depth = 1
    while cursor < len(attr) and depth:
        if attr[cursor] == "(":
            depth += 1
        elif attr[cursor] == ")":
            depth -= 1
        cursor += 1
    return attr[match.end() : cursor - 1]


def pyo3_seed_defaults() -> list[tuple[str, str, str]]:
    """``(file, rust fn, default token)`` for every pyo3 signature ``seed=``."""
    found: list[tuple[str, str, str]] = []
    for path in sorted(_SRC.rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(_SRC).as_posix()
        for attr, end in _attributes(text):
            if "pyo3" not in attr or "signature" not in attr:
                continue
            body = _signature_body(attr)
            if body is None:
                continue
            tokens = re.findall(r"\bseed\s*=\s*([^,\s)]+)", body)
            if not tokens:
                continue
            function = re.search(r"\bfn\s+([A-Za-z0-9_]+)", text[end : end + 2000])
            assert function is not None, relative
            for token in tokens:
                found.append((relative, function.group(1), token))
    return found


def test_pyo3_signature_seed_defaults_are_zero_or_one():
    found = pyo3_seed_defaults()
    tokens = {token for _file, _fn, token in found}
    assert tokens <= {"0", "1", "None"}, tokens
    zero = sorted(f"{file}::{fn}" for file, fn, token in found if token == "0")
    optional = sorted(f"{file}::{fn}" for file, fn, token in found if token == "None")
    assert zero == sorted(SEED_ZERO)
    assert optional == sorted(SEED_OPTIONAL)
    assert all(token in {"0", "1"} for _file, _fn, token in found if token != "None")
