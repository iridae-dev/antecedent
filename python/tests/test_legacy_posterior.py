"""The Python facade returns the io-core legacy-posterior conversion.

The column-major fixture is the one
``crates/antecedent/src/analysis/legacy_posterior.rs`` locks against
``convert_legacy_posterior``: columns ``[0, 1, 2]`` and ``[0, 2, 4]``, target
order swapped, draw-major rows ``[0, 0]``, ``[2, 1]``, ``[4, 2]``. Python does
not rebuild those rows.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from antecedent import _native
from antecedent.errors import CausalError, CausalTypeError
from antecedent.joint_distribution import DistributionIdentity, ScientificQuantity
from antecedent.legacy_posterior import LegacyPosteriorBinding, convert_legacy_posterior

_DRAWS = [0.0, 1.0, 2.0, 0.0, 2.0, 4.0]
_ROWS = np.array([[0.0, 0.0], [2.0, 1.0], [4.0, 2.0]])
_OTHER_MEANINGS = (
    "parameter_posterior",
    "posterior_predictive",
    "interventional_predictive",
    "estimator_sampling",
    "bootstrap",
    "empirical_outcome",
)


def _outcome(name: str) -> ScientificQuantity:
    return ScientificQuantity.outcome(
        name, units="dimensionless", population="target", regime="do(a=1)", functional="mean"
    )


def _identity(semantic: str, quantities: tuple[ScientificQuantity, ...]) -> DistributionIdentity:
    return DistributionIdentity(
        semantic=semantic,  # type: ignore[arg-type]
        quantities=quantities,
        alignment="joint",
        source_id="py-posterior",
        provider_id="legacy-conversion",
        rng_id="source-posterior-draws",
        snapshot_id="checked-snapshot",
        causal_contract_id="checked-contract",
    )


def _encoded(quantity_names: list[str], draws: list[float], n_draws: int) -> bytes:
    artifact = _native.PosteriorArtifact(
        n_draws=n_draws,
        mean=[1.0, 2.0] if len(quantity_names) == 2 else [1.0],
        sd=[1.0, 2.0] if len(quantity_names) == 2 else [0.0],
        q025=[0.05, 0.1] if len(quantity_names) == 2 else [1.0],
        q975=[1.95, 3.9] if len(quantity_names) == 2 else [1.0],
        draws=draws,
        backend_id="conjugate",
        identification="NonparametricallyIdentified",
        quantity_names=quantity_names,
        hessian_condition=1.0,
    )
    return _native.encode_posterior_artifact(artifact)


def _bindings() -> tuple[LegacyPosteriorBinding, LegacyPosteriorBinding]:
    return (
        LegacyPosteriorBinding({"effect": {"name": "ate_x"}}, _outcome("ate_x")),
        LegacyPosteriorBinding({"effect": {"name": "ate_y"}}, _outcome("ate_y")),
    )


def test_facade_returns_the_converted_artifact_without_rebuilding_the_law() -> None:
    ate_y, ate_x = _outcome("ate_y"), _outcome("ate_x")
    law = convert_legacy_posterior(
        _encoded(["ate_x", "ate_y"], _DRAWS, 3),
        _identity("causal_functional_posterior", (ate_y, ate_x)),
        _bindings(),
    )
    np.testing.assert_array_equal(np.asarray(law), _ROWS)
    assert law.semantic == "causal_functional_posterior"
    assert law.calibration == "not_measured"
    assert law.trust == "unverified"
    receipt = law.to_dict()["legacy_posterior"]
    assert receipt["quantities"] == [
        {"effect": {"name": "ate_x"}},
        {"effect": {"name": "ate_y"}},
    ]
    assert receipt["draws_encoding"] == "f64_le_colmajor"
    assert law.to_dict()["legacy_bindings"][0]["source"] == {"effect": {"name": "ate_x"}}


def test_parameter_predictive_bootstrap_and_structural_meanings_stay_distinct() -> None:
    encoded = _encoded(["ate_x", "ate_y"], _DRAWS, 3)
    quantities = (_outcome("ate_y"), _outcome("ate_x"))
    for semantic in _OTHER_MEANINGS:
        with pytest.raises(CausalError) as caught:
            convert_legacy_posterior(encoded, _identity(semantic, quantities), _bindings())
        assert caught.value.reason_code == "distribution_meaning_mismatch"


def test_an_unnamed_outcome_quantity_refuses() -> None:
    artifact = _native.PosteriorArtifact(
        n_draws=2,
        mean=[1.0, 2.0],
        sd=[0.0, 0.0],
        q025=[1.0, 2.0],
        q975=[1.0, 2.0],
        draws=[1.0, 1.0, 2.0, 2.0],
        backend_id="conjugate",
        identification="NonparametricallyIdentified",
        quantity_names=[" ", "ate_y"],
        hessian_condition=1.0,
    )
    encoded = _native.encode_posterior_artifact(artifact)
    with pytest.raises(CausalError) as caught:
        convert_legacy_posterior(
            encoded,
            _identity("causal_functional_posterior", (_outcome("ate_y"), _outcome("blank"))),
            (
                LegacyPosteriorBinding({"effect": {"name": " "}}, _outcome("blank")),
                LegacyPosteriorBinding({"effect": {"name": "ate_y"}}, _outcome("ate_y")),
            ),
        )
    assert caught.value.reason_code == "distribution_meaning_mismatch"


def test_an_unnamed_coefficient_and_a_summary_only_posterior_refuse() -> None:
    coefficient = _native.PosteriorArtifact(
        n_draws=2,
        mean=[1.0],
        sd=[0.0],
        q025=[1.0],
        q975=[1.0],
        draws=[1.0, 1.0],
        backend_id="conjugate",
        identification="NonparametricallyIdentified",
        quantity_names=["coef_0"],
        hessian_condition=1.0,
    )
    identity = _identity(
        "parameter_posterior",
        (
            ScientificQuantity.covariate(
                "beta",
                units="dimensionless",
                population="target",
                regime="do(a=1)",
                functional="coefficient",
            ),
        ),
    )
    binding = (
        LegacyPosteriorBinding({"coefficient": {"index": 0, "name": None}}, identity.quantities[0]),
    )
    with pytest.raises(CausalError) as unnamed:
        convert_legacy_posterior(
            _native.encode_posterior_artifact(coefficient), identity, binding
        )
    assert unnamed.value.reason_code == "distribution_meaning_mismatch"
    summary = _native.PosteriorArtifact.from_moments(
        n_draws=4,
        mean=[1.0],
        sd=[0.0],
        q025=[1.0],
        q975=[1.0],
        backend_id="conjugate",
        identification="NonparametricallyIdentified",
        quantity_names=["ate"],
        hessian_condition=1.0,
    )
    with pytest.raises(CausalError, match="explicitly bound draws"):
        convert_legacy_posterior(
            _native.encode_posterior_artifact(summary),
            _identity("causal_functional_posterior", (_outcome("ate"),)),
            (LegacyPosteriorBinding({"effect": {"name": "ate"}}, _outcome("ate")),),
        )


def test_a_source_id_that_is_not_the_artifact_refuses() -> None:
    identity = dataclasses.replace(
        _identity("causal_functional_posterior", (_outcome("ate_y"), _outcome("ate_x"))),
        source_id="other-posterior",
    )
    with pytest.raises(CausalError) as caught:
        convert_legacy_posterior(_encoded(["ate_x", "ate_y"], _DRAWS, 3), identity, _bindings())
    assert caught.value.reason_code == "quantity_semantics_mismatch"


def test_bindings_must_be_legacy_posterior_bindings() -> None:
    identity = _identity("causal_functional_posterior", (_outcome("ate"),))
    with pytest.raises(CausalTypeError):
        convert_legacy_posterior(b"unused", identity, ["ate"])  # type: ignore[arg-type]
