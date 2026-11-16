"""A posterior-artifact prior on the outcome nuisance of the cross-fitted AIPW score.

The closed-form oracles (the per-fold posterior mode, the hand-computed doubly robust estimate and
consistency under a misspecified prior) are the Rust tests
``crates/antecedent-estimate/tests/dml_outcome_prior_reference.rs``; this file pins the Python
surface: the wire, the fail-fast validation and the native round trip.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalError, CausalValueError
from antecedent.estimators import DML
from antecedent.outcomes import require_success
from antecedent.priors import PriorMapping

from _refusal import assert_registered_refusal

QUERY = ant.AverageEffect("t", "y")
GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]


def _confounded(n: int = 400, seed: int = 3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * z))).astype(np.float64)
    y = 2.0 * t + z + 0.3 * rng.normal(size=n)
    return {"z": z, "t": t, "y": y}


def _artifact() -> bytes:
    source = require_success(
        ant.analyze(
            _confounded(seed=1),
            graph=GRAPH,
            query=QUERY,
            inference=ant.Bayesian(n_draws=4000, backend="conjugate"),
            refute=False,
            execution=ant.Execution(seed=1),
        )
    )
    return bytes(source.export(quantity="posterior"))


def test_the_dataclass_wires_the_prior_and_its_mapping():
    mapping = PriorMapping.named_parameters([("intercept", "intercept"), ("coef_z", "coef_z")])
    cfg = DML(prior_from=b"artifact", mapping=mapping)
    wire = cfg._wire()
    assert wire["prior_artifact"] == b"artifact"
    assert wire["prior_mapping"] == mapping.to_dict()
    assert DML()._wire() == {}


def test_a_prior_the_aipw_score_cannot_carry_fails_fast():
    with pytest.raises(CausalValueError, match="partially_linear|route_not_supported"):
        DML(prior_from=b"artifact", score="partially_linear")
    with pytest.raises(CausalValueError, match="mapping requires prior_from"):
        DML(mapping=PriorMapping.identical())
    with pytest.raises(CausalValueError, match="artifact bytes"):
        DML(prior_from="not bytes")  # type: ignore[arg-type]
    with pytest.raises(CausalValueError, match="empty"):
        DML(prior_from=b"")


def test_a_named_prior_from_a_bayesian_fit_regularizes_the_outcome_nuisance():
    data = _confounded(seed=2)
    mapping = PriorMapping.named_parameters([("intercept", "intercept"), ("coef_z", "coef_z")])
    plain = require_success(
        ant.analyze(data, graph=GRAPH, query=QUERY, estimator=DML(), refute=False)
    )
    regularized = require_success(
        ant.analyze(
            data,
            graph=GRAPH,
            query=QUERY,
            estimator=DML(prior_from=_artifact(), mapping=mapping),
            refute=False,
        )
    )
    # The prior shapes efficiency, not identification: both fits recover the effect.
    assert abs(regularized.effect - 2.0) < 0.2, regularized.effect
    assert abs(regularized.effect - plain.effect) < 0.2
    text = " ".join(str(a) for a in (regularized.assumptions or []))
    assert "dml.outcome_nuisance_prior" in text or "outcome-regression nuisance" in text


def test_a_prior_whose_coordinates_do_not_map_is_refused_with_the_next_attempt():
    # The source carries a treatment coefficient the per-arm outcome regression does not have, so
    # the identical coefficient subspace does not bind.
    with pytest.raises(CausalError) as caught:
        require_success(
            ant.analyze(
                _confounded(seed=2),
                graph=GRAPH,
                query=QUERY,
                estimator=DML(prior_from=_artifact()),
                refute=False,
            )
        )
    message = f"{getattr(caught.value, 'reason_code', '')} {caught.value}"
    assert "quantity_semantics_mismatch" in message
    assert "named_parameters" in message
    assert_registered_refusal(caught.value)
