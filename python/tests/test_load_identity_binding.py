"""``acceptance.identity_bound`` is true only when a caller-supplied identity was compared."""

import antecedent as ant
import pytest
from antecedent.outcomes import require_success
from test_sequential_evaluation import sealed


def exported():
    _problem, artifact = sealed()
    return bytes(artifact.export("sequential-evaluation-1")), artifact.identities


def test_an_empty_expectation_binds_nothing():
    data, _identities = exported()
    with pytest.raises(ant.CausalError, match="identity field"):
        require_success(ant.load(data, expected={}))


def test_a_mapping_without_an_identity_field_binds_nothing():
    data, _identities = exported()
    with pytest.raises(ant.CausalError, match="identity field"):
        require_success(ant.load(data, expected={"execution": None}))
    with pytest.raises(ant.CausalError, match="identity field"):
        require_success(ant.load(data, expected={"model": None}))


def test_no_expectation_loads_unbound():
    data, _identities = exported()
    loaded = require_success(ant.load(data))
    assert not loaded.acceptance.identity_bound


def test_a_correct_identity_is_bound():
    data, identities = exported()
    loaded = require_success(ant.load(data, expected={"model": identities["model"]}))
    assert loaded.acceptance.identity_bound
    consumed = {name: identities[name] for name in ("model", "policy", "evaluation")}
    loaded = require_success(ant.load(data, expected=consumed))
    assert loaded.acceptance.identity_bound


def test_an_identity_the_consumer_does_not_take_is_refused():
    data, identities = exported()
    assert "behaviour" in identities
    with pytest.raises(ant.CausalError, match="unknown expectation fields"):
        require_success(ant.load(data, expected=dict(identities)))


def test_a_wrong_identity_is_refused():
    data, _identities = exported()
    with pytest.raises(ant.CausalError):
        require_success(ant.load(data, expected={"model": "0" * 64}))
