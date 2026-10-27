"""Recovery evidence keeps actual fitting origins and explicit joint draw pairing."""

import json
import os
import subprocess
import sys

import antecedent as ant
import numpy as np
import pytest
from antecedent.cyclic import (
    AsymptoticNormal,
    EvidenceSchema,
    EvidenceVariable,
    FlatCoefficients,
    KnownDisturbance,
    RecoveryPosterior,
    RecoveryPrior,
    Set,
    World,
)
from antecedent.outcomes import require_success

from test_cyclic import _recovery


def _source(seed=19):
    posterior = RecoveryPosterior(
        RecoveryPrior(FlatCoefficients(), KnownDisturbance({"A": 1.0, "B": 1.0})),
        draws=8,
    )
    return require_success(
        _recovery(posterior, execution=ant.Execution(seed=seed, threads=1)).execute()
    )


def _branch(source, level):
    return require_success(
        source.law.prepare(
            [World(47, Set("A", level))],
            variables=["A", "B"],
            outcome="B",
        ).execute()
    )


def _evidence(result, identifier, *, population="original population", units="meter"):
    schema = EvidenceSchema(
        identifier,
        population,
        (
            EvidenceVariable("A", units),
            EvidenceVariable("B", units),
        ),
    )
    world = require_success(result.interventions())[47]
    return world.evidence(schema, variable=1, port="response")


def test_sibling_targets_preserve_native_origins_and_explicit_outcome_composition():
    source = _source()
    left = require_success(_evidence(_branch(source, 3.0), "do-A3"))
    right = _evidence(_branch(source, 4.0), "do-A4")
    other = require_success(right)
    origins = left.provenance["recovery_origins"]
    assert origins == other.provenance["recovery_origins"]
    assert left.provenance["source_identity"] != other.provenance["source_identity"]
    assert left.identity != other.identity
    assert origins["evidence"] != left.provenance["source_identity"]
    independent = left.joint_law(right, dependence="independent")
    assert independent.refusal.code == "evidence_double_counted"
    assert left.joint_law(right).refusal.code == "joint_law_required"
    joint = require_success(left.joint_law(right, dependence="aligned"))
    draws = np.asarray(joint)
    expected = np.column_stack(
        (
            left.to_dict()["columns"][0]["samples"],
            other.to_dict()["columns"][0]["samples"],
        )
    )
    np.testing.assert_array_equal(draws, expected)
    assert joint.semantic == "causal_functional_posterior"
    assert joint.shape == (8, 2)
    assert joint.weights is not None
    weights = np.asarray(joint.weights)
    means = np.sum(weights[:, None] * expected, axis=0)
    covariance = np.sum(weights * (expected[:, 0] - means[0]) * (expected[:, 1] - means[1]))
    assert joint.covariance(0, 1) == pytest.approx(covariance, abs=1e-12)


def test_equal_draw_counts_do_not_prove_alignment_or_independence():
    left = require_success(_evidence(_branch(_source(19), 3.0), "left"))
    right = require_success(_evidence(_branch(_source(20), 4.0), "right"))
    a, b = left.provenance["recovery_origins"], right.provenance["recovery_origins"]
    assert a["evidence"] == b["evidence"]
    assert a["alignment"] != b["alignment"]
    assert left.joint_law(right, dependence="aligned").refusal.code == "joint_law_required"
    assert left.joint_law(right, dependence="independent").refusal.code == "evidence_double_counted"


def test_population_units_and_sampling_ports_require_their_scientific_bridges():
    source = _source()
    first, second = _branch(source, 3.0), _branch(source, 4.0)
    left = require_success(_evidence(first, "left"))
    population = _evidence(second, "other-pop", population="other population")
    units = _evidence(second, "other-unit", units="second")
    assert left.joint_law(population, dependence="aligned").refusal.code == "joint_law_required"
    assert left.joint_law(units, dependence="aligned").refusal.code == "joint_law_required"
    sampling = require_success(_recovery(AsymptoticNormal(8)).execute())
    evidence = require_success(_evidence(_branch(sampling, 3.0), "sampling"))
    assert evidence.joint_law().refusal.code == "joint_law_required"


def test_version_three_consumer_rederives_origins_and_preserves_exact_encoding(tmp_path):
    source = _source()
    left = require_success(_evidence(_branch(source, 3.0), "left"))
    right = require_success(_evidence(_branch(source, 4.0), "right"))
    artifact = right.export()
    loaded = require_success(ant.load(artifact, expected=artifact.expectation))
    assert loaded.provenance["recovery_origins"] == right.provenance["recovery_origins"]
    assert bytes(loaded.export()) == bytes(artifact)
    np.testing.assert_array_equal(
        np.asarray(require_success(left.joint_law(loaded, dependence="aligned"))),
        np.asarray(require_success(left.joint_law(right, dependence="aligned"))),
    )
    path = tmp_path / "evidence.ant"
    path.write_bytes(bytes(artifact))
    script = """
import json, pathlib, sys
from antecedent.cyclic import consume_evidence
from antecedent.outcomes import require_success
value = require_success(consume_evidence(pathlib.Path(sys.argv[1]).read_bytes(), expected_identity=sys.argv[2]))
print(json.dumps(dict(value.provenance['recovery_origins'])))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path), right.identity],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == dict(right.provenance["recovery_origins"])
