"""One retained native nuisance owner, explicit export, and independent canonical loading."""

import gc
import json
import os
import subprocess
import sys

import antecedent as ant
import pytest
from antecedent import _native
from antecedent import nuisance_scores as ns
from antecedent.errors import (
    CausalCancelledError,
    CausalResourceError,
    CausalSerializationError,
    CausalUnsupportedError,
)
from antecedent.execution import Execution, ResourceLimits
from antecedent.execution_attempt import observe_native_attempts
from antecedent.outcomes import require_success

from test_prepared_nuisance_score_lifecycle import _data, _mode, _sigmoid, _study, _temporal_study


def test_native_owner_no_eager_encoding_or_refit_on_views_and_export():
    attempted = observe_native_attempts(
        lambda: ns.treatment_score_posterior(_study(), draws=32, summary_rows=[0, 150])
    )
    product = require_success(attempted.unwrap())
    assert attempted.report.operations["posterior_fit"].completed == 1
    assert attempted.report.operations["posterior_draw"].completed == 32
    assert not product._native.encoding_retained()
    read = observe_native_attempts(
        lambda: (
            product.inspect(),
            repr(product),
            product._repr_html_(),
            dict(product.values.items()),
        )
    )
    read.unwrap()
    assert read.report.operations["posterior_fit"].attempted == 0
    assert not product._native.encoding_retained()
    exported = observe_native_attempts(product.export)
    artifact = exported.unwrap()
    assert exported.report.operations["posterior_fit"].attempted == 0
    assert product._native.encoding_retained()
    assert artifact.kind == "array_fit_replay_v1"
    assert artifact.expectation.identity == product.request_digest
    assert bytes(product.export()) == bytes(artifact)
    inspected = ant.artifacts.inspect(artifact)
    assert inspected.kind == "array_fit_replay_v1"
    assert "body" not in inspected.metadata
    assert inspected.metadata["validation"] == "requires_independent_refit"


@pytest.mark.parametrize("temporal", [False, True])
def test_canonical_native_replay_once_has_no_live_study_and_retains_views(temporal):
    if temporal:
        study, _ = _temporal_study()
        product = require_success(ns.temporal_treatment_score(study, [[0], [1.5]], draws=32))
    else:
        product = require_success(
            ns.treatment_score_posterior(_study(), draws=32, summary_rows=[0, 150])
        )
    artifact = product.export()
    loaded_attempt = observe_native_attempts(
        lambda source=artifact: ant.load(source, expected=source.expectation)
    )
    loaded = require_success(loaded_attempt.unwrap())
    assert isinstance(loaded, ns.NuisanceScoreProduct)
    assert loaded_attempt.report.operations["posterior_fit"].completed == 1
    assert loaded_attempt.report.operations["posterior_draw"].completed == 32
    assert loaded.to_dict() == product.to_dict()
    assert loaded.acceptance.verified and loaded.acceptance.identity_bound
    assert loaded.acceptance.replayable and not loaded.acceptance.resumable
    assert loaded.inspect().stage == "consumed"
    assert bytes(loaded.export()) == bytes(artifact)
    assert not hasattr(loaded, "study") and not hasattr(loaded, "_prepared")
    view = loaded.values
    first = next(iter(view))
    expected = view[first]
    del loaded, product, artifact
    gc.collect()
    assert view[first] == expected


def test_known_nuisance_mode_and_input_mutation_isolation_through_canonical_replay():
    import numpy as np

    data = _data()
    prepared = _study(data)
    original = {key: value.copy() for key, value in data.items()}
    data["z"][:] = 17
    data["t"][:] = 0
    product = require_success(
        ns.treatment_score_posterior(prepared, draws=16, summary_rows=[0, 150])
    )
    beta = _mode(np.column_stack([np.ones(200), original["z"]]), original["t"], np.eye(2) / 4)
    assert [product["mode.0"], product["mode.1"]] == pytest.approx(beta, abs=1e-5)
    assert product["row0.at_mode"] == pytest.approx(_sigmoid(beta[0]), abs=1e-5)
    artifact = product.export()
    loaded = require_success(ant.load(artifact, expected=artifact.expectation))
    assert loaded.to_dict() == product.to_dict()
    unbound = require_success(ant.load(bytes(artifact)))
    assert unbound.acceptance.verified and not unbound.acceptance.identity_bound


def test_expected_digest_is_checked_before_refitting_and_corruption_is_operational():
    product = require_success(ns.treatment_score_posterior(_study(), draws=16))
    artifact = product.export()
    with pytest.raises(CausalSerializationError):
        ant.load(bytes(artifact), expected="0" * 64)
    with pytest.raises(CausalSerializationError):
        ant.load(bytes(artifact)[:-5])
    attempt = observe_native_attempts(
        lambda: _native.consume_array_fit_artifact(bytes(artifact), max_numbers=0)
    )
    with pytest.raises(CausalResourceError):
        attempt.unwrap()
    assert attempt.report.operations["posterior_fit"].attempted == 0


def test_native_export_and_replay_obey_resource_cancel_and_original_observer_errors():
    product = require_success(ns.treatment_score_posterior(_study(), draws=16))
    zero = Execution(limits=ResourceLimits(0))
    with pytest.raises(CausalResourceError):
        product.export(execution=zero)
    assert not product._native.encoding_retained()
    artifact = product.export()
    with pytest.raises(CausalResourceError):
        ant.load(artifact, execution=zero)
    token = ant.CancellationToken()
    token.cancel()
    canceled = Execution(cancel=token)
    with pytest.raises(CausalCancelledError):
        product.export(execution=canceled)
    with pytest.raises(CausalCancelledError):
        ant.load(artifact, execution=canceled)
    original = CausalUnsupportedError("observer sentinel", reason_code="effect_not_identified")

    def observer(_fraction, _stage):
        raise original

    with pytest.raises(CausalUnsupportedError) as raised:
        ant.load(artifact, execution=Execution(on_progress=observer))
    assert raised.value is original
    assert any(frame.name == "observer" for frame in raised.traceback)


def test_fresh_process_canonical_load_replays_scientific_recipe(tmp_path):
    product = require_success(
        ns.treatment_score_posterior(_study(), draws=16, summary_rows=[0, 150])
    )
    artifact = product.export()
    path = tmp_path / "nuisance.ant"
    path.write_bytes(bytes(artifact))
    script = """
import json, pathlib, sys
import antecedent as ant
from antecedent.outcomes import require_success
p = require_success(ant.load(pathlib.Path(sys.argv[1]).read_bytes(), expected=sys.argv[2]))
print(json.dumps({'values':dict(p.values.items()),'verified':p.acceptance.verified,'bound':p.acceptance.identity_bound,'stage':p.inspect().stage,'digest':p.request_digest}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path), product.request_digest],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["values"] == dict(product.values.items())
    assert result["verified"] and result["bound"]
    assert result["stage"] == "consumed" and result["digest"] == product.request_digest


def test_nuisance_owner_does_not_replace_or_mutate_the_known_causal_effect():
    data = _data()
    data["y"] = 2.0 * data["t"] + 0.7 * data["z"]
    study = ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        estimator="linear.adjustment.ate",
        inference=ant.Frequentist(bootstrap=0),
        execution=Execution(seed=13, threads=1),
        refute=False,
    )
    original = require_success(ant.estimate(study))
    assert original.estimate.ate == pytest.approx(2.0, abs=1e-8)
    nuisance = require_success(ns.treatment_score_posterior(study, draws=16, summary_rows=[0]))
    assert nuisance.answer.value is None and nuisance.answer.bounds is None
    assert nuisance.target == "propensity_score_nuisance"
    assert not nuisance.inspect().identification.available
    repeated = require_success(ant.estimate(study))
    assert repeated.estimate.ate == pytest.approx(2.0, abs=1e-8)
    assert repeated.estimate.ate == original.estimate.ate
