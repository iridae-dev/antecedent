"""Run published lifecycle examples and check their scientific continuation."""

import re
from pathlib import Path

import antecedent as ant
import numpy as np
import pytest
from antecedent.errors import CausalValueError, EffectNotIdentified

ROOT = Path(__file__).resolve().parents[2]


def blocks(path):
    return re.findall(
        r"```python\n(.*?)```", (ROOT / path).read_text(encoding="utf-8"), flags=re.DOTALL
    )


def execute(code, namespace, name):
    exec(compile(code, name, "exec"), namespace)


def test_readme_retains_the_original_answer_and_loads_without_live_state():
    namespace = {}
    execute(blocks("README.md")[0], namespace, "README.md")
    original, updated = namespace["result"], namespace["updated"]
    assert updated.answer.value - original.answer.value == pytest.approx(1.0)
    assert ant.estimate(namespace["study"]).answer.value == original.answer.value
    assert namespace["loaded"].study is None
    assert namespace["loaded"].acceptance.identity_bound


def test_migration_examples_execute_with_declared_application_inputs():
    # Balance the covariate exactly across treatment groups so the point
    # target is analytic; this makes no interval-coverage assertion.
    z = np.repeat(np.linspace(-1.0, 1.0, 80), 2)
    treatment = np.tile([0.0, 1.0], 80)
    frame = {"treatment": treatment, "outcome": 2 * treatment + z, "z": z}
    namespace = {
        "frame": frame,
        "new_data": {**frame, "outcome": frame["outcome"] + treatment},
        "graph": [("treatment", "outcome"), ("z", "outcome")],
        "query": ant.AverageEffect("treatment", "outcome"),
        "other_query": ant.AverageEffect("z", "outcome"),
    }
    published = blocks("docs/migration.md")
    # The leading blocks run in order against the application inputs above. The trailing blocks
    # are fragments that continue an application's own cancellation token or transport catalog;
    # they must still be valid Python.
    assert len(published) == 10
    examples, fragments = published[:8], published[8:]
    for index, fragment in enumerate(fragments):
        compile(fragment, f"migration:fragment-{index}", "exec")
    execute(examples[0], namespace, "migration:lifecycle")
    assert namespace["result"].answer.value == pytest.approx(2.0)
    assert namespace["updated_result"].answer.value == pytest.approx(3.0)
    assert ant.estimate(namespace["study"]).answer.value == pytest.approx(2.0)
    assert namespace["alternative"].query == namespace["other_query"]

    execute(examples[1], namespace, "migration:scientific-profiles")
    profile_study = namespace["profile_study"]
    profile_outcome = namespace["profile_outcome"]
    assert profile_outcome.accepted
    assert profile_outcome.answer.value == pytest.approx(2.0)
    profile_result = profile_outcome.value
    assert profile_result.performance.latency_mode == "interactive"
    assert profile_result.performance.bootstrap_replicates_requested == 25
    assert profile_study.inspect().stage == "prepared"

    execute(examples[2], namespace, "migration:outcomes")
    refused = namespace["refused"]
    assert refused.refusal.code == "effect_not_identified"
    assert isinstance(namespace["refusal_error"], EffectNotIdentified)
    assert isinstance(namespace["wrong_call"], CausalValueError)

    execute(examples[3], namespace, "migration:persistence")
    assert namespace["loaded"].study is None
    assert namespace["loaded"].acceptance.identity_bound

    execute(examples[4], namespace, "migration:posterior")
    posterior = namespace["posterior_record"]
    assert posterior.study is None
    assert posterior.acceptance.status == "sealed"
    assert posterior.acceptance.identity_bound
    assert not posterior.acceptance.replayable
    assert posterior.value.n_draws == 32
    assert namespace["ate_record"].mean == pytest.approx(
        namespace["posterior_result"].value.posterior.effect_mean
    )

    execute(examples[5], namespace, "migration:population")
    population, targeted = namespace["population_result"], namespace["targeted"]
    assert population.answer.value == pytest.approx(2.0)
    assert targeted.answer.value == pytest.approx(2.0)
    assert population.inspect().target_weights_id is None
    assert targeted.inspect().target_weights_id is not None
    assert namespace["targeted_loaded"].study is None
    assert namespace["targeted_loaded"].acceptance.identity_bound
    assert (
        namespace["targeted_loaded"].inspect().target_weights_id
        == targeted.inspect().target_weights_id
    )

    execute(examples[6], namespace, "migration:shared-batch")
    original_batch = namespace["batch_result"]
    changed_batch = namespace["changed_batch"]
    assert original_batch.member(0).answer.value == pytest.approx(2.0)
    assert changed_batch.member(0).answer.value == pytest.approx(3.0)
    assert original_batch.member(1).answer.value == pytest.approx(-1.0)
    assert namespace["selected_member"].answer.value == pytest.approx(2.0)
    assert namespace["loaded_member"].study is None
    assert namespace["loaded_member"].acceptance.identity_bound

    execute(examples[7], namespace, "migration:cyclic")
    assert namespace["feedback_result"].value[2].mean("B") == pytest.approx(1.0)
