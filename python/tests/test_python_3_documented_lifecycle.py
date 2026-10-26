"""Run published lifecycle examples and check their scientific continuation."""

import re
from pathlib import Path

import antecedent as ant
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def blocks(path):
    return re.findall(r"```python\n(.*?)```", (ROOT / path).read_text(), flags=re.DOTALL)


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
    examples = blocks("docs/migrations/2.4-to-3.0-python.md")
    assert len(examples) == 7
    execute(examples[0], namespace, "migration:preparation")
    assert namespace["result"].answer.value == pytest.approx(2.0)
    assert namespace["updated_result"].answer.value == pytest.approx(3.0)
    assert ant.estimate(namespace["study"]).answer.value == pytest.approx(2.0)
    execute(examples[1], namespace, "migration:persistence")
    assert namespace["loaded"].study is None
    assert namespace["loaded"].acceptance.identity_bound
    execute(examples[2], namespace, "migration:cyclic")
    assert namespace["result"].value[2].mean("B") == pytest.approx(1.0)
    execute(examples[3], namespace, "migration:posterior")
    posterior = namespace["posterior_record"]
    assert posterior.study is None
    assert posterior.acceptance.status == "sealed"
    assert posterior.acceptance.identity_bound
    assert not posterior.acceptance.replayable
    assert posterior.value.n_draws == 32
    assert namespace["ate_record"].mean == pytest.approx(
        namespace["posterior_result"].value.posterior.effect_mean
    )
    execute(examples[4], namespace, "migration:population")
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
    execute(examples[5], namespace, "migration:shared-batch")
    original_batch = namespace["batch_result"]
    changed_batch = namespace["changed_batch"]
    assert original_batch.member(0).answer.value == pytest.approx(2.0)
    assert changed_batch.member(0).answer.value == pytest.approx(3.0)
    assert original_batch.member(1).answer.value == pytest.approx(-1.0)
    assert namespace["selected_member"].answer.value == pytest.approx(2.0)
    assert namespace["loaded_member"].study is None
    assert namespace["loaded_member"].acceptance.identity_bound

    execute(examples[6], namespace, "migration:scientific-profiles")
    profile_study = namespace["profile_study"]
    profile_outcome = namespace["profile_result"]
    assert profile_outcome.accepted
    assert profile_outcome.answer.value == pytest.approx(2.0)
    profile_result = profile_outcome.value
    assert profile_result.performance.latency_mode == "interactive"
    assert profile_result.performance.bootstrap_replicates_requested == 25
    assert profile_study._declarations["inference"] == ant.Frequentist(
        profile="interactive", bootstrap=25
    )
    assert profile_study.inspect().stage == "prepared"
