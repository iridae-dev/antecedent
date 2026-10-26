"""Notebook exploration to independently consumed scientific bytes.

Run with an installed Antecedent development wheel. No files are retained after
this example; a real producer stores container bytes and expectations separately.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import antecedent as ant
import numpy as np


def main() -> None:
    rng = np.random.default_rng(24)
    z = rng.normal(size=240)
    treatment = rng.binomial(1, 0.5, len(z)).astype(float)
    data = {
        "z": z,
        "treatment": treatment,
        "outcome": 2 * treatment + z + rng.normal(scale=0.2, size=len(z)),
    }
    study = ant.prepare(
        data,
        graph=[("z", "treatment"), ("z", "outcome"), ("treatment", "outcome")],
        query=ant.AverageEffect("treatment", "outcome"),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    )
    assert study.last_result is None
    assert study.identification.verdict == "identified"
    print(study.inspect_plan())
    result = study.estimate()
    checked = result.refute(suite="cheap", seed=24)
    assert checked.answer.value == result.answer.value
    print(checked.inspect())

    # Update observations while retaining the scientific question and design.
    updated = study.refresh({**data, "outcome": data["outcome"] + treatment})
    assert np.isclose(updated.answer.value, result.answer.value + 1)
    assert result.refute(suite="cheap", seed=24).answer.value == result.answer.value
    # Named experimental assignments and unit IDs are frozen design bindings.
    from antecedent.interference import BernoulliAssignment

    ids = np.array([f"unit-{i}" for i in range(len(z))])
    experiment_data = {
        "outcome": data["outcome"],
        "arm": treatment.astype(bool),
        "id": ids,
    }
    design_query = ant.experiment.RandomizedEffect(
        "outcome",
        ant.experiment.ExperimentDesign(BernoulliAssignment(0.5), "arm", "id", "id"),
    )
    experiment = ant.prepare(experiment_data, query=design_query, refute=False)
    original = experiment.estimate()
    changed_design = {**experiment_data, "arm": ~experiment_data["arm"]}
    try:
        experiment.refresh(changed_design)
    except ant.CausalError as error:
        assert "arm" in str(error) and "reprepare" in str(error)
    else:
        raise AssertionError("changing retained assignments must be refused")
    assert experiment.last_result is original
    assert (
        experiment.refresh({"outcome": data["outcome"]}).answer.value
        == original.answer.value
    )
    # Repreparation returns a new study, resolves the names and repeats checks.
    revised = experiment.reprepare(changed_design)
    assert revised is not experiment and revised.last_result is None
    assert np.isclose(revised.estimate().answer.value, -original.answer.value)

    # A mean producer with incomplete coordinates requires a declaration.
    from antecedent import decision, joint_distribution, joint_law_propagation

    producer = joint_law_propagation.mixture(
        [joint_law_propagation.MixtureComponent("model", "evidence", 1.0, 2.0, 3.0)]
    )
    quantity = joint_distribution.ScientificQuantity.of(
        "outcome",
        "outcome",
        units="score",
        population="declared_target",
        regime="model",
    )
    mean = producer.as_mean_source(quantity=quantity)
    contract = decision.Contract(
        actions=[
            decision.Action("act", inputs=(quantity,), utility=decision.x(0)),
            decision.Action("wait", inputs=(quantity,), utility=-decision.x(0)),
        ],
        utility_units="score",
        target_population="declared_target",
        criterion=decision.Criterion.expected_utility(),
    )
    assert contract.evaluate(mean).source.means == (2.0,)

    envelope = checked.artifact()
    assert isinstance(checked.export(), bytes)
    assert envelope.data == checked.export()
    assert ant.artifacts.inspect(envelope).acceptance.status == "unavailable"
    consumed = ant.load(envelope, expected=envelope.expectation)
    assert consumed.answer.value == checked.answer.value
    assert consumed.acceptance.identity_bound
    with TemporaryDirectory() as directory:
        artifact_path = Path(directory) / "analysis.bin"
        expected_path = Path(directory) / "expected.json"
        artifact_path.write_bytes(envelope.data)
        expected_path.write_text(json.dumps(envelope.expectation.to_dict()["identity"]))
        script = """import json,sys; from pathlib import Path; import antecedent as ant
r=ant.load(Path(sys.argv[1]).read_bytes(), expected=json.loads(Path(sys.argv[2]).read_text()))
assert r.acceptance.identity_bound
print(json.dumps(r.answer.value))
"""
        replay = subprocess.run(
            [sys.executable, "-c", script, str(artifact_path), str(expected_path)],
            check=True,
            text=True,
            capture_output=True,
        )
        assert json.loads(replay.stdout) == checked.answer.value
    print("Fresh-process consumption preserved the answer and identity bindings.")


if __name__ == "__main__":
    main()
