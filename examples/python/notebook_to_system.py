"""Notebook exploration to independently consumed scientific bytes.

Run with an installed Antecedent. No files are retained after
this example; a real producer stores container bytes and expectations separately.
"""

from __future__ import annotations

# %%
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import antecedent as ant
import numpy as np
from antecedent.outcomes import require_success


def first_proof(view):
    """The first retained proof node of a prepared identification."""
    if view.kind == "proof":
        return view
    for child in view.children():
        selected = first_proof(child)
        if selected is not None:
            return selected
    return None


# %% [markdown]
# ## Prepare, inspect and estimate
#
# Preparation retains the scientific question and identification. Each estimate returns its own
# outcome; refutation belongs to that execution snapshot.

# %%
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
assert study.inspect().stage == "prepared"
proof = first_proof(study.identification)
assert proof.status == "nonparametrically_identified"
assert [item["name"] for item in proof.adjustment(estimand=0).items] == ["z"]
print(study.inspect())
result = require_success(ant.estimate(study))
checked = result.refute(suite="cheap")
assert checked.answer.value == result.answer.value
print(checked.inspect())

# %% [markdown]
# ## Rebind new observations
#
# Branch on new observations while retaining the scientific question and design. The shifted
# outcome moves the estimate by exactly one.

# %%
updated = require_success(
    ant.estimate(ant.rebind(study, data={**data, "outcome": data["outcome"] + treatment}))
)
assert np.isclose(updated.answer.value, result.answer.value + 1)
assert result.refute(suite="cheap").answer.value == result.answer.value

# %% [markdown]
# ## Guard a frozen experimental design and prepare a new one
#
# Named experimental assignments and unit IDs are frozen design bindings. Rebinding the outcome
# alone works; changing the assignment is refused, and a new preparation repeats the design checks.

# %%
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
original = ant.estimate(experiment)
changed_design = {**experiment_data, "arm": ~experiment_data["arm"]}
try:
    ant.rebind(experiment, data=changed_design)
except ant.CausalError as error:
    assert "arm" in str(error) and "ant.prepare" in str(error)
else:
    raise AssertionError("changing retained assignments must be refused")
unchanged = ant.rebind(experiment, data={"outcome": data["outcome"]})
assert ant.estimate(unchanged).answer.value == original.answer.value
revised = ant.prepare(changed_design, query=design_query, refute=False)
assert revised is not experiment and revised.inspect().stage == "prepared"
assert np.isclose(ant.estimate(revised).answer.value, -original.answer.value)

# %% [markdown]
# ## Compose a declared scientific mean
#
# A mean producer with incomplete coordinates requires a declaration of what it measures. The
# decision contract then evaluates against that declared source.

# %%
from antecedent import decision, joint_distribution, joint_law_propagation

producer = require_success(
    joint_law_propagation.mixture(
        [joint_law_propagation.MixtureComponent("model", "evidence", 1.0, 2.0, 3.0)]
    )
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
assert require_success(contract.evaluate(mean)).source.means == (2.0,)

# %% [markdown]
# ## Persist bytes and independently consume them
#
# A result carrying refutation reports loads intact and keeps its answer, but a replay is
# withheld and names what it cannot replay. The refutation-free result is then loaded in a fresh
# process from its bytes and expectation alone.

# %%
checked_envelope = checked.export()
withheld = require_success(ant.load(checked_envelope, expected=checked_envelope.expectation))
assert withheld.acceptance.recognized
audited = withheld.replay()
assert not audited.acceptance.replayed
assert "dependencies.refutation_replay" in audited.acceptance.not_replayable

envelope = result.export()
assert bytes(result.export()) == envelope.data
assert ant.artifacts.inspect(envelope).acceptance.status == "unavailable"
consumed = require_success(ant.load(envelope, expected=envelope.expectation))
assert consumed.answer.value == result.answer.value
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
    assert json.loads(replay.stdout) == result.answer.value
print("Fresh-process consumption preserved the answer and identity bindings.")
