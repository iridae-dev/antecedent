# Migrate from 3.0 to 3.1

The [2.4 to 3.0 guide](migration.md) still applies. These are the 3.1 calls. Each block is self-contained and checks the value or refusal it describes.

## Declare a slope the trial estimates

`estimate_cuped_effect(outcomes, covariate, assignment, propensity)` still runs, warns, and does not go through the compiler. Declare the same covariate on the query. The compiled estimator is `randomized.estimated_cuped_ht_itt` (`Estimator.RANDOMIZED_ESTIMATED_CUPED_HT_ITT`). The result publishes the pooled slope as `adjustment_coefficient` and names the route in `uncertainty`.

Nuisance models you supply are a different declaration: [Use your own outcome and treatment models](tutorials/external-learners.md). `analyze_trial_aipw` and `analyze_joint_cells` are the compiled forms of those learner routes.

<!-- exec: migrate-cuped -->
```python
import warnings

import numpy as np

import antecedent as ant
from antecedent import interference
from antecedent.experiment import (
    EstimatedCUPED,
    ExperimentDesign,
    RandomizedEffect,
    estimate_cuped_effect,
)
from antecedent.outcomes import require_success

rows = 40
assigned = [i % 2 == 0 for i in range(rows)]
baseline = np.sin(0.2 * np.arange(rows))
outcome = 1.0 + 1.5 * baseline + np.asarray(assigned, dtype=float)
design = ExperimentDesign(
    interference.BernoulliAssignment(0.5),
    assigned,
    [f"u{i}" for i in range(rows)],
    [f"r{i}" for i in range(rows)],
)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", FutureWarning)
    direct = estimate_cuped_effect(outcome, baseline, assigned, 0.5)
assert any("estimate_cuped_effect" in str(item.message) for item in caught)
fit = require_success(
    ant.analyze(
        {"outcome": outcome, "baseline": baseline},
        query=RandomizedEffect(
            "outcome", design, estimated_cuped=EstimatedCUPED("baseline")
        ),
        refute="none",
    )
)
effect = fit.randomized_effect
assert effect.adjustment_coefficient == direct.adjustment_coefficient
assert "estimated_cuped" in effect.uncertainty
assert (
    ant.Estimator.RANDOMIZED_ESTIMATED_CUPED_HT_ITT
    == "randomized.estimated_cuped_ht_itt"
)
```

## Compile the decision, then run it

`Contract.compile` binds the contract to a prepared study and states the demand before any fit. `execute` is the fit. On this line, the higher dose ranks first.

<!-- exec: migrate-decision -->
```python
import numpy as np

import antecedent as ant
from antecedent import decision
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

dose = np.linspace(0.0, 2.0, 48)
prepared = ant.prepare(
    {"t": dose, "y": 0.5 + 1.5 * dose},
    graph=[("t", "y")],
    query=ant.ResponseCurve("t", "y", grid=[0.5, 1.5]),
    inference=ant.Frequentist(bootstrap=0),
    refute="none",
)


def mean_at(value: float) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id="y",
        variable_name="y",
        role="outcome",
        units="points",
        population_id="target",
        regime_id=f"do(t={value})",
        horizon=0,
        functional_id="mean",
    )


contract = decision.Contract(
    actions=(
        decision.Action("low", inputs=(mean_at(0.5),), utility=decision.x(0)),
        decision.Action("high", inputs=(mean_at(1.5),), utility=decision.x(0)),
    ),
    utility_units="points",
    criterion=decision.Criterion.expected_utility(),
    target_population="target",
)
compiled = require_success(contract.compile(prepared))
assert compiled.representation == "mean"
decided = require_success(compiled.execute())
assert decided.choice == "high"
assert decided.standing == "best"
assert (
    decided.statement
    == "Action high ranks first on expected utility of a frequentist point estimate."
)
```

`standing` and `statement` are the words on every decision result. `best` is a unique first rank. The wire names added for trade-off ranges (`ClassPriorIdentityWire::Uniform`, `ClassPriorIdentityWire::Annotated`, `StructuralWeightBasisWire::PosteriorWithDeclaredClassPrior`, `StructuralWeightBasisWire::SampledClassMeasure`, `JointDesignInference::IndependentUnitBootstrap`) are new variants. Existing wire bytes, labels, and identities are unchanged.

## Resume the study, or load the claim

`save_compiled_study` writes the program. `open_study` with that identity can estimate again, and the resumed estimate equals the sealed claim. A container that is not that program opens without an estimate. `load` of an analysis artifact reads the claim and does not restore the study. Pass the identity the consumer retained (`artifacts.artifact_identity`) as `retained_identity=`.

<!-- exec: migrate-resume -->
```python
import antecedent as ant
from antecedent import compiled_study

outcome = ant.analyze(
    {"a": [0.0, 0.0, 1.0, 1.0], "y": [0.0, 0.0, 2.0, 2.0]},
    graph=[("a", "y")],
    query=ant.AverageEffect("a", "y"),
    inference=ant.Frequentist(bootstrap=0),
    refute="none",
)
assert outcome.answer.value == 2.0
saved = outcome.study.save_compiled_study()
resumed = compiled_study.open_study(saved.data, saved.expected_identity)
assert resumed.resumable
assert resumed.estimate == resumed.claim_estimate == 2.0

opaque = compiled_study.write(b"opaque", ())
opened = compiled_study.open_study(opaque.data, opaque.expected_identity)
assert opened.resumable is False
assert opened.estimate is None

artifact = outcome.export()
kept = ant.artifacts.artifact_identity(artifact.data)
loaded = ant.load(
    artifact.data, expected=artifact.expectation, retained_identity=kept
)
assert loaded.study is None
assert loaded.answer.value == 2.0
```

## Read a resource refusal

A limit that stops the work before a scientific result exists raises `errors.CausalResourceError`. The message names the pool. An admission budget of `memory_bytes=0` says `admission wait`. The registered code [`execution_resource_limit`](reference/refusal-codes.md) is that refusal when a route reports a reason code. No effect follows from either form.

<!-- exec: migrate-resource -->
```python
import antecedent as ant
from antecedent.errors import CausalResourceError
from antecedent.execution import ResourceLimits

try:
    ant.prepare(
        {"t": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 2.0, 0.0, 2.0]},
        graph=[("t", "y")],
        query=ant.AverageEffect("t", "y"),
        inference=ant.Frequentist(bootstrap=0),
        refute="none",
        execution=ant.Execution(limits=ResourceLimits(memory_bytes=0)),
    )
except CausalResourceError as refusal:
    assert "admission wait" in str(refusal)
else:
    raise AssertionError("a zero memory budget must refuse")
```

## A law that states no support mask

`DecisionInput.from_distribution` reads the artifact's mask as coordinate support. An artifact with no mask states no coordinate. Without a native execution record behind the law, each coordinate is `missing_evidence`.

<!-- exec: migrate-mask -->
```python
import numpy as np

from antecedent import composition
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)


def coordinate(regime: str) -> ScientificQuantity:
    return ScientificQuantity.of(
        "outcome", "y", units="units", population="target", regime=regime
    )


identity = DistributionIdentity(
    semantic="interventional_predictive",
    quantities=(coordinate("do(a=0)"), coordinate("do(a=1)")),
    alignment="joint",
    source_id="study",
    provider_id="caller",
    rng_id="deterministic_exact",
    snapshot_id="snap",
    causal_contract_id="checked-contract",
)
law = JointDistributionArtifact(
    identity,
    np.array([[1.0, 3.0], [2.0, 4.0]]),
    calibration="not_measured",
    trust="externally_attested",
)
source = composition.DecisionInput.from_distribution("law", law)
assert [item.status for item in source.support] == [
    "missing_evidence",
    "missing_evidence",
]
```

## Recalculation receipt version

A recalculation receipt records its planner rule in the version word. Version 1 is the first-change rule, and its bytes are the bytes from before this release, so an older reader still accepts it. Version 2 is the removal-first rule, written when a removed input sat beside an earlier change. `RECALC_RECEIPT_ARTIFACT_VERSION` is 1 and `RECALC_RECEIPT_REMOVAL_FIRST_VERSION` is 2. `planner_rule_of_version` maps those two words and refuses any other. A reader older than this release refuses a version 2 receipt.
