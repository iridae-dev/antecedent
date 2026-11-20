# From a new experiment to the next study

A programme needs to improve its outcome by 1.1 units to pay for itself. Your first trial puts the gain just below that threshold. A second experiment reports a larger gain. How should you combine the two results, and does the programme now look worth running?

Run `python examples/python/experiment_to_next_study.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/experiment_to_next_study.ipynb). See the [complete example](#the-complete-example).

For an introduction to expected utility and the value of another study, see [stop, continue or run longer](decide-under-uncertainty.md).

## The task

The first trial assigns 32 independent clusters at random, 16 to the programme and 16 to a reference group. Each cluster contributes one average outcome. Subtract the reference group’s mean from the programme group’s mean to estimate the gain.

The simulated reference mean is 1.0 and the programme mean is 2.0. Their difference gives you a known value against which to check the fit.

The example uses the following inputs at each step.

| Declared | Learned |
| --- | --- |
| The randomized design: worlds, independent sampling unit, population, no spillover between clusters | 512 retained draws of the two arm means from one Gaussian fit |
| A later experiment's observed mean, its variance and sample size | Posterior weights on those draws after the experiment |
| A mapping from that experiment's estimand to the retained contrast | Expected value of a further study for each candidate |

## The retained response law

Each draw from the fit contains a possible mean for each arm. Subtract the two means within each draw. The spread of those differences measures uncertainty about the programme’s gain.

```python
result = require_success(prepared.execute({"inference": "randomized_gaussian", "prior": {...}}))
draws = result.report()["law"]["Moments"]["Bayesian"]["draws"]
```

`prepared` comes from `joint_panel.prepare_native_joint_design` with the randomized design and the 32 units; the prior is a Gaussian with mean 0 and variance 4 on each arm mean. The contrast of a draw is its active mean minus its reference mean (`COEFFICIENTS = [-1.0, 1.0]`). The mean contrast across the draws is about 0.96: the fixture's 1.0, shrunk slightly by the declared prior and moved by the Monte Carlo error of 512 draws. The example asserts that it lies within 0.1 of 1.0. The spread across draws is the fit's uncertainty about the contrast.

## Map the experiment's estimand onto the retained response

Before combining results, state how the quantity measured by the new experiment relates to the one you fitted. Here you assume that the new experiment independently estimates the same programme-minus-reference difference in the same units and setting. A `BridgeContract` records that mapping.

In this example the bridge declares an identity mapping (`AffineMapping(0.0, 1.0)`), the premise that an independent Gaussian sample mean estimates this exact contrast, a fixed mapping with no extra mapping uncertainty, and the exposure setting the two share. You supply `intercepts` and `slopes`, one per retained draw, so a mapping that varies across draws would update the law differently from a fixed one.

A bridge whose target is a different functional than the response being updated is refused. The Rust chain test checks, for instance, that a bridge supported for a one-window experiment cannot update a response defined over two windows.

The `EvidenceLedger` registers three sources with `add_source`: the fitted model, the new experiment and the mapping. The update depends on all three, and current permission to use it follows their standing (see [replay and current decisions](replay-and-current-decisions.md)).

## The changed response

```python
updated = result.condition_experiment(
    coefficients=COEFFICIENTS, outcome_units="outcome_unit", bridge=bridge, ledger=ledger,
    observed_mean=OBSERVED, observation_variance=0.2, sample_size=4, ...
)
report = updated.report(ledger)
```

Keep each draw’s pair of means and change its weight. A draw gets more weight if its predicted contrast is close to the new measurement. The Gaussian likelihood uses variance `0.2 / 4` to determine how close is close enough. The script checks the normalized weights against a separate calculation to `1e-12`.

The example sets the new measurement 0.3 units above the old mean. After reweighting, the estimated gain rises from about 0.96 to 1.17. Both experiments still contribute to that estimate.

## The changed decision

Suppose the programme should run when the contrast exceeds a break-even of 1.1 outcome units. The contract has two actions, `hold` with utility zero and `act` with utility `contrast − 1.1`, and compares them by expected utility.

At 0.96 units, the estimated gain fell short of the 1.1 break-even point and the decision favored `hold`. At 1.17, the expected net gain is about 0.07 units. `updated.decide(contract, ledger)` now selects `act`.

## Rank the next studies

Two candidate studies share one sampling model (4 observations of variance 1.0) and differ only in cost.

```python
ranking = updated.rank_next_samples(actions=actions, candidates=candidates, ledger=ledger, replicates=64)["candidates"]
```

The candidates are `cheap` (cost 0.01 in utility units) and `costly` (cost 0.1). Each is valued through the update it would cause, on the decision above. Its `gross_evsi` is the expected improvement in expected utility from deciding after the study rather than now, and `net_value` is that minus its cost. The two studies have equal gross value, so the cheaper ranks first, and the example checks `net_value = gross_evsi − cost`.

The example also checks the gross value against independent numerical integration. It computes the one-dimensional integral of the best action's advantage over the predicted observation, with a bound on the quadrature error, and requires the Monte Carlo value to agree within that bound plus six Monte Carlo standard errors. That check is what makes the ranking more than a list: a study's value is a number that two different computations produce.

For studies in another regime, the companion method `rank_next_studies` takes `PanelFutureStudy` declarations, each with its own supported bridge. A study whose bridge is unsupported, or whose mapping evidence was withdrawn, is refused.

## Read the update's status

```python
report["scope"]            # model_conditional_experimental_update;calibration_not_measured
report["modular_policy"]   # frozen_joint_importance
report["calibration"]      # not_measured
```

Read `report["scope"]` to see how to describe the updated result. Its scope is conditional on the fitted model and the mapping you supplied. `report["identification"]["required_assumptions"]` lists the design assumptions that still apply.

A more precise estimate can change which action you choose. Establishing that the contrast measures a causal effect still requires a design that identifies it.

## What this established

Under a stated mapping, a later experiment changed a retained response and, with it, a decision, without refitting the model. A further study's value was computed and checked against independent integration, and the update stayed a model-conditional experimental update with calibration `not_measured`.

## What this did not establish

The update did not identify the contrast as a causal effect, and it did not measure the calibration of the updated law. The mapping between the experiment and the retained contrast was declared, not tested, and the later experiment's mean was set by the example, not drawn from the simulated world.

## Where to go next

To choose among studies that might change a decision, see the [design ranking guide](../design-ranking.md). For a worked example that starts with the decision itself, see [decide under uncertainty](decide-under-uncertainty.md).

For one connected marketing investigation, [beyond the marketing mix](beyond-marketing-mix.md) follows a three-channel panel through spending proposals, experimental evidence, a business decision and the next study.

## The complete example

<!-- example: examples/python/experiment_to_next_study.py -->

[Open the Python script](../../examples/python/experiment_to_next_study.py)
