# From a new experiment to the next study

A response you have already estimated is not final when a new experiment arrives. This tutorial takes a retained response law, brings in a later experiment through an explicit mapping between their estimands, and follows the consequences: the response changes, a decision changes, and the candidate further studies can be ranked by what they are worth and what they cost. Along the way the update's status stays visible. The evidence changes the result; it does not turn the result into an identified causal claim.

## The task

A programme is compared with a reference by randomizing 32 independent clusters, half to each arm. The outcome is a cluster-level average in a declared outcome unit. The question is the whole-programme contrast: the population mean under the active programme minus the population mean under the reference.

The data are simulated. The reference arm averages 1.0 and the active arm 2.0, so the fixture's true contrast is 1.0. That known truth is useful because every later step can be checked against arithmetic you can do by hand. It is a truth of the stated model, not a claim about any real programme.

Two things are declared and three are learned.

| Declared | Learned |
| --- | --- |
| The randomized design: worlds, independent sampling unit, population, no spillover between clusters | 512 retained draws of the two arm means from one Gaussian fit |
| A later experiment's observed mean, its variance and sample size | Posterior weights on those draws after the experiment |
| A mapping from that experiment's estimand to the retained contrast | Expected value of a further study for each candidate |

## The retained response law

The fit returns draws, not only a point estimate. Each draw is a coherent pair of arm means, so any contrast of them has an uncertainty that comes from the same fit.

```python
result = require_success(prepared.execute({"inference": "randomized_gaussian", "prior": {...}}))
draws = result.report()["law"]["Moments"]["Bayesian"]["draws"]
```

The mean contrast across the draws is close to 0.985: the fixture's 1.0, shrunk slightly by the declared prior. The spread across draws is the fit's uncertainty about the contrast.

## Map the experiment's estimand onto the retained response

Before an experiment can update anything, state what it measures in the retained response's terms. An experiment's observed mean is a statement about its own estimand: a sample mean in its own setting. The retained law is a statement about the whole-programme population mean contrast. A `BridgeContract` says how one becomes the other, and nothing is updated without one.

In this example the bridge declares an identity mapping (`AffineMapping(0.0, 1.0)`), the premise that an independent Gaussian sample mean estimates this exact contrast, a fixed mapping with no extra mapping uncertainty, and the exposure setting the two share. You supply `intercepts` and `slopes`, one per retained draw, so a mapping that varies across draws would update the law differently from a fixed one.

A bridge whose target is a different functional than the response being updated is refused. The Rust chain test checks, for instance, that a bridge licensed for a one-window experiment cannot update a response defined over two windows.

The ledger registers three sources: the fitted model, the new experiment and the mapping. The update depends on all three, and current permission to use it follows their standing (see [replay and current decisions](replay-and-current-decisions.md)).

## The changed response

```python
updated = result.condition_experiment(
    coefficients=COEFFICIENTS, outcome_units="outcome_unit", bridge=bridge, ledger=ledger,
    observed_mean=OBSERVED, observation_variance=0.2, sample_size=4, ...
)
report = updated.report(ledger)
```

The update is frozen importance conditioning. Each retained draw keeps its values and receives a weight proportional to how well its predicted experimental mean matches the observed one, under a Gaussian likelihood with variance `0.2 / 4`. The model is not refit. The example recomputes those weights with independent arithmetic and requires agreement to `1e-12`.

The experiment's mean was declared as 0.3 above what the retained law expected. The posterior mean contrast moves from about 0.985 to about 1.2. It does not move all the way to the observed mean, because the retained law and the experiment each carry information and the weights combine them.

## The changed decision

Suppose the programme should run when the contrast exceeds a break-even of 1.1 outcome units. The contract has two actions, `hold` with utility zero and `act` with utility `contrast − 1.1`, and compares them by expected utility.

Before the experiment the retained mean contrast (about 0.985) is below break-even, so the same rule holds. After it, `updated.decide(contract, ledger)` returns a uniquely optimal `act`, and `act`'s expected utility equals the posterior contrast minus the break-even. The experiment changed the decision. Whether that change is worth trusting is the question the next section answers.

## Read the update's status

```python
report["scope"]            # model_conditional_experimental_update;calibration_unmeasured
report["modular_policy"]   # frozen_joint_importance
report["calibration"]      # unmeasured
```

The update reweights draws from a fit whose design premises are listed in `report["identification"]`. It does not re-derive identification, and it does not measure calibration. So the result of the update is a model-conditional experimental update, and the field says so. The decision above is a decision under that status.

This is the boundary the tutorial exists to teach. Evidence can legitimately change a response and a decision without upgrading the claim. The Rust chain test checks that asking the updated source for identified-causal permission is refused; the Python report exposes the same status as the `scope` field. What would change the status is not more weighting but a design whose premises establish identification for the quantity you are deciding on, which is a different study.

## Rank the next studies

Two candidate studies share one sampling model (4 observations of variance 1.0) and differ only in cost.

```python
ranking = updated.rank_next_samples(actions=actions, candidates=candidates, ledger=ledger, replicates=64)["candidates"]
```

Each candidate is valued through the update it would cause, on the decision above. Its `gross_evsi` is the expected improvement in expected utility from deciding after the study rather than now, and `net_value` is that minus its cost. The two studies have equal gross value, so the cheaper ranks first, and the example checks `net_value = gross_evsi − cost`.

The example also checks the gross value against independent numerical integration. It computes the one-dimensional integral of the best action's advantage over the predicted observation, with a bound on the quadrature error, and requires the Monte Carlo value to agree within that bound plus six Monte Carlo standard errors. That check is what makes the ranking more than a list: a study's value is a number that two different computations produce.

The same call accepts studies in another regime through `rank_next_studies` and `PanelFutureStudy`, each with its own licensed bridge. A study whose bridge is unlicensed, or whose mapping evidence was withdrawn, is refused.

<!-- example: examples/python/experiment_to_next_study.py -->

## What this established

Under a stated mapping, a later experiment changed a retained response and, with it, a decision, without refitting the model. A further study's value was computed and checked against independent integration, and the update stayed a model-conditional experimental update with calibration unmeasured.

The next scientific question is which of two things you need: a study that sharpens this decision (the ranking above), or a study whose design identifies the contrast you are deciding on, which would change the status rather than the weights. The [design ranking guide](../design-ranking.md) covers ranking studies in general, and [decide under uncertainty](decide-under-uncertainty.md) follows a decision and the value of more evidence from its start.
