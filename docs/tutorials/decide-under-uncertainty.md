# Stop, continue or run longer

A retailer runs a geographic experiment with an advertising platform to measure incremental orders. The result suggests 1,000 additional orders per region per day, a 200% return on advertising spend and positive profit. The p-value is 0.07.

Should the retailer keep advertising, stop, or buy more evidence? At the estimated effect, the campaign makes money. To choose among those options, the retailer also needs to know how much it could lose and what another experiment would cost.

Run `python examples/python/decide_under_uncertainty.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/notebooks/decide_under_uncertainty.ipynb). See the [complete example](#the-complete-example).

## The task

The experiment measures additional orders per region per day. To decide on a rollout, the retailer needs profit in dollars across all the regions and days it plans to cover.

The data are simulated, and the mechanism is declared:

```text
orders[r] = 12,000 + 1,000 × advertised[r] + u[r]
```

The experiment assigns 20 regions to advertising and 20 to a holdout group at random. In the simulation, advertising adds 1,000 orders per region per day and affects only shoppers in the advertised region. Regional variation has a standard deviation of 1,750 orders. Both groups use the same set of equally spaced normal quantiles, so their difference in means recovers the simulated effect exactly.

| Declared | Learned |
| --- | --- |
| Assignment was randomized by region | The estimated effect and its standard error |
| No spillover between regions | Expected profit of each action |
| 25 dollars of margin per order; 20,000 dollars of advertising per advertised region per day | The chance of breaking even under a belief you state |
| A rollout of 50 regions for 180 days | The value of each candidate study, net of cost |

At 40 dollars of revenue per order, 1,000 orders and 20,000 dollars of spend give a return on ad spend of 2.0, which is the 200%. At 25 dollars of margin per order the break-even effect is 20,000 / 25 = 800 orders per region per day.

Randomization and the no-spillover premise are what make a simple comparison of regions mean the effect of advertising. A design in which regions influence one another needs a different estimand; see [randomized interference and saturation](../guides/designs/interference-and-network-exposure.md), not this comparison.

## The causal result, with its design and uncertainty

```python
result = require_success(
    analyze(data, query=AverageEffect("advertised", "orders"), graph=[("advertised", "orders")])
)
```

The query is the average effect of advertising on orders. The graph declares that nothing but chance assigns `advertised`. The example reads the estimate from `result.estimate.ate`, computes the standard error by hand from the two arms (the square root of the sum of each arm's sample variance over 20, which is 1,750 × √0.1) and prints:

```text
effect 1000 orders, standard error 553, z 1.81, p 0.071
```

The test statistic `z` is the estimate divided by its standard error, and the p-value is two-sided against a normal reference.

If advertising had zero effect and the test assumptions held, a result at least this far from zero in either direction would occur about 7% of the time. To calculate the probability that advertising loses money, you need a distribution over the effect given the data. You’ll specify one below.

A threshold of 0.05 is a convention for a hypothesis test. A business decision weighs what each outcome costs, and it need not share that threshold.

The result carries its own claim and calibration standing. Read `result.inspect().calibration` to see it. Unless it reports a matching measurement, the standard error is a reported number, not an interval with demonstrated coverage.

## The decision contract

A contract names the actions and the utility of each, in one unit.

```python
decision.Action("continue", inputs=(quantity,), utility=profit(decision.x(0))),
decision.Action("stop", inputs=(quantity,), utility=decision.x(0) * 0.0),
```

Here `profit` is `(orders_effect × 25 − 20,000) × 9,000`, in dollars over the rollout of 50 regions for 180 days. Stopping is worth zero. The input is the effect quantity from the result (`ScientificQuantity.from_effect`), which carries its units and population, so a number cannot be silently reread as something else.

Evaluating the contract on the result returns the expected utility of each action and the selection:

```python
effect_quantity = ScientificQuantity.from_effect(result, outcome_units="orders_per_region_day")
mean_contract = contract_for(effect_quantity, decision.Criterion.expected_utility())
decided = require_success(mean_contract.evaluate(result))
assert decided.selected == ("continue",)
```

```text
expected profit if we continue: 45.0 million dollars
```

`contract_for` is the example's helper that builds a `decision.Contract` from the two actions above, with utility in `usd`. Continuing is worth 45 million dollars on the point estimate (1,000 × 25 − 20,000 = 5,000 dollars per region-day, times 9,000), stopping is worth zero, and the contract selects `continue`.

The point estimate tells you which action has higher expected profit. To calculate a profit interval, regret, or the chance of losing money, you also need to specify how uncertainty about the effect is distributed.

## A risk question the result refuses

Ask for the tenth percentile of profit, a downside measure.

```python
tenth_percentile = contract_for(effect_quantity, decision.Criterion.quantile(0.1))
try:
    require_success(tenth_percentile.evaluate(result))
except decision.DecisionRefusal as refusal:
    assert refusal.reason_code == "decision_contract_unsatisfied"
```

```text
refused: decision_contract_unsatisfied
```

The refusal identifies the missing input. To calculate the tenth percentile of profit, the contract needs a distribution over possible profits.

The study compiler asks the same question earlier, before anything is fitted, and makes the next option concrete. `study_ranking.compile_decision_for_study` takes the study, the decision and the carrier the decision demands, and checks what the study's inference realization can publish. The example declares the same study twice, once with `posterior=False` (sampling) and once with `posterior=True` (conjugate Bayesian):

| Demanded of the study | Sampling realization | Conjugate Bayesian realization |
| --- | --- | --- |
| `mean`, for an affine utility of the point estimate | supplied | not exercised in the example |
| `parameter_posterior`, for expected utility and probabilities over the effect | refused; next attempt `bayesian_realization` | supplied |
| `outcome_law`, for the risk of one region's orders | refused; next attempt `interventional_law` | refused; next attempt `interventional_law` |

The sampling refusal for `parameter_posterior` has reason code `distribution_meaning_mismatch`. A confidence interval is not a posterior, and an average effect is not the law of one region's outcome. Each refusal says which carrier is missing and which realization could supply it.

## Downside risk under a belief you declare

To ask how likely a loss is, you need a distribution over the effect, and you must say where it comes from. The example declares a normal belief with the estimate as its mean and the standard error as its standard deviation. This is what a flat prior and approximately normal sampling error imply. The example represents it as a `JointDistributionArtifact` of 4,001 equally weighted atoms at the belief's quantiles, so its mean is the estimate. It is your judgment, stated in the example's code, and Antecedent labels the source as caller-built (`unverified`) with calibration `not_measured`. If you fit the study under a Bayesian realization instead, its posterior takes the place of this declaration.

Under that belief the contract can ask three things.

- **Expected utility** is the same 45 million dollars, because the belief's mean is the estimate.
- **The chance of breaking even** (`Criterion.threshold_probability(0.0)`) is about 64%, the probability that the effect reaches 800 orders. Equivalently, there is about a 36% chance continuing loses money. That is the statement the p-value was being asked to make.
- **A limit on losses.** The retailer allows at most a 10% chance of losing more than $10 million. Staying within that loss requires about 756 extra orders per region per day. Under the declared belief, the chance of reaching that level is only about 67%. The contract therefore excludes `continue` and selects `stop`.

```python
loss_cap = decision.Constraint(
    "loss-cap",
    -profit(decision.x(0)),
    bound=10_000_000.0,
    units="usd",
    min_probability=0.9,
    applies_to=("continue",),
)
```

```text
chance that continuing breaks even: 0.64
continuing reaches the loss cap with probability 0.67; required 0.90
```

The excluded action keeps its reason: `continuing.exclusions[0]` carries the `probability` reached and the probability `required`. The example checks both probabilities against the normal distribution function to within 0.002.

So expected value says continue, and a stated tolerance for loss says do not continue yet. Neither is a significance test, and the second is a policy you choose, not a fact the data established.

## Is more evidence worth its cost?

The remaining option is to run longer. Candidate studies add pairs of regions, one advertised and one held back. A pair's contrast has variance 2 × 1,750², and the current experiment is 20 such pairs. The example declares three candidates and their costs in dollars:

| Study | Extra pairs | Cost |
| --- | --- | --- |
| ten-more-pairs | 10 | 2.0 million |
| twenty-more-pairs | 20 | 4.0 million |
| forty-more-pairs | 40 | 12.0 million |

`design.rank_designs` values each by expected value of sample information (EVSI): how much better the decision becomes, in expectation, if you decide after the study instead of today. Net value subtracts the study's cost.

```python
ranked = design.rank_designs(
    studies,
    decision=declared,
    signal=signal,
    cost_map=design.CostMap("usd", "usd", 1.0),
)
print(ranked.explain())
```

`studies` holds one `design.Candidate` per row of the table, each with a `design.GaussianMeanSignal` of the pair variance. `declared` is the decision in the form the ranker reads, built by `design.DesignDecision.from_contract` with the belief as a `design.StatePrior.normal`. `ranked.candidates` is ordered by net value, and each entry carries `evsi`, `evpi` and `net_value`.

With a normal belief and a Gaussian study the value has a closed form, so the example computes it independently and requires agreement to a relative 1e-5; each candidate reports its integration method as `exact`. The values are about 11.6, 17.1 and 22.0 million dollars. Perfect information would be worth about 30.4 million dollars, an upper bound on any study. Net of cost the order is twenty pairs (about 13.1 million), then forty (about 10.0), then ten (about 9.6). The cheapest study is not the best, and the largest costs more than it adds.

Under the expected-profit rule, advertising now is worth about $45 million. Testing 20 more pairs first and then choosing whether to advertise adds about $13 million in expected profit after the cost of testing. The script prints the comparison below.

```text
decide today: continue is worth 45.0 million dollars
best study twenty-more-pairs: expected value 17.1, net 13.1
```

The ranking values expected utility only. It carries neither a hard constraint nor a quantile, and a contract that declares one is refused (`DesignDecision.from_contract` names the hard constraints). So the loss policy shapes which action is admissible today, and the ranking tells you what information is worth for the expected-value decision. They answer different questions.

If monitoring or early stopping is part of the experimental design, the inference must account for that procedure. Looking repeatedly until a conventional threshold is crossed is a different procedure from the single analysis here.

## Boundaries worth keeping

The belief in the downside-risk section is a declaration. The example's probabilities and values are exact for that belief, and they inherit its premises: randomized assignment, no spillover, an approximately normal sampling error, and a flat prior. A different belief gives different numbers, which is why it is written down where you can challenge it.

When an input to a decision is itself a corrected measurement or a pooled estimate from several studies, the same discipline applies: declare the correction, name the evidence it rests on, and carry a point value only as a point value. [Measurement models](../measurement-models.md) covers corrected measurements, and the [evidence ledger](../evidence-ledger.md) covers pooled evidence and what a revised source changes downstream.

## What this established

The choice depends on the retailer’s rule. Maximizing expected profit favors advertising. Imposing the $10 million loss limit rules it out on current evidence. For a retailer maximizing expected profit and willing to wait, testing 20 more pairs gives the largest expected gain after costs.

## What this did not establish

The belief itself was not established: the downside numbers are exact for the declared normal belief and change with it. The value of each study is an expectation under that same belief, not a forecast of what the study will find. Randomization and no spillover were declared for the fixture, not tested, and the 1,000-order effect is a property of the simulation.

## Where to go next

If the retailer can randomize 20 more pairs without spillover between regions, the next step is to [update the estimate with the new experiment](experiment-to-next-study.md). For the underlying calculations, see [decisions](../decisions.md) and [design ranking](../design-ranking.md#ranking-studies-by-value-of-information). For saved decisions, see [replay and current decisions](replay-and-current-decisions.md).

## The complete example

<!-- example: examples/python/decide_under_uncertainty.py -->

[Open the Python script](../../examples/python/decide_under_uncertainty.py)
