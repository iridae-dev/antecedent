# Stop, continue or run longer

A retailer runs a geographic experiment with an advertising platform to measure incremental orders. The result suggests 1,000 additional orders per region per day, a 200% return on advertising spend and positive profit. The p-value is 0.07.

Should the retailer stop because the effect is not significant at 0.05? Continue because the estimated return is positive? Or run the experiment longer? This tutorial answers with a causal result, a decision contract, a risk question, and the value of more evidence, and it shows where each step refuses and why.

## The task

A decision to advertise is made for a population of regions, over a stretch of days, in dollars. The experiment gives an estimate of the effect on orders; the business supplies what an order is worth and what advertising costs. Antecedent connects the two and keeps the uncertainty attached.

The data are simulated, and the mechanism is declared:

```text
orders[r] = 12,000 + 1,000 × advertised[r] + u[r]
```

There are 40 regions. Twenty were chosen at random for advertising and 20 held back. Regional variation `u` has a standard deviation of 1,750 orders per region per day, and shoppers in one region are not affected by advertising in another. The true effect is 1,000 orders per region per day. That is a truth of this fixture and useful because every later number can be checked; it says nothing about any real retailer.

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

The query is the average effect of advertising on orders. The graph declares that nothing but chance assigns `advertised`. The estimate is 1,000 orders with a standard error of about 553, a ratio of about 1.8. By construction the sample effect equals the true effect here, and the standard error is a fixed number the example computes by hand from the two arms.

The p-value of 0.07 says this: if the true effect were zero, an estimate at least this far from zero would turn up about 7% of the time under the test's assumptions. It is not a 7% probability that the campaign has no effect, and 0.93 is not the probability that it works. Those are statements about the effect given the data, and they need a belief about the effect that a p-value does not contain. The section on downside risk supplies one, explicitly.

A threshold of 0.05 is a convention for a hypothesis test. A business decision weighs what each outcome costs, and it need not share that threshold.

The result carries its own claim and calibration standing. Read `result.inspect().calibration` to see it. Unless it reports a matching measurement, the standard error is a reported number, not an interval with demonstrated coverage.

## The decision contract

A contract names the actions and the utility of each, in one unit.

```python
decision.Action("continue", inputs=(quantity,), utility=profit(decision.x(0))),
decision.Action("stop", inputs=(quantity,), utility=decision.x(0) * 0.0),
```

Here `profit` is `(orders_effect × 25 − 20,000) × 9,000`, in dollars over the rollout of 50 regions for 180 days. Stopping is worth zero. The input is the effect quantity from the result (`ScientificQuantity.from_effect`), which carries its units and population, so a number cannot be silently reread as something else.

`contract.evaluate(result)` returns expected utility. Continuing is worth about 45 million dollars on the point estimate (1,000 × 25 − 20,000 = 5,000 dollars per region-day, times 9,000). The contract's choice is `continue`.

That answer is a point value. A standard error is not a distribution of profit, and the decision reports no interval, no regret and no replayable export, because a mean retains nothing to compute them from. The next step asks for exactly what a mean cannot give.

## A risk question the result refuses

Ask for the tenth percentile of profit, a downside measure.

```python
decision.Criterion.quantile(0.1)
```

`evaluate` refuses with `decision_contract_unsatisfied`. A quantile needs a distribution; the result supplies a mean. Nothing is invented to fill the gap, and the refusal names what is missing, not that the arithmetic failed.

The study compiler asks the same question earlier, before anything is fitted, and makes the next option concrete. For the same decision it checks what the study's inference realization can publish:

| Demanded of the study | Sampling realization | Conjugate Bayesian realization |
| --- | --- | --- |
| `mean`, for an affine utility of the point estimate | supplied | not exercised in the example |
| `parameter_posterior`, for expected utility and probabilities over the effect | refused; next attempt `bayesian_realization` | supplied |
| `outcome_law`, for the risk of one region's orders | refused; next attempt `interventional_law` | refused; next attempt `interventional_law` |

A confidence interval is not a posterior, and an average effect is not the law of one region's outcome. Each refusal says which carrier is missing and which realization could supply it.

## Downside risk under a belief you declare

To ask how likely a loss is, you need a distribution over the effect, and you must say where it comes from. The example declares a normal belief with the estimate as its mean and the standard error as its standard deviation. This is what a flat prior and approximately normal sampling error imply. It is your judgment, stated in the example's code, and Antecedent labels the source as caller-built (`unverified`) with calibration unmeasured. If you fit the study under a Bayesian realization instead, its posterior takes the place of this declaration.

Under that belief the contract can ask three things.

- **Expected utility** is the same 45 million dollars, because the belief's mean is the estimate.
- **The chance of breaking even** (`Criterion.threshold_probability(0.0)`) is about 64%, the probability that the effect reaches 800 orders. Equivalently, there is about a 36% chance continuing loses money. That is the statement the p-value was being asked to make.
- **A downside policy** declared as a hard constraint: at most a 10% chance of losing more than 10 million dollars. Continuing reaches that bound with probability about 0.67 against 0.90 required, so it is excluded on today's evidence. A constraint excludes an action; it never subtracts a penalty from its utility.

So expected value says continue, and a stated tolerance for loss says do not continue yet. Neither is a significance test, and the second is a policy you choose, not a fact the data established.

## Is more evidence worth its cost?

The remaining option is to run longer. Candidate studies add pairs of regions, one advertised and one held back. A pair's contrast has variance 2 × 1,750², and the current experiment is 20 such pairs. The example declares three candidates and their costs in dollars:

| Study | Extra pairs | Cost |
| --- | --- | --- |
| ten-more-pairs | 10 | 2.0 million |
| twenty-more-pairs | 20 | 4.0 million |
| forty-more-pairs | 40 | 12.0 million |

`design.rank_designs` values each by expected value of sample information (EVSI): how much better the decision becomes, in expectation, if you decide after the study instead of today. Net value subtracts the study's cost.

With a normal belief and a Gaussian study the value has a closed form, so the example computes it independently and requires agreement. The values are about 11.6, 17.1 and 22.0 million dollars. Perfect information would be worth about 30.4 million dollars, an upper bound on any study. Net of cost the order is twenty pairs (about 13.1 million), then forty (about 10.0), then ten (about 9.6). The cheapest study is not the best, and the largest costs more than it adds.

Read the answer against the three options. Continuing now is worth about 45 million dollars in expectation. Running twenty more pairs first adds about 13 million net in expectation, and it narrows the belief that the loss policy rejected. Stopping forgoes both. On these declared inputs the best move is to run the twenty-pair study, then decide.

The ranking values expected utility only. It carries neither a hard constraint nor a quantile, and a contract that declares one is refused (`DesignDecision.from_contract` names the hard constraints). So the loss policy shapes which action is admissible today, and the ranking tells you what information is worth for the expected-value decision. They answer different questions.

If monitoring or early stopping is part of the experimental design, the inference must account for that procedure. Looking repeatedly until a conventional threshold is crossed is a different procedure from the single analysis here.

<!-- example: examples/python/decide_under_uncertainty.py -->

## Boundaries worth keeping

The belief in the downside-risk section is a declaration. The example's probabilities and values are exact for that belief, and they inherit its premises: randomized assignment, no spillover, an approximately normal sampling error, and a flat prior. A different belief gives different numbers, which is why it is written down where you can challenge it.

When an input to a decision is itself a corrected measurement or a pooled estimate from several studies, the same discipline applies: declare the correction, name the evidence it rests on, and carry a point value only as a point value. The `measurement_to_decision` example works that case by hand, with a ledger that marks pooled evidence for recomputation when one source grows.

## What this established

Under the declared design and belief, continuing is worth about 45 million dollars on average, has roughly a one-in-three chance of losing money, fails a stated downside policy, and is improved most, net of cost, by running a twenty-pair study first. The p-value of 0.07 did not decide any of that. The risk question a mean-only result could not answer refused, and a declared belief answered it with its status attached.

The next scientific question is whether the twenty-pair study is feasible under the same randomization and non-interference premises. If it is, the continued use is to [update the response with the new experiment](experiment-to-next-study.md) and decide again. To keep an exported decision reproducible while its sources change, see [replay and current decisions](replay-and-current-decisions.md).
