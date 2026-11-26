# When the causal assumptions stop holding

In January, a subscription service estimates that customers who use the product more are less likely to cancel. In April, the same analysis gives a much smaller effect. Did engagement become less valuable, or did something change which customers were using the product?

You’ll investigate the change using two simulated customer cohorts. The simulation lets you check when an apparent change in the effect is actually a change in who is being compared.

Run `python examples/python/when_assumptions_change.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/notebooks/when_assumptions_change.ipynb). See the [complete example](#the-complete-example).

The graph checks use conditional independence and adjustment for a common cause. For a worked example of adjustment, see [estimate an effect from observational data](observational-study.md).

## The task

A subscription business wants to know how much weekly engagement changes cancellations. It has five measured columns per customer, all standardized scores: `price`, `tenure`, `engagement`, `support` (contacts with the support team) and `cancellation`.

In both months, raising engagement by one unit reduces the cancellation score by 0.8. In April, an integration starts failing. Customers who depend on it spend more time in the product, contact support more often and are more likely to cancel. The table never records who uses the integration.

| Declared by a person | Learned from data |
| --- | --- |
| The direction of the one edge discovery left open in January | January's graph, apart from that edge |
| Three readings of the new link that appears in April | That April's table contradicts two independencies the January graph asserts |
| Two candidate studies, with their costs and sample sizes | Which candidate study would identify the effect |

Each monthly table contains 4,000 customers. The script fixes the random seeds so you can compare your output with the values below.

## January: discover, review, estimate

PC discovery returns five directed edges and leaves one, between `price` and `support`, undirected. The data do not fix it, so a reviewer supplies it, and the accepted graph records a new version.

```python
discovered = disc.discover(january, method=disc.PC(alpha=0.01), accept_discovered=True, execution=EXEC)
discovered.pending          # [price -- support]
accepted = discovered.review({("price", "support"): ("tail", "arrow")})

study = ant.prepare(january, graph=accepted, query=ant.AverageEffect("engagement", "cancellation"), ...)
first = require_success(ant.estimate(study))    # -0.79
```

The estimate is -0.79 against a simulated -0.8; the example asserts it is within 0.1. The graph says price is the only common cause of engagement and cancellation, and that holds in January.

## April: the same analysis, a different number

```python
rerun = require_success(ant.estimate(ant.rebind(study, data=april)))    # -0.36
```

Rebinding repeats January's analysis on April's rows: the same graph, the same adjustment set. It returns -0.36, and the example asserts a move of more than 0.3. More than half the effect appears to have gone.

It has not. The simulated effect is still -0.8. The estimate moved because customers on the failing integration are both more engaged and more likely to cancel, and nothing in the adjustment set accounts for them. No error is raised, because no computation failed. An estimate cannot report that its own graph is out of date.

## Three checks, three different questions

### Is the January claim stale?

```python
stale = require_success(drift.staleness(study, april, policy=drift.StalenessPolicy(alpha=0.01)))
stale.recommendation                       # "reidentify"
[f.subject for f in stale.findings]
# ['tenure _||_ cancellation | price, engagement', 'engagement _||_ support | price']
```

The [staleness diagnostic](../guides/staleness.md) compares April's table with what the plan was prepared on. Most of what it checks is coverage: covariate distributions, the range of the treatment and the adjustment set, missingness, schema. April passes all of that. Its customers cover the same ranges as January's, and the adjustment is still computable on them.

It also repeats, on the new rows, the test of every conditional independence the plan's graph implies. January's graph implies four. Two of them fail in April.

| Statement the graph implies | January, second sample | April |
| --- | --- | --- |
| price and tenure are independent | holds | holds |
| tenure and support are independent | holds | holds |
| engagement and support are independent given price | holds (0.01) | fails (0.28) |
| tenure and cancellation are independent given price and engagement | holds (-0.01) | fails (-0.17) |

The numbers are partial correlations. The diagnostic flags a relation when its magnitude reaches 0.1 and the test is significant after adjustment for multiple comparisons. On a second January sample, no relation meets both conditions and the recommendation is `reuse`.

The recommendation is `reidentify`, not `refresh`. Refreshing would repeat January's analysis on April's rows, which is the -0.36 above. The finding says the structure that analysis rests on no longer agrees with the data.

### How wrong is the graph as a whole?

The staleness finding is about single statements. [`evaluate_graph`](../guides/graph-falsification.md) asks a different question of the whole graph: does it explain the table better than random relabellings of its own nodes?

```python
checked = evaluate_graph(april, GRAPH, alpha=0.01, n_permutations=500, permutation_seed=7)
checked.outcome, checked.n_violations     # "not_rejected", 3 of 7
```

| Table | Verdict | Violated implications |
| --- | --- | --- |
| A second January sample | `not_rejected` | 0 of 7 |
| April | `not_rejected` | 3 of 7, including engagement and support given price |

The verdict is the same in both rows. January's graph still explains April better than its relabellings do, and `not_rejected` means better than chance, which a partly wrong graph easily is. The two results do not conflict. One says a specific statement failed; the other says the graph as a whole is still far from arbitrary. Both are true of a graph that is mostly right and missing one cause.

### Is that sampling noise?

Two tables can give two graphs when nothing has changed. `BootstrapStability` re-runs the search on resamples of one table and reports how often each pair of variables is joined.

```python
stability = disc.BootstrapStability(disc.PC(alpha=0.01), replicates=200)
stability.run(january, execution=EXEC).edges    # engagement, support joined in 2.5% of resamples
stability.run(april, execution=EXEC).edges      # joined in 99%
```

The engagement–support edge appears in 2.5% of January resamples and 99% of April resamples. The April search keeps finding it even as the sample changes. Those percentages measure the stability of the search within each month.

## Discover again, allowing for what is not measured

PC assumes every common cause is in the table. FCI does not, and marks with a circle each edge end the data leave open.

```python
fci = disc.FCI(alpha=0.01, fdr=False)
```

| Edge | January | April |
| --- | --- | --- |
| engagement, cancellation | tail, arrow: engagement causes cancellation | circle, circle |
| engagement, support | not joined | joined |

In April the search still finds engagement and cancellation related. It no longer says which way the relation runs, or whether something unmeasured sits behind both. That is a weaker result than January's, and the weaker result is the accurate one.

A changed discovery result is evidence to weigh. It does not replace the accepted graph, and `AcceptedGraph` never rediscovers unless asked.

## Three readings, three answers

The new link between engagement and support can be read at least three ways, and each is a different causal claim.

| Reading | Graph check on April | Effect of engagement |
| --- | --- | --- |
| Engaged customers contact support more | `rejected` | -0.36 |
| Support contacts change engagement | `not_rejected`, 1 of 5 violated | -0.61 |
| An unmeasured cause drives engagement, support and cancellation | not testable this way | refused: `effect_not_identified` |

The first two are complete graphs, so each is tested with `evaluate_graph` and estimated with `analyze`. The data discard the first. The second survives the check and gives -0.61.

The third is declared as a graph with unmeasured common causes, and the question is put to it as it stands.

```python
hidden_cause = ant.Admg.from_edges(
    NAMES, GRAPH,
    [("engagement", "support"), ("engagement", "cancellation"), ("support", "cancellation")],
)
refused = ant.analyze(april, graph=hidden_cause, query=QUERY, ...)
refused.refusal.code    # "effect_not_identified"
```

The refusal is the answer under that reading: with a hidden cause of both engagement and cancellation, no adjustment over the measured columns recovers the effect.

The second graph passes the check but gives −0.61. We know from the simulation that the true effect is −0.8 and that the failing integration influences both variables. With real data, you would need evidence from outside this table to distinguish a direct relationship from a hidden common cause.

The readings are put to the question one at a time here. `GraphPosterior.from_graphs` can hold several complete graphs with weights and report the mass on which an effect is not identified, but its atoms cannot carry a directed and a bidirected edge on the same pair, which is exactly what the third reading declares between engagement and cancellation. That reading has to be asked as its own graph.

## What would settle it

Asked to describe, the refused call also returns the [evidence the failure determines](../guides/fallback.md).

```python
described = ant.analyze(
    april, graph=hidden_cause, query=QUERY, on_refusal="describe",
    evidence_candidates=[experiment, more_rows], ...
)
[action["id"] for action in described.evidence_plan.next_actions]
```

```text
collect_panel
find_instrument
measure_variable:common_cause_of_engagement_cancellation
measure_variable:common_cause_of_engagement_support
run_experiment:engagement
```

`common_cause_of_engagement_cancellation` describes a role in the proposed graph. Measuring a cause in that position could resolve the ambiguity. Finding out whether the failing integration plays that role takes knowledge of the service.

Two candidate studies are declared, with costs that are yours to supply: ten thousand more rows of the same table, and an experiment that sets engagement for 2,000 customers. Each is checked against the identification theorem of its route.

```python
plan = described.evidence_plan.plan
plan.best.labels, plan.best.gate    # ("engagement experiment",), "verified_sufficient"
plan.unlocks.unverified             # ("ten thousand more rows",)
```

The experiment is certified. More rows are not, at an eighth of the cost. The plan is hypothetical identification, not an estimate: its ordering is `point_only` and its calibration `not_measured`. It also lists four assumptions that no study establishes, such as the validity of an instrument.

Because the mechanism is simulated, both studies can be run.

| Study | Estimate | Simulated truth |
| --- | --- | --- |
| 14,000 observational rows, January's graph | -0.34 | -0.8 |
| 2,000 customers with engagement assigned at random | -0.83 | -0.8 |

More rows sharpen the wrong number. The experiment assigns engagement independently of everything else, so in its table engagement has no causes to adjust for, measured or not.

Setting engagement directly is a simplification. A real programme would more likely encourage engagement at random, which makes the assignment an instrument and adds the assumptions an instrument carries. The [design ranking guide](../design-ranking.md) covers comparing studies by what they are worth as well as what they identify.

## The same change with nothing to catch it

The checks above had something to test for two reasons. The unmeasured cause also moved `support`, and `tenure` moves engagement without touching cancellation, which gives the graph independencies the data could contradict.

Keep only `price`, `engagement` and `cancellation` and neither is available. The graph over those three columns joins every pair, so it implies no independence at all.

| Check on the three-column table | Result |
| --- | --- |
| January's estimate | -0.79 |
| The same analysis on April | -0.36 |
| Staleness | `reuse`; `implied_independence` listed under `not_evaluated` |
| Graph check | `uninformative`: `no_testable_implication` |

The estimate is as wrong as before and no check contradicts it. What the staleness report can do is say so: its `not_evaluated` entry reads "the plan's graph implies no conditional independence among its columns, so the data cannot contradict it". A `reuse` with that entry beside it is a weaker statement than a `reuse` without. This is the ordinary case, not a contrived one: a table that holds a treatment, an outcome and the confounders someone thought of carries no test of whether they thought of all of them.

## Who moves between the steps

None of these steps triggered the next. The staleness diagnostic recommended identifying again and did not launch discovery. Discovery did not replace the accepted graph. The evidence plan did not run a study. A host application made each call, chose which reading to put the question to, and kept the record the example prints at its end: one line per step, with the month, the step and what came back. Antecedent supplies each computation and its limits; when to run them and what to do with the result stay with the application and the people behind it.

## What this established

Applying January’s analysis to April moved the estimate from −0.79 to −0.36 even though the simulated effect stayed at −0.8. Two conditional independencies failed in April. The support-contact records helped expose the change, but identifying the effect under the hidden-cause explanation took an experiment.

## What this did not establish

It did not show that January's graph was wrong in January. It did not identify the unmeasured cause. And it was able to notice the change only because that cause left traces among the other measured columns. With three columns the same change contradicted nothing, and the most a check could do was say it had nothing to test. That limit is the reason the assumptions are stated beside every estimate, and the reason an experiment is worth its cost when the answer matters.

## Where to go next

For keeping several structures in play with weights, see [graph uncertainty](../graph_uncertainty.md). For the refusal and the description that can accompany it, see [when the effect is not identified](../guides/fallback.md). To value a study against a decision, continue with [from a new experiment to the next study](experiment-to-next-study.md).

## The complete example

<!-- example: examples/python/when_assumptions_change.py -->

[Open the Python script](../../examples/python/when_assumptions_change.py)
