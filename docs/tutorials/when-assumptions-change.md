# When the causal assumptions stop holding

Every estimate Antecedent returns is conditional on a causal structure someone declared or accepted. This tutorial follows what happens when the world stops matching that structure. One claim is taken from the graph that supported it, through new data that no longer fits, to a set of competing explanations, a refusal, and the study that would make the question answerable again.

Nothing in the first analysis is at fault. Its arithmetic stays correct and reproducible throughout. What changes is whether the number may still be read as a causal effect.

## The task

A subscription business wants to know how much weekly engagement changes cancellations. It has five measured columns per customer, all standardized scores: `price`, `tenure`, `engagement`, `support` (contacts with the support team) and `cancellation`.

The data are simulated, so the truth is known. One unit of engagement lowers the cancellation score by 0.8, in January and in April alike. In April an unmeasured cause is switched on: customers who depend on an integration that has started failing. They use the product more, contact support more and cancel more. That variable is never in the table.

| Declared by a person | Learned from data |
| --- | --- |
| The direction of the one edge discovery left open in January | January's graph, apart from that edge |
| Three readings of the new link that appears in April | That April's table contradicts two independencies the January graph asserts |
| Two candidate studies, with their costs and sample sizes | Which candidate study would identify the effect |

The simulated truth is a property of the stated mechanism. It is there so each step can be checked, not as a claim about any real business.

## January: discover, review, estimate

PC discovery returns five directed edges and leaves one, between `price` and `support`, undirected. The data do not fix it, so a reviewer supplies it, and the accepted graph records a new version.

```python
discovered = disc.discover(january, method=disc.PC(alpha=0.01), accept_discovered=True, execution=EXEC)
discovered.pending          # [price -- support]
accepted = discovered.review({("price", "support"): ("tail", "arrow")})

study = ant.prepare(january, graph=accepted, query=ant.AverageEffect("engagement", "cancellation"), ...)
first = require_success(ant.estimate(study))    # -0.79
```

The estimate is -0.79 against a simulated -0.8. The graph says price is the only common cause of engagement and cancellation, and that holds in January.

## April: the same analysis, a different number

```python
rerun = require_success(ant.estimate(ant.rebind(study, data=april)))    # -0.36
```

Rebinding repeats January's analysis on April's rows: the same graph, the same adjustment set. It returns -0.36. More than half the effect appears to have gone.

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

The numbers are partial correlations. A statement exceeds only when it is significant after adjustment for every test in the assessment and its correlation reaches 0.1, so a second January sample comes back `reuse` with nothing found.

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

An edge that appears in 2.5% of January resamples and 99% of April ones is not something sampling variation explains. The frequencies are a bootstrap distribution of the search, not a probability that the edge exists.

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

Two things follow. The second reading passes its check and is still wrong: -0.61 is not -0.8, because the third reading is the true one here. And the table cannot separate the second reading from the third, since the graph check assumes no hidden causes and so cannot be put to the third at all. Choosing between them takes knowledge from outside the table, or new evidence.

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

The unmeasured cause appears as `common_cause_of_engagement_cancellation`. Antecedent can establish that a variable in that position is needed. It cannot say that the variable is a failing integration. Naming it is the investigator's work.

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

<!-- example: examples/python/when_assumptions_change.py -->

## Who moves between the steps

None of these steps triggered the next. The staleness diagnostic recommended identifying again and did not launch discovery. Discovery did not replace the accepted graph. The evidence plan did not run a study. A host application made each call, chose which reading to put the question to, and kept the record the example prints at its end. Antecedent supplies each computation and its limits; when to run them and what to do with the result stay with the application and the people behind it.

## What this established

A January estimate, reproduced exactly on April's data, was no longer justified as a causal effect. April covered the same ranges as January; two independencies the graph implies failed, and the staleness diagnostic asked to identify again; a resampled search showed the failure was not noise; and a search allowing hidden causes withdrew the direction January relied on. Of three readings of the new data, one was discarded, one gave an estimate that was wrong, and one refused. The refusal named an experiment as sufficient and more observational rows as not, and in the simulation that is how it turned out.

It did not show that January's graph was wrong in January. It did not identify the unmeasured cause. And it was able to notice the change only because that cause left traces among the other measured columns. With three columns the same change contradicted nothing, and the most a check could do was say it had nothing to test. That limit is the reason the assumptions are stated beside every estimate, and the reason an experiment is worth its cost when the answer matters.

For keeping several structures in play with weights, see [graph uncertainty](../graph_uncertainty.md). For the refusal and the description that can accompany it, see [when the effect is not identified](../guides/fallback.md). To value a study against a decision, continue with [from a new experiment to the next study](experiment-to-next-study.md).
