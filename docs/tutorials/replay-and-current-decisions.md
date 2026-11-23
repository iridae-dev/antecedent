# Replay old numbers, refuse current use

Last quarter your team decided to run a programme. Now the trial behind that decision has been withdrawn. An auditor needs to reproduce the original calculation, and your service needs to decide whether it can still recommend the programme.

You’ll use `antecedent.mixed_decision` to answer both questions. It combines a trial, a change over time and a cost estimate, and records which conclusions depend on each source.

Run `python examples/python/replay_and_current_decisions.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/replay_and_current_decisions.ipynb). See the [complete example](#the-complete-example).

## The task

Run the programme (`act`) if its expected benefit covers its cost. The example values the programme as `benefit × margin + temporal effect − cost` and compares that with declining (`decline`).

| Piece | Evidence | What it contributes |
| --- | --- | --- |
| Margin | A cluster-randomized trial | The effect on the outcome that the programme is meant to move |
| Temporal effect | A checked two-step history over a period | A period-specific change that adds to the margin |
| Cost | A conjugate Gaussian model fitted to cost observations | What the programme costs per unit |

Antecedent fits the three quantities from simulated data. You give the margin a weight of one and assume the estimates from the three sources are independent.

## Build the decision and read it

```python
workflow = md.run_mixed_decision(md.Dependence.independent(), execution=Execution(seed=SEED))
routes = workflow.routes
decision = workflow.decision()
```

`md` is `antecedent.mixed_decision`. `decision.mean` and `decision.variance` are those of the linear combination. `decision.evaluation` compares expected utility for `act` and `decline` at that mean, and `workflow.routes` summarises what each route estimated. The dependence is never defaulted: `Dependence.independent()` sets every cross term to zero, and `Dependence.covariance`, `Dependence.correlated` and `Dependence.shared_parameter` are the other ways to say how the routes relate.

The example prints the verdict, the chosen action, the combined mean and its variance, and asserts how they fit together:

- the combined mean equals `decision.margin_mean + routes.temporal_effect - routes.cost_mean` to within 1e-9, the combination with a benefit weight of one;
- the verdict is `uniquely_optimal`;
- the chosen action is `act` when the combined mean is positive and `decline` otherwise.

The variance and the bounds on the combination are not coverage statements, and this workflow reports no calibration record. Treat the variance as the declared combination of the three routes' published variances, not as an interval with measured coverage.

## Change the utility, refit nothing

```python
receipt = workflow.utility_only(2.0)
assert receipt == md.UtilityReplay(0, 0, 0, 0, 0)
```

Double the weight on the margin. The new combined mean is `2.0 * margin + temporal effect - cost`. Check `md.UtilityReplay` to see what ran again. All five fit and identification counters are zero, and the estimates of margin, temporal effect and cost match their earlier values. Changing the cost prior with `prior_change` would refit the cost posterior. Changing the period with `period_change` would refit the temporal mechanisms.

## Export, then withdraw a source

```python
exported = workflow.export_decision()
loaded = workflow.load(exported.artifact, exported.identity)
replayed = md.replay_decision(exported.artifact, exported.identity, execution=Execution(seed=SEED))
verification = workflow.withdraw_cluster()
```

Save `exported.identity` separately from the artifact bytes. A consumer can then verify it received the expected artifact. `workflow.load` also checks the current standing of the sources in the evidence ledger. `replay_decision` needs only the bytes to reproduce the historical mean and variance.

`withdraw_cluster` then records that the trial is no longer a valid source. The returned `CompositionVerification` classifies every claim:

- `cluster_trial`, `margin_bridge` and `decision` are `withdrawn`. The margin and the decision depended on the trial.
- `temporal_drift` and `bayesian_cost` are `still_valid`. Neither depended on the trial, and they keep their estimates.

Read a claim's class with `verification.classify(key).klass`. For the decision, `verification.classify("decision").authority_retained` is `False`.

## Two different questions, two different answers

Ask the same exported decision two questions.

**Can the numbers be reproduced?** Yes. `replay_decision` returns exactly the sealed mean and variance, and the assertion in the example checks it after the withdrawal.

**May the service use the decision today?** After withdrawal, `workflow.decision()`, `workflow.export_decision()`, `workflow.load(...)` and changes to utility all raise `MixedDecisionRefusal`. Each refusal includes a detail beginning with `evidence_ledger.` so the caller can trace the problem to the withdrawn evidence.

The auditor can therefore reproduce the figure in last quarter’s report. A fresh recommendation would need a valid source for the margin.

## Where this stops

The behaviour is specific to this workflow. The refusal follows from how this workflow records its sources and what depends on what; it was tested for withdrawing the trial. Withdrawing the temporal or cost source, or using a different composition, is a different question for you to ask of that composition's own claims; [composition verification](../composition-verification.md) classifies claims for any graph you build.

To revisit the original decision, record a successor trial in the ledger. If you only need a result about temporal drift or cost, check its standing separately.

## Replay a fitted decision

A compiled decision can also contain the fit that produced its outcomes. In the [fitted feedback example](feedback-three-questions.md#compare-actions-after-fitting-the-feedback), the program binds the training rows, the actions, and the decision contract. Its consumer rebuilds the fit and repeats the comparison using the retained identity.

Before handing off a result, keep the program identity separately from the bytes. If the trial rows or action assumptions change, prepare a fresh program for that question. The old artifact still describes the old computation. The evidence-standing check in this tutorial answers whether that computation can be used today.

## What this established

For this decision, changing a utility refits no route, an export replays from its bytes without a ledger, and withdrawing a source ends current use while leaving the historical numbers reproducible.

## What this did not establish

This example exercises withdrawal of the trial in this particular workflow. Other combinations have their own dependencies. The reported variance also relies on the declared independence of the three estimates and has no measured coverage record.

## Where to go next

The next question is how to restore standing: bring in a new trial through the ledger and decide again. The [mixed decisions guide](../guides/mixed-decisions.md) shows the declared dependences and shared-parameter policies; the [evidence ledger](../evidence-ledger.md) shows how sources and withdrawals are recorded.

## The complete example

<!-- example: examples/python/replay_and_current_decisions.py -->

[Open the Python script](../../examples/python/replay_and_current_decisions.py)
