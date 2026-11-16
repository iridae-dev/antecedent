# Replay old numbers, refuse current use

You can reproduce last quarter's decision numbers exactly and still not be allowed to act on them today. This tutorial builds one decision from three kinds of evidence, changes its utility without refitting anything, exports it, and then withdraws one source. The exported numbers still replay. Current use of the decision refuses.

The scope is one supported workflow, `antecedent.mixed_decision`. It shows how current permission follows the evidence in that workflow. It is not a general promise that withdrawing any source anywhere retires everything downstream.

- **You will learn** how a decision is combined from separately fitted routes under a declared dependence, how a receipt shows what a change recomputed, and why reproducing an exported decision and being permitted to use it now are separate questions with separate answers.
- **Before you start:** [installation](../start/installation.md). [Stop, continue or run longer](decide-under-uncertainty.md) introduces decisions; the [evidence ledger](../evidence-ledger.md) explains sources and their standing.
- **Time:** the example runs in under a minute.
- **Run it:** `python examples/python/replay_and_current_decisions.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/replay_and_current_decisions.ipynb). The complete script is embedded [below](#the-complete-example).

## The task

A programme may be run (`act`) or not (`decline`). The decision compares the two at a single combined number: `benefit × margin + temporal effect − cost`.

| Piece | Evidence | What it contributes |
| --- | --- | --- |
| Margin | A cluster-randomized trial | The effect on the outcome that the programme is meant to move |
| Temporal effect | A checked two-step history over a period | A period-specific change that adds to the margin |
| Cost | A conjugate Gaussian model fitted to cost observations | What the programme costs per unit |

The routes are fitted by Antecedent from a simulated fixture. The fixture's numbers are truths of a stated model, useful because every piece can be checked, and make no empirical claim. Two things are declared: how the three routes depend on one another (here, independently) and the benefit weight on the margin (here, one). Everything else is learned.

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

A new benefit weight of 2 is a change to the decision's utility, not to the science. The receipt (`md.UtilityReplay`) counts the temporal factor builds and temporal identifications, and the cost model's fits, posterior draws and identifications, that the change performed, so a reviewer can check the claim instead of trusting it. All five counts are zero: the trial, the temporal mechanisms and the cost posterior are the objects they were before, and only the decision is recomputed. The example confirms it from the other side: the margin mean and each route's estimate are equal to their earlier values, and the new combined mean is `2.0 * margin + temporal effect - cost`. `prior_change` (a new cost prior) refits only the cost posterior, and `period_change` refits only the temporal mechanisms; each reports its own receipt.

## Export, then withdraw a source

```python
exported = workflow.export_decision()
loaded = workflow.load(exported.artifact, exported.identity)
replayed = md.replay_decision(exported.artifact, exported.identity, execution=Execution(seed=SEED))
verification = workflow.withdraw_cluster()
```

`export_decision` binds the artifact to the current standing of its sources in an evidence ledger. Keep `exported.identity` apart from the bytes; a consumer checks the bytes against the identity it retained. `workflow.load` consumes the artifact against the workflow's current source standing; before the withdrawal it returns the mean and variance that were exported. `replay_decision` recomputes the combined mean and variance from the bytes alone, so it needs no ledger and can grant no authority.

`withdraw_cluster` then records that the trial is no longer a valid source. The returned `CompositionVerification` classifies every claim:

- `cluster_trial`, `margin_bridge` and `decision` are `withdrawn`. The margin and the decision depended on the trial.
- `temporal_drift` and `bayesian_cost` are `still_valid`. Neither depended on the trial, and they keep their estimates.

Read a claim's class with `verification.classify(key).klass`. For the decision, `verification.classify("decision").authority_retained` is `False`.

## Two different questions, two different answers

Ask the same exported decision two questions.

**Can the numbers be reproduced?** Yes. `replay_decision` returns exactly the sealed mean and variance, and the assertion in the example checks it after the withdrawal.

**May the decision be used now?** No. `workflow.decision()`, `workflow.export_decision()`, `workflow.load(...)` and further utility changes each raise `MixedDecisionRefusal`. The example prints one line per refused call, of the form `refused: <reason_code> evidence_ledger.<detail>`. The example asserts that every `detail` begins with `evidence_ledger.`, which says the refusal comes from the ledger's standing for a source, not from a numerical fault. The decision's inputs have not become wrong; one of them no longer has standing as current evidence for this use.

The distinction is the point. A replay shows that a computation is reproducible. Current permission says that the evidence behind it still counts. Withdrawing the trial removes the second without touching the first, because the first never depended on the ledger.

## Where this stops

The behaviour is specific to this workflow. The refusal follows from how this workflow records its sources and what depends on what; it was tested for withdrawing the trial. Withdrawing the temporal or cost source, or using a different composition, is a different question for you to ask of that composition's own claims; [composition verification](../composition-verification.md) classifies claims for any graph you build.

A refusal also does not say what to do next. The next step is scientific: replace the trial with new evidence (a successor source in the ledger), or decide that the remaining claims support a smaller question. A replay of the old numbers is a record of what was decided, not a route back to permission.

## What this established

For this decision, changing a utility refits no route, an export replays from its bytes without a ledger, and withdrawing a source ends current use while leaving the historical numbers reproducible.

## What this did not establish

It did not show how withdrawal propagates in other compositions, or which smaller question the remaining claims can still answer. The decision's variance is a declared combination of route variances with no calibration record, and every number is a fixture number of a simulated model.

## Where to go next

The next question is how to restore standing: bring in a new trial through the ledger and decide again. The [mixed decisions guide](../guides/mixed-decisions.md) shows the declared dependences and shared-parameter policies; the [evidence ledger](../evidence-ledger.md) shows how sources and withdrawals are recorded.

## The complete example

<!-- example: examples/python/replay_and_current_decisions.py -->
