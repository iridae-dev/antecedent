# Replay old numbers, refuse current use

You can reproduce last quarter's decision numbers exactly and still not be allowed to act on them today. This tutorial builds one decision from three kinds of evidence, changes its utility without refitting anything, exports it, and then withdraws one source. The exported numbers still replay. Current use of the decision refuses.

The scope is one supported workflow, `antecedent.mixed_decision`. It shows how current permission follows the evidence in that workflow. It is not a general promise that withdrawing any source anywhere retires everything downstream.

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
decision = workflow.decision()
```

`decision.mean` and `decision.variance` are those of the linear combination. `decision.evaluation` compares expected utility for `act` and `decline` at that mean, and `workflow.routes` summarises what each route estimated. The dependence is never defaulted: `Dependence.independent()` sets every cross term to zero, and covariance, correlation and shared-parameter declarations are the other ways to say how the routes relate.

The variance and the bounds on the combination are not coverage statements, and this workflow reports no calibration record. Treat the variance as the declared combination of the three routes' published variances, not as an interval with measured coverage.

## Change the utility, refit nothing

```python
receipt = workflow.utility_only(2.0)
assert receipt == md.UtilityReplay(0, 0, 0, 0, 0)
```

A new benefit is a change to the decision's utility, not to the science. The receipt counts every temporal build, identification and posterior fit the change performed, so a reviewer can check the claim instead of trusting it. All five counts are zero: the trial, the temporal mechanisms and the cost posterior are the objects they were before, and only the decision is recomputed. `prior_change` (a new cost prior) refits only the cost posterior, and `period_change` refits only the temporal mechanisms; each reports its own receipt.

## Export, then withdraw a source

```python
exported = workflow.export_decision()
replayed = md.replay_decision(exported.artifact, exported.identity, execution=Execution(seed=SEED))
verification = workflow.withdraw_cluster()
```

`export_decision` binds the artifact to the current standing of its sources in an evidence ledger. Keep `exported.identity` apart from the bytes; a consumer checks the bytes against the identity it retained. `replay_decision` recomputes the combined mean and variance from the bytes alone, so it needs no ledger and can grant no authority.

`withdraw_cluster` then records that the trial is no longer a valid source. The returned `CompositionVerification` classifies every claim:

- `cluster_trial`, `margin_bridge` and `decision` are `withdrawn`. The margin and the decision depended on the trial.
- `temporal_drift` and `bayesian_cost` are `still_valid`. Neither depended on the trial, and they keep their estimates.

## Two different questions, two different answers

Ask the same exported decision two questions.

**Can the numbers be reproduced?** Yes. `replay_decision` returns exactly the sealed mean and variance, and the assertion in the example checks it after the withdrawal.

**May the decision be used now?** No. `workflow.decision()`, `export_decision()`, `load(...)` and further utility changes each raise `MixedDecisionRefusal`. Its `reason_code` is registered and its `detail` begins with `evidence_ledger.`, which says the refusal comes from the ledger's standing for a source, not from a numerical fault. The decision's inputs have not become wrong; one of them has stopped being licensed for this use.

The distinction is the point. A replay shows that a computation is reproducible. Current permission says that the evidence behind it still counts. Withdrawing the trial removes the second without touching the first, because the first never depended on the ledger.

<!-- example: examples/python/replay_and_current_decisions.py -->

## Where this stops

The behaviour is specific to this workflow. The refusal follows from how this workflow records its sources and what depends on what; it was tested for withdrawing the trial. Withdrawing the temporal or cost source, or using a different composition, is a different question for you to ask of that composition's own claims; [composition verification](../composition-verification.md) classifies claims for any graph you build.

A refusal also does not say what to do next. The next step is scientific: replace the trial with new evidence (a successor source in the ledger), or decide that the remaining claims support a smaller question. A replay of the old numbers is a record of what was decided, not a route back to permission.

## What this established

For this decision, changing a utility refits no route, an export replays from its bytes without a ledger, and withdrawing a source ends current use while leaving the historical numbers reproducible. The next question is how to restore standing: bring in a new trial through the ledger and decide again. The [mixed decisions guide](../guides/mixed-decisions.md) shows the declared dependences and shared-parameter policies; the [evidence ledger](../evidence-ledger.md) shows how sources and withdrawals are recorded.
