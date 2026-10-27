# One decision from mixed evidence

`antecedent.mixed_decision` combines three separately fitted routes into one expected-utility
decision: a cluster-randomized trial (the margin), a checked two-step temporal history (the period
effect) and a conjugate Gaussian cost model, joined through an affine bridge with declared mapping
uncertainty. The three published means combine in one linear combination, `benefit * margin +
temporal - cost`. Its variance follows the dependence you declare between the routes. The bounds on
the combination are not coverage statements, and nothing here reports a calibration record.

```python
import antecedent as ant
from antecedent import mixed_decision as md

workflow = md.run_mixed_decision(
    md.Dependence.independent(), execution=ant.Execution(seed=7)
)
decision = workflow.decision()
print(decision.mean, decision.variance)
print(decision.evaluation.verdict, decision.evaluation.chosen)   # uniquely_optimal ('act',)
```

`workflow.routes` summarises what each route estimated. The dependence is never defaulted
silently: `Dependence.independent()`, `Dependence.covariance(...)` and `Dependence.correlated(...)`
declare cross terms, and `Dependence.shared_parameter(...)` takes the correlation from the joint
posterior of a parameter the margin and the cost contrast share, under a `Policy`
(`coherent_joint`, `cut` or `feedback`). A covariance that is not symmetric and positive
semidefinite is refused with `MixedDecisionRefusal`.

## Change one input and see the work it costs

```python
print(workflow.utility_only(2.0))          # a new benefit: nothing is refit
print(workflow.prior_change(0.5))          # a new cost prior: only the cost posterior refits
```

Each replay returns a receipt that counts the fits, identifications and score recomputations it
performed, so a reviewer can check that only the affected route was redone.
`workflow.period_change(start, end)` likewise refits only the temporal mechanisms.

## Export, load and replay

```python
current = workflow.decision()                         # after the replays above
exported = workflow.export_decision()                 # bound to current source authority
loaded = workflow.load(exported.artifact, exported.identity)
assert loaded.mean == current.mean

# An independent consumer replays the numbers and grants no authority.
mean, variance = md.replay_decision(
    exported.artifact, exported.identity, execution=ant.Execution(seed=7)
)
assert (mean, variance) == (current.mean, current.variance)
```

Keep `exported.identity` apart from the bytes. A wrong identity or an unsupported version raises
`MixedDecisionRefusal`; corrupt bytes raise `CausalSerializationError`.
`workflow.export_historical_replay()` exports the same inputs for a replay that deliberately
carries no current standing.

## Authority follows the evidence

The decision is authorised by the current standing of its sources, held in an evidence ledger.
Withdrawing the trial retires the margin and the decision, leaves the temporal and cost claims
standing, and makes any earlier export fail to load:

```python
verification = workflow.withdraw_cluster()            # a CompositionVerification
try:
    workflow.decision()
except md.MixedDecisionRefusal as refusal:
    print(refusal.detail)                             # evidence_ledger.result_not_valid
```

See [composition verification](../composition-verification.md) for how the withdrawn and surviving
claims are classified, and [decisions](../decisions.md) for the contract a decision is built from.
