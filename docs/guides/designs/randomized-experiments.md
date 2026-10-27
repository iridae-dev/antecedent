# Analyze a randomized or factorial experiment

Estimate the effect of an assignment you randomized, with uncertainty taken from the randomization itself. Use this guide when you know how units were assigned (a coin flip per account, a fixed number of treated clusters, a 2x2 grid of two messages) and want the intention-to-treat effect, the main effects and interaction of two factors, or contrasts among several actions.

The recorded assignment law is what identifies the effect. Antecedent reads that law from an `ExperimentDesign` and does not infer it from the data.

## Estimate an intention-to-treat effect

The data below are simulated: assignment lifts revenue by exactly 2.0, so the estimate has a known target. Row-aligned design inputs can be named by the data column that holds them; `prepare` resolves the names and freezes the design.

```python
import antecedent
import numpy as np
from antecedent import analyze
from antecedent.experiment import ExperimentDesign, RandomizedEffect
from antecedent.interference import BernoulliAssignment

rng = np.random.default_rng(0)
n = 400
assigned = rng.random(n) < 0.5
revenue = 2.0 * assigned.astype(float) + rng.normal(size=n)
account_id = np.array([f"account-{i}" for i in range(n)])

design = ExperimentDesign(
    BernoulliAssignment(0.5),
    realized_assignment="assigned",
    assignment_units="account_id",
    outcome_units="account_id",
)
query = RandomizedEffect("revenue", design)
data = {"revenue": revenue, "assigned": assigned, "account_id": account_id}

result = analyze(data, query=query)
fit = result.value.randomized_effect
fit.effect, fit.interval_95, fit.uncertainty, fit.support_status
```

The runnable version is `examples/python/experiment_randomized_effect.py`. The same study then accepts outcome-only data without changing the design:

```python
fresh = 2.0 * assigned.astype(float) + rng.normal(size=n)
updated = antecedent.estimate(antecedent.rebind(result.study, data={"revenue": fresh}))
```

## Read the result

- `fit.effect` is the intention-to-treat contrast: the effect of being assigned, whether or not the unit complied.
- `fit.uncertainty` names the variance construction and `fit.interval_95` is present only when the design meets that construction's support rule. When it does not, you get the point estimate and a labelled variance, and the diagnostic names the rule that was not met.
- `fit.support_status` says whether the interval is among the executing support evidence listed in the [graphless support matrix](../../graphless-support-matrix.md). A result can execute and be useful while remaining outside that matrix; the status stays attached.
- `result.value.calibration` reports whether a matching coverage measurement applies to the interval; see [Uncertainty](../../concepts/uncertainty.md).

## Choose the assignment design

The design you declare chooses the variance construction. Name the one that matches how assignment actually happened.

| Assignment | Declare | Contrast and variance |
| --- | --- | --- |
| Independent coin flip per unit | `BernoulliAssignment(p)` (one probability, or one per unit) | Horvitz-Thompson contrast with the assignment-design variance |
| Fixed number treated | `CompleteRandomization(treated)` | Difference in means with Neyman's conservative variance |
| Fixed number treated within blocks | `StratifiedRandomization({block: treated})`, plus `blocks=` on the design | Block-weighted differences with blockwise variance |
| Whole clusters assigned | `ClusterRandomization(clusters, treated_clusters)` | Cluster totals with a conservative cluster-level variance; at least two clusters per arm |
| Two binary factors in four fixed cells | `FactorialRandomization(second_factor_assignment, cell_counts)` | Both main effects and the interaction, with cellwise Neyman variances; at least two units per cell |
| Three or more named actions | `MultiArmExperimentDesign` | Horvitz-Thompson arm means and every contrast against the first action |
| Sequences of periods | `SwitchbackEffect` | Sequence-clustered sandwich variance; assumes no carryover and no interference between sequences |

Interval thresholds differ by design (for example, about 30 units per arm for a complete randomization and about 30 independent clusters per arm for a cluster design). Intervals are pointwise, never simultaneous across several contrasts.

### A 2x2 factorial experiment

For two binary factors randomized in four fixed cells, `FactorialRandomization` runs through the same `RandomizedEffect` query. It reports the primary main effect as `effect`, the second factor's main effect as `second_factor_effect`, and the interaction as `factorial_interaction`.

```python
primary = [False, False, True, True, False, False, True, True]
second = [False] * 4 + [True] * 4
factorial = antecedent.experiment.FactorialRandomization(
    second, (2, 2, 2, 2), ("no_message", "message")
)
```

`design` is then an `ExperimentDesign` built on `factorial` as its assignment, as in the example above. With very small cells the result is point-only.

If both factors were assigned by independent Bernoulli draws, `antecedent.factorial.estimate` is a direct utility for the cell means, both main effects and the interaction. The fixture in `examples/python/factorial_cell_means.py` is balanced with known cell means, so its main effects (4 and 5) and interaction (4) are exact:

```python
from antecedent import factorial

result = factorial.estimate(
    {"y": outcome},
    factor_a=factor_a,
    factor_b=factor_b,
    design=factorial.FactorialDesign(0.5, 0.5),
    outcome="y",
)
result.cell_means, result.factor_a_effect, result.factor_b_effect, result.interaction_effect
```

This direct utility reports variance upper bounds and no interval.

## Variants

- **Precision adjustment.** `RandomizedEffect(..., cuped=FixedCUPED(...))` and `ancova_covariates=(...)` use pre-assignment covariates to reduce variance under independent Bernoulli assignment. The covariates must be measured before assignment; they cannot be post-treatment variables. ANCOVA reports an independent-row HC0 variance. Multi-arm designs do not combine with either.
- **Exact randomization test.** `exact_randomization_test=True` (or `experiment.exact_randomization_test` directly, up to 20 units) tests the sharp null of no effect for any unit by enumerating assignments. It is a test, not a confidence interval.
- **Sampling weights.** `RandomizedEffect(..., weights="w")` targets the weighted population for an unadjusted Bernoulli or complete design. See [sampling weights](../sampling-weights.md).
- **Noncompliance.** `experiment.ComplierEffect` takes observed treatment receipt under Bernoulli encouragement and reports the intention-to-treat effect, the first stage, and the Wald complier effect under exclusion and monotonicity, which you declare. `experiment.TreatmentOnTreated` is the recipient-target version under one-sided noncompliance. Both report an influence-function variance and no interval.

## Boundaries

- A unit assigned by something other than the declared law (an analyst's choice, a rule that read the outcome) is not a randomized experiment, and the declared design would misdescribe it. Estimate it as an observational effect instead; [First analysis](../../start/first-analysis.md) shows the flow.
- A zero or negative first stage refuses the complier effect: with no effect of assignment on receipt there is no complier effect to scale.
- Factorial designs beyond two binary factors in four fixed cells (block or multi-arm factorials, three or more factors) are not supported; use separate contrasts instead.
- Outcome-adaptive assignment, interim looks and stopping rules are not fixed designs: an analysis that presumes a fixed sample is refused for such data. See [Inference from adaptive experiments](../../adaptive-inference.md).
- Contrasts at several time points under a randomized sequence are [temporal designs](../temporal-designs.md).

## Related

- [Plan power, sample size and minimum detectable effect](../power-and-sample-size.md) for the design you are about to run.
- [Interference and network exposure](interference-and-network-exposure.md) when units affect each other.
- [Reading a result](../../start/reading-a-result.md) for the answer, assumptions and uncertainty fields.
