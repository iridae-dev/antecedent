# Noncausal associations: what the observed data say, and nothing more

`CausalQuery::Association` (Rust) and `antecedent.descriptive.Association` (Python) describe how
an outcome varies with predictors in the observed data. The route is compiled like every other:
it lowers through all six layers, seals an operation, executes only from its retained lowered
program, and stamps the program's identity on its result. It answers a different question from a
causal query. Identification is **not claimed**, the result is never the effect of an
intervention, and it ranks no action. Every standard error, spread and posterior it reports is
unlicensed: no coverage record names any of them, and the calibration label is `unmeasured`.

```python
import antecedent as ant
from antecedent.descriptive import Association

result = ant.analyze(table, query=Association("y", "x"))
result.value.association    # AssociationEstimate
result.answer.kind          # "associational"
```

Use it when a causal query was refused and the data still describe something worth stating: pass
`refused_after=` the refusal's reason code and the description carries the reason beside
it; `on_refusal="describe"` on the causal call does this for you (see [the fallback guide](fallback.md)). The description answers none of what the refusal declined to claim.

## The selection rule

The analysis is chosen by one declared, deterministic rule (`association_selection_rule`). The same
query on the same data always picks the same analysis, and the program records it.

1. **Outcome scale.** A declared `outcome_scale=` wins. Otherwise a schema `binary` column is
   binary; a column whose observed complete values are exactly 0 and 1 is binary; a `count` or
   unspecified column of non-negative integers is a count; every other real column is
   continuous. A categorical, ordinal or vector outcome is refused with the next attempt (code one
   level as a 0/1 indicator).
2. **Functional.** `auto` reads the predictors. No predictor is the marginal law of the outcome.
   Predictors and conditioning columns that are all discrete in the schema give the law in each
   stratum. A numeric predictor gives the regression of the outcome scale. An explicit functional
   is honoured.

| `functional=` | Continuous outcome | 0/1 outcome | Count outcome |
| --- | --- | --- | --- |
| `conditional` | mean in each stratum | probability in each stratum | mean in each stratum |
| `coefficients` | linear regression | logistic regression (log-odds) | Poisson regression (log-rate) |
| `pearson_correlation`, `distance_correlation` | one predictor, nothing held fixed | same | same |
| `predictive` | linear model, log score and CRPS | logistic model, Brier score | Poisson model, log score and CRPS |

A predictive model is scored on held-out folds with the proper scores of the predictive
assessment contract, beside the training-mean baseline on the same cases. A table is split by a
seeded random assignment; a series is split into consecutive blocks. The scores are point values.

## Rows or a series

A table's rows are independent units (`ProgramCell` `AssociationStatic`). With
`Association(..., temporal=True)` the rows are the consecutive points of one series
(`AssociationTemporal`). The data modality changes the uncertainty, not the analysis:

| | Table | Series |
| --- | --- | --- |
| Analytic standard error | sandwich (HC1), linearised influence for strata | Newey-West sandwich, Newey-West influence for strata |
| `bootstrap=` | row bootstrap and percentile spread | circular-block bootstrap, block length from the outcome's own persistence |
| Declared design | independent subjects, or none | dependent time series, or none; any other design is refused naming the next attempt |
| Bayesian, Gaussian outcome, no `weight=` | conjugate Normal-Inverse-Gamma marginals | Bayesian bootstrap on contiguous blocks |
| Bayesian, anything else | Bayesian bootstrap (Dirichlet row weights) | Bayesian bootstrap on contiguous blocks |

The independent-error conjugate likelihood does not describe a dependent series, which is why a
series uses the block Bayesian bootstrap whatever the outcome. The conjugate prior is weakly
informative and scaled to the outcome (`prior_scale` in standard deviations of the outcome per
standard deviation of a predictor).

## What it will not do

An association is refused wherever a causal quantity is required: the claim kind is
`associational`, the contract states identification as `not_claimed`, a scientific bridge refuses
it as a source (`association_as_intervention`), and a decision or experiment-planning goal over it
is refused. It is no substitute for an identified effect, and a good predictive score is not
evidence of a causal relation. The assumptions it carries say so beside every number:
`descriptive_only`, `no_confounding_adjustment`, the sampling premise of its modality, and
`uncalibrated_uncertainty`.
