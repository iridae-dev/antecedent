# Describe the data and plan evidence when the effect is not identified

A causal query the declared structure does not identify ends in a typed refusal. The refusal is
the answer to the causal question, and it stays so. Two things are still worth having beside it:
a plain description of what the observed data show, and the evidence that would make the causal
question answerable. Both are opt-in, and neither replaces the refusal.

```python
import numpy as np
import antecedent as ant

rng = np.random.default_rng(11)
z = rng.normal(size=240)
t = (z + rng.normal(size=240) > 0).astype(float)
data = {"z": z, "t": t, "y": 1.0 + 2.0 * t + z + rng.normal(scale=0.3, size=240)}

# z -> t, z -> y, t -> y, and an unmeasured common cause of t and y (t <-> y).
graph = ant.Admg.from_edges(
    ["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")], [("t", "y")]
)
query = ant.AverageEffect("t", "y")

# Default: the refusal is the answer.
refused = ant.analyze(data, graph=graph, query=query)
print(refused.accepted, refused.refusal.code)   # False effect_not_identified

outcome = ant.analyze(data, graph=graph, query=query, on_refusal="describe")
outcome.refusal           # the unchanged refusal (code, remedy, fields)
outcome.associational     # a labelled-noncausal outcome: outcome.associational.answer.kind
outcome.evidence_plan     # obligations, next actions, templates (and a ranked plan)
print(ant.fallback.described(outcome).explain())
```

## What comes back

`analyze(..., on_refusal="describe")` returns the refusal `Outcome` with the two extra parts attached only when the causal
query is refused for identification (`effect_not_identified`, `cross_world_not_identified`). A
query that succeeds returns its ordinary outcome. `antecedent.fallback.described(outcome)` reads the refusal,
association and evidence plan together as an `AnalysisWithFallback` for `explain()`, `report()` and `to_dict()`. A search a budget stopped, a cancellation, a
resource limit, an invalid argument and every data or estimation error are raised exactly as
without the option.

* `refusal` is the original refusal object, unchanged. It is the answer to the causal question.
* `associational` is the compiled associational route
  ([noncausal associations](#describe-the-observed-data-directly)) over the same outcome, treatment and
  adjustment variables. The adjustment variables are chosen by a declared rule: the measured
  parents of the treatment in the declared directed or bidirected graph, plus the query's
  declared effect modifier. The study is built through the same compiler as any other (`Study`
  compile, retained execution, plan view), is labelled noncausal
  (`answer.kind == "associational"`), carries the refusal's reason code in its identification
  layer (`refused_after`), and keeps the inference you chose (Bayesian posterior or analytic and
  resampling standard errors, all unmeasured for coverage). It ranks no action.
* `evidence_plan` is the evidence the failure determines. `obligations` are the verified evidence
  obligations, each with its proof step; `next_actions` are the evidence-gathering actions they
  imply (`measure_variable:<name>`, `run_experiment:<name>`, `find_instrument`,
  `collect_panel`); `templates` are the unfinished studies they determine. An assumption (an
  instrument's validity, the time-invariance a panel relies on) is listed as an obligation and is
  satisfied by no study.

Each of `associational` and `evidence_plan` is either present or a `FallbackUnavailable` with a
registered code, a stable `fallback.*` detail and the next attempt: no association form exists for
the query kind (a counterfactual, a distribution), a variable is not measured, the data are a
panel, the declared graph is not a directed or bidirected one, or the declared graph identifies
the effect (so the refusal came from the estimator and no structural gap is owed).

## Ranking the options that would close the gap

Declare the studies you could run and the plan ranks them. Cost, timing, recruitment and sample
size are yours; nothing is defaulted.

```python
from antecedent import repair

trial = repair.StudyCandidate.experiment(
    "randomized trial", population="study", interventions=["t"], measured=["y"],
    cost=9, sample_size=100, recruitment="consecutive recruitment", timing="baseline",
    unit="patient", cost_unit="usd", sample_budget=100, feasible=True,
)
outcome = ant.analyze(
    data, graph=graph, query=query, on_refusal="describe", evidence_candidates=[trial]
)
plan = outcome.evidence_plan.plan
plan.best.labels, plan.best.gate     # each option is checked by the theorem of its route
```

The plan is hypothetical identification, not an estimate: the ordering is `point_only` and its
calibration `unmeasured`. Plan again over other candidates, or with a declared value and cost
map, with `antecedent.design.evidence_plan(outcome.evidence_plan.contract, ...)`.

## What the description is not

The associational result is **not an answer to the causal question.** It describes the observed
law of the columns, with whatever confounding, selection and reverse direction that carries. It is
not the effect of an intervention, it is not identified, and it ranks no action; a scientific
bridge refuses it wherever a causal quantity is required. `repr`, `explain()` and `report()` of the
described object say so, and the host record (`antecedent.host.fallback_response`) shows
`causal_status: unsupported` beside the status of the noncausal claim.


## Describe the observed data directly

`CausalQuery::Association` (Rust) and `antecedent.descriptive.Association` (Python) describe how
an outcome varies with predictors in the observed data. It answers a different question from a
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
it; `on_refusal="describe"` on the causal call does this for you (see above). The description answers none of what the refusal declined to claim.

### The selection rule

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

### Rows or a series

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

### What it will not do

An association is refused wherever a causal quantity is required: the claim kind is
`associational`, the contract states identification as `not_claimed`, a scientific bridge refuses
it as a source (`association_as_intervention`), and a decision or experiment-planning goal over it
is refused. It is no substitute for an identified effect, and a good predictive score is not
evidence of a causal relation. The assumptions it carries say so beside every number:
`descriptive_only`, `no_confounding_adjustment`, the sampling premise of its modality, and
`uncalibrated_uncertainty`.

## From Rust

```rust
use antecedent::analysis::fallback::{FallbackOutcome, FallbackRequest};

match study.analyze_with_fallback(&FallbackRequest::default(), &ctx)? {
    FallbackOutcome::Causal(result) => { /* the ordinary result */ }
    FallbackOutcome::Described(fallback) => {
        // fallback.refusal, fallback.associational, fallback.evidence
        println!("{}", antecedent::report::render_fallback_report(
            &fallback, antecedent::report::ReportFormat::Markdown));
    }
    _ => {}
}
```

Pass `FallbackRequest { evidence: Some(request) }` to rank declared studies. Any other error
(`CausalError::Cancelled`, a resource limit, a capped search) is returned as the error it is.
