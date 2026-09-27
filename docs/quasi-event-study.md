# Staggered-adoption event study

## Retained two-period panel DiD

`antecedent.prepare(data, query=antecedent.quasi.PanelDifferenceInDifferences(...))`
and `antecedent.analyze(...)` execute a balanced two-period panel DiD through
the retained native `Study` route. The query names the outcome, subject, stable
treatment, pre/post indicator, and optionally a higher-level cluster column.
Each subject must have exactly one pre and post outcome; rows, design values,
and identifiers must align. The result's `panel_did` section reports the
difference in mean subject-level changes and a cluster score-sandwich standard
error using the finite-cluster multiplier `G/(G - 1)`. At least two clusters
must contribute to each treatment group. It makes no p-value or interval claim
and remains unlicensed in the support matrix. The identifying assumptions are
parallel untreated trends, no anticipation, consistency, stable assignment,
and no interference. Staggered adoption, repeated cross-sections, missing
waves, and other panel shapes are refused by this route.

The same standard-error contract is used by `estimate_panel_did` for balanced
two-period panels. It defaults to subject clusters and accepts an optional
higher-level cluster column. `estimate_group_time_att` uses the same contract
for post-adoption cohort-period contrasts.

`antecedent.prepare(data, query=StaggeredAdoption(..., event_study=True))` and
`antecedent.analyze(...)` retain all cohort-specific event-time contrasts in
one native `Study`. The result's `panel_did.effects` holds the full curve,
including descriptive pre-adoption contrasts. The direct
`estimate_staggered_event_study` utility uses the same native estimator.
Both use the immediately
pre-adoption period (`g - 1`) as the reference and cohort 0 as the never-treated
comparison group. The API reports cohort and calendar period, event time,
effect, and treated/control subject counts for every comparison.

```python
import antecedent
from antecedent.quasi import StaggeredAdoption

query = StaggeredAdoption("outcome", "subject", "period", "cohort", event_study=True)
result = antecedent.analyze(data, query=query)
effects = result.panel_did.effects
```

The retained result and artifact bind the subject, cohort, period, and cluster
vectors. A prepared analysis requires the same design rows on refresh. The
top-level scalar is the first post-adoption contrast in cohort/period order;
the full event study is `result.panel_did.effects`.

This is an unlicensed point workflow with `unlicensed_point_utility` support status.
It assumes cohort-specific parallel untreated trends, no anticipation,
absorbing treatment after adoption, valid never-treated controls, no
interference, a balanced panel, and independent sampling clusters. The API
cannot establish these assumptions. It reports pointwise cluster-robust
standard errors from cluster-aggregated influence contributions, with the
finite-cluster multiplier `G/(G - 1)`. Subject IDs are the default clusters;
an optional higher-level cluster column can be supplied. At least two distinct
clusters must contribute to each cohort and control group for every row.
Pre-adoption contrasts are descriptive diagnostics only; they are not a
parallel-trends test. Neither route provides p-values, confidence intervals,
or calibrated inference. Cluster IDs must be non-empty and constant within
subject.

The input uses the same columns as `StaggeredAdoption`: outcome, subject ID,
calendar period, and first-treatment cohort, with cohort 0 reserved for units
never treated during the observed panel. Every unit must have one observation
at every common period, and each treated cohort must have its `g - 1` baseline
and adoption period observed. Designs without never-treated controls or with
unsupported timing are refused.
