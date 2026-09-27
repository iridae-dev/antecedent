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
must contribute to each treatment group, and clusters must be nested within
treatment groups. A separate pointwise 95% normal interval is available when
each group has at least 30 independent clusters and the standard error is
positive. This is interval evidence outside the current graphless support
matrix. The identifying assumptions are parallel untreated trends, no
anticipation, consistency, stable assignment,
and no interference. Staggered adoption and missing waves are refused by this
query; repeated cross sections use its explicit sampling mode.

The retained route serves balanced panels and repeated cross sections through
`result.panel_did`. It defaults to subject clusters and accepts an optional
higher-level cluster column. `estimate_group_time_att` uses the same cluster
score contract for post-adoption cohort-period contrasts.

`antecedent.prepare(data, query=StaggeredAdoption(..., event_study=True))` and
`antecedent.analyze(...)` retain all cohort-specific event-time contrasts in
one native `Study`. The result's `panel_did.effects` holds the full curve,
including descriptive pre-adoption contrasts.
It uses the immediately
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

Supported post-adoption contrasts have separate pointwise 95% cluster intervals
when each side has at least 24 independent clusters and a positive standard
error. These are not a simultaneous band and remain outside the graphless
support matrix. The design assumes cohort-specific parallel untreated trends,
no anticipation,
absorbing treatment after adoption, valid never-treated controls, no
interference, a balanced panel, and independent sampling clusters. The API
cannot establish these assumptions. It reports pointwise cluster-robust
standard errors from cluster-aggregated influence contributions, with the
finite-cluster multiplier `G/(G - 1)`. Subject IDs are the default clusters;
an optional higher-level cluster column can be supplied. At least two distinct
clusters must contribute to each cohort and control group for every row.

The retained result reports a descriptive joint pre-period statistic when at
least two non-reference leads have finite, positive cluster standard errors:
the maximum across leads of `abs(effect / cluster SE)`. It reports an explicit
unavailable diagnostic otherwise. This statistic has no calibrated p-value or
cutoff. A small value cannot establish parallel untreated trends, while a
large value can direct scrutiny to pre-treatment differences. Cluster IDs must
be non-empty and constant within subject.

The input uses the same columns as `StaggeredAdoption`: outcome, subject ID,
calendar period, and first-treatment cohort, with cohort 0 reserved for units
never treated during the observed panel. Every unit must have one observation
at every common period, and each treated cohort must have its `g - 1` baseline
and adoption period observed. Designs without never-treated controls or with
unsupported timing are refused.
