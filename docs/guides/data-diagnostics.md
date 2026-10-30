# Inspect data quality before estimation

Preparation automatically collects a bounded, fit-free `DataDiagnostics` report.
The report describes suspicious measurements under the prepared question's roles.
It never drops rows, changes an adjustment set, chooses an estimator, or changes
identification or scientific standing. Review findings need human interpretation;
there is no p-value or automatic rejection of a scientific model.

```python
import numpy as np
import antecedent as ant

rows = {"t": np.arange(200, dtype=float) % 2,
        "y": np.arange(200, dtype=float) % 10}
study = ant.prepare(rows, graph=[("t", "y")],
                    query=ant.AverageEffect("t", "y"),
                    estimator="linear.adjustment.ate", refute=False)
report = study.data_diagnostics
for finding in report.findings:
    print(finding.code, finding.observation, finding.criterion, finding.next_action)
```

`report.columns` contains typed column summaries; `report.checks` records evaluated,
inapplicable, disabled and unavailable checks. Missing values and infinities count
as non-finite. Undefined moments stay absent. The `blocking` severity is reserved
for adapted existing preflight execution requirements, such as a rank-deficient
regression design. Collecting that finding never itself refuses preparation.

Prepared inspection and `repr(study)` show bounded finding counts. Python emits
no warnings. Reports and their nested collections are immutable Python values.

## Choose checks and thresholds

```python
suite = ant.DataDiagnostics(skewness=3.0, duplicate_rows=False)
print(suite.checks)  # Native registry: stable codes, policy families and applicability.
study = ant.prepare(rows, graph=[("t", "y")],
                    query=ant.AverageEffect("t", "y"),
                    estimator="linear.adjustment.ate", refute=False,
                    data_diagnostics=suite)

# Disable automatic collection, then evaluate explicitly when needed.
quiet = study.reconfigure(data_diagnostics=False)
assert quiet.data_diagnostics is None
from antecedent.outcomes import require_success
explicit = require_success(quiet.diagnose_data(suite=suite))

# Standalone diagnostics work even when causal preparation would refuse.
standalone = ant.diagnose_data(rows, roles={"t": "treatment", "y": "outcome"},
                              contrasts={"t": (0.0, 1.0)})
```

Standalone calls return a report. Prepared calls return an `Outcome`, following
`diagnose()` and `diagnose_fit()`. `suite=` accepts a `DataDiagnostics` declaration.
`prepare_many(..., data_diagnostics=...)` accepts the same setting and exposes a
tuple of retained member reports; shared table profiles are computed once per
compatible policy during batch preparation. Explicit batch diagnostics return an
`Outcome` containing a tuple of reports.

| Check | Default review rule |
| --- | --- |
| Missingness | At least 5% missing/non-finite cells, with at least one missing cell |
| Constant | One distinct observed measurement |
| Near constant | Modal fraction at least 99% |
| Repeated values | Modal fraction at least 50%, unless already near constant |
| Shape | Absolute population skewness above 2; population excess kurtosis above 10 |
| Low resolution | At least 1,000 observed values, at most 20 distinct values, and distinct fraction at most 1% |
| Duplicate rows | At least one repeated row over the inspected columns, excluding its first copy |
| Treatment imbalance | Smallest declared contrast-arm fraction below 5% |

Concentration and shape checks require at least 100 observed sample values.
Constants can be described on smaller samples. Shape and resolution warnings are
scoped to suitable scalar numeric value types; binary and categorical values are
not interpreted as continuous measurements. Unspecified numeric resolution is
informational; `data.unexpected_discretization` requires a declared continuous
variable. These thresholds are descriptive defaults, not calibrated scientific
acceptance criteria. Population moments use divisor `n`; no bias correction or
normality test is implied.

## Cost and coverage

Basic counts scan all rows. Shape, sample distinctness, missingness patterns and
duplicate rows inspect at most 4,096 evenly spaced row indices by default. This
selection is deterministic across seeds and threads and makes no random-sampling
or representativeness guarantee. Every relevant check and finding records whether
it used a sample. Sample duplicate counts and cardinality are never extrapolated
to the full table. Duplicate equality normalizes signed zero and uses one missing
token for missing or non-finite cells, preserving typed integer and categorical
equality. It covers all retained inspected columns; it
does not establish that two records represent the same real-world observation.

Existing preflight relationships run automatically only on complete designs with
at most 4,096 rows and 64 adjustment-design columns. A wider or taller design is
explicitly not evaluated; a sample rank is never passed off as full-table rank.
Use `DataDiagnostics(max_sample_rows=None)` in an explicit call to request
full-table profiling and remove these bounded preflight limits. Memory admission
and cancellation still apply. Explicit calls raise operational errors when the
requested work cannot be admitted. Automatic diagnostic resource admission
failures produce unavailable coverage instead of refusing the causal analysis.

Temporal/event inputs are profiled as retained table values, with no assertion
about timestamp regularity, independent observations, or temporal model validity.
Panel and multi-environment preparations retain an explicit not-evaluated report;
individual tables can be inspected with the standalone API.

## Evidence and lifecycle

`report.to_json()` exports the native report, including its version, content
snapshot identity, diagnostic context, policy, measurements and coverage. Its
canonical identity binds these values. No wall-clock timestamp or execution seed
enters the report identity. Diagnostic settings do not enter scientific program
or claim identities.

Rebinding data and refreshing a prepared study rebuild diagnostic evidence for the
new retained snapshot. Reports already read from the original study remain
unchanged. A failed refresh publishes neither new data nor a replacement report.
The standalone Rust API is `antecedent::diagnose_data`; prepared Rust handles expose
`data_diagnostics()` and `diagnose_data(&policy, &ctx)`. Configure collection with
`StudyBuilder::data_diagnostics(Some(policy))` or disable it with `None`; batch
builders expose the same setting.

[Preflight diagnostics](preflight-diagnostics.md) retain their existing execution
requirements and relationship checks. [Staleness](staleness.md) compares snapshots
under its own policy. Fitted diagnostics and scientific refuters remain explicit;
this suite fits no model and runs neither of them.
