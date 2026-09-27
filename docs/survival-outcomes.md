# Survival outcomes: current 2.1.0 slice

The direct survival API is an explicitly limited point-estimation utility for
two-arm, individually randomized studies with right-censored follow-up. It
computes arm-specific Kaplan–Meier step curves and restricted mean survival
time (RMST) through a user-specified horizon. A declared treatment assignment
is required before the result is described as causal.

```python
import antecedent

query = antecedent.survival.SurvivalOutcome(
    duration="follow_up_days",
    event_observed="event",
    treatment="treated",
    tau=180,
    randomized=True,
)
summary = antecedent.survival.estimate_survival(data, query)
summary.rmst_difference
summary.times, summary.control_survival, summary.treated_survival
```

The same unadjusted survival and competing-risk queries can use the retained
`prepare` / `analyze` flow when the caller explicitly supplies the marginal
observation assumption. The result exposes `result.survival`, a structured
answer and a portable study artifact:

```python
from antecedent.observation import IndependentGiven

query = antecedent.survival.SurvivalOutcome(
    duration="follow_up_days",
    event_observed="event",
    treatment="treated",
    tau=180,
    randomized=True,
    observation_assumption=IndependentGiven(()),
)
result = antecedent.analyze(data, query=query)
result.survival.rmst_difference
```

The unweighted retained route requires `randomized=True` and
`IndependentGiven(())` for marginally independent censoring, including when
there is no delayed entry. It publishes no scalar ATE.

Rows with `event_observed=False` are treated as right-censored: they remain in
the risk set through their recorded duration and do not count as events. The
implementation requires both arms to have observed follow-up at least through
`tau`. It assumes individual random assignment, independent right censoring
within each arm, consistency, and no interference. These assumptions are
reported but cannot be verified from the table.

The direct utility remains point-only. The retained `prepare` / `analyze`
route can compute **pointwise** two-sided 95% percentile intervals for the
RMST treatment-minus-control contrast and the survival-probability contrast
at `tau` when `bootstrap` is explicitly requested:

```python
result = antecedent.analyze(data, query=query, bootstrap=399, seed=17)
result.survival.rmst_difference_interval
result.survival.survival_at_tau_difference_interval
result.survival.bootstrap_replicates_ok
```

The native estimator resamples whole subjects separately within the two
randomized arms and recomputes the complete risk-set estimator. The seed makes
the draws repeatable. It requires at least eight subjects per arm, 199–100,000
requested replicates, and at least 90% replicates satisfying the estimator's
original support contract. A failed support gate refuses the interval instead
of silently dropping it. The intervals cover the **two scalar contrasts**
separately. The result's
`uncertainty` string and artifact identify the method and valid replicate
count. The support matrix still treats these survival queries as outside its
licensed axes, so the Python result labels this interval available but
unlicensed. Longitudinal or observational survival estimands remain unsupported.

For the unweighted retained route, `bootstrap=399` also gives a **simultaneous**
95% band across the full reported survival-difference curve when each arm has
at least 80 subjects and at least 399 valid replicates satisfy the estimator's
support contract (at least 90% of requested draws):

```python
band = result.survival.difference_band
band.times, band.difference, band.lower, band.upper
```

The band resamples whole subjects within arms and uses a single critical
radius from the bootstrap distribution of maximum curve deviations. Its grid,
endpoints, and accepted replicate
count are frozen in the artifact. In 400 repeated-sampling known-truth studies,
the band covered the full survival-difference grid in 373. It is distinct from
the pointwise RMST and fixed-horizon intervals. Fewer than 80 subjects per arm,
delayed entry, or caller-supplied fixed censoring weights return an explicit
`band_unavailable_reason`; these cases can still report eligible scalar
intervals. The band remains outside the support-matrix axes.

Repeated-sampling native fixtures exercise 400 uncensored two-arm studies and
240 studies with fixed, heterogeneous known censoring probabilities. Each
fixture has an analytic contrast and requires at least 90% empirical coverage
for the nominal 95% interval. These checks provide evidence for the declared
designs; they do not establish finite-sample coverage under every censoring
or event-time law.

## Caller-supplied censoring weights

`estimate_survival_ipcw` provides a direct weighted product-limit path for
right-censored randomized studies. It takes a strictly increasing time grid
from zero through `tau`, including each recorded event/censor time, and a
subjects-by-grid matrix `censoring_survival`. Each matrix entry is the
subject-specific probability of remaining uncensored immediately before that
grid time. The caller must fit and validate these probabilities; the API does
not estimate censoring models or verify their provenance.

```python
weighted = antecedent.survival.estimate_survival_ipcw(
    data,
    query,
    times=[0.0, 30.0, 60.0, 180.0],
    censoring_survival=subject_survival_of_censoring,
    minimum_probability=0.01,
)
weighted.rmst_difference
weighted.censoring_survival_provenance
```

The same native IPCW kernel now runs through retained `prepare` / `analyze`.
Place the subject-specific probabilities in data columns, one per time point,
and specify their order on the query. The observation assumption may name the
variables under which censoring is independent:

```python
from antecedent.observation import IndependentGiven

query = antecedent.survival.SurvivalOutcome(
    "follow_up_days", "event", "treated", tau=180, randomized=True,
    observation_assumption=IndependentGiven(("baseline_risk",)),
    known_censoring=antecedent.survival.KnownCensoringSurvival(
        times=(0, 30, 60, 180),
        columns=("g_0", "g_30", "g_60", "g_180"),
        minimum_probability=0.01,
    ),
)
result = antecedent.analyze(data, query=query)
result.survival.rmst_difference
```

The study artifact freezes the grid, column identities, positivity floor, and
conditioning variables. Prepared refresh reads the censoring columns from the
new table in the same row order. This path also supports competing-risk
cumulative incidence. The retained route also combines known censoring survival
with delayed entry when `IndependentGiven(())` declares marginally independent
entry and censoring. It uses `(entry, exit]` risk sets and keeps each subject's
entry, exit, event cause, and fixed censoring-survival row together in resamples.
Conditional entry remains unsupported. The direct IPCW utilities do not accept
delayed entry.

For retained IPCW survival, `bootstrap=399` produces the same RMST and
fixed-horizon pointwise intervals. The supplied censoring probabilities stay
with their subject in every resample; they are **held fixed**, never refit.
The result and artifact label them
`caller_supplied_fixed_not_fitted_or_verified`. These intervals therefore
exclude uncertainty from fitting or validating the censoring model. The
reported assumptions name conditional independent censoring, correct supplied
probabilities, and censoring positivity.

The native estimator uses inverse-`G` weighted event and risk counts at each
observed event time, then integrates its right-continuous curve for RMST. It
refuses probabilities below the declared positivity floor, probabilities
outside `(0, 1]`, non-monotone rows, and weighted event counts exceeding the
risk set. The reported minimum `G` and event-time risk-set sizes are diagnostics,
not inference. Direct utility results remain point-only; retained results can
report the pointwise intervals described above. Assumptions include correct
caller-supplied conditional censoring survival, sequential censoring
positivity, independent censoring given the supplied history, random assignment,
consistency, and no interference.
For combined delayed entry and fixed known censoring, the observation claim is
the marginal empty-variable form and the supplied censoring probabilities must
be correct. This combined retained route reports the same scalar pointwise
intervals with `bootstrap=299`; the censoring probabilities are held fixed.
Separate 400-study known-truth fixtures covered the RMST contrast in 383,
survival at `tau` in 376, and target-cause incidence in 386 studies. It reports
no simultaneous curve band and does not include uncertainty from fitting `G`.

The same supplied-`G` contract is available for competing-risk cumulative
incidence via `estimate_cumulative_incidence_ipcw(data, competing_query,
times=..., censoring_survival=...)`. It uses inverse-`G` weighted target-cause
failures and all-cause event-free survival in the Aalen–Johansen recursion;
positive non-target causes remain competing events. It requires at least two
distinct observed causes, a present target cause, both randomized arms through
`tau`, and the same time-grid and positivity checks. The result reports minimum
event-time risk-set sizes and minimum supplied `G`. Its direct utility remains
point-only; the retained route offers a pointwise interval for the target-cause
incidence difference at `tau` under `bootstrap=399`. A 240-study known-G,
two-event-time, two-cause repeated-sampling fixture checks its nominal 95%
interval against an analytic truth with a 90% empirical-coverage floor.
Unweighted competing-risk incidence differences also expose the simultaneous
`difference_band` under the 80-subject/399-draw gate. A 400-study fixture
covered the complete incidence-difference grid in 382 studies. Fixed known-G
weights and delayed entry have no simultaneous band.

## Delayed entry and left truncation

Both survival and cumulative-incidence queries accept a `delayed_entry` column.
This reuses the observation contract's explicit `IndependentGiven` assumption;
the supported claim is the marginal empty-variable form:

```python
query = antecedent.survival.SurvivalOutcome(
    duration="follow_up_days",
    event_observed="event",
    treatment="treated",
    tau=180,
    randomized=True,
    delayed_entry="entry_day",
    observation_assumption=antecedent.observation.IndependentGiven(()),
)
summary = antecedent.survival.estimate_survival(data, query)
```

The event-time risk set follows counting-process intervals `(entry, duration]`:
subjects entering exactly at an event time do not join that time's risk set;
right-censored subjects remain in it through their censoring time. The native
kernel rejects negative or non-finite entry times and entry times that are not
strictly earlier than the recorded event/censor time. RMST starts at time zero,
so each randomized arm must include at least one time-zero entrant; the
estimator also requires observed follow-up through `tau` in each arm. These
guards avoid presenting unsupported extrapolation from a left-truncated sample
as an RMST from the origin.

The empty `IndependentGiven(())` claim asserts marginal independent entry and
right censoring; it does not verify either condition. Conditional delayed entry
and interval censoring remain refused. The retained unweighted route accepts
`bootstrap=299` for pointwise RMST and fixed-horizon survival or target-cause
incidence contrasts: resampling keeps each subject's entry and exit together and
recomputes counting-process risk sets. In a 400-study left-truncated survival
fixture with independent entry, the RMST and survival-at-`tau` intervals covered
known truth in 375 and 379 studies. A separate two-cause, 400-study fixture
covered the target-cause incidence contrast in 380 studies. This evidence
supports the declared marginal-entry design, not arbitrary dependent
truncation. The retained combined delayed-entry/fixed-known-G route adds its
own pointwise evidence described above; delayed-entry curves have no
simultaneous band.

## Competing risks

For competing events, use a cause-code column: zero denotes right censoring;
each positive integer denotes one observed event type. Select one positive
target code. At least two event causes must appear in the data, and the target
cause must be present. The native estimator updates the target cumulative
incidence with the Aalen–Johansen risk-set recursion, treating every positive
cause code as an event that removes the subject from the event-free risk set.
Other causes are therefore competing events, never ordinary censoring.

```python
query = antecedent.survival.CompetingRisksOutcome(
    duration="follow_up_days",
    event_cause="event_code",  # 0 = censored; 1, 2, ... = distinct event causes
    treatment="treated",
    target_cause=1,
    tau=180,
    randomized=True,
)
cif = antecedent.survival.estimate_cumulative_incidence(data, query)
cif.incidence_difference
```

The direct utility is **point-only**. The retained route can provide the
pointwise incidence-difference interval and the unweighted simultaneous
difference band described above. Its assumptions include
individual random assignment, complete and
distinct coding of competing causes, independent right censoring within each
arm, consistency, and no interference. The implementation requires both arms
to have observed follow-up through `tau`. Delayed entry uses the same explicit
`IndependentGiven(())` contract and `(entry, duration]` risk-set rule above;
cause misclassification and covariate-adjusted censoring are unsupported.
