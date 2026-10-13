# Sampled counterfactual transport

General counterfactual transport identifies a nested or conditional
counterfactual probability of a target population from labelled observational
and interventional laws of the target and of source populations. The exact
route evaluates that identification on supplied finite laws. This page covers
the sampled route: the same stored derivation evaluated on laws estimated from
counts, with the sample's uncertainty kept apart from the identification.

Nothing on this route is calibrated. It publishes a point estimate and an
uncalibrated standard error, and it keeps the interval closed.

## Four claims, four fields

| Claim | What it says | What it does not say |
| --- | --- | --- |
| Structural identification | the stored derivation identified the query from the declared graph, selection diagram and regime family | anything about the sample |
| Exact finite-law evaluation | the arithmetic of the derivation on the estimated tables is exact | that the tables are the population's laws |
| Point estimate | the plug-in (or smoothed plug-in) value estimates the population probability | a coverage statement |
| Statistical inference | withheld: `cell_not_licensed` with detail `counterfactual_transport_sampled.interval_not_licensed` | an interval, posterior or standard-error calibration |

A result is `plug_in_point_estimate` for unsmoothed count shares and
`smoothed_plug_in_point_estimate` when any table was smoothed. A request for an
unseen-mass envelope reports a structural range and may report no point at all.

## Evidence

Evidence is a dense count table per cited population, regime and intervention
level over exactly the variables the derivation cites, with a declared sampling
design:

* independent units, or strata whose totals were fixed by design (the stratum
  proportions are then not sampled);
* declared structural zeros: cells that are impossible, which must have count
  zero and are never smoothed or filled;
* sample-size and sparsity limits (minimum total per table, largest unseen
  share).

Counts are converted to finite laws. The exact route's refusals are preserved
on the estimated tables: a conditioning event of zero estimated mass is a
`transport_support_failure`, a cited regime with no table is
`transport_missing_provider`, and a table the derivation does not cite is
refused rather than ignored.

## Unseen and sparse cells

An unseen cell is a zero count that is not a declared structural zero. It is
never zero-filled silently. The request declares one rule:

| Rule | Result | Label |
| --- | --- | --- |
| refuse | any table with an unseen cell is refused (`transport_support_failure`, `counterfactual_transport_sampled.unseen_cell`) | |
| additive smoothing `a` | `p = (n + a) / (N + K a)` over the `K` non-structural cells (the posterior mean under a symmetric Dirichlet(`a`) prior); `a` is recorded | smoothed |
| unseen-mass envelope `d` | the range of the derivation over the finite family in which each table keeps its observed shares or moves mass `d` onto one unseen cell; members with no defined value are counted and reported | structural envelope |

The envelope is a range over that family, not an interval and not a proven outer
bound over every placement of the mass: the functional is a ratio of sums of
products, so its extremes over the whole polytope need not lie in the family.

## Factors that read fewer coordinates

In the model a district factor is a function of its parents. Read from a sample
it differs across the coordinates it does not read by sampling noise, so the
exact route's agreement requirement cannot hold. The sampled route evaluates
such a factor as the mass-weighted average, under the table's own mass given the
parents, over the unread coordinates. The two coincide when the tables satisfy
the graph. The number of averaged factors and any tolerance repair (a negative
consistency complement set to zero, a probability clamped into `[0, 1]`) are
reported with the result.

## Dependence between tables

Overlap is declared for every pair of tables:

* `independent`: disjoint units. Two tables declared independent that hold
  identical observations are refused as `evidence_double_counted`.
* `shared_units`: the same units. Counts determine the covariance only when the
  two tables are the same observations; then they are one dataset (one
  multinomial). Shared units between different tables are refused
  (`route_not_supported`, `counterfactual_transport_sampled.shared_units_unlinked`).
* `unknown`, or nothing declared: refused (`sampling_dependence_unknown`).
  Independence is never assumed.

A declaration can be derived from an evidence ledger
(`overlap_declaration_from_ledger`): disjoint declared observations are
independent, shared observations are shared units, and sources whose
observations are not declared stay unknown.

## Uncertainty method

One method is used: the delta method for the multinomial cell proportions of
each independent dataset, through the whole functional (identification, ratios
and the averaging above). The gradient is a central difference at relative step
`1e-5` of each positive cell. A perturbed evaluation that is undefined withholds
the standard error with a stated reason; it is never dropped. The bootstrap was
not chosen: it needs a replicate stream per population, regime and replicate and
a rule for empty strata in each replicate, and the delta method is deterministic
and replayable bit for bit. The price is the linearization near the boundary
(sparse tables, small conditioning mass), which is exactly what calibration must
measure.

The standard error carries the label `uncalibrated_standard_error`. At smoothed
tables it is the standard error at the smoothed table and omits the smoothing
bias. An envelope request has none.

## Interval route

No interval is published. The coordinate that would license one is the whole
method across the sample-size grid, including estimated-table zero and sparse
cells; it is allocated and unmeasured (`parity/promotion.toml`, records
`counterfactual.transport_sampled` and `counterfactual.transport_sampled_interval`).
A nominal normal interval exists only behind the `calibration-internal` feature
of `antecedent-estimate` for the calibration harness. If calibration fails at a
coordinate, the point estimate ships labelled and the interval stays closed.

## Artifact

`sampled_counterfactual_transport_v1` binds the counts, the derivation and its
inputs (model class, query, evidence catalog), the unseen-cell rule and limits,
the overlap declaration, the uncertainty method and the estimate with its four
labels. A fresh consumer holding only the bytes and an expected identity
re-decides the derivation and recomputes the estimate and standard error from
the counts bit for bit. Counts edited and resealed, an overlap declaration
flipped, a swapped derivation and an edited stored estimate are refused.

## Limits

At most 64 count tables per request, 65536 cells per table, and 65536 envelope
members; the derivation's own bounds (8 variables, 4 levels each, 4 sources)
apply. Continuous variables, temporal dependence, clustered or weighted designs
and selection into the sample are separate rows.
