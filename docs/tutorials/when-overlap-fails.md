# When the comparison group does not overlap

How much did a job-training programme change participants’ earnings? One way to estimate the effect is to find people who did not receive training but had similar employment histories. That comparison becomes difficult when the two groups earned very different amounts before training even began.

The National Supported Work (NSW) participants had very low earnings before the programme. Most people in the comparison survey, the Panel Study of Income Dynamics (PSID), earned more. You’ll see what happens when you limit the weights given to unusual cases, then when you restrict the analysis to people with comparable histories.

Run `python examples/python/when_overlap_fails.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/notebooks/when_overlap_fails.ipynb). See the [complete example](#the-complete-example).

For a comparison where both groups are well represented, see [estimate an effect from observational data](observational-study.md).

## The question

- **Unit and population:** men who entered the NSW demonstration, a United States job-training programme of the mid-1970s.
- **Treatment and target:** entering the programme. The estimand is the effect on the treated: the average change in 1978 earnings for those who entered, relative to what they would have earned without it.
- **Outcome:** real earnings in 1978, in dollars.
- **Time:** the covariates (age, schooling, race, ethnicity, marital status, degree status, earnings in 1974 and 1975) precede the programme; the outcome follows it.

You declare that each covariate may affect both entering the programme and 1978 earnings, and that the programme may affect 1978 earnings. That is the assumption of no unmeasured confounding given those eight variables; nothing below tests it.

NSW also assigned people at random, so a randomised estimate of the same effect on the treated exists. Lalonde (1986) asked whether non-experimental comparison groups reproduce it. This tutorial uses that benchmark only at the end.

## Where the data come from

The manifest [`examples/data/nsw_psid.toml`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/data/nsw_psid.toml) records the origin, citations, per-file SHA-256, retrieval date, preprocessing and columns. The publisher states no support, so Antecedent ships no copy. `examples/data/fetch.py` downloads the files when you set `ANTECEDENT_DATA=download` and refuses any file that fails its checksum; the [data README](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/data/README.md) describes the cache and the offline path.

Without the verified files, the script generates the same ten columns for 185 participants and 2,490 comparison respondents. The simulation reproduces the overlap problem and gives you a known training effect to check. The first line of output identifies the dataset.

```text
DATA: synthetic stand-in (fixture effect on the treated $...).
DATA: NSW treated group with the PSID comparison group (checksums verified).
```

Every step below runs the same way on either table: the same refusal code, the same `limited` support and `partial` status, an excluded fraction strictly between 0 and 1. The dollar figures quoted below are from the verified real files; a stand-in run prints different ones.

## Declare the graph and the query

The graph and the query are declared once and reused by every request:

```python
GRAPH = [(z, "treat") for z in COVARIATES] + [(z, "re78") for z in COVARIATES]
GRAPH.append(("treat", "re78"))
QUERY = ant.AverageEffect("treat", "re78", target_population=population.Treated())


def ask(data, estimator):
    return ant.analyze(
        data,
        graph=GRAPH,
        query=QUERY,
        estimator=estimator,
        refute="none",
        execution=ant.Execution(seed=11),
    )
```

`COVARIATES` contains the eight pre-programme measurements. `population.Treated()` asks about the people who entered training. Keep the graph and the query fixed as you try different estimators and overlap policies.

## Ask, and read the refusal

Ask for the effect on the treated with propensity weighting and no overlap policy:

```python
refused = ask(data, PropensityWeighting(bootstrap=100))
refusal = refused.refusal
assert refusal is not None and refusal.code == "positivity_violated"
rows = refusal.fields["refusal_fields"]["boundary_count"]
```

`ant.analyze` returns an outcome whose `refusal` is set; it does not raise. The code is `positivity_violated`, and on the real files the message reads:

> the propensity model gives 139 rows a probability of treatment within 1e-8 of 0 or 1, so those rows have essentially no chance of the other treatment and an inverse-probability weight is unsupported there.

The propensity model estimates a person’s chance of entering training given their recorded characteristics. For these 139 rows it puts that chance essentially at a boundary. The usual weighting calculation relies on comparisons in parts of the data where both groups are represented. To proceed when scores reach a boundary, you must choose how to handle those rows.

`refusal.remedy` lists what would support an answer:

- declare an overlap trim, which estimates on the population with common support (the target changes, and the excluded fraction is reported);
- declare a clip that bounds the weights;
- restrict the population or covariates to where both treatments occur;
- obtain data or a design that produces both treatments there.

A host application can switch on the code and read the counts from `refusal.fields["refusal_fields"]` (`reason`, `boundary_count`, `boundary_margin`) without parsing the message.

## Declare a clip, and read what the claim says about it

Declare the clip explicitly, and read the claim that comes back with the estimate:

```python
clipped = ant.outcomes.require_success(
    ask(data, PropensityWeighting(bootstrap=100, overlap=Overlap(clip=0.01)))
)
claim = clipped.inspect().contract["claim"]
assert claim["support_domain"] == "limited"
assert host.claim_status(claim) == "partial"
```

Clipping moves scores below 0.01 up to 0.01 and scores above 0.99 down to 0.99 before computing weights. On the survey data the estimated effect is about −$183. Check `clipped.estimate.overlap_ess`. After weighting, the effective sample size is only about 153 out of 2,675 rows. Report that the estimate depends on clipping. Its support is `limited` and `host.claim_status(claim)` returns `partial`.

## Follow the remedy: declare a trim

The first remedy in the refusal is a trim. Declare it, and run two estimators on the same trimmed rows:

```python
trim = Overlap(clip=None, trim=0.01)
trimmed = ant.outcomes.require_success(ask(data, PropensityWeighting(bootstrap=100, overlap=trim)))
aipw = ant.outcomes.require_success(ask(data, Aipw(bootstrap=100, overlap=trim)))
excluded = trimmed.estimate.overlap_excluded_fraction
```

Rows whose estimated propensity falls outside [0.01, 0.99] are excluded, and `estimate.overlap_excluded_fraction` reports the share. On the real files that is 73% of rows.

Trimming changes who the estimate describes. Among participants whose scores fall in the retained range, weighting gives an estimated gain of about $1,610 (standard error about $940). AIPW gives about $2,282 on the same rows. Report the restricted population with either result. Both claims have `limited` support and `partial` status.

## How far the answer moves is itself the finding

From clipped weighting to trimmed AIPW, the estimates span about $2,465. Two things change across that range. Trimming restricts the population, and switching to AIPW changes the estimator within that population. Report which people each estimate describes and how many rows trimming excluded.

## The last remedy: a design that produces both treatments

The refusal's final option, data or a design that produces both treatments for people like the participants, is exactly what NSW supplied: a randomised experiment. On the real files the example prints the experimental difference in mean 1978 earnings between the same participants and the randomised controls, about 1,794 dollars with a standard error of about 671. On the stand-in there is no experiment; the example prints the fixture effect over all 185 treated rows instead and notes that the trimmed estimate targets a different population, the treated rows with common support.

Read that comparison carefully. That the trimmed weighting estimate lands near it does not make weighting the right method here, and the clipped estimate's miss does not say which assumption failed. Dehejia and Wahba (1999) and later work found that estimates on these data depend strongly on the comparison group and the method. The experiment answers the question because its design creates the overlap the observational data lack.

## What this established

On the verified data, clipping and trimming even give different signs for the earnings effect. The refusal led you to the source of that sensitivity. Earnings histories differ so much between the groups that the choice of whom to compare matters to the result.

## What this did not establish

The observational data did not establish the effect on all participants: no declared overlap policy recovers information about participants with no comparable comparison rows. Nothing here tested the no-unmeasured-confounding assumption, which every estimate above shares. Agreement between one trimmed estimate and the experiment is not evidence that weighting is valid on these data.

## Where to go next

For an observational analysis where overlap holds and the workflow runs end to end, see [estimate an effect from observational data](observational-study.md). For the overlap options themselves, see [overlap and E-value diagnostics](../guides/tier-diagnostics.md) and [refusals](../concepts/refusals.md).

## The complete example

<!-- example: examples/python/when_overlap_fails.py -->

[Open the Python script](../../examples/python/when_overlap_fails.py)
