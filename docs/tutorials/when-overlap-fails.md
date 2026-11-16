# When the comparison group does not overlap

Some observational questions cannot be answered with the data you have, at least not as asked. This tutorial asks one, gets a refusal, and follows what the refusal says. Each step changes what is being estimated or what would settle it, and the tutorial keeps track of that.

The data are a classic test case: the National Supported Work (NSW) job-training programme, compared with respondents to the Panel Study of Income Dynamics (PSID). The people who entered the programme had very low earnings; most PSID respondents did not. That gap is what this tutorial is about.

- **You will learn** how to read a `positivity_violated` [refusal](../concepts/refusals.md) and its remedy, what a declared overlap clip and a declared overlap trim each do to the estimand and to the claim's status, and why the spread across those choices belongs in the report.
- **Before you start:** [installation](../start/installation.md) and [a first analysis](../start/first-analysis.md). [Estimate an effect from observational data](observational-study.md) shows the same workflow where overlap holds.
- **Time:** the example runs in under a minute.
- **Run it:** `python examples/python/when_overlap_fails.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/when_overlap_fails.ipynb). The complete script is embedded [below](#the-complete-example).

## The question

- **Unit and population:** men who entered the NSW demonstration, a United States job-training programme of the mid-1970s.
- **Treatment and target:** entering the programme. The estimand is the effect on the treated: the average change in 1978 earnings for those who entered, relative to what they would have earned without it.
- **Outcome:** real earnings in 1978, in dollars.
- **Time:** the covariates (age, schooling, race, ethnicity, marital status, degree status, earnings in 1974 and 1975) precede the programme; the outcome follows it.

You declare that each covariate may affect both entering the programme and 1978 earnings, and that the programme may affect 1978 earnings. That is the assumption of no unmeasured confounding given those eight variables; nothing below tests it.

NSW also assigned people at random, so a randomised estimate of the same effect on the treated exists. Lalonde (1986) asked whether non-experimental comparison groups reproduce it. This tutorial uses that benchmark only at the end.

## Where the data come from

The manifest [`examples/data/nsw_psid.toml`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/data/nsw_psid.toml) records the origin, citations, per-file SHA-256, retrieval date, preprocessing and columns. The publisher states no support, so Antecedent ships no copy. `examples/data/fetch.py` downloads the files when you set `ANTECEDENT_DATA=download` and refuses any file that fails its checksum; the [data README](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/data/README.md) describes the cache and the offline path.

Without verified files the example runs on a synthetic stand-in with the same ten columns, the same sizes (185 treated rows and 2,490 comparison rows) and the same overlap problem. Its effect on the treated is a fixture truth of the generator and says nothing about the real programme. The first line of output says which table was used:

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

`COVARIATES` is the eight pre-programme columns. `target_population=population.Treated()` makes the estimand the effect on the treated. Only the estimator, and the overlap policy it carries, changes from one step to the next, so every difference in the answers below comes from that choice.

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

This is a statement about the data, not a numerical failure. The effect on the treated compares each participant with comparison rows like them. For 139 PSID rows the model finds no participant like them, so the data say nothing about what treatment would have done there, and weighting would divide by numbers indistinguishable from zero. Weighting clips ordinary scores to [0.01, 0.99] by default, but a default does not decide a positivity question: that choice is yours to declare.

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

Now the clip governs the boundary rows: propensities used in the weights are bounded to [0.01, 0.99]. On the real files weighting returns an effect on the treated of about −183 dollars. Read the claim before the number. The effective sample size behind the weights (`clipped.estimate.overlap_ess`) is about 153 of 2,675 rows, and the claim's support domain is `limited`: the clip moved scores at the boundary, so the answer rests on it. `host.claim_status(claim)` is `partial`, which tells an application to relay that limit with the number rather than use it as stated.

## Follow the remedy: declare a trim

The first remedy in the refusal is a trim. Declare it, and run two estimators on the same trimmed rows:

```python
trim = Overlap(clip=None, trim=0.01)
trimmed = ant.outcomes.require_success(ask(data, PropensityWeighting(bootstrap=100, overlap=trim)))
aipw = ant.outcomes.require_success(ask(data, Aipw(bootstrap=100, overlap=trim)))
excluded = trimmed.estimate.overlap_excluded_fraction
```

Rows whose estimated propensity falls outside [0.01, 0.99] are excluded, and `estimate.overlap_excluded_fraction` reports the share. On the real files that is 73% of rows.

The answer changes meaning as well as value. It is now the effect on the participants who have comparable comparison rows, not on all participants, and the claim's support is again `limited`, so its status is `partial`. Weighting gives about 1,610 dollars (standard error about 940); AIPW, on the same trimmed rows, about 2,282.

## How far the answer moves is itself the finding

The example prints the largest minus the smallest of the three estimates (clipped weighting, trimmed weighting, trimmed AIPW). On the real files they span about 2,465 dollars. The data have not determined the effect; the overlap choice and the estimator have. Report that spread with any single number, and report the excluded fraction with any trimmed number, because a reader needs both to know what the estimate is about.

## The last remedy: a design that produces both treatments

The refusal's final option, data or a design that produces both treatments for people like the participants, is exactly what NSW supplied: a randomised experiment. On the real files the example prints the experimental difference in mean 1978 earnings between the same participants and the randomised controls, about 1,794 dollars with a standard error of about 671. On the stand-in there is no experiment; the example prints the fixture effect over all 185 treated rows instead and notes that the trimmed estimate targets a different population, the treated rows with common support.

Read that comparison carefully. That the trimmed weighting estimate lands near it does not make weighting the right method here, and the clipped estimate's miss does not say which assumption failed. Dehejia and Wahba (1999) and later work found that estimates on these data depend strongly on the comparison group and the method. The experiment answers the question because its design creates the overlap the observational data lack.

## What this established

Antecedent refused a request whose answer would have rested on rows with no comparable counterpart. It named how many rows and why, and listed what would support an answer. A declared clip produced a number whose claim says it rests on the clip. Following the trim remedy produced an estimate for a stated, smaller population, with the exclusion reported and the claim marked partial. Comparing overlap choices showed how much the answer depends on them. The randomised benchmark showed what a design with real overlap provides.

## What this did not establish

The observational data did not establish the effect on all participants: no declared overlap policy recovers information about participants with no comparable comparison rows. Nothing here tested the no-unmeasured-confounding assumption, which every estimate above shares. Agreement between one trimmed estimate and the experiment is not evidence that weighting is valid on these data.

## Where to go next

For an observational analysis where overlap holds and the workflow runs end to end, see [estimate an effect from observational data](observational-study.md). For the overlap options themselves, see [overlap and E-value diagnostics](../guides/tier-diagnostics.md) and [refusals](../concepts/refusals.md).

## The complete example

<!-- example: examples/python/when_overlap_fails.py -->
