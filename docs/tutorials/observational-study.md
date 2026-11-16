# Estimate an effect from observational data

Nobody assigned these smokers to quit, so the question has to be answered from people who chose for themselves. This tutorial asks one such question on real public data, follows the assumptions the answer depends on, and ends by stating what the numbers do and do not say.

- **You will learn** how to declare a graph and a missing-data policy, inspect a prepared study before estimating, read overlap diagnostics and calibration status with an estimate, change the estimator, the target population and the missing-data policy on the same prepared study, and read an E-value.
- **Before you start:** [installation](../start/installation.md) and [a first analysis](../start/first-analysis.md). The example needs NumPy and no other package.
- **Time:** the example runs in under a minute.
- **Run it:** `python examples/python/observational_study.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/notebooks/observational_study.ipynb). The complete script is embedded [below](#the-complete-example).

## The question

- **Unit and population:** cigarette smokers aged 25 to 74 who had a baseline examination in the first National Health and Nutrition Examination Survey (1971-75) and a follow-up interview in 1982 (the NHEFS cohort).
- **Treatment:** quitting smoking between the two visits.
- **Estimand:** the average effect of quitting, in kilograms: the average, over this population, of weight change if everyone quit minus weight change if no one did.
- **Outcome:** weight in 1982 minus weight in 1971, in kg.
- **Time:** all covariates are measured at baseline, before quitting; the outcome follows it.

The question and the data follow Hernán and Robins, *Causal Inference: What If*, chapter 12.

## Where the data come from

The manifest [`examples/data/nhefs.toml`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/data/nhefs.toml) records the origin, citation, SHA-256, retrieval date, preprocessing and the columns used. The publisher states no support, so Antecedent ships no copy. `examples/data/fetch.py` downloads the file when you set `ANTECEDENT_DATA=download` and refuses any file that fails its checksum; the [data README](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/data/README.md) describes the cache and the offline path.

Without a verified file the example runs on a synthetic stand-in with the same columns, 1,629 rows and 63 missing outcomes. Every simulated quitter gains exactly 3.0 kg more, so its average effect of quitting, 3.0 kg, is a fixture truth of the generator, and it says nothing about smokers. The first line of output says which table was used:

```text
DATA: synthetic stand-in (fixture average effect of quitting 3.0 kg).
DATA: NHEFS smoking-cessation subset (checksum verified).
```

The kilogram figures quoted below are from the verified real file. A stand-in run follows the same steps and prints different figures; what it adds is a check the real data cannot offer, described [at the end](#compare-with-the-published-analysis-carefully).

The example then prints the unadjusted difference in mean weight change, quitters minus others, among people with a recorded outcome. On the real file it is 2.54 kg. This is a description of the two groups, not an effect: quitters differ from non-quitters in age, smoking history and weight before anyone quits.

## Declare the graph

You declare that each baseline covariate may affect both quitting and weight change, and that quitting may affect weight change. The covariates are sex, race, age, education, cigarettes per day, years of smoking, exercise, daily activity and starting weight.

```python
covariates = [k for k in data if k not in ("qsmk", "wt82_71")]
graph = [(z, "qsmk") for z in covariates] + [(z, "wt82_71") for z in covariates]
graph.append(("qsmk", "wt82_71"))
query = ant.AverageEffect("qsmk", "wt82_71")
```

`qsmk` is the quitting indicator and `wt82_71` the weight change. `ant.AverageEffect("qsmk", "wt82_71")` names the estimand stated above.

That declaration is the assumption of no unmeasured confounding given those covariates. Nothing in this tutorial tests it.

The table has more columns than the list of covariates because the propensity model needs room to bend. Age, smoking intensity, years smoked and starting weight enter linearly and squared; education, exercise and activity are categories, so each non-reference level gets an indicator column. This is the specification the book uses. The graph is the same however the covariates are encoded; the encoding changes what the models can fit.

## Prepare and inspect

Prepare the analysis with propensity weighting, 200 bootstrap replicates and `missing="complete_case"`, then inspect it before estimating:

```python
study = ant.prepare(
    data,
    graph=graph,
    query=query,
    estimator=PropensityWeighting(bootstrap=200),
    missing="complete_case",
    inference=ant.Frequentist(bootstrap=200),
    execution=ant.Execution(seed=7),
)
print(study.inspect())
```

`ant.prepare` compiles the declaration without estimating anything. Inspection reports the identification status, the adjustment set and what the declared missing-data policy will do, so a declaration that cannot identify the effect is visible before any model is fitted. After estimation the example asserts the two facts that matter here: the status is `NonparametricallyIdentified`, and the adjustment set is exactly the declared covariates. Identification is a statement about the graph you declared: if the graph is right, adjusting for these columns removes confounding. It is not evidence that the graph is right.

## Missing outcomes are a declared policy

Of the 1,629 people, 63 have no 1982 weight. Antecedent does not drop them silently: `missing="complete_case"` is a declaration, and the result reports the accounting and the assumption the policy rests on.

```python
result = ant.outcomes.require_success(ant.estimate(study))
missing = result.missing_data
```

```text
Missing data (complete_case): 1566 of 1629 rows kept, 63 dropped.
```

`missing.accounting` holds `rows_total`, `rows_kept` and `rows_dropped`; `missing.assumption` states what the policy assumes, and the example prints it on the next line. Complete-case analysis is adequate when the rows that remain represent the population the question is about. If who drops out depends on things that also drive weight change, it may not be. The book's analysis handles this with a loss-to-follow-up model; this one declares the simpler policy and checks it later.

## Estimate and read the diagnostics

Propensity weighting reweights the sample so the covariates of quitters and non-quitters match. The example reads the point estimate from `result.answer.value`, the bootstrap standard error from `result.estimate.se_bootstrap`, and forms a 99% Wald interval, the estimate plus or minus 2.576 standard errors:

```python
est = result.estimate
se = est.se_bootstrap or est.se_analytic
effect = result.answer.value
lo, hi = effect - 2.576 * se, effect + 2.576 * se
```

On the real file the average effect of quitting is 3.44 kg with a standard error of about 0.52, against the unadjusted 2.54 kg. The interval covers sampling variation under the declared graph and missing-data policy. It does not cover the possibility that either is wrong.

Read the overlap diagnostics before the number. The smallest estimated propensity (`est.overlap_propensity_min`) is about 0.05 and the effective sample size (`est.overlap_ess`) is about 1,013 of 1,566 rows, so weights are moderate and no row stands in for a large share of the sample. The example asserts that the smallest propensity is above 0.01. That is the contrast with [the NSW comparison](when-overlap-fails.md), where propensities reach the boundary and Antecedent refuses without a declared overlap policy. Here the data cover both groups.

Also read `result.calibration.status`, which the example prints. It says whether a measured coverage record supports this interval construction; the interval is reported whatever the status says. [Reading a result](../start/reading-a-result.md#calibration-status) lists the statuses.

## Continued use: the effect among those who quit

Reconfigure the same study with `target_population=population.Treated()`:

```python
treated_query = ant.AverageEffect("qsmk", "wt82_71", target_population=population.Treated())
among_quitters = ant.outcomes.require_success(
    ant.estimate(ant.reconfigure(study, query=treated_query))
)
```

The question becomes: for people who actually quit, how much did quitting change their weight? On the real file the answer is 3.34 kg. `ant.reconfigure` returns a new prepared study and leaves `study` as it was, so the graph is declared once. A different population gives a different estimand, not a different estimate of the same thing, so Antecedent treats it as a new query on the prepared study.

## Check the answer with other estimators

Reconfigure the estimator and leave everything else as declared:

```python
aipw = ant.outcomes.require_success(
    ant.estimate(ant.reconfigure(study, estimator=Aipw(bootstrap=200)))
)
regression = ant.outcomes.require_success(
    ant.estimate(ant.reconfigure(study, estimator="linear.adjustment.ate"))
)
imputed = ant.outcomes.require_success(
    ant.estimate(ant.reconfigure(study, missing="multiple_imputation"))
)
```

On the real file AIPW (propensity weighting combined with an outcome model) gives 3.25 kg, and outcome regression gives a similar value. The third request changes the missing-data policy instead of the estimator: multiple imputation fills in the 63 missing outcomes and gives 3.43 kg. Close agreement between estimators that make different modelling bets supports the numerical answer; it does not test the no-unmeasured-confounding assumption, which all of them share.

## Sensitivity: how strong would an unmeasured confounder need to be?

The E-value converts the estimate and its standard error into the minimum strength of association, with both quitting and weight change, that an unmeasured confounder would need to explain the estimate away. It reads the effect on a standardized scale, dividing by the outcome standard deviation among those who did not quit:

```python
sd = float(np.nanstd(outcome[qsmk == 0], ddof=1))
sens = v.evalue(effect / sd, standard_error=se / sd, scale="standardized_mean_difference")
```

`sens.range.break_even` is the E-value, on the risk-ratio scale, and `sens.range.assumption` names the untestable assumption it concerns; the example prints both and asserts that the E-value is above 1. A large E-value says the conclusion is hard to overturn with one confounder; it cannot say whether such a confounder exists.

## Compare with the published analysis, carefully

Hernán and Robins report an effect of about 3.4 kg (95% CI about 2.4 to 4.5) by inverse-probability weighting. That is close to 3.44 kg, but the analyses are not identical: theirs also weights for loss to follow-up, and this one declares a complete-case policy. Agreement between two analyses that share a graph shows the computation is reproducible. It says nothing about whether either graph is right, and neither number is presented here as causal truth.

On the stand-in the comparison is of a different kind. The generator's effect is known, so the example asserts that the 99% interval contains 3.0 kg. That checks the computation against a truth the fixture supplies; real data supply none.

## What cannot be tested

Two assumptions carry this answer, and the data cannot check either by themselves.

- **No unmeasured confounding.** Whether someone quits may depend on motivation, health scares or diet, none of which is a column. Adding more estimators does not touch this. What would help is a measured proxy for such a cause, a design that assigns quitting, or a sensitivity analysis such as the E-value, which states how strong such confounding would need to be.
- **Complete-case missingness.** Multiple imputation gives 3.43 kg, close to the complete-case value. That agreement shows the answer does not move much between two ways of handling the missing rows. It does not show that the missing outcomes are ignorable.

## What this established

Antecedent identified the average effect of quitting under the declared graph and estimated it at about 3.4 kg (standard error about 0.5) on the 1,566 people with a recorded outcome. Overlap was good, other estimators and an imputation check gave similar values, and the effect among those who quit was a stated, slightly different quantity.

## What this did not establish

None of this established that the graph is right or that the missing outcomes are ignorable. The estimate is conditional on both declarations, and the E-value only states how strong a violation of the first would need to be.

## Where to go next

The next question is whether the assumptions hold. For that you need evidence the data here do not contain: measured proxies for the unmeasured causes, a follow-up model for who drops out, or an experiment. For an analysis where overlap fails and the answer has to change shape, see [when the comparison group does not overlap](when-overlap-fails.md). For reading the claim and its status, see [reading a result](../start/reading-a-result.md).

## The complete example

<!-- example: examples/python/observational_study.py -->
