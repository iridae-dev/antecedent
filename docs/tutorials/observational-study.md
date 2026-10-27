# Estimate an effect from observational data

Nobody assigned these smokers to quit, so the question has to be answered from people who chose for themselves. This tutorial asks one such question on real public data, follows the assumptions the answer depends on, and ends by stating what the numbers do and do not say.

## The question

- **Unit and population:** cigarette smokers aged 25 to 74 who had a baseline examination in the first National Health and Nutrition Examination Survey (1971-75) and a follow-up interview in 1982 (the NHEFS cohort).
- **Treatment:** quitting smoking between the two visits.
- **Estimand:** the average effect of quitting, in kilograms: the average, over this population, of weight change if everyone quit minus weight change if no one did.
- **Outcome:** weight in 1982 minus weight in 1971, in kg.
- **Time:** all covariates are measured at baseline, before quitting; the outcome follows it.

The question and the data follow Hernán and Robins, *Causal Inference: What If*, chapter 12.

## Where the data come from

The manifest [`examples/data/nhefs.toml`](https://github.com/iridae-dev/antecedent/blob/main/examples/data/nhefs.toml) records the origin, citation, SHA-256, retrieval date, preprocessing and the columns used. The publisher states no licence, so Antecedent ships no copy. `examples/data/fetch.py` downloads the file when you set `ANTECEDENT_DATA=download` and refuses any file that fails its checksum; the [data README](https://github.com/iridae-dev/antecedent/blob/main/examples/data/README.md) describes the cache and the offline path.

Without a verified file the example runs on a synthetic stand-in with the same columns, 1,629 rows and 63 missing outcomes. Its average effect of quitting, 3.0 kg, is a fixture truth of the generator, and it says nothing about smokers. The numbers below are from the verified real file.

## Declare the graph

You declare that each baseline covariate may affect both quitting and weight change, and that quitting may affect weight change. The covariates are sex, race, age, education, cigarettes per day, years of smoking, exercise, daily activity and starting weight.

That declaration is the assumption of no unmeasured confounding given those covariates. Nothing in this tutorial tests it.

The squares and indicators exist so the propensity model can bend. Age, smoking intensity, years smoked and starting weight enter linearly and squared; education, exercise and activity are categories, so each non-reference level gets an indicator column. This is the specification the book uses. The graph is the same however the covariates are encoded; the encoding changes what the models can fit.

## Prepare and inspect

Prepare the analysis with propensity weighting, 200 bootstrap replicates and `missing="complete_case"`, then inspect it before estimating. Inspection reports the identification status, the adjustment set and what the declared missing-data policy will do.

## Missing outcomes are a declared policy

Of the 1,629 people, 63 have no 1982 weight. Antecedent does not drop them silently: `missing="complete_case"` is a declaration, and the result reports the accounting (1,629 rows, 1,566 kept, 63 dropped) and the assumption the policy rests on. Complete-case analysis is adequate when the rows that remain represent the population the question is about. If who drops out depends on things that also drive weight change, it may not be. The book's analysis handles this with a loss-to-follow-up model; this one declares the simpler policy and checks it later.

## Estimate and read the diagnostics

The unadjusted difference in mean weight change, quitters minus others, is 2.54 kg. Propensity weighting reweights the sample so the covariates of quitters and non-quitters match, and gives an average effect of quitting of 3.44 kg with a bootstrap standard error of about 0.52.

Read the overlap diagnostics before the number. The smallest estimated propensity is about 0.05 and the effective sample size is about 1,013 of 1,566 rows, so weights are moderate and no row stands in for a large share of the sample. That is the contrast with [the NSW comparison](when-overlap-fails.md), where propensities reached 1e-9 and Antecedent refused without a declared overlap policy. Here the comparison is supported by the data. Also read `result.inspect().calibration`: the interval's calibration status is reported with the result rather than assumed.

## Check the answer with other estimators

On the same declaration, AIPW (propensity weighting combined with an outcome model) gives 3.25 kg, and outcome regression gives a similar value. Close agreement between estimators that make different modelling bets supports the numerical answer; it does not test the no-unmeasured-confounding assumption, which all of them share.

## Continued use: the effect among those who quit

Reconfigure the same study with `target_population=population.Treated()`. The question becomes: for people who actually quit, how much did quitting change their weight? The answer is 3.34 kg. A different population gives a different estimand, not a different estimate of the same thing, so Antecedent treats it as a new query on the prepared study.

## Sensitivity: how strong would an unmeasured confounder need to be?

The E-value converts the estimate and its standard error into the minimum strength of association, with both quitting and weight change, that an unmeasured confounder would need to explain the estimate away. It reads the effect on a standardized scale, dividing by the outcome standard deviation. A large E-value says the conclusion is hard to overturn with one confounder; it cannot say whether such a confounder exists.

## Compare with the published analysis, carefully

Hernán and Robins report an effect of about 3.4 kg (95% CI about 2.4 to 4.5) by inverse-probability weighting. That is close to 3.44 kg, but the analyses are not identical: theirs also weights for loss to follow-up, and this one declares a complete-case policy. Agreement between two analyses that share a graph shows the computation is reproducible. It says nothing about whether either graph is right, and neither number is presented here as causal truth.

<!-- example: examples/python/observational_study.py -->

## The boundary that teaches: what cannot be tested

Two assumptions carry this answer, and the data cannot check either by themselves.

- **No unmeasured confounding.** Whether someone quits may depend on motivation, health scares or diet, none of which is a column. Adding more estimators does not touch this. What would help is a measured proxy for such a cause, a design that assigns quitting, or an analysis that declares a bound on how large the confounding could be, which is what the E-value expresses.
- **Complete-case missingness.** Multiple imputation fills in the 63 missing outcomes from the declared model and gives 3.43 kg, close to the complete-case value. That agreement shows the answer does not move much between two ways of handling the missing rows. It does not show that the missing outcomes are ignorable.

## What this established

Antecedent identified the average effect of quitting under the declared graph and estimated it at about 3.4 kg (standard error about 0.5) on the 1,566 people with a recorded outcome. Overlap was good, other estimators and an imputation check gave similar values, and the effect among those who quit was a stated, slightly different quantity.

The next question is whether the assumptions hold. For that you need evidence the data here do not contain: measured proxies for the unmeasured causes, a follow-up model for who drops out, or an experiment. For an analysis where overlap fails and the answer has to change shape, see [when the comparison group does not overlap](when-overlap-fails.md). For reading the claim and its status, see [reading a result](../start/reading-a-result.md).
