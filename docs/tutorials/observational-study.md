# Estimate a training programme's effect from observational data

This tutorial asks what a job-training programme did to the later earnings of the people who entered it, using comparison people who did not. It uses a classic public dataset, with a record of where the data come from and how to verify them, and it stays honest about what observational data can and cannot show.

## The question

- **Unit and population:** adult men eligible for the National Supported Work (NSW) demonstration, a United States job-training programme of the mid-1970s.
- **Treatment and target:** entering the programme. The estimand is the effect on the treated: the average change in 1978 earnings for those who entered, relative to what they would have earned without it.
- **Outcome:** real earnings in 1978, in dollars.
- **Time:** the covariates (age, schooling, race, ethnicity, marital status, degree status, earnings in 1974 and 1975) precede the programme; the outcome follows it.

The data have an unusual property. NSW assigned people to the programme at random, so a randomised estimate exists. Lalonde (1986) asked whether non-experimental comparison groups, here drawn from the Panel Study of Income Dynamics (PSID), reproduce it. Here the programme's participants are compared with PSID respondents who never had the chance to enter it. The observational analysis targets the same estimand as the experiment, the effect on the participants.

## Where the data come from

The manifest [`examples/data/nsw_psid.toml`](https://github.com/iridae-dev/antecedent/blob/main/examples/data/nsw_psid.toml) records the origin, citations, version, per-file SHA-256, retrieval date, preprocessing and columns. The publisher states no licence, so Antecedent ships no copy: `examples/data/fetch.py` downloads the files on request and refuses any that fail their checksum. Without verified files the example runs on a synthetic stand-in with the same ten columns and sizes. The [data README](https://github.com/iridae-dev/antecedent/blob/main/examples/data/README.md) describes the cache, the offline path and the `ANTECEDENT_DATA=download` switch.

The stand-in is a simulation. Its effect on the treated, the average over participants of an effect that varies with schooling (1,000 dollars plus 100 per year below 11 years), is a fixture truth of the generator, useful because it lets the run check that the method recovers a known value. It says nothing about the real programme.

## Declare the graph

You declare that age, schooling, race, ethnicity, marital status, degree status and the two pre-programme earnings each may affect both entering the programme and 1978 earnings, and that the programme may affect 1978 earnings. Nothing else is declared. This is the assumption of no unmeasured confounding given those eight variables. Antecedent does not learn it from the table, and no diagnostic below can confirm it.

## When the comparison group does not overlap

Before the main analysis, the example weights the whole PSID comparison group. On the real files that request refuses. Most PSID respondents were older, more educated and earning far more than anyone who entered the programme, so the propensity model gives them a probability of entry within 1e-8 of zero. Weighting would divide by those probabilities. Antecedent stops instead of reporting a number that rests on them, and the refusal says why: the fitted scores are too extreme to weight.

The refusal also says what would help. A comparison row with essentially no chance of treatment carries no information about what the participants would have earned without the programme. The example therefore keeps every participant and only the comparison rows whose age, schooling and 1974 and 1975 earnings fall inside the participants' range. The rule reads pre-programme covariates only, never the outcome. Because every participant stays, the target is unchanged: it is still the effect on the participants. On the real files 1,479 of the 2,490 comparison rows remain. This trimming is a choice you make and report; a different reasonable rule gives a different comparison group and can give a different answer.

## Prepare and inspect

The example prepares the analysis and prints its inspection before estimating. Identification comes first: with this graph the programme effect is identified by adjusting for the eight covariates, an identification that holds only if the declared graph does. Then overlap. The query declares the treated population as its target, so the comparison rows are reweighted toward the participants. Programme participants were selected for low earnings, so few comparison rows resemble them; `overlap_ess` reports the effective number of rows the weights rely on, and the smallest estimated propensity shows how extreme the weights are.

## Estimate with uncertainty

Inverse-probability weighting estimates the effect, with a bootstrap standard error. The example prints a Wald interval and `result.calibration.status`. Read that status before relying on the interval's coverage: an interval is calibrated only when a matching measured record applies. A second estimator, doubly robust AIPW, uses the same declaration as a cross-check. Both fit their nuisance models on the whole sample, without cross-fitting.

## Probe the assumption

The cheap refuter suite runs with the estimate. The example also computes an E-value: how strongly an unmeasured confounder would have to be associated with both entering the programme and earnings to explain the estimate away. The E-value is a range of assumptions, not a probability that confounding is absent; the example prints its calibration field.

<!-- example: examples/python/observational_study.py -->

## Compare with the experiment

On the real files the example also prints the randomised NSW difference in means, about 1,794 dollars with a standard error of about 671. It estimates the effect on the participants, the same estimand as the observational analysis, so the numbers are directly comparable. On the trimmed comparison group, weighting gives about 747 and AIPW about 1,858.

Read that carefully. The two observational estimators use the same graph, the same rows and the same target, yet they differ by more than a thousand dollars, because the overlap that remains is thin: the weights rest on about 155 effective rows. That AIPW lands near the experimental number does not make it the right method, and the distance of the weighting estimate does not show which assumption failed. Dehejia and Wahba (1999) and later work found that results on these data depend strongly on the comparison group and the method. The lesson is about assumptions and overlap, not a score for an estimator.

## A boundary that teaches

Compare the unadjusted difference with the adjusted estimate. If they differ a great deal, selection into the programme on observed earnings explains it. What the table cannot show is selection on something unobserved, such as motivation, that affects both entry and earnings. Overlap is a separate limit, and on these data it is the binding one: where no comparison row resembles a participant, no weighting recovers the effect there, and the estimate leans on a few rows. Better evidence for either limit is a measured confounder, a design that creates variation unrelated to confounders, or an experiment, which is what NSW supplied.

## What was established and what comes next

The computation established an identification status under a declared graph, an effect estimate with a sampling interval, and how much of the sample carries that estimate. On the stand-in it also showed that the method recovers a known effect on the treated when the graph is true. It did not establish the effect of the real programme as causal truth. The next question is whether the result survives a stronger comparison group or an independent design; see [descriptive comparison](../guides/descriptive-comparison.md) and [target-native validation](../guides/target-native-validation.md) for related routes.
