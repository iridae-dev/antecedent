# Predictive assessment: proper scores, calibration diagnostics and held-out checks

A predictive law is assessed against realized observations. The assessment
reports how the law performed on those observations, under a declared
validation design, and nothing more. It is a model-independent contract, not a
model zoo and not a forecasting application: laws come from native finite and
closed-form families, or are supplied by an external provider and declared
through the provider contract.

A good score is not causal validity, not a probability that a claim holds and
not a guarantee for other observations.

## Boundary

| Item | Supported | Refused |
| --- | --- | --- |
| Law families | Gaussian, Bernoulli, categorical, exact finite real, equally weighted sample, provider-evaluated density and cumulative values, exact finite and sample joint laws | a law that exists only as a callback: `predictive_assessment.law_callback_not_built` |
| Target | each law and the observations name quantity, horizon, population and units | a law and the observations that differ in any of them: `quantity_semantics_mismatch`; no conversion is inferred |
| Observations | distinct identities, finite outcomes of one kind | repeated identities, mixed outcome kinds |
| Design | held-out split, rolling origin, blocked or clustered, simulation stream | a rolling-origin design whose horizon is not the target's |
| Dependence | iid (declared), serial with a truncation lag, cluster | iid for a rolling-origin or blocked design: `sampling_dependence_unknown`; a lag shorter than horizon minus one; one cluster |
| Overlap | held out verified, in sample, unverified, not fitted, decided from row identities | held-out requirement with an in-sample predictor (`evidence_double_counted`) or an undeclared one (`evidence_overlap_unknown`) |
| Uncertainty | point scores, paired differences, a labelled statistic, and the interval of an expected score or score difference inside a measured design | an unscoped interval request, an interval for an in-sample or unverified predictor, an undeclared dependence, a design no coverage record measured, a significance level: `cell_not_licensed` |
| Meaning | a loss of a law on the observations | a causal claim or the probability of a claim: `route_not_supported` |

## Scores

Every score is a loss (smaller is better) and is strictly proper on the laws
it is defined for: the true law has the smallest expected score.

| Score | Defined for | Closed form used as oracle |
| --- | --- | --- |
| log score `-ln p(y)` | Gaussian, Bernoulli, categorical, finite, provider-evaluated | Gaussian `ln sigma + ln sqrt(2 pi) + z^2/2` |
| CRPS | Gaussian, Bernoulli, finite, sample | Gaussian `sigma (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi))`; Bernoulli equals the Brier score |
| Brier | Bernoulli, categorical | `(p - y)^2` (binary), `sum_k (p_k - 1{y = k})^2` |
| energy | joint finite, joint sample | `E|X - y| - E|X - X'|/2` |

A sample is scored as the law it is: the empirical distribution of its draws.
Its CRPS is the energy form `(1/m) sum |x_i - y| - (1/(2 m^2)) sum |x_i - x_j|`,
computed as the integral of the squared difference of distribution functions (a
sum of non-negative terms), and is proper for that law. As an estimator of the
CRPS of the law the draws were taken from, the energy form is biased upward by
`E|X - X'| / (2 m)`. The fair form (`EnsembleCrpsForm::Fair`, divisor
`m (m - 1)`; Ferro 2014) is unbiased for iid draws and is not a proper score of
the empirical law; it is offered as an estimator and labelled as one.

A log score is infinite where the law gives the realized outcome probability
zero; that is refused, never averaged. A sample has no density, so its log
score is refused until a density is supplied.

## Score differences

Two predictors scored on the same observations are paired. The object of
interest is the mean of `d_i = S_A,i - S_B,i`; its variance comes from the
differences, not from the two predictors' separate variances, because they
share the observations.

| Declared dependence | Variance of the mean |
| --- | --- |
| `iid` | `s^2 / n` |
| `serial`, truncation lag `L` | `(gamma_0 + 2 sum_{k=1}^{L} (1 - k/(L+1)) gamma_k) / n` (Bartlett kernel; Newey and West 1987) |
| `cluster` | `G/(G-1) sum_g (S_g - n_g mean)^2 / n^2` over `G >= 2` clusters |

The Diebold-Mariano style statistic `mean / se` is a labelled diagnostic. No
p-value or coverage statement is attached to the report. With no declared
dependence the standard error is withheld; independence is never assumed. A
difference also states how many observations the two predictors' fit sets
share, when both declare them.

## Interval of an expected score

The report describes the assessed cases. The interval answers a different
question: what expected score does a law have under the declared sampling
design, given these cases. It is asked for one predictor's score (or one pair's
difference) at one level.

| Declared dependence | Interval | Reference |
| --- | --- | --- |
| `iid` | `mean +- t * sqrt(s^2 / n)` | Student-t, `n - 1` degrees of freedom |
| `serial`, lag `L`, `(L + 1) / n <= 0.1` | `mean +- t * se` with the Bartlett long-run variance | Student-t, `n / (L + 1) - 1` degrees of freedom |
| `cluster`, at least ten clusters | `mean +- t * se` with the cluster-robust variance | Student-t, `G - 1` degrees of freedom |

It is published only for a predictor verified held out of the cases (or never
fitted) and only where a current coverage record measured the construction at
the design, score, law family (Gaussian here), level (0.95) and number of cases.
The record is read from the shipped coverage table: a declaration cannot open
the route. The measured design is a fixed Gaussian forecast against known
generators, for the CRPS, the log score and the CRPS difference of two
forecasts. The number of cases must lie inside the range measured for the
declared dependence; there is no extrapolation above the largest count or below
the smallest, and a count outside it refuses with `cell_not_licensed`.

| Declared dependence | Measured generator | Cases measured |
| --- | --- | --- |
| `iid` | independent standard normal outcomes | 400 to 1,600 |
| `cluster` | clusters of five cases with a shared cluster effect | 800 to 3,200 (160 to 640 clusters) |
| `serial` | stationary AR(1) outcomes, `phi = 0.5`, lag `floor(n^(1/3))` | 1,600 to 6,400 |

The minimum of ten clusters and the bandwidth limit of the serial interval are
requirements of the construction, not a measured range: a cluster count or
series length that satisfies them but lies outside the measured case counts
still refuses. A serial interval for the CRPS of a single forecast is a named
boundary (0.925 at 6,400 cases) and is not published; the serial log score and
CRPS difference are. A heavy-tailed truth is also recorded as a named boundary
and does not license the route. The interval states its assumptions: cases
independent or dependent as declared, a finite score variance, a law fixed
before the cases were seen, and for a rolling-origin design the average
expected score over the forecast origins.

## Calibration diagnostics

All diagnostics are labelled `diagnostic` with `calibration = unmeasured`: no
coordinate of this route has measured calibration.

- **PIT histogram.** For a discrete or ensemble law each case spreads its unit
  mass uniformly over `[F(y-), F(y)]` (non-randomized PIT; Czado, Gneiting and
  Held 2009), so no randomness enters.
- **Reliability.** Binned forecast probability against observed frequency of a
  binary event.
- **Coverage and sharpness.** The fraction of outcomes inside the central
  prediction interval at each declared level, the mean interval width and the
  mean predictive standard deviation.
- **Drift.** A separate route: the two-sample KS distance and the population
  stability index of the PIT between two windows. Temporal assessments report
  the first against the second half of the cases. No significance level is
  attached.

## Overlap with the data a law was fitted on

Held out means held out of the data the law was fitted or tuned on, decided
from row-level identities. The facade resolves the fit set and the assessed
observations from the evidence ledger (derivation lineage applied) with
`fit_declaration_from_ledger` and `observation_keys_from_ledger`.

| Status | Meaning |
| --- | --- |
| `held_out_verified` | the declared fit set is disjoint from the assessed observations |
| `in_sample` | some assessed observations were used to fit the law; labelled, or refused when held-out observations are required |
| `unverified` | what the law was fitted on is not declared (the usual case of a provider law); never promoted to held out |
| `not_fitted` | the law was fixed without data |

## Exact finite-law scoring

When the generating law is finite, `exact_expected_score` returns the exact
expected score of a forecast: a finite sum, with no sampling, so no sampling
interval exists or is reported.

## Bayesian predictive checks

`PosteriorPredictiveCheck::replicates` returns the replicates its discrepancy
statistics summarize (same parameter draws and noise stream). Each row's
replicates are a sample law scored and checked through this contract, so there
is one scoring surface. The replicates are drawn at parameters fitted to the
observed rows: declare the fit set accordingly, and an assessment of the rows
the posterior was conditioned on is `in_sample`.

## Artifact

`predictive_assessment_v1` stores the laws, the observations (in canonical case
order), the declarations and the results. A consumer re-scores every case
(bit for bit on one platform; within `1e-12` relative across platforms, where
`exp`, `ln` and `erfc` may differ in the last bits) and recomputes every
identity. A resealed edit of a law, an outcome, a declaration or a stored value
is refused, and a consumer that retained the identity independently refuses a
consistently resealed assessment of other laws or observations. The artifact
carries no interval and states `point_only`, `unmeasured` and
`not_established` (causal validity).

## Relation to the scientific bridge

The refusal to read a model score as a probability that a claim holds is the
one the scientific bridge states for the same request. Here it is
`predictive_assessment.score_not_probability`; reading a score as causal
validity is `predictive_assessment.not_a_causal_claim`. Causal validity needs
an identification argument and refutation; the probability of a claim needs a
declared prior and likelihood.

## Python

`antecedent.predictive_assessment` builds the declarations and presents the
results; every score, identity and refusal is Rust's.

```python
from antecedent import predictive_assessment as pa

target = pa.Target("outcome", "cohort", "mmHg")
observed = pa.Observations(target, pa.keys("cohort", range(4)), "real", [0.0, 1.0, 0.0, 1.0])
result = pa.assess(
    observed,
    [pa.Predictor("standard_normal", [pa.Gaussian(0.0, 1.0)] * 4, target,
                  fit=pa.FitDeclaration.not_fitted())],
    pa.ValidationDesign.simulation_stream("seeded generator"),
    scores=("crps", "log_score"),
    dependence=pa.Dependence.iid(),
)
result.report.predictor("standard_normal").score("crps").mean   # 0.4180681674...
artifact = result.export()
pa.consume_predictive_assessment_artifact(artifact, result.identity)
```

`pa.keys(domain, rows)` returns the evidence ledger's observation identities, so
`pa.FitDeclaration.from_ledger(ledger, sources)` and `pa.ledger_observations`
decide held out from the same rows. `result.score_interval("forecast", "crps")`
(and `reference=`, `level=`) returns the interval inside a measured design and
raises `cell_not_licensed` elsewhere; `result.score_interval()` with no
predictor is the unscoped request and always raises. `result.causal_validity()`
and `result.score_as_probability()` raise the typed refusals of the closed
routes. `pa.replicate_laws` turns posterior predictive
replicates into sample laws for the same scoring surface.

## Closed routes

| Route | Reason | What would license it |
| --- | --- | --- |
| unscoped interval request, or a scoped one outside a measured design | `cell_not_licensed` | name the predictor, score and level, declare the dependence, and assess a held-out or unfitted predictor inside a design a current coverage record measured || callback law | `route_not_supported` | the law as a native family, or as provider-evaluated values declared through the provider contract |
| score as causal validity or as a probability | `route_not_supported` | an identification argument and refutation; a declared prior and likelihood |

## References

- Gneiting and Raftery (2007), Strictly Proper Scoring Rules, Prediction, and Estimation, JASA 102(477).
- Brier (1950), Verification of Forecasts Expressed in Terms of Probability, Monthly Weather Review 78(1).
- Matheson and Winkler (1976), Scoring Rules for Continuous Probability Distributions, Management Science 22(10).
- Ferro (2014), Fair Scores for Ensemble Forecasts, QJRMS 140.
- Diebold and Mariano (1995), Comparing Predictive Accuracy, JBES 13(3).
- Newey and West (1987), A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix, Econometrica 55(3).
- Gneiting, Balabdaoui and Raftery (2007), Probabilistic Forecasts, Calibration and Sharpness, JRSSB 69(2).
- Czado, Gneiting and Held (2009), Predictive Model Assessment for Count Data, Biometrics 65(4).
- Dawid (1984), Statistical Theory: The Prequential Approach, JRSSA 147(2).
