# Release-boundary composition fixture

One end-to-end scenario whose numbers are checkable by hand, run through public
Rust and Python APIs only (`crates/antecedent/tests/composition_release_fixture.rs`,
`python/tests/test_composition_release_fixture.py`). `expected.json` holds the
declared inputs, the hand-derived values and the refusals every mutation must
produce; both test suites read it, and `consumer.py` is the fresh-process
consumer used by the Python suite.

## Scenario

A binary exposure is recorded with misclassification (sensitivity 0.9,
specificity 0.8). Of 1000 respondents 410 are recorded exposed, so the
corrected prevalence is

    pi = (0.41 + 0.8 - 1) / (0.9 + 0.8 - 1) = 0.3.

Two disjoint studies report the exposure's risk difference on the outcome:
0.10 and 0.20, each with variance 0.0004 (500 rows each). The inverse-variance
pool is 0.15 with variance 0.0002. With a declared baseline risk of 0.10 the
composed finite model has

    P(Y = 1 | do(X = 0)) = 0.10,  P(Y = 1 | do(X = 1)) = 0.25,
    P(Y = 1) = 0.7 * 0.10 + 0.3 * 0.25 = 0.145,

so the attributable excess risk is `0.145 - 0.10 = 0.3 * 0.15 = 0.045`.

The downstream decision reads that mean through a caller-declared mean source.
`fund_programme` has utility `1000 * excess - 30 = 15`; `hold` has utility 0.
`fund_programme` is uniquely optimal.

New observations enlarge study B to 600 rows with summary 0.07 and variance
0.0002. The pool becomes `(0.10/0.0004 + 0.07/0.0002) / (1/0.0004 + 1/0.0002)
= 600 / 7500 = 0.08`, the excess risk 0.024 and `fund_programme` is worth -6:
the decision flips to `hold`, but only after the claim graph classified every
downstream claim `recompute` and the pool was recomputed with the replaced
result named.

## Shared evidence and dependence

A third study that re-uses 200 of study A's rows is refused when declared
independent (`evidence_double_counted`) and accepted with a declared covariance
of 0.0002: the pool is 0.15 with variance `(0.0004 + 0.0002) / 2 = 0.0003`,
counted over 1000 declared and 800 distinct observations. A vendor claim with
undeclared observations is an unknown overlap and refuses.

## Point-only

Nothing here ships an interval, a posterior decision or a ranking guarantee.
Every uncertainty request is refused with `cell_not_licensed`; calibration is
unmeasured. The variance of a pooled estimate is a labelled point quantity.
