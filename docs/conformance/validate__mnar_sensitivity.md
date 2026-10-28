# Missing-not-at-random sensitivity

**Suite path:** `conformance/validate/mnar_sensitivity`

Known-truth pins for the departure-from-MAR sensitivity analysis of a two-arm mean contrast.

- Pattern-mixture shift: the effect is linear in the shift, `effect(delta) = MAR + delta (l1 f1 - l0 f0)`
  with `f_a` the arm's missing fraction. On the retained data (MAR effect 1.5, `f1 = 0.4`, `f0 = 0.5`) the
  common-loading sign tips at `delta = 15` and, with the treated arm alone departing, at `-3.75`.
- Selection model by exponential tilting: for a binary treated outcome with observed rate one half the
  missing-pattern mean is `expit(theta)`, so the decision "effect above 0.3" tips at `theta = ln(5/3)`.
- A seeded continuous sample evaluates the selection model through the selection mechanism itself
  (`f(y | R=0)` proportional to `f_obs(y) (1 - pi(y)) / pi(y)` with an arbitrary intercept that drops out),
  with the sampling variance taken from a numerical influence function rather than a closed form.

`reference.py` is standard-library Python and imports no Antecedent code. Run
`python3 conformance/validate/mnar_sensitivity/reference.py --check`; `--write` regenerates
`expected.json`, which `crates/antecedent-validate/tests/mnar_sensitivity.rs` compares the analysis against.

## Expected summary

Top-level keys: `cases, closed_forms, level, oracle, schema_version, tolerance_class` (6 fields).
