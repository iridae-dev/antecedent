# Checked IV recalculation AR diagnostic oracle

**Suite path:** `conformance/recalculation/design_ar`

The observed DAG contains Z→T, U→T, U→Y and T→Y. Independent Z is Bernoulli(1/2), and U,V,E are independent standard normals. T=strength*Z+U+.5V and Y=2T+.8U+.5E. The target coefficient is 2. Source identification retains only Z as excluded instrument. The existing checked Wald-functional lowering uses an intercept and Z; it does not include U as an additional regressor. This remains a valid marginal IV effect because Z is independent of U,V,E.

The finite-F constants in expected.json were computed independently of Antecedent with Python standard-library math.lgamma and composite Simpson integration (4096 even panels) of the Student t density from zero to its 0.975 quantile, with 80 bisections in [0,4]. Squaring that quantile gives F(1,n-2)'s 0.95 critical value. No production incomplete-beta, F quantile, normal generator or AR inversion is used to construct these pins.

The harness reads these pins, generates actual iid frames, executes original checked IV fitting, and compares reported intervals/union withholding to the original source engine. An independent centered raw-data quadratic checks AR membership at the known coefficient. Finite, unbounded, union and failed cases remain in the full denominator; no finite-interval-only conditioning is performed. All measurement tests are ignored until the final calibration step. This is a numerical/error-rate diagnostic, not an additional released calibration license.

## Expected summary

Top-level keys: `effect, finite_f_oracles, strengths` (3 fields).
