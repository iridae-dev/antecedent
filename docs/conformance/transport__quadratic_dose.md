# Exact quadratic dose mean

**Suite path:** `conformance/transport/quadratic_dose`

Oracle kind: `closed_form`. m(d)=1+.5d+.25d², m'(d)=.5+.5d and m(3)-m(1)=3. Rust/Python fixtures span levels, derivatives, contrasts and admissible fixed bandwidths. The local polynomial design exactly spans this conditional mean; weighted least squares reproduces it in conditional expectation, so smoothing bias is zero under the explicitly declared model. Rows and residuals cannot authenticate that assumption.

Run `cargo test -p antecedent-estimate --test dose_grid_functional --offline`, `cargo test -p antecedent-io --test dose_grid_artifact --offline`, and `cd python && uv run pytest tests/test_dose_grid.py`. Independent artifacts bind the added premise and required feature; old omitted-field requests preserve their original digest. Mutations refuse. Sampling sandwich uncertainty remains unmeasured; this is not a coverage oracle or a general bias correction.

## Expected summary

Top-level keys: `conditional_mean_coefficients, contrast, from, oracle_kind, to` (5 fields).
