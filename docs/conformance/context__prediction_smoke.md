# Prediction smoke

**Suite path:** `conformance/context/prediction_smoke`

Conditional prediction at X_{t-1}=1.0 for Y ← 2·X_{t-1}; mean prediction ≈ 2.
The fixture is unconfounded and linear, which is the only reason the associational
prediction coincides with the interventional mean here.

## Expected summary

Top-level keys: `mean_prediction_target, tol, tolerance_class` (3 fields).
