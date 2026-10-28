# Prediction smoke

Conditional prediction at X_{t-1}=1.0 for the noise-free law Y_t = 0.01 + 2·X_{t-1}
(X_t = 0.5·X_{t-1} + 0.1). The fit is exact, so the mean prediction is 2.01 to
floating-point round-off (tolerance 1e-8); there is no sampling error to budget for.
The fixture is unconfounded and linear, which is the only reason the associational
prediction coincides with the interventional mean here.
