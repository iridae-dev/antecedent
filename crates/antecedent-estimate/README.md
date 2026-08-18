# antecedent-estimate

Estimators for identified causal functionals. Estimators consume an
`IdentifiedEstimand` — they never choose confounders or assert
identifiability.

Frequentist surface: linear and GLM adjustment, g-computation, propensity
methods (IPW, matching, AIPW), instrumental variables (Wald, 2SLS),
front-door two-stage, sharp regression discontinuity, and temporal
adjustment, mediation, and prediction. Continuous causal responses
(Kennedy-style doubly robust curves, derivatives, elasticities) live in
`response`. Bayesian estimation covers g-computation, HMC GLMs, prior
transfer, and graph-by-effect posterior envelopes.

Observation mechanisms (`observation`), trial-to-target transport
(`transport`), and randomized interference (`interference`) are explicit
stage APIs: they change what identifies the estimand and are never inferred
from the data.

Standard errors are analytic or bootstrap; overlap diagnostics and clipping
policies are reported, not silent. See
[docs/capabilities.md](https://github.com/iridae-dev/antecedent/blob/main/docs/capabilities.md)
for the full estimator inventory.
