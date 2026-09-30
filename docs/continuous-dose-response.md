# Conditional continuous-dose response utility

`policy.ConditionalDoseResponse`, through
`analyze(data, query=...).continuous_dose_response`, estimates response levels
at caller-specified dose targets within supplied baseline groups. The native
estimator
uses an Epanechnikov kernel, 0.75(1 - u²), centered at each target and inverse supplied density
weights, then reports a Hájek local mean. The caller provides the continuous
dose density evaluated at each observed dose and declares whether it is known
or externally estimated.

Each group/target result reports local row count, effective sample size,
minimum local density, maximum normalized weight, and descriptive local outcome
spread. The route refuses targets without the configured minimum local rows
and rejects non-positive densities. Bandwidth controls the compact local
window. Outcomes and density models are not fit by this route.

Identification requires conditional exchangeability given the supplied
baseline groups, consistency, no interference, a correct density, and adequate
continuous-dose overlap at each target. The result is point-only; local outcome
spread is not a standard error. This describes a conditional dose-response
curve and does not select or evaluate a learned continuous-dose policy value.
