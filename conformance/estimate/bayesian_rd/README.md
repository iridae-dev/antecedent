# Bayesian sharp RD reference

This fixture checks the posterior jump under the fixed-bandwidth local-linear Gaussian model
`Y ~ 1 + T + (R-c) + T(R-c)` with sharp assignment `T=1{R>=c}`. The synthetic outcome has
a known jump of 2.5 at cutoff zero. The estimator conditions on its declared rectangular
bandwidth and plug-in residual variance. Its claim is limited to this model and bandwidth.

The paired unit test checks known truth, refusal when the sharp-assignment rule fails,
bandwidth sensitivity, and prior sensitivity.
