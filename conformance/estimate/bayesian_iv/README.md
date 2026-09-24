# Bayesian IV joint structural reference

This fixture exercises the public joint Gaussian structural IV model implemented by
`antecedent-estimate::bayesian_iv`. Treatment is generated from one binary instrument and a
Gaussian structural disturbance. The outcome has a fixed unit loading on that same disturbance.
The target treatment coefficient is 2.0.

The joint posterior conditions on known unit stage variances and an isotropic Normal prior.
The fixed disturbance loading is a substantive model restriction. A separate free-loading
control-function prototype overcovered in repeated sampling and is not the public route.

The staged test checks known truth, posterior propagation, and the first-stage F gate.
Separate 300-replicate low-level and public exact-law checks cover the true effect at nominal 95%.
