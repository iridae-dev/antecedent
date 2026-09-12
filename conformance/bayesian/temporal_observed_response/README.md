# Bayesian observed temporal response

The consuming Rust fixture generates X independently N(0,1) and
Y_t = 2 + 1.5 X_(t-1) + .75 X_(t-2) + .5 epsilon_t. Selection is
Bernoulli(logistic(.4 X_(t-1))). Censoring bounds are independent Gaussian
or conditional Gaussian given X_(t-1); left-censored pairs reflect outcome
and bounds. Recorded missing placeholders are deliberately arbitrary.

A pulse at -1 gives horizon-one mean 2+1.5*d+.75*mean(X) and horizon-two
mean 2+.75*d+1.5*mean(X), with empirical exogenous standardization. The
Sequence sets X_-2=1 and X_-1=2, giving 5.75. Gibbs inference uses the
observed Gaussian likelihood, integrating latent outcomes and mechanism
parameters. Observation nuisance likelihood factors out under trajectory
ignorability, distinct parameters and independent nuisance priors.

`crates/antecedent/tests/temporal_observed_bayesian.rs` consumes all five
pairs and the Sequence; `python/tests/test_temporal_observed_bayesian.py`
checks native prepared execution and composite response/posterior artifacts.
The numerical tests establish these fixed-fixture estimates and sampler
diagnostics, not a general repeated-sampling coverage guarantee.
