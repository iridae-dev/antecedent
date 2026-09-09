# known_truth_mixtures

Frozen, analytic data-generating processes for the staged graph-posterior
effect paths. The Bayesian aggregator mixes per-atom `bayesian.gcomp` draws;
the Frequentist aggregator mixes per-atom `linear.adjustment.ate` point
estimates. Both report `E[τ | identified]` and retain unidentified mass.

The static fixture uses `Y = 2T + 2Z + epsilon` with a balanced deterministic
factorial design.  The unadjusted DAG therefore has effect 3, while the DAG that
adjusts for `Z` has effect 2.  Their posterior weights are 0.5 and 0.3.  A valid
reverse-causal `Y -> T` atom has weight 0.2, has no admissible backdoor adjustment
under the licensed identifier, and cannot be upgraded to identified by a prior.
The reported effect envelope is `E[tau | identified] = 2.625`; the separate
`unidentified_mass` remains 0.2.

The temporal fixture uses `defect_t = 0.9 * pressure_{t-1}`.  A valid lag-one DBN
atom has weight 0.7.  A second valid DBN atom adds the stationary autoregressive
edge `pressure_{t-1} -> pressure_t` and has weight 0.3.  Its treatment ancestry
continues across every finite unfolding boundary, so `TemporalBackdoorIdentifier`
reaches its finite-history cap and returns `NotCertified`; the atom remains
unidentified rather than being dropped or upgraded by a prior.  Pulse and
single-step Sustained both have effect 0.9, conditional on the identified atom,
and both retain 0.3 unidentified mass.

The consuming tests construct the posterior atoms directly, then execute both
`Study::run()` and `Study::prepare()` followed by the prepared estimate and
same-schema refresh methods.  Thus the fixture pins the structural mixture, not
a discovery smoke envelope or agreement between two Antecedent paths.
