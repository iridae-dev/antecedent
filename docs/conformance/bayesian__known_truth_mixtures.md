# known_truth_mixtures

**Suite path:** `conformance/bayesian/known_truth_mixtures`

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
unidentified rather than being dropped or upgraded by a prior.  Pulse,
single-step Sustained, and two-step Sustained `[-2, -1]` all have effect 0.9,
conditional on the identified atom (the earlier sustained time has no path),
and all retain 0.3 unidentified mass. Multi-step cheap/full refuse rather than
collapsing to the last step.

The temporal mediation fixture uses the lag-one linear SEM
`M_t = 0.8 T_{t-1}`, `Y_t = 0.25 T_{t-1} + 0.55 M_t` (same series as
`temporal_mediation_grid`). The identified atom is that template (weight 0.7);
the second atom adds `T_{t-1} -> T_t` and is `NotCertified` (weight 0.3).
Each atom uses that atom's `I(h)` cache; adjustment sets are not unioned.
The reported mediated envelope is `0.44` with `0.3` unidentified mass retained.
Priors do not upgrade the unidentified atom.

The consuming tests construct the posterior atoms directly, then execute both
`Study::run()` and `Study::prepare()` followed by the prepared estimate and
same-schema refresh methods.  Thus the fixture pins the structural mixture, not
a discovery smoke envelope or agreement between two Antecedent paths.

## Expected summary

Top-level keys: `static_average_effect, temporal_effect, temporal_mediation, temporal_sustained_multistep, tolerance_class` (5 fields).
