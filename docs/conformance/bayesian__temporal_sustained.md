# Bayesian temporal sustained conformance

**Suite path:** `conformance/bayesian/temporal_sustained`

Lag-2 SCM `defect_t = 0.7 * pressure_{t-2} + eps_t`, `eps_t ~ N(0, 0.05^2)` i.i.d.,
drawn from a fixed-seed deterministic PRNG (SplitMix64) so the fixture stays exactly
reproducible run to run; conjugate Bayesian g-comp on the unfolded temporal design via
the staged `Study::prepare()` / `PreparedStudy::estimate_series()` path, queried with
`TemporalPolicy::Sustained { from: -2, until: -2 }`.

The noise scale (`sigma=0.05`) is chosen so the posterior/replicate spread is
non-degenerate — large enough for the `data.subset` stability refuter to be a
meaningful check rather than comparing against near-zero floating-point variance —
while staying small enough (about 14x below the 0.05 tolerance, in standard-error
terms) that the posterior mean recovers 0.7 with a comfortable, deterministic margin
at n=400. See `expected.json`'s `derivation` field for the full reasoning.

This is deliberately distinct from `temporal_pulse` (lag 1, `Pulse{at:-1}`):
the graph only has a lag-2 edge, so a test that accidentally reused the Pulse
lag-1 wiring or fixture would fail to identify or would recover the wrong
coefficient. Posterior mean should be within 0.05 of the true coefficient
0.7; finite `P(effect < 0)`; posterior artifact round-trip; prior/posterior
predictive checks present once `refute != None`.

`from == until` (a single-step window) is the only Sustained form the current
`TemporalLinearAdjustment`/`BayesianGComputationAte::from_prepared_estimation`
pipeline supports — `until > from` is refused
(`refuse_multi_step_schedule`, `antecedent-estimate/src/temporal_adjustment.rs`)
because a single-column regression cannot honor the multi-node contrast a
true multi-step Sustained estimand requires. This fixture does not attempt to
exercise multi-step accumulation; that remains unlicensed and untested.

## Expected summary

Top-level keys: `backend, derivation, expected_ate, horizon_steps, n, n_draws, outcome, policy, policy_note, require_artifact_round_trip, require_finite_p_below_zero, require_ppc_present_when_refute_enabled, scm, tolerance, treatment, treatment_lag, true_effect_per_unit` (17 fields).
