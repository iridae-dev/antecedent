# Discover latent soft interventions

You prepare a bounded discovery study on a static DAG you supply. Pass two independent arm tables, a randomized recorded action, eligible equations, and measured DAG roots available before execution. The executed version of the action is unobserved. Call `antecedent.latent_soft_intervention.prepare`. In Rust, call `Study::prepare_latent_soft_intervention`. The analysis assumes version assignment is independent of structural disturbances given the declared roots. Randomizing the recorded action supports the comparison between arms. Conditional version exogeneity is a separate premise.

Inspect the plan before you call `execute`. Read the plan’s `support` field ([plan view](plan-view.md)) to see whether your installed build supports this analysis. The [support matrix](../support-matrix.md) explains how support is decided. Forecasts reuse the retained fit. They accept only the declared context. You get the joint Gaussian mixture for the recorded action as delivered, a baseline law at the same context, and a mean difference. Call `forecast.action_marginal(variable)` with one coordinate. Its mean and variance condition on this fitted forecast. They carry its query identity as shared evidence.

Baseline equations are fitted on training controls. The fit then compares one or two execution versions. Each version allows additive shifts to at most two candidate equations. An outcome-independent held-out action partition is used for comparison with a context-dependent Gaussian alternative. The result can prefer a split while leaving the choice of target equations unresolved. The fitted weights are conditional version probabilities. The runner-up target gap is descriptive. Those quantities cannot resolve observational equivalence. They cannot support assigning a particular version.

Forecasts require overlap in both arms' training context. A failed split does not confer support on the restricted single-shift forecast. The call can withhold that forecast when the evaluation data show a material context or covariance difference. Either version count can also fail the held-out check of sequential conditional outcome probabilities. This diagnostic can withhold a forecast even when the split comparison favors two versions. Optional whole-fit resampling preserves the original split. It carries the dependence between baseline equations, intervention shifts, and membership probabilities across the same draws. Draws describe diagnostic sampling spread. There is no confidence interval.

The export includes the source rows. Retain its expectation separately. The replay checks the context, the source and the prepared analysis before fitting. It then reruns the forecast and verifies the output. In Python, replay through `antecedent.load`.

The initial evidence register includes equal downstream means with different targeted equations, a single-version case, a variance-only case, a resource refusal, and joint refit replay. The independent checks are declared in the scientific registry. Whether the analysis is supported comes from those checks' records. Read the plan’s `support` field or the [support matrix](../support-matrix.md).

## Two rollouts with the same headline effect

Imagine a team randomized an onboarding change against its usual flow. The action log records who got the change. In the simulated rollout, one version helps people finish setup and the other improves follow-up. Each adds three points to the sum of the two steps. The mean completion score alone gives little reason to look for two versions.

Supply both arm tables, the steps that could have changed, and the context measured before the rollout. Read `discovery.evidence` and `discovery.hypotheses` together. A fit can favor a split while leaving the changed equations uncertain. A forecast, when admissible, compares the action as delivered with the baseline at the same context.

The example also generates a rollout in which setup is the only step that changes. This is a useful comparison when interpreting an unresolved split. An unresolved fit leaves open how many versions were delivered. The script prints the result for each dataset. It replays any admissible forecast from its saved identity.

Run `python examples/python/hidden_rollout_variants.py` from a checkout. The example prepares and fits both rollout variants in a supported release build.

<!-- example: examples/python/hidden_rollout_variants.py -->
