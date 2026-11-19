# Discover latent soft interventions

The native compiler can prepare a bounded discovery study on a supplied static DAG. The inputs are two independent arm tables, a randomized recorded action, eligible equations, and measured DAG roots available before execution. The executed version of the action is unobserved. The analysis assumes version assignment is independent of structural disturbances given the declared roots. Randomizing the recorded action supports the comparison between arms. Conditional version exogeneity is a separate premise.

Use `antecedent.latent_soft_intervention.prepare` in Python or `Study::prepare_latent_soft_intervention` in Rust. Inspect the lowered plan before calling `execute`. Forecasts reuse the retained fit and accept only the declared context. They give the joint Gaussian mixture for the recorded action as delivered, a baseline law at the same context, and a mean difference. The public result reports the current release-support status. Before the capability earns support, release builds let callers inspect the plan and refuse execution. Development builds can execute the plan and report the evidence refusal they bypassed. Use `forecast.action_marginal(variable)` to pass one coordinate to the existing known-weight propagation route. Its mean and variance condition on this fitted forecast and carry its query identity as shared evidence.

The algorithm fits baseline equations on training controls. It then compares one or two execution versions, each allowing additive shifts to at most two candidate equations. An outcome-independent held-out action partition is used for comparison with a context-dependent Gaussian alternative. The result can prefer a split while leaving the choice of target equations unresolved. The fitted weights are conditional version probabilities, and the runner-up target gap is descriptive. Those quantities cannot resolve observational equivalence or support assigning a particular version.

Forecasts require overlap in both arms' training context. A failed split does not confer support on the restricted single-shift forecast. The route can withhold that forecast when the evaluation data show a material context or covariance difference. Either version count can also fail the held-out check of sequential conditional outcome probabilities. This diagnostic can withhold a forecast even when the split comparison favors two versions. Optional whole-fit resampling preserves the original split and carries the dependence between baseline equations, intervention shifts, and membership probabilities across the same draws. Draws describe diagnostic sampling spread without a confidence interval.

The export includes the source rows. Retain its expectation separately. The native consumer checks the context, source, and compiled program before fitting, then reruns the forecast and verifies the output. Python can dispatch the replay through `antecedent.load`.

The initial evidence register includes equal downstream means with different targeted equations, a single-version case, a variance-only case, a resource refusal, and joint refit replay. The independent checks are declared in the scientific registry and use its generated lane filter. For the audit, the evidence-only support snapshot lets the scientific lane exercise the gate without asserting a passing release record. Release support is derived after the independent checks earn their records.

## Two rollouts with the same headline effect

Imagine a team randomized an onboarding change against its usual flow. The action log records who got the change. In the simulated rollout, one version helps people finish setup and the other improves follow-up. Each adds three points to the sum of the two steps, so the mean completion score alone gives little reason to look for two versions.

Supply both arm tables, the steps that could have changed, and the context measured before the rollout. Read `discovery.evidence` and `discovery.hypotheses` together. A fit can favor a split while leaving the changed equations uncertain. A forecast, when admissible, compares the action as delivered with the baseline at the same context.

The example also generates a rollout in which setup is the only step that changes. This is a useful comparison when interpreting an unresolved split. An unresolved fit leaves open how many versions were delivered. The script prints the result for each dataset and replays any admissible forecast from its saved identity.

Run `python examples/python/hidden_rollout_variants.py` from a checkout. Use a development build while execution awaits release support.

<!-- example: examples/python/hidden_rollout_variants.py -->
