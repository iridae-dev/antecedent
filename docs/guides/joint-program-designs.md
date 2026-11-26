# Update a joint response with experiments and designs

Declare the design that assigned whole programs. Then use an experiment's result to update a retained joint response and decide what to study next. A fitted curve can evaluate any schedule. Whether that evaluation is the causal effect of the whole program depends on how programs were assigned. This page covers the designs that identify whole-program contrasts. It also covers how an experiment on those quantities updates a retained Bayesian joint law.

Both start from the objects in [fit and compare coordinated intervention programs](joint-intervention-responses.md). The conceptual route from an experiment to a next study is in the [experiment-to-next-study tutorial](../tutorials/experiment-to-next-study.md).

## Declare a whole-program design

Call `prepare_native_joint_design(recipe)`. The design keeps its own estimator. The target keeps its meaning: population, history, cohort or structural feature. It is not converted into an average effect. The recipe's `kind` is one of `randomized`, `switchback`, `longitudinal`, `gaussian_longitudinal`, `instrumented`, `instrumented_local`, `difference_in_differences` and `synthetic_control`. Source records are explicit. Every design premise is explicit. `execute(inference)` keeps the identification proof with a compatible statistical procedure. Call `prepare_joint_design` for a randomized or longitudinal recipe over the shared joint target. Call `prepare_joint_design_for_panel` to attach a design to a prepared panel.

- **Randomized whole-program assignment** identifies each program's response by known positive assignment probabilities, as in the [joint intervention tutorial](../tutorials/joint-interventions.md). Randomizing each period does not randomize a whole trajectory.
- **Continuous observational histories** use explicit conditional mechanisms, not categorical discretization. Covariate and outcome formulas may depend on recorded histories and treatment-affected covariates. The check covers causal ordering, exact intervention schedules, joint policy density rank and declared support. The full episode bootstrap refits all mechanisms and policy support. It keeps each failed slot. The Bayesian counterpart uses proper conditional Gaussian and inverse-gamma priors. It shares one complete mechanism draw across all regimes.
- **Instrumented structural features** keep the declared homogeneous structural moment as the target. `instrumented_local` keeps an encouragement-specific complier effect under exclusion and monotonicity. It does not need a homogeneous response model. Frequentist inference uses the paired intention-to-treat and receipt Wald ratio. It withholds its normal interval for weak support. The Bayesian version keeps paired empirical score-law ratio, numerator and denominator draws. These are Bayesian-bootstrap weights, not a principal-strata likelihood. A finite sampled first-stage check is not a global support certificate. No claim is made that ratio posterior moments exist. Any downstream expected-utility update must establish the moments it needs.

A design result exports its observations, physical schedules, population, assumptions, inference declaration and full joint law. An explicit bootstrap or posterior declaration retains its seed and other controls. For a Gaussian longitudinal design, a bare `frequentist` inference declaration runs 64 whole-episode bootstrap refits with seed 0. It does not use the execution seed. It does not guarantee Monte Carlo precision. Its law retains the replicate and failure slots. The bare inference declaration does not serialize a seed field. Use the `gaussian_longitudinal_bootstrap` inference declaration with explicit `replicates` and `seed` to select those controls. `replay_joint_design` returns historical numbers with no current authority. `replay_native_joint_design` does the same. An independently prepared target can consume the artifact as current.

## Update the response with an experiment

An experiment measures a quantity that may not be the response's own estimand. Updating requires a declared estimand bridge: a supported `BridgeContract` from the experiment's population, window, dose or duration to the retained contrast, with its transport premise. It also requires the mapping's uncertainty and its dependence on the fitted source. It requires the evidence origins in an `EvidenceLedger`. The update reweights complete source and mapping draw slots of a frozen Bayesian joint law. It does not refit.

```python
experiment = result.condition_experiment(
    bridge=bridge,                      # a BridgeContract
    ledger=ledger,                      # an EvidenceLedger holding the evidence origins
    observed_mean=observed,
    observation_variance=variance,
    sample_size=n_experiment,
    mapping_dependency=dependency,      # the fitted source the mapping depends on
    intercepts=intercepts,              # paired per-draw affine mapping
    slopes=slopes,
    model="gaussian experiment likelihood",
    experiment="geo test",
    mapping="dose to response",
    functional="terminal_mean",
)
```

This is a signature sketch, not a runnable program. See the [experiment-to-next-study tutorial](../tutorials/experiment-to-next-study.md) and its script [`experiment_to_next_study.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/experiment_to_next_study.py) for a complete, executing flow. `experiment_quantity` returns the exact quantity the bridge must name. Terminal means, window averages and explicit weighted time sums keep their exact projected quantity. A changed population, dose, duration or window needs the explicit estimand translation and transport premise in the bridge. The numeric affine map is an input model, not an assumed scaling law. A same-population horizon extrapolation still needs that premise. Removing the translation refuses the update. Naming another experimental population refuses the update. Supplying a transport premise for another target refuses the update.

A frequentist sampling law cannot enter this Bayesian update. The update carries conditional scope even when the original design identified its estimand.

### Native designs

For switchback histories, structural IV feature contrasts, cohort DiD effects and donor-reference rollout contrasts, call `NativeJointDesignResult.experiment_quantity`. Call `condition_experiment`. Both bind a `NativeExperimentBinding`. The binding is the exact linear functional of the retained native coordinates, with their original response units. An IV structural contrast stays structural. A cohort effect stays cohort-specific. A donor contrast keeps its treated-unit and reference interpretation through the update, the decision and the future-study ranking. Antecedent does not manufacture population world means to fit a different interface.

An encouragement-specific local IV ratio keeps its complier target. Its unconditioned ratio law does not establish an expected value. Conditioning on a Gaussian experimental likelihood with known positive variance needs `local_ratio_regularization` in the binding. That declaration is global bounds on the mapping intercept and slope, and the substantive bounded-support assumption. Each retained paired draw is checked against those bounds. Missing or contradicted bounds refuse the update.

### Decide, then rank further studies

Call `PanelExperiment.decide(contract, ledger)`. It evaluates a decision contract on the updated law. `rank_next_samples` values additional samples of the observed estimand. `rank_next_studies` values candidate studies. Each study may sit in its own regime: another population, window, dose, duration, exposure regime, recency or spend. Describe it with a `PanelFutureStudy` and a `PanelFutureRegime`. The regime carries its own supported bridge, paired mapping and cost. The current posterior is reweighted by each study's likelihood. Your terminal utility is evaluated on every whole response row. The net value is the gain over deciding now minus the declared cost. An unsupported, unaligned or stale regime refuses the ranking. It is never replaced by the observed estimand.

For deterministic operational restrictions, derive the terminal action domain from the constrained decision's admissible actions before ranking. Excluded actions stay excluded in the future-data calculation. Receipts carry `admissible_action_identity`, computed from the serialized action domain (names, intercepts and coefficients). Changing an action cannot inherit another domain's identity. When one action remains, the information value is exactly zero.

Extracted experimental sources require current ledger authority for authorization, decisions and search. Withdrawing or superseding evidence can refuse current use. `export(ledger)` keeps the original numerical law, bridge, likelihood and historical origins. `replay_design_experiment` returns historical numbers. `replay_native_experiment` returns historical numbers. `verify_current` binds your original scientific source and current ledger.

## Check coverage of the procedures

Call `joint_panel.experiment_acceptance_matrix()`. You get the machine-readable list of design, inference, mapping, translation and decision cases the update must handle. A row is an obligation, not a passing result. An outstanding implementation stays visible as a gap in its row. Neither the matrix nor an execution of a case measures statistical calibration.

## Related

- [Fit and compare coordinated intervention programs](joint-intervention-responses.md)
- [Search and certify intervention programs](joint-program-search.md)
- [Scientific bridges](../scientific-bridges.md) and [evidence ledger](../evidence-ledger.md)
- [From a new experiment to the next study](../tutorials/experiment-to-next-study.md)

For a complete marketing investigation using a shared panel fit, see [beyond the marketing mix](../tutorials/beyond-marketing-mix.md).
