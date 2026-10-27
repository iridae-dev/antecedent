# Update a joint response with experiments and designs

Declare the design that assigned whole programs, then use an experiment's result to update a retained joint response and decide what to study next. A fitted curve can evaluate any schedule; whether the evaluation is the causal effect of the whole program depends on how programs were assigned. This page covers the designs that identify whole-program contrasts, and how an experiment on those quantities updates a retained Bayesian joint law.

Both start from the objects in [fit and compare coordinated intervention programs](joint-intervention-responses.md). The conceptual route from an experiment to a next study is in the [experiment-to-next-study tutorial](../tutorials/experiment-to-next-study.md).

## Declare a whole-program design

`prepare_native_joint_design(recipe)` compiles a design through its own estimator, so the target keeps its meaning (population, history, cohort or structural feature) rather than being converted into an average effect. The recipe's `kind` is one of `randomized`, `switchback`, `longitudinal`, `gaussian_longitudinal`, `instrumented`, `instrumented_local`, `difference_in_differences` and `synthetic_control`. Source records and every design premise are explicit; `execute(inference)` keeps the identification proof with a compatible statistical procedure. `prepare_joint_design` is the corresponding entry for randomized and longitudinal recipes over the shared joint target; `prepare_joint_design_for_panel` attaches a design to a prepared panel.

- **Randomized whole-program assignment** identifies each program's response by known positive assignment probabilities, as in the [joint intervention tutorial](../tutorials/joint-interventions.md). Randomizing each period does not randomize a whole trajectory.
- **Continuous observational histories** use explicit conditional mechanisms, not categorical discretization. Covariate and outcome formulas may depend on recorded histories and treatment-affected covariates. The compiler checks causal ordering, exact intervention schedules, joint policy density rank and declared support. The full episode bootstrap refits all mechanisms and policy support and keeps each failed slot; the Bayesian counterpart uses proper conditional Gaussian and inverse-gamma priors and shares one complete mechanism draw across all regimes.
- **Instrumented structural features** keep the declared homogeneous structural moment as the target. `instrumented_local` keeps an encouragement-specific complier effect under exclusion and monotonicity and does not need a homogeneous response model. Frequentist inference uses the paired intention-to-treat and receipt Wald ratio and withholds its normal interval for weak support. The Bayesian version keeps paired empirical score-law ratio, numerator and denominator draws; these are Bayesian-bootstrap weights, not a principal-strata likelihood, and a finite sampled first-stage check is not a global support certificate. No claim is made that ratio posterior moments exist, so any downstream expected-utility update must establish the moments it needs.

A design result exports its observations, physical schedules, population, assumptions, prior, seed and full joint law. `replay_joint_design` and `replay_native_joint_design` return historical numbers with no current authority; an independently prepared target can consume the artifact as current.

## Update the response with an experiment

An experiment measures a quantity that may not be the response's own estimand. Updating requires a declared estimand bridge (a licensed `BridgeContract` from the experiment's population, window, dose or duration to the retained contrast, with its transport premise), the mapping's uncertainty and dependence on the fitted source, and the evidence origins in an `EvidenceLedger`. The update reweights complete source and mapping draw slots of a frozen Bayesian joint law; it does not refit.

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

This is a signature sketch, not a runnable program; see the [experiment-to-next-study tutorial](../tutorials/experiment-to-next-study.md) for a complete, executing flow. `experiment_quantity` returns the exact compiled quantity the bridge must name. Terminal means, window averages and explicit weighted time sums keep their exact projected quantity. A changed population, dose, duration or window needs the explicit estimand translation and transport premise in the bridge; the numeric affine map is an input model, not an assumed scaling law. A same-population horizon extrapolation still needs that premise. Removing the translation, naming another experimental population or supplying a transport premise for another target refuses the update.

A frequentist sampling law cannot enter this Bayesian update, and the update carries conditional scope even when the original design identified its estimand.

### Native designs

For switchback histories, structural IV feature contrasts, cohort DiD effects and donor-reference rollout contrasts, `NativeJointDesignResult.experiment_quantity` and `condition_experiment` bind a `NativeExperimentBinding`: the exact linear functional of the retained native coordinates, with their original response units. An IV structural contrast stays structural, a cohort effect stays cohort-specific and a donor contrast keeps its treated-unit and reference interpretation through the update, decision and future-study ranking. Antecedent does not manufacture population world means to fit a different interface.

An encouragement-specific local IV ratio keeps its complier target, and its unconditioned ratio law does not establish an expected value. Conditioning on a Gaussian experimental likelihood with known positive variance needs `local_ratio_regularization` in the binding: global bounds on the mapping intercept and slope and the substantive bounded-support assumption. Each retained paired draw is checked against those bounds. Missing or contradicted bounds refuse the update.

### Decide, then rank further studies

`PanelExperiment.decide(contract, ledger)` evaluates a decision contract on the updated law. `rank_next_samples` values additional samples of the observed estimand; `rank_next_studies` values candidate studies, each optionally in its own regime (another population, window, dose, duration, exposure regime, recency or spend), described by a `PanelFutureStudy` with a `PanelFutureRegime` carrying its own licensed bridge, paired mapping and cost. The current posterior is reweighted by each study's likelihood, your terminal utility is evaluated on every whole response row, and the net value is the gain over deciding now minus the declared cost. An unlicensed, unaligned or stale regime refuses the ranking; it is never replaced by the observed estimand.

For deterministic operational restrictions, derive the terminal action domain from the constrained decision's admissible actions before ranking, so excluded actions stay excluded in the future-data calculation. Receipts carry `admissible_action_identity`, computed from the serialized action domain (names, intercepts and coefficients), so changing an action cannot inherit another domain's identity. When one action remains, the information value is exactly zero.

Extracted experimental sources require current ledger authority for authorization, decisions and search; withdrawing or superseding evidence can refuse current use. `export(ledger)` keeps the original numerical law, bridge, likelihood and historical origins. `replay_design_experiment` and `replay_native_experiment` return historical numbers; `verify_current` binds your original scientific source and current ledger.

## Check coverage of the procedures

`joint_panel.experiment_acceptance_matrix()` returns the machine-readable list of design, inference, mapping, translation and decision cases the update workflow must handle. A row is an obligation, not a passing result, and an outstanding implementation stays visible as a gap in its row. Neither the matrix nor an execution of a case measures statistical calibration.

## Related

- [Fit and compare coordinated intervention programs](joint-intervention-responses.md)
- [Search and certify intervention programs](joint-program-search.md)
- [Scientific bridges](../scientific-bridges.md) and [evidence ledger](../evidence-ledger.md)
- [From a new experiment to the next study](../tutorials/experiment-to-next-study.md)
