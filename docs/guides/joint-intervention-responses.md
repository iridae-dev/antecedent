# Fit and compare coordinated intervention programs

Compare several coordinated interventions, such as a channel mix held over a horizon. Take the uncertainty from one shared fitted law. Name the population, the outcome and the intervention worlds. Name the reference, the time horizon and the initial history. The result keeps that question with its identification status, or with an explicit model-conditional justification. It also keeps the numerical execution and the resource plan.

If you have not seen a joint comparison before, start with the [joint intervention tutorial](../tutorials/joint-interventions.md), which builds one on simulated data with a known answer. For what a world is, see [interventions and worlds](../concepts/interventions-and-worlds.md). This page is the task reference for the panel and static joint workflows in `antecedent.joint_panel`. Choosing among programs is covered in [search and certify intervention programs](joint-program-search.md); experimental updates and whole-program designs are in [update a joint response with experiments and designs](joint-program-designs.md). Joint binary interventions on cross-sectional data are in [joint cells](joint-cells.md).

Runnable examples: [`joint_intervention_program.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/joint_intervention_program.py) compares a two-channel program with its baseline over a 24-week horizon on a simulated panel with a known terminal-week effect, and [`joint_panel_workflow.rs`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/joint_panel_workflow.rs) runs the same kind of question from Rust.

## Declare channels and interactions

The panel family supports an arbitrary finite set of named channels and an explicit sparse set of interactions, including higher-order expressions. Write them with `intervention_programs.Expr`:

```python
from antecedent.intervention_programs import minimum, var

interactions = {
    "joint": var("a") * var("b") * var("c"),
    "modified": var("a") * minimum(var("b"), 0.25),
}
```

A variable read here means the **lag-weighted, saturated channel value**, not raw exposure. Foreign variables, generated values, lagged feature reads and callbacks are refused. Antecedent does not expand every subset of channels for you. Declare the interactions you want.

The default fitted family uses finite distributed lags, shared Michaelis-Menten saturation, group intercepts, seasonal effects, an additive observed confounder and Gaussian AR(1) dependence within groups. The zero-lag reduction fixes the lag and serial parameters. Frequentist and Bayesian realizations keep their distinct uncertainty meanings.

Pass `family=` to `joint_panel.prepare` to choose a broader mechanism. A request that names none reproduces the default fit exactly. The choices are:

- truncated geometric adstock (one decay per channel) in place of the free simplex of lag weights;
- a free positive Hill shape per channel in place of Michaelis-Menten;
- one half-saturation per declared class of groups (all groups, or one class per group), partially pooled toward the first class under a prior;
- spatially correlated group intercepts with a fitted kernel range, from a declared group distance matrix.

Under spatial dependence, frequentist uncertainty uses the model-based inverse information, because groups are no longer independent clusters. The parametric bootstrap repeats the whole fit, including temporal change selection. The Bayesian realization places proper priors on every added coordinate. Hill shape, geometric decay and class half-saturation can vary through the same piecewise or smooth temporal path. A scalar response expression supports integer Hill shapes only. Other shapes are refused.

A family changes the fitted mechanism, never what identifies a world. Worlds are identified by the declared design, such as a known whole-program randomized assignment. A fit of any family leaves a confounded panel without an identified claim.

## Prepare, estimate and reuse

Call `joint_panel.prepare`. Call `inspect()` to read the question and its design attempts. Nothing is fitted yet. Call `estimate()`. You get a `PanelResult` that keeps every draw for every world and every period. Failed draws remain failed slots on the shared uncertainty axis.

After you discard the builder, the prepared result still has what it needs to run. `retarget` reuses a frozen fit when the execution context permits caching. A compatible world or utility change reuses that fit on the same condition. Changed source data, priors, initial laws, mechanisms or relevant numerical policies invalidate their actual dependencies. Explicitly disabled caches stay disabled. Execution checks cancellation and resource admission.

To get one functional per world and per draw, without holding the full table, use Rust. In Rust, call `PreparedStudy::estimate_projected`. In Rust, call `JointPanelResult::project_streamed`. Both read the same aligned draws in bounded tiles. Failed slots stay in place. Each reduced value equals projecting the full output.

## Compare worlds on one fitted law

Call `PanelResult.scheduled_contrast`. It builds a contrast of scheduled programs on the shared draws. Its interval carries the cross-world covariance. Subtracting marginal interval endpoints does not give contrast uncertainty. The functional is `"terminal_mean"` or a one-key mapping such as `{"window_average": {"from": 0, "until": 3}}`. The interval is withheld when a draw slot failed. The interval is also withheld when the fit does not authorize a complete law. The value is conditional on the fitted mechanism.

General linear contrasts keep finite nonzero world coefficients and an explicit reference. Time-window and change projections keep the question you asked, including history, horizon, coefficients and units. A scalar dose-by-time surface needs a declared policy covering every modelled exposure period. An undeclared background-channel fix is refused. Implicit post-pulse continuation is refused. A multichannel schedule needs an explicit world and an explicit program.

### Joint worlds from a prepared static response

Start from a static `InterventionResponse` study that still holds its adjustment proof. Call `compile_joint_response`. You get several worlds on one shared Gaussian fit. The result is their joint law, including the cross-world covariance:

```python
import math
import numpy as np
import antecedent as ant
from antecedent import joint_panel
from antecedent.outcomes import require_success

n = 240
z = np.array([math.sin(i / 17.0) for i in range(n)])
t = z + np.array([math.cos(i / 11.0) for i in range(n)])
y = 1.0 + 2.0 * t + 0.8 * z + 0.1 * np.array([math.sin(i / 3.7) for i in range(n)])
study = ant.prepare(
    {"t": t, "y": y, "z": z},
    graph=[("z", "t"), ("z", "y"), ("t", "y")],
    query=ant.InterventionResponse("y", intervention=ant.intervention.Set("t", 0.25)),
    estimator="response.intervention_gcomp",
    refute="none",
    inference=ant.Frequentist(bootstrap=0),
    execution=ant.Execution(seed=4),
)
worlds = [
    {"id": i, "reference": i == 0, "levels": {"t": level}} for i, level in enumerate((-0.5, 0.0, 0.5))
]
compiled = require_success(
    joint_panel.compile_joint_response(
        study, outcome="y", worlds=worlds,
        contrast={"kind": "linear", "terms": [[0, -2.0], [1, 1.0], [2, 1.0]], "reference": 0},
    )
)
print(compiled.inspect()["fitted"])            # False: compilation fits nothing

law = require_success(compiled.execute(execution=ant.Execution(seed=7)))
print(law.means, law.uncertainty, law.calibration)   # sampling_covariance, not_measured
value, standard_error = law.contrast                   # one linear functional of the joint law

other = require_success(compiled.retarget(outcome="y", worlds=worlds[:2]))   # reuse the fit
replayed = require_success(compiled.consume_current(bytes(law.export())))
assert replayed.means == law.means
```

Each world assigns exactly the study's intervention coordinates inside their observed range. A level outside that range is refused. It is not extrapolated. The contrast is a `difference` or a `linear` combination with an explicit reference. The target keeps the Gaussian-linear, fixed-empirical-covariate premise. Worlds whose conditional joint support is not established stay model-conditional. On those worlds, `identified_claim` is false. `retarget` moves only the world projections. `consume_current` replays an exported law against this same result. It refuses a law made for a different target.

This static joint Gaussian response is additive in its treatments. It refuses a contrast whose world weights cancel under that model (the product interaction), because the answer would be zero with zero standard error by construction. Estimate an interaction from the [cell-mean contrast](joint-cells.md) or from the nonlinear panel family. A linear contrast of the same worlds that the additive mean can move is executed from the retained fit.

## Read the scientific standing

A fitted response does not establish causal identification. The result records the source and design proofs it supports. It records its attempts to identify joint assignment. Unsupported worlds stay unresolved or model-conditional. Per-period assignment does not silently identify a whole trajectory.

### Identification under a declared Gaussian restriction

Call `with_unrolled_graph_attempt` on a declared unrolled point graph. You get a verdict from general identification. That verdict is a nonparametric functional of the observed law. The fitted Gaussian/AR panel realizes it only if the observed conditional laws are the panel's own. Declare that restriction on the target before you prepare the request. Call `declare_joint_panel_gaussian_restriction`. The fit's identification is then a separate attempt, `joint_adjustment.gaussian_restriction`. The result keeps it as identification under a parametric restriction. It is never nonparametric identification.

The restriction states that, given the observed context, the outcome is a saturated lag-weighted sum of the channels (and declared products) plus a covariate and season effect, with Gaussian group intercepts and stationary Gaussian AR(1) errors independent of the whole program. The claim is refused, with a typed error, when:

- the restriction is not declared (the verdict stays recorded and the fit stays model-conditional);
- general ID reports a hedge (the declaration does not identify through a hedge);
- the unrolled graph names a carryover reach other than the fitted lag order, serial outcome dependence that the fitted errors do not have (or lack it where they do), or an assignment that responds to past outcomes, which the fitted mechanism does not realize;
- the mechanism is time-varying, which has no residual check.

After the fit, residual checks run on the point mechanism. One check is Jarque-Bera normality of the AR(1)-whitened innovations. The other is an equal-count-bin lack-of-fit check of the mean against each channel's lag-weighted exposure and the covariate. The family-wise level is 0.01, split evenly over the tests run. A flagged check withdraws the identified claim from the result. `require_identified` refuses. The attempt is `Rejected`. A check that does not flag is not evidence that the restriction holds. No calibration of these checks is measured. `retarget` rebuilds the identification and drops the restricted claim. Rebind the graph on the retargeted handle.

From Python, declare the restriction with `joint_panel.prepare(..., gaussian_restriction=True)`. Run the graph with `PreparedPanel.with_unrolled_graph(periods=..., channels=..., outcome_periods=[...], carryover_reach=..., outcome_persistence=..., latent_confounding=[(("channel", 0, 5), ("outcome", 5))])`. A refusal raises `CausalValueError`. A hedge or an undeclared restriction is visible in `inspect()["attempts"]` and in `inspect()["identified"]`. Tests: `python/tests/test_joint_panel_restriction.py`.

### Categorical channels and missing cells

Call `joint_panel.dummy_code_panel`. Or pass `categorical=[PanelCategoricalChannel(...)]` to `prepare`. Either call rewrites a categorical assignment as one 0/1 indicator channel per non-reference level. The declared level set must equal the labels the units assign. A declared level no unit assigns is refused by name (`python/tests/test_joint_panel_categorical.py`). Call `joint_panel.prepare_imputed` to multiply impute missing outcomes and confounders. Frequentist results pool the response vectors with Rubin's rules. Bayesian results pool them by stacking posteriors. A missing assignment or season level is refused, not imputed (`python/tests/test_joint_panel_imputation.py`).

`inspect()` lists the design attempts in order, including the ones that failed: the joint Gaussian route, observational joint adjustment and known randomized assignment of whole programs. Each refused attempt names the next one (for example, collecting positive-probability whole-program assignments). A confounded design with unsupported joint worlds therefore keeps its fitted mechanism but withholds a causal claim for those worlds.

Panel point means are fitted plug-in estimates, including in Bayesian runs. In Rust, `mean_source` keeps the scientific standing and the exact projected question for a decision or a search. `law_source` does the same. `project_*_source` does the same. A model-conditional result must be consumed explicitly. Accessors named `numerical_*` expose numerical components. They do not grant current scientific permission.

Time-varying workflows evaluate piecewise candidates and a regularized smooth alternative. They retain selection uncertainty. Unique attribution to exposure, saturation, composition or mechanism is refused without a separating design. Explicit reference-world response changes stay available under their own scope.

## Explain a change in response

Call `PanelResult.decompose_reference`. It evaluates a four-component reference decomposition of a change between two programs on the same retained fitted law. Declare:

- `before_world` and `after_world` (world ids)
- population weights in source group order
- `time_weights`
- an `order` permutation of exposure, saturation, population and mechanism
- `periods`, mapping before and after periods to parameter indices (all zero for a stationary fit)

Independently fitted marginal draws are never paired by index.

Exposure changes schedules and initial histories. Saturation changes the half-saturation parameter. Population changes scenario covariates, seasons and group weights. Mechanism changes the remaining parameters, lag weights and group effects. The result keeps all 16 hybrid responses. It keeps the contributions from all 24 replacement orders. It keeps exact Shapley averages and your sequential decomposition on every retained draw. `order_range` describes variation across orderings, separately from statistical uncertainty. `joint_uncertainty()` returns the joint four-component covariance and the total-change variance from complete aligned realizations. It also returns the requested and complete slot counts. It does not blend ordering sensitivity with statistical uncertainty. It does not hide failed slots. Use frequentist bootstrap inference when the requested law must repeat the entire original fit.

The decomposition is a model-conditional reference calculation and does not establish unique causal attribution. To find out what the design does support:

- `attempt_identified_reference` returns the identified decomposition, or the coarser contrast you supply as a fallback. The per-hybrid missing premises sit beside it.
- `attempt_design_adaptive_reference` runs the finest decomposition the retained design supports. A component step runs only when both hybrids it differences are identified. Each missing premise comes back as a typed obligation. The obligation names the design or analysis that would earn it: a design identifying the base worlds, an explicit hybrid world, a period-specific mechanism intervention, a population transport proof or a retargeted time functional. For temporal fits it also splits the law by the specification each draw selected. That split is descriptive. It grants no coverage.
- `attempt_explanation` returns the coarser window change of differences that a finer explanation falls back to. The finer names are `exposure`, `saturation`, `composition` and `mechanism`. `grants_unique_causal_attribution` is false.

Exact interventions with the same empirical population, parameter path and identified final response inherit their world proof. Changed contexts, population laws and parameter interventions need their own transport premises. Equal dose ranges do not supply those premises. A fitted change curve does not supply them either.

`JointDesignResult.decompose_reference_components` keeps the whole response covariance for frequentist runs. It propagates that covariance through Shapley, sequential and ordering-specific covariance matrices. Posterior sources keep aligned draws and any finite-law importance weights. Complete-procedure bootstrap sources keep each paired refit slot, failures included. `decompose_population_reference` handles covariate-distribution changes through identified stratum response worlds and explicit before and after probabilities. Its `SelectionDiagram` must confine population selection to an exogenous baseline root. All other response mechanisms must be declared invariant. Selection at the outcome mechanism produces a scientific refusal. Baseline latent confounding produces a scientific refusal. `attempt_population_reference` runs a coarser intervention contrast under its own visible target. Reference artifacts export and replay historical numbers. Call `replay_reference_decomposition`. Call `replay_population_reference`. Current consumption also compares the design, the exact statistical execution, and your bindings and scope.

## Limits

Frequentist cluster-sandwich and local Gaussian realizations are approximations, not posterior draws. Bayesian consumption requires its convergence checks. The fitted panel publishes no supported interval. Its parameter law is model-conditional. Nothing measures its coverage. `calibration` reads `not_measured`. The reliability of a selected decision is not measured. The reliability of a next-study ranking is not measured. Successful numerical execution does not supply that reliability.

## Intervals with a calibration statement

In Rust, five readings return a 95% interval. Each reading also returns the calibration key the result derives for it, and that key's standing against the shipped coverage records:

| Reading | Interval |
| --- | --- |
| `NativeJointDesignResult::contrast_interval_reading` | A prespecified contrast of a compiled native design: Student-t with Satterthwaite degrees of freedom for whole-program randomization, a normal interval for switchback, instrument, cohort and longitudinal designs, the conformal projection for synthetic rollouts, the Wald interval for a complier effect, or the equal-tailed posterior interval. |
| `RandomizedSelectionResult::selected_value_reading`, `InstrumentedSelectionResult::selected_value_reading` | The held-out value of the selected program, plus and minus the normal quantile times its standard error. |
| `JointDesignResult::population_reference_region` | A simultaneous region for the four Shapley contributions of a population reference decomposition (Bonferroni, so at least nominal). |
| `JointDesignResult::updated_contrast_reading` | The weighted equal-tailed posterior interval of a contrast after an experiment update. |

A reading is `calibrated` only at the level, construction, sample-size range, resampling or posterior size and support condition its coverage record measured. Anywhere else it still returns the interval. It says `not_measured` or `outside_measured_scope`. The conditions the key checks:

- A switchback needs every requested history observed. The design refuses otherwise. It also needs at least 32 sequences per requested history, from 512 sequences. With fewer, the inverse-probability estimator is skewed. Its normal interval under-covers (0.933 at 64 sequences, 0.941 at 256).
- A joint 95% conformal region over two synthetic rollouts needs at least 39 pre-periods. It is refused below that. It was measured from 40.
- A synthetic posterior interval depends on the declared inverse-gamma prior of the residual variance. It was measured with the prior mean at the residual variance. The key requires the declared mean to be within a factor of two of the pre-period fit error. A prior mean several times too large over-covers.
- Whole-program randomization with equal weights was measured from 128 units with at least 32 per program. With unequal population weights it was measured from 8 units with at least four per program. There the interval uses a leverage-corrected variance.
- A complier effect needs an estimated first stage of at least 0.4, from 800 units.
- Every Bayesian interval was measured with its likelihood scale declared at the generating value. The library cannot check that declaration.
- The updated-contrast interval was measured with the truth drawn from the declared prior and an identity translation. Its coverage is an average over that prior.

In Rust, `NativeJointDesignResult::support` still describes the result as a whole. `calibration` does the same. Both are unchanged. The reading carries the interval's own standing. The measured scope of each reading is recorded in `parity/capability_records.toml` under `joint_response.interval_readings`.

Frozen artifacts keep the numerical law, dependencies, uncertainty meaning and failed slots. Independent historical replay grants no current authorization. It does not attest that stored fit numbers were generated honestly. A current consumer supplies the expected program, the frozen law and the relevant evidence authority.

## Related

- [Search and certify intervention programs](joint-program-search.md)
- [Update a joint response with experiments and designs](joint-program-designs.md)
- [Joint cells](joint-cells.md) and [joint distributions](../joint-distributions.md)
- [Compare a coordinated intervention program with its baseline](../tutorials/joint-interventions.md)
- [Interventions and worlds](../concepts/interventions-and-worlds.md)

For a complete marketing investigation using a shared panel fit, see [beyond the marketing mix](../tutorials/beyond-marketing-mix.md).
