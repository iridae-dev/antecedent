# Fit and compare coordinated intervention programs

Compare several coordinated interventions, such as a channel mix held over a horizon, with uncertainty taken from one shared fitted law. A request binds the population, outcome, intervention worlds, reference, time horizon and initial history. The compiled program keeps that question together with its identification status (or an explicit model-conditional justification), the numerical execution and the resource plan.

If you have not seen a joint comparison before, start with the [joint intervention tutorial](../tutorials/joint-interventions.md), which builds one on simulated data with a known answer. For what a world is, see [interventions and worlds](../concepts/interventions-and-worlds.md). This page is the task reference for the panel and static joint workflows in `antecedent.joint_panel`. Choosing among programs is covered in [search and certify intervention programs](joint-program-search.md); experimental updates and whole-program designs are in [update a joint response with experiments and designs](joint-program-designs.md). Joint binary interventions on cross-sectional data are in [joint cells](joint-cells.md).

## Declare channels and interactions

The panel family supports an arbitrary finite set of named channels and an explicit sparse set of interactions, including higher-order expressions. Write them with `intervention_programs.Expr`:

```python
from antecedent.intervention_programs import minimum, var

interactions = {
    "joint": var("a") * var("b") * var("c"),
    "modified": var("a") * minimum(var("b"), 0.25),
}
```

A variable read here means the **lag-weighted, saturated channel value**, not raw exposure. Foreign variables, generated values, lagged feature reads and callbacks are refused for this binding. Antecedent does not expand every subset of channels for you; declare the interactions you want.

The default fitted family uses finite distributed lags, shared Michaelis-Menten saturation, group intercepts, seasonal effects, an additive observed confounder and Gaussian AR(1) dependence within groups. The zero-lag reduction fixes the lag and serial parameters. Frequentist and Bayesian realizations keep their distinct uncertainty meanings.

Pass `family=` to `joint_panel.prepare` to choose a broader mechanism; a request that names none reproduces the default fit exactly. The choices are:

- truncated geometric adstock (one decay per channel) in place of the free simplex of lag weights;
- a free positive Hill shape per channel in place of Michaelis-Menten;
- one half-saturation per declared class of groups (all groups, or one class per group), partially pooled toward the first class under a prior;
- spatially correlated group intercepts with a fitted kernel range, from a declared group distance matrix.

Under spatial dependence, frequentist uncertainty uses the model-based inverse information, because groups are no longer independent clusters. The parametric bootstrap repeats the whole fit, including temporal change selection, and the Bayesian realization places proper priors on every added coordinate. Hill shape, geometric decay and class half-saturation can vary through the same piecewise or smooth temporal path. A compiled scalar response expression supports integer Hill shapes only; other shapes are refused.

A family changes the fitted mechanism, never what identifies a world. Worlds are identified by the declared design, such as a known whole-program randomized assignment. A fit of any family leaves a confounded panel without an identified claim.

## Prepare, estimate and reuse

`joint_panel.prepare` compiles the request and `inspect()` reports the question and its design attempts without fitting. `estimate()` fits once and returns a `PanelResult` that retains every draw for every world and every period. Failed draws remain failed slots on the shared uncertainty axis.

A prepared program owns what it needs to execute after the original builder is discarded. `retarget` and compatible world or utility changes reuse a frozen fit when the execution context permits caching. Changed source data, priors, initial laws, mechanisms or relevant numerical policies invalidate their actual dependencies, and explicitly disabled caches stay disabled. Execution checks cancellation and resource admission.

To reduce to one functional per world and draw without holding the full table, the Rust facade offers `PreparedStudy::estimate_projected` and `JointPanelResult::project_streamed`. The same plan runs over the same aligned draws in bounded tiles, failed slots stay in place, and each reduced value equals projecting the full output.

## Compare worlds on one fitted law

`PanelResult.scheduled_contrast` assembles a contrast of scheduled programs on the shared draws, so its interval carries the cross-world covariance. Subtracting marginal interval endpoints does not give contrast uncertainty. The functional is `"terminal_mean"` or a one-key mapping such as `{"window_average": {"from": 0, "until": 3}}`. The interval is withheld when a draw slot failed or the fit does not authorize a complete law. The value is conditional on the fitted mechanism.

General linear contrasts keep finite nonzero world coefficients and an explicit reference. Time-window and change projections keep their actual prefix-world questions, including history, horizon, coefficients and units. Scalar dose-by-time surfaces need a declared policy covering every modelled exposure period; undeclared background-channel fixes and implicit post-pulse continuation are refused. Multichannel schedules use explicit world and program contracts.

### Joint worlds from a prepared static response

For a static `InterventionResponse` study whose adjustment proof is retained, `compile_joint_response` lowers several worlds onto one shared Gaussian fit and returns their joint law, including the cross-world covariance:

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
print(law.means, law.uncertainty, law.calibration)   # sampling_covariance, unmeasured
value, standard_error = law.contrast                   # one linear functional of the joint law

other = require_success(compiled.retarget(outcome="y", worlds=worlds[:2]))   # reuse the fit
replayed = require_success(compiled.consume_current(bytes(law.export())))
assert replayed.means == law.means
```

Each world assigns exactly the study's intervention coordinates inside their observed range; a level outside it is refused, never extrapolated. The contrast is a `difference` or a `linear` combination with an explicit reference. The target keeps the Gaussian-linear, fixed-empirical-covariate premise, so worlds whose conditional joint support is not established stay model-conditional (`identified_claim` is false). `retarget` moves only the world projections; `consume_current` replays an exported law against the same compiled program and refuses one made for a different target.

This static joint Gaussian response is additive in its treatments. It refuses a contrast whose world weights cancel under that model (the product interaction), because the answer would be zero with zero standard error by construction. Estimate an interaction from the [cell-mean contrast](joint-cells.md) or from the nonlinear panel family. A linear contrast of the same worlds that the additive mean can move is executed from the retained fit.

## Read the scientific standing

A fitted response does not establish causal identification. The compiler records the source and design proofs it supports and its attempts to identify joint assignment. Unsupported worlds stay unresolved or model-conditional; per-period assignment does not silently identify a whole trajectory.

`inspect()` lists the design attempts in order, including the ones that failed: the joint Gaussian route, observational joint adjustment and known randomized assignment of whole programs. Each refused attempt names the next one (for example, collecting positive-probability whole-program assignments). A confounded design with unsupported joint worlds therefore keeps its fitted mechanism but withholds a causal claim for those worlds.

Panel point means are fitted plug-in estimates, including in Bayesian runs. The Rust `mean_source`, `law_source` and `project_*_source` ports keep scientific standing and the exact projected question for decision and search consumers, and model-conditional consumption must be explicit. Accessors named `numerical_*` expose numerical components without granting current scientific permission.

Time-varying workflows evaluate piecewise candidates and a regularized smooth alternative while retaining selection uncertainty. Unique attribution to exposure, saturation, composition or mechanism is refused without a separating design; explicit reference-world response changes stay available under their own scope.

## Explain a change in response

`PanelResult.decompose_reference` evaluates a four-component reference decomposition of a change between two programs on the same retained fitted law. Declare `before_world` and `after_world` (compiled world ids), population weights in source group order, explicit `time_weights`, an `order` permutation of exposure, saturation, population and mechanism, and `periods` mapping before and after periods to parameter indices (all zero for a stationary fit). Independently fitted marginal draws are never paired by index.

Exposure changes schedules and initial histories; saturation changes the half-saturation parameter; population changes scenario covariates, seasons and group weights; mechanism changes the remaining parameters, lag weights and group effects. The result keeps all 16 hybrid responses, the contributions from all 24 replacement orders, exact Shapley averages and your sequential decomposition on every retained draw. `order_range` describes variation across orderings, separately from statistical uncertainty. `joint_uncertainty()` returns the joint four-component covariance and total-change variance from complete aligned realizations, with requested and complete slot counts. It does not blend ordering sensitivity with statistical uncertainty or hide failed slots. Use frequentist bootstrap inference when the requested law must repeat the entire original fit.

The decomposition is a model-conditional reference calculation and does not establish unique causal attribution. To find out what the design does support:

- `attempt_identified_reference` returns the identified decomposition or the coarser contrast you supply as a fallback, with the per-hybrid missing premises beside it.
- `attempt_design_adaptive_reference` runs the finest decomposition the retained design supports. A component step is executed exactly when both hybrids it differences are identified. Each missing premise comes back as a typed obligation with the design or analysis that would earn it: a design identifying the base worlds, an explicit hybrid world, a period-specific mechanism intervention, a population transport proof or a retargeted time functional. For temporal fits it also splits the law by the specification each draw selected; that split is descriptive and grants no coverage.
- `attempt_explanation` executes the coarser window change of differences that a finer explanation (`exposure`, `saturation`, `composition` or `mechanism`) falls back to, with `grants_unique_causal_attribution` false.

Exact interventions with the same empirical population, parameter path and identified final response inherit their world proof. Changed contexts, population laws and parameter interventions need their own transport premises; equal dose ranges or a fitted change curve do not supply them.

`JointDesignResult.decompose_reference_components` keeps the whole response covariance for frequentist runs, propagating it through Shapley, sequential and ordering-specific covariance matrices. Posterior sources keep aligned draws and any finite-law importance weights; complete-procedure bootstrap sources keep each paired refit slot, failures included. `decompose_population_reference` handles covariate-distribution changes through identified stratum response worlds and explicit before and after probabilities. Its `SelectionDiagram` must confine population selection to an exogenous baseline root, with all other response mechanisms declared invariant. Selection at the outcome mechanism, or baseline latent confounding, produces a scientific refusal; `attempt_population_reference` executes a coarser intervention contrast under its own visible target. Reference artifacts export and replay historical numbers (`replay_reference_decomposition`, `replay_population_reference`); current consumption additionally compares the design, the exact statistical execution and your bindings and scope.

## Limits

Frequentist cluster-sandwich and local Gaussian realizations are approximations, not posterior draws. Bayesian consumption requires its convergence checks. Calibration of the whole procedure and the reliability of a selected decision need their own measured evidence; successful numerical execution does not supply it, and `calibration` reads `unmeasured` until that evidence exists.

Frozen artifacts keep the numerical law, dependencies, uncertainty meaning and failed slots. Independent historical replay grants no current authorization and does not attest that stored fit numbers were generated honestly. Current consumers supply the expected compiled program, frozen law and relevant evidence authority.

## Related

- [Search and certify intervention programs](joint-program-search.md)
- [Update a joint response with experiments and designs](joint-program-designs.md)
- [Joint cells](joint-cells.md) and [joint distributions](../joint-distributions.md)
- [Compare a coordinated intervention program with its baseline](../tutorials/joint-interventions.md)
- [Interventions and worlds](../concepts/interventions-and-worlds.md)
