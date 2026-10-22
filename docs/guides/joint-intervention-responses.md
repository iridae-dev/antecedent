# Joint intervention responses

Joint response workflows evaluate coordinated interventions through the causal
compiler. A request binds the population, outcome and units, intervention worlds,
reference levels, conditioning, time horizon and initial history. The compiled
program retains that question together with identification or an explicit
model-conditional justification, numerical execution and resource planning.

## Channels and programmable interactions

The panel family supports an arbitrary finite set of named channels and an
explicit sparse set of interactions, including higher-order expressions. Use the
existing Rust `ValueExpr` or Python `intervention_programs.Expr` syntax:

```python
from antecedent.intervention_programs import minimum, var

interactions = {
    "joint": var("a") * var("b") * var("c"),
    "modified": var("a") * minimum(var("b"), 0.25),
}
```

These feature reads mean **lag-weighted, saturated channel values**, rather than
raw exposure. Rust makes this binding explicit through
`PanelInteractionSpec::from_saturated_features`. Foreign variables, generated
values, lagged feature reads and callbacks are refused for this binding. The
compiler does not automatically expand every subset of channels.

The fitted group/time family uses finite distributed lags, shared
Michaelis–Menten saturation, group intercepts, seasonal effects, an additive
observed confounder and Gaussian AR(1) dependence within groups. The zero-lag
reduction fixes the lag and serial parameters. Frequentist and Bayesian
realizations preserve their distinct uncertainty meanings.

## Compile, execute and reuse

Rust callers use `Study::prepare_joint_panel` or an existing prepared study's
`compile_joint_panel`. Python exposes preparation, inspection, estimation,
retargeting and historical artifact consumption through `antecedent.joint_panel`.

Inspection does not fit. A prepared program owns what it needs to execute after
the original builder is discarded. Compatible world or utility changes reuse a
frozen fit when the execution context permits caching. Changed source data,
priors, initial laws, mechanisms or relevant numerical policies invalidate their
actual dependencies. Explicitly disabled caches remain disabled.

Execution checks cancellation and resource admission. Retained numerical and
scientific outputs keep their admission until their final owner is dropped.
Failed draws remain failed slots on the shared uncertainty axis.

## Contrasts, time windows and surfaces

General linear contrasts retain finite nonzero world coefficients and an explicit
reference. Shared covariance or aligned draws preserve cross-world uncertainty;
subtracting marginal interval endpoints does not produce contrast uncertainty.

Time-window and change projections retain their actual prefix-world questions,
including history, horizons, coefficients and units. Scalar dose-by-time surfaces
require a declared policy covering every modeled exposure period. Undeclared
background-channel fixes and implicit post-pulse continuation are refused. Joint
multichannel schedules use explicit world/program contracts.

## Scientific standing and downstream use

A fitted response does not establish causal identification. The compiler records
supported source/design proofs and joint-assignment attempts. Unsupported worlds
remain unresolved or model-conditional; per-period assignment does not silently
identify an entire trajectory.

Inspection lists the design attempts in order, including the ones that failed: the
joint Gaussian route, observational joint adjustment, and known randomized
assignment of whole programs. Every refused or unsupported attempt names the next
attempt (for example, collecting positive-probability whole-program assignments).
A confounded design with unsupported joint worlds therefore keeps its fitted
mechanism, but withholds a causal claim for those worlds.

The static joint Gaussian response is additive in its treatments. It refuses a contrast
whose world weights cancel under that model (the product interaction), because the answer
would be zero with zero standard error by construction; an interaction is estimated from
the cell-mean family contrast, or from the nonlinear panel family. A linear contrast of
the same worlds that the additive mean can move is executed from the retained fit.

Rust `mean_source`, `law_source` and `project_*_source` ports retain scientific
standing and the exact projected question for decision and search consumers.
Model-conditional consumption must be explicit. Accessors named `numerical_*`
expose numerical components without granting current scientific permission.
Panel point means are fitted plug-in estimates, including in Bayesian runs.
Gaussian Bayesian mean ports average posterior draws and are labeled accordingly.

Time-varying workflows evaluate piecewise candidates and a regularized smooth
alternative while retaining selection uncertainty. Unique attribution to exposure,
saturation, composition or mechanism is refused without a separating design;
explicit reference-world response changes remain available under their own scope.

Finite search certifies only the supplied feasible candidate set under its retained
law. Continuous search reports a best-found feasible point and its stopping
reason, without inventing a global optimality gap. Comparative or robust criteria
that need a finite set or identified set retain those requirements.

Experimental updates require a licensed estimand bridge, declared mapping
uncertainty/dependence and evidence origins. The supported modular update
reweights a frozen Bayesian joint law; next-study ranking supports additional
independent Gaussian sampling of the same mapped estimand. Extracted experimental
sources require caller-current ledger authority for authorization, decisions and
search. Withdrawal or supersession can refuse current use.

Advanced search, explanation and experimental-update interfaces are currently
available in Rust. Python currently exposes the core prepared panel lifecycle and
historical artifact interchange.

## Uncertainty and artifact limits

Frequentist cluster-sandwich/local Gaussian realizations are approximations, not
posterior draws. Bayesian law consumption requires its actual convergence checks.
Whole-procedure calibration and selected-decision reliability require their own
measured evidence; successful numerical execution does not supply that evidence.

Frozen artifacts preserve the numerical law, dependencies, uncertainty meaning and
failed slots. Independent historical replay grants no current authorization and
does not attest that stored fit numbers were generated honestly. Current consumers
supply the expected compiled program, frozen law and relevant evidence authority.
