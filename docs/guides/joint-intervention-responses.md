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

The retained result keeps every draw by every world by every period. Callers that need
one prespecified functional per world and draw can use `PreparedStudy::estimate_projected`,
`JointPanelResult::project_streamed` or `project_frozen_streamed`: the same shared plan runs
over the same aligned draws in bounded tiles, the tile's rows are reduced and dropped,
failed slots stay in place, and every reduced value equals projecting the full output.

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

`JointPanelResult::decompose_reference` (Python: `PanelResult.decompose_reference`)
now evaluates a complete four-component reference decomposition. Supply before/after
programs, population weights, explicit time weights, a replacement order, and
before/after parameter-period maps into the **same** retained fitted law. Stationary
fits use parameter index zero; temporal maps preserve their joint model/path and
selection realizations. Independently fitted marginal draws are never paired by index.

The exposure block changes schedules and initial histories; saturation changes the
half-saturation parameter; population changes scenario covariates, seasons and group
weights; mechanism changes the remaining parameters, lag weights and group effects.
The result retains all 16 hybrid responses, contributions from all 24 replacement
orders, exact Shapley averages and the caller's sequential decomposition on every
retained draw. `order_range` describes variation across orderings, separately from
statistical uncertainty. Failed parent draw slots remain failed. Frozen execution
programs can replay the same projection through `decompose_reference` without fitting.
`joint_uncertainty()` returns the joint four-component covariance and total-change
variance from complete aligned realizations, alongside requested/complete slot counts.
It does not combine ordering sensitivity with statistical uncertainty or conceal failures.
Use `PanelInference::FrequentistBootstrap` when the requested frequentist law must repeat
the entire original fit. The local Gaussian option has its own approximation meaning.
Temporal laws retain their original change/smoothing selection; Bayesian realizations
retain the actual joint parameter/path draws.

`decompose_reference_checked` additionally checks every hybrid against the retained
identification product. Exact interventions with the same empirical population,
parameter path and identified final response inherit their original world proof.
Changed contexts, population laws and parameter interventions require their own transport
premises; equal dose ranges or a fitted change curve do not establish those premises.
The conditional calculation still executes when these finer causal obligations remain.
`attempt_identified_reference_decomposition` accepts an explicit coarser request and
actually evaluates it if finer identification fails, preserving the per-hybrid reasons.
Identified reference responses do not establish unique causal attribution among factors.

`attempt_design_adaptive_reference` runs the finest decomposition the retained design
supports. A component step (one of exposure, saturation, population or mechanism moved
from its before to its after value, the other three held at a stated context) is
executed on every aligned law slot exactly when both hybrids it differences are
identified. Components with no identified step are listed; each missing premise is
returned as a typed obligation together with the design or analysis that would earn it:
a design that identifies the base worlds, an explicit hybrid world, a period-specific
mechanism intervention, a population transport proof or a retargeted time functional.
When the full decomposition is not identified and a coarser request is supplied, the
total-response contrast is executed on the same fit as well. Executed steps can be
rebuilt by an independent consumer from the stored fit alone.

For temporal fits the same call splits the decomposed law by the specification each
draw selected (`decomposition_selection`): mass on each change period, on smooth paths
and on no change, the mean and variance of every contribution within a specification, and
the variance owed to the selection itself. The split is descriptive and grants no
coverage. `decompose_reference_streamed` returns the point decomposition and the law's
joint moments without holding any per-draw table or paired path.

For broader scientific sources, `decompose_scientific_reference` consumes one actual
`ScientificDecisionSource` with all required world responses and joint uncertainty.
`ReferenceComponentBindings` assigns acted-on variables to three disjoint exposure,
saturation and mechanism blocks, while population follows each world's population and
conditioning bindings. The adapter checks every binary hybrid against the two endpoints:
static interventions, temporal schedules and structural mechanism replacements retain
separate semantics. The original compiler must supply any identification and transport
proofs; an array of caller-asserted support flags cannot replace them. The result retains
the complete scientific source and its question identity. The panel convenience method
`decompose_compiled_components` derives that source from actual retained terminal means
or the aligned law. This permits supported reference decomposition under the chosen
scope without asserting uniquely determined causal attribution.

The design-native `JointDesignResult.decompose_reference_components` preserves the
whole response covariance in frequentist runs: exact linear maps propagate it into
Shapley, sequential and all 24 ordering-specific covariance matrices. Posterior sources
retain aligned draws and any finite-law importance weights. Complete-procedure bootstrap
sources retain their fitted point and every paired refit slot, including failure diagnostics;
failed slots never become zero covariance or silently disappear. The continuous full-history
g-computation owner supplies these same paired refits and its distinct proper conditional
model posterior directly through the compiled world adapter. Versioned reference artifacts
retain the original design, explicit component recipe, requested scope and numerical
replay digest. `replay_reference_decomposition` returns historical numbers without a
current-authorizing source; current consumption also compares the independently retained
design, exact statistical execution (including prior and seed), and caller-specified
bindings/scope. Changing an ordering cannot silently reuse an
artifact for a different scientific question.

`decompose_population_reference` handles actual covariate-distribution changes using
identified stratum response worlds and explicit before/after probabilities. Its
`SelectionDiagram` must confine population selection to an exogenous baseline root,
with no baseline parent or latent common cause. The diagram declares all other response
mechanisms invariant; the compiler checks these structural conditions and every
constant-baseline source program before mixing its response. The returned transport
proof retains the graph, functional and exact population-law identity. Selection at the
outcome mechanism or baseline latent confounding produces a scientific refusal;
`attempt_population_reference` can execute a caller-specified coarser intervention
contrast under its own visible target. Rust and Python expose population artifact
export, independent historical replay and caller-current consumption.

The checked fixture in `conformance/response/joint_panel/population_reference.json`
contains an actual 16-arm source codebook and observations for
`Y=(1+L) M A/(A+S)`, with baseline weights changing from `(0.8,0.2)` to `(0.2,0.8)`.
Rust and Python compare all four contributions against separately calculated subset
weights, including a population-only change that contributes solely to population.
The full measurement procedure is registered in
`parity/joint_response_calibration.toml`; final calibration remains tied to the stable
candidate and supplies no implementation-stage release license.



Finite search certifies only the supplied feasible candidate set under its retained
law. General continuous search reports a best-found feasible point and its stopping
reason. `JointPanelResult::compile_certified_search` (Python:
`PanelResult.certify_program`) instead derives an arithmetic objective from the
actual retained point response, including histories, lags, saturation, interactions,
population and time weights. Declare every scheduled control explicitly, using equal
bounds for a fixed coordinate. Supported arithmetic receives outward interval bounds,
deterministic branch-and-bound and retained pruning counts/frontier. Compiler-derived
derivative signs and additive factorization permit reductions; structurally proven
concavity permits supporting-plane dual bounds with nonnegative multipliers. A resource
stop retains a valid partial bound; only a gap meeting the requested tolerance certifies
optimality. Unsupported expressions remain available to the general search route.

Certificates preserve the incumbent's actual response source and original fitted
dependency. Their versioned artifacts replay both the frozen panel evaluator and the
numerical proof. Current consumption additionally compares current fit and domain
identities. The certificate concerns the fitted scalar objective. Selected-value
reliability remains separate. Comparative or robust criteria
that need a finite set or identified set retain those requirements.

The selection owner provides fixed-set joint bootstrap max-t intervals and a complete
held-out procedure executor. The executor separates actual original observational
units/time spans, repeats generation, fitting, feasibility, search and selection, then
evaluates that fixed action on reserved observations. Whole-procedure cluster resampling
retains failures, ties, instability and training-minus-heldout optimism. Bootstrap
diagnostics do not assert coverage at ties. The evaluation owner's justified sampling
law supplies fixed-action uncertainty; independent RNG seeds cannot replace reserved
observational data.

`prepare_randomized_selection` applies this procedure to the native joint randomized
estimator. It enforces hard per-program costs before choosing a winner and supports
training-only screening or fixed-family simultaneous intervals. Reserved evaluation
units have distinct original identities and an explicit common sampling frame.
`select_instrumented_program` fits the native joint structural-feature IV model,
generates a prespecified or training-dependent raw-dose domain, derives and searches
its actual polynomial response under hard restrictions, and independently refits the
held-out units to evaluate the selected structural contrast. The target remains the
declared homogeneous structural moment population; it does not turn local IV effects
into population effects. Both interfaces retain every resampling failure, instability
and training optimism and replay from the original observations.

`PanelResult.search_program` attempts the compiler-derived certificate and executes
the existing bounded best-found search if the expression has no supported interval
evaluator. Reports retain the concrete refusal and omit a global gap for that route.
Cancellation, invalid scopes and inadequate admission remain typed execution errors.

The complete calibration procedures are registered in
`parity/joint_response_calibration.toml`; `scripts/joint_response_calibration.py`
lists their actual Cargo commands. Final measurement requires the exact clean stable
candidate and at least 400 complete data sets per registered cell. Implementation
checks and registered procedures grant no release calibration license.

Experimental updates require a licensed estimand bridge, declared mapping
uncertainty/dependence and evidence origins. The supported modular update
reweights a frozen Bayesian joint law. `rank_next_studies` values candidate studies,
each optionally in its own regime: another population, window, dose, duration,
exposure regime, recency or spend, with its own licensed bridge (including the transport
premise that regime needs), its own paired per-draw mapping and its own cost. The current
posterior is reweighted by that study's likelihood, the caller's terminal utility is
evaluated on every whole response row, and the net value is the gain over deciding now
minus the declared cost. `update_for_future_study` executes the update a candidate would
cause for a stated outcome, so the moved response and the changed decision are
observable. An unlicensed, unaligned or stale regime refuses the ranking; it is never
replaced by the observed estimand. `rank_next_samples` remains the additional-sample
case of the observed estimand. Extracted experimental
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

A retained Bayesian result can be updated directly with
`JointPanelResult::condition_experiment`, using the same frozen modular likelihood
as the prepared experiment workflow. The update reweights complete source and
mapping slots; it does not fit the panel again. Terminal means, window averages
and explicit weighted time sums retain their exact projected quantity. A changed
population, dose, duration or historical window requires the explicit estimand
translation and transport premise in the bridge. The numeric affine map is an
input model, rather than an assumed dose or duration scaling law.

`JointLawExperimentResult::condition` also accepts an authenticated native design
posterior with one checked world-mean coordinate per target world. It derives the
assembled contrast from the scientific target and sparse projection, then uses
the same paired likelihood, current evidence checks, decision owner and
next-study integration. A frequentist sampling law cannot enter this Bayesian
adapter. Its modular update carries conditional scope even when the original
design identifies its original estimand. Export retains the original numerical
law, bridge, likelihood and historical origins; replay returns historical numbers,
and `consume_current` additionally binds the caller's original scientific source
and current evidence ledger.

The public experiment-chain fixture executes terminal, window and weighted
projections across deterministic and dependent uncertain mappings, including
changed population, dose, duration and time settings. Each row checks posterior
weights and expectations against independent likelihood-ratio arithmetic, terminal
utilities and operational exclusions against exhaustive enumeration, and
next-study values against a separate predictive-mixture quadrature with a stated
integration error bound. The standard-library reference in
`conformance/response/joint_panel/experiment_reference.py` separately computes
known-truth population/time/dose estimands and retained finite-law update,
constrained-decision and future-study reference values. These checks do not grant
a release calibration license.

`joint_experiment_acceptance_matrix()` generates the required machine-readable
protocol from `JOINT_DESIGN_CAPABILITIES`, rather than a hand-selected list of
passing designs. It crosses every design with inference meaning, mapping
uncertainty, population/time/dose/duration translation and decision/feasibility
cases, and includes the panel's three advertised time projections. An outstanding
implementation stays in its row as an implementation gap. A frequentist sampling
law being offered as a Bayesian prior is recorded separately as a scientific
refusal. Matrix rows state what must execute; they are not test results.

Experimental translation can also carry different source and target population
and horizon coordinates themselves. This route requires a
`ParameterInterventionResponse` bridge, a complete `EstimandBridge` binding the
actual experimental population, and an explicit transport premise applying to
the target. A same-population horizon extrapolation still requires that premise.
Removing the estimand translation, naming another experimental population or
supplying a transport premise for another target refuses the update.

For switchback histories, structural IV feature contrasts, cohort-specific DiD
ATTs and donor-reference rollout contrasts, `NativeJointDesignResult` offers
`experiment_quantity` and `condition_native_experiment` in Rust
(`experiment_quantity` and `condition_experiment` in Python). A
`NativeExperimentBinding` names the exact linear functional of retained native
coordinates and explicitly declares their original response units. The quantity
identity includes the original target kind and projection. An IV structural
contrast remains structural, a cohort ATT remains cohort-specific, and a donor
contrast retains its treated-unit/reference interpretation throughout the update,
decision and future-study ranking. This consumer does not manufacture population
world means to fit a different source interface.

The native consumer exports the complete original design/data/statistical
artifact and experimental inputs. Historical replay reruns both the original
native procedure and the modular update. Current consumption additionally matches
the actual source, target projection and observed experiment and checks the
caller's current ledger. Each source/mapping dependency binds the statistical
prior, simulation seed and retained joint law as well as the original design and
data, so another posterior's draw slots cannot inherit a mapping by sharing a
prepared source.

The machine-readable experiment acceptance protocol is available as
`joint_experiment_acceptance_matrix()` in Rust and
`antecedent.joint_panel.experiment_acceptance_matrix()` in Python. It includes
all advertised designs, both statistical modes, deterministic and dependent
mapping laws, five translation cases and three terminal decision cases.
A protocol row is an obligation, not a passing result.

Run `python3 scripts/run_joint_panel_experiment_acceptance.py` on a stable
checkout to execute the independent references and write
`conformance/response/joint_panel/experiment_acceptance_report.json`. The runner
hashes the actual procedure sources, records each successful executable case's
source dependency and reference checks, and refuses changed sources or missing
mandatory cases. Original finite-state and continuous-history g-computation,
equal-weight and unequal-weight geo populations, and factorial and multi-arm
assignment each require their own evidence. A frequentist sampling-law refusal
records the early statistical guard; downstream Bayesian stages are explicitly
inapplicable in that case. Neither the generated protocol nor this implementation
acceptance report grants a Q08/E14 release calibration license.

An encouragement-specific local IV ratio keeps its original complier target.
Its unconditioned ratio law does not automatically establish an expected value.
The native experiment consumer can condition that proper prior on a positive
known-variance Gaussian experimental likelihood when the binding explicitly
supplies `local_ratio_regularization`: global bounds on the mapping intercept
and slope and the substantive bounded-support assumption. Every retained paired
realization is checked against those bounds. With a nonzero target coefficient,
the Gaussian likelihood dominates every polynomial ratio tail, establishing
posterior moments without asserting any unconditioned prior moment. Missing or
contradicted bounds refuse the update. These assumptions and their exact identity
are retained in the target, update artifact and current-consumption checks.

The separate Python `prepare_native_joint_design(recipe, **execution)` entry point
compiles the design's actual target. Its versioned `kind` recipe selects
`randomized`, `switchback`, `longitudinal`, `gaussian_longitudinal`,
`instrumented`, `instrumented_local`, `difference_in_differences` or
`synthetic_control`. `execute(inference)` retains the same identification proof
with a compatible statistical procedure. Rust `export_artifact` (Python `.export()`), historical replay
and independently prepared current consumption retain original observations,
physical schedules, population, assumptions, prior, seed and full joint law.
The machine-readable contracts and independent evidence are listed in
`parity/fixtures/joint_design_procedures.toml`.

Continuous observational histories use explicit conditional mechanisms rather
than categorical discretization. Covariate and outcome formulas can depend on
recorded histories, interactions between treatment times and treatment-affected
covariates. The compiler checks causal ordering, exact intervention schedules,
joint policy density rank and declared support. Full episode bootstrap refits
all mechanisms and policy support, retaining each failed slot. The Bayesian
counterpart uses proper conditional Gaussian/inverse-gamma priors and shares
one complete mechanism draw across all requested regimes.

`instrumented_local` retains an encouragement-specific complier effect under
exclusion and monotonicity. It does not require a homogeneous population response
model. Frequentist inference uses the paired intention-to-treat/receipt Wald
ratio and withholds its normal interval for weak support. The Bayesian procedure
retains paired empirical score-law ratio, numerator and denominator draws.
These Bayesian-bootstrap weights are not a principal-strata likelihood that
enforces monotonicity, and a finite sampled first-stage check is not a global
posterior support certificate. The native output makes no claim that ratio
posterior moments exist. Any downstream expected-utility update must establish
its required moments separately.

Future-study receipts also retain `admissible_action_identity`, computed from
the actual serialized terminal action domain used in every future-data update.
For deterministic operational restrictions, derive that domain from the actual
constrained decision's admissible actions before ranking studies. Excluded actions
stay excluded in the future-data calculation. The acceptance matrix checks the
constrained ranking against its own reference, including exact zero information
value when only one action remains. Python receipts bind declared action names,
intercepts and coefficients, so renaming or changing an action cannot inherit
another action domain's identity.
