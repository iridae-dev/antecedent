# Model feedback loops and ask about equilibria

Ask what an intervention does in a system whose variables influence each other in a loop: supply and price, a thermostat and a room, interacting populations. A directed acyclic graph cannot state these systems. With `antecedent.cyclic` you declare simultaneous equations. You solve their equilibrium under named intervention worlds. When you have clamp experiments, you fit the mechanisms from data.

If you are new to feedback models, work through [one feedback system, three questions](../tutorials/feedback-three-questions.md) first. It derives by hand why observing, clamping, and shocking give different answers. It runs the linear model through `LinearCyclicModel`. For what a world is, see [interventions and worlds](../concepts/interventions-and-worlds.md). This page covers the rest of the surface.

Every answer is conditional on the specified or fitted model. Nothing here identifies a causal structure nonparametrically. The statistical outputs carry no nominal coverage claim. [The development notes](../development/cyclic-systems.md) record the mathematical scope of each family.

| Task | Start from |
| --- | --- |
| Solve a specified linear loop under interventions | `LinearCyclicModel.worlds` ([tutorial](../tutorials/feedback-three-questions.md)) |
| Solve a specified nonlinear loop, with shared disturbances | [`EquilibriumModel.worlds`](#model-nonlinear-systems-and-shared-disturbances) |
| Fit mechanisms from equilibrium data with known clamps | [`EquilibriumModel.prepare`](#fit-a-loop-from-clamp-experiments) |
| Fit continuous-time linear dynamics from trajectories | [`OuMechanisms.fit_trajectories`](#fit-a-continuous-time-ou-model-from-trajectories) |
| Simulate or fit a nonlinear stochastic differential equation | [`StochasticDifferentialEquation`](#simulate-and-fit-a-nonlinear-stochastic-differential-equation) |

## Read a world's answer

On a `LinearCyclicModel`, call `model.worlds([...], component_id=..., source_id=..., certifying_evidence=[...])`. That call prepares the program you keep. Call `prepared.interventions()` to run it. Each world reports its certificate. It reports how it was solved. It reports any failed or withheld mass. A clamp that leaves no unique solution is a `no_solution` world with failed mass one. It is never a dropped world. Reading its mean raises `CyclicRefusal`. Its `next_attempt` names what to try. `Set` is a hard clamp. `Shift` is an additive shock on the disturbance with feedback still active. `Soft` is a named mechanism replacement.

`source_id` records where the model came from. It does not certify the component. `certifying_evidence` must name other evidence. `certificate="report_uncertified"` answers a uniquely solvable world. The result says that world is uncertified.

## Model nonlinear systems and shared disturbances

`EquilibriumModel` takes named `Equation` mechanisms and a row-major joint disturbance covariance. An activation is `identity`, `tanh`, `sine`, or `ReLU`. You can also pass a portable compositional expression. Each `Equation(id, intercept, weights, activation)` is the simultaneous equation of one variable. It is the intercept plus the activation applied to the weighted sum of the variables, plus that variable's disturbance. Worlds share the disturbance draws you supply, one row per joint draw. A counterfactual and its factual world see the same noise.

```python
from antecedent.cyclic import Equation, EquilibriumModel, Set, World
from antecedent.outcomes import require_success

feedback = EquilibriumModel(
    (
        Equation("a", 0.0, (0.0, 0.4), "identity"),
        Equation("b", 0.0, (0.5, 0.0), "identity"),
    ),
    (1.0, 0.0, 0.0, 1.0),
)
worlds = require_success(feedback.worlds(
    [World(0), World(1, Set("a", 2.0))], [[0.0, 0.0], [1.0, 0.0]]
).interventions())
print(worlds[0].draws("b"), worlds[1].draws("b"))
```

Each entry of `draws` is the equilibrium value of `b` for one disturbance row. A row with no unique solution keeps its slot as `None`. It counts toward the world's failed mass. In world 0 the loop is active (`a = 0.4 b + e_a`, `b = 0.5 a + e_b`). In world 1 `a` is clamped at 2. Its equation is deleted. `b = 1.0` on both rows.

Numerical initialization is never a selection law. When a system has several equilibria, the weights over roots must be declared. None are inferred from solver behavior. See the family list in the development notes for the polynomial, multiple-root, ODE, and Ornstein-Uhlenbeck families.

## Fit a loop from clamp experiments

Fitting needs independent equilibrium replicates with known hard clamps. `EquilibriumModel.prepare` keeps aligned intervention worlds together with the joint-likelihood fit. The held-out joint intervention is evaluation only.

```python
from antecedent.outcomes import require_success

prepared = template.prepare(
    [[], joint_clamps],          # worlds: no intervention, then a joint clamp
    [[0.0, 0.0, 0.0]],           # disturbance draws shared by the worlds
    factual=[0.2, 0.3, -0.1],    # optional fully observed factual state
    parameters=[{"kind": "weight", "equation": 0, "parent": 1}],
    rows=rows,                   # {"values": [...], "clamps": [...]}
)
result = require_success(prepared.execute())
result.to_dict()["fit"]["converged"]
```

In this fragment `template` is an `EquilibriumModel` holding the starting parameter values. `joint_clamps` is a list of `{"variable": index, "value": number}` clamps. `rows` are the training rows. [`examples/python/general_cyclic_systems.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/general_cyclic_systems.py) is the complete runnable version. It has an independent generator of overlapping loops, a held-out joint intervention, and a shared-noise counterfactual. [`examples/python/cyclic_paper_experiments.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/cyclic_paper_experiments.py) reproduces the published synthetic protocols with their deviations stated. For a linear law recovered from clamp experiments, use `prepare_experimental_recovery` with an explicit draw origin ([`examples/python/recovered_feedback_law.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/recovered_feedback_law.py)). `inspect()` reports the rows a clamp design leaves unresolved before any fit.

## Fit a continuous-time OU model from trajectories

`OuMechanisms.fit_trajectories` fits `dX = (M X + b) dt + L dW` from continuous trajectories rather than independent stationary snapshots. Each unit is observed at strictly increasing, possibly irregular times under a known piecewise-constant intervention schedule (`OuScheduleSegment`). Between observations the Gaussian transition is exact. The mean uses a matrix exponential. The process covariance uses a van Loan block exponential. There is no time-discretization error. A Kalman filter evaluates the likelihood under an explicit linear Gaussian observation model (`OuObservationModel`). That model has matrix `H`, an offset, and known noise. It covers partial observation. Components with no direct observation are reported with their filtered uncertainty.

```python
from antecedent.cyclic import OuMechanisms, OuObservationModel, OuPredictionDesign, OuTrajectory
from antecedent.outcomes import require_success

prepared = OuMechanisms(("a", "b"), drift, forcing, factor).fit_trajectories(
    [OuTrajectory("unit-0", times, values)],      # irregular times, vector observations
    [{"kind": "drift", "equation": 0, "parent": 1}, {"kind": "diffusion", "equation": 0, "source": 0}],
    observation=OuObservationModel((1.0, 0.0), (0.05,)),   # only `a` observed, noise 0.05
    predictions=[OuPredictionDesign("held-out", (0.5, 1.0, 2.0), schedule=schedule)],
)
fit = require_success(prepared.execute()).to_dict()["answer"]["result"]["fit"]
fit["fitting"], fit["identification"], fit["numerics"]
```

The result keeps three records apart, and none implies another:

- `fitting`: optimizer termination. The fields are `converged`, `score_norm`, and `line_search_exhausted`. The score is analytic. It uses Fréchet derivatives of the matrix exponential carried through the Kalman recursion. It uses the steady-state equations for a stationary start. The information is the Gauss-Newton form, positive semidefinite by construction, named in `derivative_method`. A fit that stops on its bound is returned with `converged: false`. Its evidence columns record `fit_convergence`.
- `identification`: `identified`, `not_identified`, or `model_conditional`. The status comes from the rank of the observed information. It comes from the observability of the pair `(H, A)`, the state dimensions no observation depends on. It comes from the sampling-interval aliasing check. An oscillation with `max |Im lambda| * dt >= pi` at the smallest observed interval is not unique. Unobserved equations are listed. The assumptions the result is relative to are stated.
- `numerics`: stability of the fitted drift. It includes the accuracy witness `|exp(A) exp(-A) - I|` of every exponential used. It includes the conditioning of the innovation covariances and of the information.

A design that cannot identify a requested coordinate is refused with `dynamics_not_identified`. It is not returned with a wide interval. No unit observed at two times is refused. Use the stationary families for snapshots. An information matrix that is singular at the converged optimum is refused. That singularity is a coordinate of an unobserved state the observations cannot determine, or a redundant diffusion factor. Parameterize `L` as lower-triangular, since only `L L^T` is identified.

Predictions of held-out intervention trajectories and of sustained-intervention stationary laws carry two separate uncertainty meanings. They are never summed. The evidence selectors are `ou_trajectory_state` and `ou_trajectory_stationary`, next to `likelihood`. `state_covariance` and `observation_variance` are process variability at the fitted parameters. The `*_parameter_*` fields are the fitted parameters' sampling uncertainty. They are propagated by the delta method from the observed information. They are available only when the information is positive definite. A stationary world whose drift is unstable keeps its slot with the reason.

## Simulate and fit a nonlinear stochastic differential equation

`StochasticDifferentialEquation` declares `dX = f(X, u(t), theta) dt + g(X, u(t), theta) dW`. The drift is `f`. The diffusion `g` is diagonal and state-dependent. It is written in the portable mechanism expression language `SdeExpression`. The operations are add, multiply, tanh, sin, and softplus. The expression uses states, scheduled inputs, and named parameters. A diffusion may be wrapped as `sqrt(max(expression, 0))` (`SdeDiffusion(..., square_root=True)`) for demographic noise. External inputs `u(t)` are declared piecewise-constant or linear schedules (`SdeInput`). States may carry reflecting or clamping bounds (`SdeBound`). That is how continuous count-like states stay nonnegative. Dimension is bounded at 16 states.

Validation is part of the model. Interval derivative arithmetic over the declared state box and input envelopes certifies finite Lipschitz constants of the drift and the diffusion (`result["simulation"]["certificate"]`). A nonlinearity that multiplies unbounded states is refused with `route_not_supported` / `sde.unbounded_growth`. Declare bounds. A square-root diffusion whose argument can reach zero is admitted. It is certified only as Holder-1/2. The certificate carries no strong-convergence guarantee. It says so.

The documented epidemic is an SIR system with nonlinear transmission `beta (1 + a u(t)) S I / N`. It has a seasonal input `u(t)`. It has square-root noise whose variance is the rate of the events changing each compartment. That noise is a diagonal chemical-Langevin approximation. It ignores the correlation the shared infection event induces between `S` and `I`:

```python
from antecedent.cyclic import (
    SdeAction, SdeContrast, SdeInput, SdeSimulation, SdeTrajectory, SdeWorld,
    StochasticDifferentialEquation,
)
from antecedent.outcomes import require_success

sir = StochasticDifferentialEquation.epidemic(population=1000.0, beta=0.6, amplitude=0.3, gamma=0.25)
lockdown = SdeWorld("lockdown", inputs=(SdeInput("season", (0.0,), (-1.0,), envelope=(-1.0, 1.0)),))
suppress = SdeWorld("suppress", actions=(SdeAction("I", (5.0,), (20.0,)),))   # do(I = 20) from t = 5
prepared = sir.simulate(
    (950.0, 50.0, 0.0),
    SdeSimulation(step=0.05, steps=400, paths=2000, seed=11, output_every=80),
    interventions=[lockdown, suppress],
    contrasts=[SdeContrast("lockdown_effect", "lockdown", "baseline")],
)
simulation = require_success(prepared.execute()).to_dict()["answer"]["result"]["simulation"]
```

The simulation uses Euler-Maruyama or diagonal Milstein on the declared fixed step. The generator is counter-based, addressed by `(seed, path, step, component)`. The same request replays bit for bit. Every world reuses the same Brownian increments. A `SdeContrast` is a paired difference with its own Monte Carlo standard error. Milstein is admitted only when each diffusion depends on no other state. Otherwise the refusal is `sde.milstein_noncommutative`. `do` actions must lie on the step grid (`sde.action_time_off_grid`). Replacement input schedules must lie inside the declared envelope (`sde.input_outside_certified_envelope`). Each world reports the mean and the variance. It reports the joint covariance across states. Optionally it reports the final state of every path.

Three error sources stay in separate fields and are never summed:

- `monte_carlo_standard_error`: sampling error of the finite ensemble.
- `numerical`: with `check_step`, the same Brownian paths are re-run at half the step. The paired mean difference is `weak_difference`, with its standard error. The pathwise root-mean-square difference is `strong_rms_difference`. The weak-order-one Richardson mean and a convergence flag estimate the discretization effect of the declared step.
- `parameter_uncertainty` (after a fit): variance across asymptotic-normal draws of the fitted parameters of each world's ensemble mean. The draws are simulated with common random numbers.

Fitting uses the Euler Gaussian transition quasi-likelihood for dense, fully observed trajectories (`fit`): `x' | x ~ N(x + f dt, diag(g^2 dt))`. Known holds (`do`) are excluded from the density. It is a pseudo-likelihood, consistent as the observation interval shrinks. The result carries a third record next to `fitting` and `identification`. `discretization` reports the interval-times-rate product `dt * ||J_f||`. Above `coarse_flag` (0.1) the fit is a model-conditional approximation. Above `coarse_refusal` (0.5) it is refused with `sde_fit.observation_interval_too_coarse`, because no justified procedure exists there without latent paths. A vanishing transition variance is refused (`sde_fit.degenerate_transition`). Unobserved transitions are refused as `dynamics_not_identified`. Parameters the likelihood does not read are refused as `dynamics_not_identified`. With an `SdeObservationModel` (noise and partial observation) and a Gaussian `filter_prior`, the likelihood becomes the extended Kalman innovation likelihood of the Euler-discretized model, substepped by `filter_step`. It linearizes the dynamics. Such a fit is always `model_conditional`. Standard errors are the inverse observed information of the pseudo-likelihood.

```python
prepared = sir.fit(
    [SdeTrajectory("unit-0", times, rows)], ["beta", "amplitude", "gamma"],
    (950.0, 50.0, 0.0), SdeSimulation(step=0.05, steps=400, paths=500, seed=3, output_every=80),
    interventions=[lockdown], contrasts=[SdeContrast("lockdown_effect", "lockdown", "baseline")],
    parameter_draws=64, parameter_seed=5,
)
result = require_success(prepared.execute()).to_dict()["answer"]["result"]
result["fit"]["discretization"], result["parameter_uncertainty"]["contrasts"]
```

The result evidence selectors are `likelihood` and `world`. `likelihood` is the quasi log likelihood, with the three records as diagnostics. `world` is an ensemble mean, with its Monte Carlo standard error, step-halving evidence, and parameter spread as diagnostics. Everything is conditional on the declared drift, diffusion, and bounds. None of the three error fields is a confidence interval of the mean. None is a credible interval of the mean.

## Reuse, branch and export

- `inspect()` is cheap after execution. It reports preparation work, fits actually performed, and the SCC schedules.
- `reconfigure(...)` changes worlds without refitting. `rebind(rows)` replaces the training observations through a new preparation. It refits.
- `execute()` returns an outcome containing a native-owned `CyclicResult`. `result.to_dict()` explicitly materializes its report. `result.export()` produces a typed artifact. `bytes(artifact)` exposes its encoding. Store the identity separately from those bytes. Evidence projections use `evidence.to_dict()` for their explicit report conversion too.
- `consume_artifact(artifact, expected_identity=...)` checks identity and integrity. It recomputes in a fresh process. Built-in mechanisms are portable. Process-local callbacks refuse export.
- Pass resource limits and cancellation through `execution=Execution(...)`.
- Systems with hundreds of variables are solved sparsely. `SolvePolicy` bounds the Jacobian's nonzeros (`max_nnz`), the block order (`max_block_size`), and the factor fill (`max_factor_fill`). It bounds total iterations (`max_total_iterations`) and wall clock (`max_wall_clock_ms`). Exceeding one bound is a refusal that names its own `solver_*_budget` code. Memory is bounded by the execution's `ResourceLimits`, not by the policy. Each solved world carries an `evidence` dictionary. It has the block structure, resource accounting, contraction, and spectral bounds. It has the Jacobian's nonsingularity basis, the unique-root basis, and an `ill_conditioned` flag (`condition_warning`).

```python
from antecedent.cyclic import consume_artifact
from antecedent.outcomes import require_success

artifact = result.export()
replayed = require_success(
    consume_artifact(bytes(artifact), expected_identity=result.identity)
)
assert replayed.to_dict() == result.to_dict()
```

## Use a result in a decision

`prepared.evidence(schema, selectors)` projects the execution you kept onto native evidence coordinates without refitting. The projection feeds `evidence.decide_means(contract)`, `evidence.joint_law()`, and `evidence.decide_enclosures(contract)`. Roots acquire no implicit weights. Unresolved boxes stay inside robust utility bounds. Impossible posterior evidence refuses a decision. See [decisions](../decisions.md) and [composition](../composition.md).

In Rust, call `Study::prepare_cyclic`. Call `Study::prepare_cyclic_family`. Both use the same `PreparedStudy` handle. See [cyclic systems for developers](../development/cyclic-systems.md) and [`examples/rust/feedback_three_questions.rs`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/feedback_three_questions.rs).

## Refusals and failed worlds

A world that cannot be solved is an outcome, not an exception. It keeps its slot with status `no_solution` or `withheld` and its failed mass. Only reading a value from it raises `CyclicRefusal`. Its `next_attempt` names what to try. A numerical failure inside the solver budgets is likewise a world outcome you keep. Non-convergence is one such failure. A singular Jacobian is another. Requests that cannot run at all are refused with a registered code ([list](../reference/refusal-codes.md#other-registered-refusal-codes)):

| Code | When |
| --- | --- |
| `solver_jacobian_nnz_budget`, `solver_block_size_budget`, `solver_factor_fill_budget`, `solver_iteration_budget`, `solver_wall_clock_budget`, `solver_model_order_budget` | a solve exceeds the corresponding `SolvePolicy` bound, or (`solver_model_order_budget`) the number of variables exceeds an admitted bound such as the dense disturbance covariance's |
| `dynamics_not_identified` | a trajectory design that does not identify a requested coordinate (no unit observed at two times, a singular information matrix, unobserved transitions) |
| `route_not_supported` | a model outside a family's certified scope, with a namespaced detail such as `sde.unbounded_growth`, `sde.milstein_noncommutative`, `sde.action_time_off_grid` or `sde.input_outside_certified_envelope` |

An SDE fit additionally refuses `sde_fit.observation_interval_too_coarse` and
`sde_fit.degenerate_transition`, described [above](#simulate-and-fit-a-nonlinear-stochastic-differential-equation).

## Related

- [One feedback system, three questions](../tutorials/feedback-three-questions.md)
- [Interventions and worlds](../concepts/interventions-and-worlds.md)
- [Decision contracts](../decisions.md)
