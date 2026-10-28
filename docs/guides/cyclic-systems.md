# Model feedback loops and ask about equilibria

Ask what an intervention does in a system whose variables influence each other in a loop: supply and price, a thermostat and a room, interacting populations. A directed acyclic graph cannot state these systems. With `antecedent.cyclic` you declare simultaneous equations, solve their equilibrium under named intervention worlds and, when you have clamp experiments, fit the mechanisms from data.

If you are new to feedback models, work through [one feedback system, three questions](../tutorials/feedback-three-questions.md) first: it derives by hand why observing, clamping and shocking give different answers and runs the linear model through `LinearCyclicModel`. For what a world is, see [interventions and worlds](../concepts/interventions-and-worlds.md). This page covers the rest of the surface.

Every answer is conditional on the specified or fitted model. Nothing here identifies a causal structure nonparametrically, and the statistical outputs carry no nominal coverage claim. [The development notes](../development/cyclic-systems.md) record the mathematical scope of each family.

## Read a world's answer

`model.worlds([...], component_id=..., source_id=..., certifying_evidence=[...])` prepares the retained program and `prepared.interventions()` executes it. Each world reports its certificate, the route that solved it, and any failed or withheld mass. A clamp that leaves no unique solution is a `no_solution` world with failed mass one, never a dropped world; reading its mean raises `CyclicRefusal` whose `next_attempt` names what to try. `Set` is a hard clamp, `Shift` an additive shock on the disturbance with feedback still active, and `Soft` a named mechanism replacement.

`source_id` records where the model came from. It does not certify the component: `certifying_evidence` must name other evidence, and `certificate="report_uncertified"` answers a uniquely solvable world while marking it uncertified.

## Model nonlinear systems and shared disturbances

`EquilibriumModel` takes named `Equation` mechanisms (identity, tanh, sine or ReLU
activations, or a portable compositional expression) and a row-major joint
disturbance covariance. Worlds share the disturbance draws you supply, so a
counterfactual and its factual world see the same noise.

```python
from antecedent.cyclic import Equation, EquilibriumModel, Set, World

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

Numerical initialization is never a selection law. When a system has several
equilibria, the weights over roots must be declared; none are inferred from solver
behavior. See the family list in the development notes for the polynomial,
multiple-root, ODE and Ornstein-Uhlenbeck routes.

## Fit a loop from clamp experiments

Fitting needs independent equilibrium replicates with known hard clamps.
`EquilibriumModel.prepare` retains aligned intervention worlds together with the
joint-likelihood fit; the held-out joint intervention is evaluation only.

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

`examples/python/general_cyclic_systems.py` is the complete runnable version: an
independent generator of overlapping loops, a held-out joint intervention and a
shared-noise counterfactual. `examples/python/cyclic_paper_experiments.py`
reproduces the published synthetic protocols with their deviations stated. For a
linear law recovered from clamp experiments, use `prepare_experimental_recovery`
with an explicit draw origin; `inspect()` reports the rows a clamp design leaves
unresolved before any fit.

## Fit a continuous-time OU model from trajectories

`OuMechanisms.fit_trajectories` fits `dX = (M X + b) dt + L dW` from continuous
trajectories rather than independent stationary snapshots. Each unit is observed at
strictly increasing, possibly irregular times under a known piecewise-constant
intervention schedule (`OuScheduleSegment`). Between observations the Gaussian transition is
exact (matrix exponential for the mean, van Loan block exponential for the process
covariance), so there is no time-discretization error. A Kalman filter evaluates the
likelihood under an explicit linear Gaussian observation model
(`OuObservationModel`: matrix `H`, offset, known noise), which covers partial observation;
components with no direct observation are reported with their filtered uncertainty.

```python
from antecedent.cyclic import OuMechanisms, OuObservationModel, OuPredictionDesign, OuTrajectory

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

- `fitting`: optimizer termination (`converged`, `score_norm`, `line_search_exhausted`).
  The score is analytic (Fréchet derivatives of the matrix exponential carried through the
  Kalman recursion, and the steady-state equations for a stationary start); the information
  is the Gauss-Newton form, positive semidefinite by construction, named in
  `derivative_method`. A fit that stops on its bound is returned with `converged: false` and
  its evidence columns are marked (`fit_convergence`).
- `identification`: `identified`, `not_identified` or `model_conditional`, from the rank of
  the observed information, the observability of the pair `(H, A)` (state dimensions no
  observation depends on) and the sampling-interval aliasing check (an oscillation with
  `max |Im lambda| * dt >= pi` at the smallest observed interval is not unique). Unobserved
  equations are listed; the assumptions the result is relative to are stated.
- `numerics`: stability of the fitted drift, the accuracy witness `|exp(A) exp(-A) - I|` of
  every exponential used, and the conditioning of the innovation covariances and of the
  information.

A design that cannot identify a requested coordinate is refused with
`dynamics_not_identified` rather than returned with a wide interval: no unit observed at
two times (use the stationary families for snapshots), or an information matrix that is
singular at the converged optimum (a coordinate of an unobserved state the observations
cannot determine, or a redundant diffusion factor; parameterize `L` as lower-triangular,
since only `L L^T` is identified).

Predictions of held-out intervention trajectories and of sustained-intervention stationary
laws (also evidence selectors `ou_trajectory_state` and `ou_trajectory_stationary`, next to
`likelihood`) carry two separate uncertainty meanings that are never summed: `state_covariance` and
`observation_variance` are process variability at the fitted parameters, and the
`*_parameter_*` fields are the fitted parameters' sampling uncertainty propagated by the
delta method from the observed information. They are available only when the information is
positive definite. A stationary world whose drift is unstable keeps its slot with the reason.

## Simulate and fit a nonlinear stochastic differential equation

`StochasticDifferentialEquation` declares `dX = f(X, u(t), theta) dt + g(X, u(t), theta) dW`
with drift `f` and a diagonal, state-dependent diffusion `g` written in the portable
mechanism expression language (`SdeExpression`: add, multiply, tanh, sin, softplus, with
states, scheduled inputs and named parameters). A diffusion may be wrapped as
`sqrt(max(expression, 0))` (`SdeDiffusion(..., square_root=True)`) for demographic noise.
External inputs `u(t)` are declared piecewise-constant or linear schedules (`SdeInput`);
states may carry reflecting or clamping bounds (`SdeBound`), which is how continuous
count-like states stay nonnegative. Dimension is bounded at 16 states.

Validation is part of the model. Interval derivative arithmetic over the declared state box
and input envelopes certifies finite Lipschitz constants of the drift and the diffusion
(`result["simulation"]["certificate"]`). A nonlinearity that multiplies unbounded states is
refused with `route_not_supported` / `sde.unbounded_growth`: declare bounds. A square-root
diffusion whose argument can reach zero is admitted but certified only as Holder-1/2, so the
certificate carries no strong-convergence guarantee and says so.

The documented epidemic is an SIR system with nonlinear transmission
`beta (1 + a u(t)) S I / N`, a seasonal input `u(t)` and square-root noise whose variance is
the rate of the events changing each compartment (a diagonal chemical-Langevin
approximation that ignores the correlation the shared infection event induces between `S`
and `I`):

```python
from antecedent.cyclic import (
    SdeAction, SdeContrast, SdeInput, SdeSimulation, SdeTrajectory, SdeWorld,
    StochasticDifferentialEquation,
)

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

The simulation uses Euler-Maruyama or diagonal Milstein on the declared fixed step with a
counter-based generator addressed by `(seed, path, step, component)`: the same request
replays bit for bit, and every world reuses the same Brownian increments, so a `SdeContrast`
is a paired difference with its own Monte Carlo standard error. Milstein is admitted only
when each diffusion depends on no other state (otherwise `sde.milstein_noncommutative`);
`do` actions must lie on the step grid (`sde.action_time_off_grid`) and replacement input
schedules inside the declared envelope (`sde.input_outside_certified_envelope`). Each world
reports the mean, variance, the joint covariance across states and, optionally, the final
state of every path.

Three error sources stay in separate fields and are never summed:

- `monte_carlo_standard_error`: sampling error of the finite ensemble.
- `numerical`: with `check_step`, the same Brownian paths are re-run at half the step. The
  paired mean difference (`weak_difference`, with its standard error), the pathwise
  root-mean-square difference (`strong_rms_difference`), the weak-order-one Richardson mean
  and a convergence flag estimate the discretization effect of the declared step.
- `parameter_uncertainty` (after a fit): variance across asymptotic-normal draws of the fitted
  parameters of each world's ensemble mean, simulated with common random numbers.

Fitting uses the Euler Gaussian transition quasi-likelihood for dense, fully observed
trajectories (`fit`): `x' | x ~ N(x + f dt, diag(g^2 dt))`, with known holds (`do`) excluded
from the density. It is a pseudo-likelihood, consistent as the observation interval shrinks,
so the result carries a third record next to `fitting` and `identification`:
`discretization` reports the interval-times-rate product `dt * ||J_f||`. Above `coarse_flag`
(0.1) the fit is a model-conditional approximation; above `coarse_refusal` (0.5) it is
refused with `sde_fit.observation_interval_too_coarse`, because no justified procedure
exists there without latent paths. A vanishing transition variance is refused
(`sde_fit.degenerate_transition`), unobserved transitions or parameters the likelihood does
not read are refused as `dynamics_not_identified`. With an `SdeObservationModel` (noise and
partial observation) and a Gaussian `filter_prior`, the likelihood becomes the extended
Kalman innovation likelihood of the Euler-discretized model, substepped by `filter_step`; it
linearizes the dynamics, so such a fit is always `model_conditional`. Standard errors are the
inverse observed information of the pseudo-likelihood.

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

The result evidence selectors are `likelihood` (the quasi log likelihood, with the three
records as diagnostics) and `world` (an ensemble mean, with its Monte Carlo standard error,
step-halving evidence and parameter spread as diagnostics). Everything is conditional on the
declared drift, diffusion and bounds; none of the three error fields is a confidence or
credible interval of the mean.

## Reuse, branch and export

- `inspect()` is cheap after execution; it reports preparation work, fits actually
  performed and the SCC schedules.
- `reconfigure(...)` changes worlds without refitting; `rebind(rows)` replaces the
  training observations through a new preparation and refits.
- `execute()` returns an outcome containing a native-owned `CyclicResult`.
  `result.to_dict()` explicitly materializes its report; `result.export()` produces
  a typed artifact, and `bytes(artifact)` exposes its encoding. Store the identity
  separately from those bytes. Evidence projections use `evidence.to_dict()` for
  their explicit report conversion too.
- `consume_artifact(artifact, expected_identity=...)` checks identity and integrity
  and recomputes in a fresh process. Built-in mechanisms are portable; process-local
  callbacks refuse export.
- Pass resource limits and cancellation through `execution=Execution(...)`.
- Systems with hundreds of variables are solved sparsely. `SolvePolicy` bounds the
  Jacobian's nonzeros (`max_nnz`), block order (`max_block_size`), estimated memory
  (`max_bytes`), factor fill (`max_factor_fill`), total iterations
  (`max_total_iterations`) and wall clock (`max_wall_clock_ms`); exceeding one is a
  refusal that names its own `solver_*_budget` code. Each solved world carries an
  `evidence` dictionary with the block structure, resource accounting, contraction
  and spectral bounds, the Jacobian's nonsingularity basis, the unique-root basis and
  an `ill_conditioned` flag (`condition_warning`).

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

`prepared.evidence(schema, selectors)` projects retained execution onto native
evidence coordinates without refitting; the projection feeds
`evidence.decide_means(contract)`, `evidence.joint_law()` and
`evidence.decide_enclosures(contract)`. Roots acquire no implicit weights, unresolved
boxes stay inside robust utility bounds, and impossible posterior evidence refuses
a decision. See [decisions](../decisions.md) and [composition](../composition.md).

Rust uses `Study::prepare_cyclic` and `Study::prepare_cyclic_family` on the same
`PreparedStudy` handle; see [cyclic systems for developers](../development/cyclic-systems.md).


## Related

- [One feedback system, three questions](../tutorials/feedback-three-questions.md)
- [Interventions and worlds](../concepts/interventions-and-worlds.md)
- [Decision contracts](../decisions.md)
