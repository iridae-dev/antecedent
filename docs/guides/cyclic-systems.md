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
