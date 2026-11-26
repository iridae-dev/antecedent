# One feedback system, three questions

Suppose A raises B and B, in turn, raises A. When you see a high value of A, what value of B should you expect? How much would B change if you held A one unit higher? What if you gave A a one-unit push and let the feedback continue?

Two linear equations are enough to work out all three answers by hand. You’ll then ask Antecedent for each answer and examine a second loop in which an intervention leaves the equations with no solution.

Run `python examples/python/feedback_three_questions.py`, or open the [notebook](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/notebooks/feedback_three_questions.ipynb). See the [complete example](#the-complete-example).

You can check the calculations by solving two linear equations by substitution.

## Declare the system

The two mechanisms are

```text
A = 0.4 B + U_A
B = 0.5 A + U_B
```

`U_A` and `U_B` represent influences from outside the loop. They are independent, each with mean zero and variance one. Raising A by one unit raises B by 0.5. That change feeds back into A with strength 0.4 × 0.5 = 0.2.

A **world** is a named set of simultaneous changes to the system, and a question is asked of a world. World 0 changes nothing: it is the system as declared. The other worlds in this tutorial use two kinds of change:

- A **clamp** (`Set`) holds a variable at a value you choose. The clamped variable's own mechanism is deleted, so nothing flows into it any more. The loop is cut.
- A **shock** (`Shift`) adds an amount to a variable's disturbance. The mechanisms stay as they are, so the loop is still active.

That gives us two ways to intervene. We can also observe the system running on its own.

## The three questions and their answers by hand

Before running anything, derive each answer. Solve the two equations together by substituting one into the other.

**Observation: what is the slope of B on A in passive data?** From `A = 0.4 (0.5 A + U_B) + U_A` we get `A = 1.25 (U_A + 0.4 U_B)`, and then `B = 0.5 A + U_B = 0.625 U_A + 1.25 U_B`. So

```text
Var(A)     = 1.25² × (1 + 0.4²) = 1.8125
Cov(A, B)  = 1.25 × 0.625 + 0.5 × 1.25 = 1.40625
slope      = 1.40625 / 1.8125 ≈ 0.776
```

The slope exceeds both interventional answers below because A also carries information about `U_B`: a high `U_B` raises B, and B feeds back into A. A regression of B on A mixes the two routes.

**Clamp: what is the mean change in B when A is set from 0 to 1?** With A held at `a`, its equation is gone and `B = 0.5 a + U_B`, so the mean of B is `0.5 a`. Moving the clamp from 0 to 1 changes the mean of B by 0.5. The disturbance `U_B` does not depend on `a`, and there is no route from B back to a clamped A.

**Shock: what is the mean change in B after a unit mean shock to `U_A`?** A shock keeps the loop, so the push is amplified each time it goes round. In means, `E[A] = 0.4 E[B] + 1` and `E[B] = 0.5 E[A]`. Substituting, `E[A] = 1 / (1 − 0.2) = 1.25`, and `E[B] = 0.5 × 1.25 = 0.625`.

| Question in this model | Answer |
| --- | ---: |
| Slope of B on A in the passive world | ≈ 0.776 |
| Mean change in B when A is clamped from 0 to 1 | 0.5 |
| Mean change in B after a unit mean shock to `U_A` | 0.625 |

Holding A fixed cuts the path from B back to A. After a shock, that path still amplifies the initial change. In passive observations it also lets a disturbance that started in B show up in A. That is why the observational slope and the two intervention effects differ.

## Ask Antecedent

The declaration is the same two equations, written as direct effects keyed `(into, from)`: `("B", "A"): 0.5` means `B = 0.5 A + ...`. The disturbance covariance is the identity, written out as a row-major 2 × 2 matrix.

```python
model = LinearCyclicModel.from_direct_effects(
    ["A", "B"],
    {("A", "B"): 0.4, ("B", "A"): 0.5},
    noise_covariance=[1.0, 0.0, 0.0, 1.0],
)
```

Declare the four worlds: the system untouched, A clamped at 0, A clamped at 1, and a unit shock to A. Preparation only checks and retains the question; it solves nothing, and `prepared.inspect()["preparation_evaluations"]` is zero.

```python
prepared = model.worlds(
    [
        World(0),
        World(1, Set("A", 0.0)),
        World(2, Set("A", 1.0)),
        World(3, Shift("A", 1.0)),
    ],
    component_id="feedback-pair",
    source_id="specification",
    certifying_evidence=["bench-derivation"],
)
```

`source_id` records where the model came from; it does not certify the model. `certifying_evidence` names the other evidence the component rests on. Here it is a hand derivation. Executing the program returns one answer per world, keyed by world id:

```python
worlds = require_success(prepared.interventions())
observed = worlds[0]
association = observed.covariance("A", "B") / observed.covariance("A", "A")
clamp_effect = worlds[2].mean("B") - worlds[1].mean("B")
shock_effect = worlds[3].mean("B")
```

World 0 is the untouched system, so the slope comes from its covariances. The clamp effect is a difference between two worlds, A held at 1 and A held at 0. The shock effect is the mean of B in world 3, because the mean of B in the untouched system is zero.

The example asserts all three against the hand derivation to within 1e-10: `association` is 1.40625 / 1.8125, `clamp_effect` is 0.5 and `shock_effect` is 0.625. It also asserts that the shock moves the mean of A to 1.25. It then prints the three values with the two solution methods (values rounded here):

| Printed key | Value |
| --- | --- |
| `association_of_B_with_A` | 0.776 |
| `clamp_A_from_0_to_1` | 0.5 |
| `unit_shock_to_U_A` | 0.625 |
| `clamp_schedule` | `acyclic_order` |
| `shock_schedule` | `strongly_connected_blocks` |

### Different questions run different programs

Read the solution method alongside each world’s result.

```python
assert worlds[1].actions[0]["kind"] == "hard_clamp"
assert worlds[3].actions[0]["kind"] == "additive_shock"
assert worlds[1].solved_by["method"] == "acyclic_order"
assert worlds[3].solved_by["method"] == "strongly_connected_blocks"
assert all(world.certified and world.failed_mass == 0.0 for world in worlds.values())
```

A clamp turns the cyclic system into an acyclic one: A is fixed, then B follows. A shock leaves A and B in one strongly connected block that must be solved together. Each world also carries a certificate. With `K` the matrix of direct effects, the certificate Antecedent checks for these linear worlds is that the spectral radius of `|K|` is below one, which shows the equilibrium is unique. Here it is √0.2 ≈ 0.45. Every answer in this family is certified, with failed mass zero.

## An intervention with no solution

Not every intervention on a feedback system has an answer. Take three variables where A depends on B and C, and each of those depends on A:

```text
A = B + C + U_A
B = A + U_B
C = A + U_C
```

Without intervention this system has a unique equilibrium, `A = −(U_A + U_B + U_C)`. The disturbances are again independent with unit variance, the default when `noise_covariance` is omitted. The spectral radius of `|K|` is √2, above one, so the certificate is not shown. Antecedent answers world 0 but marks it uncertified when asked to (`certificate="report_uncertified"`). Without that choice the default, `certificate="required"`, is to withhold the world: withheld mass means the certificate was not shown, which is different from the world having no solution.

Now clamp C at 1. C's mechanism is deleted and `C = 1`. What remains is

```text
A = B + 1 + U_A
B = A + U_B
```

Substituting the second into the first gives `0 = 1 + U_A + U_B`. A has dropped out. For any disturbance values the equation either fails or holds for every A; for almost every draw it fails. There is no equilibrium.

Declare both worlds and ask Antecedent:

```python
model = LinearCyclicModel.from_direct_effects(
    ["A", "B", "C"],
    {("A", "B"): 1.0, ("A", "C"): 1.0, ("B", "A"): 1.0, ("C", "A"): 1.0},
)
worlds = require_success(
    model.worlds(
        [World(0), World(1, Set("C", 1.0))],
        component_id="overlapping-loops",
        source_id="specification",
        certifying_evidence=["bench-derivation"],
        certificate="report_uncertified",
    ).interventions()
)
assert list(worlds) == [0, 1]
assert not worlds[0].certified
```

Both worlds come back. World 0 answers and is marked uncertified. The clamped world is kept instead of dropped:

```python
clamped = worlds[1]
assert clamped.status == "no_solution"
assert clamped.failed_mass == 1.0
assert clamped.solved_by["method"] == "not_solved"
```

The whole probability mass of the world is failed mass, and reading a mean from it raises a refusal rather than returning a number:

```python
try:
    clamped.mean("B")
except CyclicRefusal as refusal:
    next_attempt = refusal.next_attempt
```

For the clamped world the script prints `status = no_solution` and `failed_mass = 1.0`. To proceed, you need to specify what happens when the system cannot settle. The remedy depends on the equations.

- A **dynamic law** says how the loop evolves in time. The question then becomes where it is at a stated time rather than where it settles.
- A **selection rule** picks one state among several when there are several. Here there are none, so this applies to nonlinear loops with multiple roots rather than to this linear one.
- A **mechanism change** asks a different question: instead of fixing C, change how C responds to A so that the remaining system has a solution.

A world with no solution, a world whose certificate was withheld, and a world that answered but is uncertified are three distinct states. They are reported with distinct masses (failed, withheld, solved) and next attempts, and a world without a solution is never reported as a number.

## Continue with the result

A result can leave the process that produced it. Export the executed program, then consume the artifact against the identity you stored separately. The consumer recomputes the law and checks it against that identity:

```python
result = require_success(prepared.execute())
replayed = require_success(
    consume_family_artifact(bytes(result.export()), expected_identity=result.identity)
)
assert replayed.to_dict() == result.to_dict()
```

A successful replay shows that the consumer reproduced the calculation from the saved equations. Testing whether those equations describe an actual feedback loop takes observations or experiments.

To learn a feedback law from data instead of declaring it, see [`examples/python/recovered_feedback_law.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/recovered_feedback_law.py): it recovers the same two-variable law from a passive sample and two randomized clamp experiments, and attaches fit uncertainty to the answers. The [cyclic systems guide](../guides/cyclic-systems.md) covers nonlinear loops, shared disturbances and experiment-based fitting.

## Compare actions after fitting the feedback

Now suppose you have trial rows instead of a fully specified loop. In the Rust example, advertising was fixed at several known levels and the trial recorded wishlists and traffic. The model fits one feedback coefficient from those rows. Compare ordinary and promoted placements using the same fitted coefficients and background disturbances in both worlds. This lets the decision ask how those two choices differ under the same conditions.

Read the fitting diagnostics and the number of paired outcome draws before interpreting the verdict. The declared noise covariance and the feedback equations are premises of this example. The consumer rebuilds the fit and checks the decision against the identity saved at export. To see how replay differs from current permission to use a result, continue with [Replay old numbers, refuse current use](replay-and-current-decisions.md#replay-a-fitted-decision).

Run `cargo run -p antecedent --example fitted_feedback_decision`.

[Read the complete Rust example](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/fitted_feedback_decision.rs).

## What this established

In this loop the observational slope is about 0.776. Raising the clamp by one unit changes mean B by 0.5. A one-unit shock changes it by 0.625 because the feedback continues. In the three-variable example, clamping C leaves no equilibrium and Antecedent reports the failed world.

## What this did not establish

None of this says that the equations describe a real system: every number is a consequence of the declared model. The certificate shows that the declared equations have a unique equilibrium, not that the coefficients are right, and a replayed artifact reproduces the computation, not its premises.

## Where to go next

The next question is where the equations come from. Experiments that clamp one variable at a time can identify each direction of a linear loop under a declared recipe, with fit uncertainty that is a statement about the fitted law and not about the world. The [cyclic systems guide](../guides/cyclic-systems.md) shows that route.

## The complete example

<!-- example: examples/python/feedback_three_questions.py -->

[Open the Python script](../../examples/python/feedback_three_questions.py)

The same computation is available in Rust as [`examples/rust/feedback_three_questions.rs`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/feedback_three_questions.rs) through the cyclic world API of `antecedent-model`. It asserts the same three values and the same failed mass.

```sh
cargo run -p antecedent --example feedback_three_questions
```
