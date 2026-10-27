# One feedback system, three questions

Two quantities push on each other: A rises when B rises, and B rises when A rises. Ask "what does A do to B?" of such a system and the answer depends on what you mean. This tutorial asks three different questions of one feedback system, gets three different answers, and shows why each is the right answer to its own question. It then asks a fourth question that has no answer, and shows how Antecedent keeps that visible instead of dropping it.

Nothing in this tutorial is estimated from data. The system is declared by its equations, so every number is an exact consequence of those equations: a fixture truth of the stated model, not a claim about any real system.

## Declare the system

The two mechanisms are

```text
A = 0.4 B + U_A
B = 0.5 A + U_B
```

`U_A` and `U_B` are disturbances: everything that moves a variable apart from the other one. Here they are independent, centered at zero, with variance one. The loop gain is 0.4 × 0.5 = 0.2: a push on A comes back to A, reduced to a fifth.

A **world** is a named set of simultaneous changes to the system, and a question is asked of a world. World 0 changes nothing: it is the system as declared. The other worlds in this tutorial use two kinds of change:

- A **clamp** (`Set`) holds a variable at a value you choose. The clamped variable's own mechanism is deleted, so nothing flows into it any more. The loop is cut.
- A **shock** (`Shift`) adds an amount to a variable's disturbance. The mechanisms stay as they are, so the loop is still active.

Clamping and shocking are different interventions on the same system and ask different questions.

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

Three questions, three answers. The clamp cuts A's incoming mechanism, so it removes the feedback. The shock leaves the loop active, so it keeps it. The observational slope answers neither: it describes how the two variables co-vary when the system runs untouched.

## Ask Antecedent

The declaration is the same two equations, written as direct effects. `("B", "A"): 0.5` means `B = 0.5 A + ...`; the disturbance covariance is the identity.

```python
model = LinearCyclicModel.from_direct_effects(
    ["A", "B"],
    {("A", "B"): 0.4, ("B", "A"): 0.5},
    noise_covariance=[1.0, 0.0, 0.0, 1.0],
)
```

Declare the four worlds. Preparation only checks and retains the question; it solves nothing, and `inspect()` reports `preparation_evaluations` of zero.

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

The example asserts all three against the hand derivation to within 1e-10: `association` is 1.40625 / 1.8125, `clamp_effect` is 0.5 and `shock_effect` is 0.625. It also asserts that the shock moves A by 1.25.

### Different questions run different programs

The questions are not just different numbers. Each world reports how it was solved:

```python
assert worlds[1].solved_by["method"] == "acyclic_order"
assert worlds[3].solved_by["method"] == "strongly_connected_blocks"
```

A clamp turns the cyclic system into an acyclic one: A is fixed, then B follows. A shock leaves A and B in one strongly connected block that must be solved together. Each world also carries a certificate. The loop gain gives a spectral bound of √0.2 ≈ 0.45, below one, so the equilibrium is unique; showing that bound is the certificate Antecedent checks for these linear worlds. Every answer here is certified, with failed mass zero.

## An intervention with no solution

Not every intervention on a feedback system has an answer. Take three variables where A depends on B and C, and each of those depends on A:

```text
A = B + C + U_A
B = A + U_B
C = A + U_C
```

Without intervention this system has a unique equilibrium, `A = −(U_A + U_B + U_C)`. Its loop gain is large enough that the contraction bound is not shown, so Antecedent answers world 0 but marks it uncertified when asked to (`certificate="report_uncertified"`). Without that choice the default is to withhold the world: withheld mass means the certificate was not shown, which is different from the world having no solution.

Now clamp C at 1. C's mechanism is deleted and `C = 1`. What remains is

```text
A = B + 1 + U_A
B = A + U_B
```

Substituting the second into the first gives `0 = 1 + U_A + U_B`. A has dropped out. For any disturbance values the equation either fails or holds for every A; for almost every draw it fails. There is no equilibrium.

Antecedent keeps this world instead of dropping it:

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

The refusal names what to do next. For a singular clamped loop the next attempt is to declare an equilibrium selection rule or a dynamic law for the clamped loop, or to intervene with a mechanism change that keeps the free block regular. Each of these is a modelling choice that you make, because the model as declared says nothing about where a clamped system with no equilibrium goes:

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

A replay recomputes the same numbers. It does not turn a model into evidence: the artifact says what was asked of what, not that the equations describe a real system.

To learn a feedback law from data instead of declaring it, see `examples/python/recovered_feedback_law.py`: it recovers the same two-variable law from a passive sample and two randomized clamp experiments, and attaches fit uncertainty to the answers. The [cyclic systems guide](../guides/cyclic-systems.md) covers nonlinear loops, shared disturbances and experiment-based fitting.

## The complete example

<!-- example: examples/python/feedback_three_questions.py -->

The same computation is available in Rust as `examples/rust/feedback_three_questions.rs` through the cyclic world API of `antecedent-model`. It asserts the same three values and the same failed mass.

```sh
cargo run -p antecedent --example feedback_three_questions
```

## What this established, and what comes next

In the stated feedback model, the passive slope is about 0.776, clamping A from 0 to 1 changes the mean of B by 0.5, and a unit mean shock to `U_A` changes it by 0.625. The three differ because the interventions differ, not because of any estimation error: nothing was estimated. A clamp that leaves a loop with no equilibrium is retained as failed mass with a named next attempt.

The next question is where the equations come from. Experiments that clamp one variable at a time can identify each direction of a linear loop under a declared recipe, with fit uncertainty that is a statement about the fitted law and not about the world. The [cyclic systems guide](../guides/cyclic-systems.md) shows that route.
