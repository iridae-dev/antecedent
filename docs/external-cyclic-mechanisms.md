# Registered external cyclic mechanisms

A feedback model may use a mechanism the library does not own. There are two ways to
supply one, and they differ in what an artifact may claim.

| | Process-local callback | Registered external mechanism |
| --- | --- | --- |
| Where the code lives | The host process | The host process, registered by the caller |
| What a saved artifact records | Nothing: `export` is refused (`policy_callback_not_portable`) | A mechanism identity: implementation id, version, content digest, dependencies, signature, parameter schema, derivative declaration, Lipschitz bounds |
| Independent replay | Impossible | Possible with a caller-supplied trusted resolver, after verification |
| Standing | Numerical only | `verified_external`; never `native` |

The process-local path ([`LocalCyclicModel`](guides/cyclic-systems.md)) is unchanged.

## What is recorded

`ExternalMechanismSpec` (Rust, `antecedent_model::external_mechanism`) is the whole record:

| Field | Meaning |
| --- | --- |
| `implementation_id`, `version` | Versioned name of the implementation. |
| `content_digest` | BLAKE3 of the implementation's content (its source, bytecode or any canonical byte description the caller chooses). |
| `dependencies` | Declared dependencies, each with a version and digest. |
| `reads` | Endogenous coordinates read, in input order. The output is one finite scalar. |
| `parameter_names` | The parameter schema; fitted values live in the system, not in the implementation. |
| `derivative` | `analytic` (the implementation supplies an input gradient) or `finite_difference` with a relative step in `(0, 1]` and a `forward` or `central` scheme. |
| `input_lipschitz` | Optional global bounds on `|d output / d input|`. |

The artifact (`external_cyclic_equilibrium`) stores this record, the fitted parameters, the
fit request and the answers. It has no field that could carry an implementation, and the
encoder has no input that could supply one. `inspect_external_cyclic_artifact` reads the
recorded identities without a resolver and runs nothing.

## Replay needs a trusted resolver

Replay takes a `TrustedMechanismResolver` (Rust trait) or, in Python, a
`MechanismRegistry` or any callable `resolver(spec) -> implementation | None`. The caller's
act of supplying a resolver is the act of trust: this library does not sandbox the code it
runs. What the library adds is verification. Before anything runs, the resolved
implementation must equal the record in its id, version, content digest, dependencies and
declarations. Otherwise replay is refused and nothing is executed:

| Situation | Reason code |
| --- | --- |
| The artifact names an external mechanism and no resolver was supplied | `mechanism_resolver_missing` |
| The resolver has no implementation for the recorded id and version | `mechanism_implementation_unavailable` |
| The implementation differs in version, content digest, dependencies, signature, parameter schema, derivative declaration or Lipschitz bounds | `mechanism_implementation_changed` |
| The resolver itself failed (raised, or returned a non-implementation) | `mechanism_resolver_failed` |
| A declaration is malformed or incomplete | `mechanism_declaration_invalid` |

The digest binds replay to the exact content the caller registered. It does not make that
content correct or safe.

## Fitting

`fit_external_feedback` maximizes the joint equilibrium likelihood from known hard-clamp
rows. The likelihood needs `log |det(I - J)|`:

* The Jacobian uses the declared analytic input gradient, or the declared finite-difference
  step and scheme.
* The Gaussian data term has an analytic parameter score when the implementation supplies a
  parameter gradient, otherwise a central difference.
* The score of the determinant term is analytic when the mechanism declares analytic
  derivatives and supplies the mixed derivatives of its input gradient
  (`input_gradient_parameter_derivative`): `-sum_j (M^-1)_{j,e} dJ_{e,j}/dtheta` with
  `M = I - J` of the equation's block. Otherwise it is a central difference in the fitted
  parameters with the fit policy's relative `difference_step`. The fit records the basis
  (`score_basis`: `analytic`, `analytic_data_term_central_difference_determinant` or
  `central_difference`).
* A density is normalized only when every free block has a global contraction. The
  contraction bound comes from the declared `input_lipschitz` (and the built-in mechanisms'
  own bounds). A mechanism without a bound gives an infinite bound, so the likelihood is
  refused (`Uncertified`) rather than treating one inverse branch of a multiple-equilibrium
  system as a density.

Local convergence is not a global optimum, and the fit is conditional on the specified graph,
covariance and mechanism family.

## Solvers

* **Equilibrium.** `ResolvedFeedbackSystem::solve` takes the same `SolvePolicy` as the built-in
  solver and applies its budgets (Jacobian nonzeros, block size, memory, factor fill, total
  iterations, wall clock), resource admission, cancellation and sparse-LU Newton, and fills the
  same `SolveEvidence` fields, using the verified mechanisms' derivatives. Existence and uniqueness are marked only when the declared
  Lipschitz bounds give a block infinity-norm contraction below one; otherwise the root is
  numerical. The certificate is as trustworthy as the declaration, and the evidence says it
  is declared (`lipschitz_basis: declared`).
* **ODE.** `ExternalOdeSystem` uses the same Dormand-Prince integrator as the built-in
  `OdeSystem`, with timed hard clamps. A declared drift Lipschitz bound caps the first
  proposed step at its reciprocal and is reported as `drift_lipschitz`; error control governs
  every step.

## Standing

A system with an external mechanism is `verified_external` in the report, the evidence
records and the claim string (`verified_external_model_conditional`). No path converts it to
`native`; native means every mechanism is a built-in, serializable library mechanism.
Calibration is `unmeasured`.

## Python

```python
import math
from antecedent import external_mechanisms as em
from antecedent.cyclic import Equation

registry = em.MechanismRegistry()
gain = em.register_mechanism(
    "lab/saturating_gain", "1.0",
    lambda x, p: p[0] * math.tanh(x[0]),
    reads=[1], parameters=["gain"],
    input_gradient=lambda x, p: [p[0] * (1 - math.tanh(x[0]) ** 2)],
    input_lipschitz=[0.9],
    registry=registry,
)
model = em.ExternalFeedbackModel(
    equations=(em.ExternalEquation("response", gain, parameters=(0.0,)),
               Equation("feedback", 0.0, (0.2, 0.0), "tanh")),
    noise_covariance=((1.0, 0.0), (0.0, 1.0)),
)
artifact = model.export("fit", worlds, noise, resolver=registry,
                        fit=em.ExternalFeedbackModel.fit_spec(targets, rows))
# Later, in another process, with a fresh registry that registers the same implementation:
report = em.consume_artifact(artifact.data, artifact.identity, resolver=fresh_registry)
```

`content` defaults to the source of the callables; pass `content=` when source is unavailable.
`inspect_artifact(data)` shows the recorded identities without any resolver. The generic `antecedent.load(data, expected=identity)` knows the `external_cyclic_equilibrium` kind but has no way to pass a resolver, so it refuses with `mechanism_resolver_missing`; use `consume_artifact(..., resolver=)`. `execute`/`export` take `require_contraction` (default false): true refuses any block without a declared contraction instead of using Newton.

## Reference

`conformance/gcm/external_mechanism_feedback/reference.py` derives the fitted gain
(least-squares slope), the fixed points (bisection) and a joint log density (explicit 2x2
determinant) with the standard library only; the tests compare the library with it.
