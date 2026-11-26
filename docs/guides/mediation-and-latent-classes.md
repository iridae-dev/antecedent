# Split an effect into direct and indirect parts, or into latent classes

Use this page to ask how an effect is built. Two outcome-model routes answer that question:

| You want | Use | You declare |
|---|---|---|
| The part of a binary treatment's effect that passes through a continuous mediator, and the part that does not | [`nonlinear_mediation.mediation_effects`](#natural-direct-and-indirect-effects-through-a-continuous-mediator) | `MediationPremises`: no unmeasured confounding and cross-world independence |
| The effect of a treatment within each of `K` unobserved classes | [`latent_class.latent_class_effects`](#per-class-effects-of-a-finite-mixture) | The number of classes and `within_class_randomization=True` |

Neither route reads a graph: the identifying premises are declarations that the data cannot test. Both return model-conditional point estimates, so the numbers are only as good as the premises and the parametric models stated below. Their bootstrap uncertainty is labelled with its calibration status, which is `not_measured` unless a coverage record covers it.

## Natural direct and indirect effects through a continuous mediator

A binary treatment `A` acts on an outcome `Y` directly and through a continuous mediator `M`. Write `M(a)` for the mediator a unit would have under treatment `a`, and `Y(a, m)` for its outcome with the treatment set to `a` and the mediator set to `m`. `mediation_effects` estimates the natural effects, averaged over the observed rows of the covariates `X`:

- natural direct effect, `NDE = E[Y(1, M(0))] - E[Y(0, M(0))]`
- natural indirect effect, `NIE = E[Y(1, M(1))] - E[Y(1, M(0))]`
- total, `NDE + NIE`

These are cross-world quantities: `Y(1, M(0))` pairs the outcome under treatment with the mediator under control, which no experiment observes for one unit. You must therefore declare the premises that identify them. `MediationPremises.sequentially_ignorable()` declares that the covariates leave no unmeasured confounding of the treatment-outcome, treatment-mediator or mediator-outcome relations, that no mediator-outcome confounder is itself affected by the treatment, and the cross-world independence `Y(a, m) _||_ M(a') | X`. A premises object that declares an unmeasured confounder, names a treatment-induced confounder or leaves the cross-world independence undeclared is refused. Only the natural effects are estimated; requesting the interventional estimand refuses. The mediator is modelled as linear-Gaussian in the treatment and covariates, and the outcome as a polynomial in the standardized mediator with a treatment interaction.

The grid below is simulated from a known structural model (`M = 0.5 A + 0.5 X + e`; `Y` quadratic in `M` with an `A*M` interaction), crossing treatment, five `X` values and four residual levels, so the answer has a closed form: NDE = 0.8 + 0.3 * 0.5 * 1 = 0.95.

```python
import math
import numpy as np
from antecedent.nonlinear_mediation import MediationPremises, mediation_effects

alpha, gamma, sigma2 = 0.5, 0.5, 0.64
rows = 2.0 * 5.0 * 4.0
v = sigma2 * (rows - 3.0) / rows
lo = 0.4
hi = math.sqrt(2.0 * v - lo * lo)
a, m, y, x = [], [], [], []
for arm in (0.0, 1.0):
    for xv in (-1.0, 0.0, 1.0, 2.0, 3.0):
        for e in (-hi, -lo, lo, hi):
            mv = alpha * arm + gamma * xv + e
            y.append(1.0 + 0.8 * arm + 0.6 * mv + 0.4 * mv**2 + 0.3 * arm * mv + 0.7 * xv)
            a.append(arm)
            m.append(mv)
            x.append(xv)

result = mediation_effects(
    np.array(a), np.array(m), np.array(y), {"x": np.array(x)},
    premises=MediationPremises.sequentially_ignorable(),
    quadrature_nodes=8,
    bootstrap_replicates=0,
)
result.natural_direct, result.natural_indirect, result.total   # NDE is 0.95
```

The integral over the cross-world mediator law is a Gauss-Hermite rule. Antecedent computes the estimate with `n` and `2n` nodes (`quadrature_nodes=n`, default 16) and reports the disagreement as `result.integration_error`; a disagreement above `integration_tolerance` (default `1e-6`, as a fraction of the outcome standard deviation) is refused rather than reported. `bootstrap_replicates=` (default 200; `0` reports no standard errors) adds seeded bootstrap standard errors and percentile bootstrap intervals for the three effects (`interval_status == "percentile_bootstrap_interval"`); each interval in `result.intervals` carries its own calibration status in `.status`. The row bootstrap describes sampling variation given the premises and model forms, not doubt about them.

`result.overlap` reports the rows in each arm and `mediator_support_violation`, the share of the control-arm mediator law that falls outside the treated arm's observed mediator range, where the outcome model extrapolates. A share above `max_support_violation` (default 0.25) or an arm with fewer than `min_arm_count` rows (default 10) is refused. `result.export()` with `consume_mediation_artifact(..., expected_identity=result.identity)` recomputes the answer from a retained artifact.

### Mediation refusals

Each is a `MediationRefusal` (a [structured refusal](../reference/refusal-codes.md#structured-refusals-and-their-remedies)) with these `code` and `detail` values:

| Code | Detail | When |
|---|---|---|
| `effect_not_identified` | `nonlinear_mediation.confounding` | The premises declare an unmeasured confounder. |
| `cross_world_not_identified` | `nonlinear_mediation.treatment_induced_confounding`, `.cross_world_independence_not_declared` | A treatment-induced mediator-outcome confounder is named, or the cross-world independence is not declared. |
| `route_not_supported` | `nonlinear_mediation.interventional_effects_not_supported` | The interventional estimand is requested. |
| `arm_not_populated` | `nonlinear_mediation.overlap` | An arm has fewer than `min_arm_count` rows. |
| `effect_not_identified` | `nonlinear_mediation.overlap` | More than `max_support_violation` of the cross-world mediator law lies outside the treated arm's observed mediator range. |
| `mechanism_fit_not_converged` | `nonlinear_mediation.integration_error` | The `n` and `2n` node estimates disagree by more than the tolerance; raise `quadrature_nodes`. |
| `design_rank_deficient` | `nonlinear_mediation.rank_deficient`, `.too_few_rows` | A regression has no more rows than coefficients, or its design is rank deficient. |

## Per-class effects of a finite mixture

`latent_class_effects` models the population as a mixture of `K` latent regimes (`2 <= K <= 4`, declared, never selected). Within class `k`, `Y | A, X ~ N(alpha_k + tau_k A + gamma_k' X, sigma_k^2)` with class weights `pi_k` that are constants. `tau_k` is the class-specific effect: the change in the class mean outcome per unit of treatment. You declare `within_class_randomization=True` as the premise that treatment is as-if randomized within a class; the route cannot check it.

The data below are simulated: two classes far enough apart that every responsibility is 0 or 1, with `pi = (0.6, 0.4)` and `tau = (1, -2)`.

```python
import numpy as np
from antecedent.latent_class import latent_class_effects

NOISE = (0.5, -0.5, 0.25, -0.25)
y, a, x = [], [], []

def push(arm, xv, mean):
    for e in NOISE:
        y.append(mean + e); a.append(arm); x.append(xv)

for arm in (0.0, 1.0):
    for xv in (-1.0, 0.0, 1.0):
        push(arm, xv, arm + 0.5 * xv)                 # class P: effect +1
for arm in (0.0, 1.0):
    for xv in (-1.0, 1.0):
        push(arm, xv, 20.0 - 2.0 * arm - 0.5 * xv)    # class Q: effect -2

result = latent_class_effects(
    np.array(y), np.array(a), {"x": np.array(x)},
    classes=2, within_class_randomization=True, seed=11, bootstrap_replicates=0,
)
[(c.weight, c.effect) for c in result.classes]   # [(0.4, -2.0), (0.6, 1.0)]
result.mixture_effect                            # -0.2
result.responsibilities                          # (n, K) posterior class probabilities
```

Labels carry no meaning, so classes are reported in a canonical order: ascending effect. `mixture_effect` is the weight-averaged class effect. `result.separation` and `result.min_effect_gap` say how well the classes are separated; `min_effect_gap == 0` means two classes are not separately identified in effect. The estimate also assumes that `K` is the true number of classes, that the Gaussian-linear class model holds and that the class weights do not vary with covariates. EM finds a local maximum from seeded restarts, not a certified global one, so the result is a fit of the declared mixture. A class effect is an average effect within an unobserved group, not an individual effect, and a class is not a subgroup you can observe: read `responsibilities` as model probabilities. Bootstrap standard errors are seeded, label-aligned and `not_measured`, and no interval is produced.

### Latent-class refusals

Each is a `LatentClassRefusal` (a [structured refusal](../reference/refusal-codes.md#structured-refusals-and-their-remedies)):

| Code | Detail | When |
|---|---|---|
| `required_option_missing` | `latent_class.randomization_not_declared` | `within_class_randomization=True` is not passed. |
| `population_not_estimable` | `latent_class.weak_class` | A class weight is below `min_class_weight` (default 0.05), or the mean largest responsibility is below `min_separation` (default 0.7). |
| `population_not_estimable` | `latent_class.degenerate_class`, `.constant_outcome` | Every start collapses a class (no separable class structure), or the outcome is constant. |
| `effect_not_identified` | `latent_class.treatment_constant` | The treatment does not vary. |
| `mechanism_fit_not_converged` | `latent_class.not_converged`, `.likelihood_not_monotone` | No start converges within `max_iterations` (default 500), or the EM log-likelihood decreases. |

## Look for different ways an intervention was delivered

If the recorded action was randomized and the way it was delivered is missing, start with [the two rollout examples](latent-soft-interventions.md#two-rollouts-with-the-same-headline-effect). They show how to compare eligible steps that could have changed after the action.

## Related

- [Estimate an effect when treatment has several levels](treatment-levels.md) for categorical and joint treatments.
- [Heterogeneous effects and policies](heterogeneous-effects-and-policies.md) when the variation is by observed covariates.
- [Reading a result](../start/reading-a-result.md) for how to read a not_measured calibration status.
- [`staged_static_kinds.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/staged_static_kinds.py) runs the graph-identified linear counterpart, `MediationEffect` with the `mediation.linear` estimator, on a declared DAG.
