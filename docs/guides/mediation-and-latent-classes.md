# Split an effect into direct and indirect parts, or into latent classes

Two outcome-model routes answer questions about how an effect is built. `antecedent.nonlinear_mediation` splits the effect of a binary treatment into the part that passes through a continuous mediator and the part that does not. `antecedent.latent_class` asks for the effect of a treatment within each of a declared number of unobserved classes. Both are model-conditional point estimates: the numbers are only as good as the premises you declare, and every standard error is a diagnostic with calibration `unmeasured`.

## Natural direct and indirect effects through a continuous mediator

A binary treatment `A` acts on an outcome `Y` directly and through a continuous mediator `M`. `mediation_effects` estimates the natural effects, averaged over the observed covariate rows:

- natural direct effect, `NDE = E[Y(1, M(0))] - E[Y(0, M(0))]`
- natural indirect effect, `NIE = E[Y(1, M(1))] - E[Y(1, M(0))]`
- total, `NDE + NIE`

These are cross-world quantities, so you must declare the premises that identify them. `MediationPremises.sequentially_ignorable()` states sequential ignorability and the cross-world independence; confounding flags you leave out, a treatment-induced confounder and an undeclared cross-world assumption are refused, and the interventional estimand is closed. The mediator is modelled as linear-Gaussian in the treatment and covariates, and the outcome as a polynomial in the standardized mediator with a treatment interaction.

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

The integral over the cross-world mediator law is a Gauss-Hermite rule. Antecedent computes the estimate with `n` and `2n` nodes and reports the disagreement; a disagreement above the tolerance is refused rather than reported. `bootstrap_replicates=` adds a seeded bootstrap standard error with interval status `closed_calibration_unmeasured`: no interval is produced. Weak overlap is refused. `result.export()` with `consume_mediation_artifact(..., expected_identity=result.identity)` recomputes the answer from a retained artifact.

## Per-class effects of a finite mixture

`latent_class_effects` models the population as a mixture of `K` latent regimes (`2 <= K <= 4`, declared, never selected). Within class `k`, `Y | A, X ~ N(alpha_k + tau_k A + gamma_k' X, sigma_k^2)` with class weights `pi_k` that are constants. `tau_k` is the class-specific effect. You declare `within_class_randomization=True` as the premise that treatment is as-if randomized within a class; the route cannot check it.

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

Labels carry no meaning, so classes are reported in a canonical order: ascending effect. `mixture_effect` is the weight-averaged class effect. `result.separation` and `result.min_effect_gap` say how well the classes are separated, and a class weight too small to estimate refuses. EM finds a local maximum from seeded restarts, not a certified global one, so the result is a fit of the declared mixture. A class effect is an average effect within an unobserved group, not an individual effect, and a class is not a subgroup you can observe: read `responsibilities` as model probabilities. Bootstrap standard errors are seeded, label-aligned and `unmeasured`.

## Related

- [Estimate an effect when treatment has several levels](treatment-levels.md) for categorical and joint treatments.
- [Heterogeneous effects and policies](heterogeneous-effects-and-policies.md) when the variation is by observed covariates.
- [Reading a result](../start/reading-a-result.md) for how to read an unmeasured calibration status.
