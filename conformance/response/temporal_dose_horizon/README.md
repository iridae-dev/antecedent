# Temporal dose × horizon known-truth pin

The stationary linear temporal SCM is

```text
T_s = U_s
Y_s = 1 + 2 T_{s-1} + 3 T_{s-2}
```

with template edges `T@lag1 -> Y@lag0` and `T@lag2 -> Y@lag0`.
The frozen deterministic realization uses the period-four exogenous sequence
`U_s = [0, 1, 0, -1]` repeated for 242 observations. This sequence is not a
treatment autoregression: each `T_s` is an exogenous input. Over the exact
lag-aligned windows used by the estimator, the adjacent treatment columns have
zero mean and zero cross-product, so the two OLS fits recover the structural
coefficients exactly.

The policy is `Pulse { at: -1 }`. Horizon 1 evaluates `Y_0`, and horizon 2
evaluates `Y_1`. Therefore

```text
E[Y_0 | do(T_{-1}=d)] = 1 + 2d
E[Y_1 | do(T_{-1}=d)] = 1 + 3d
```

For doses `[0, 1]` and horizons `[1, 2]`, the dose-major surface is
`[1, 1, 3, 4]`: `value[dose_index * 2 + horizon_index]`. The flattened grid
stores coordinate pairs `[dose, horizon]` in the same order.

This truth matches `TemporalResponseEstimator`: each horizon is re-anchored as
a lagged OLS design, then linear g-computation replaces the treatment column
with the requested dose and averages the fitted rows. It does not claim a
recursive simulator or a multi-step longitudinal g-formula.

`Soft(constant=1)` has the same path `[3, 4]` as `Set(1)`.
`Soft(additive_shift=1)` averages g-computation at each observed treatment plus
one; the aligned treatment means are zero here, so its path is also `[3, 4]`.
A single-step, one-variable `Sequence` resolves to the same overlay and is
the only licensed `Sequence` shape. A multi-step or nested `Sequence` fails
closed with a stable error rather than collapsing to one step; longer
sequences and multi-step temporal policies are not evidenced by this
fixture.

At horizon 1, the two-point surface contrast
`mean(dose=1) - mean(dose=0) = 2` matches the `PulseEffect` value for
active `1` versus control `0`; both dispatch through `TemporalLinearAdjustment`
and agree numerically on this fixture. The licensed single-step
`SustainedEffect` window at offset `-1` recovers the same contrast. This is
observed numerical agreement via shared adjustment machinery, not a
derivation of one from the other and not a separate response estimand;
multi-step Sustained is not evidenced here.

Pointwise 95% bands on the surface are pinned in the fixture (`surface.lower` /
`surface.upper`); index 0 is dose zero at horizon 1 and has strictly positive
width (regression guard for the old zero-width-at-dose-0 bug).
