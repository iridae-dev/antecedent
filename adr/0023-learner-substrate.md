# ADR 0023 — Learner substrate

- Status: Accepted
- Date: 2026-09-20
- Extends: [0001](0001-linear-algebra-backend.md)

## Context

Antecedent already owns identification, orthogonal scores, sandwich
covariance, and `ExecutionContext` (thread budget, RNG, memory, kernel
policy). Nuisance fits are hard-coded GLM/OLS/GAM. Cross-fitted AIPW copies
training rows. There is no pluggable prediction contract.

The 2.0 user problem is large tabular adjustment sets with nonlinear
covariate relationships — EconML-like flexibility — without making
Antecedent a Rust scikit-learn or a GPU tensor framework.

ADR 0001 already chose faer as the dense linear-algebra backend behind
library-owned views. A second LA or ndarray stack would fork that decision.

## Decision

Add `antecedent-learn` for **prediction**, not causality.

```text
antecedent-estimate  →  LearnerFactory / FittedPredictor
antecedent-learn     →  DesignView + providers
antecedent-stats     →  faer helpers (OLS, ridge, logistic, …)
```

`antecedent-estimate` never names Forust, SmartCore, Linfa, or Burn.

### Stack

1. **faer** remains the statistical linear-algebra backend (ADR 0001).
   Upgrade 0.21 → 0.24. Do not grow `DenseLinearAlgebra` into a generic
   tensor API. Thin internal helpers wrap operations Antecedent actually
   uses (`least_squares`, `weighted_least_squares`, `ridge_solve`, `gram`,
   `crossprod`, `stable_rank`).
2. **Antecedent-owned learner interface.** `LearnerSpec` is public
   (`Linear`, `Ridge`, `Logistic`, `GradientBoostedTrees`, `RandomForest`,
   `Auto`, `NeuralNet`). Resolution to a provider is internal. Provenance records
   `implementation` and crate version. `NeuralNet` is optional (`ml-neural` →
   `antecedent-learn-burn`) and is not part of `ml-full`.
3. **Forust** is the first nonlinear provider (milestone F): borrowed
   column-major matrix plus row index vector; squared-loss and log-loss;
   hidden behind `GradientBoostedTrees`.
4. **SmartCore** is provider #2 (milestone H): forests / ExtraTrees only.
   Not its linear algebra. Not its XGBoost (MSE-only objective).

**Not foundational:** Linfa (MSRV 1.87, ndarray), Burn / Candle / CUDA /
WGPU. GPU is a later optional provider that still emits `Vec<f64>` OOF
predictions into Antecedent scores.

### Interchange

`DesignView` is the common matrix contract: borrowed dense (wrapping
`F64MatrixView`) or sparse CSR, plus `RowSelection` (`&[u32]`). Adapters
translate. `CompiledDesign` stays the causal design in `antecedent-stats`.

`PredictionTask` is `Regression` or `BinaryProbability`. Capabilities are
checked before fit.

Learned preprocessing fits inside each cross-fit training fold.

### Parallelism

`ExecutionContext` remains the only budget owner. A later `ComputeLease`
prevents fold × learner × faer oversubscription. An Antecedent-owned Rayon
pool is a later investigation, not a prerequisite of A–C.

### First-party models

Antecedent implements OLS, WLS, ridge, logistic, elastic net, and basis
regression — the models whose inference Antecedent must control. Flexible
tree learners are adapters.

## Consequences

- New crate `antecedent-learn`; feature flags `ml-faer` (default), later
  `ml-gbdt` / `ml-forest`.
- `antecedent-estimate` depends on `antecedent-learn` only when DML/DR
  bind (milestone G). A–C do not rewire AIPW.
- Public APIs never expose foreign matrix types.
- Transport T6/X4 consume this substrate; they do not invent a second
  provider stack.
- Comparison docs and CHANGELOG product claims wait until G.
