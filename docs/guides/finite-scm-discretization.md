# Discretizing a composed Gaussian model into a finite SCM

The finite counterfactual routes (exact enumeration, finite counterfactual
transport) need variables with finitely many levels. A composed linear-Gaussian
model has a closed form. The discretization is the bounded bridge between them,
with a computed error bound against the closed form.

Rust surface: `antecedent_model::discretized_scm::DiscretizedLinearGaussian`.
Provenance record: `model.discretized_linear_gaussian_scm`.

## Method

You declare a number of levels per variable (2 to 64; never defaulted). Each
variable is cut at its `i/n` quantiles under the composition's observational
regime into equal-probability cells; the outer cells are open and the support
is not truncated. A cell's value is its conditional mean. A child's table row
for a parent configuration is the Gaussian mechanism evaluated at the parents'
cell values and integrated over the child's cells. `do(V = level)` fixes the
variable at that level's value; the closed form is computed at the same value.

## Error bound

For each variable the bound `|err_v| <= e_v + sum_p |b_vp| |err_p|` is evaluated
in topological order from the mean error `e_v` of each table. It is a computed
guarantee for the mean of every variable and for a linear contrast of means. It
says nothing about variances, probabilities or counterfactual quantities of the
discretized law.

## Label

Point-only (`BRIDGE_INFERENCE`). No interval, posterior or calibration; the
parameter uncertainty declared on the source mechanisms is not propagated
through the discretization. `PARITY_TABLE` in the module records, per
transported or derived route, which paradigm each route supports and refuses
the paradigm a request names that the route does not.

## What is refused, and the remedy

| Refused request | Detail | Remedy |
| --- | --- | --- |
| Levels not declared or outside 2..=64 | `discretized_scm.levels_undeclared`, `levels_out_of_range` | Declare levels per variable |
| Joint space too large | `discretized_scm.state_space_too_large` | Fewer levels or fewer variables |
| Non-Gaussian, finite, mixed or bounded-support mechanism | `discretized_scm.non_gaussian_mechanism`, `support_not_real_line` | Use the route native to that family |
| Mechanism loading on a shared latent cause | `discretized_scm.latent_confounding` | Use a Markovian composition |
| Interval or posterior on the discretized value | `discretized_scm.point_only` | Use the point and its bound |
| A paradigm the route is closed to | `paradigm_parity.paradigm_closed` | Use the paradigm the parity row names |
