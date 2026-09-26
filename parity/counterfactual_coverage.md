# Counterfactual coverage for the 2.1 development branch

| Query | Graph and evidence | Execution | Uncertainty or refusal |
| --- | --- | --- | --- |
| Existing unit-level counterfactual | Licensed fixed Markovian DAG and compatible SCM evidence | Shared-exogenous abduction, action, prediction | Existing support registry governs inference and validation |
| Natural direct effect `Y₁,M₀ − Y₀,M₀` | Explicit three-node Markovian `X→M`, `X→Y`, `M→Y` DAG; tabular linear Gaussian SCM; Frequentist; no refutation suite | One abduced exogenous table shared by both worlds; known linear SCM target truth; Rust and Python staged execution and artifact replay | Point-only; no sampling interval reported |
| Per-unit abduced disturbance read, not a conditional-mean plug-in | Same fixed nested DAG; non-separable outcome basis fit by the route (`fit_non_separable_nested_outcome`, `LinearSpline`) | The frozen mediator carries each unit's abduced disturbance into a non-additive outcome, so the direct effect reads that per-unit draw; the route matches the per-unit abduction truth and differs from the conditional-mean plug-in value | Point-only; no interval |
| Incompatible nested worlds | Counterfactual query outside the named compatible mediation contract | Refused | Stable `cross_world_not_identified` reason |
| Nested effect with accepted, uncertain, latent-confounded, or temporal graph | Graph/evidence outside the fixed explicit DAG contract | Refused | No counterfactual-ID or transported claim |
| Nested effect with Bayesian inference or cheap/full validation | No joint posterior SCM or licensed refutation route | Refused | No posterior or calibration claim |
| Other path-specific, temporal, or transported counterfactuals | Contracts not established by this slice | Refused | No inferred cross-world composition |

The named cell has deterministic evidence against its known SCM target truth, frozen shared-world
semantics, contract/body inconsistency rejection, fitted-snapshot tampering, and
artifact replay/consumption in
`crates/antecedent/tests/nested_counterfactual_route_evidence.rs`. The support
registry remains the coordinate-level license; a missing implementation is not
a zero-probability counterfactual event.

Under the licensed separable linear-Gaussian outcome the natural direct effect
is `a·(active − control)` and does not read the abducted disturbance, so sharing
one exogenous draw across both worlds cannot change the point value: that cell
alone cannot exhibit the shared-exogenous property. The property is made
observable and proven discriminating with a non-separable outcome mechanism the
route can fit (`NestedCounterfactualOperation::with_non_separable_outcome`,
`crates/antecedent/src/gcm.rs`). On a structural model whose outcome is a
non-additive function of treatment and mediator (a treatment-interacted mediator
term), the frozen mediator carries its unit-level abduced disturbance into the
direct effect. The evidence tests
`nested_route_shares_abduced_exogenous_draw_under_nonseparable_outcome`,
`nested_shared_equals_independent_under_separable_outcome`, and
`nested_non_separable_outcome_refuses_when_basis_underdetermined`
(all in `crates/antecedent/src/gcm.rs`) show the route's value equals the
shared-abduction truth (forward-simulated independently through the fitted SCM),
differs by more than a fixed margin from the disturbance-free value that a
non-shared exogenous draw would produce, collapses to no difference when the
outcome is separable or the mediator carries no disturbance, and refuses with a
typed error rather than fabricating an effect when the basis is
underdetermined. For the mean natural direct effect, sharing the abduced draw across the two
worlds versus drawing it independently is immaterial by linearity of
expectation: an independent per-unit redraw with the same marginal yields the
same mean, for any outcome mechanism. What the route reads is each unit's
abduced disturbance rather than a conditional-mean plug-in, and that changes the
mean only when the outcome is non-additive in the mediator (a treatment-
interacted convex term). The evidence tests discriminate exactly that per-unit
abduction against the plug-in; they do not, and could not, distinguish shared
from independent draws for this point estimand.

No additional compatible fixed-DAG cell is required for the 2.1 release claim:
the natural direct effect is the one named fixed-DAG nested cell, and the
route's use of per-unit abduced disturbances (not conditional-mean plug-ins) is
substantiated on that same fixed DAG by the non-separable evidence above (an execution-engine capability of the route,
not a separately licensed Study coordinate — the licensed frequentist cell stays
linear-Gaussian and point-only). Natural indirect effects, other nested world
pairs, and transported nested outcomes remain separate future cells and must not
be inferred from this row.
