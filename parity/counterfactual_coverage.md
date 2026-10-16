# Counterfactual coverage

| Query | Graph and evidence | Execution | Uncertainty or refusal |
| --- | --- | --- | --- |
| Existing unit-level counterfactual | Licensed fixed Markovian DAG and compatible SCM evidence | Shared-exogenous abduction, action, prediction | Existing support registry governs inference and validation |
| Natural direct effect `Y₁,M₀ − Y₀,M₀` | Explicit three-node Markovian `X→M`, `X→Y`, `M→Y` DAG; tabular linear Gaussian SCM; Frequentist; no refutation suite | One abduced exogenous table shared by both worlds; known linear SCM target truth; Rust and Python staged execution and artifact replay | Point-only; no sampling interval reported |
| Per-unit abduced disturbance read, not a conditional-mean plug-in | Same fixed nested DAG; non-separable outcome basis fit by the route (`fit_non_separable_nested_outcome`, `LinearSpline`) | The frozen mediator carries each unit's abduced disturbance into a non-additive outcome, so the direct effect reads that per-unit draw; the route matches the per-unit abduction truth and differs from the conditional-mean plug-in value | Point-only; no interval |
| Incompatible nested worlds | Counterfactual query outside the named compatible mediation contract | Refused | Stable `cross_world_not_identified` reason |
| Nested effect on the static slice with an accepted, graph-posterior or latent-confounded (ADMG, CPDAG, PAG) structure | Structure outside the supplied fixed explicit static DAG contract (`parity/support_closed.toml`) | Refused | No counterfactual-ID or transported claim |
| Nested effect on a temporal structure | Separate cells in `parity/support_licensed.toml` (`TemporalDag`, `TemporalCpdag`, `TemporalPag`, one-lag unfolded linear points); not this row's fixed Markovian mediation DAG | Exactly the licensed coordinates there; every other temporal coordinate refused (`parity/support_closed.toml`) | Point-only; no interval; no fixed-population cross-world or transported claim is inferred from it |
| Natural direct effect, same explicit three-node DAG, Bayesian conjugate configuration (`parity/support_licensed.toml`, `NestedCounterfactualEffect` / `Dag` / `explicit` / `Bayesian`; evidence `bayesian_natural_direct_effect_publishes_the_posterior_mean_without_an_interval` in `crates/antecedent/tests/nested_counterfactual_route_evidence.rs`) | Same fixed nested DAG; isotropic conjugate Gaussian prior of the Study configuration; a shared coefficient prior, an unmapped prior artifact or an external power prior is refused by the mediation prior contract | The posterior mean of the same effect; posterior draws are formed to compute that mean and then dropped | Point-only: no posterior interval, no posterior object and no coverage record. The mean is not a retained joint posterior SCM and is not a cross-world law for any other query (it is not a source for general cross-world execution); `result.interval` and `result.posterior` are absent |
| Nested effect with cheap or full validation | No compatible cross-world refuter suite (`parity/support_closed.toml`) | Refused | No calibration or refutation claim |
| Other path-specific or transported counterfactuals, and temporal coordinates not licensed above | Contracts not established by this slice | Refused | No inferred cross-world composition |

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

No additional compatible fixed-DAG cell is required:
the natural direct effect is the one named fixed-DAG nested cell, and the
route's use of per-unit abduced disturbances (not conditional-mean plug-ins) is
substantiated on that same fixed DAG by the non-separable evidence above (an execution-engine capability of the route,
not a separately licensed Study coordinate — the licensed frequentist cell stays
linear-Gaussian and point-only). Natural indirect effects, other nested world
pairs, and transported nested outcomes remain separate future cells and must not
be inferred from this row.

## Promoted cells

| Query | Graph and evidence | Execution | Uncertainty or refusal |
| --- | --- | --- | --- |
| Path-specific edge-intervention contrast `E[Y₁] − E[Y₀]` (`parity/promotion.toml`, `counterfactual.path_specific_edge`); the natural direct effect is the edge set `{X→Y}` and the natural indirect effect `Y(0, M(1)) − Y(0, M(0))` is `{X→M, M→Y}` | A supplied explicit Markovian DAG of at most eight fully observed variables; a tabular sample; linear-Gaussian mechanisms, or a non-separable outcome basis. No latent confounding, accepted, uncertain, equivalence-class or temporal structure | Stage API `antecedent::cross_world` / `antecedent.cross_world` (not a Study query, so no support-matrix coordinate): check the query on the graph and return a witness; one coupled abduction-action-prediction over one exogenous term per unit and variable shared by both worlds, each edge reading its parent from the world its route names; result with the witness; artifact a consumer replays (same evaluator, plus a separate closed-form recomputation of the point for both mechanism families: ordinary least squares, or a Givens-QR ridge-stabilized basis regression) | Point-only; no sampling interval or posterior. Edge sets with a recanting witness refuse `cross_world_not_identified` (the one nonidentification finding); query shapes outside the two-world contract (three worlds, a mixed-outcome contrast, a baseline that reads another world) refuse `route_not_supported` (`cross_world.query_outside_contract`), not as unidentified; structures outside the cell refuse `cell_not_licensed`; interval or Bayesian requests refuse `estimator_inference_mismatch` |
| Effect of treatment on the treated `P(Y_x = y \| X = x′)`, `x ≠ x′`, its distribution over `Y` and the contrast `E[Y_x \| X = x′] − E[Y \| X = x′]` (`parity/promotion.toml`, `counterfactual.admg_identification`) | A supplied explicit ADMG (or DAG) of at most six finite-discrete variables with at most four levels each, one treatment and one outcome; the observational joint as an exact law or a count table. Only this query shape: not general counterfactual identification on ADMGs | Stage API `antecedent::counterfactual_id` / `antecedent.counterfactual_id`: ID* on `{Y_x = y, X = x′}` with each district term identified from `P(V)` by ID, under one bounded search budget; exact evaluation of the functional; `counterfactual_id_admg_v1` artifact replayed under its stored limits with a separate direct-sum recomputation | Point-only; no interval or posterior. Sound (enumerated latent-SCM truth). When ID* stops and the treatment is binary, the consistency complement `P_x(y) − P(y, X = x)` answers wherever ID identifies `P_x(y)`. A query neither route identifies refuses `route_not_supported` (`counterfactual_id.conflicting_subscripts` or `counterfactual_id.counterfactual_hedge`) with a checkable obstruction: not identified by ID*, NOT a proof of non-identifiability (ID* composed with ID is not complete from `P(V)`); completeness from `P(V)` is not claimed. Other shapes refuse `route_not_supported`; path-specific effects on ADMGs are not licensed (`counterfactual_id.path_specific_deferred`); a within-one-world contradiction is an exact zero |

What the cell is, and is not. A cross-world query is an explicit value: worlds (each with hard interventions and per-edge routes), the shared-exogenous coupling and the observed contrast; the graph enters only when the query is checked. The check returns a machine-checkable witness that names the assumptions (`consistency`, `markovian_no_latent_confounding`, `mechanism_invariance_across_worlds`, `no_recanting_witness`), the rerouted edges, the coupled worlds and the counterfactual nodes the estimand needs; the artifact stores it and the consumer recomputes it from the stored graph and query and requires it to equal the stored one. The evidence in `crates/antecedent/tests/cross_world_edge_contrast.rs` and `python/tests/test_cross_world.py` covers exact SCM truth for all eight edge sets of the mediation DAG; a non-separable mechanism on which per-unit shared abduction is observable (the answer matches the per-unit structural truth and differs from the conditional-mean plug-in); per-unit contrasts that use one shared exogenous term across worlds for the mediator and for a noisy outcome (checked on the unit effects, since the mean point is permutation invariant), on both the linear-Gaussian and the non-separable mechanism, in Rust and in Python; an edge-level answer compared with the node-level counterfactual total from the existing counterfactual engine (equal for every edge, off by exactly the omitted path's effect otherwise, neither the total nor additive under interaction); exact reproduction of the existing natural direct effect cell on the exact, the noisy and the non-separable fixtures; the recanting-witness refusal and the out-of-contract refusals; well-posed-abduction refusals; cancellation before the fit, between nodes of the evaluation and at chosen polls in the middle of the fit; the independent recomputation on both families (agreement on every edge set, a mutated mechanism it catches and replay does not, designs it declines); and typed artifact mutation tests, including re-sealed semantic edits, a data digest that binds the table, and the row bound enforced at export and at consumption.

What is and is not enforced. The witness names the assumption `consistency` (a unit's observed values are its values under the treatment it received). In this pipeline it holds by construction: abduction defines each exogenous term as the residual that regenerates the observed value, so a factual round trip would be a tautology and is not performed. It, the Markovian assumption, mechanism invariance and correct specification of the fitted mechanisms are named premises that no test or check in this cell verifies from the data (a test pins that mechanisms fitted on one SCM evaluate data from another without refusal). What the code enforces is well-posed abduction (exact inversion of an invertible mechanism; otherwise `invalid_argument` / `cross_world.invalid_query`), the recanting-witness criterion, the graph and query contract, and the artifact digests. Replay by the artifact consumer detects corruption and re-sealed edits that change the derivation or the point; it does not detect a producer that seals a wrong table or query on purpose, and a bug shared by the fitter and the evaluator is caught only by the separate recomputation, which covers both mechanism families (skipped, with `independently_verified` false, for a singular or ill-conditioned design). Cancellation is polled before every node, candidate family and cross-validation fold of the fit and before every node of the evaluation; one fold's design solve is not interrupted. It is not general counterfactual identification: latent-confounded (ADMG) structure and the identification of arbitrary nested queries are not claimed here and remain outside this cell. The existing natural direct effect Study cell above is unchanged.
