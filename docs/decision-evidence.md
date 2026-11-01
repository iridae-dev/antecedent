# Experimental evidence for decision-relevant outcomes

An experiment often estimates the effect of a change on several outcomes at once (order frequency, basket value, unit cost) while the decision turns on a quantity built from all of them. This page describes what Antecedent keeps at each step from estimate to decision, and what it refuses.

Rust: `antecedent::decision_evidence`. Python: `antecedent.decision_evidence`.

## The five steps

| Step | Object | What it says | What it never says |
| --- | --- | --- | --- |
| Compile | `compile_joint_experiment` | the unit of analysis, which adjustments the design admits, and the identification and statistical obligations, before anything is fitted | that any obligation holds |
| Estimate | `JointExperimentEvidence` | the arm means of several outcomes of one experiment with the covariance of their estimators, the design, the exposure assumption | a joint law when only marginals were estimated |
| Derive | `derive`, `DerivedQuantity` | a function of the outcomes in the decision contract's own `UtilityExpr` language: a delta-method sampling interval and, separately, the range over declared assumptions | a support license or a calibrated interval; the components' precision does not transfer to a nonlinear function of them |
| Decide | `decide`, `DecisionFrame` | `robust`, `assumption_dependent` or `unresolved`, and what is unresolved; assumption-dependent classifications can include unresolved intermediate values | a probability; an assumption range is never turned into one |
| Plan | `plan_evidence` | what additional units or a narrowed assumption would do to the decision | one ranking across kinds that are not comparable |

## Design-aware precision adjustment

Fixed-coefficient CUPED composes with complete cluster randomization (`fixed_cuped_cluster_totals`; licensed row in the graphless support matrix). The adjustment is applied to cluster totals, so the cluster stays the unit of analysis and the estimand stays the unadjusted per-row contrast. A coefficient learned from the experiment's own outcomes is not admissible for clusters: fitting it on rows would treat rows as independent and change the unit of analysis. `compile_joint_experiment` lists both verdicts with reasons.

## Joint estimates

`complete_cluster_joint_contrast` estimates the per-row arm mean of every outcome and the full covariance of those estimators from the same assigned clusters (the conservative Neyman covariance; it dominates the exact one for every linear combination). Dropping the cross-component covariance changes the answer for a margin; a test pins that.

Evidence from different experiments combines only through `merge_independent`, and only when independence is declared and the premise (population, regimes, horizon, design) matches. Shared or unknown dependence is refused (`evidence_overlap_unknown`).

## Derived quantities

A quantity is evaluated in each arm at that arm's component means; the contrast is treated minus control. It is a function of means, not the mean of a function. Each input is one of:

- `Measured(component)`: the arm mean of an outcome;
- `Fixed(value)`: known exactly;
- `Assumed(lower, upper, assumption_id)`: not measured, bounded by a declared assumption, shared across arms or not;
- `Missing(what)`: the answer is unresolved, never defaulted.

The answer carries:

- `sampling`: delta-method estimate, standard error and 95% interval through the joint covariance. Calibration is `unmeasured` and `license` is `none`.
- `assumption_range`: the extremes of the contrast over the declared assumptions. Exact when the expression reads each input once (vertex enumeration of a multilinear function); otherwise a guaranteed outer bound, labeled so. It is not a probability law.
- `unresolved`: why no resolved answer exists: `missing_input`, `joint_law_missing`, `too_few_independent_units`, `sampling_unavailable` (a kink or unstable covariance calculation prevents a reliable interval), `sampling_straddles_threshold`.

A statistical interval needs at least 30 independently assigned units in each arm. The number of independent units, not rows, limits precision: more rows inside the same clusters leave the answer `unresolved`.

## Interference and scope boundaries

Between-unit exposure must be declared: `no_between_unit_interference` (an assumption, not a finding), `bounded_spillover` (a per-outcome bound on the bias of the contrast, which enters as an assumption range), or `unspecified` (refused). Bounds on component contrasts do not determine nonlinear functions of arm means, so those combinations are refused. For nonlinear quantities use `CenteredFixedCuped` and center each cluster's pretreatment total using the pretreatment mean across all target rows; `centered_fixed_cuped_totals` implements this calculation. The Python cluster constructor accepts `centered_fixed_cuped`, `cluster_sizes`, and `pretreatment_totals` and performs the centering. Uncentered fixed CUPED remains available for linear contrasts only. Alternatively, declare `bounded_arm_spillover` with separate per-component treated and control arm bias bounds; these assumptions support nonlinear sensitivity analysis without converting the bounds into empirical intervals. A request for another population, contrast or horizon is refused with its next attempt. A deployment-policy quantity is refused unless a structural bridge assumption is declared and is then reported as an assumption of the answer.

## Reuse and withdrawal

Stage digests chain (premise, adjustment, estimate, derived quantity, decision). A change invalidates its own stage and every later one:

- a decision threshold or action change reuses the estimate and the derived quantity;
- a new transformation reuses the estimate;
- new observations or a new adjustment reuse the premise;
- a change of the scientific premise reuses nothing.

`EvidenceLedger` records which evidence is in force. Withdrawing an experiment unsupports every composed evidence id built from it, and evidence not recorded in the ledger is not support.

## Planning

`MoreUnits` candidates without an outstanding assumption range are valued with the existing Normal-Normal `FutureStudyProgram`; candidates with such a range are refused rather than evaluated at its midpoint. The belief is centered on the current contrast with the delta-method variance, and the per-unit variance is implied by the current number of independent units. This is a declared planning model, not a calibrated posterior. `ResolveAssumption` candidates re-run the decision with the declared narrower range; they are best cases, not expectations. The two kinds are reported side by side.

## Maintained example

`crates/antecedent/tests/decision_evidence_workflow.rs` and `python/tests/test_decision_evidence.py` run one scenario in both languages: free shipping tested on 80 clusters of 12 customers, three dependent outcomes, margin `orders * (basket value - unit cost)`. With unit cost measured the decision is robust; with unit cost only bounded to $4-$12 and a $2.50 bar it is assumption-dependent; with 24 clusters (the same rows per cluster) it is unresolved; narrowing the cost range alone does not resolve it, and the plan refuses to value more clusters at an arbitrary cost midpoint.

## Not done

No marketplace-specific module, no default substantive assumption, no learned cluster-level adjustment, no calibration claim for derived intervals, no sensitivity bound converted to a probability. Interference beyond a declared no-interference or bounded-spillover assumption is refused rather than modeled.

The calibration gate names two ignored future measurement designs for the executed cluster-margin interval, with and without centered fixed CUPED. Their candidate basis identifies the transformation, adjustment, design and number of assignment units; registration does not itself grant calibration or a support license.

## Matched pairs, switchbacks and proxy effects

`compile_matched_pair_design` accepts declared cluster identities, disjoint pairs, preassignment matching variables, fixed cluster sizes, assignment probabilities and assumption identities. The current supported randomization is one treated cluster per pair with probability one half and independent assignment across pairs. `evidence_from_matched_pairs` jointly estimates the arm means and the full covariance between outcomes **and arms**. It preserves the pair as the independent assignment unit: at least 30 pairs are required for the paired delta-method construction with a Student reference, which remains unlicensed and uncalibrated. Arbitrary blocks, other probabilities, and automatic matching are not supported.

`compile_joint_switchback` retains the sequence and period identities, marginal assignment probabilities, washout rule, no-carryover assumption and independent-sequence assumption. `evidence_from_joint_switchback` reuses the existing sequence-level switchback estimator; its full covariance is reconstructed from that estimator's sequence variance. It requires at least 30 independent, equally sized sequences and the existing switchback overlap and interval gates. Additional periods in the same sequences do not count as additional independent assignments. Only zero washout is supported; nonzero washout is refused rather than treating assignment-dependent exclusions as innocuous. The result uses the same sequence-count Student reference as the marginal estimator; its nonlinear interval does not inherit the marginal estimator's license or calibration.

Both designs enter the same `derive` and `decide` functions as complete-cluster evidence. Their complete design declarations travel with the evidence and contribute to scientific identity. A derived interval is withheld when cross-arm covariance cancellation is numerically unstable. `MoreUnits` planning is refused for these designs until a design-specific planning model is supplied. Python users can inspect `compile_experimental_design` and then call `JointEvidence.from_experimental_design` with the same declaration.

`decide_with_proxy` accepts an explicitly scoped causal-effect mapping of the form `target effect = multiplier × proxy effect + residual`, where both coefficients are bounded by declared assumptions. A predictive association alone is refused. The result distinguishes a range over mapping assumptions at the observed proxy effect from an outer envelope over the proxy's sampling interval; neither is presented as a probability distribution or a calibrated interval. Experimental and mapping evidence identities remain separate, and `ProxyDecision.unsupported_evidence` checks both against the evidence ledger. The mapping supports a decision on the target effect itself; it does not supply arm means for a nonlinear objective, deployment extrapolation, an extra spillover range, or uncertainty represented by an unsupported joint distribution.

## Lagged excursions and whole regimes

`compile_joint_excursion` prepares a sequentially randomized unit-period schedule with a recorded assignment law, positivity floor, outcome lag and explicit independence and follow-up assumptions. `evidence_from_joint_excursion` uses the existing `temporal_design::lagged_pairs`, `two_arm_scores` and cluster-robust ratio covariance for all outcomes together. Earlier and later treatments follow the declared law; carryover into the lagged outcome is part of the excursion target. It does not use the switchback route or assume no carryover. Internal missing lagged outcomes are refused; unpaired end-of-follow-up decisions remain visible through the retained design and paired count.

A `Target::Excursion` identifies both the lag and assignment law. Requests for a different lag, an ordinary experimental contrast, or a sustained deployment are refused. The number of independent units with paired decisions—not the number of unit-period rows—controls the existing 30-unit threshold for a derived interval. The temporal covariance and its nonlinear transformation remain unlicensed and uncalibrated. Python callers use `compile_experimental_design` with `kind="temporal_excursion"`, followed by `JointEvidence.from_experimental_design`; `evaluate` retains the excursion target automatically.

`compile_joint_static_regime` prepares two whole static treatment paths, the measured history, method and identification assumptions. `evidence_from_joint_static_regime` calls the existing temporal regime point estimator (`g_formula`, `inverse_probability_weighting`, or `doubly_robust`) for each outcome. It carries the complete paths, sequential exchangeability, positivity, consistency, no-interference, follow-up and model-correctness assumptions. The g-formula relies on its specified outcome models; inverse weighting relies on its specified treatment models; the doubly robust route declares the corresponding either-model condition. By default the separately fitted point values do **not** provide a compatible joint sampling law; derived values are retained, but decisions are unresolved with `joint_law_missing`. An explicit bootstrap option refits all outcomes on the same resampled subjects and retains every failed slot. It supplies a joint sampling covariance only when all refits succeed; a derived normal-reference delta interval is unlicensed and uncalibrated. This route does not treat a marginal interval as a paired covariance or convert posterior uncertainty into sampling uncertainty. History-adaptive policies and retained posterior draws require their own integration.

`TemporalEffectBridge` explicitly identifies a source excursion or static regime and a contrasting fixed schedule, static or adaptive regime, excursion, or model-conditional response, including both active and reference schedules or rules. It can map effects only through declared bounded multiplier and residual assumptions, with mapping-evidence lineage. It does not establish equality of the two estimands, identify a sustained program from a pulse, or infer causal identification from a predictive association. The bridge's range is assumption-dependent; it is not a probability law or a calibrated interval. A bare deployment assumption ID is not a temporal bridge. Python callers can use `decide_with_temporal_bridge` with the same explicit descriptors and mapping.

Ignored lagged-excursion and static-regime bootstrap calibration designs are registered in the calibration gate under a distinct cluster-ratio construction and independent-unit scope. Registration does not execute calibration, attest coverage, or confer a support license; the pre-existing frozen calibration-family review must be refreshed through its own process before that gate can accept a changed roster.

## Retained compiler sources

The compiler-backed producers retain the original assignment and outcome source, the exact intervention and population identities, and the declared identification assumptions. Matched-pair, switchback, excursion and static-regime sources use `PreparedNativeJointDesign`; the static-regime bootstrap retains its aligned sampling refits and never presents them as posterior draws. The caller-context `compile_and_evidence_from_*` functions keep the native result and its admission alive with the numerical decision component. Artifact replay is historical; independently matching the caller-current source and checking the evidence ledger are separate requirements.

The Python producers retain a replayable compiler artifact and check its reproduced numerical evidence before evaluation. Cluster inputs require explicit cluster identities, fixed cluster sizes, assignment-law identity and consistency assumption. Fixed CUPED additionally requires original pretreatment totals and a preassignment assumption; centered adjustment is performed by the native compiler owner. Matched-pair, switchback and excursion declarations also require an explicit consistency assumption. Compiler resource and cancellation refusals remain distinct from malformed scientific input. These source bindings do not themselves license a decision, establish interval calibration, or authorize a source that the current evidence ledger has withdrawn.

`JointEvidence.from_experimental_design` accepts `bootstrap_replicates` (at least 99) and an explicit `bootstrap_seed` on a static-regime design. The Rust equivalent is `compile_and_evidence_from_joint_static_regime_bootstrap`. All outcomes use the same resampled subjects; failed refits and their reasons remain in the compiled artifact. If any refit fails, the result remains point-only. The default without those options is point-only. Neither the normal-reference interval formed from the bootstrap covariance nor a nonlinear transformation inherits an existing calibration record.

For a Bayesian joint panel response, `JointEvidence.excursion_quantity` supplies the exact excursion coordinate for an explicit experiment-to-response bridge. `PanelResult.condition_excursion` consumes the retained excursion through the existing conditional panel update. The caller declares the Gaussian sampling model and its known variance as a separate assumption; the excursion's estimated covariance is not silently substituted for that variance. The update checks the source identity and current ledger, exact observed excursion mean and independent-unit count, and the full source-to-target translation. Its result remains model-conditional and uncalibrated; the bridge does not identify a sustained program from a pulse by itself.

For prospective temporal studies, `more_independent_units` ranks additional complete independent
subjects, matched pairs, or assignment sequences under the explicitly named assumption that the
future study has the same design and per-unit variance. It uses the current joint sampling
variance in a normal planning model; it is not a posterior or a calibrated guarantee. Additional
periods within existing units are not additional independent units. A point-only result or an
unresolved assumption range cannot be turned into this planning distribution.

`JointEvidence.from_adaptive_regime` compares two prespecified, deterministic policies over
finite discrete state and action histories. The action table is keyed by the complete recorded
state and previous-action history, and the g-formula integrates later states under each policy.
Each reachable history must have observed support. The native compiler retains the policy tables,
independent episodes and identifying assumptions. A whole-subject jackknife supplies joint
covariance only if every deletion remains supported; otherwise the result remains point-only.
The interval is unlicensed and uncalibrated, and adaptive policy evidence is distinct from a
static schedule or randomized excursion.

The Python recipe names `population`, `state_variable_id`, `action_variable_id`, and `outcomes`
(as `[component, variable, units]` triples). Each `episodes` entry contains an `id`, equally long
`states` and `actions` arrays, and terminal `outcomes` in that coordinate order. The two ordered
`policies` each have an `id` and `rules`; a rule's `states` includes the current pre-action state,
`past_actions` contains the previous actions, and `action` is the prescribed discrete action.
The recipe also requires named `policy_prespecification`, `positivity`, `exchangeability`, `consistency`,
`followup`, and `isolation`
assumptions. Episode counts and policy-history work are bounded, and the exact source is retained
in the compiled artifact.
