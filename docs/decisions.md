# Evaluate actions with a decision contract

Compare actions under a utility you declare, using the quantities an analysis
produced. A decision contract names the actions, the utility of each in terms of
scientific quantities, any hard constraints and the criterion that ranks them.
Antecedent evaluates the contract on whatever the source can support: a mean
answers the expectation of an affine utility, aligned joint draws answer
probabilities, quantiles, regret and nonlinear utilities. A decision keeps the
standing of its inputs. A decision on an external mean stays a decision on an
attested point, and a sensitivity range stays an assumption range.

For the worked story from a causal result to a stop, continue or run-longer
choice, see [Decide under uncertainty](tutorials/decide-under-uncertainty.md).
For decisions that combine several fitted routes, see
[Decide from mixed evidence](guides/mixed-decisions.md).

## Decide from an analysis result

`Contract.evaluate` takes a supported analysis result. Declare outcome units once
in its scientific coordinates or in `analyze(outcome_units=..., dose_units=...)`;
the contract supplies an unambiguous matching outcome, unit and population when no
binding was retained. Undeclared dose units mean `native_numeric_scale`: the
original numeric intervention scale, without a physical unit claim or conversion.

```python
import numpy as np
from antecedent import AverageEffect, ResponseCurve, analyze, decision
from antecedent.joint_distribution import ScientificQuantity

rng = np.random.default_rng(23)
a = rng.normal(size=400)
data = {"a": a, "y": 2.0 * a + rng.normal(scale=0.2, size=400)}

result = analyze(data, query=ResponseCurve("a", "y", grid=[-0.5, 0.0, 0.5]), graph=[("a", "y")])
print(result.value.claim())      # what the result is, in words

low = ScientificQuantity.from_response_dose(result, -0.5, outcome_units="mmHg")
high = ScientificQuantity.from_response_dose(result, 0.5, outcome_units="mmHg")
contract = decision.Contract(
    actions=(
        decision.Action("low", inputs=(low,), utility=decision.x(0)),
        decision.Action("high", inputs=(high,), utility=decision.x(0)),
    ),
    utility_units="mmHg",
    criterion=decision.Criterion.expected_utility(),
    target_population="target",
)

decided = contract.evaluate(result)
decided.value.selected           # ("high",): E[Y | do(a=0.5)] is about +1, do(a=-0.5) about -1
```

What this decision is and is not:

- A response curve is a **mean grid**. A mean answers only the expectation of an
  affine utility. The decision therefore has no regret, no `evpi` (it is `None`)
  and no replayable export: `decided.export()` raises `CausalUnsupportedError`
  with `reason_code == "route_not_supported"`.
- A quantile or probability criterion over a mean grid is refused by name
  (`decision_contract_unsatisfied`, detail `native_claims.source_not_supplied`,
  expected `joint_draws,marginal_draws`, supplied `mean`). Supply aligned joint
  draws (a `JointDistributionArtifact`) to ask a distributional question.
- A checked static `AverageEffect` result supplies one contrast coordinate,
  created with `ScientificQuantity.from_effect(result, outcome_units=...)`.
  Its functional is `mean_difference`, and its regime names active minus
  control. It cannot supply two absolute outcome means. Use that coordinate
  in an affine utility and call `contract.evaluate(result)`; inspect
  `decided.original_execution` for the immutable original native diagnostics
  and effect uncertainty. That uncertainty is not an interval for utility.

`ScientificQuantity.from_response(result, outcome_units=...)` returns every grid
coordinate at once; `ScientificQuantity.outcome / treatment / utility` build a
coordinate by hand with only conventional fields defaulted. Scientific meaning
(units, population, regime) is never defaulted.

## Decide from an external claim

When another system supplies the numbers, bind its grid to an identified program
first (see [external scientific objects](guides/build/scientific-interfaces.md)),
then decide on the bound claim:

```python
wait, _, treat = claim.quantities
utility = decision.x(0) * 2.0 - 1.0
contract = decision.Contract(
    actions=(
        decision.Action("wait", inputs=(wait,), utility=utility),
        decision.Action("treat", inputs=(treat,), utility=utility),
    ),
    utility_units="utility",
    criterion=decision.Criterion.expected_utility(),
    target_population="target",
)
result = contract.evaluate(claim)
result.value.selected             # ("treat",): 2 * 5 - 1 = 9 against 2 * 1 - 1 = 1
result.value.stages_behind()      # includes causal_contract, external_provider, claim
```

The decision is a point on an externally attested claim. It carries the claim's
lineage plus the decision contract, so the foreign provider stays visible in
`result.lineage`. It is not exportable on its own; to carry it forward use
`composition_bundle.mean_decision` (see [composition](composition.md)). If the
decision depends on an assumption you cannot test, carry it through
[sensitivity and robustness](sensitivity-and-robustness.md) before acting. If it
must travel to another population, see
[transport and counterfactuals](transport-counterfactuals.md). If it is not yet
clear-cut, [rank the next study](design-ranking.md).

## The contract in detail

### What a contract declares

- **Actions** by stable semantic ID and `ActionKind` (intervention, policy,
  regime, study, external). A label never defines identity.
- **Inputs** per action: ordered `ScientificQuantity` coordinates under the
  action's regime, all inside the decision's target population and horizon.
- **Utility** per action as a closed `UtilityExpr` (constants, inputs, sum,
  difference, product, negation, min, max), in the contract's `utility_units`.
  A closed expression is replayable from the contract alone; a callback is not.
- **Hard constraints** `P(expr <= bound) >= min_probability`. They exclude
  actions and never become penalties.
- A **criterion** (posterior expected utility, expected loss, threshold
  probability, quantile, minimax over an identified set, maximin over
  structures, regret, expected regret) and a **structural policy**
  (`RequireInvariantBestAction`, `Maximin`, `BayesOverStructures`,
  `ReportOnly`). CVaR is not offered.

`validate` refuses fewer than two actions, blank or duplicate IDs, inputs outside
the target scope, unknown inputs, non-finite parameters and out-of-range
probabilities before anything is evaluated.

### What each functional needs

`DecisionFunctional::requirement(expr)` states which source representation can
answer it: a mean answers the expectation of an affine utility; a variance needs
covariance or draws; a quantile or threshold probability needs a quantile
function, CDF or draws; a nonlinear function of several inputs needs aligned
joint draws, because pairing marginal draws by index is invalid. When the answer
comes from draws the requirement flags that a Monte Carlo error receipt is
needed. `DecisionContract::source_requirement` combines the actions: one
nonlinear action makes the comparison need draws, and a regret criterion needs
joint draws across actions. A typed functional does not by itself license a
native support row.

### Identity

`DecisionContract::identity` is a canonical BLAKE3 digest. Reordering actions or
constraints leaves it unchanged; any semantic edit to an action, quantity,
utility, constraint, criterion, scope, units or structural policy changes it.

### Evaluation on aligned draws

`decision_eval::evaluate_contract(contract, source)` evaluates a contract on an
aligned joint `DistributionArtifact`. Each action's utility is computed per
draw from the artifact's rows, so a nonlinear utility over several quantities
sees genuine joint realizations: with `P*Q` and enumerated rows where
`E[P]E[Q] = 4.5` but `E[PQ] = 2`, only joint rows choose correctly, and
re-pairing the same marginals flips the choice. It refuses non-joint draws
(`joint_law_required`), a coordinate it cannot find or that is masked
(`quantity_semantics_mismatch`), and a source whose meaning cannot answer an
outcome-law input (`distribution_meaning_mismatch`), such as a posterior over a
mean offered where an outcome law is read.

Hard constraints exclude actions before the criterion ranks the rest; an
excluded action keeps its untouched utility and carries the constraint, the
probability it reached and the probability required. Implemented criteria,
each scored by its own route: posterior expected utility, expected loss,
threshold probability, quantile, regret (maximum over draws) and expected
regret. Minimax over an identified set and maximin over structures need
structure inputs that one draw source does not carry, so they refuse with
`route_not_supported`.

`DecisionResult` reports per-action admissibility and exclusions, expected
utility, criterion value, standard error, expected and maximum regret, EVPI over
the admissible actions, effective draws, the source lineage and the
assumptions. `UniquelyOptimal` names the unique best value under the supplied
finite law, `Indistinguishable` reports equal values, and `NoAdmissibleAction`
reports that every action failed a constraint. For a sampled law these are
point rankings only. Monte Carlo standard errors and statistical ranking
guarantees are withheld: a distribution's calibration label does not certify
independent draws, effective MCMC sample size, or decision calibration. An exact
finite law has zero numerical sampling error. Loss decisions compute regret
relative to the smallest loss and EVPI as the reduction in expected loss.
Zero-weight atoms do not enter worst-case regret or quantiles.

### Structural uncertainty

`decision_structural::evaluate_structural(contract, atoms)` evaluates the
contract in each structure (a graph, completion or supplied scenario) and
combines the structures under the contract's `StructuralPolicy`:

- `RequireInvariantBestAction` names an action only when it is uniquely best in
  every structure, otherwise `NoInvariantBest` lists each structure's leader.
- `Maximin`, and the `MaximinOverStructures` criterion, choose the best worst
  case. `MinimaxOverIdentifiedSet` reads the contract's utilities as losses and
  chooses the smallest worst-case expected loss; the two can differ.
- `BayesOverStructures` weights by supplied structure probabilities and refuses
  without them (`ProbabilitiesRequired`): completion counts are not
  probabilities.
- `ReportOnly` returns each structure's answer without choosing.

An atom is `Evaluated`, `Unidentified` or `Unevaluated` (with the registered
reason code when its evidence could not answer the contract). Unidentified and
unevaluated mass is reported beside the evaluated mass and is never
renormalized away: a Bayes choice states the evaluated mass it rests on, and a
worst case or invariance claim that needs every structure returns
`InsufficientScience` when one is unresolved. A hard constraint that fails in
any structure excludes the action, and when every action is excluded the verdict
is `NoAdmissibleAction`. Per action the result keeps the criterion value in
each structure, its range, the probability-weighted value, the mass where it
leads and the structures that excluded it.

### Artifacts and replay

`DecisionContractArtifact` stores the contract and its canonical identity in the
checksummed container. Loading requires the identity the consumer retained
independently; an edit resealed with its own identity, truncation and oversized
input all refuse, while reordering unordered actions keeps the identity.

`DecisionResultArtifact` binds a result to its contract identity and to a BLAKE3
digest of the source draws (coordinates, rows and weights), and keeps the
provider, snapshot, RNG and causal-contract identities of the source. It loads
only under the consumer's retained contract identity and source digest.
`replay(contract, source)` recomputes the decision and requires the stored
result to match exactly, so an edited number or verdict does not replay even if
it was resealed under the same identities. A fresh process rebuilds the contract
and source identity from constants and replays the decision without the producer.

A result over structures has its own artifact (see below).

#### Python surface

`antecedent.decision` follows the Python flow rather than mirroring the Rust
types. A `Contract` is built from `Action`s, a closed utility `Expr` written with
ordinary operators (`decision.x(0) * decision.x(1)`, `decision.maximum(...)`),
`Constraint`s and a `Criterion`. `contract.evaluate(joint_distribution)` returns
a `Decision` with `verdict`, `actions`, `evpi`, `assumptions`, `explain()` and
`export()`; `decision.replay(bytes, contract=..., source=...)` recomputes a stored
result exactly. Python only builds declarations: identity, validation,
evaluation, artifacts and refusals are Rust's, and each refusal raises
`DecisionRefusal`, a `CausalUnsupportedError` with the registered `reason_code`,
`detail` and `offending` action input. `python/tests/test_decision.py` asserts
the same enumerated fixture as the Rust integration tests.

### Refusals

Decision refusals share one shape, `antecedent_contract::StructuredRefusal` (an alias
of `ExternalRefusal`): registered `code`, `stage`, namespaced `detail`,
`offending` action input (`action[input]`), `expected` and `supplied` semantics,
`capability` and `remedy`. `DecisionContractError`, `DecisionEvalError`,
`StructuralError` and `DesignError` each have a `to_refusal()` in
`decision_refusal.rs`; the existing reason codes and `reason_code()` are
unchanged.

| Family | Namespace | Stage |
| --- | --- | --- |
| Contract declaration and source requirement | `decision_contract` | `declare` |
| Evaluation (quantity, joint law, meaning, structure, utility) | `decision_evaluation` | `evaluate` |
| Structural evaluation | `decision_structural` | `structural` |
| Design evaluation | `design_refusal` | `design` |

Every detail is a plain string literal of the form `namespace.snake_case`, never
built with `format!`, so it can be matched against the declared refusals. A missing source lists the needed representations, comma-joined, in
`expected`. 
### Provenance chain and mean-only sources

`DecisionResultArtifact::provenance_chain()` names what stands behind a decision,
parents before children: `contract:<causal_contract_id>` (causal contract),
`provider:<provider_id>#<snapshot_id>` (external provider),
`distribution:<source_digest>` (distribution artifact), `decision:<contract
identity>` (decision contract) and the reported `decision_result` claim, derived
from the distribution and the decision. A different source digest is a different
distribution link. `result_to_json` carries the same links as a `lineage` array,
derived at serialization from the stored identities and re-derived on load, so
replay still compares like with like. In Python, `Decision.lineage` and
`Decision.stages_behind(link="decision_result")` expose it.

`evaluate_contract_on_means(contract, MeanSource)` evaluates a contract on one
mean per coordinate, such as an external response grid. A mean answers only the
expectation of an affine utility: the criterion must be expected utility or
expected loss, every utility must be affine, and hard constraints refuse
(`decision_contract_unsatisfied`, `decision.mean_source_insufficient`). Inputs
whose functional is `outcome` are not found in a mean grid, and a mean source
that does declare an `outcome` coordinate refuses with a meaning mismatch. The
result claims no sampling error, no regret and no EVPI, and calls two actions
indistinguishable only on exactly equal values. In Python,
`Contract.evaluate(claim)` accepts a `BoundExternalClaim`; its `Decision` lineage
is the claim's own lineage plus the decision contract and `decision_result`.
Such a decision is not exportable or replayable (`route_not_supported`,
`decision.mean_source_not_replayable`). Ranking a study with `design.rank_designs` is covered by
[design ranking](design-ranking.md).

### Structural result artifact

`StructuralResultArtifact` (`antecedent_design::decision_structural_artifact`) stores a `StructuralDecisionResult` in the checksummed container (`decision_structural_result_v1`, bounded at 8 MiB). It retains the contract identity, each atom's id, probability and status (evaluated, unidentified, or unevaluated with its reason), the BLAKE3 digest of each evaluated atom's draws, the declared structural policy, the per-action per-structure values, ranges and masses, the unidentified, unevaluated and evaluated masses, and the verdict. `from_bytes` loads only under the consumer's own contract identity and atom id/digest list, and `replay` recomputes `evaluate_structural` and requires a byte-identical body, so a resealed verdict or number is refused.

Structural Bayes choices require all positive probability mass to be evaluated.
Partial weighted values remain visible without renormalization, and an omitted
remainder is counted as unevaluated mass. Unknown payoffs can reverse a partial
ranking, so it produces `InsufficientScience`. A zero-probability atom cannot
exclude a Bayes action. Bayesian averaging of component quantiles or maximum
regrets is refused: neither is the corresponding functional of a mixture law.

### Cohort-mean treatment contrast of a fitted effect model

A fitted effect model predicts conditional treatment contrasts `tau(x)` (a CATE map). Those
points are differences: they do not give two absolute potential-outcome means and they do not
give a law of the contrast across units. The one decision a cohort of them supports is the
expectation of an affine utility of the cohort's weighted mean contrast `sum_i w_i tau(x_i)`.

`antecedent::analysis::cohort_decision` binds that quantity natively:

1. `FittedEffectModel::load` verifies an exported analysis result (contract section, executed
   claim, no attested evidence) and retains its portable fitted payload, the original
   contrast, the producer's fitted-row support box and the original diagnostics and
   assumptions. A result with no support contract is refused for row-level support; it is
   never assumed supported. `FittedEffectModel::from_compact_export` retains a caller-built
   linear compact model that declares a conditional contrast; such a model is a declaration,
   so its sources are attested and never native.
2. `CohortBinding` declares the cohort: strict named feature rows (none missing, none extra),
   unique row identities, a target population, finite normalized weights and a snapshot
   digest that names every field. `FittedEffectModel::decision_source(cohort, outcome_units,
   options, ctx)` requires explicit outcome units (never inferred, never converted), checks
   every row against the support contract and issues one `CohortDecisionSource`.
3. The source's coordinate is the declared outcome under the model's contrast regime,
   functional `mean_difference`, population of the cohort and the cohort snapshot digest as
   conditioning, so a decision contract names exactly the cohort it decides for.
   `cohort_decision_input` and `evaluate_cohort_decision` hand it to the support-aware
   evaluator; `cohort_forward_claim` offers it as a mean forward source of an inverse query.

The value is point-only. No utility standard error, regret, value of perfect information or
interval is inherited. A probability, quantile, tail expectation, nonlinear utility, hard
constraint or joint law is refused (`decision_contract_unsatisfied` /
`joint_law_required`, `cohort_decision.law_not_supplied` / `cohort_decision.joint_law_required`).
This is closed by logic, not pending a measurement: one expectation determines no law (Jensen's
inequality separates `E[g(tau)]` from `g(E[tau])`), and the individual-level law of
`Y(1) - Y(0)` is not identified from marginals. The refusal names the carrier that does hold a
law: per-draw cohort utilities under a declared shared law (`DecisionDraws`, producer
`cohort_decision`), with the decision reliability routes.

Export and load. `CohortDecisionSource::to_bytes` writes `cohort_decision_source_v1`, embedding
the model payload, the cohort rows and the result. `consume_cohort_decision_source` recomputes
every number under an independently retained `CohortExpectation` (identity, and optionally
parent claim, population, units, contrast regime and cohort digest), so a changed row, weight,
coefficient, unit, population or parent is refused with its own detail even when the container
was resealed with recomputed checksums.

A cohort from another population is accepted only under a `CovariateShiftDeclaration`: the
conditional contrast is declared invariant given exactly the model's features, with every row
inside the support contract. The result is labelled `Extrapolative`, so the default support
policy evaluates no action on it and a weaker policy must be chosen explicitly. Without the
declaration the refusal names the unmet premise (`transport_missing_evidence`,
`cohort_decision.shift_declaration_required`). The support box is a necessary condition for
overlap, not a density-ratio overlap diagnostic.

Adjacent uncertainty route. For a linear compact model with a retained coefficient covariance
`V`, the variance of the weighted cohort-mean contrast is `phibar' V phibar` with
`phibar = sum_i w_i phi(x_i)`. It is reported as `uncalibrated_standard_error`
(`model_based_sqrt_phibar_v_phibar`, calibration unmeasured) on the source only. It conditions
on the declared cohort rows and weights and on the fitted structure and excludes cohort
sampling variation, the support-selection step, misspecification and extrapolation. It is
never an interval by itself and is not passed to a decision input.

The interval is a separate, measured output of a natively fitted linear effect. Fitting the
conditional effect by ordinary least squares of the outcome on the declared basis and its
treatment interaction, with an HC3 sandwich covariance and the retained row count `n` and
design width `p`, gives `phibar' b_hat +/- t_(n-p) sqrt(phibar' V phibar)` for the declared
cohort's mean of the true conditional effect
(`antecedent_estimate::cohort_linear_effect`). It is reported only as a measured scalar bound
to a currently attesting coverage record: randomized or observed-confounded binary treatment,
two to four modifiers, heteroskedastic noise, cohorts of 50 to 200 rows of the fitted
population, levels 0.90 and 0.95. A caller-built compact export is point-only and refuses
`cell_not_licensed` (`cohort_decision.interval_requires_native_fit`); a shifted cohort and a
portable model refuse `cohort_decision.interval_not_licensed`. The native-fit marker is never
trusted: the measured envelope retains the fit rows (`n`, `p`, and a producer digest binding the
coefficient vector and covariance) and the consumer refits and requires the export to reproduce
exactly, so a hand-built export cannot claim a native fit. In Python,
`fit_native_cohort_effect(...)` retains the rows,
`decision_source_from_native_fit(...)` binds a cohort, and `source.interval(level=...)` returns
the measured result or a typed refusal. A misspecified effect basis and
a cohort of high-leverage rows are recorded boundaries, not claims.

## Decide with transported, composed and cohort sources

Transported, composed and cohort sources reach a decision or an inverse query through `crates/antecedent/src/analysis/` in Rust and
through `antecedent.transported_response`, `antecedent.inverse_sources`,
`antecedent.transport.revalidation`, `antecedent.transport.sensitivity`,
`antecedent.discretized_composition` and `antecedent.cross_world_execution` in
Python. Every value is a point
(or a labelled spread) under declared premises; none is an interval and none is
upgraded by being passed on.

| Step | Entry | Keeps | Recomputes | Refuses |
|---|---|---|---|---|
| analyze, then decide with a transported source | `BoundDecisionContract::evaluate_transported(&TransportedResponse, TransportedDemand, &SupportPolicy)` | the receipt (route, source and target populations, diagram identities, assumptions), the extrapolative support label and the declared exclusions as part of the bound decision identity | the affine expected utility of the transported mean, or the probability criterion of a transported law, and the decision identity | a law request of a point (`transported_decision.law_not_supplied`), an interval (`cell_not_licensed`), the default policy (`transported_decision.transported_not_accepted`), a support rule stronger than the label (`transported_decision.support_rule_unmet`), an uncertainty requirement a point cannot meet (`transported_decision.uncertainty_requirement`) |
| transported source for an inverse query | `analysis::inverse_sources::{composed_source, cohort_source, transported_source, evaluate, seal}` and `PreparedMechanismComposition::inverse_forward_source` | the origin (kind, representation, support, receipt, assumed mechanisms), bound into the sealed query and re-derived on consume | the per-action forward value (replayed from the composition artifact) and the claim digest | a quantile or probability of a point (`inverse_forward_source.point_only_law_requested`), an interval (`inverse_forward_source.interval_not_licensed`), a transported source under a native-only policy (`inverse_forward_source.transported_not_accepted`) |
| refresh, then revalidate transported claims | `transport_revalidation::{TransportSceneBuilder, revalidate}`, `TransportClaimState::observe` | the declared selection diagram, structure and quantity; claims that do not count a changed source | every claim counting a changed source, diagram or target (`recompute`); a state change enters through `invalidation_effects_for_claims` | a calibration record established at another target (`withdrawn`, `calibration_from_another_target`) and an unsupported target (every claim withdrawn); a refused observation leaves the state unchanged |
| assumption range on a transported value | `transport_sensitivity::{transport_range, transport_sensitivity_artifact}` | the value, its quantity and receipt, the declared bound | the band at each grid point and the sealed sensitivity artifact on consume | a transported law (`transport_sensitivity.law_response_not_scalar`), a structural premise (`transport_sensitivity.no_scalar_family`), a sampling interval composed with the range (`transport_sensitivity.interval_not_composed`) |
| refit, then discretize, contrast, export and consume | `PreparedMechanismComposition::discretize`, `PreparedDiscretizedComposition::{mean_contrast, refresh, export}`, `consume_discretized_composition` | the declared level counts and method | the cell edges and values, the tables, the computed mean-error bound and the identity; a consumer re-derives them from the carried premises | a refit whose variables no longer match the plan, an interval or posterior (`discretized_scm.point_only`), a latent-confounded, non-Gaussian or oversized composition |
| execute a cross-world row, then account for its uncertainty | `CrossWorldExecution::whole_path_uncertainty(UnitSampling)` | the executed request and result the account is bound to | the enumerated exogenous-posterior spread and the declared sampling component, re-derived by a consumer | undeclared unit sampling (`cross_world_uncertainty.unit_sampling_undeclared`), a sampled or transported path, an account of other units (`cross_world_uncertainty.account_mismatch`) |

A transported point export is `transported_decision_artifact` (a consumer re-evaluates
the decision; a law is not exportable). A transported cohort effect, a fitted mechanism
carry, a transported policy or adaptive-trial value, a discovered-class envelope and a
provider handoff enter these steps as a declared `TransportedResponse` with the
producing result's digest identity.

## Routes at a glance

Each row gives the question a route answers, its scope bounds, its claim label, the Python module that owns it and its refusal namespace.

- **Point-only** means the value is a point value (or an enumerated finite-law
  quantity). No interval, coverage or probability claim comes with it.
- **No claim** means the route states a request or a classification and
  computes no estimate.
- A standard error, p-value, interval or Monte Carlo error reported by these routes is a **diagnostic with calibration unmeasured**. Read it as a diagnostic; its nominal level is not a coverage claim.

A refusal carries a registered reason code plus a namespaced detail
(`<namespace>.<snake_case>`); the namespace named per route is the stable
thing to switch on.

### Decisions

| Route | Question | Scope bounds | Claim | Python entry | Refusal namespace |
| --- | --- | --- | --- | --- | --- |
| Decision contract | Which action has the best expected utility under a durable, replayable declaration? | At least two actions with stable semantic ids; closed utility expressions; inputs inside the target population and horizon; a hard constraint excludes an action and never becomes a penalty. | point-only | `antecedent.decision` (`Contract`, `Action`, `Criterion`) | `decision_contract` |
| Typed functionals | What is the expectation, probability or quantile of a utility on an aligned finite law? | A mean answers only the expectation of an affine utility; a probability or quantile needs a quantile function, CDF or draws; a nonlinear function of several inputs needs aligned joint draws, and independent marginals refuse. | point-only | `antecedent.decision` | `decision_evaluation` |
| Admissibility and support | Can declared exclusions, per-input support rules and an uncertainty requirement remove actions? | Rules remove actions and never penalize; support statuses are ordered by severity; an uncertainty requirement is met only by a claim of exactly its own kind. | point-only | `antecedent.decision` | `decision_admissibility` |
| Structural policy | What does a declared policy choose when the graph structure is uncertain? | Maximin, report-only and Bayes-over-structures; Bayes refuses without genuine atom probabilities, and unevaluated mass is never normalized away. | point-only | `antecedent.decision_robust` | `decision_structural` |
| Claim adapters and identified sets | Can scenario, weighted-scenario and CPDAG-completion reports, and identified-set intervals, enter a decision? | Supplied sets only; an identified set has no probability law so Bayes over it refuses; completion counts are never probabilities. | point-only | `antecedent.scenario_decision` | `decision_adapters`, `decision_claims` |
| Robust verdicts and result artifact | Is the choice robust across structures, support-robust, graph-dependent or insufficiently claimed, and can the result be replayed? | Verdicts over supplied structures; an external-callback value keeps its receipt through export and replay and is never labelled native. | point-only | `antecedent.decision_robust` | `decision_robustness` |

Two limits are worth stating because they are easy to misread. A typed
functional does not by itself license a native support row, and a decision over
an external mean claim is labelled point-only attested, not native.

### Signals, EVSI and design ranking

| Route | Question | Scope bounds | Claim | Python entry | Refusal namespace |
| --- | --- | --- | --- | --- | --- |
| Study candidate | What evidence would a proposed study collect, with sample, timing, unit and cost identities? | A candidate observing separate regimes cannot claim a joint factor; missing unit or cost semantics refuse. | no claim | `antecedent.repair` (`StudyCandidate`, `ExpectedEvidence`) | `study_candidate` |
| External signal | Is a signal's observation law and posterior update coherent, with external trust kept external? | A predictive law without a coherent update, or a wrong candidate or sample id, refuses; trust is never native. | point-only | `antecedent.design` | `signal_provider` |
| Prior and signal planning | Can a prior bank and a candidate signal be combined into a checked planning input? | Pooling is declared; an observation shared between a prior source and the candidate, or between two sources, refuses; a changed population needs a declared transport policy. | point-only | `antecedent.priors` | `prior_signal` |
| EVSI | What are the expected value of sample information, of perfect information and the net value under a declared cost map? | Exact integration under a declared utility-unit cost map; currency without a cost-to-utility map, and a changed terminal action set, refuse a net-value comparison. | point-only | `antecedent.design` (`rank_designs(..., decision=)`, basis `net_value`) | `evsi` |
| Design ranking artifact | Can a ranking of candidate studies be exported and replayed by a fresh consumer? | Native values are recomputed on replay; external values are retained as attested; an altered signal fingerprint, overlapping source data or an incompatible cost unit refuses. | point-only | `antecedent.design` (`consume`) | `design_ranking` |

The Monte Carlo error of a sampled EVSI and any rank uncertainty are
diagnostics with calibration unmeasured, not licensed claims.

### Repair

| Route | Question | Scope bounds | Claim | Python entry | Refusal namespace |
| --- | --- | --- | --- | --- | --- |
| Evidence obligations | What evidence does a failed identification contract owe, by exact population, regime, variables and proof step? | A request, never a verdict; separate marginal studies cannot satisfy one required joint-regime factor. | no claim | `antecedent.repair` (`EvidenceObligation`) | `evidence_obligations` |
| Identification repair | Which candidate study repairs a contract (transport, back-door, z-transport, front-door, instrument, hedge or sensitivity), as classified by each family's theorem checker? | Subset search under a `SearchBudget`; the subset size is at most 4; exhaustion means not found within the budget, never impossible. | no claim | `antecedent.repair` | `identification_repair` |
| Repair search receipt | Can a stored search receipt be replayed by a fresh consumer? | The consumer trusts nothing stored: it rebuilds the family and re-runs the checker on each stored delta; truncated work is stored as unevaluated. | no claim | `antecedent.repair` | `repair_artifact` |
| Proposal receipt | Can the evidence a failed contract owes be linked, per proposed study, to the ranking that values it? | Identities are bound and the family checker is re-run on arrival; a hypothetical derivation is never treated as available evidence. | point-only | `antecedent.proposals` | `proposal_receipt` |

### Sensitivity composition and inverse queries

| Route | Question | Scope bounds | Claim | Python entry | Refusal namespace |
| --- | --- | --- | --- | --- | --- |
| Sensitivity-decision composition | Over a declared assumption grid, where does the leading action switch? | Grid points only; a tie coordinate is reported exactly and an interpolated crossing is labelled an interpolation. A declared assumption range is not a sampling interval and never a probability; composing the two is refused. | point-only | `antecedent.sensitivity_decision` | `sensitivity_decision_composition` |
| Functional inverse query | What is the least action on a declared grid whose functional meets a target? | A finite declared grid; point, interval-region, identified-set, all-scenario and posterior-probability feasibility are separate fields, and a found grid point is never a global feasibility claim. A mean cannot satisfy a probability or quantile target. | point-only | `antecedent.inverse_query` | `functional_inverse_query` |

### Dose grid and chain recovery

| Route | Question | Scope bounds | Claim | Python entry | Refusal namespace |
| --- | --- | --- | --- | --- | --- |
| Dose-grid functional | What are the level, derivative or contrast of a randomized-dose response curve at named doses, and where is the data too thin to say? | A randomized continuous dose, a Gaussian-kernel local quadratic at a caller-declared fixed bandwidth, one named functional per call, per-dose support reported. | point-only | `antecedent.dose_grid` (`dose_functional`, `dose_support_table`) | `dose_grid` |
| Chain observation recovery | Is the law of two binary variables recoverable from the observed pattern law when one response indicator causes the other? | An m-graph with exactly one response edge; a self-censoring edge is refused with a verified two-model witness, and a graph outside the class is `route_not_supported`, never nonrecoverable. | point-only | `antecedent.recovery_chain` (`recover_chain`) | `recovery_chain` |

For the dose grid the pointwise normal intervals and standard errors are
diagnostics with calibration unmeasured, the smoothing bias is not included, and
the simultaneous band is closed (`dose_grid.simultaneous_band_closed`) until
its own calibration exists. A sampled provider composed on chain recovery, and
its intervals, stay closed.

### Estimators that feed decisions, and compact export

These estimators report point values. Every standard error, covariance
entry, Wald statistic, p-value (raw and Holm) and bootstrap standard error they
report is a diagnostic with calibration unmeasured.

| Route | Question | Scope bounds | Claim | Python entry | Refusal namespace |
| --- | --- | --- | --- | --- | --- |
| Vector treatment coefficients | What are the joint coefficients of several treatments estimated together, with their full covariance? | One shared adjustment block and one row snapshot; a different adjustment set, snapshot or row count, a constant or collinear treatment, or a rank-deficient block refuses. | point-only | `antecedent.vector_treatment` | `vector_treatment` |
| Categorical treatment contrasts | What are level-versus-reference and pairwise contrasts, with an omnibus test and a declared monotonicity test? | A declared level with too few rows refuses by name; nothing is dropped or merged. The monotonicity p-value is a conservative bound. | point-only | `antecedent.categorical_treatment` | `categorical_treatment` |
| Nonlinear mediation | What are the natural direct, indirect and total effects of a binary treatment through a continuous mediator with a nonlinear outcome? | Requires sequential ignorability and declared cross-world independence; a Gauss-Hermite quadrature check refuses on disagreement; the interventional estimand is refused. | point-only | `antecedent.nonlinear_mediation` | `nonlinear_mediation` |
| Latent-class effects | What are the per-class treatment effects of a K-class mixture of linear outcome models? | At most four classes; treatment as-if randomized within class; EM finds a local maximum from seeded restarts, not a certified global one; a class effect is not an individual effect. | point-only | `antecedent.latent_class` | `latent_class` |
| Compact runtime export | Can a fitted linear-in-coefficients effect model be exported as an identity-bound runtime object that predicts only inside its declared support? | A finite basis, a coefficient vector and a covariance; a query outside the support or inside a mask region refuses. The model-based standard error carries no extrapolation or misspecification uncertainty. | point-only | `antecedent.compact_export` (`CompactExport`) | `compact_export` |

For nonlinear mediation the bootstrap standard error is reported with interval
status `closed_calibration_unmeasured`: no interval is produced.

## What these routes do not claim

- No route here produces a calibrated interval, credible interval or coverage
  statement.
- A structural or assumption range is a range over declared cases. It is never
  a probability.
- A route that composes an external value keeps that value's trust label
  (externally attested unless an exact-request receipt is retained) and never
  upgrades it to native.
- Declaring a route is not executing it. How these routes combine is described in
  [composition](composition.md); related pages are
  [external scientific objects](guides/build/scientific-interfaces.md) and
  [joint distributions](joint-distributions.md).
