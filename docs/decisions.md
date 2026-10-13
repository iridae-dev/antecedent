# Decision contracts

`antecedent_design::DecisionContract` declares a decision problem durably. It is
separate from the callback-based `DecisionProblem`, which stays for
`evaluate_decision` and the preposterior routes.

## What a contract declares

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

## What each functional needs

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

## Identity

`DecisionContract::identity` is a canonical BLAKE3 digest. Reordering actions or
constraints leaves it unchanged; any semantic edit to an action, quantity,
utility, constraint, criterion, scope, units or structural policy changes it.

The contract and result artifacts are not part of this layer yet.

## Evaluation on aligned draws

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

## Structural uncertainty

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

## Artifacts and replay

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

A result over structures (`StructuralDecisionResult`) has no artifact yet.

## Python surface

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

## Refusals

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
built with `format!`, so the promotion gate can match it against the declared
refusals. A missing source lists the needed representations, comma-joined, in
`expected`. Typed design signal, update and cost-unit failures do not exist yet,
so `design_signal_invalid` and `design_cost_units_mismatch` are not claimed.

## Provenance chain and mean-only sources

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
`decision.mean_source_not_replayable`). `python/tests/test_lifecycle.py` runs
identify, bind, inspect, export, decide and a fresh-process reload. Ranking a
study with `design.rank_designs` is covered by [design ranking](design-ranking.md)
and the [lifecycle](lifecycle.md).

## Structural result artifact

`StructuralResultArtifact` (`antecedent_design::decision_structural_artifact`) stores a `StructuralDecisionResult` in the checksummed container (`decision_structural_result_v1`, bounded at 8 MiB). It retains the contract identity, each atom's id, probability and status (evaluated, unidentified, or unevaluated with its reason), the BLAKE3 digest of each evaluated atom's draws, the declared structural policy, the per-action per-structure values, ranges and masses, the unidentified, unevaluated and evaluated masses, and the verdict. `from_bytes` loads only under the consumer's own contract identity and atom id/digest list, and `replay` recomputes `evaluate_structural` and requires a byte-identical body, so a resealed verdict or number is refused.

Structural Bayes choices require all positive probability mass to be evaluated.
Partial weighted values remain visible without renormalization, and an omitted
remainder is counted as unevaluated mass. Unknown payoffs can reverse a partial
ranking, so it produces `InsufficientScience`. A zero-probability atom cannot
exclude a Bayes action. Bayesian averaging of component quantiles or maximum
regrets is refused: neither is the corresponding functional of a mixture law.

## Cohort-mean treatment contrast of a fitted effect model

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
`joint_law_required`, `cohort_decision.law_not_supplied` / `cohort_decision.joint_law_required`)
and names the producer of aligned outcome draws that would license it.

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
never an interval and is not passed to a decision input; every interval request is refused
`cell_not_licensed` (`cohort_decision.interval_not_licensed`). The whole-method design that
would open the route (a linear-CATE DGP with a declared cohort sampling plan, the whole method
including cohort weighting and support-row selection, nominal level 0.95, cohort sizes 50 to
5000 and fitted-box coverage fractions 0.5 to 1.0, boundary and misspecified-truth probes) is
recorded in the `cohort_decision.interval` promotion record and is not measured.
