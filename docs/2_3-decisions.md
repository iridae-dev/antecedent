# Decision contracts (2.3)

`antecedent_design::DecisionContract` declares a decision problem durably. It is
separate from the 2.2 callback-based `DecisionProblem`, which stays for
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
assumptions. The verdict is `UniquelyOptimal`, `Indistinguishable` (the leader
cannot be separated from the others within two paired standard errors) or
`NoAdmissibleAction`. An exact finite law has no sampling error, so ties there
are only exact ties; the same rows read as a Monte Carlo sample are
indistinguishable when noise covers the gap.

Not yet in the result: structural and support robustness, decision uncertainty
beyond the standard error, graph-dependent choice and the artifacts.

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

Decision refusals share one shape, `antecedent_core::StructuredRefusal` (an alias
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
identify, bind, inspect, export, decide and a fresh-process reload; ranking a
study is not covered because EVSI and design ranking do not exist yet.
