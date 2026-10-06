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

Evaluation, `DecisionResult` and the contract and result artifacts are separate
steps and are not part of this layer yet.
