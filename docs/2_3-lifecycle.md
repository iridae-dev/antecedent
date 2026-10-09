# The 2.3 lifecycle: analyze, bind, decide, plan, bundle

This page is the practitioner path through the 2.3 Python surface. Each step
hands a typed object to the next, and each step keeps the label of the claim it
received. Nothing on the path upgrades a claim: an external mean stays
externally attested, a point value stays a point value, and a sensitivity range
stays an assumption range.

The snippets are taken from `python/tests/test_decision_flow_bridge.py`,
`python/tests/test_lifecycle.py` and `python/tests/test_composition_bundle.py`,
which execute them against hand-derived values.

```text
analyze  ->  bind external  ->  inspect the claim  ->  decide
                                                         |
                          bundle and export  <-  rank a study
```

A result can enter the decision step from either end. A native `analyze` result
answers a decision directly (step 1). A foreign provider's numbers enter through
a bound external claim (step 2). Both are `Contract.evaluate` sources.

## 1. Analyze and decide directly

`Contract.evaluate` takes a supported analysis result. Declare outcome units once
in its scientific coordinates or in `analyze(outcome_units=..., dose_units=...)`;
the contract supplies an unambiguous matching outcome/unit/population when no
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
print(result.claim())            # what the result is, in words

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
decided.selected                 # ("high",): E[Y | do(a=0.5)] is about +1, do(a=-0.5) about -1
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

## 2. Bind an external result

When another system supplies the numbers, Antecedent checks the causal claim and
binds the provider's grid to an identified contract. Identification comes first.

```python
import antecedent as ac
from antecedent import external

ident = ac.identify(
    graph=[("x", "a"), ("x", "y"), ("a", "y")],
    names=["x", "a", "y"],
    query=ac.ResponseCurve("a", "y", grid=[0.0, 1.0, 2.0]),
)
spec = external.response(
    ident,
    outcome_units="mmHg",
    population="target",
    require_evidence=("factor:z",),
    require_assumptions=("ignorability",),
)
provider = external.ProviderObject(
    provider_id="lab", object_id="curve", version="v3", snapshot="snap-9",
    request="req-1", meaning="interventional_predictive", capabilities=("mean",),
)
claim = spec.bind(
    external.Response(
        provider=provider,
        values=[1.0, 3.0, 5.0],           # attested closed form: E[Y | do(a)] = 1 + 2a
        evidence=("factor:z",),
        assumptions=("ignorability",),
        attested_by="lab",
    )
)
```

## 3. Inspect the claim

```python
inspection = claim.inspect()
inspection.native                 # False: always
[link.id for link in inspection.lineage][-1]    # "claim"
claim.identity                    # a string; retain it independently of the bytes
claim.identity_fields             # the mapping the string encodes
data = claim.export()
same = spec.load(data, expected_identity=claim.identity)
```

`BoundExternalClaim.identity` is a string and `identity_fields` the mapping it
encodes; `ExternalSpec.load(bytes, expected_identity=...)` accepts either. The
claim is never `native_licensed`, and any uncertainty the provider declared stays
a provider declaration, not an Antecedent interval. See
[external scientific objects](2_3-external-science.md).

## 4. Decide on the claim

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
result.selected                   # ("treat",): 2 * 5 - 1 = 9 against 2 * 1 - 1 = 1
result.stages_behind()            # includes causal_contract, external_provider, claim
```

The decision is labelled **point-only attested**. It carries the claim's lineage
plus the decision contract, so the foreign provider stays visible in
`result.lineage`. It is not exportable on its own; to carry it forward use
`composition_bundle.mean_decision` (step 6). See
[decisions](2_3-decisions.md) and [decision breadth](2_3-decisions-breadth.md).

If the decision depends on an assumption you cannot test, carry it through
[sensitivity and robustness](2_3-sensitivity-and-robustness.md) before acting. If
the claim must travel to another population, see
[transport and counterfactuals](2_3-transport-counterfactuals.md).

## 5. Rank the next study

A decision that is not yet clear-cut raises the next question: which study is
worth running? `DesignDecision.from_contract` extracts the affine action
utilities from a contract, and `rank_designs` values each candidate study by the
expected value of sample information (EVSI), net of cost.

```python
from antecedent import design
from antecedent.joint_distribution import ScientificQuantity

def q(name):
    return ScientificQuantity(
        variable_id=f"schema:{name}", variable_name=name, role="outcome",
        units="dimensionless", population_id="target", regime_id="observational",
        horizon=0, functional_id="state",
    )

state = q("state")
guess = decision.Contract(
    actions=(
        decision.Action("guess0", inputs=(state,), utility=1.0 - decision.x(0)),
        decision.Action("guess1", inputs=(state,), utility=decision.x(0)),
    ),
    utility_units="utility",
    criterion=decision.Criterion.expected_utility(),
    target_population="target",
)
declared = design.DesignDecision.from_contract(guess, prior=design.StatePrior.draws([0.0, 1.0]))
```

Candidates, signal, cost map and the ranking itself are in
[design ranking](2_3-design-ranking.md). The result says what its order means in
`ranking.basis`: `"identification"`, `"net_value"` or `"evsi"`. Frozen oracle for
the guess decision: signals of accuracy 3/4 and 5/8 have EVSI 1/4 and 1/8; with a
utility-unit cost of 1/10 the net values are 3/20 and 1/40.

```python
# candidates, signal and cost_map are built as on the design ranking page
ranked = design.rank_designs(candidates, decision=declared, signal=signal, cost_map=cost_map)
ranked.basis                       # "net_value"
ranked.best.id
design.consume(ranked.export(), expected_identity=ranked.expectation())
```

`from_contract` refuses what it cannot represent, by name: an action whose
utility is not affine in the state, a criterion other than expected utility, and
hard constraints.

## 6. Bundle and export

A bundle links the parts of a composed decision as a typed graph. Every edge
carries the digest of its upstream node, so a changed upstream identity changes
every dependent node and the bundle identity.

```python
from antecedent import composition_bundle as cb

result_bytes = cb.mean_decision(contract, claim)     # point-only attested result
builder = cb.Bundle.builder()
builder.add_artifact("external_claim", claim, node_id="claim")
builder.add_artifact("decision_contract", contract.export(artifact_id="contract"), node_id="contract")
builder.add_artifact("decision_result", result_bytes, node_id="result")
bundle = builder.connect("claim", "result").connect("contract", "result").build()

data, identity = bundle.export(artifact_id="my-bundle"), bundle.identity

# In another process, under the identity retained independently of the bytes:
consumed = cb.consume_bundle(data, expected_identity=identity)
consumed.require_verified()
consumed.claim_label                      # "point_only_attested"
consumed.value("result", "treat.expected_utility")   # 9.0
```

`add_artifact(artifact)` infers the node kind from the container when you do not
name it (`add_artifact("auto", artifact)` is equivalent), and an artifact object
whose `export(artifact_id=...)` returns container bytes may be passed instead of
bytes. A decision over aligned joint draws is labelled `joint_draw`; a decision on
an external mean is `point_only_attested`, and a joint-law question asked of a
mean-only claim fails as `unsupported_law`. A bundle never claims that a
serialized result recreates an executable study. See
[composition](2_3-composition.md).

## Handling refusals along the path

Every step can refuse, and a refusal is information, not an error string. The
stage modules raise a subclass of `antecedent.errors.StructuredRefusal`, which
exposes the same machine-readable fields on every class:

```python
from antecedent import msm_sensitivity
from antecedent.errors import CausalUnsupportedError, StructuredRefusal

try:
    # strata as on the sensitivity page; a perturbation bound below one is refused
    msm_sensitivity.msm_ate_sensitivity(strata, 0.5)
except StructuredRefusal as refusal:
    refusal.code        # "invalid_argument" here; a scientific refusal has e.g. "cell_not_licensed"
    refusal.detail      # "msm_sensitivity.lambda_below_one"
    refusal.offending   # the field, quantity, node or identity at fault, or None
    refusal.remedy      # what to change in order to proceed, or None
except CausalUnsupportedError as refusal:
    # ExternalRefusal and the design-ranking refusals are CausalUnsupportedError with the
    # same fields: reason_code, remedy, and (where present) detail / offending / stage.
    refusal.reason_code, refusal.remedy
```

Branch on `code` and `detail`, not on the message text. A refusal whose code is
`invalid_argument` rejects a malformed argument rather than a scientific route and
is also a `ValueError`. See
[Refusal and partial knowledge](refusal-and-partial-knowledge.md#structured-refusals-and-their-remedies)
for the base class, the families and the common remedies.

## Where each label comes from

| Step | Object | Label it carries |
| --- | --- | --- |
| Analyze | `analyze(...)` result | the result's own claim; read `result.claim()` and `result.calibration` |
| Bind | `BoundExternalClaim` | externally attested (`verified_extension` only with an exact-request receipt); never native |
| Decide | `Decision` | point-only; no interval, coverage or probability claim |
| Sensitivity | `MsmResult`, `SensitivityArtifact` | assumption range, calibration unmeasured |
| Rank | `DesignRankingResult` | point-only values; Monte Carlo error and rank uncertainty are diagnostics |
| Bundle | `Bundle` | `joint_draw` or `point_only_attested`; certifies decoding, identities and bindings only |

The whole path is one runnable script,
[`decision_lifecycle.py`](../examples/python/decision_lifecycle.py). Runnable
examples for the later steps:
[`composition_bundle.py`](../examples/python/composition_bundle.py) (bundle and
consume) and [`repair_obligations.py`](../examples/python/repair_obligations.py)
(the evidence a failed contract owes, which is where study candidates come from).

Next: [sensitivity and robustness](2_3-sensitivity-and-robustness.md),
[transport and counterfactuals](2_3-transport-counterfactuals.md),
[design ranking](2_3-design-ranking.md) and
[recalculation](2_3-recalculation-capabilities.md).
