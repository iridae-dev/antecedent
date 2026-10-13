# Scientific bridges

A claim answers a question about one scientific quantity. A consumer often asks
about another: the construct a measurement stands for, the behavior a stated
preference predicts, the terminal outcome a surrogate tracks, the causal
quantity a prediction approximates, the response a parameter encodes, the
utility an endpoint is scored against. The claim does not answer those
questions by being renamed. A **bridge** is the explicit, separately identified
derivation that states the extra assumptions under which the source answers the
downstream question.

A bridge produces a *new* claim. The source claim is not modified and stays
separately inspectable; the derived claim has its own identity (it binds the
contract, every declared assumption and the source claim identity), its own
provenance chain, and a handoff receipt that lists what the bridge did not
establish. Everything is **point-only**: a bridge carries one mean. No
interval, posterior, predictive law, identified set or joint outcome law is
produced.

## The contract

| Part | What it states |
|---|---|
| kind | one of six bridges (below) |
| source and target | the full `ScientificQuantity` coordinate and the *class* of each (construct, measured outcome, stated preference, realized behavior, surrogate, terminal outcome, predictive quantity, causal quantity, parameter, response, endpoint, utility) |
| value kinds | what kind of object the source value is and the consumer requires |
| evidence scope | the population and regime the bridge's own evidence (validation trials, measurement model, linkage study) was established for |
| mapping | identity, a declared affine map, the trial-level least-squares line, or a binary-misclassification correction |
| assumptions | declared, each with an identity, a statement and the evidence it rests on |
| transportability premise | required when the evidence scope is not the claim's population and regime |

## The six bridges

| Bridge | Principal assumption | Mapping | Independent truth in the tests |
|---|---|---|---|
| latent construct and measured outcome | the measurement model, named by its identity | Rogan-Gladen inversion and its forward map | 0.31 recorded at sensitivity 0.9, specificity 0.8 is 11/70 latent |
| stated preference and realized behavior | a behavioral linkage | identity, or a declared calibration | stated 0.5 under calibration 0.6 is 0.3 |
| surrogate and terminal outcome | validation of the surrogate | trial-level least squares inside the validated range | four trials: slope 0.92, intercept 0.06, 0.8 maps to 0.796 |
| predictive quantity and causal quantity | a named identification | identity | association 0.54 refused, standardized effect 0.30 bridged |
| model parameter and intervention response | the parameter's interpretation | declared affine contrast | coefficient 2.5, contrast +0.4 is 1.0 |
| scientific endpoint and decision utility | a declared utility function | declared affine utility | 0.3 at 30,000 per unit is 9,000 |

An affine map of a mean is exact for the mean of the mapped quantity because
expectation commutes with affine maps. A nonlinear map is not carried: the map
of a mean is not the mean of the mapped law.

## Refusals

Every refusal has a registered reason code, a `scientific_bridge.*` detail and
the declaration or evidence that would license the request.

| Request | Code | Detail |
|---|---|---|
| association as intervention | `bridge_substitution_refused` | `association_as_intervention` |
| proxy as construct | `proxy_offered_as_construct` | `proxy_as_construct` |
| stated preference as realized behavior | `bridge_substitution_refused` | `stated_preference_as_realized_behavior` |
| predictive importance as causal effect | `bridge_substitution_refused` | `predictive_importance_as_causal_effect` (closed) |
| model score as probability | `bridge_substitution_refused` | `model_score_as_probability` (closed) |
| scientific endpoint as utility | `bridge_substitution_refused` | `scientific_endpoint_as_utility` |
| a missing assumption | `bridge_assumption_missing` | `assumption_missing`, `assumption_evidence_missing` |
| a transported bridge without its premise | `bridge_assumption_missing` | `transport_premise_missing` |
| population, regime, horizon, conditioning, quantity or units mismatch | `bridge_binding_mismatch` | `population_mismatch`, `regime_mismatch`, ... |
| validation minimums not met, an extrapolated surrogate effect, an unattainable recorded prevalence | `bridge_mapping_not_licensed` | `validation_trials_too_few`, `outside_validated_range`, `outside_attainable_range`, ... |
| a mean, law, posterior, identified set or joint law where another is required | `bridge_substitution_refused` | `value_kind_substitution` |
| an interval, posterior or joint law of a bridged claim | `cell_not_licensed` | `interval_not_licensed` |

The two closed pairs refuse whatever is declared: no calibration bridge is
licensed for a model score, and no assumption turns the importance of a feature
in a predictive model into a causal effect. Ask the causal question directly.

## Mean, law, posterior, set, joint law

A mean, a predictive law, a posterior, an identified set and a joint outcome
law are not interchangeable, and no bridge relaxes that rule. A bridge carries
a value of one kind to a quantity of the same kind, and only a point mean has an
executed route. A consumer that requires a law from a bridged claim is refused
at `BridgedClaim::require_value_kind`.

## Transport

A surrogate relationship validated in one population and used in another needs
its own transportability premise. The contract states it: the relation the
bridge uses is the same in the claim's population and regime as in the one it
was established for, with an identity and the evidence it rests on. Without it
the bridge is refused naming the premise; with it the derived claim is labelled
transported and the premise travels as an unresolved obligation. The premise is
a declaration. The bridge performs no selection-diagram derivation of
invariance, and a transported value is never presented as native evidence.

## Lifecycle

`PreparedScientificBridge::prepare` checks the contract once, before any
number is formed. `derive` with a refreshed source claim about the same
quantity keeps the contract, its identity, the assumptions and the verdict and
recomputes the value, the fit and the claim identity; it refuses a claim about
another quantity or units. `with_contract` is the only way assumptions, mapping
or scope change and re-runs every check. A refused derivation or replacement
leaves the prepared bridge as it was. A withdrawn or superseded source makes the
derived claim inapplicable and preserves its history.

## Artifact

`scientific_bridge_v1` stores the contract (with the full measurement model
where a correction is used), the source claim as the bridge saw it and the
derived identity, value and fit. A consumer rebuilds both, applies every
licensing check again, derives the claim again and requires the stored values to
match bit for bit, against an identity it retained independently of the bytes.
Replay protects against corruption, unsealed edits and resealed edits that
change the answer. It does not protect against a producer that seals a bridge
whose declared assumptions are false: those are declarations.

## Python

`antecedent.scientific_bridge` builds the contract and presents the result; every check,
mapping, identity and refusal is Rust's.

```python
from antecedent import scientific_bridge as sb

bridge = sb.prepare_scientific_bridge(contract)      # checks before any number is formed
claim = bridge.derive(sb.SourceClaim.declared("biomarker-effect", 0.8, surrogate))
claim.value, claim.fit.r_squared, claim.unresolved   # 0.796, 529/535, ("assumption:validation",)
exported = claim.export("surrogate-bridge")
again = sb.consume_scientific_bridge_artifact(exported.artifact, exported.identity)
claim.require_value_kind("predictive_law")            # refused: value_kind_substitution
claim.interval()                                      # refused: cell_not_licensed
```

Refusals are `ScientificBridgeRefusal` (a `CausalUnsupportedError`) with `reason_code`, `detail`
and `remedy`. A bridged claim reaches `Contract.evaluate` and `InverseQuery.evaluate` only through
the caller-declared `MeanSource` / `MeanClaim` adapter; see
[producer-consumer-matrix.md](producer-consumer-matrix.md).

## What is not licensed

No interval or posterior for a bridged value (whole-method coverage is not
measured; the mapping's own sampling uncertainty is not propagated), no law or
set transformation, no calibration of a model score, no individual-level
surrogacy, no extrapolation of a trial-level relationship beyond the validated
range, no differential measurement error without the claim's arm.
