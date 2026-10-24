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
establish. A bridge of a point claim carries **one mean**; the interval of that mean is a
separate, scoped request (below) that needs the source claim's standard error. A bridge whose
source is the aligned draws of a posterior or a predictive law carries **those draws**
([Bridges of distributions](#bridges-of-distributions)). No identified set or joint outcome law is
produced.

## The contract

| Part | What it states |
|---|---|
| kind | one of six bridges (below) |
| source and target | the full `ScientificQuantity` coordinate and the *class* of each (construct, measured outcome, stated preference, realized behavior, surrogate, terminal outcome, predictive quantity, causal quantity, parameter, response, endpoint, utility) |
| value kinds | what kind of object the source value is and the consumer requires |
| evidence scope | the population and regime the bridge's own evidence (validation trials, measurement model, linkage study) was established for |
| mapping | identity, a declared affine map, the trial-level least-squares line, or a binary-misclassification correction |
| mapping uncertainty | optional: the variances of a declared affine map's intercept and slope and their covariance; absent is an exact map |
| source dependence | optional: the source estimate and the mapping declared independent (naming the evidence), or the covariance of the slope estimator and the source estimator; absent is not a declaration of independence |
| estimand bridge | optional: the translation of an experiment's population, window, dose, duration and exposure regime into the bridge's target, with the recency and spend differences |
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
| an interval of a bridged claim with no level named, or a posterior or joint law | `cell_not_licensed` | `interval_not_licensed` |
| an interval of a bridged value with no source standard error | `cell_not_licensed` | `interval_source_uncertainty_required` |
| an interval at a design no coverage record measured | `cell_not_licensed` | `interval_scope_outside_measured` |
| draws offered by an associational, refused, bounded or incomplete claim | `cell_not_licensed` | `source_not_draws` |
| draws that are a sampling, bootstrap or empirical law | `distribution_meaning_mismatch` | `draws_meaning_mismatch` |
| draws that are independent marginals | `joint_law_required` | `draws_not_aligned` |
| draws that are non-finite, constant under a declared covariance, or a target that repeats a coordinate of the table | `scientific_bridge_invalid` | `draws_unusable` |
| draws outside the validated surrogate range, or no admissible rates for a recorded prevalence | `bridge_mapping_not_licensed` | `draws_outside_domain` |
| an uncertain map, a trial-level line or a correction with no declared dependence on the source | `sampling_dependence_unknown` | `source_dependence_missing` |
| a trial-level line or a correction that declares its mapping uncertainty a second time | `scientific_bridge_invalid` | `mapping_uncertainty_invalid` |
| a declared covariance no joint law has | `bridge_mapping_not_licensed` | `dependence_not_a_covariance` |
| a bridge that is not an affine map offered as a chain stage | `route_not_supported` | `chain_stage_not_affine` |

The two closed pairs refuse whatever is declared: no calibration bridge is
licensed for a model score, and no assumption turns the importance of a feature
in a predictive model into a causal effect. Ask the causal question directly.

## Mean, law, posterior, set, joint law

A mean, a predictive law, a posterior, an identified set and a joint outcome
law are not interchangeable, and no bridge relaxes that rule. A bridge carries
a value of one kind to a quantity of the same kind. A point mean and the aligned
draws of a posterior or a predictive law have executed routes; an identified set
and a joint outcome law have none. A consumer that requires a law from a bridged
mean is refused at `BridgedClaim::require_value_kind`, and one that requires a
mean from bridged draws at `BridgedDistributionClaim::require_value_kind`.

## Bridges of distributions

When the source is a posterior or a predictive law, the claim is a table of aligned draws (a column
of a `DistributionArtifact` whose rows are joint realizations), and so is the bridged claim: the
same table with the bridged column in place of the source column. Row `i` of the output is the
image of row `i` of the input, every other column keeps its place, and the table's alignment
identity is the source's, so the bridged column still pairs by index with every column it was
sampled with. A bridge that broke the pairing would silently turn a joint law into marginals, so
independent marginals are refused.

The value kinds are `posterior` (posterior draws of a parameter or a causal functional) and
`predictive_law` (posterior predictive draws), and the distribution's own meaning must agree with
the contract's. A bootstrap, a sampling law or an empirical law is neither. An associational claim
is never a bridge source. The column is found by its quantity coordinate, not its position.

What each mapping adds to the spread of its source:

| Mapping | Draws |
| --- | --- |
| identity, declared affine with no mapping uncertainty | exact: `a + b m` for each draw `m`, bit for bit |
| declared affine with mapping uncertainty | each row draws the intercept `a` and slope `b` from their declared normal law, conditionally on that row's source draw through the declared covariance `c` of the slope and the source estimate |
| trial-level line, `k >= 5` trials | the Student-t law of the least-squares coefficients on `k - 2` degrees of freedom (a scale mixture, one `sqrt(nu / chi2_nu)` per row), covariance `s^2 (X'X)^-1`; a predictive law adds the trial residual `s`; every draw must lie in the validated range, and a law is neither truncated nor extrapolated |
| measurement correction | sensitivity and specificity drawn from Beta laws of their declared means and variances, redrawn per row until the pair makes the recorded prevalence attainable (`1 - Sp <= q <= Se`, Youden above its minimum); a row no rates produce is refused |

The source's spread comes only from its draws and the mapping's spread only from the mapping, so
neither is counted twice: the only link is the declared dependence. A trial-level line or a
correction takes its uncertainty from its own evidence and so declares none a second time, and an
uncertain map that declares no dependence is refused rather than assumed independent. For a normal
mapping law the moments of the bridged draws are those of `a + b M` under the trivariate covariance
`[[Va, d, 0], [d, Vb, c], [0, c, V]]`: mean `a0 + b0 mu + c` and variance
`Va + mu^2 Vb + b0^2 V + 2 mu b0 c + 2 mu d + Vb V + c^2`, which is the delta-method variance of
`BridgeContract::propagate_affine_uncertainty` (exact here). The coefficients are drawn from a
deterministic stream of the seed, so the mapped table is a function of the contract, the source
table and the seed.

The bridged table is read as an `uncalibrated_spread` with calibration `unmeasured`. An interval is
requested through the same measured-coverage gate as the interval of a bridged mean, on the mean twin
of the bridge at the source's mean with the standard deviation of the draws as the source standard
error. A map with uncertain coefficients (propagated by draws, which no record measured), a declared
slope-source covariance (the records were measured with independent evidence) and any design outside
a record are refused `interval_scope_outside_measured`. The declared assumptions stay unresolved
obligations, together with the assumption that the mapping's evidence is independent of the other
columns of the table, which the bridge cannot check.

### As a stage of a propagation chain

`BridgedDistribution::chain_stage` expresses an affine bridge as a measurement-mapping stage of a
propagation chain. The chain reads the source draws, the drawn slope and the drawn intercept (three
columns of one aligned table) and applies `source * slope + intercept` with the slope and intercept
as the stage's own inputs, so a chain of bridge, transport and utility propagates the source, the
mapping's coefficients and every later stage jointly, row by row: the utility is the mean of the
utility of each joint draw and not the utility of the mean. The declared slope covariance enters
the chain as a shared factor of the source and the slope. A measurement correction (a ratio of its
inputs) and a predictive law through a trial-level line (a per-row residual) are not affine maps of
their inputs; derive the bridged draws and supply them to the chain as a source.

### As a node of a composition bundle

A bridge artifact is a `scientific_bridge` bundle node, replayed by its own consumer. A bridge of
draws publishes the bridged aligned table (`law = joint_draw`, trust `unverified`, the digest a
decision over it retains), so a decision result rests on the bridge node as on a distribution; a
decision computed over another realization of the bridge, or a bridge of a table other than the
distribution upstream of it, is refused `composition_bundle.tampered_quantity`. A bridged mean
publishes `mean_only`. An `observation_query` node is replayed through the observation-query
consumer and a parameter-redraw query publishes its aligned draws the same way; a
`propagation_chain` node is replayed stage by stage and publishes its answer's value and spread (the
record keeps summaries, not the output column).

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
where a correction is used, and the mapping uncertainty, source dependence and
estimand bridge where declared), the source claim as the bridge saw it and the
derived identity, value and fit. A contract that declares any of the three, and a
bridge of draws, are written in format version 2 under the feature marker
`scientific_bridge_extended_v1`; every other bridge keeps its version-1 bytes and
identity, and a version-1 reader refuses version 2 before interpreting its
payload. A bridge of draws embeds the source draw table, the seed of the mapping's
random stream and the summary of the mapped table; the consumer maps the stored
draws again and requires the contract identity, claim identity, mean, variance,
draw count, alignment and the digest of the mapped table to equal the stored
values bit for bit. A consumer rebuilds both, applies every
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
exported = claim.artifact("surrogate-bridge")
again = sb.consume_scientific_bridge_artifact(exported.data, exported.expectation.identity)
claim.require_value_kind("predictive_law")            # refused: value_kind_substitution
claim.interval()                                      # refused: cell_not_licensed (unscoped)

claim = bridge.derive(sb.SourceClaim.declared("biomarker-effect", 0.8, surrogate, standard_error=0.1))
claim.interval(0.95)                                  # the interval of the bridged value, inside a
                                                      # measured design; refused outside it
```

A bridge of draws is prepared with `prepare_distribution_bridge` and bridges a
`JointDistributionArtifact` (the `consume_scientific_bridge_artifact` of its export returns a
`ConsumedBridgeDistribution`):

```python
contract = sb.BridgeContract(
    ..., mapping=sb.AffineMapping(intercept=0.5, slope=2.0),
    mapping_uncertainty=sb.MappingUncertainty("calibration-fit", slope_variance=0.04),
    source_dependence=sb.SourceDependence.independent("disjoint-samples"),
    source_value="posterior", target_value="posterior",
)
bridged = sb.prepare_distribution_bridge(contract).derive(posterior_draws, seed=7)
bridged.distribution()        # the aligned draws of the terminal outcome
bridged.reading               # "uncalibrated_spread"
parts = bridged.chain_stage("terminal")   # sources and a measurement-mapping stage for pc.propagate
bridged.interval(0.95)        # refused outside a measured design
```

Refusals are `ScientificBridgeRefusal` (a `CausalUnsupportedError`) with `reason_code`, `detail`
and `remedy`. A mean-valued bridged claim retains its target quantity and can reach
`Contract.evaluate` and `InverseQuery.evaluate` directly when its units, population, regime
and functional meaning match the consumer. `claim.as_mean_source()` exposes the canonical
`results.MeanSource` explicitly. The handoff remains point-only; see
[producer-consumer-matrix.md](producer-consumer-matrix.md).

## Interval of a bridged value

A bridged value is a function of the source estimate and of what the mapping
was estimated from, so its interval has two parts, both stated and nothing
else: the source claim's **declared standard error** (a declared point value
states none, so the request refuses without one) and the mapping's own
estimation uncertainty.

| Mapping | Interval | Reference |
| --- | --- | --- |
| identity, declared affine | the source interval mapped (ends ordered); the map is declared exact | normal |
| trial-level line, `k` validation trials | `value +- t * sqrt(s^2 (1 + 1/k + (x - xbar)^2 / Sxx) + slope^2 se^2)`, the prediction interval of a new trial's terminal effect | Student-t, `k - 2` degrees of freedom |
| binary-misclassification correction | delta method: `Var(p) = (se_q^2 + p^2 Var(Se) + (1 - p)^2 Var(Sp)) / J^2`, `J = Se + Sp - 1`, clipped to `[0, 1]` | normal |

An identity or affine map adds no uncertainty and nothing to measure, so its
interval is licensed as an exact transform of the declared source interval.
The estimating mappings are published only where a current coverage record
measured the construction (surrogate bridges over 15, 20 and 40 validation
trials; corrections with rates estimated from 100, 200 and 400 validation
units, Youden index at least 0.3, rate standard errors at most 0.05; level
0.95), and the record is read from the shipped coverage table. The trial
effects are treated as exact points, as the mapping does; the declared
assumptions of the bridge stay unresolved obligations; invariance across
populations is a declared premise; none of this is individual-level surrogacy.
The interval is not stored in the artifact: the source standard error is held
beside the derived result and is supplied again to a consumer that wants it.

## What is not licensed

No interval outside a measured design or without the source standard error (for a
bridge of draws: none for an uncertain map or a declared source-mapping covariance),
no identified set or joint outcome law, no bridge of a bootstrap, sampling or empirical
law, no associational claim as a source, no truncation or extrapolation of draws
outside a mapping's validated range, no calibration of a model score, no individual-level
surrogacy, no extrapolation of a trial-level relationship beyond the validated
range, no differential measurement error without the claim's arm.
