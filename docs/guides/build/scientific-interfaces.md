# Connect scientific components and bring in outside results

Two components can agree on array shape, dtype and a variable name and still disagree on units, population, intervention regime, support, whether a value is a mean or a law, or what its interval means. This guide covers the declarations that make those things explicit, so a connection is accepted or refused before anything runs, and the routes by which a number computed outside Antecedent can reach a result.

Antecedent checks declarations against each other. It does not check that a declared population is the one the data came from, that a declared calibration record exists, or that a declared evidence set is complete. Those are claims of the artifacts the declarations describe.

## Name the quantity

A `ScientificQuantity` names one scalar coordinate by its stable variable ID, display name, role, units, population, intervention regime, horizon, functional, conditioning and scale or transform. Identity never relies on position or display label. Two coordinates are the same only when variable ID, population, regime, horizon, functional, conditioning and transform all match, and units are declared, never converted.

Quantities appear wherever a value changes hands: each coordinate of a joint distribution artifact, each requested and returned value of an external result, the utilities and outcomes of a decision contract, and optionally each value of a response artifact. When a failed identification needs more evidence, `repair.obligations(contract)` names it in the same coordinates, so a study must cover the required joint factor in the named regime. Separate marginal observations cannot replace it. See [decisions, design and repair](../../decisions.md#repair).

## Declare what a component produces and consumes

An interface contract declares a component and its ports. Each port states a quantity, the estimand it answers, its support, the operations on offer or required (`point`, `mean`, `law`, `draws`, `joint_draws`, `covariance`, `interval`, `posterior`; none implies another), what its uncertainty means, its calibration label, the measurement models it is established under and the evidence it rests on.

`check_connection` accepts a producer and a consumer only when every declaration is compatible, and otherwise returns every defect with its registered reason code and the declaration that would license the connection. This pair matches on name and units and fails scientifically: the producer's population is a trial cohort, and the consumer needs a law where the producer offers a mean.

```python
from antecedent import interface_contract as ic
from antecedent.interface_contract import InterfaceContractRefusal
from antecedent.joint_distribution import ScientificQuantity


def quantity(population):
    return ScientificQuantity.outcome(
        "ldl_change", units="mg/dL", population=population, regime="do(statin=1)"
    )


producer = ic.InterfaceContract(
    "estimator",
    produces=[
        ic.ProducedPort(
            "effect",
            quantity("trial_cohort"),
            "estimand:ate",
            ic.SupportDomain.real(),
            operations=("mean",),
        )
    ],
)
consumer = ic.InterfaceContract(
    "decision",
    consumes=[
        ic.ConsumedPort(
            "effect",
            quantity("adults"),
            "estimand:ate",
            ic.SupportDomain.real(),
            operations=("law",),
            uncertainty="interventional_predictive",
        )
    ],
)

try:
    ic.check_connection(producer, "effect", consumer, "effect")
except InterfaceContractRefusal as error:
    print(error.reason_code)
    for refusal in error.refusals:
        print(refusal.detail, refusal.expected, refusal.supplied)
```

The refusal lists three defects, in order: `interface_contract.population_mismatch` (expected `adults`, supplied `trial_cohort`), `interface_contract.operation_unavailable` and `interface_contract.uncertainty_meaning_missing`. Each carries a remedy.

Read these refusals by family:

| Family | The defect |
| --- | --- |
| `interface_contract_quantity_mismatch` | Another variable, role, units, population, regime, horizon, functional, conditioning, transform, semantic identity or support. |
| `interface_contract_assumption_mismatch` | Different measurement models, a consumer that needs declared evidence the producer does not declare, or a missing provenance stage. |
| `interface_contract_operation_unavailable` | A required operation is not offered, such as a mean where a law is needed. |
| `interface_contract_uncertainty_mismatch` | The required uncertainty meaning is missing, of another paradigm or another object, such as a sampling interval where a posterior is needed. |
| `interface_contract_calibration_not_held` | The consumer relies on measured calibration and the producer does not hold it. |
| `interface_contract_invalid` | The declaration is malformed. |

An accepted connection carries the producer's label and evidence unchanged. **A connection never raises a label**: a point-only producer stays point-only, an unmeasured interval stays unmeasured, and a consumer that relies on measured calibration is refused instead of being served an unmeasured interval.

`check_composition(components, connections)` checks a whole set before any execution. It refuses unknown or duplicated components and ports, a consumed port fed twice, a feedback cycle (reported with its path, as `interface_contract_cyclic_composition`), calibration declared `inherited` whose upstream does not hold a measured record, and two inputs required to be disjoint that share an evidence source. Cyclic scientific composition is not supported, and a loop is never unrolled or iterated. The check is bounded at 1024 components and 8192 connections.

A contract has one canonical, versioned identity independent of port declaration order, and it exports as a checksummed artifact that a consumer re-validates and checks against an identity it retained independently. The full field-by-field reference, the specification and its second reference consumer are in [scientific interface contracts](../../interface-contracts.md).

## Adapt declarations you already have

Bridge, measurement-model and mechanism-composition declarations are instances of the one interface-contract schema and are decided by the same connection check. An adapter takes its port facts explicitly: support, estimand identity, uncertainty meaning and measurement models. Nothing is inferred. An unstated support is real-valued, no uncertainty meaning is stated, and no measurement model is claimed. The produced calibration is `unmeasured` when a value carries uncertainty and `point_only` otherwise; it is never measured by default, and a bridge never raises a calibration its source did not hold.

Whether two components share evidence is decided by the evidence ledger's row-level overlap, not by comparing identity strings. The same source, or distinct sources that share observations (a subset, a resample, a derived source), are refused as `interface_contract.shared_evidence` where disjoint evidence is required. A source the ledger does not hold, or whose observations cannot be computed, is `dependence_unknown`: unknown is never read as disjoint. Without a ledger, a disjointness requirement refuses instead of being skipped. Rows of several inputs may be paired only when they share an alignment identity, and a covariance or declared independence is never a route to row pairing.

| Refused request | Detail | What to do |
| --- | --- | --- |
| Shared observations where disjoint evidence is required | `interface_contract.shared_evidence` | Use disjoint sources. |
| Overlap that cannot be computed | `interface_contract.dependence_unknown` | Register the sources in the ledger. |
| Independence declared where overlap is unknown | `joint_law_propagation.evidence_overlap_unknown` | Decide the overlap in the ledger first. |
| Inputs with different alignment identities | `alignment_mismatch` | Supply draws from one joint law. |
| A bridge value kind that names no operation | A port with no operations | Declare a carried value kind. |

## Bring in a number computed elsewhere

A number computed outside Antecedent reaches a result through one of four routes. All four share the provider-trust vocabulary, the verification receipt, the provider identity and the shared artifact container. They differ in what is being brought in and what you already hold.

| | Registry and handoff | `external.bind` | Recalculation callbacks | Distribution artifacts and priors |
| --- | --- | --- | --- | --- |
| Use it when | A Python package runs the estimator (EconML, CausalML, a lab model) and Antecedent should record what ran, or an adjustment set is handed to a learner you fit yourself. | You already hold the numbers (a response grid, a scalar effect, a column of per-row effects) and want them bound to an identified contract. | The outside mean must be recomputed when a dependency changes (a new seed, grid or provider version) under declared replay rules. | The outside object is a joint law, draws or a prior that a decision or composition consumes. |
| Entry point | `providers.register`, `handoff.adjustment(...).attach(...)` | `external.response(...)` then `spec.bind(Response)` | `recalc_external.ExternalCallbackSession` | `JointDistributionArtifact`, `DistributionIdentity`, prior artifacts |
| Verification | `providers.verify(name, fixtures, evidence_origin=)` issues a receipt, and a request outside the receipt stays attested. | `Response(probes=...)` verifies the exact provider object; `attested_by=` attests. | Always attested. A stateful or side-effecting policy is not replayable. | Trust comes from recorded evidence matched on provider and snapshot, never from a claimed label. |
| What you get | `ProviderResult` and, for a labelled effect, `ExternalEstimate` with `.claim`. | `BoundExternalClaim` (grid) or `BoundExternalColumn` (per row), with lineage and support labels. | A bound claim and an artifact that replays. | A distribution artifact and its trust label, consumed by composition. |

Choose by what you hold. If you have the numbers already, bind them. If something must run Python to produce them and you want that run recorded and optionally verified, use the registry; `handoff.attach` is the bring-your-own-fit form of it. If the number has to follow a changing dependency, use a callback. If it is a law or a prior rather than an effect, use a distribution artifact. Results from every route load through `antecedent.artifacts.loads` as their declared header, and a decoded header never raises trust: the `trust` there is the producer's claim.

### Bind a response grid

`external.response(identification, outcome_units=...)` derives the required coordinates from the identified query, so the units you give are the units the bound claim carries. This example binds a response grid with the closed form E[Y | do(a)] = 1 + 2a:

```python
import antecedent as ant
from antecedent import external

edges = [("x", "a"), ("x", "y"), ("a", "y")]
names = ["x", "a", "y"]
identification = ant.identify(
    graph=edges, names=names, query=ant.ResponseCurve("a", "y", grid=[0.0, 1.0, 2.0])
)
spec = external.response(
    identification,
    outcome_units="mmHg",
    population="target",
    require_evidence=("factor:z",),
    require_assumptions=("ignorability",),
)
provider = external.ProviderObject(
    provider_id="lab",
    object_id="curve",
    version="v3",
    snapshot="snap-9",
    request="req-1",
    meaning="interventional_predictive",
    capabilities=("mean",),
)
claim = spec.bind(
    external.Response(
        provider=provider,
        values=[1.0, 3.0, 5.0],
        evidence=("factor:z",),
        assumptions=("ignorability",),
        attested_by="lab",
    )
)
print(claim.trust, claim.native, claim.support_status)
```

The claim's `trust` is `externally_attested`, `native` is `False` and the support status is `missing_evidence`, because the provider declared no support. Its lineage runs from the causal contract through each evidence factor and the exact provider execution to the claim, so an independent consumer can answer where the numbers came from. External trust stays external: the bound claim supplies no interval, calibration or native licence, and provider-declared uncertainty stays a provider declaration.

The binding refuses an unidentified contract, a provider of the wrong kind, a different graph, a different coordinate (population, regime, horizon, conditioning, transform or units), a different distribution meaning, non-finite values, missing evidence or assumptions, and a provider claiming native licence. An observational law offered for an interventional coordinate is accepted only with a checked equivalence for the same graph and regime. `P(Y | T = v)` can stand for `P(Y | do(T = v))` over a grid only when the conditioning maps to exactly one interventional regime and every other field agrees; asserting the justification is yours.

A per-unit effect is not a grid: one coordinate per row would put thousands of full quantities in the identity. A `ColumnResponse` binds one quantity and the digest of the ordered values, with a provider-declared support per row or for all rows. The artifact carries no rows, so a consumer that holds the column checks it against the digest and one changed row is a different claim.

### Handle refusals

Every refusal on these routes carries a registered `reason_code` and, where one is known, a `remedy`. Binding and verification raise `ExternalRefusal`, and the registry and adjustment handoff raise `ProviderRefusal`, both a `CausalUnsupportedError`. Branch on `reason_code` and read `remedy` for what to change:

| Code | The outside result was refused because |
| --- | --- |
| `external_capability_missing` | The exact operation was not declared. A mean never implies a CDF or quantile, and sampling is not an approximation of either. |
| `external_verification_failed` | A probe was missing or outside its predeclared tolerance. |
| `external_binding_mismatch` | The result does not match the checked contract. |
| `quantity_semantics_mismatch` | A coordinate differs in population, regime, horizon, units or another field. |
| `distribution_meaning_mismatch` | The law means something other than what the contract accepts. |
| `effect_not_identified` | The contract itself is unidentified. |
| `invalid_argument` | The declaration is malformed. |

### Verification and trust

Trust states stay apart: `externally_attested`, `verified_extension` and, for Antecedent's own routes, native licence. A verified extension is never native, and metadata or a stored artifact cannot mint native trust. Verification takes independent probe values (shape, normalization, moments, known truth, seeded behaviour, support, update coherence, monotonicity, uncertainty), and each required probe must be present once and within its predeclared tolerance. You compute the probe values independently; verification does not invoke providers. A receipt covers only the exact contract it digested, so a different request, snapshot, version or capability set needs its own check. Neither check supplies an interval or sampling calibration for a CDF or quantile.

## Callbacks: a function Antecedent does not own

A callback is a law, a posterior or a composition of them that the library can ask questions of but cannot inspect. The declaration is an instance of the same interface-contract schema and the same provider contract. There is no separate callback format and no second registry. Each operation (`mean`, `sample`, `cdf`, `quantile`, `posterior_update`) is negotiated on its own: a mean does not supply a law, and draws do not supply a distribution function or an update.

What matters to a host:

* A stored answer is reused only when identities prove it is the same question of the same declared callback. A callback with opaque or undeclared dependencies is never reused, and a side-effecting callback is refused rather than run to find out.
* Planning and verification never invoke a side-effecting or stateful callback. A side-effecting callback runs only with an explicit opt-in and is retried only under a declared idempotent contract.
* Output is data. It is accepted only after validation (shape and count, finiteness, support, a distribution function within [0, 1] and non-decreasing, a non-decreasing quantile function), and a malformed law is refused whole. Nothing is repaired, sorted or clipped.
* Invocation counts are exact. The wall-clock bound is enforced after the call returns, because an in-process closure cannot be preempted, so a late result is discarded. A host that must survive a hung callback runs it in a process it can terminate.
* A result is **diagnostic** unless its contract is replayable, its dependencies complete and its provider attested or verified against its receipt. A diagnostic result is inspectable and no claim is built on it.
* A callback supplies values, not coverage. An interval from a callback law, and a posterior or probability guarantee, are not licensed.

A transported claim can be offered to a callback input that is declared to accept transported claims, through `transport_handoff.hand_to_callback`; the claim is never native target data and its calibration label is never raised (see [provider-sourced transport](../transported-decisions-and-measurement.md)). A callback result exports as a portable artifact holding the declaration, the request and the recorded output. It is a declaration and a portable output, not a promise that a closure runs elsewhere, and a loaded artifact's stored trust is the producer's claim that you recompute against the declaration you hold. The full protocol, reuse table and bounds are in [arbitrary law, posterior and nested callbacks](../../provider-callbacks.md). Because a closure runs with your authority, read [callback security](../../callback-security.md) before supplying one, and never construct a provider from text a user or a model produced.

## What stays closed

| Request | Answer |
| --- | --- |
| A connection with a feedback cycle | Refused, with the cycle path. |
| Measured calibration relied on where the upstream does not hold it | `interface_contract_calibration_not_held`. |
| A mean offered where a law is required, or a sampling interval for a posterior | Refused as an unavailable operation or an uncertainty mismatch. |
| An interval or posterior for a composed quantity | Not licensed. No whole-method coverage has been measured at a composed coordinate, and a component interval does not cover a composition. |
| A native-licensed label on an outside result | Refused. |

To carry a bound or composed result between services, export it and accept it with the rules in [save, hand off and accept results](artifacts-and-handoff.md). To use a library's estimator beside Antecedent's identification, see [work with other causal libraries](interoperability.md).
