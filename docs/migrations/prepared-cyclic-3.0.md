# Prepared cyclic execution in 3.0

Cyclic preparation now compiles and retains a typed program without fitting,
sampling, simulating or encoding an artifact. `inspect()` remains cheap after
execution: `preparation_evaluations` counts preparation work and stays zero.
For fitted equilibrium preparations, `fits_performed` starts at zero and counts
fits actually performed by that handle during execution.

`Study::prepare_cyclic_family` retains the original family request, source
identity and applicable equilibrium, dynamic or equation compiler products.
`estimate` executes that retained program; `export` then encodes its result.
The prepared owner no longer retains an encoded artifact or a complete report.
Independent artifact consumers execute the same program. Finite-law evaluation,
Gaussian density fitting and canonical integration now live in the estimation
engines; IO authenticates the graph source and exact executable expression, then
delegates to those engines. Existing public IO law/report paths remain reexports.
The numerical component APIs accept caller-established program metadata and grant
no identification or scientific authority by themselves. Existing version-one artifacts retain their replay
semantics and request identity checks.

Python `rebind(data)` changes observation fields on a new prepared family;
`reconfigure(worlds=..., scientific=..., method=..., uncertainty=...)` changes
scientific declarations on a new branch. System handles use `rebind(rows)` for
training rows and `reconfigure(worlds=..., noise=..., initial=..., model=..., policy=...)`
for their declarations. Compatible branches share a frozen statistical product only
when its native fit dependencies match. Changing response worlds can reuse bootstrap parameters, posterior draws
or an OU mechanism fit. Changing observations, mechanism parameters, priors,
factual initial laws or fitting policies invalidates the relevant fit. Changing
the execution kernel also refits. Numerical initialization used only to propagate
an already fitted response is distinct from a factual initial-law dependency.
The prepared owner uses synchronized caches and can outlive the original builder.

These changes preserve each family's model-conditional interpretation, failed
draw slots, unresolved solution regions and unmeasured calibration. Compilation
does not add identification or scientific authorization. See
[repair acceptance](compiler-integration-acceptance-3.0.md) for the reviewed
execution/resource scope and verification. Aggregate execution accounting and statistical release
attestation remain separate obligations.

Returned equilibrium and family reports retain admission reservations after their
prepared handle is dropped. Cancellation leaves those live output reservations
in place until their last owner is released. Frozen fits keep their reservation
with their data, and copied fitted models are admitted before cloning. Output
bounds include draw/world/noise axes, OU world covariances, trajectory times and
bounded isolation regions. A structural shape traversal and explicit concrete slot-width bounds account for
numeric contents and failed `Option` slots, rather than using JSON spelling as
a memory estimate. Serde can erase sequence referent types; large failed-fit
slots therefore use the actual Rust `Option<CyclicStructureFit>` width.
The byte counts are conservative allocation reservations, not process resident
memory. Resource guards are local, omitted from artifacts, and ignored by
scientific equality; independent consumers acquire their own output leases.

Export counts the bounded logical payload and admits serialization/container
scratch before allocating it. The returned artifact byte vector becomes
caller-owned after export; the report's resource guard covers its numerical
report, rather than promising accounting for an independently retained vector.

The public IO `evaluate_cyclic_functional` and
`evaluate_gaussian_cyclic_functional` convenience workflows now execute through
the retained family compiler and return `CyclicFamilyReport`. Match its `answer`
field to access the finite or Gaussian numerical report. Keeping the outer report
keeps the output reservation alive. Numerical component evaluation remains available from `antecedent-estimate` without acquiring causal authority.

A recovered linear law uses the same prepared family owner. Preparation retains the
draws and does not fit. Interventions execute from that law; failed and
inadmissible draws stay in the result. An acyclic free support uses the DAG
schedule. A directed cycle is still refused by that algorithm, with its existing
next attempt, and only then compiled under the cyclic license. The recovered law
stays model-conditional. Compilation grants no scientific license.

## Typed worlds, recovered laws and dynamic inference

The family program also accepts worlds named by variable id and core
interventions: `linear_worlds`, `equilibrium_worlds` and `ode_worlds`. Each world
is admitted at preparation; an action the family cannot perform is refused with
its next attempt and is never rewritten into another action. Results carry the
scientific world id, the uniqueness certificate (never physical stability), the
solve route, and failed and withheld mass. `interventions()` on the prepared
family returns these worlds for every family that has them.

`experimental_recovery` fits a linear law from declared clamp-experiment regimes
at execution, draws its asymptotic-normal, bootstrap or identified-subspace
posterior uncertainty, and evaluates hard-clamp worlds from the portable law.
Unresolved directions stay unresolved in every draw. `recovered_linear` gains an
optional draw origin; a request without one keeps its encoding and identity.
Evidence projections put a declared sampling law on the sampling port, a posterior
on the parameter-posterior port, and the law within one draw on the interventional
port. Supplied draws keep their previous projection.

`dynamic_inference` adds a bootstrap sampling law, observed information, or a
posterior to ODE, stationary OU and shared-mechanism OU fits; changing only the
method or the quantities reuses the fit. The single-system handle converts to a
family program with `to_family`, which shares rebinding, evidence and replay.

Fit reuse is keyed by the typed fit dependencies of each family, not by the
encoded request with fields removed. The standalone recovery-plan byte format in
`antecedent-estimate` (`export_retained_recovery_plan`, `consume_recovery_artifact`)
is retired; recovered laws are exchanged through the family artifact.

## Lowered programs, contracts and the version-two record

Every cyclic preparation (`prepare_cyclic`, `prepare_cyclic_family`,
`prepare_recovered_linear`, `prepare_experimental_recovery`) now obtains its plan
from a cyclic lowering with the same six layers as a compiled route. Keyed worlds
become scientific targets in a feedback vocabulary: a `Cyclic` graph family;
equilibrium, continuous-time and stationary frames; and structural world actions
(hard clamp, mechanism replacement, additive shock, initial-state change) that are
never rewritten into one another. Identification is not a graphical product: a
specified or fitted model stays conditional on that model, a clamp-experiment
recovery carries its design digest and the directions the design leaves
unresolved, and a theorem-scoped cyclic input/output SCM identification keeps its
source. Live preparation and execution metadata retain a local lowered program
identity. That stamp is not written to an artifact and is ignored by report
equality. A loaded result does not reconstruct a prepared handle or invent that
local identity.

Prepared cyclic handles expose `contract()` with layered identities (target,
identification, identification product, program, inference binding, observation,
data snapshot), the `cyclic.solution.v1` tag and, for an experimental recovery, a
recovery identity over the experiment data snapshot, the regime contract and the
recovery derivation. Existing acyclic targets keep their canonical bytes and
identities; the new vocabulary is append-only.

Family export now writes a version-two record: the request and report section
`cyclic_family_v2` and a `cyclic_family.contract` section. A consumer verifies the
contract from bytes (`verify_cyclic_contract_against_body`), re-derives every
payload from the request and replays the answer; a tampered or consistently
resealed contract is refused. Version-one records keep replaying and carry no
contract (`consume_cyclic_family_contract` refuses them with a next attempt).

A certified cyclic product enters an interface-contract composition as one
declared component (`check_interface_composition_with_cyclic`,
`CyclicFamilyContract::as_component`). Feedback among components stays refused.

## Row design, contracts and composition from Python

`ExperimentalFitOptionsWire` gains an optional `sampling_design`. Absent, the
encoding and the request identity are unchanged. Declared, request compilation
refuses an asymptotic-normal or posterior origin under a clustered or time-series
design (`reason=estimator_inference_mismatch`), with a bootstrap under the declared
design as the next attempt, and refuses a bootstrap that resamples under another
design. Python `prepare_experimental_recovery` writes a declared design into the
fit options for a row-independent origin and no longer decides the rule itself;
the refusal keeps its reason code and next attempt.

`PreparedStudy::design_unresolved` returns the rows a clamp design leaves
unresolved before any fit. Python preparation `inspect()` reports the declared
`design_unresolved` and `unresolved_directions`. Execution does not rewrite these
declarations: the fitted law's actual draw statuses and unresolved directions
belong to `result.law`.
Python `PreparedCyclicFamily.contract()` and `PreparedCyclicSystem.contract()`
return the native contract, and `interface_contract.check_composition_with_cyclic`
binds `check_interface_composition_with_cyclic`.

## Retained Python results and bounded exploration

`prepared.execute()` returns an `Outcome[CyclicResult]`; use
`antecedent.outcomes.require_success` to select its successful result, or inspect
the outcome's refusal. Executing retains a native report without immediately
encoding an artifact. `result.inspect()` returns bounded metadata, while
`result.to_dict(execution=...)` explicitly materializes the complete report.
`result.export(execution=...)` returns an `ExportedArtifact`; `bytes(artifact)`
selects its encoding. These are separate materialization boundaries.

`result.assumptions(offset=0, limit=64)` returns a `CyclicPage` of retained native
assumption records. `evidence.columns(offset=0, limit=64)` pages selected quantity,
world and uncertainty-port metadata. Each immutable page exposes `items`, `count`,
`offset`, `next_offset`, and `metadata`; `page.next()` reads the next page or returns
`None`. The maximum limit is 256. `page.to_dict()` converts only that selected
page. Reading summaries, pages, representations or already executed worlds does
not fit, sample or solve again.

For experimental recovery, `result.law` is a `RecoveredCyclicLaw` sharing the
execution owner. Its `inspect()` preserves draw origin, draw count and failed
mass. `law.draw_status(offset=0, limit=64)` and
`law.unresolved(draw, offset=0, limit=64)` page actual retained draw records;
`law.to_dict(execution=...)` explicitly converts the full law. A result without
an experimental recovered law raises a typed argument error for `.law`.
Prepare additional hard-clamp targets directly from that retained owner:

```python
branch = result.law.prepare(
    [World(47, Set("A", 3.0))], variables=["A", "B"], outcome="B",
)
continued = ant.estimate(branch)
```

This continuation assumes the variables declared by the original experiment.
The branch retains the original recovery recipe, draw order, failed slots and
uncertainty origin. Changing its worlds does not fit or sample another law.
A law from `ant.load(...)` supports the same operation without reconstructing
a live study. New observations or a different inference declaration require
the original experimental preparation. See the runnable
[recovered feedback law example](../../examples/python/recovered_feedback_law.py)
for the experiments, analytic intervention checks and independent load.
Cross-branch composition still requires explicit shared-evidence and draw
alignment declarations; sharing a law does not establish independence.

`result.interventions()` selects the existing native worlds without executing
them again. A `WorldAnswer` exposes its status and certificate; a
`RecoveredWorldAnswer` retains draw origin and the original weighted failed or
inadmissible slots. Numerical readers select the appropriate kind of answer:
`world.mean("B")` applies to a Gaussian-law world, while
`world.numeric(row=0, field="values")` borrows one equilibrium row. A failed slot
has no numeric buffer and returns `None`. `result.numeric(world, row=0, draw=0,
counterfactual=False, field="values")` selects a buffer directly. Returned
memoryviews are read-only and keep their exact native execution owner alive
after the result and preparation are dropped. `world.to_dict()` converts that
world explicitly; it does not convert its siblings.

Selected-world evidence composes through `world.evidence(schema, variable,
time=None, execution=...)`, or through `result.evidence(schema, selectors,
execution=...)` for several explicit selectors. `EvidenceSchema` declares the
population and physical units; selectors retain world ids, time coordinates and
the scientific uncertainty port. A Gaussian-world selector projects its mean,
with variance retained as a diagnostic; it does not silently pass its entire
Gaussian distribution. `evidence.decide_means(contract)` accepts the applicable
point/mean decision contract, and `evidence.joint_law()` requires an eligible
complete aligned law. Failed posterior slots, bootstrap sampling laws and
numerical root sets cannot silently become outcome distributions. Numerical
enclosure decisions use the distinct `evidence.decide_enclosures(contract)`
operation and preserve unresolved regions.

## A specified feedback model, selected evidence and independent loading

This model has feedback A→B→A with coefficient product 0.2. Its observational
equilibrium mean is zero. Clamping A to 2 removes A's equation and gives a mean
of 1 for B. The calculation is conditional on the specified linear model and
disturbance covariance.

```python
import math

import antecedent as ant
from antecedent.cyclic import (
    EvidenceSchema, EvidenceVariable, LinearCyclicModel, World,
)
from antecedent.intervention import Set
from antecedent.outcomes import require_success

policy = ant.Execution(seed=19, threads=1)
model = LinearCyclicModel(
    ("A", "B"),
    (0.0, 0.4, 0.5, 0.0),
    (1.0, 0.0, 0.0, 1.0),
)
prepared = model.worlds(
    [World(0), World(7, Set("A", 2.0))],
    component_id="feedback",
    source_id="specified-linear-model",
    certifying_evidence=["coefficient product 0.2; invertible free system"],
    execution=policy,
)
result = require_success(prepared.execute(execution=policy))
worlds = require_success(result.interventions())
assert math.isclose(worlds[0].mean("B"), 0.0, abs_tol=1e-10)
assert math.isclose(worlds[7].mean("B"), 1.0, abs_tol=1e-10)

schema = EvidenceSchema(
    "feedback-B-means",
    "specified model population",
    (EvidenceVariable("A", "meter"), EvidenceVariable("B", "meter")),
)
evidence = require_success(worlds[7].evidence(schema, "B", execution=policy))
assert evidence.quantities[0].units == "meter"
assert evidence.quantities[0].population_id == "specified model population"
assert evidence.columns(limit=1).items[0]["world"] == 7

selected = worlds[7].numeric(field="mean")
assert selected.readonly
artifact = result.export(execution=policy)
# Retain this producer-side expectation independently when exchanging bytes.
expected = artifact.expectation
loaded = require_success(ant.load(bytes(artifact), expected=expected, execution=policy))
loaded_worlds = require_success(loaded.interventions())
assert math.isclose(loaded_worlds[7].mean("B"), 1.0, abs_tol=1e-10)
assert loaded.inspect().stage == "consumed"
del prepared, result, worlds
assert math.isclose(selected[1], 1.0, abs_tol=1e-10)
```

`ant.load` invokes the independent native consumer, including the format's
verification and replay. Passing an artifact does not implicitly supply an
expected identity; expectations remain explicit inputs. Loaded results expose
acceptance through the enclosing outcome and inspection through the common
interface, without restoring a live study or dataset.

Version-two family artifacts retain their verified contract identities. Supported
version-one family records and standalone system records retain their actual
source/request identity but do not contain that family contract. Inspection marks
missing contract metadata unavailable, and assumption pages carry
`available=False` with `reason="not_present_in_source_artifact"`; no substitute
digest is presented as the missing scientific identity. The local lowered
program identity is absent after loading even when a version-two contract is
available. Independent replay checks the encoded claim; it does not supply
calibration or release attestation.
