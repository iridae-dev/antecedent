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

Rust `rebind` and Python family `refresh` share a frozen statistical product only
when its fit dependencies match. Changing response worlds can reuse bootstrap parameters, posterior draws
or an OU mechanism fit. Changing observations, mechanism parameters, priors,
factual initial laws or fitting policies invalidates the relevant fit. Changing
the execution kernel also refits. Numerical initialization used only to propagate
an already fitted response is distinct from a factual initial-law dependency.
The prepared owner uses synchronized caches and can outlive the original builder.

These changes preserve each family's model-conditional interpretation, failed
draw slots, unresolved solution regions and unmeasured calibration. Compilation
does not add identification or scientific authorization. See
[repair acceptance](compiler-integration-acceptance-3.0.md) for the reviewed
execution/resource scope and verification. Aggregate Y15 and statistical release
attestation remain separate backlog obligations.

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
source. `inspect()` and every report carry the lowered program identity. The stamp
is local: it is not written to an artifact and is ignored by report equality.

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
unresolved before any fit. Python `inspect()` reports them as `design_unresolved`
and, until a fit on the handle, as `unresolved_directions` (previously `None`).
Python `PreparedCyclicFamily.contract()` and `PreparedCyclicSystem.contract()`
return the native contract, and `interface_contract.check_composition_with_cyclic`
binds `check_interface_composition_with_cyclic`.
