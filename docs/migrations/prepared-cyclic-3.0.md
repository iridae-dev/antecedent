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
