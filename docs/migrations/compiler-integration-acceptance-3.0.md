# Compiler integration repair acceptance

This repair addresses the reviewed 3.0 compiler integration, authority, resource,
and observation defects. It does not promote new causal methods or calibrations.
H03, H04, H12 and Y15 remain open for their unrelated aggregate obligations.

## Execution boundaries

| Workflow | Scientific checks | Retained execution | Physical execution and resources | Independent consumption |
| --- | --- | --- | --- | --- |
| Checked Gaussian adjustment | Target, functional, identification and assumptions retained together | Public compile and prepare share one product; changed rows rebind without a second identification | Costed choice dispatches direct or materialized assembly; row-level uncertainty refuses streaming; source, fit and result reservations survive owner disposal | Existing result consumers retain their contract checks and historical observations |
| Bayesian dose grid | Randomized-design and explicit coefficient-model premises; still unlicensed and unmeasured | Prepared numerical and IO programs; compatible targets reuse the fit, changed fit dependencies invalidate it | Caller admission/cancellation; independent grid leases, shared fit guards and guarded numerical/artifact results | Recompiles and executes the same program, checks request/data identities and bitwise numerical replay |
| Cyclic families | Original SCC, dynamic, constraint and observational functional checks; model-conditional and unresolved labels preserved | Typed programs separate inspection, execution and encoding; compatible targets reuse fits | Reports and fits own leases; draw/world/time/failed-slot bounds precede execution; encoding scratch is admitted | Recompiles source proofs and executes the same typed programs; IO delegates finite/Gaussian calculations to estimation engines |
| Decisions and contract-bound bridges | Current ledger authority and bridge licensing | Withdrawal, supersession and trust loss invalidate dependent current decisions | Independent descendants remain usable | Current decision artifacts bind source/result identities and revisions to caller-supplied authority; historical numerical replay grants no current permission |
| Ordinary and prepared batches | Existing per-workflow checks | Checked shared scheduler | Serial/parallel admission, bounded inner work, cancellation and cleanup; live reservations cannot be reclaimed by competing jobs | Native observations distinguish attempts, estimates and caller declarations |

The standalone weighted Gaussian capacity component executes its existing streamed
reduction and verifies backend selection and capability refusals. It is a numerical
component, not a causal workflow or an interchangeable substitute for retained-row
adjustment uncertainty. Lazy and accelerator alternatives remain unavailable.

## Cyclic family acceptance evidence

Integrated cyclic families have committed fixtures under `parity/cyclic/` (the
specified feedback pair and its three distinct quantities, the overlapping-loop
clamp kept as failed mass, asymptotic-normal, bootstrap and unresolved-direction
recovered laws, a dynamic-inference bootstrap, and the version-one record). They
are written by `cargo test -p antecedent --test cyclic_family_fixtures -- --ignored
write_`. `cyclic_independent_consumer` replays the committed bytes against the
separately committed identity on an independent CPU context, bit for bit, and
refuses identity, report, contract and version tampering. `cyclic_dag_regression`
keeps acyclic support on the DAG schedule. Python replays each fixture in a fresh
process. The capability row is `integrated_families` in
`parity/cyclic/acceptance.toml`; the producer/consumer cells are in
`parity/producer_consumer.toml`. These checks do not measure calibration.

## Ownership and compatibility

Returned numerical reports retain reservations until their last shared owner is
released. Exported artifact byte vectors become caller-owned after export;
serialization/container scratch is admitted while encoding. These are conservative
allocation reservations, not allocator instrumentation or process RSS. Caller
copies and Python-decoded representations are outside the native allocation account.
Native NumPy views keep their lease in the actual array base, including slices.

Native execution V2 records optional copied bytes, retained admission volumes and
observed output storage mode. V1 and absent observations remain readable; unknown
formats are refused. Imported receipts stay historical. Failure/cancellation facts
are available through Rust observed APIs and Python `error.execution_facts`.
No receipt grants additional scientific permission.

See the focused migration notes for authority, compiled execution, dose-grid,
cyclic and native execution interface changes. New 3.0 IO convenience entry points
return guarded compiled results; internal bare-wire conversions are not public
execution APIs.

## Verification and outstanding release evidence

The affected Rust libraries passed 2,961 selected unit tests (80 existing ignored
tests and the separately documented attestation-dependent test were not selected).
The public workflow/artifact selection passed 326 integration tests. A final
136-test regression selection passed, including ownership, withdrawal, dose-grid,
cyclic, encoding refusal, backend dispatch and native-wire checks. These selections
overlap and are not an additive count of distinct tests.

Clippy passed with warnings denied for the changed libraries and their test targets.
The affected facade/estimation/IO all-feature build and public API snapshots passed.
The test-registry, Python surface (428 entries), checked-execution inventory
(723 cells, 791 citations), and cyclic fixture/catalog inventory checks passed.
Inventory validation is not a claim that every cited test was executed.

The final retained-resource-policy selection passed eight tests. The affected
Python selection passed 244 distinct tests, including fresh-process artifact
consumption, Rust/Python agreement, cancellation/refusal observations and NumPy
ownership. One new receipt test initially supplied fewer draws than the existing
minimum; correcting its fixture made the targeted retry pass. The selection also
exposed one unchanged attestation-dependent bridge test, documented below.

Whole-branch release gates, calibration campaigns and unfinished 3.0 feature
acceptance are outside this repair. The checks focus on the changed workflows,
rather than treating this partial 3.0 branch as a release candidate.

The broad unit run exposed the existing
`every_measured_transport_variant_matches_original_execution_basis_bit_for_bit`
expectation of calibrated transport. The coverage snapshot has zero attesting
records and is byte-for-byte unchanged from repair baseline `04afda690`.
The production measured route correctly refuses with
`coverage_record_not_attesting`. Its positive test and the calibration registry
are deliberately unchanged. Remeasurement/release attestation is separate TODO
work, outside this repair; implementation acceptance must not be presented as
statistical promotion.

The Python test
`test_a_scoped_interval_needs_the_source_standard_error_and_a_measured_design`
likewise expects an attested scoped interval. It receives
`coverage_record_not_attesting` from the unchanged coverage snapshot. Neither
this expectation nor the scientific licensing checks were weakened.

Verification used `cargo nextest run --profile ci` with the affected library,
public integration and changed-family selections; `cargo clippy --lib --tests --
-D warnings` for the seven affected crates; and `cargo check --all-features` for
the facade, estimation and IO crates. Public snapshots were regenerated and
checked with `bash scripts/gate_public_api.sh`. The Python tests ran through
`python/.venv/bin/python -m pytest -q` against the freshly built release extension.
The checked-execution and cyclic checks validated their inventories; they did
not execute the entire release evidence matrix. Formatting was applied to the
changed Rust files, and `git diff --check` passed.
