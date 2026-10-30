# Coverage validation provenance

Coverage measurement is scoped to the execution that produced it. New v2 records
include that execution's support validation coordinate in both metadata and the
record identity. A Study execution uses its identity-bound executed refutation
suite; an unverified or missing execution stamp stays unknown. A non-Study
procedure may state `not_applicable` only when its producer establishes that it
has no support-matrix validation axis. Unknown and legacy records never imply
`none`.

Historical measurements remain unchanged. `parity/coverage_validation_bindings.toml`
proposes reviewed bindings keyed by exact record id, measurement commit and test
identity. Each binding names the test and support-bridge source blobs at that
commit, the attested calibration surface blob, and a source-review rationale.
Validation checks the registry identity and every blob. Each row has its own
review status: only rows marked `approved` are exported to the snapshot used
by the runtime and collector; `pending_review` rows grant no authority. Matching a binding never changes its
record's coverage, grid, estimator, attestation, or test; current-code
attestation and all existing construction and scope checks remain required.

Validation mode does not transfer between `none`, `cheap`, and `full` without
separate construction-specific evidence. Existing fixture equality checks alone
do not establish equal interval delivery and equal refusal or selection behavior
across the full calibration distribution. Unresolved historical records and
procedures without an established mode remain readable but cannot claim an exact
facade validation match.

Historical on-axis bindings also record the source structure when proven.
Explicit, accepted, and graph-posterior measurements bind only their respective
support cells. The old `fixed` construction spelling alone does not establish
whether the scored input was explicit or accepted. New accepted records include
an accepted-structure marker in their identity; the resolver fails closed when
that marker or the matching source is missing.
