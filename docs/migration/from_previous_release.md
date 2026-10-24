# Migration from the previous release

The previous release is Antecedent 2.4. The published predecessor is 2.4.1. 2.4.0 stays the historical control. This note is the migration for callers and artifacts moving from that release onto the current contracts.

Family interchange, previous-release goldens, and fresh-process consumers are not decided here.

The durable container format is unchanged. Readers that already migrate container sources 0.1 through 0.5 still do that work. This note does not add a container rewrite.

## Route records

The public route manifest is `parity/capabilities/route_manifest.toml`. Its `[[route]]` records are the inventory of the previous release: one per query and graph class, with a status, a design/time model and an inference text. Its `[[method]]` records are the current inventory: one per licensed (query, graph class, estimator, inference), and one per graphless, execution-only or fitting-gap route, plus the facade methods that have no previous route. A method names the previous route it carries in `route`, its time frame in `design`, and its realization in `inference`.

`antecedent_io::release_migration::migrate_route_record` carries one previous record onto the current methods. The method records decide the mapping; the previous text is read only to say which axes it named.

| Previous record | Outcome |
| --- | --- |
| Closed, no current method | `RefusalKept`: the recorded reason is kept as written |
| One current method carries the route | `OneToOne`: the method, its estimator, status and axes |
| Several methods carry the route | `Split`: one `MethodAssignment` per method, in inventory order |
| Not licensed, no current method | `Retired`: the route left the inventory; stop requesting it |
| Licensed, no current method | Error `RouteWithoutCurrentMethod`: request the query through the method that replaced it |
| Licensed, no carrying method is licensed | Error `LicenseNotCarried`: license one of the methods, or accept its execution-only standing |
| Status outside the manifest vocabulary | Error `UnknownStatus` |

An axis is a time frame (static or temporal) crossed with a realization (frequentist or Bayesian). Each assignment lists the axes its method occupies. A previous record that names an axis none of its methods occupies reports it in `unserved_axes`; the method record is the authority for its time frame, so this is a correction of an over-named text and not a break. A method that keeps less than a licensed route (an execution-only Bayesian response beside a licensed frequentist one) has `narrowed` set. Methods that name no previous route are additions and are listed by `added_methods`.

The mapping over the current manifest:

| Previous status | Records | Outcome |
| --- | --- | --- |
| Licensed | 92 | split by estimator and realization (two or more methods each); none is a break |
| Execution-only | 83 | one-to-one |
| Graphless | 38 | one-to-one |
| Fitting gap | 1 | one-to-one |
| Closed | 71 | refusal kept |

A licensed route splits across its estimators: the average-effect Dag route is fourteen methods. Four execution-only records name an axis their method does not occupy (the temporal attribution routes name the static graph their lag-aligned fit unfolds to, the simulator names temporal rollouts, and the regime-survival route names static regimes); each reports it in `unserved_axes`. One method, the joint Bayesian response of the co-determined intervention-response route, is `narrowed`. Forty-three facade methods have no previous route and are additions. No record is a break.

`consume_previous_route_records` reads a previous-release route artifact, `{"routes":[{"id","status","design_time_model","inference"}]}`, and returns one outcome or error per record, in order, so one break does not hide the others. `crates/antecedent-io/tests/release_migration_consumers.rs` reads the manifest as plain text and carries every route through the migration; it also loads the frozen artifact `tests/fixtures/previous_release_routes_v1.json`.

## Changed identities

### Score-row ids

On the previous release these public fields stored each row id in four bytes:

- `ScoreTableWire::row_index`
- `ScoreTableKey::row_index`
- `CandidateSelectionWire::estimate_rows`
- `CandidateSelectionWire::screen_rows`
- `CheckedAipwRowsWire::rows`

They are eight-byte ids now. There is no four-byte alias.

The score-table row-index digest, kind `score_reuse.row_index`, hashes those bytes explicitly. Recompute it from the eight-byte ids. A retained four-byte digest of a nonempty index does not match and is refused. An empty index has no width bytes, so the two digests coincide; that is byte equality, not a widening of a stored digest.

A shared-design batch key still digests a dense four-byte fold position. That key is not the score-table row index.

The checked AIPW row digest hashes the CBOR body of the row binding. This note does not claim those CBOR bytes changed for an integer that already encoded the same value.

### Scientific target digest

The previous release has no scientific-target digest to translate. The current digest is the canonical payload under `antecedent.scientific_target.v1` at payload format 3. Recompute it from the target. A carried digest that is not that value is refused. Format 2 is not format 3: the payload version is inside the digest, so every scientific-target digest moves, including a contrast whose worlds were already hashed by content.

What the digest contains:

- A world, a world collection, and a dose × time surface are content-based. A caller-chosen world id or a reference flag does not enter. A world collection is the set of those contents, so id order does not enter. A dose × time surface hashes each cell's content in grid order and does not hash the cell id.
- A contrast hashes the content of its signed worlds. Caller-chosen world labels and the reference flag do not enter that digest. The reference is the baseline term of the formula.

## Changed refusal reasons

None were renamed. `parity/support_closed.toml` is the same file as in the published 2.4.1 tree. A reason the manifest repeats is kept as written. Examples that are not rewritten:

- `policy_route_not_licensed` (a transform or a non-constant assignment on `gcm.model.intervention_programs`; the program stays simulable)
- `Attribution is licensed only for an explicit Dag at validation none.`
- `AnomalyAttribution and ChangeAttribution require a supplied static Dag. The attribution engine has no lagged latent slice.`

## Reduced retention

These refusals are `antecedent_state::retention::RetentionMode::require`. This note does not add a second retention type.

Full reusable state is the default. It still serves every resident operation. It refuses a disk read, because that read was not declared.

Summary-only keeps reuse of the summary. It can no longer serve:

- row reconstruction
- refit from retained rows
- exact row eviction
- in-place column insert
- in-place weight or mask update
- reuse of full state

It also refuses a read of the declared disk artifact. That read is not a loss relative to full reusable state, which never served one.

Disk-backed output is `antecedent_state::SpillRegistry::spill`, which writes a prepared result (the retained rows and source identity of a refresh summary) as a digest-checked frame under one byte budget. The returned `SpillHandle` serves only a read of the declared disk artifact (`reload`). It can no longer serve the six operations above, and it can no longer serve reuse of the summary; each refusal names the reload that reaches the same end. A spill that does not fit the whole budget is refused and evicts nothing. A spill that fits evicts the least recently used results, and reloading an evicted result is refused with the rebuild from the source rows. A file that was changed or truncated is refused.

The refusal text is `{mode} cannot serve {operation}`, with the mode names `full reusable state`, `summary-only`, and `disk-backed output`.

## Break inventory and paths

Every intentional break has one explicit path, `antecedent_io::release_migration::break_path`. Each path names the error a retained value gets, and a test calls the function that enforces it.

| Break | Path | Error |
| --- | --- | --- |
| Score-row ids are eight bytes | Recompute the digest from the eight-byte row ids | `MigrationError::ScoreRowDigestNotTranslated` |
| Scientific-target digest at the current payload format | Recompute the digest from the target | `MigrationError::CarriedTargetDigestNotTranslated` |
| World identities are content-based | Recompute the digest from the target; world ids and the reference flag do not enter | `MigrationError::CarriedTargetDigestNotTranslated` |
| Reduced retention refuses later operations | Reload the rows or the disk artifact, and run the operation on the resident state | `antecedent_state::retention::RetentionRefusal` |
| A licensed route has no current method | Request the query through the method that replaced it | `RouteRecordError::RouteWithoutCurrentMethod` |
| A licensed route is carried only by unlicensed methods | License one of the methods, or accept its execution-only standing | `RouteRecordError::LicenseNotCarried` |
| Refusal reasons | None changed | none |

## What the route inventory does not list

Searches of the route manifest notes found no `changed`, `identity`, `breaking`, `retention`, or `unavailable` entry. The score-row width, the scientific-target digest, and the reduced-retention refusals are the breaks named by the public wire, the target writer, and the retention modes. They are not rows of the route manifest.
