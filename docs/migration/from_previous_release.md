# Migration from the previous release

The previous release is Antecedent 2.4. The published predecessor is 2.4.1. 2.4.0 stays the historical control. This note is the migration for callers and artifacts moving from that release onto the current contracts.

Family interchange, previous-release goldens, and fresh-process consumers are not decided here.

The durable container format is unchanged. Readers that already migrate container sources 0.1 through 0.5 still do that work. This note does not add a container rewrite.

## Route axes

Migration children are the four axes, one child each. A geometric cell is not a child.

| Axis | Standing |
| --- | --- |
| Static frequentist | Still unmigrated |
| Static Bayesian | Still unmigrated |
| Temporal frequentist | Still unmigrated |
| Temporal Bayesian | Still unmigrated |

Licensed and graphless routes are not yet on the current contracts. Closed, execution-only, and fitting-gap routes are not relabelled as migrated. Asking to migrate an axis returns an error that names the axis.

The public route manifest is `parity/capabilities/route_manifest.toml`, the inventory of the published 2.4.1 tree. Its header is:

> Public route manifest for the published 2.4.1 tree.

No route record in that file names a changed identity, a changed refusal reason, or a break. The fields on each record are target, graph class, design/time model, inference, uncertainty, decision ports, artifact consumer, status, owning module, and notes.

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

Disk-backed output is a declaration. It stores no path and spills nothing. It serves only a read of the declared disk artifact. It can no longer serve the six operations above, and it can no longer serve reuse of the summary.

The refusal text is `{mode} cannot serve {operation}`, with the mode names `full reusable state`, `summary-only`, and `disk-backed output`.

## What the route inventory does not list

Searches of the route manifest notes found no `changed`, `identity`, `breaking`, `retention`, or `unavailable` entry. The score-row width, the scientific-target digest, and the reduced-retention refusals are the breaks named by the public wire, the target writer, and the retention modes. They are not rows of the route manifest.
