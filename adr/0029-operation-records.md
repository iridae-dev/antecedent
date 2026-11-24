# ADR 0029: Operation records

- Status: Accepted
- Date: 2026-11-24
- Extends: [0027](0027-rust-verification-route.md)
- Amends: [0028](0028-one-compiled-path.md) (operation and assurance records)

## Context

Every verified operation is to be described with the same fields: inputs, output,
premises, transformation, justification, implementation, obligations and limitations.
Planned work reads these descriptions (ROADMAP.md): machine-readable obligations that
compose, derivation steps and estimation building blocks with stated premises, links
from a result to the operations behind it, and a search over typed operations. The descriptions must be attached to the operations themselves, not kept in a
second model of the library that can drift from it.

Each assurance record carried one `operation` block for its whole property. A property
covers several operations (the relay rule, the rank it compares, the function that
applies it), so the block described none of them exactly, and an operation covered by
two properties had two descriptions.

ADR 0028 said an operation record is a computed view over the layers of a lowered
program. The operations verified today are kernel functions in `antecedent-contract`,
`antecedent-core` and `antecedent-io-core` (claim standing, obligation composition,
revalidation, identities, acceptance). No lowered program contains them, and most have no
machine-readable statement of their inputs or premises anywhere.

## Decision

- One record per verified operation, in `parity/operation_records.toml`. A verified
  operation is one an assurance record lists in its `operations`; that list replaces the
  per-property `operation` block.
- A record is keyed by the operation's existing identity, its defining Rust path, the
  path the assurance records already used. `file` and `item` (a `scripts/rust_items.py`
  locator) attach it to the code, as `formal/identity_sites.toml` attaches identity
  sites. No new operation vocabulary is introduced: every field points at an existing
  item, refusal token, assurance record, test, plan route or support unit.
- Fields are generated wherever a source already holds the fact, and hand-written only
  where none does:
  - `justification`: every check of every assurance record listing the operation, with
    its kind, tool and theorem or harness names, and whether the check's digest binds the
    operation's item; and the checks of an identity site on the same item. Kinds stay
    separate; nothing is merged into one "verified" label.
  - `implementation`: the canonical public path and facade re-exports (from the public-API
    snapshots), the compiled-path disposition (`parity/compiled_path.toml`), the plan
    route with its support unit and entry (`parity/compiled_units.toml`, the program-family
    table), and the assurance records whose digests bind the item.
  - `inputs`, `output`, `transformation`: from the capability record of the operation's
    support unit when the operation is a plan route; written by hand for kernel
    functions, which have no capability record.
  - `premises`: written by hand, but each names the token (reason code, refusal variant
    or refusing call) the operation refuses or restricts with when the premise fails,
    and the checker finds that token in the item; a premise the operation cannot check
    says why (`unchecked`). Premises may name the adversarial or unit tests that remove
    them.
  - `obligations`: what the output carries to its users, each naming the identifier it is
    carried in, found in the item.
- An empty list field needs a stated reason (`empty.<field>`).
- `scripts/operation_records.py` regenerates the generated fields (`--write`) and checks
  (`check`, static gate) that every verified operation has a record, every record
  describes a verified operation, every reference resolves, an assurance check that
  binds an operation's item lists the operation, and the generated fields are current.
  `--self-test` (gate self-tests) feeds it deliberately broken records.
- For compiled operations the record stays the computed view ADR 0028 describes: once an
  operation is a plan route, its implementation and its inputs, output and transformation
  come from the route registry and its unit, and hand-written text for them is refused.

## Consequences

- An operation covered by several properties has one description, and each property's
  checks appear in its justification.
- A record never raises a result's standing and makes no calibration claim. Its
  justification says which kind of check stands behind the operation, and whether that
  check binds the production item or only a specification; an obligation is discharged
  only by a checked rule or valid evidence, never by the existence of a record or proof.
- Renaming or moving a verified function, or removing the refusal a premise names, fails
  the static gate until the record follows.
- The record does not store the item's token digest; the assurance records bind it, and
  `check` fails for a verified operation whose item no check binds (`bound_by = []`), so
  editing any verified operation stales an assurance record. An operation a proof only
  models is bound through the proof check's `model_items`: the digest binds the code, and
  the record states that the binding is to a model, not a translation (`binds = "model"`).
- The bound artifact loader forms that `acceptance.retained_artifact` covers are described
  through `load_expecting` and `bind_retained_artifact`, not one record each.
