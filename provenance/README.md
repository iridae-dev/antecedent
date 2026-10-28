# Algorithm provenance

Every substantive algorithm must have a machine-readable provenance record in
this directory.

For a worked reading of a method record and its numerical evidence, see
[How Antecedent checks its scientific methods](../docs/guides/scientific-verification.md#aipw-start-with-the-method-record).

Copy [`_template.toml`](_template.toml) to a descriptive name
(e.g. `discovery.pcmci.toml`) and fill all fields truthfully. The `feature_id`
must match the filename stem, and should match the corresponding
[`parity/`](../parity) capability id where one exists.

## Rules

- Disclose prior exposure to upstream implementations.
- Set `source_translation = false` and `copied_* = false` for clean-room work.
- Cite papers with DOI and relevant sections.
- List independent test sources (synthetic SCMs, paper examples, black-box
 comparison against pinned baselines).
- Never commit upstream GPL source, translated GPL tests, or fixtures with
 unclear redistribution status.

The sections below state what those rules mean in practice. Each one exists
because a record in this directory got it wrong.

## What `papers` may contain

`papers` is **primary literature only** — the publication that introduces the
method being implemented.

A library's own software paper is not the source of the algorithm it wraps; it
belongs in `upstream_implementations_observed`. Citing it under `papers` credits
the wrapper instead of the method, and reads as provenance while supplying none.

`sections` must name the specific algorithm, theorem, lemma, definition, or
section the implementation actually follows, established by reading the code —
not filler like `["method"]`. It is the least verifiable part of a record, and so
the easiest to get quietly wrong.

## DOIs

Verify that a DOI resolves to the title recorded next to it. Two registries are
involved, and querying the wrong one makes a valid DOI look nonexistent:

- publisher DOIs — `https://api.crossref.org/works/<doi>`
- arXiv DOIs (`10.48550/arXiv.NNNN.NNNNN`) — `https://api.datacite.org/dois/<doi>`

**A DOI that cannot be verified must be omitted, not guessed.** Omit the key
entirely; never write `doi = ""`. Many legitimate sources have no DOI at all —
pre-digital books, and AAAI/UAI/JMLR proceedings of certain years. An absent DOI
on a correctly titled paper is honest; a fabricated one turns a gap into a false
claim.

Format-checking is not verification. A well-formed DOI can resolve to an
unrelated paper, and has: records here once carried an arXiv DOI that resolved to
a paper on learning Lagrangian dynamics from images, an Econometrica DOI recorded
under the title of a different paper by overlapping authors, and a Wiley-style
DOI constructed from a book's ISBN that resolved to nothing at all. Every one
passed structural checks for as long as it existed. Resolve the DOI and compare
the returned title.

## `upstream_implementations_observed`

Every entry is a table, never a bare string:

```toml
{ project = "tigramite", exposure = "black-box comparison only" }
```

`exposure` is exactly one of `"none"`, `"previous familiarity"`,
`"black-box comparison only"`. A bare string silently drops the exposure claim —
the specific thing the top-level [`README`](../README.md) promises to record.
Project names and pinned versions come from
[`parity/baselines/`](../parity/baselines).

Do not claim exposure stronger than the evidence supports, and do not claim
black-box comparison for a module no fixture actually compares — leave the array
empty (`[]`) when there is no upstream counterpart.

`exposure = "none"` is correct when an algorithm is implemented from a
publication that a library also implements: the paper is the source, and the
library is disclosed only for completeness. See
[`kernels.special_functions.toml`](kernels.special_functions.toml), where the
cited coefficient tables are the artifact and the library that shares them was
not consulted.

## `test_sources`

Name real paths and verify they exist. Prefer fixture directories under
[`conformance/`](../conformance) and crate test files; fixtures carrying a
`reference` block are the authoritative record of what was compared against
what. Do not write test-source prose when a path exists.

Re-check paths when repairing an old record. The conformance tree has been
reorganized since some records were written, and several pointed at directories
that no longer existed — claiming comparison evidence that could not be located.

## When there is no primary paper

Some modules genuinely have no primary: dispatch and normalization layers,
folklore diagnostics, textbook material, original engineering. For those,
`papers = []` is correct and honest.

It must be an explicit judgment, not an empty field. State the reasoning in a
comment in the record — what the module does, why no paper introduces it, and
which related record carries the citation if one does. Without that, an empty
list is indistinguishable from an unfilled one.

Resist the adjacent temptation: a paper that is merely topically nearby is not a
source. See
[`state.rolling_mechanism_diag.toml`](state.rolling_mechanism_diag.toml), which
declines to cite Page's 1954 CUSUM work because the module computes a
fixed-window max-|partial sum| scan, not a sequential control-chart procedure.

### Recording the judgment: `[non_algorithm_review]`

A comment is not checked. A record with `papers = []` that was read against its modules and found
to be a wrapper, contract, derivation or closed variant carries a `[non_algorithm_review]` table
that the gate reads (`scripts/provenance_links.py::non_algorithm_review_problems`):

```toml
[non_algorithm_review]
status = "justified_non_algorithm"
kind = "wrapper"                      # wrapper | contract | derivation | closed_variant
rationale = "what the module does"
no_independent_algorithm = "why nothing numerical of its own remains"
covered_by = ["record.stem"]          # records that carry the numerics (required for wrapper)
derivation = "..."                    # required for derivation: the identity relied on
inline_tests = true                   # only when the mapped file holds its own #[cfg(test)] assertions
record_sha256 = "..."

[[non_algorithm_review.artifacts]]
role = "implementation"               # the module files that were read
path = "crates/.../module.rs"
sha256 = "..."
[[non_algorithm_review.artifacts]]
role = "assertion"                    # required for derivation and closed_variant
path = "crates/.../tests/module.rs"
sha256 = "..."
```

- `wrapper` delegates every numerical step to the `covered_by` records. `contract` is a declaration,
  binding, lineage or refusal check over supplied identities and numbers. `derivation` is closed-form
  or enumerative arithmetic on a textbook identity that the table states. `closed_variant` implements
  nothing and refuses; its assertion must execute the refusal.
- The gate fails a mapping with an unknown kind, a missing rationale, a `covered_by` that names a
  missing record or the record itself, a missing mapped module, a stale module or record digest, a
  record that lists papers, a record that also carries `method_review`, or a derivation or closed
  variant with no executing assertion.
- Before mapping, read the module: a file that contains its own square roots, logarithms,
  exponentials, solves or sampling is a method and needs `papers` and a `[method_review]`, or a
  `candidate: ..., not verified` entry in `parity/provenance_review_triage.toml` while the source is
  unconfirmed. Do not map such a record to quiet it.
- A mapping states that no independent source is owed. It is not an implementation review of a
  primary source and the evidence report counts it separately from `method_review`.
- After editing the record or a mapped file, run
  `scripts/bind_review_digests.py mapping provenance/<id>.toml`, then
  `scripts/provenance_review_triage.py --write`.

## Records that name the modules they cover

A record may carry `modules = [...]`, the implementation files its method is. `scripts/provenance_links.py` links a method to its records by that field, by the module path or file name in the record, by a feature id, or by the explicit table of modules an existing record documents without naming. A module that executes or orchestrates other methods and implements no algorithm of its own is listed with its reason instead.

## Method review

`parity/method_specs/specs.toml` records, for each admitted method, the provenance records that cover its module and whether its public surface is in the reviewed API snapshot (`parity/public_api/`). `scripts/gate_provenance_schema.py` fails an admitted method that links no record and gives no reason, a linked record that does not state all four clean-room flags false, a record whose `modules` names a missing file, and a method whose public items are in no reviewed snapshot. The admission state is the one the specifications derive from the executing evidence, so a method cannot be advertised as licensed or executing without a provenance record and a reviewed surface.
