# Algorithm provenance

Every substantive algorithm must have a machine-readable provenance record in
this directory.

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
