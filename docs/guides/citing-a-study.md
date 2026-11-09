# Cite the methods behind a study

When you write up a study, cite the methods that produced each number and say where the implementation departs from the published method. Every executed Antecedent result carries that information. It names the provenance records behind the route it ran: one for the identification strategy and one for the estimator. Capability-specific routes have their own records, read with `antecedent.citations.capability_provenance(capability_id)`. Each record gives the published sources, the documented deviations from the published method, what its review did not cover, and the record id and digest.

This page shows how to read that information from a result, how to build one de-duplicated bibliography for a whole study, and what to put in your methods section.

## Read a result's provenance

```python
import antecedent as ant
from antecedent.estimators import Aipw

outcome = ant.analyze(data, graph=graph, query=ant.AverageEffect("t", "y"), estimator=Aipw())

for record in outcome.method_provenance.records:
    print(record.id, record.cited_for, record.digest[:12])
    for source in record.sources:
        print("  source:", source.title, source.year, source.doi)
    for text in record.deviations:
        print("  deviation:", text)
    for text in record.scope_limits:
        print("  review scope:", text)
```

For a backdoor AIPW analysis, the result names two records:

| Record | Cited for | What it says |
| --- | --- | --- |
| `identify.backdoor` | identifier `backdoor.adjustment` | The adjustment-set identification and its sources. |
| `estimate.aipw` | estimator `aipw` | Chernozhukov et al. (2018) and Funk et al. (2011), with two deviations: no cross-fitting on the ATT/ATC, trimmed and predicate-target paths, so the DML rate conditions are not inherited there; and nuisance models fixed to a GLM propensity and OLS outcomes rather than arbitrary learners. |

`outcome.method_provenance` is the provenance of the route the result *executed*: the identifier and estimator the plan resolved, not only the ones you declared. In Rust the same data is `StudyResult::method_provenance()`. It is distinct from the result's `provenance` field, which records the execution's own provenance graph.

## Build a study's bibliography

A study usually has several results. `result.citations()` lists one result's sources. `antecedent.citations.bibliography(...)` collects the sources of every result and lists each source once:

```python
from antecedent.citations import bibliography

for citation in bibliography([primary, sensitivity, subgroup]):
    print(citation.reference)
    for caveat in citation.caveats:
        print(f"  - {caveat.text} ({caveat.record})")
```

Sources are the same when their DOIs agree. Sources without a DOI are matched by title, ignoring case and punctuation. A merged entry keeps every record that cites the source, with the record's digest, and every caveat those records document. `citation.reference` is a one-line reference in the form `Authors (year). Title. https://doi.org/…`. In Rust, use `antecedent::bibliography(results)` or `CompletedBatch::bibliography()`.

## What to write

Cite each source in the bibliography. For each caveat, say which part of the published method was not implemented as published. For example:

> Average effects were estimated with augmented inverse-probability weighting (Chernozhukov et al., 2018; Funk et al., 2011) as implemented in Antecedent (provenance record `estimate.aipw`, digest `f8819b29…`). Nuisance models were a logistic propensity and per-arm linear outcome regressions. The ATT was fit without cross-fitting, so the DML convergence guarantees are not inherited for that estimand.

Report the record id and digest. The digest is the SHA-256 that a record's review binds. It changes when the record's sources or deviations change, so a reader can tell which version of the record your study used.

## What the provenance contains, and what it does not

- **Sources** are the primary publications the record cites. Authors and year come from the DOI registry and are retained in `provenance/citation_metadata.json`. `doi_verified` is true when the DOI resolved to the recorded title. A source without a verified DOI has no DOI, rather than a guessed one.
- **Deviations** are the record's `implementation_deviations`, in the record's own words.
- **Scope limits** state what the record's bounded method review did not cover. They describe the review, not a refusal: an execution outside that scope still runs if it is supported.
- **Wrapper records** delegate their numerics to other records. Those records are cited too, with the role `covered_by`.
- **Unlinked components** are identifier or estimator ids whose strategy-table pointer names no provenance record. They are listed in `unlinked` rather than given a guessed citation.

Provenance is descriptive. It changes no number, support status or [calibration annotation](../concepts/uncertainty.md). It does not replace the [support status](../support-matrix.md) or the [independent numerical checks](scientific-verification.md) behind a supported route.

## Exported artifacts

An exported `analysis_result` artifact with a contract carries its provenance in an `analysis_result.method_provenance` section, so a result's citations travel with it. Whoever loads the artifact reads them the same way as on a live result:

```python
loaded = ant.load(artifact, expected=artifact.expectation)
for citation in loaded.citations():
    print(citation.reference)
```

`loaded.method_provenance` holds the records, and `bibliography(...)` accepts loaded results beside live ones. In Rust, `replay_analysis_result` returns the section as `method_provenance`, and `decode_method_provenance_section` decodes it from a decoded artifact. Every reader checks that the section names the identifier and estimator the contract declares and ignores it otherwise, so a loaded result without a bound section has empty provenance. The section records what the producer's table cited at export; it is additive, so older readers ignore it, and it takes no part in acceptance or replay.

## Where the table comes from

`scripts/generate_method_provenance.py` builds the table from the existing links: the provenance pointer on each identifier and estimator row in `crates/antecedent/src/strategy_table/ids.rs` (resolved through `scripts/provenance_links.py`), the admitted methods in `parity/method_specs/specs.toml`, the capability records in `parity/capability_records.toml`, and the records in `provenance/`. It writes `crates/antecedent-io-core/data/method_provenance.json`. `scripts/gate_provenance_schema.sh` fails when that file is stale. After editing a provenance record, regenerate it:

```bash
python3 scripts/generate_method_provenance.py
```

`python3 scripts/verify_provenance_citations.py --fetch` retains authors and year for a newly cited DOI; `--bibliographic` adds them to cached entries that lack them. Both are network operations.
