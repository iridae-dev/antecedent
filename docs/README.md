# Documentation

Antecedent’s narrative docs. The identification-first engine covers contrasts and
continuous causal responses; observation, transport, and interference stay
explicit stage contracts. Package version **0.6.1**.

| Doc | Contents |
|-----|----------|
| [Causal responses](causal-responses.md) | Curves, derivatives, support, uncertainty, observation mechanisms |
| [Transport and interference](transport-interference.md) | Selection diagrams, trial generalization, assignment designs, exposure mappings |
| [Capabilities](capabilities.md) | Full inventory: graphs, discovery, identification, estimation, validation, design |
| [Support matrix](support-matrix.md) | Licensed / n/a / refused cells (generated) |
| [Comparison](comparison.md) | Antecedent vs. DoWhy, EconML, Tigramite, causal-learn — and when to use each |
| [Architecture](architecture.md) | Invariants, crates, analysis pipeline, execution model |
| [Development](development.md) | CI vs local gates, tests, performance rules, versions |
| [Artifacts](artifacts.md) | Wire format, migration, graph interchange (including response format 0.3) |
| [Prior bank](priors.md) | External prior catalog, compose, conflict, transport |
| [API naming](api_naming.md) | Rust ↔ Python capability dictionary |
| [Hot paths](hot_paths.md) | Benches, baselines, allocation contracts |
| [Conformance](conformance/README.md) | Generated from `conformance/` fixtures |
| [Security review](security_review.md) | Unsafe, deps, licensing evidence |
| [0.6.1 release notes](release-notes/v0.6.1.md) | Patch: envelope/ingest correctness and hot paths |
| [0.6.0 release notes](release-notes/v0.6.0.md) | Contract cut; matrix is the license |
| [0.5.2 release notes](release-notes/v0.5.2.md) | Performance pass and localized correctness |
| [0.5.1 release notes](release-notes/v0.5.1.md) | Honesty gates and row diagnostics |
| [0.5.0 release notes](release-notes/v0.5.0.md) | Causal-response release |
| [Roadmap](../ROADMAP.md) | Post-0.5 path to 1.0 and after |

Decisions: [adr/](../adr/README.md).

## How docs are published

| Surface | Host | Builder |
|---------|------|---------|
| Narrative (`docs/`) | [Read the Docs](https://antecedent.readthedocs.io/) | MkDocs — `mkdocs.yml`, `.readthedocs.yaml` |
| Python API | [RTD `/python/`](https://antecedent.readthedocs.io/en/latest/python/antecedent.html) | `pip install antecedent` + `pdoc` in RTD `post_build` |
| Rust API | [docs.rs/antecedent](https://docs.rs/antecedent) | `cargo doc` on crates.io publish |

Release `docs.tar.gz` still bundles markdown + rustdoc + pdoc for offline use; the
live Python API is on Read the Docs, not behind a download.

Local narrative preview:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Regenerate conformance docs:

```bash
python3 scripts/generate_conformance_docs.py
```

Python stubs live next to the package (`python/antecedent/*.pyi`).
Rust ↔ Python names: [api_naming.md](api_naming.md).
