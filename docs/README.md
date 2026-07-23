# Documentation

| Doc | Contents |
|-----|----------|
| [Architecture](architecture.md) | Invariants, crates, analysis pipeline, execution model |
| [Development](development.md) | CI vs local gates, tests, performance rules, versions |
| [Artifacts](artifacts.md) | Wire format, migration, graph interchange |
| [Prior bank](prior_bank.md) | External prior catalog, compose, conflict, transport |
| [API naming](api_naming.md) | Rust ↔ Python capability dictionary |
| [Hot paths](hot_paths.md) | Benches, baselines, allocation contracts |
| [Conformance](conformance/README.md) | Generated from `conformance/` fixtures |
| [Security review](security_review.md) | Unsafe, deps, licensing evidence |

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
