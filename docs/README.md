# Documentation

Start with the [documentation home](index.md).

- [Python quickstart](python-workflow.md): install, estimate, and read an answer.
- [Rust quickstart](rust-quickstart.md): run an example and build the API reference.
- [Examples](examples.md): choose a workflow by the question you want to answer.
- [Supported analyses](supported-analyses.md): check support and understand refusals.
- [Python workflow reference](python-options.md): configure or integrate an analysis.
- [1.10 release notes](release-notes/v1.10.0.md): changes in this version.

## How docs are published

| Surface | Host | Builder |
|---------|------|---------|
| Narrative (`docs/`) | [Read the Docs](https://antecedent.readthedocs.io/) | MkDocs — `mkdocs.yml`, `.readthedocs.yaml` |
| Python API | [RTD `/python/`](https://antecedent.readthedocs.io/en/latest/python/antecedent.html) | source checkout + `pdoc` in RTD `post_build` |
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
