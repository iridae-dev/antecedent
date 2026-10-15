# Documentation sources

These Markdown sources build the Antecedent documentation site with MkDocs.

Generated pages are `support-matrix.md` and `conformance/`; regenerate them
from their source registries before committing documentation changes. Release
notes live in `release-notes/`, one `vX.Y.Z.md` per version.

Keep durable user guides in `guides/` and release-specific product changes in
`release-notes/`. The top level contains established public pages whose URLs
are already in use. Put candidate-specific run logs and measurements with their
CI run or evidence registry, rather than adding cut checklists or temporary
status pages to the published docs tree. The standing release procedure is in
`development.md`.

Publication exclusions are declared in `scripts/docs_publication.toml`. Evidence-linked audit sources stay at their existing paths so calibration references remain stable; MkDocs and `scripts/package_docs.py` exclude them from shipped outputs. Bounds tokens are resolved from `parity/promotion.toml` at publication time. Never replace them with copied constants in source guides.

Complete runnable code can be embedded with `<!-- example: examples/python/name.py -->`; the build reads the tested source rather than maintaining a second copy.

Local preview (with a matching installed Antecedent):

```bash
python -m pip install -r requirements-docs.txt
python scripts/generate_example_catalog.py --check
python scripts/check_docs_experience.py
python scripts/render_notebooks.py
mkdocs build --strict
mkdocs serve
```

The build splits generated release inventories into virtual reference appendices while retaining generator-owned source markers. Notebook HTML is built into `target/doc-notebook-previews`, never committed into source notebooks. Re-run rendering after changing a notebook or package.
