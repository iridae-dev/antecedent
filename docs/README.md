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
