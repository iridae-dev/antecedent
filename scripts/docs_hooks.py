"""MkDocs publication transforms; never rewrite scientific evidence sources."""

import hashlib
import json
import re
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOUND = re.compile(r"\{\{bound:([\w.]+)\}\}")


def publication_exclusions():
    return tomllib.loads((ROOT / "scripts/docs_publication.toml").read_text(encoding="utf-8"))["excluded"]


@lru_cache(maxsize=1)
def bound_values():
    records = tomllib.loads((ROOT / "parity/promotion.toml").read_text(encoding="utf-8"))["record"]
    return {
        f"{record['id']}.{key}": value
        for record in records
        for key, value in record.get("bounds", {}).items()
    }


def render_bounds(markdown):
    values = bound_values()

    def substitute(match):
        key = match.group(1)
        if key not in values:
            raise ValueError(f"Unknown documentation bound: {key}")
        return str(values[key])

    rendered = BOUND.sub(substitute, markdown)
    if "{{bound:" in rendered:
        raise ValueError("Unresolved documentation bound")
    return rendered


def on_config(config):
    bound_values.cache_clear()
    config.extra["documentation_version"] = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))[
        "workspace"
    ]["package"]["version"]
    from pathspec import GitIgnoreSpec

    config.exclude_docs = GitIgnoreSpec.from_lines(f"/{path}" for path in publication_exclusions())
    return config


EXAMPLE = re.compile(r"<!-- example: ([\w./-]+) -->")


def render_examples(markdown):
    def substitute(match):
        path = (ROOT / match.group(1)).resolve()
        if not path.is_relative_to(ROOT / "examples") or not path.is_file():
            raise ValueError(f"Unknown documentation example: {match.group(1)}")
        return "```python\n" + path.read_text(encoding="utf-8").rstrip() + "\n```"

    return EXAMPLE.sub(substitute, markdown)


def on_page_markdown(markdown, page, config, files):
    return render_bounds(render_examples(markdown))


def split_release(markdown, source_uri):
    if not source_uri.startswith("release-notes/") or "## Claims by cell" not in markdown:
        return markdown, None
    highlights, inventory = markdown.split("## Claims by cell", 1)
    name = Path(source_uri).stem
    appendix_uri = f"reference/release-claims-{name}.md"
    # Inventory originally sits beside the release page; the virtual reference
    # page is at the same depth, preserving its ../ relative links.
    appendix = f"# {name} claim and support inventory\n\n" + "## Claims by cell" + inventory
    highlights += f"## Complete claim inventory\n\n[Per-cell claims, explicit refusals and support inventory](../{appendix_uri}) are generated from the release evidence. Read those exact scopes before using a new route.\n"
    return highlights, (appendix_uri, appendix)


def on_files(files, config):
    from mkdocs.structure.files import File, InclusionLevel

    for file in list(files):
        if file.src_uri.startswith("release-notes/") and file.src_uri.endswith(".md"):
            text = file.content_string
            highlights, appendix = split_release(text, file.src_uri)
            if appendix:
                file.content_string = highlights
                files.append(
                    File.generated(
                        config,
                        appendix[0],
                        content=appendix[1],
                        inclusion=InclusionLevel.NOT_IN_NAV,
                    )
                )
    files.append(
        File.generated(
            config,
            "assets/support-matrix.json",
            content=(ROOT / "crates/antecedent/data/support_matrix.json").read_bytes(),
            inclusion=InclusionLevel.NOT_IN_NAV,
        )
    )
    preview_dir = ROOT / "target/doc-notebook-previews"
    notebooks = sorted((ROOT / "examples/notebooks").glob("*.ipynb"))
    record_path = preview_dir / "execution.json"
    if not record_path.is_file():
        raise ValueError(
            "Notebook execution record missing: run python scripts/render_notebooks.py first"
        )
    records = {record["notebook"]: record for record in json.loads(record_path.read_text(encoding="utf-8"))}
    release = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["workspace"]["package"]["version"]
    for notebook in notebooks:
        record = records.get(notebook.name, {})
        if (
            record.get("source_sha256") != hashlib.sha256(notebook.read_bytes()).hexdigest()
            or record.get("antecedent_version") != release
            or record.get("outputs", 0) < 1
        ):
            raise ValueError(
                f"Stale or unexecuted notebook preview: {notebook.name}; run python scripts/render_notebooks.py"
            )
        preview = preview_dir / (notebook.stem + ".html")
        if not preview.is_file():
            raise ValueError(
                "Notebook previews missing: run python scripts/render_notebooks.py first"
            )
        files.append(
            File.generated(
                config,
                f"notebook-previews/{preview.name}",
                abs_src_path=str(preview),
                inclusion=InclusionLevel.NOT_IN_NAV,
            )
        )
    return files
