#!/usr/bin/env python3
"""Generate the repository/site example catalogs from examples/catalog.toml."""

import argparse
import html
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- generated:example-catalog:begin -->"
END = "<!-- generated:example-catalog:end -->"


def version():
    prep = ROOT / "docs/release-notes/preparation.toml"
    if prep.exists():
        target = tomllib.loads(prep.read_text(encoding="utf-8")).get("target_version")
        if target:
            return target
    return tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))["workspace"]["package"]["version"]


def entries():
    return tomllib.loads((ROOT / "examples/catalog.toml").read_text(encoding="utf-8"))["example"]


def render(site):
    rows = entries()
    topics = list(dict.fromkeys(row["topic"] for row in rows))
    topics.sort(key=lambda topic: (topic != "Start and reuse", topic == "Performance", topic))
    out = [
        BEGIN,
        "",
        "All datasets are simulated. Runtime estimates assume a warm installed environment and exclude installation and Rust's first compilation. First native imports and notebook-kernel setup can add substantial time; machine, workload and optional dependencies affect elapsed time. Assertions check these examples, not your causal assumptions.",
        "",
    ]
    for topic in topics:
        out += [
            "## " + topic,
            "",
            "| Example and question | Level / estimated runtime | Dependencies beyond Antecedent | Run / open |",
            "| --- | --- | --- | --- |",
        ]
        for row in rows:
            if row["topic"] != topic:
                continue
            links = []
            for kind in ("python", "rust", "notebook"):
                if kind not in row:
                    continue
                path = row[kind]
                ref = "main" if row.get("unreleased") else f"v{version()}"
                target = (
                    f"https://github.com/iridae-dev/antecedent/blob/{ref}/{path}"
                    if site
                    else path.removeprefix("examples/")
                )
                links.append(f"[{kind}]({target})")
                if kind == "notebook":
                    links.append(
                        f"[Colab](https://colab.research.google.com/github/iridae-dev/antecedent/blob/{ref}/{path})"
                    )
                    if site:
                        links.append(f"[preview](notebook-previews/{Path(path).stem}.html)")
            deps = ", ".join(row["dependencies"]) or "none"
            out.append(
                f"| **{row['title']}** — {row['summary']} <details><summary>Claim scope</summary>{html.escape(row['claim'])}</details> | {row['level']} / {row['runtime']} | {deps} | {' · '.join(links)} |"
            )
        out.append("")
    out += [
        "For exact claim settings, inspect each result and check the support matrix. Optional handoff dependencies are needed to exercise their external half; an absent dependency prints a skip.",
        "",
        END,
    ]
    return "\n".join(out) + "\n"


def expected_page(site):
    release = version()
    major, minor, _ = map(int, release.split("."))
    next_minor = f"{major}.{minor + 1}"
    prefix = (
        f"# Choose an example\n\nInstall [the matching package](installation.md), then start with [your first Python analysis](python-workflow.md) or [the Rust quickstart](rust-quickstart.md). For the complete workflow, follow [notebook to system](guides/notebook-to-system.md).\n\n```bash\npython examples/python/<name>.py\ncargo run -p antecedent --example <name>\n```\n\nNotebook previews show executed outputs; source downloads and Colab links are beside each notebook. See [the repository setup](https://github.com/iridae-dev/antecedent/blob/v{release}/examples/README.md#python-environment) for the local environment. See [DoWhy interoperability](interop_dowhy.md) for the reviewed graph handoff.\n\n"
        if site
        else f'# Examples\n\n## Python environment\n\nUse Python 3.11 or later. Released-package setup:\n\n<!-- generated:release:python-pin:begin -->\n```bash\npython -m pip install "antecedent>={release},<{next_minor}" pandas matplotlib ipykernel jupyterlab\n```\n<!-- generated:release:python-pin:end -->\n\nFor a source build, follow [installation](../docs/installation.md#build-from-source). Select the notebook kernel for your installed environment and restart after upgrading an imported native extension.\n\n```bash\npython examples/python/analysis_workflow.py\npython examples/python/notebook_to_system.py\njupyter lab examples/notebooks\ncargo run -p antecedent --example ate_quickstart\n```\n\nFollow [the notebook-to-system tutorial](../docs/guides/notebook-to-system.md), [bring your own data](../docs/guides/your-data.md), or [measurement to decision](../docs/tutorials/measurement-to-decision.md). The published site includes executed notebook previews; the source notebooks remain free of saved outputs.\n\n'
    )
    return prefix + render(site)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    errors = []
    for site, path in [
        (False, ROOT / "examples/README.md"),
        (True, ROOT / "docs/examples.md"),
    ]:
        expected = expected_page(site)
        if args.check:
            if path.read_text(encoding="utf-8") != expected:
                errors.append(str(path.relative_to(ROOT)))
        else:
            path.write_text(expected, encoding="utf-8")
    if errors:
        raise SystemExit("Stale example catalogs: " + ", ".join(errors))
    print("Example catalogs current")


if __name__ == "__main__":
    main()
