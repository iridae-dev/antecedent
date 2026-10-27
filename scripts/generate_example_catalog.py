#!/usr/bin/env python3
"""Generate the repository/site example catalogs from examples/catalog.toml."""

import argparse
import html
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- generated:example-catalog:begin -->"
END = "<!-- generated:example-catalog:end -->"

# Topic order follows the documentation: the front door first, then the task guides.
TOPIC_ORDER = [
    "Start here",
    "Tutorials",
    "Data and inputs",
    "Estimate effects",
    "Experiments and study designs",
    "Structure and discovery",
    "Structural models and feedback",
    "Joint interventions and search",
    "Transport",
    "Decisions and study planning",
    "Evidence and composition",
    "Prepare, branch and reuse",
    "Build with Antecedent",
]


def version():
    prep = ROOT / "docs/release-notes/preparation.toml"
    if prep.exists():
        target = tomllib.loads(prep.read_text(encoding="utf-8")).get("target_version")
        if target:
            return target
    return tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))[
        "workspace"
    ]["package"]["version"]


def entries():
    return tomllib.loads((ROOT / "examples/catalog.toml").read_text(encoding="utf-8"))[
        "example"
    ]


def render(site):
    rows = entries()
    present = {row["topic"] for row in rows}
    unknown = present - set(TOPIC_ORDER)
    if unknown:
        raise SystemExit(
            f"Unknown example topics (add to TOPIC_ORDER): {sorted(unknown)}"
        )
    topics = [topic for topic in TOPIC_ORDER if topic in present]
    out = [
        BEGIN,
        "",
        "Examples run on simulated data with a declared mechanism, so the known truth can be asserted; a simulated truth is a fixture truth of that stated model, not a claim about a real system. One tutorial reads a real public dataset on request and falls back to a synthetic stand-in offline, and its output says which table it used. Runtime estimates assume a warm installed environment and exclude installation and Rust's first compilation. First native imports and notebook-kernel setup can add substantial time; machine, workload and optional dependencies affect elapsed time. Assertions check each example against its stated truth, not your causal assumptions.",
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
                        links.append(
                            f"[preview](notebook-previews/{Path(path).stem}.html)"
                        )
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
        f"# Choose an example\n\nEach example asks one question and checks one property of the answer. Start with the tutorials: they are the maintained front door, and each pairs with a page that explains the science. Install [the matching package](start/installation.md), then run an example from a repository checkout.\n\n```bash\npython examples/python/<name>.py\ncargo run -p antecedent --example <name>\n```\n\nNotebook previews show executed outputs; source downloads and Colab links are beside each notebook. See [the repository setup](https://github.com/iridae-dev/antecedent/blob/v{release}/examples/README.md#python-environment) for the local environment. For your own data, see [your first analysis](start/first-analysis.md); for a graph handoff to another library, see [DoWhy interoperability](interop_dowhy.md).\n\n"
        if site
        else f'# Examples\n\n## Python environment\n\nUse Python 3.11 or later. Released-package setup:\n\n<!-- generated:release:python-pin:begin -->\n```bash\npython -m pip install "antecedent>={release},<{next_minor}" pandas matplotlib ipykernel jupyterlab\n```\n<!-- generated:release:python-pin:end -->\n\nFor a source build, follow [installation](../docs/start/installation.md#build-from-source). Select the notebook kernel for your installed environment and restart after upgrading an imported native extension.\n\n## Run an example\n\n```bash\npython examples/python/feedback_three_questions.py\ncargo run -p antecedent --example feedback_three_questions\njupyter lab examples/notebooks\n```\n\nThe tutorials are the front door: [one feedback system and three questions](../docs/tutorials/feedback-three-questions.md), [a joint intervention program](../docs/tutorials/joint-interventions.md), [an experiment and the next study](../docs/tutorials/experiment-to-next-study.md), [deciding under uncertainty](../docs/tutorials/decide-under-uncertainty.md), [replay and current decisions](../docs/tutorials/replay-and-current-decisions.md), [an application component](../docs/tutorials/application-component.md) and [an observational study](../docs/tutorials/observational-study.md).\n\n## Datasets\n\nMost examples simulate their data from a declared mechanism so the known answer can be checked. Examples that read a real public dataset describe its origin, licence, version and checksums in [`data/`](data/README.md), and run offline on a labelled synthetic stand-in. The published site includes executed notebook previews; the source notebooks remain free of saved outputs.\n\n'
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
