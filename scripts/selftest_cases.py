#!/usr/bin/env python3
"""Broken-input self-tests for the ledger and docs gates.

    python3 scripts/selftest_cases.py docs     # gate_docs_support_matrix.sh --self-test
    python3 scripts/selftest_cases.py schema   # gate_parity_schema.sh --self-test
    python3 scripts/selftest_cases.py citations  # gate_coverage_citations.sh --self-test

Each case builds a disposable overlay of the repo (every tracked file hard
linked, `python/` linked so the built extension is reused), copies the files it
breaks into it, runs the unchanged gate script from the overlay, and requires
the gate to fail with the case's expected message. A positive control (the overlay with no mutation) must pass, so a gate
that fails on everything cannot satisfy its self-test. The working tree is
never modified.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["package"]["version"]

Mutation = Callable[[str], str]


def _repo_files() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return [f for f in listed.split("\0") if f and (ROOT / f).is_file()]


def build_overlay(tmp: Path) -> Path:
    """Hard-link (or copy) every tracked and unignored file into a fresh tree."""
    ov = tmp / "repo"
    ov.mkdir()
    for rel in _repo_files():
        if rel.startswith("python/"):
            continue
        dest = ov / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.hardlink_to(ROOT / rel)
        except OSError:
            shutil.copy2(ROOT / rel, dest)
    # `python/` stays a link to the checkout: a gate's `uv run` there resolves
    # to the real project directory (POSIX cwd is canonical) and reuses the
    # built extension instead of rebuilding it once per case.
    (ov / "python").symlink_to(ROOT / "python")
    return ov


def materialize(ov: Path, rel: str) -> Path:
    """Make `rel` an independent, writable copy inside the overlay."""
    if rel.startswith("python/"):
        raise SystemExit(f"self-test cannot mutate {rel}: python/ is shared with the checkout")
    target = ov / rel
    if target.exists() or target.is_symlink():
        target.unlink()  # never write through a hard link into the checkout
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / rel, target)
    return target


def run_gate(ov: Path, script: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["bash", str(ov / "scripts" / script)], cwd=ov, capture_output=True, text=True
    )
    return proc.returncode, proc.stdout + proc.stderr


def case(
    script: str,
    name: str,
    mutations: dict[str, Mutation],
    expected: list[str],
    *,
    must_fail: bool = True,
) -> bool:
    """Run `script` on an overlay with `mutations`; check the verdict and messages."""
    with tempfile.TemporaryDirectory() as tmp:
        ov = build_overlay(Path(tmp))
        for rel, mutate in mutations.items():
            path = materialize(ov, rel)
            before = path.read_text()
            after = mutate(before)
            if after == before:
                print(f"SELF-TEST FAIL: {name}: mutation of {rel} changed nothing")
                return False
            path.write_text(after)
        code, out = run_gate(ov, script)
    if not must_fail:
        if code != 0:
            print(f"SELF-TEST FAIL: {name}: expected {script} to pass:\n{out}")
            return False
        print(f"self-test ok: {name}: passes")
        return True
    if code == 0:
        print(f"SELF-TEST FAIL: {name}: broken input passed {script}")
        return False
    missing = [e for e in expected if e not in out]
    if missing:
        print(f"SELF-TEST FAIL: {name}: failed without {missing}:\n{out}")
        return False
    print(f"self-test ok: {name}: fails with {expected}")
    return True


# ---------------------------------------------------------------- mutations


def append(line: str) -> Mutation:
    return lambda text: text.rstrip("\n") + "\n\n" + line + "\n"


def replace(old: str, new: str) -> Mutation:
    def apply(text: str) -> str:
        if old not in text:
            raise SystemExit(f"self-test fixture drift: {old!r} not found")
        return text.replace(old, new, 1)

    return apply


def drop_calibration_pair() -> Mutation:
    """Remove a row's `calibration` / `calibration_reason` line, whichever the
    collector last wrote: the row's state follows the coverage registry."""

    def apply(text: str) -> str:
        kept = re.sub(r"(?m)^calibration(?:_reason)? = .*\n", "", text)
        if kept == text:
            raise SystemExit("self-test fixture drift: no calibration pair in the block")
        return kept

    return apply


def drop_block(header: str, row_id: str) -> Mutation:
    """Remove the `[[header]]` block whose first key is `id = row_id`."""

    def apply(text: str) -> str:
        pattern = re.compile(
            rf"\[\[{re.escape(header)}\]\]\nid = \"{re.escape(row_id)}\"\n.*?(?=\n\[\[|\Z)",
            re.S,
        )
        out, n = pattern.subn("", text, count=1)
        if n != 1:
            raise SystemExit(f"self-test fixture drift: [[{header}]] {row_id} not found")
        return out

    return apply


def in_block(header: str, row_id: str, fn: Mutation) -> Mutation:
    """Apply `fn` to the text of one `[[header]]` block only."""

    def apply(text: str) -> str:
        pattern = re.compile(
            rf"\[\[{re.escape(header)}\]\]\nid = \"{re.escape(row_id)}\"\n.*?(?=\n\[\[|\Z)",
            re.S,
        )
        m = pattern.search(text)
        if not m:
            raise SystemExit(f"self-test fixture drift: [[{header}]] {row_id} not found")
        return text[: m.start()] + fn(m.group(0)) + text[m.end() :]

    return apply


def changelog_current(line: str) -> Mutation:
    def apply(text: str) -> str:
        head = f"## [{VERSION}]"
        i = text.index(head)
        j = text.index("\n", i)
        return text[: j + 1] + "\n" + line + "\n" + text[j + 1 :]

    return apply


# -------------------------------------------------------------------- cases


def docs_cases() -> list[bool]:
    g = "gate_docs_support_matrix.sh"
    return [
        case(g, "control", {}, [], must_fail=False),
        case(
            g,
            "walkthrough_deferral",
            {
                "docs/v1.10-practitioner-walkthrough.md": append(
                    "Panel promotions stay a separately gated workstream."
                )
            },
            ["forbidden deferral_prose 'separately gated'"],
        ),
        case(
            g,
            "adr_follow_on",
            {
                "adr/0022-causal-compiler-contract.md": append(
                    "Consuming gates are follow-on work on this decision."
                )
            },
            ["forbidden deferral_prose 'follow-on'"],
        ),
        case(
            g,
            "changelog_current_section",
            {"CHANGELOG.md": changelog_current("- Panel class response is deferred.")},
            ["CHANGELOG.md: forbidden deferral_prose 'deferred'"],
        ),
        case(
            g,
            "readme_version_deferral",
            {"README.md": append("Weighted exports land in 1.11.")},
            ["README.md: forbidden"],
        ),
        case(
            g,
            "claim_sentence_edited",
            {"docs/python-workflow.md": replace("retains a reusable study", "retains a study")},
            ["docs/python-workflow.md: claim reusable_headline occurs 0 times"],
        ),
        case(
            g,
            "later_release_deferral",
            {"docs/capabilities.md": append("Panel ATE is planned for the next release.")},
            ["forbidden deferral_prose 'next release'"],
        ),
        # Stating a limitation is not deferring work.
        case(
            g,
            "known_limitations_allowed",
            {
                "docs/capabilities.md": append(
                    "## Known limitations\n\nThe known limitation of the ADMG path is that "
                    "Frequentist distributions are Rust-only."
                )
            },
            [],
            must_fail=False,
        ),
        # Earlier CHANGELOG sections are frozen history and are not rescanned.
        case(
            g,
            "changelog_history_frozen",
            {"CHANGELOG.md": append("## [0.0.1]\n\n- Frequentist DBN mixing was deferred.")},
            [],
            must_fail=False,
        ),
    ]


NEW_ESTIMATOR_ROW = """
[[capabilities]]
id = "estimate.selftest_estimator"
group = "estimation"
description = "A new estimator row that states no calibration"
owner = "estimate"
status = "pending"
"""


def gridded_record(grid: list[tuple[int, float, bool]], boundary: bool) -> str:
    """A coverage record measured on the sample-size grid, as the collector writes
    one: `grid` holds `(n, observed, boundary)` per point, smallest first."""
    suite = "crates/antecedent-estimate/src/calibration_coverage.rs"
    points = "".join(
        f"  {{ point = {k}, n_min = {n}, n_max = {n}, observed = {obs}, mcse = 0.011, "
        f'replicates = 400, boundary = {str(b).lower()}, role = "gated" }},\n'
        for k, (n, obs, b) in enumerate(grid)
    )
    governing = min([p for p in grid if p[2]] or grid, key=lambda p: p[1])
    rid = "cov.average_effect.dag.frequentist.analytic_se.l95"
    return f"""[[record]]
id = "{rid}.linear_adjustment_analytic_ci_coverage.selftest"
query = "AverageEffect"
graph_class = "Dag"
structure = "fixed"
modality = "tabular"
inference = "Frequentist"
estimator = "linear.adjustment.ate"
interval_method = "analytic_se"
se_kind = "homoskedastic"
dependence = "iid"
posterior = ""
functional = "all_observed.mean"
identification = "point"
nominal = 0.95
n_min = {min(n for n, _, _ in grid)}
n_max = {max(n for n, _, _ in grid)}
replicates_min = 0
posterior_draws_min = 0
unidentified_mass_max = 0.0
observed = {governing[1]}
mcse = 0.011
replicates = 400
boundary = {str(boundary).lower()}
role = "gated"
grid = [
{points}]
dgp = "{suite}::confounded_scm"
test = "{suite}::linear_adjustment_analytic_ci_coverage"
facets = ["core", "suite.calibration_coverage"]
calibration_sha = "{"1" * 40}"
"""


def schema_cases() -> list[bool]:
    g = "gate_parity_schema.sh"
    return [
        case(g, "control", {}, [], must_fail=False),
        case(
            g,
            "required_row_deleted",
            {"parity/compiler.toml": drop_block("capabilities", "compiler.e2e_licensed_cells")},
            ["required composition row compiler.e2e_licensed_cells missing"],
        ),
        case(
            g,
            "estimator_row_without_calibration",
            {
                "parity/estimate.toml": in_block(
                    "capabilities",
                    "estimate.linear_regression",
                    drop_calibration_pair(),
                )
            },
            ["estimate.linear_regression: exactly one of calibration / calibration_reason"],
        ),
        case(
            g,
            "new_estimation_group_row_is_obligated",
            {"parity/estimate.toml": append(NEW_ESTIMATOR_ROW.strip())},
            ["estimate.selftest_estimator: exactly one of calibration / calibration_reason"],
        ),
        case(
            g,
            "calibration_on_non_estimation_row",
            {
                "parity/estimate.toml": in_block(
                    "capabilities",
                    "estimate.refute.placebo",
                    replace(
                        'group = "refutation_and_sensitivity"\n',
                        'group = "refutation_and_sensitivity"\n'
                        'calibration_reason = "estimator_grid_not_measured"\n',
                    ),
                )
            },
            ['estimate.refute.placebo: calibration fields only on group = "estimation" rows'],
        ),
        case(
            g,
            "no_interval_reported_out_of_scope",
            {
                "parity/reason_codes.toml": replace(
                    'queries = ["Counterfactual", "AnomalyAttribution", "ChangeAttribution"]',
                    'queries = ["Counterfactual", "AnomalyAttribution"]',
                )
            },
            ["no_interval_reported on a ChangeAttribution cell"],
        ),
        case(
            g,
            "required_job_not_a_job",
            {
                "parity/release.toml": replace(
                    'required_jobs = ["rust", "gates", "python-lint", "python-wheels"]',
                    'required_jobs = ["rust", "gates", "python-lint", "python-wheels", "pull_request"]',
                )
            },
            ["required job 'pull_request' missing from ci.yml"],
        ),
        case(
            g,
            "required_job_removed_from_ci",
            {".github/workflows/ci.yml": replace("  python-lint:\n", "  python-lint-renamed:\n")},
            ["required job 'python-lint' missing from ci.yml"],
        ),
        case(
            g,
            "publish_without_attestation",
            {
                ".github/workflows/publish-crates.yml": replace(
                    "        run: bash scripts/gate_calibration_attestation.sh\n",
                    "        run: echo skipped\n",
                )
            },
            ["publish-crates.yml: no step runs gate_calibration_attestation.sh"],
        ),
        case(
            g,
            "wheels_built_before_attestation",
            {".github/workflows/publish-release.yml": replace("    needs: prepare\n", "")},
            ["publish-release.yml: job 'wheels' does not depend on the attestation job 'prepare'"],
        ),
        case(
            g,
            "attestation_on_shallow_clone",
            {
                ".github/workflows/publish-release.yml": replace(
                    "          fetch-depth: 0\n", "          fetch-depth: 1\n"
                )
            },
            ["publish-release.yml: job 'prepare' checks out without fetch-depth: 0"],
        ),
        case(
            g,
            "reason_code_list_stale",
            {
                "parity/reason_codes.toml": append(
                    '[[code]]\nid = "selftest_unlisted"\napplies_to = ["runtime_refusal"]\n'
                    'meaning = "Not in the generated list."\nmax_uses = 0'
                )
            },
            ["reason_codes_data.rs is stale against"],
        ),
        case(
            g,
            "record_facets_narrowed_by_hand",
            {
                "parity/coverage_records.toml": replace(
                    'facets = ["core", "mechanism", "suite.v19_static_calibration"]',
                    'facets = ["core", "suite.v19_static_calibration"]',
                )
            },
            ["are not the derived", "collect_coverage_records.py --retag"],
        ),
        case(
            g,
            "record_grid_missing_a_point",
            {
                "parity/coverage_records.toml": append(
                    gridded_record([(150, 0.95, False), (300, 0.9475, False)], False)
                )
            },
            ["grid must hold one measurement per sample-size grid point"],
        ),
        case(
            g,
            "record_grid_failing_point_averaged_into_a_pass",
            {
                "parity/coverage_records.toml": append(
                    gridded_record(
                        [(150, 0.9, True), (300, 0.9475, False), (600, 0.95, False)], False
                    )
                )
            },
            ["boundary must be true exactly when a grid point is"],
        ),
        case(
            g,
            "record_grid_that_does_not_scale_its_sample_size",
            {
                "parity/coverage_records.toml": append(
                    gridded_record(
                        [(300, 0.95, False), (300, 0.9475, False), (600, 0.95, False)], False
                    )
                )
            },
            ["grid sample sizes must strictly increase point to point"],
        ),
        case(
            g,
            "record_without_facets",
            {
                "parity/coverage_records.toml": replace(
                    'facets = ["core", "suite.v19_temporal_frequentist"]\n', ""
                )
            },
            ["missing facets"],
        ),
        case(
            g,
            "raw_reason_literal",
            {
                # Before any `#[cfg(test)]`: production code, not a test fixture.
                "crates/antecedent/src/analysis/prepared.rs": replace(
                    "\nuse ", '\nconst _BYPASS: &str = "reason=bogus: typed by hand";\nuse '
                )
            },
            ['raw "reason=..." literal'],
        ),
    ]


COND_DAG_BAYES = (
    "0.892 (record `cov.conditional_effect.dag.bayesian.posterior_quantile.l90."
    "conditional_effect_dag_bayesian_nominal_90_coverage`)"
)


def citation_cases() -> list[bool]:
    g = "gate_coverage_citations.sh"
    lic = "parity/support_licensed.toml"
    return [
        case(g, "control", {}, [], must_fail=False),
        case(
            g,
            "coverage_figure_unattributed",
            {lic: replace(COND_DAG_BAYES, "0.892")},
            ["coverage figure 0.892 has no record citation"],
        ),
        case(
            g,
            "coverage_figure_disagrees_with_record",
            {lic: replace(COND_DAG_BAYES, COND_DAG_BAYES.replace("0.892", "0.931"))},
            ["coverage figure 0.931 does not match its cited record(s) (0.8925)"],
        ),
        case(
            g,
            "unknown_record_cited",
            {lic: replace(COND_DAG_BAYES, COND_DAG_BAYES.replace("l90.", "l80."))},
            ["cites unknown coverage record"],
        ),
        case(
            g,
            "attribution_for_a_later_clause_does_not_cover",
            {
                lic: replace(
                    COND_DAG_BAYES, "0.892; the probe measured 0.180 (not a registry value)"
                )
            },
            ["coverage figure 0.892 has no record citation"],
        ),
        # Known-truth values, SEs, locations and disclosed probe figures are never rejected.
        case(
            g,
            "known_truth_and_disclosures_allowed",
            {
                lic: replace(
                    COND_DAG_BAYES,
                    COND_DAG_BAYES
                    + "; P(Y=1|do(T=1)) = 0.625, mean SE 0.0416 vs Monte Carlo SD 0.0410, "
                    "total 0.356; a skewed-treatment probe covered 0.180 (not a registry value)",
                )
            },
            [],
            must_fail=False,
        ),
    ]


def main(argv: list[str]) -> int:
    suites = {"docs": docs_cases, "schema": schema_cases, "citations": citation_cases}
    if len(argv) != 1 or argv[0] not in suites:
        print(f"usage: {sys.argv[0]} {{{'|'.join(suites)}}}")
        return 2
    results = suites[argv[0]]()
    if not all(results):
        print(f"{argv[0]} self-test: {results.count(False)} case(s) failed")
        return 1
    print(f"{argv[0]} self-test: ok ({len(results)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
