"""Session-wide guard: a run that mostly skips is not a passing run.

A broken or missing `antecedent._native` makes module-level skips (or collection
errors) out of the tests that exercise it, and `pytest` can still exit 0 with "N
skipped". The handful of opt-in tests skip by design; if more than
`ANTECEDENT_MAX_SKIP_FRACTION` (default 5 %) of the collected tests skip, the
session fails.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from _repo_text import REPO_ROOT


def _scientific_nodes() -> set[tuple[str, str]]:
    """(path relative to the repository, test name) of every registered scientific assertion.

    Derived from the same registries as `scripts/evidence_audit.py scientific_rows`;
    a checkout without `scripts/` has no registry and marks nothing."""
    audit = REPO_ROOT / "scripts" / "evidence_audit.py"
    if not audit.is_file():
        return set()
    done = subprocess.run(
        [sys.executable, str(audit), "scientific-python-nodes"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        raise pytest.UsageError(
            "cannot derive scientific assertions from the registries:\n" + done.stderr[-2000:]
        )
    return {tuple(line.split("::", 1)) for line in done.stdout.splitlines() if "::" in line}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark registered scientific assertions so the default lane deselects them.

    Marking happens before `-m` deselection runs (this hook is registered first)."""
    if not items:
        return
    scientific = _scientific_nodes()
    for item in items:
        path = getattr(item, "path", None)
        if path is None:
            continue
        try:
            relative = path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            continue
        if (relative, getattr(item, "originalname", item.name)) in scientific:
            item.add_marker(pytest.mark.scientific)


@pytest.fixture
def repo_root():
    """Repository root. Tests that only need the path import REPO_ROOT."""
    return REPO_ROOT


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    collected = session.testscollected
    if reporter is None or collected < 20 or exitstatus != 0:
        return
    skipped = len(reporter.stats.get("skipped", []))
    bound = float(os.environ.get("ANTECEDENT_MAX_SKIP_FRACTION", "0.05"))
    if skipped > bound * collected:
        reporter.write_line(
            f"FAILED: {skipped} of {collected} collected tests skipped "
            f"(more than {bound:.0%}); is antecedent._native importable?",
            red=True,
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
