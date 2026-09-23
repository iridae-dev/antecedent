"""Session-wide guard: a run that mostly skips is not a passing run.

A broken or missing `antecedent._native` makes module-level skips (or collection
errors) out of the tests that exercise it, and `pytest` can still exit 0 with "N
skipped". The handful of opt-in tests skip by design; if more than
`ANTECEDENT_MAX_SKIP_FRACTION` (default 5 %) of the collected tests skip, the
session fails.
"""

from __future__ import annotations

import os

import pytest


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
