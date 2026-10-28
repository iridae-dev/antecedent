"""Fault controls for the `python-internal` lane report check."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import check_python_internal_lane as lane


def case(module: str, name: str, outcome: str = "pass", reason: str = "") -> str:
    body = {
        "pass": "",
        "skip": f'<skipped type="pytest.skip" message="{reason}">{reason}</skipped>',
        "fail": '<failure message="assert 1 == 2">assert 1 == 2</failure>',
    }[outcome]
    return f'<testcase classname="tests.{module}" name="{name}">{body}</testcase>'


def report(cases: list[str]) -> str:
    return f'<testsuites><testsuite name="pytest">{"".join(cases)}</testsuite></testsuites>'


def healthy() -> list[str]:
    cases = [case(module, "test_a") for module in lane.MODULES]
    cases.append(case(lane.MODULES[0], "test_default", "skip", "normal released wheel refusal boundary"))
    return cases


class InternalLaneReport(unittest.TestCase):
    def test_a_lane_where_every_module_runs_passes(self):
        self.assertEqual(lane.junit_problems(report(healthy())), [])

    def test_a_module_that_collected_nothing_fails(self):
        cases = [c for c in healthy() if lane.MODULES[2] not in c]
        problems = lane.junit_problems(report(cases))
        self.assertEqual(len(problems), 1)
        self.assertIn(lane.MODULES[2], problems[0])
        self.assertIn("no test was collected", problems[0])

    def test_an_empty_report_fails_every_module(self):
        self.assertEqual(len(lane.junit_problems(report([]))), len(lane.MODULES))

    def test_a_module_that_skipped_every_test_for_a_missing_hook_fails(self):
        for reason in (
            "requires calibration-internal acceptance wheel",
            "internal candidate lifecycle is not a released positive",
            "private candidate hook requires calibration-internal; public measured route is available",
            "original checked lifecycle requires isolated calibration-internal wheel",
        ):
            cases = [c for c in healthy() if lane.MODULES[1] not in c]
            cases += [case(lane.MODULES[1], "test_a", "skip", reason), case(lane.MODULES[1], "test_b", "skip", reason)]
            problems = lane.junit_problems(report(cases))
            self.assertEqual(len(problems), 1, reason)
            self.assertIn("internal hook is missing", problems[0])

    def test_one_hook_skip_among_passes_still_fails(self):
        cases = healthy() + [case(lane.MODULES[3], "test_b", "skip", "requires calibration-internal acceptance wheel")]
        self.assertEqual(len(lane.junit_problems(report(cases))), 1)

    def test_a_module_whose_tests_all_skip_for_another_reason_fails(self):
        cases = [c for c in healthy() if lane.MODULES[4] not in c]
        cases.append(case(lane.MODULES[4], "test_a", "skip", "some other reason"))
        problems = lane.junit_problems(report(cases))
        self.assertEqual(len(problems), 1)
        self.assertIn("every collected test skipped", problems[0])

    def test_a_failing_test_is_left_to_pytest_not_hidden_as_a_skip(self):
        cases = [c for c in healthy() if lane.MODULES[5] not in c] + [case(lane.MODULES[5], "test_a", "fail")]
        self.assertEqual(lane.junit_problems(report(cases)), [])

    def test_an_unparseable_report_fails(self):
        self.assertEqual(len(lane.junit_problems("<testsuites>")), 1)


class InternalLaneSources(unittest.TestCase):
    def tree(self, extra: dict[str, str] | None = None, drop: str | None = None) -> Path:
        root = Path(tempfile.mkdtemp(prefix="internal-lane-"))
        tests = root / "python/tests"
        tests.mkdir(parents=True)
        for module in lane.MODULES:
            if module != drop:
                (tests / f"{module}.py").write_text(
                    'import pytest\nX = pytest.mark.skipif(True, reason="requires calibration-internal acceptance wheel")\n',
                    encoding="utf-8",
                )
        for name, text in (extra or {}).items():
            (tests / f"{name}.py").write_text(text, encoding="utf-8")
        return root

    def test_the_repository_lane_list_is_current(self):
        self.assertEqual(lane.source_problems(lane.REPO_ROOT), [])

    def test_a_listed_module_that_is_gone_fails(self):
        problems = lane.source_problems(self.tree(drop=lane.MODULES[0]))
        self.assertEqual(len(problems), 1)
        self.assertIn("does not exist", problems[0])

    def test_a_new_module_that_skips_on_the_internal_feature_must_join_the_lane(self):
        extra = {"test_new_hook": 'import pytest\n@pytest.mark.skipif(True, reason="private candidate hook requires calibration-internal")\ndef test_x(): pass\n'}
        problems = lane.source_problems(self.tree(extra))
        self.assertEqual(len(problems), 1)
        self.assertIn("test_new_hook", problems[0])

    def test_a_module_without_such_a_skip_is_not_required_in_the_lane(self):
        self.assertEqual(lane.source_problems(self.tree({"test_other": "def test_x(): pass\n"})), [])


if __name__ == "__main__":
    unittest.main()
