"""The shared instrumented runner: scheduling, durations and coverage merging."""
import tempfile
import unittest
from pathlib import Path

import evidence_capture as ec


class SchedulingTests(unittest.TestCase):
    def test_longest_first_keeps_untimed_work_in_order_after_timed_work(self):
        seconds = {"b": 5.0, "d": 50.0}
        self.assertEqual(ec.longest_first(list("abcd"), seconds.get), ["d", "b", "a", "c"])

    def test_the_last_recorded_duration_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t" / "timings.tsv"
            ec.record_duration(path, "x", 1.0)
            ec.record_duration(path, "x", 9.0)
            path.open("a").write("malformed\n")
            self.assertEqual(ec.last_durations(path), {"x": 9.0})

    def test_coverage_of_several_runs_is_their_union(self):
        merged = ec.merge_coverage({"a": {(1, 1)}}, None, {"a": {(2, 1)}, "b": {(3, 3)}})
        self.assertEqual(merged, {"a": {(1, 1), (2, 1)}, "b": {(3, 3)}})


if __name__ == "__main__":
    unittest.main()
