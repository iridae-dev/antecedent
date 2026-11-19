"""Replicate shards and duration-estimate ordering; never runs a measurement.

The shard merge is checked on a synthetic tally that follows the harness's accumulation
(`CoverageTally::record` / `skip` / `bind`): a run over replicates `0..n` equals the merge of
disjoint shards of it followed by the extension over the tail."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import calibration_estimates as estimates  # noqa: E402
import calibration_groups as groups  # noqa: E402
import calibration_shards as shards  # noqa: E402

KEY = "calibration_test::synthetic@0.950000"


def replicate(rep: int) -> tuple[str, float, float, int]:
    """A deterministic outcome of replicate `rep` (a function of the index alone)."""
    digest = hashlib.blake2b(f"replicate {rep}".encode(), digest_size=16).digest()
    a, b, c = struct.unpack("<HHI", digest[:8])
    lo = a / 65535.0 * 3.0 - 1.5
    width = (b + 1) / 65535.0 * 1.7 + 1e-9 * (c % 97)
    kind = "skip" if c % 17 == 0 else "refused" if c % 11 == 0 else "scored"
    return kind, lo, lo + width, 80 + c % 40


def fresh() -> dict:
    return {"covered": 0, "scored": 0, "skipped": 0, "with_interval": 0, "length_sum": 0.0, "events": [],
            "lengths": [], "n_min": 2**64 - 1, "n_max": 0, "bound": 0}


def feed(tally: dict, rep: int) -> None:
    """One replicate, as `record` / `skip` / `bind` fold it."""
    kind, lo, hi, rows = replicate(rep)
    fingerprint = hashlib.blake2b(f"{kind} {lo!r} {hi!r}".encode(), digest_size=16).hexdigest()
    if kind == "skip":
        tally["skipped"] += 1
        tally["events"].append(fingerprint)
        return
    tally["bound"] += 1
    tally["n_min"], tally["n_max"] = min(tally["n_min"], rows), max(tally["n_max"], rows)
    tally["scored"] += 1
    tally["events"].append(fingerprint)
    if kind == "refused":
        return
    tally["with_interval"] += 1
    tally["length_sum"] += hi - lo
    tally["lengths"].append(hi - lo)
    if lo <= 0.0 <= hi:
        tally["covered"] += 1


def state(tally: dict, map_end: int, *, shard: bool) -> dict:
    out = {
        "key": KEY, "level": 0.95, "covered": tally["covered"], "scored": tally["scored"],
        "skipped": tally["skipped"], "with_interval": tally["with_interval"],
        "length_sum": tally["length_sum"], "map_end": map_end,
        "event_digests": "".join(tally["events"][:1000]), "prefix_digests": {},
        "decision_context": {"config": "frozen", "grid_point": 1},
        "record": {"construction": {"interval_method": "synthetic"}, "validation": "none" if tally["bound"] else None,
                   "n_min": tally["n_min"], "n_max": tally["n_max"], "replicates_min": None,
                   "posterior_draws_min": None, "unidentified_mass_max": 0.0, "bound": tally["bound"]},
    }
    if shard:
        out["lengths"] = tally["lengths"]
    return out


def single_run(start: int, end: int, prior: dict | None = None) -> dict:
    """The state a run over `start..end` writes, extending `prior` when it has one."""
    tally = fresh()
    if prior is not None:
        tally.update({k: prior[k] for k in ("covered", "scored", "skipped", "with_interval", "length_sum")})
        tally["events"] = [prior["event_digests"][i:i + 32] for i in range(0, len(prior["event_digests"]), 32)]
        record = prior["record"]
        tally.update(n_min=record["n_min"], n_max=record["n_max"], bound=record["bound"])
    for rep in range(start, end):
        feed(tally, rep)
    return state(tally, end, shard=False)


def shard_run(start: int, end: int) -> dict[str, dict]:
    tally = fresh()
    for rep in range(start, end):
        feed(tally, rep)
    return {KEY: state(tally, end, shard=True)}


def comparable(s: dict) -> dict:
    return {k: v for k, v in s.items() if k != "prefix_digests"}


class ShardMergeTest(unittest.TestCase):
    def test_merged_shards_and_tail_equal_one_run(self):
        for n, count in ((400, 8), (400, 3), (97, 5), (2000, 10)):
            ranges, (tail, end) = shards.plan_ranges(0, n, count)
            merged = shards.merge(None, [(a, b, shard_run(a, b)) for a, b in ranges])
            prior = merged[KEY]
            self.assertEqual(prior["map_end"], tail)
            extended = single_run(tail, end, prior)
            self.assertEqual(comparable(extended), comparable(single_run(0, n)), (n, count))
            # The length sum is identical to the bit, not merely close.
            self.assertEqual(struct.pack("<d", extended["length_sum"]),
                             struct.pack("<d", single_run(0, n)["length_sum"]))

    def test_partial_sums_would_not_be_identical(self):
        # Adding each shard's own sum is not the sequential sum: the merge re-folds lengths.
        differing = 0
        for n, count in ((400, 8), (400, 3), (97, 5), (2000, 10), (1000, 7), (640, 6)):
            ranges, _ = shards.plan_ranges(0, n, count)
            naive = 0.0
            for a, b in ranges:
                naive += shard_run(a, b)[KEY]["length_sum"]
            merged = shards.merge(None, [(a, b, shard_run(a, b)) for a, b in ranges])[KEY]["length_sum"]
            sequential = single_run(0, ranges[-1][1])["length_sum"]
            self.assertEqual(merged, sequential)
            differing += naive != sequential
        self.assertGreater(differing, 0)

    def test_recheck_extension_of_a_first_run_by_shards(self):
        first = single_run(0, 400)
        ranges, (tail, end) = shards.plan_ranges(400, 2000, 6)
        merged = shards.merge({KEY: first}, [(a, b, shard_run(a, b)) for a, b in ranges])[KEY]
        self.assertEqual(comparable(single_run(tail, end, merged)), comparable(single_run(0, 2000)))

    def test_fingerprint_prefix_stops_at_the_retained_limit(self):
        ranges, (tail, end) = shards.plan_ranges(0, 2000, 4)
        merged = shards.merge(None, [(a, b, shard_run(a, b)) for a, b in ranges])[KEY]
        self.assertEqual(len(merged["event_digests"]), 1000 * 32)
        self.assertEqual(merged["event_digests"], single_run(0, tail)["event_digests"])

    def test_ranges_are_disjoint_cover_the_range_and_keep_a_short_tail(self):
        for start, end, count in ((0, 400, 10), (400, 2000, 4), (0, 40, 10), (0, 41, 3)):
            ranges, (tail, stop) = shards.plan_ranges(start, end, count)
            self.assertEqual(stop, end)
            self.assertEqual(ranges[0][0], start)
            for (_, b), (a, _) in zip(ranges, ranges[1:] + [(tail, 0)], strict=False):
                self.assertEqual(b, a)
            self.assertTrue(all(b > a for a, b in ranges))
            self.assertLessEqual(end - tail, max(count, -(-(end - start) // 20)))
            self.assertGreaterEqual(end - tail, 1)

    def test_a_range_too_small_to_split_is_not_split(self):
        self.assertIsNone(shards.plan_ranges(0, 20, 10))
        self.assertIsNone(shards.plan_ranges(0, 400, 1))

    def test_mismatched_shards_are_refused(self):
        a, b = shard_run(0, 50), shard_run(50, 100)
        with self.assertRaises(shards.ShardError):
            shards.merge(None, [(0, 50, a), (60, 100, b)])  # gap
        with self.assertRaises(shards.ShardError):
            shards.merge(None, [(0, 50, a), (50, 90, b)])  # the state says it ended at 100
        with self.assertRaises(shards.ShardError):
            shards.merge(None, [(50, 100, b)])  # no prior and not from 0
        other = shard_run(50, 100)
        other[KEY]["record"]["construction"] = {"interval_method": "other"}
        with self.assertRaises(shards.ShardError):
            shards.merge(None, [(0, 50, a), (50, 100, other)])
        other = shard_run(50, 100)
        other[KEY]["decision_context"] = {"config": "another"}
        with self.assertRaises(shards.ShardError):
            shards.merge(None, [(0, 50, a), (50, 100, other)])
        with self.assertRaises(shards.ShardError):
            shards.merge({KEY: single_run(0, 40)}, [(50, 100, b)])  # the prior ends before the shard

    def test_a_tally_the_run_never_wrote_is_refused(self):
        extra = shard_run(50, 100)
        extra["calibration_test::other@0.950000"] = {**extra[KEY], "key": "calibration_test::other@0.950000"}
        with self.assertRaises(shards.ShardError):
            shards.merge({KEY: single_run(0, 50)}, [(50, 100, extra)])

    def test_merged_file_round_trips_as_a_prior(self):
        ranges, _ = shards.plan_ranges(0, 400, 5)
        merged = shards.merge(None, [(a, b, shard_run(a, b)) for a, b in ranges])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "merged.tallies.jsonl"
            shards.write_states(path, merged)
            again = shards.load_states(path)
        self.assertEqual(again[KEY], {k: v for k, v in merged[KEY].items() if k != "lengths"})
        self.assertNotIn("lengths", again[KEY])


class ShardedExtensionTest(unittest.TestCase):
    def test_run_in_ranges_then_tail_reproduces_the_single_run_tallies(self):
        """`sharded_extension` against a fake test binary that follows the harness's env contract."""
        import calibration_runner as runner

        calls = []

        def fake(args, env):
            start = int(env.get(shards.START_ENV, "0"))
            end = int(env[shards.END_ENV]) if shards.END_ENV in env else int(env["ANTECEDENT_CALIBRATION_NSIM"])
            sharding = shards.END_ENV in env
            out = Path(env[shards.TALLY_ENV])
            if sharding:
                content = shard_run(start, end)
            else:
                prior = shards.load_states(Path(env[shards.PRIOR_ENV]))[KEY] if shards.PRIOR_ENV in env else None
                content = {KEY: single_run(start, end, prior)}
            out.write_text("".join(json.dumps(s) + "\n" for s in content.values()), encoding="utf-8")
            calls.append((start, end, sharding, env["ANTECEDENT_CALIBRATION_THREADS"]))
            return 0, "test result: ok. 1 passed; 0 failed\n", {"f.rs": {(1, 2)}} if sharding else {"f.rs": {(3, 4)}}, None

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            env = {"ANTECEDENT_CALIBRATION_NSIM": "400", "ANTECEDENT_CALIBRATION_THREADS": "2"}
            result = runner.sharded_extension(fake, [], env, start=0, end=400, shards=4, prior=None,
                                              tally_out=tmp / "t.jsonl", work_dir=tmp / "shards")
            self.assertIsNotNone(result)
            status, _, regions, _, _ = result
            self.assertEqual(status, 0)
            self.assertEqual(regions, {"f.rs": {(1, 2), (3, 4)}})  # executed regions of every run
            written = shards.load_states(tmp / "t.jsonl")[KEY]
            self.assertEqual(comparable(written), comparable(single_run(0, 400)))
            plain = sorted(c for c in calls if not c[2])
            self.assertEqual(len(plain), 1)  # one asserting tail
            self.assertEqual(plain[0][3], "4")  # the tail runs on the shards' cores
            parts = sorted(c[:2] for c in calls if c[2])
            self.assertEqual(parts[0][0], 0)
            self.assertEqual(parts[-1][1], plain[0][0])  # the shards end where the tail begins
            self.assertTrue(all(c[3] == "1" for c in calls if c[2]))
            # The recheck extends the sharded first pass the way it extends a plain one.
            again = runner.sharded_extension(fake, [], {**env, "ANTECEDENT_CALIBRATION_NSIM": "2000"}, start=400,
                                             end=2000, shards=4, prior=tmp / "t.jsonl",
                                             tally_out=tmp / "r.jsonl", work_dir=tmp / "recheck")
            self.assertEqual(again[0], 0)
            self.assertEqual(comparable(shards.load_states(tmp / "r.jsonl")[KEY]), comparable(single_run(0, 2000)))

    def test_a_failed_shard_falls_back_to_the_whole_run(self):
        import calibration_runner as runner

        def failing(args, env):
            return 101, "test result: FAILED. 0 passed; 1 failed\n", None, "the test binary failed"

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.assertIsNone(runner.sharded_extension(
                failing, [], {"ANTECEDENT_CALIBRATION_NSIM": "400"}, start=0, end=400, shards=4,
                prior=None, tally_out=tmp / "t.jsonl", work_dir=tmp / "s"))


def task_group(index: int, label: str, grid: bool = False):
    return groups.Group(index=index, label=label, long=False, grid=grid)


class EstimateOrderingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.short = task_group(1, "t: short")
        self.long = task_group(2, "t: long")
        self.mid = task_group(3, "t: mid", grid=True)
        self.file = self.root / "estimates.tsv"
        self.file.write_text("# estimate\n" + "t: short\t10\nt: long\t1000\nt: mid [grid point 0]\t100\n"
                             "t: mid [grid point 1]\t200\nt: mid [grid point 2]\t300\n", encoding="utf-8")

    def tasks(self):
        return [(self.short, None), (self.long, None), *[(self.mid, p) for p in (0, 1, 2)]]

    def patched(self, **extra):
        env = {groups.ESTIMATES_ENV: str(self.file), **extra}
        return (patch.dict(os.environ, env), patch.object(groups, "TIMINGS", self.root / "none.tsv"),
                patch.object(groups, "PILOT_TIMINGS", self.root / "none-pilot.tsv"))

    def test_no_file_and_no_timings_keep_gate_order(self):
        with patch.dict(os.environ, {}, clear=False) as env:
            env.pop(groups.ESTIMATES_ENV, None)
            with patch.object(groups, "TIMINGS", self.root / "none.tsv"), \
                    patch.object(groups, "PILOT_TIMINGS", self.root / "none-pilot.tsv"):
                self.assertEqual(groups.longest_first(self.tasks()), self.tasks())

    def test_estimates_order_the_first_run_longest_first(self):
        a, b, c = self.patched()
        with a, b, c:
            order = [groups.task_label(g, p) for g, p in groups.longest_first(self.tasks())]
        self.assertEqual(order, ["t: long", "t: mid [grid point 2]", "t: mid [grid point 1]",
                                 "t: mid [grid point 0]", "t: short"])

    def test_a_recorded_timing_replaces_its_estimate(self):
        a, b, c = self.patched()
        timings = self.root / "timings.tsv"
        timings.write_text("t: short\t5000.0\t0\tsha\t1\t400\n", encoding="utf-8")
        with a, c, patch.object(groups, "TIMINGS", timings):
            order = [groups.task_label(g, p) for g, p in groups.longest_first(self.tasks())]
        self.assertEqual(order[0], "t: short")

    def test_a_named_estimate_file_that_is_missing_or_malformed_is_an_error(self):
        absent = {groups.ESTIMATES_ENV: str(self.root / "absent.tsv")}
        with patch.dict(os.environ, absent), self.assertRaises(SystemExit):
            groups.estimated_durations()
        self.file.write_text("t: short\tfast\n", encoding="utf-8")
        named = {groups.ESTIMATES_ENV: str(self.file)}
        with patch.dict(os.environ, named), self.assertRaises(SystemExit):
            groups.estimated_durations()

    def test_shard_counts_split_only_tasks_longer_than_the_share(self):
        a, b, c = self.patched()
        with a, b, c:
            counts = groups.shard_counts(self.tasks(), 4)
        # total 1610 / 4 jobs = 402.5 s share: only the 1000 s task exceeds it.
        self.assertEqual(counts, {"t: long": 3})
        with a, b, c:
            self.assertEqual(groups.shard_counts(self.tasks(), 1), {})
            self.assertEqual(groups.shard_counts([(task_group(9, "t: unknown"), None)], 4), {})

    def test_estimate_rows_follow_the_planner_labels(self):
        rows, unknown = estimates.estimate_rows(
            [{"harness": "h", "fn": "one", "secs": [5.0]}, {"harness": "h", "fn": "grid", "secs": [1, 2, 3]},
             {"harness": "h", "fn": "gone", "secs": [1]}],
            {"h: one", "h: grid"})
        self.assertEqual(rows, [("h: one", 5.0), ("h: grid [grid point 0]", 1.0),
                                ("h: grid [grid point 1]", 2.0), ("h: grid [grid point 2]", 3.0)])
        self.assertEqual(unknown, ["h: gone"])
        self.assertEqual(groups.task_label(SimpleNamespace(label="h: grid"), 1), "h: grid [grid point 1]")


if __name__ == "__main__":
    unittest.main()
