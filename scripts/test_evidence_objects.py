"""Engineering tests for the sharded, content-addressed evidence object store."""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import evidence_objects as eo


class ObjectStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def roundtrip(self, value):
        objects: dict[str, bytes] = {}
        encoded = eo.encode(value, objects)
        eo.write(self.root, objects)
        store = eo.Store(self.root)
        self.assertEqual(eo.decode(encoded, eo.Decoded(store.get)), value)
        return encoded, objects

    def test_short_values_stay_inline_and_long_ones_are_stored(self):
        encoded, objects = self.roundtrip({"a": 1, "b": [1, 2]})
        self.assertEqual((encoded, objects), ({"a": 1, "b": [1, 2]}, {}))
        encoded, objects = self.roundtrip({"text": "x" * 2000})
        self.assertEqual(list(encoded), ["@"])
        self.assertEqual(list(objects), [encoded["@"]])

    def test_long_lists_share_chunks_across_values(self):
        rows = [{"row": i, "pad": "p" * 40} for i in range(400)]
        _, first = self.roundtrip(rows)
        _, second = self.roundtrip(rows[:200] + [{"row": -1}] + rows[200:])
        self.assertGreater(len(set(first) & set(second)), len(first) // 2)

    def test_a_value_that_looks_like_a_reference_is_refused(self):
        with self.assertRaises(ValueError):
            eo.encode({"@": "a" * 64}, {})

    def test_shards_append_once_and_refuse_misaddressed_lines(self):
        _, objects = self.roundtrip({"text": "y" * 2000})
        self.assertEqual(eo.write(self.root, objects), [])
        (address,) = objects
        shard = self.root / eo.shard_name(address)
        shard.write_bytes(shard.read_bytes().replace(b"y", b"z", 1))
        with self.assertRaises(eo.CorruptObject):
            eo.Store(self.root).get(address)

    def test_compact_keeps_only_what_is_named(self):
        _, kept = self.roundtrip({"text": "k" * 2000})
        _, dropped = self.roundtrip({"text": "d" * 2000})
        store = eo.Store(self.root)
        self.assertEqual(store.compact(set(kept)), sorted(dropped))
        self.assertEqual(eo.Store(self.root).addresses(), set(kept))

    def test_a_write_racing_a_compaction_lands_in_the_compacted_shard(self):
        _, kept = self.roundtrip({"text": "k" * 2000})
        (address,) = kept
        prefix = address[:2]
        late = {"text": "late", "pad": 0}
        while True:
            data = eo._canonical(late)
            if eo.address_of(data).startswith(prefix):
                break
            late["pad"] += 1
        writer, compactor = eo.Store(self.root), eo.Store(self.root)
        self.assertTrue(writer.has(address))
        real_flock = eo.fcntl.flock

        def compact_first(handle, operation):
            # The writer opened the old shard; the compaction replaces it before the lock.
            eo.fcntl.flock = real_flock
            compactor.compact({address})
            real_flock(handle, operation)

        eo.fcntl.flock = compact_first
        try:
            writer.put({eo.address_of(data): data})
        finally:
            eo.fcntl.flock = real_flock
        self.assertEqual(eo.Store(self.root).addresses(), {address, eo.address_of(data)})

    def test_a_committed_shard_may_only_grow_by_valid_lines(self):
        def git(*args):
            subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)
        git("init", "-q")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "base")
        _, objects = self.roundtrip({"text": "c" * 2000})
        name = eo.shard_name(next(iter(objects)))
        git("add", name)
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "shard")
        self.assertTrue(eo.appended_only(self.root, name))
        path = self.root / name
        path.write_bytes(path.read_bytes() + b"0" * 64 + b" {}\n")
        self.assertFalse(eo.appended_only(self.root, name))


if __name__ == "__main__":
    unittest.main()
