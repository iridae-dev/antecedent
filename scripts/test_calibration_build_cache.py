"""Same-version worktrees cannot borrow binaries compiled from different source bytes."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from _calibration_build_cache import guarded_release_cache, source_inputs


class SharedCargoSourceContextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.target = self.base / "target"
        self.output = self.target / "cached-binary"

    def checkout(self, name: str, answer: str) -> Path:
        root = self.base / name
        source = root / "crates/example/src/lib.rs"
        source.parent.mkdir(parents=True)
        source.write_text(answer, encoding="utf-8")
        (root / "Cargo.toml").write_text(
            '[workspace]\nmembers=["crates/example"]\n', encoding="utf-8"
        )
        return root

    def timestamp_build(self, root: Path) -> str:
        """Independent simplified Cargo rule: old source mtime reuses an existing output."""
        source = root / "crates/example/src/lib.rs"
        if not self.output.exists() or source.stat().st_mtime_ns > self.output.stat().st_mtime_ns:
            self.output.write_bytes(source.read_bytes())
        return self.output.read_text(encoding="utf-8")

    def test_older_different_checkout_cannot_reuse_a_wrong_source_binary(self) -> None:
        original = self.checkout("original", "answer=1")
        later = self.checkout("later", "answer=2")
        # Both checkouts already existed before the original was compiled.
        with guarded_release_cache(original, self.target):
            self.assertEqual(self.timestamp_build(original), "answer=1")
        self.assertEqual(self.timestamp_build(later), "answer=1")  # Demonstrates the bug.
        bytes_before = (later / "crates/example/src/lib.rs").read_bytes()
        with guarded_release_cache(later, self.target):
            self.assertEqual(self.timestamp_build(later), "answer=2")
        self.assertEqual((later / "crates/example/src/lib.rs").read_bytes(), bytes_before)

    def test_unchanged_inputs_and_python_only_changes_reuse_the_cache(self) -> None:
        root = self.checkout("first", "answer=1")
        with guarded_release_cache(root, self.target):
            self.timestamp_build(root)
        original_mtime = self.output.stat().st_mtime_ns
        (root / "scripts").mkdir()
        (root / "scripts/tool.py").write_text("# orchestration only\n", encoding="utf-8")
        with guarded_release_cache(root, self.target) as receipt:
            self.assertEqual(receipt["invalidated_inputs"], [])
            self.timestamp_build(root)
        self.assertEqual(self.output.stat().st_mtime_ns, original_mtime)

    def test_changed_old_mtime_asset_is_invalidated_without_touching_other_sources(self) -> None:
        root = self.checkout("first", "answer=1")
        asset = root / "crates/example/lookup.json"
        asset.write_text("[1]", encoding="utf-8")
        with guarded_release_cache(root, self.target):
            self.timestamp_build(root)
        source = root / "crates/example/src/lib.rs"
        previous_mtime = source.stat().st_mtime_ns
        asset.write_text("[2]", encoding="utf-8")
        os.utime(asset, ns=(1, 1))
        with guarded_release_cache(root, self.target) as receipt:
            self.assertEqual(receipt["invalidated_inputs"], ["crates/example/lookup.json"])
            self.assertGreater(asset.stat().st_mtime_ns, 1)
        self.assertEqual(source.stat().st_mtime_ns, previous_mtime)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "answer=1")

    def native_inputs(self, root: Path) -> list[Path]:
        paths = []
        for name in ("Cargo.toml", "build.rs", "pyproject.toml", "src/lib.rs", "src/data.json"):
            path = root / "python" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original", encoding="utf-8")
            paths.append(path)
        return paths

    def test_native_source_with_an_old_mtime_cannot_reuse_a_stale_extension(self) -> None:
        root = self.checkout("native", "answer=1")
        native = self.native_inputs(root)[3]
        output = self.target / "cached-extension"

        def native_build() -> str:
            if not output.exists() or native.stat().st_mtime_ns > output.stat().st_mtime_ns:
                output.write_bytes(native.read_bytes())
            return output.read_text(encoding="utf-8")

        with guarded_release_cache(root, self.target):
            self.assertEqual(native_build(), "original")
        native.write_text("changed native implementation", encoding="utf-8")
        os.utime(native, ns=(1, 1))
        self.assertEqual(native_build(), "original")
        with guarded_release_cache(root, self.target) as receipt:
            self.assertEqual(receipt["invalidated_inputs"], ["python/src/lib.rs"])
            self.assertEqual(native_build(), "changed native implementation")

    def test_extending_a_context_adds_only_native_inputs_and_ignores_python_outputs(self) -> None:
        root = self.checkout("native", "answer=1")
        native = self.native_inputs(root)
        before_native = {
            name: digest
            for name, digest in source_inputs(root).items()
            if not name.startswith("python/")
        }
        with (
            patch("_calibration_build_cache.source_inputs", return_value=before_native),
            guarded_release_cache(root, self.target),
        ):
            self.timestamp_build(root)
        source = root / "crates/example/src/lib.rs"
        crate_mtime = source.stat().st_mtime_ns
        package = root / "python/antecedent"
        package.mkdir()
        for name in ("feature.py", "_native.so", "__pycache__/feature.pyc"):
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not a native Rust build input")
        with guarded_release_cache(root, self.target) as receipt:
            self.assertEqual(
                set(receipt["invalidated_inputs"]),
                {path.relative_to(root).as_posix() for path in native},
            )
        self.assertEqual(source.stat().st_mtime_ns, crate_mtime)
        with guarded_release_cache(root, self.target) as receipt:
            self.assertEqual(receipt["invalidated_inputs"], [])

    def test_competing_checkout_prebuild_waits_for_the_previous_context(self) -> None:
        first = self.checkout("first", "answer=1")
        second = self.checkout("second", "answer=2")
        waiting = threading.Event()
        entered = threading.Event()

        def contender() -> None:
            waiting.set()
            with guarded_release_cache(second, self.target):
                entered.set()
                self.timestamp_build(second)

        with guarded_release_cache(first, self.target):
            self.timestamp_build(first)
            thread = threading.Thread(target=contender)
            thread.start()
            self.assertTrue(waiting.wait(1))
            self.assertFalse(entered.wait(0.05))
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(entered.is_set())
        self.assertEqual(self.output.read_text(encoding="utf-8"), "answer=2")

    def test_interrupted_prebuild_releases_lock_and_keeps_sources_invalidated(self) -> None:
        root = self.checkout("first", "answer=1")

        def failed_prebuild() -> None:
            with guarded_release_cache(root, self.target):
                raise RuntimeError("failed")

        self.assertRaisesRegex(RuntimeError, "failed", failed_prebuild)
        with guarded_release_cache(root, self.target) as receipt:
            self.assertEqual(receipt["invalidated_inputs"], [])
            self.assertEqual(self.timestamp_build(root), "answer=1")


if __name__ == "__main__":
    unittest.main()
