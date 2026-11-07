"""Checks of untracked Cargo configuration and compile-time overrides."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from compile_time_environment import noncanonical_compile_environment


class CompileTimeEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "parent" / "project"
        self.root.mkdir(parents=True)
        self.home = Path(self.temp.name) / "home"
        self.home.mkdir()
        self.env = {"HOME": str(self.home), "PATH": "/usr/bin"}

    def test_canonical_environment_and_tracked_project_config_are_allowed(self):
        config = self.root / ".cargo" / "config.toml"
        config.parent.mkdir()
        config.write_text('[resolver]\nincompatible-rust-versions = "fallback"\n')
        env = self.env | {"RUSTFLAGS": "-D warnings", "CARGO_PROFILE_TEST_OPT_LEVEL": "2"}
        self.assertEqual(noncanonical_compile_environment(self.root, env), [])

    def test_compiler_native_and_python_overrides_are_rejected_without_values(self):
        env = self.env | {
            "RUSTC_WRAPPER": "/secret/wrapper", "CC": "my-cc", "PYO3_PYTHON": "python",
            "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER": "linker", "SDKROOT": "sdk",
        }
        issues = noncanonical_compile_environment(self.root, env)
        for name in ("RUSTC_WRAPPER", "CC", "PYO3_PYTHON", "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER", "SDKROOT"):
            self.assertIn(f"environment override {name}", issues)
        self.assertNotIn("/secret/wrapper", str(issues))

    def test_zstd_cannot_select_ambient_pkg_config(self):
        env = self.env | {"ZSTD_SYS_USE_PKG_CONFIG": "1"}
        self.assertIn(
            "environment override ZSTD_SYS_USE_PKG_CONFIG",
            noncanonical_compile_environment(self.root, env),
        )

    def test_libc_build_overrides_are_refused(self):
        names = ("LIBC_CI", "LIBC_BUILD_VERBOSE", "RUST_LIBC_UNSTABLE_FREEBSD_VERSION")
        issues = noncanonical_compile_environment(self.root, self.env | dict.fromkeys(names, "1"))
        for name in names:
            self.assertIn(f"environment override {name}", issues)

    def test_target_prefixed_native_overrides_are_rejected_without_values(self):
        env = self.env | {
            "HOST_CC": "private compiler", "TARGET_CFLAGS": "private flags",
            "HOST_ARFLAGS": "private ar flags", "CRATE_CC_NO_DEFAULTS": "1",
            "NVCC": "private nvcc", "CXXSTDLIB": "private library",
            "HOST_UNRELATED": "allowed", "TARGET_UNRELATED": "allowed",
        }
        issues = noncanonical_compile_environment(self.root, env)
        for name in ("HOST_CC", "TARGET_CFLAGS", "HOST_ARFLAGS", "CRATE_CC_NO_DEFAULTS", "NVCC", "CXXSTDLIB"):
            self.assertIn(f"environment override {name}", issues)
        self.assertNotIn("environment override HOST_UNRELATED", issues)
        self.assertNotIn("environment override TARGET_UNRELATED", issues)
        self.assertNotIn("private", str(issues))

    def test_parent_and_cargo_home_configs_are_rejected(self):
        parent_config = self.root.parent / ".cargo" / "config.toml"
        home_config = self.home / ".cargo" / "config"
        for path in (parent_config, home_config):
            path.parent.mkdir(parents=True)
            path.write_text('')
        issues = noncanonical_compile_environment(self.root, self.env)
        self.assertIn(f"ambient Cargo config {parent_config.resolve()}", issues)
        self.assertIn(f"ambient Cargo config {home_config}", issues)

    def test_missing_or_symlinked_ambient_config_and_cargo_home_override(self):
        other = Path(self.temp.name) / "other"
        other.mkdir()
        (other / "config.toml").symlink_to(other / "missing")
        issues = noncanonical_compile_environment(self.root, self.env | {"CARGO_HOME": str(other)})
        self.assertIn("environment override CARGO_HOME", issues)
        self.assertIn(f"ambient Cargo config {other / 'config.toml'}", issues)


if __name__ == "__main__":
    unittest.main()
