"""Synthetic source scans only; no Cargo or external services."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parent

class SourceScans(unittest.TestCase):
    def test_lint_scans_quoted_brackets_spaces_and_untracked_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "bin"
            binary.mkdir()
            fake = binary / "git"
            fake.write_text('#!/bin/sh\ncase " $* " in *" --others "*" -z "*) cat "$INDEX_DATA";; *) exit 4;; esac\n')
            fake.chmod(0o755)
            names = ["crates/example/src/space name.rs", "crates/example/src/new.rs"]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('#![allow(clippy::float_cmp, reason = "valid ] bracket")]\n')
            # The new untracked path lacks a reason. A commented fake allow is ignored.
            (root / names[1]).write_text('// #![allow(clippy::float_cmp)]\n#![allow(clippy::cast_sign_loss)]\n')
            data = root / "git-index"
            data.write_bytes(b"\0".join(name.encode() for name in names) + b"\0")
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ.get("PATH", ""), INDEX_DATA=str(data))
            result = subprocess.run([sys.executable, str(SCRIPTS / "gate_lint_allows.py")], cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("space name.rs", result.stdout)
            self.assertIn("new.rs", result.stdout)
            self.assertIn("cast_sign_loss", result.stdout)
            self.assertNotIn("float_cmp carries no", result.stdout)

    def test_root_hidden_entries_are_excluded_but_github_is_scanned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "parity").mkdir()
            (root / "parity/upstream_name_allowlist.toml").write_text('names=["distinctiveupstream"]\nscan_suffixes=[".toml"]\n')
            (root / ".secret.toml").write_text('value="distinctiveupstream"\n')
            (root / ".github").mkdir()
            (root / ".github/workflow.toml").write_text('value="distinctiveupstream"\n')
            binary = root / "bin"
            binary.mkdir()
            fake = binary / "git"
            fake.write_text('#!/bin/sh\nprintf ".secret.toml\\0.github/workflow.toml\\0"\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ.get("PATH", ""))
            result = subprocess.run([sys.executable, str(SCRIPTS / "gate_upstream_names.py")], cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(".github/workflow.toml", result.stdout)
            self.assertNotIn(".secret.toml", result.stdout)
