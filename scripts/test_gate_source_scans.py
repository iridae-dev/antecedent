"""Synthetic source scans only; no Cargo or external services."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parent

class SourceScans(unittest.TestCase):
    def test_posterior_consumer_contract_rejects_missing_replay_evidence(self):
        from transport_replay_contract import posterior_consumer_problems

        route = {
            "route": "antecedent.transport.advanced.consume_smoothed_dose_posterior",
            "stage": "consume",
            "claim": "none",
            "evidence_assertion": "test_posterior_artifact_is_consumed_independently_and_refuses_changes",
        }
        path = SCRIPTS.parent / "python/tests/test_smoothed_dose_transport.py"
        body = path.read_text(encoding="utf-8")
        self.assertEqual(posterior_consumer_problems(route, body), [])
        self.assertTrue(posterior_consumer_problems(route, body.replace(
            "del query, data, study, result", "del query, data, study"
        )))
        self.assertTrue(posterior_consumer_problems({**route, "claim": "calibrated"}, body))

    def test_evidence_collection_includes_calibration_markers_without_running_tests(self):
        from test_evidence import _python_collection

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "python/tests/test_example.py"
            test_file.parent.mkdir(parents=True)
            test_file.write_text("def test_example(): pass\n", encoding="utf-8")
            with patch("test_evidence.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = "tests/test_example.py::test_example\n"
                run.return_value.stderr = ""
                code, output = _python_collection(test_file, root)
            args = run.call_args.args[0]
            self.assertIn("--collect-only", args)
            self.assertEqual(args[args.index("-o") + 1], "addopts=")
            self.assertEqual(code, 0)
            self.assertIn("test_example", output)

    def test_libtest_names_follow_enabled_path_module_aliases(self):
        from test_evidence import libtest_name

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            root = src / "lib.rs"
            root.write_text("mod engine;\n", encoding="utf-8")
            (src / "engine.rs").write_text(
                '#[cfg(test)]\n#[path = "fixture.rs"]\nmod tests;\n', encoding="utf-8"
            )
            fixture = src / "fixture.rs"
            fixture.write_text("#[test]\nfn proves_it() { assert!(true); }\n", encoding="utf-8")
            self.assertEqual(libtest_name(fixture, "proves_it", root), "engine::tests::proves_it")

    def test_lint_scans_quoted_brackets_spaces_and_untracked_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "bin"
            binary.mkdir()
            fake = binary / "git"
            fake.write_text('#!/bin/sh\ncase " $* " in *" --others "*" -z "*) cat "$INDEX_DATA";; *) exit 4;; esac\n', encoding="utf-8")
            fake.chmod(0o755)
            names = ["crates/example/src/space name.rs", "crates/example/src/new.rs"]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('#![allow(clippy::float_cmp, reason = "valid ] bracket")]\n', encoding="utf-8")
            # The new untracked path lacks a reason. A commented fake allow is ignored.
            (root / names[1]).write_text('// #![allow(clippy::float_cmp)]\n#![allow(clippy::cast_sign_loss)]\n', encoding="utf-8")
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
            (root / "parity/upstream_name_allowlist.toml").write_text('names=["distinctiveupstream"]\nscan_suffixes=[".toml"]\n', encoding="utf-8")
            (root / ".secret.toml").write_text('value="distinctiveupstream"\n', encoding="utf-8")
            (root / ".github").mkdir()
            (root / ".github/workflow.toml").write_text('value="distinctiveupstream"\n', encoding="utf-8")
            binary = root / "bin"
            binary.mkdir()
            fake = binary / "git"
            fake.write_text('#!/bin/sh\nprintf ".secret.toml\\0.github/workflow.toml\\0"\n', encoding="utf-8")
            fake.chmod(0o755)
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ.get("PATH", ""))
            result = subprocess.run([sys.executable, str(SCRIPTS / "gate_upstream_names.py")], cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(".github/workflow.toml", result.stdout)
            self.assertNotIn(".secret.toml", result.stdout)
