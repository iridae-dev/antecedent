"""The predictive score scope audit: the repository agrees, and each disagreement is found."""

from __future__ import annotations

import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from _repo import REPO_ROOT, ensure_scripts_on_path

ensure_scripts_on_path()
import check_predictive_score_scope as scope  # noqa: E402

FILES = (
    scope.DESIGN_FILE,
    scope.DOC_FILE,
    scope.PYTHON_FILE,
    scope.LAW_FILE,
    scope.RECORDS_FILE,
)


def mirror(root: Path) -> None:
    for rel in FILES:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, target)
    notes = root / "docs" / "release-notes"
    notes.mkdir(parents=True, exist_ok=True)
    for note in (REPO_ROOT / "docs" / "release-notes").glob("*.md"):
        shutil.copyfile(note, notes / note.name)


class PredictiveScoreScopeTest(unittest.TestCase):
    def audit_with(self, rel: str, old: str, new: str) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mirror(root)
            path = root / rel
            text = path.read_text(encoding="utf-8")
            self.assertIn(old, text)
            path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
            with patch.object(scope, "ROOT", root):
                return scope.audit()

    def test_the_repository_agrees(self) -> None:
        self.assertEqual(scope.audit(), [])

    def test_an_unqualified_upward_in_a_release_note_is_found(self) -> None:
        findings = self.audit_with(
            "docs/release-notes/v2.4.0.md",
            "iid 400 to 1,600 cases,",
            "iid 400 cases upward,",
        )
        self.assertTrue(any(f.startswith("unqualified_upward") for f in findings), findings)

    def test_a_widened_shipped_range_is_found(self) -> None:
        records = tomllib.loads((REPO_ROOT / scope.RECORDS_FILE).read_text(encoding="utf-8"))
        low, high = scope.record_ranges(records)[0]["iid"]
        findings = self.audit_with(
            scope.DOC_FILE, f"| {low:,} to {high:,} |", f"| {low:,} to {high * 2:,} |"
        )
        self.assertTrue(any(f.startswith("shipped_range_mismatch") for f in findings), findings)

    def test_a_changed_design_range_is_found(self) -> None:
        findings = self.audit_with(
            scope.DESIGN_FILE, "Design::Cluster => 1_600", "Design::Cluster => 3_200"
        )
        self.assertTrue(any(f.startswith("design_range_mismatch") for f in findings), findings)


if __name__ == "__main__":
    unittest.main()
