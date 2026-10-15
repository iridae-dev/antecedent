#!/usr/bin/env python3
"""Publication-boundary regressions using actual generated docs and previews."""

import copy
import json
import subprocess
import sys
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from docs_hooks import ROOT, on_files, publication_exclusions
from generate_example_catalog import version


class PublicationTests(unittest.TestCase):
    def config(self):
        return SimpleNamespace(
            site_dir="/tmp/antecedent-docs-publication-test",
            use_directory_urls=True,
            plugins=SimpleNamespace(_current_plugin=None),
        )

    def test_current_executed_previews_are_admitted(self):
        from mkdocs.structure.files import Files

        files = on_files(Files([]), self.config())
        expected = {
            f"notebook-previews/{p.stem}.html"
            for p in (ROOT / "examples/notebooks").glob("*.ipynb")
        }
        self.assertTrue(expected.issubset({file.src_uri for file in files}))

    def test_stale_version_source_or_unexecuted_output_is_refused(self):
        from mkdocs.structure.files import Files

        records = json.loads((ROOT / "target/doc-notebook-previews/execution.json").read_text(encoding="utf-8"))
        for key, wrong in [
            ("source_sha256", "changed-source"),
            ("antecedent_version", "0.0.0"),
            ("outputs", 0),
        ]:
            with self.subTest(field=key):
                changed = copy.deepcopy(records)
                changed[0][key] = wrong
                with (
                    patch("docs_hooks.json.loads", return_value=changed),
                    self.assertRaisesRegex(ValueError, "Stale or unexecuted"),
                ):
                    on_files(Files([]), self.config())

    def test_release_archive_excludes_interim_sources_and_resolves_bounds(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "markdown.tar.gz"
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts/package_docs.py"),
                    str(target),
                ],
                check=True,
                capture_output=True,
            )
            with tarfile.open(target, encoding="utf-8") as archive:
                names = set(archive.getnames())
                for excluded in publication_exclusions():
                    self.assertNotIn("docs/" + excluded, names)
                for member in archive:
                    self.assertNotIn(b"{{bound:", archive.extractfile(member).read(), member.name)
                self.assertEqual(
                    archive.extractfile("README.md").read(),
                    (ROOT / "README.md").read_bytes(),
                )
                release = version()
                notes = archive.extractfile(f"docs/release-notes/v{release}.md").read()
                inventory = archive.extractfile(
                    f"docs/reference/release-claims-v{release}.md"
                ).read()
                self.assertNotIn(b"generated:release-claims:cells", notes)
                self.assertIn(b"generated:release-claims:cells", inventory)
                self.assertIn("docs/assets/retained-executions.svg", names)
                self.assertIn("docs/assets/measurement-decision.svg", names)


if __name__ == "__main__":
    unittest.main()
