"""The CI-run acceptance rule behind the release candidate's CI attestation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import ci_workflow as ci  # noqa: E402

HEAD = "a" * 40
WORKFLOW = """
jobs:
  rust:
    name: Rust ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-14]
  gates:
    name: Domain gates
"""


class CiAttestationTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workflow = Path(directory.name) / "ci.yml"
        self.workflow.write_text(WORKFLOW, encoding="utf-8", newline="\n")
        self.required = ["rust", "gates"]

    def green(self) -> dict:
        names = ["Rust ubuntu-latest", "Rust macos-14", "Domain gates"]
        return {"headSha": HEAD, "jobs": [{"name": n, "conclusion": "success"} for n in names]}

    def problems(self, payload: dict, head: str | None = HEAD) -> list[str]:
        return ci.check_run_payload(payload, head, self.required, self.workflow)

    def test_a_run_is_accepted_only_when_it_built_this_commit_with_every_required_job_green(
        self,
    ) -> None:
        self.assertEqual(self.problems(self.green()), [])
        wrong_sha = {**self.green(), "headSha": "0" * 40}
        self.assertTrue(any("headSha" in p for p in self.problems(wrong_sha)))
        missing_leg = self.green()
        missing_leg["jobs"] = [j for j in missing_leg["jobs"] if j["name"] != "Rust macos-14"]
        self.assertTrue(
            any("Rust macos-14" in p and "missing" in p for p in self.problems(missing_leg))
        )
        failed = self.green()
        failed["jobs"][2]["conclusion"] = "failure"
        self.assertTrue(any("Domain gates" in p and "failure" in p for p in self.problems(failed)))
        by_id = {
            "headSha": HEAD,
            "jobs": [{"name": n, "conclusion": "success"} for n in self.required],
        }
        self.assertTrue(
            self.problems(by_id), "workflow ids are not the display names a run reports"
        )
        self.assertEqual(
            self.problems(wrong_sha, head=None),
            [],
            "a payload without a pinned SHA checks jobs only",
        )


if __name__ == "__main__":
    unittest.main()
