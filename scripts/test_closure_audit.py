"""Closure audit: unit tests."""

from __future__ import annotations

import unittest

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import closure_audit as audit  # noqa: E402

CHILD = {"id": "FAM.method.static.bayesian", "gap": "none"}


def row(
    identifier: str, status: str, closure: str, reason: str = "", next_attempt: str = ""
) -> dict:
    return {
        "id": identifier,
        "method": "M",
        "design": "static",
        "inference": "bayesian",
        "route": "r",
        "status": status,
        "closure": closure,
        "reason": reason,
        "next_attempt": next_attempt,
    }


def record(identifier: str, status: str, missing: list[str]) -> dict:
    return {
        "id": identifier,
        "list": "mandatory",
        "mandatory": True,
        "status": status,
        "missing": missing,
    }


class ClosureAudit(unittest.TestCase):
    def test_every_unresolved_mandatory_row_is_listed_with_its_missing_element(self) -> None:
        records = [
            record("OPEN.static.bayesian", "open", ["no consumer is named"]),
            record("SHUT.static.frequentist", "reconciled", []),
        ]
        report = audit.audit([CHILD], {}, [], records)
        self.assertEqual(
            report["unresolved_mandatory"],
            [
                {
                    "id": "OPEN.static.bayesian",
                    "list": "mandatory",
                    "missing": ["no consumer is named"],
                }
            ],
        )
        self.assertTrue(
            any("OPEN.static.bayesian" in line for line in audit.release_findings(report))
        )

    def test_a_counterpart_without_a_verified_attempt_is_reported(self) -> None:
        rows = [
            row(
                "M#static.bayesian",
                "closed_with_next_attempt",
                "narrowed_promise",
                "no law",
                "derive the law",
            )
        ]
        report = audit.audit([CHILD], {}, rows, [])
        self.assertEqual(report["missing_attempt"][0]["next_attempt"], "derive the law")
        self.assertEqual(report["missing_attempt"][0]["side"], "static.bayesian")
        executing = audit.audit([CHILD], {}, [row("M#a", "executing_unlicensed", "delivered")], [])
        self.assertEqual(executing["missing_attempt"], [])

    def test_a_missing_consumer_is_reported_per_child(self) -> None:
        report = audit.audit([CHILD], {CHILD["id"]: ["no artifact consumer is named"]}, [], [])
        self.assertEqual(report["missing_consumer"][0]["id"], CHILD["id"])
        self.assertEqual(report["families"]["FAM"]["unresolved_evidence"], 1)
        self.assertEqual(audit.audit([CHILD], {CHILD["id"]: []}, [], [])["missing_consumer"], [])

    def test_a_narrowed_promise_is_listed_with_its_adjusted_claim(self) -> None:
        rows = [row("M#a", "licensed", "narrowed_promise", "point only", "measure the interval")]
        report = audit.audit([CHILD], {}, rows, [])
        self.assertEqual(report["adjusted_claim"][0]["reason"], "point only")
        self.assertEqual(report["adjusted_claim"][0]["next_attempt"], "measure the interval")
        bare = audit.audit([CHILD], {}, [row("M#b", "licensed", "narrowed_promise")], [])
        self.assertTrue(any("without a reason" in line for line in audit.release_findings(bare)))

    def test_the_report_is_deterministic_and_names_the_audited_attempts(self) -> None:
        records = [
            record("ATT.static.bayesian", "reconciled", []),
            record("OTHER.linear.static.bayesian", "reconciled", []),
        ]
        report = audit.audit([CHILD], {}, [], records, {"ATT.static.bayesian"})
        self.assertEqual([a["id"] for a in report["audited_attempt"]], ["ATT.static.bayesian"])
        scope = [
            {"id": "ATT.first", "source": "attempt"},
            {"id": "ATT.static.bayesian", "source": "method_children"},
            {"id": "OTHER.static.bayesian", "source": "method_children"},
            {"id": "FLAG.static.bayesian", "source": "method_children", "obligation": "flagship"},
        ]
        self.assertEqual(
            audit.audited_ids(scope), {"ATT.first", "ATT.static.bayesian", "FLAG.static.bayesian"}
        )
        self.assertEqual(audit.render(report), audit.render(report))


if __name__ == "__main__":
    unittest.main()
