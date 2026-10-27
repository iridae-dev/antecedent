"""Residual-limitation generator and checker: build-free unit tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import check_residual_limitations as check  # noqa: E402
import residual_limitations as gen  # noqa: E402
from _repo import load_toml, read_text  # noqa: E402

TRACKER = "\n".join(
    [
        "- [ ] **A1** open item that carries work",
        "- [x] **B2 — closed item.** done",
        "- [ ] **Engineering Pass — architecture.** open unnumbered item",
    ]
)


def row(**fields: str) -> dict:
    base = {
        "id": "r1",
        "kind": "narrowed_promise",
        "source": "s",
        "locator": "C3.x",
        "text": "t",
        "item": "C3",
        "owner": "A1",
        "decision": "",
        "rationale": "",
    }
    return base | fields


def codes(errors: list[str]) -> list[str]:
    return [e.split(":")[0] for e in errors]


class Tracker(unittest.TestCase):
    def test_ids_titles_and_state_are_read(self) -> None:
        self.assertEqual(
            gen.parse_tracker(TRACKER), {"A1": True, "B2": False, "Engineering Pass": True}
        )


class Scanning(unittest.TestCase):
    def test_narrowed_promise_and_theory_rows_become_limitations(self) -> None:
        rows = [
            {
                "id": "P9.a.temporal.frequentist",
                "closure": "narrowed_promise",
                "status": "executing_unlicensed",
                "reason": "no interval",
                "next_attempt": "derive it",
            },
            {
                "id": "T#static.bayesian",
                "closure": "delivered",
                "status": "not_applicable_with_theory_reason",
                "reason": "no static graph",
                "next_attempt": "",
            },
            {
                "id": "ok",
                "closure": "delivered",
                "status": "licensed",
                "reason": "",
                "next_attempt": "",
            },
        ]
        found = gen.counterpart_limitations(rows)
        self.assertEqual([r["kind"] for r in found], ["narrowed_promise", "theory_reason"])
        self.assertEqual(found[0]["item"], "P9")
        self.assertIn("Next attempt: derive it", found[0]["text"])

    def test_only_unmeasured_sentences_of_a_limitations_field_are_kept(self) -> None:
        doc = {
            "cell": [
                {"id": "a", "limitations": "Fits a model. No coverage record covers the interval."},
                {"id": "b", "limitations": "No coverage record covers the interval. Other."},
                {"id": "c", "limitations": "Assumes no-unmeasured confounding."},
            ]
        }
        found = gen.evidence_limitations("parity/x.toml", doc)
        self.assertEqual([r["text"] for r in found], ["No coverage record covers the interval."])
        self.assertEqual(found[0]["locator"], "2 rows")

    def test_source_phrases_are_found_outside_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "crates/a/src").mkdir(parents=True)
            (root / "crates/a/tests").mkdir(parents=True)
            (root / "crates/a/src/lib.rs").write_text(
                "// Coverage has not been measured.\nfn f() {}\n", encoding="utf-8"
            )
            (root / "crates/a/tests/t.rs").write_text(
                "// coverage has not been measured\n", encoding="utf-8"
            )
            found = gen.phrase_limitations(root)
        self.assertEqual([r["source"] for r in found], ["crates/a/src/lib.rs"])
        self.assertEqual(found[0]["text"], "Coverage has not been measured.")

    def test_route_next_attempts_become_limitations(self) -> None:
        routes = [
            {"id": "r", "counterpart_reason": "why", "counterpart_next_attempt": "do"},
            {"id": "q"},
        ]
        found = gen.route_limitations(routes)
        self.assertEqual([r["id"] for r in found], ["route.r.route_counterpart"])


class Assignment(unittest.TestCase):
    def test_first_matching_rule_owns_and_unmatched_rows_stay_unowned(self) -> None:
        rows = [
            {
                "id": "a",
                "kind": "evidence_limit",
                "source": "s",
                "locator": "",
                "text": "t",
                "item": "",
            },
            {
                "id": "b",
                "kind": "source_phrase",
                "source": "s",
                "locator": "",
                "text": "t",
                "item": "",
            },
        ]
        rules = [
            {"name": "first", "kind": "^evidence_limit$", "owner": "A1"},
            {"name": "second", "kind": "evidence", "owner": "B2"},
        ]
        out = gen.assign(rows, rules)
        self.assertEqual((out[0]["owner"], out[0]["rule"]), ("A1", "first"))
        self.assertEqual((out[1]["owner"], out[1]["decision"]), ("", ""))

    def test_a_rule_without_patterns_matches_nothing(self) -> None:
        self.assertFalse(gen.rule_matches({"name": "x", "owner": "A1"}, row()))

    def test_rendering_round_trips_through_toml(self) -> None:
        import tomllib

        text = gen.render(gen.assign([row(text='quote " and \\ slash')], []))
        self.assertEqual(tomllib.loads(text)["limitation"][0]["text"], 'quote " and \\ slash')


class Rejections(unittest.TestCase):
    tracker = gen.parse_tracker(TRACKER)

    def test_a_mapped_row_passes(self) -> None:
        self.assertEqual(check.problems([row()], self.tracker), [])
        scope = row(owner="", decision="scope", rationale="a recorded boundary")
        self.assertEqual(check.problems([scope], self.tracker), [])

    def test_unmapped_is_rejected(self) -> None:
        self.assertEqual(codes(check.problems([row(owner="")], self.tracker)), ["unmapped"])

    def test_self_ownership_is_rejected(self) -> None:
        errors = check.problems([row(item="A1", owner="A1")], self.tracker)
        self.assertEqual(codes(errors), ["self_owned"])

    def test_a_closed_owner_is_rejected(self) -> None:
        self.assertEqual(codes(check.problems([row(owner="B2")], self.tracker)), ["closed_owner"])

    def test_an_absent_owner_is_rejected(self) -> None:
        self.assertEqual(codes(check.problems([row(owner="Z9")], self.tracker)), ["absent_owner"])

    def test_a_scope_decision_without_rationale_is_rejected(self) -> None:
        bare = row(owner="", decision="scope", rationale="  ")
        self.assertEqual(codes(check.problems([bare], self.tracker)), ["no_rationale"])

    def test_an_owner_together_with_a_decision_is_rejected(self) -> None:
        both = row(decision="scope", rationale="r")
        self.assertEqual(codes(check.problems([both], self.tracker)), ["ambiguous"])
        other = row(owner="", decision="defer", rationale="r")
        self.assertEqual(codes(check.problems([other], self.tracker)), ["ambiguous"])

    def test_scan_and_ledger_must_agree(self) -> None:
        scanned = [row(id="r1"), row(id="r2")]
        self.assertEqual(
            codes(check.problems([row(id="r1")], self.tracker, scanned)), ["missing_row"]
        )
        self.assertEqual(
            codes(check.problems([row(id="r3")], self.tracker, [row(id="r3")][:0])), ["missing_row"]
        )

    def test_a_rule_matching_nothing_is_rejected(self) -> None:
        rules = [{"name": "dead", "kind": "^nothing$", "owner": "A1"}]
        errors = check.problems([row()], self.tracker, [row()], rules)
        self.assertEqual(codes(errors), ["unused_rule"])

    def test_a_checked_off_item_with_a_residual_is_a_contradiction(self) -> None:
        notes = check.contradictions([row(item="B2")], self.tracker)
        self.assertEqual(len(notes), 1)
        self.assertEqual(check.contradictions([row(item="A1")], self.tracker), [])


class Repository(unittest.TestCase):
    def test_the_committed_ledger_is_complete_and_current(self) -> None:
        tracker_path, ledger_path = gen.ROOT / gen.TRACKER, gen.ROOT / gen.LEDGER
        tracker = gen.parse_tracker(read_text(tracker_path))
        stored = load_toml(ledger_path).get("limitation", [])
        self.assertEqual(check.problems(stored, tracker), [])


if __name__ == "__main__":
    unittest.main()
