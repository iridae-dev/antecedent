"""Negative evidence must not stand in for an actual detached measured replay."""

import sys
import unittest
from pathlib import Path

from transport_replay_contract import (
    replay_body_problems,
    shared_evidence_problems,
    source_replay_problems,
)

ROOT = Path(__file__).resolve().parents[1]
ASSERTION = "test_joint_bayesian_factory_replays_without_producer_inputs"
PATH = "python/tests/test_transport_route_contracts.py"


class ReplayContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = (ROOT / PATH).read_text(encoding="utf-8")

    def test_actual_detached_source_replay_is_accepted(self):
        self.assertEqual(
            replay_body_problems(self.body, ASSERTION, "joint_bayesian_transport"), []
        )

    def test_missing_or_fake_lifecycle_evidence_refuses(self):
        mutations = [
            ("del source_inputs", "del unrelated_builder"),
            ("del measured", "del unrelated_handle"),
            ("tr.joint_bayesian_transport", "tr.unlicensed_factory"),
            ("**source_inputs", "**unbound_inputs"),
            ("expected=identity", "unbound=identity"),
            ("expected=I._from_wire", "unbound=I._from_wire"),
            ("observed == original", "observed == observed"),
            ("loaded.inspect() == original", "loaded.inspect() == loaded.inspect()"),
            ('data_digest="0" * 64', "level=0.9"),
            ("loaded.source_artifact()", "payload"),
            ("subprocess.check_output", "fake.check_output"),
        ]
        for before, after in mutations:
            with self.subTest(before=before):
                self.assertIn(before, self.body)
                mutated = self.body.replace(before, after)
                self.assertTrue(
                    replay_body_problems(mutated, ASSERTION, "joint_bayesian_transport")
                )

    def test_uncalled_helper_proof_and_changed_capture_refuse(self):
        for before, after in [
            (
                "loaded = independent_replay(payload, identity, original, tmp_path)",
                "loaded = fake_replay(payload, identity, original, tmp_path)",
            ),
            ("measured.inspect()", "fake.inspect()"),
        ]:
            with self.subTest(before=before):
                mutated = self.body.replace(before, after)
                self.assertTrue(
                    replay_body_problems(mutated, ASSERTION, "joint_bayesian_transport")
                )
        # A similarly named but unused proof outside the actual helper is insufficient.
        mutated = self.body.replace("def independent_replay(", "def unused_replay(")
        self.assertTrue(
            replay_body_problems(mutated, ASSERTION, "joint_bayesian_transport")
        )

    def test_sole_shared_citation_does_not_prove_distinct_routes(self):
        first = {
            "route": "first",
            "status": "licensed",
            "evidence_test": PATH,
            "evidence_assertion": ASSERTION,
        }
        second = first | {"route": "second"}
        self.assertTrue(shared_evidence_problems([first, second]))
        self.assertEqual(
            shared_evidence_problems(
                [first, second | {"evidence_assertion": "distinct_test"}]
            ),
            [],
        )

    def test_unknown_route_wrong_stage_owner_and_provenance_refuse(self):
        route = {
            "route": "antecedent.transport.joint_bayesian",
            "stage": "uncertainty",
            "promotion_record": "2.3A.X4.joint_bayesian_transport",
            "evidence_test": PATH,
            "evidence_assertion": ASSERTION,
        }
        self.assertEqual(source_replay_problems(ROOT, route, self.body), [])
        for key, value in [
            ("route", "antecedent.transport.unlicensed"),
            ("stage", "evaluate"),
            ("promotion_record", "unowned"),
            ("evidence_test", "unregistered.py"),
        ]:
            with self.subTest(key=key):
                self.assertTrue(
                    source_replay_problems(ROOT, route | {key: value}, self.body)
                )


if __name__ == "__main__":
    sys.exit(not unittest.main(exit=False).result.wasSuccessful())
