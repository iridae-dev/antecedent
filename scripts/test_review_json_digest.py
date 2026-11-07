"""Compatibility vectors for review JSON digest framing and field exclusions."""

import unittest

from evidence_kinds import fixture_payload_digest
from provenance_links import method_review_digest, non_algorithm_digest
from review_items import review_json_digest


class ReviewJsonDigestTests(unittest.TestCase):
    def test_utf8_sorted_compact_payload_and_review_exclusions(self):
        self.assertEqual(
            review_json_digest({"z": "é", "a": [1, None, True]}),
            "5ea9634a57acb0a0b95d1b56fec10fedfdc667a387cecdccfa88b414b7baff77",
        )
        self.assertEqual(
            method_review_digest({"id": "é", "nested": {"v": 1}, "method_review": {}}),
            "e923cebb13c147c589d61a7b02c70efe44aac875ed21ba43ed31592a633ea9f5",
        )
        self.assertEqual(
            method_review_digest({"id": "é", "method_review": {}, "non_algorithm_review": {"a": 1}}),
            "e7657424d93621c1319c0fb34ce769d54912b243adec364f638d6ae0af272983",
        )
        self.assertEqual(
            review_json_digest({"float": float("nan")}),
            "cf0ac9ae2b485f45760e53cebaeec65d82879df5b9b787853c769a2ed3c7362e",
        )
        self.assertEqual(
            non_algorithm_digest({"id": "é", "method_review": {}, "non_algorithm_review": {}}),
            "76be78880f84a68f495ae2953e3c560e99b5b456064e0b75cd939c3c2f05fb95",
        )
        self.assertEqual(
            fixture_payload_digest({"oracle": {"value": "é", "reference_review": {}}}),
            "531ad0a1719d9c0e0d2cfc27bf484fb50c3ba4230b865f32df419de0ab30f456",
        )


if __name__ == "__main__":
    unittest.main()
