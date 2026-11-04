"""The documentation vocabulary gate flags retired evidence terms and spares software licences."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_docs_vocabulary as gate  # noqa: E402


class Scan(unittest.TestCase):
    def hits(self, *lines):
        return gate.scan(list(enumerate(lines, 1)), "x.md")

    def test_retired_terms_are_flagged(self):
        for line in (
            "This route is licensed by a record.",
            "The result is promoted to identified.",
            "`not_measured` until a coverage record covers it.",
            "A closed route refuses.",
            "calibration refuses the cell",
            "needs a waiver",
        ):
            self.assertEqual(len(self.hits(line)), 1, line)

    def test_software_licence_and_math_intervals_are_not(self):
        self.assertEqual(self.hits("Licensed under MIT or Apache-2.0.", "## License", "A closed interval [a, b]."), [])


if __name__ == "__main__":
    unittest.main()
