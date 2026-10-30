"""The read-only coverage gate enforces the existing shrink-only backlog."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import coverage_citations as gate


class CoverageCitationRatchet(unittest.TestCase):
    def test_new_uncited_fraction_fails_and_existing_backlog_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            licensed, records, backlog = (root / name for name in ("licensed.toml", "records.toml", "backlog.txt"))
            licensed.write_text('[[cell]]\nquery="q"\ngraph_class="g"\nstructure="s"\ninference="i"\nvalidation="v"\nlimitations="coverage 0.93"\n', encoding="utf-8")
            records.write_text('', encoding="utf-8")
            with patch.object(gate, "LICENSED", licensed), patch.object(gate, "RECORDS", records), patch.object(gate, "BACKLOG", backlog):
                self.assertTrue(any("0.93" in problem for problem in gate.check()))
                backlog.write_text("unattributed|q/g/s/i/v|0.93\n", encoding="utf-8")
                self.assertEqual(gate.check(), [])
                licensed.write_text(licensed.read_text().replace("0.93", "0.94"), encoding="utf-8")
                problems = gate.check()
                self.assertTrue(any("0.94" in problem for problem in problems))
                self.assertTrue(any("no longer occurs" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
