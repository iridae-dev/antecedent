"""Release support changes must compare expanded coordinates, not grouped lines."""

import tempfile
import unittest
from pathlib import Path

from generate_support_matrix_docs import (
    FROZEN_BEGIN,
    FROZEN_END,
    release_support_delta,
)


class ReleaseSupportDeltaTests(unittest.TestCase):
    axes = {
        "queries": ["AverageEffect"],
        "graph_classes": ["Dag", "Admg"],
        "structures": ["explicit", "accepted"],
        "inferences": ["Frequentist"],
        "validations": ["none"],
    }

    def cell(self, graph="Dag", structure="explicit"):
        return {
            "query": "AverageEffect",
            "graph_class": graph,
            "structure": structure,
            "inference": "Frequentist",
            "validation": "none",
        }

    def compare(self, cells, inventory, count=2, *, frozen=True):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "v2.2.0.md"
            baseline.write_text(
                f"{FROZEN_BEGIN if frozen else ''}\n{count} licensed of 4 meaningful cells\n\n{inventory}\n{FROZEN_END}",
                encoding="utf-8",
            )
            return "\n".join(release_support_delta(cells, self.axes, baseline))

    inventory = "- `AverageEffect` × `Dag` / `Admg` × `explicit` × `Frequentist` × validation `none`"

    def test_unchanged_inventory_does_not_repeat_old_capabilities(self):
        rendered = self.compare([self.cell(), self.cell("Admg")], self.inventory)
        self.assertIn("0 added coordinates, 0 removed coordinates", rendered)
        self.assertNotIn("- `AverageEffect`", rendered)

    def test_only_actual_additions_and_removals_are_rendered(self):
        rendered = self.compare(
            [self.cell(), self.cell(structure="accepted")], self.inventory
        )
        self.assertIn("1 added coordinates, 1 removed coordinates", rendered)
        self.assertIn("Added analysis coordinates", rendered)
        self.assertIn("`Dag` × `accepted`", rendered)
        self.assertIn("Removed analysis coordinates", rendered)
        self.assertIn("`Admg` × `explicit`", rendered)
        self.assertNotIn("`Dag` × `explicit`", rendered)

    def test_invalid_or_unfrozen_baseline_fails(self):
        for kwargs in [{"count": 3}, {"frozen": False}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.compare([], self.inventory, **kwargs)
        for inventory in [
            self.inventory + "\n" + self.inventory,
            "- malformed coordinate",
        ]:
            with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                self.compare([], inventory)

    def test_removed_axis_is_reported_even_when_absent_from_current_axes(self):
        inventory = self.inventory.replace("`Dag` / `Admg`", "`OldGraph`")
        rendered = self.compare([self.cell()], inventory, count=1)
        self.assertIn("1 added coordinates, 1 removed coordinates", rendered)
        self.assertIn("`OldGraph`", rendered)


if __name__ == "__main__":
    unittest.main()
