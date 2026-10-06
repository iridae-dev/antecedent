"""Focused guardrails for incremental coverage-record collection."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_coverage_records as collector  # noqa: E402


class AppendAttestedRecordsTest(unittest.TestCase):
    def test_append_preserves_old_rows_and_original_sha(self) -> None:
        old = {"old": {"id": "old", "calibration_sha": "old-sha", "observed": 0.8}}
        new = {"new": {"id": "new", "calibration_sha": "new-sha", "observed": 0.9}}
        merged = collector.append_attested_records(old, new)
        self.assertEqual(set(merged), {"old", "new"})
        self.assertEqual(merged["old"], old["old"])
        self.assertEqual(merged["new"], new["new"])
        self.assertEqual(set(old), {"old"})

    def test_append_refuses_collision(self) -> None:
        with self.assertRaisesRegex(SystemExit, "refuses existing record ids"):
            collector.append_attested_records({"same": {}}, {"same": {}})

    def test_selected_cell_sync_leaves_other_cells_byte_identical(self) -> None:
        source = (
            '[[cell]]\nquery = "Elasticity"\ngraph_class = "Dag"\n'
            'structure = "explicit"\ninference = "Frequentist"\nvalidation = "none"\n'
            'calibration_reason = "estimator_grid_not_measured"\n\n'
            '[[cell]]\nquery = "Other"\ngraph_class = "Dag"\n'
            'structure = "explicit"\ninference = "Frequentist"\nvalidation = "none"\n'
            'calibration = ["old"]\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "licensed.toml"
            path.write_text(source)
            original = collector.LICENSED
            collector.LICENSED = path
            try:
                changed = collector.sync_licensed_cells(
                    {
                        "new": {
                            "query": "Elasticity",
                            "graph_class": "Dag",
                            "inference": "Frequentist",
                            "structure": "fixed",
                            "nominal": 0.95,
                            "boundary": False,
                        }
                    },
                    {("Elasticity", "Dag", "explicit", "Frequentist", "none"): ["new"]},
                )
            finally:
                collector.LICENSED = original
            self.assertEqual(changed, 1)
            self.assertIn('calibration = ["new"]', path.read_text())
            self.assertEqual(
                path.read_text().split('[[cell]]')[2],
                source.split('[[cell]]')[2],
            )


def _point(k: int, observed: float, role: str, **extra) -> dict:
    """One grid point's `calibration-record` payload, as the harness emits it."""
    payload = {
        "id": "cov.classical_transport.admg.frequentist.percentile_bootstrap.l95.t",
        "query": "ClassicalTransport",
        "nominal": 0.95,
        "n_min": 200 * 2**k,
        "n_max": 200 * 2**k,
        "replicates_min": 199,
        "posterior_draws_min": 0,
        "unidentified_mass_max": 0.0,
        "observed": observed,
        "mcse": 0.011,
        "replicates": 400,
        "bound_replicates": 400,
        "boundary": False,
        "role": role,
        "grid_point": k,
    }
    payload.update(extra)
    return payload


UPPER = {"side": "upper", "target": "true upper extremal bound U"}


class OneSidedRoleTest(unittest.TestCase):
    def test_a_one_sided_record_keeps_its_side_and_target(self) -> None:
        points = {k: _point(k, obs, "one_sided", one_sided=UPPER) for k, obs in enumerate((0.99, 0.985, 1.0))}
        record = collector.merge_grid("rid", points)
        self.assertEqual(record["role"], "one_sided")
        self.assertEqual(record["one_sided"], UPPER)
        self.assertFalse(record["boundary"])
        self.assertEqual({p["role"] for p in record["grid"]}, {"one_sided"})

    def test_a_one_sided_record_must_state_its_side_and_target(self) -> None:
        points = {k: _point(k, 0.99, "one_sided") for k in range(3)}
        with self.assertRaisesRegex(SystemExit, "states one_sided"):
            collector.merge_grid("rid", points)
        bad = {k: _point(k, 0.99, "one_sided", one_sided={"side": "both", "target": "t"}) for k in range(3)}
        with self.assertRaisesRegex(SystemExit, "side = lower|upper"):
            collector.merge_grid("rid", bad)

    def test_only_a_one_sided_record_states_a_side(self) -> None:
        points = {k: _point(k, 0.95, "gated", one_sided=UPPER) for k in range(3)}
        with self.assertRaisesRegex(SystemExit, "states one_sided"):
            collector.merge_grid("rid", points)

    def test_one_sided_does_not_mix_with_another_role(self) -> None:
        points = {k: _point(k, 0.99, "one_sided", one_sided=UPPER) for k in range(2)}
        points[2] = _point(2, 0.95, "gated", one_sided=UPPER)
        with self.assertRaisesRegex(SystemExit, "another construction|incompatible roles|states one_sided"):
            collector.merge_grid("rid", points)

    def test_an_unknown_role_is_refused(self) -> None:
        points = {k: _point(k, 0.95, "two_sided_ish") for k in range(3)}
        with self.assertRaisesRegex(SystemExit, "unknown coverage role"):
            collector.merge_grid("rid", points)

    def test_the_registry_writes_one_sided_after_role(self) -> None:
        import tomllib

        record = collector.merge_grid(
            "rid", {k: _point(k, 0.99, "one_sided", one_sided=UPPER) for k in range(3)}
        )
        record.update({key: "x" for key in collector.FIELDS if key not in record})
        record.update(facets=["core"], calibration_sha="1" * 40, surface_list_blob="1" * 40)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "records.toml"
            original = collector.surface_list_blob
            collector.surface_list_blob = lambda sha: "1" * 40
            try:
                collector.write_registry({"rid": record}, out, retag=False)
            finally:
                collector.surface_list_blob = original
            text = out.read_text()
            parsed = tomllib.loads(text)["record"][0]
        self.assertEqual(parsed["one_sided"], UPPER)
        self.assertLess(text.index('role = "one_sided"'), text.index("one_sided = {"))
        self.assertLess(text.index("one_sided = {"), text.index("grid = ["))


if __name__ == "__main__":
    unittest.main()
