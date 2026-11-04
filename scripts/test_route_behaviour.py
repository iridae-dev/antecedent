"""Generated route behaviour: derivation, materialisation, stage merge and drift."""

import unittest
from pathlib import Path

import route_behaviour as rb
from capability_source import Lit

DETAILS = {"ns.a": ["invalid_argument"], "ns.b": ["transport_not_certified", "invalid_argument"]}


def emitted(*details, code=None):
    out = {}
    for d in details:
        out.setdefault(d.split(".")[0], {})[d] = [(Path("x.rs"), True, Lit(1, d, None, code))]
    return out


def record(**extra):
    return {
        "id": "r.one",
        "status": "supported",
        "inference_claim": "point_only",
        "refusals": [{"detail": "ns.a"}, {"detail": "ns.b", "when": "only here"}],
        "routes": [
            {"name": "p.run", "stage": "evaluate", "status": "supported"},
            {"name": "p.refuse", "stage": "uncertainty", "status": "closed", "reason_code": "not_supported", "claim": "none"},
            {"name": "s.cell", "stage": "evaluate", "status": "supported", "registry": "support_cells"},
        ],
        **extra,
    }


class Derive(unittest.TestCase):
    def test_pairs_come_from_the_registry_and_routes_from_the_record(self):
        data, problems = rb.derive([record()], DETAILS, emitted("ns.a", "ns.b"))
        self.assertEqual(problems, [])
        self.assertEqual(
            data["r.one"]["refusals"],
            [("invalid_argument", "ns.a"), ("transport_not_certified", "ns.b"), ("invalid_argument", "ns.b")],
        )
        self.assertEqual([r["route"] for r in data["r.one"]["routes"]], ["p.run", "p.refuse"])
        self.assertEqual(data["r.one"]["routes"][0]["claim"], "point_only")
        self.assertEqual(data["r.one"]["routes"][1]["reason_code"], "not_supported")

    def test_an_unregistered_detail_and_an_unlisted_proven_code_are_problems(self):
        rec = record(refusals=[{"detail": "ns.zzz"}, {"detail": "ns.a"}])
        _, problems = rb.derive([rec], DETAILS, emitted("ns.a", code="not_supported"))
        self.assertTrue(any("ns.zzz is not in" in p for p in problems))
        self.assertTrue(any("not_supported" in p and "does not list" in p for p in problems))

    def test_a_not_supported_record_has_no_derived_refusals(self):
        data, _ = rb.derive([record(status="not_supported")], DETAILS, {})
        self.assertNotIn("refusals", data["r.one"])


class RenderAndMerge(unittest.TestCase):
    def test_render_parse_round_trip(self):
        data, _ = rb.derive([record()], DETAILS, emitted("ns.a", "ns.b"))
        self.assertEqual(rb.parse(rb.render(data)), data)

    def test_materialise_gives_rows_their_codes_and_rejects_restated_ones(self):
        data, _ = rb.derive([record()], DETAILS, emitted("ns.a", "ns.b"))
        rec = record()
        self.assertEqual(rb.materialise_refusals([rec], data), [])
        self.assertEqual([(r["code"], r["detail"]) for r in rec["refusals"]][:2],
                         [("invalid_argument", "ns.a"), ("transport_not_certified", "ns.b")])
        self.assertEqual(rec["refusals"][1]["when"], "only here")
        rec = record(refusals=[{"code": "invalid_argument", "detail": "ns.a"}])
        self.assertTrue(rb.materialise_refusals([rec], data))

    def test_stage_rows_merge_and_may_not_restate(self):
        data, _ = rb.derive([record()], DETAILS, emitted("ns.a", "ns.b"))
        stages = {"routes": [{"route": "p.run", "notes": "n"}, {"route": "other", "stage": "identify", "status": "supported"}]}
        rows, problems = rb.stage_routes(Path("."), stages, data)
        self.assertEqual(problems, [])
        by = {r["route"]: r for r in rows}
        self.assertEqual((by["p.run"]["stage"], by["p.run"]["notes"]), ("evaluate", "n"))
        self.assertIn("p.refuse", by)
        self.assertEqual(by["other"]["stage"], "identify")
        stages["routes"][0]["status"] = "closed"
        self.assertTrue(rb.stage_routes(Path("."), stages, data)[1])


class Committed(unittest.TestCase):
    def test_the_committed_file_parses_and_covers_every_record_route(self):
        import tomllib

        from _repo import REPO_ROOT

        generated = rb.load(REPO_ROOT / rb.GENERATED)
        self.assertIsNotNone(generated)
        records = tomllib.loads((REPO_ROOT / "parity/capability_records.toml").read_text(encoding="utf-8"))["record"]
        self.assertEqual(set(generated), {r["id"] for r in records})
        for r in records:
            self.assertEqual([x["route"] for x in generated[r["id"]]["routes"]], [x["name"] for x in rb.owned_routes(r)])
            self.assertTrue(all("code" not in row for row in r["refusals"]) or r["status"] != "supported")

    def test_load_stage_routes_reads_generated_status(self):
        rows = rb.load_stage_routes()
        self.assertTrue(rows)
        self.assertTrue(all(r.get("stage") and r.get("status") for r in rows))


if __name__ == "__main__":
    unittest.main()
