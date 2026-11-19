"""A capability route closes only on supported records; an estimator route only on a cell."""

import unittest

import route_closure

POINT = {"id": "rec.point", "status": "supported", "inference_claim": "point_only"}
INTERVAL = {
    "id": "rec.interval",
    "status": "supported",
    "inference_claim": "interval",
    "coverage_records": ["cov.a", "cov.b"],
}
OUTPUT = {
    "id": "rec.output",
    "status": "supported",
    "inference_claim": "point_only",
    "inference_outputs": [{"id": "wald", "allocated_coverage_records": ["cov.c"]}],
}
BARE_INTERVAL = {"id": "rec.bare", "status": "supported", "inference_claim": "interval"}
REFUSED = {"id": "rec.refused", "status": "not_supported", "inference_claim": "none"}
RECORDS = [POINT, INTERVAL, OUTPUT, BARE_INTERVAL, REFUSED]


def route(record, **fields):
    return {"id": "r", "status": "capability", "capability_record": record, **fields}


def problems(*routes):
    return route_closure.manifest_capability_problems(list(routes), RECORDS)


class CapabilityRoutes(unittest.TestCase):
    def test_a_supported_record_closes_the_route(self):
        self.assertEqual(problems(route("rec.point")), [])
        self.assertEqual(problems(route("rec.interval")), [])
        self.assertEqual(problems(route(["rec.interval", "rec.output"])), [])

    def test_a_route_needs_a_record_that_exists_and_is_supported(self):
        self.assertIn("names no capability_record", problems({"id": "r", "status": "capability"})[0])
        self.assertIn("is not a capability record", problems(route("rec.absent"))[0])
        self.assertIn("not supported", problems(route("rec.refused"))[0])

    def test_an_interval_claim_needs_coverage_ids_on_every_record(self):
        self.assertIn("carries no coverage record id", problems(route("rec.bare"))[0])
        mixed = problems(route(["rec.interval", "rec.point"]))
        self.assertEqual(len(mixed), 1)
        self.assertIn("rec.point", mixed[0])
        # A declared interval on a point-only record is not closed by it.
        self.assertTrue(problems(route("rec.point", capability_claim="interval")))

    def test_a_declared_point_claim_reads_no_interval_from_the_record(self):
        self.assertEqual(problems(route("rec.bare", capability_claim="point_only")), [])
        self.assertIn("is not one of", problems(route("rec.point", capability_claim="wide"))[0])

    def test_the_fields_belong_to_capability_routes_only(self):
        stray = {"id": "g", "status": "graphless", "capability_record": "rec.point"}
        self.assertIn("belong to a route whose status is capability", problems(stray)[0])

    def test_an_allocated_id_without_a_measured_record_is_owed(self):
        routes = [route(["rec.interval", "rec.output"])]
        owed = route_closure.owed_measurements(routes, RECORDS, {"cov.a"})
        self.assertEqual(owed, [("r", "coverage record cov.b"), ("r", "coverage record cov.c")])
        self.assertEqual(
            route_closure.owed_measurements(routes, RECORDS, {"cov.a", "cov.b", "cov.c"}), []
        )
        point = [route("rec.bare", capability_claim="point_only")]
        self.assertEqual(route_closure.owed_measurements(point, RECORDS, set()), [])

    def test_a_mapped_gate_group_carries_the_interval_and_is_owed_until_measured(self):
        groups = {"rec.point": ["harness: wald_l95", "whole_file"]}
        self.assertEqual(problems(route("rec.point")), [])
        mapped = route_closure.manifest_capability_problems([route("rec.point")], RECORDS, groups)
        self.assertEqual(mapped, [])
        tests = {"whole_file": ["first_test", "second_test"]}
        owed = route_closure.owed_measurements(
            [route("rec.point")], RECORDS, {"cov.q.l95.x_wald_l95.v2"}, groups, tests
        )
        self.assertEqual(owed, [("r", "gate group whole_file")])
        measured = {"cov.q.l95.x_wald_l95.v2", "cov.q.l95.second_test.v2"}
        self.assertEqual(
            route_closure.owed_measurements([route("rec.point")], RECORDS, measured, groups, tests),
            [],
        )
        # The mapping makes the route an interval route: the claim is not silently a point.
        self.assertEqual(route_closure.route_claim(route("rec.point"), {"rec.point": POINT}, groups), "interval")


CELL = {
    "graph_class": "TemporalDag",
    "inference": "Bayesian",
    "estimators": ["first", "second"],
    "evidence_test": "tests/first.rs",
    "checked_execution": [{"estimator": "second", "test": "tests/second.rs"}],
}


def estimator_route(**fields):
    base = {
        "id": "second",
        "status": "supported",
        "graph_class": "TemporalDag",
        "inference": "Bayesian",
        "notes": "",
    }
    return {**base, **fields}


class EstimatorRoutes(unittest.TestCase):
    def check(self, *routes, cells=(CELL,)):
        return route_closure.estimator_route_problems(list(routes), list(cells))

    def test_an_estimator_with_evidence_on_a_matching_cell_is_supported(self):
        self.assertEqual(self.check(estimator_route()), [])
        self.assertEqual(self.check(estimator_route(id="x", notes="Estimator second. More.")), [])
        self.assertEqual(self.check(estimator_route(id="first")), [])

    def test_a_geometric_record_is_not_an_estimator_route(self):
        self.assertEqual(self.check(estimator_route(id="supported.Q.TemporalDag")), [])

    def test_an_unlisted_estimator_or_another_cell_is_refused(self):
        self.assertIn("names no estimator", self.check(estimator_route(id="third"))[0])
        self.assertIn("is on no supported", self.check(estimator_route(graph_class="Dag"))[0])
        self.assertIn("is on no supported", self.check(estimator_route(inference="Frequentist"))[0])

    def test_an_estimator_without_its_own_evidence_is_refused(self):
        bare = {**CELL, "checked_execution": []}
        self.assertIn("cites evidence", self.check(estimator_route(), cells=(bare,))[0])


if __name__ == "__main__":
    unittest.main()
