"""2.3 A exit gate, box 1: the A evidence ledger.

The test reads only committed registries (``parity/promotion_2_3.toml``,
``parity/2_3_evidence_ledger.toml``, ``parity/coverage_records.toml``,
``parity/reason_codes.toml``) and ``provenance/*.toml``; it never runs a gate script.

For every milestone-A record it asserts:

* a ``promoted`` record has a ledger cell naming provenance files that exist and parse, a
  positive, a negative and an artifact fixture whose evidence tests exist as files and name
  their assertion, and an evidence kind from the registered oracle vocabulary;
* a record whose ``inference_claim`` is an inferential one (not ``point_only``,
  ``structural_envelope`` or ``none``) is promoted only when each of its coverage records is
  collected; otherwise it is not promoted, every route is closed with a registered reason
  code and the closed public producer really refuses;
* a ``carried_forward`` record has only closed routes, and no public Python consumer name
  exists for a closed artifact.

The closure checks are status-driven: a route that the registry has opened is not called, and
its record must then be promoted. Composes the committed refusal tests
``test_closed_pilots.py``, ``test_temporal_extensions.py`` and ``test_temporal_counterfactual.py``
by copying their request builders.

Calibration is unmeasured for the calibrated A claims at this commit; this file asserts the
registry state, it does not measure anything.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from antecedent import Admg, temporal_counterfactual
from antecedent.errors import CausalUnsupportedError
from antecedent.temporal_counterfactual import transported_path_specific
from antecedent.transport import advanced as transport
from antecedent.transport.advanced import TemporalUnitPanel, temporal_dependent_interval

from _refusal import REGISTERED

ROOT = Path(__file__).resolve().parents[2]
NOT_INFERENTIAL = frozenset({"point_only", "structural_envelope", "none"})
ORACLE_KINDS = frozenset({"closed_form", "enumerated_finite_law", "semantic_contract"})
REQUIRED_ROLES = ("positive", "negative", "artifact")

Record = dict[str, Any]


def load_toml(relative: str) -> dict[str, Any]:
    with (ROOT / relative).open("rb") as handle:
        return tomllib.load(handle)


PROMOTION = load_toml("parity/promotion_2_3.toml")
LEDGER = load_toml("parity/2_3_evidence_ledger.toml")
COVERAGE_IDS = {r["id"] for r in load_toml("parity/coverage_records.toml").get("record", [])}
RUNTIME_CODES = {
    c["id"]
    for c in load_toml("parity/reason_codes.toml")["code"]
    if "runtime_refusal" in c.get("applies_to", [])
}
A_RECORDS: list[Record] = [r for r in PROMOTION["record"] if r["milestone"] == "A"]
CELLS_BY_ID: dict[str, list[Record]] = {}
for _cell in LEDGER["cell"]:
    CELLS_BY_ID.setdefault(_cell["id"], []).append(_cell)

PROMOTED = [r for r in A_RECORDS if r["status"] == "promoted"]
INFERENTIAL = [r for r in A_RECORDS if r["inference_claim"] not in NOT_INFERENTIAL]
CARRIED = [r for r in A_RECORDS if r["status"] == "carried_forward"]


def ids(records: list[Record]) -> list[str]:
    return [r["id"] for r in records]


def test_a_exit_ledger_covers_every_a_record_once() -> None:
    assert A_RECORDS, "no milestone A records"
    assert len({r["id"] for r in A_RECORDS}) == len(A_RECORDS)
    assert {r["status"] for r in A_RECORDS} <= {"promoted", "carried_forward"}
    # Records with no ledger cell cannot be promoted.
    assert all(r["id"] in CELLS_BY_ID for r in PROMOTED)


@pytest.mark.parametrize("record", PROMOTED, ids=ids(PROMOTED))
def test_a_exit_ledger_promoted_record_has_provenance_fixtures_and_ledger_cell(
    record: Record,
) -> None:
    cells = CELLS_BY_ID.get(record["id"], [])
    assert cells, f"{record['id']}: no ledger cell"
    routes = {route["name"] for route in record["routes"]}
    for cell in cells:
        assert cell["route"] in routes, (record["id"], cell["route"])
        assert cell["parity_registry"] == "parity/promotion_2_3.toml"
        # Correct parity evidence kind: a registered independent oracle, never a pending one.
        assert cell["oracle_kind"] in ORACLE_KINDS, (record["id"], cell["oracle_kind"])
        assert cell["algorithm_provenance"], record["id"]
        for path in cell["algorithm_provenance"]:
            provenance = ROOT / path
            assert provenance.is_file(), path
            with provenance.open("rb") as handle:
                parsed = tomllib.load(handle)
            assert parsed["feature_id"], path
            for source in parsed.get("test_sources", []):
                assert (ROOT / source).is_file(), (path, source)
        # The ledger's positive and artifact fixtures are fixtures of the record.
        fixtures = {f["id"]: f for f in record["fixtures"]}
        assert fixtures[cell["positive_fixture"]]["role"] == "positive"
        assert fixtures[cell["artifact_consumer_fixture"]]["role"] == "artifact"
    roles = {fixture["role"] for fixture in record["fixtures"]}
    assert set(REQUIRED_ROLES) <= roles, (record["id"], roles)
    for fixture in record["fixtures"]:
        if fixture["role"] == "calibration":
            continue
        evidence = ROOT / fixture["evidence_test"]
        assert evidence.is_file(), (record["id"], fixture["evidence_test"])
        assert fixture["evidence_assertion"] in evidence.read_text(encoding="utf-8"), fixture["id"]


@pytest.mark.parametrize("record", PROMOTED, ids=ids(PROMOTED))
def test_a_exit_ledger_promoted_record_routes_are_licensed_not_closed(record: Record) -> None:
    for route in record["routes"]:
        if route["status"] == "closed":
            # A permanent in-release closure inside a promoted record is allowed; it must be named.
            assert route.get("reason_code") in RUNTIME_CODES, route
            assert Path(ROOT / route["refusal_test"]).is_file(), route
        else:
            assert route["status"] == "licensed", route


@pytest.mark.parametrize("record", INFERENTIAL, ids=ids(INFERENTIAL))
def test_a_exit_ledger_inferential_claim_is_calibrated_or_not_promoted(record: Record) -> None:
    coverage = record.get("coverage_records", [])
    if record["status"] == "promoted":
        # Promotion of an inferential claim requires its own collected coverage record.
        assert coverage, f"{record['id']}: promoted inferential claim without coverage records"
        assert set(coverage) <= COVERAGE_IDS, (record["id"], set(coverage) - COVERAGE_IDS)
        return
    assert record["status"] == "carried_forward", record["id"]
    for route in record["routes"]:
        assert route["status"] == "closed", (record["id"], route["name"])
        assert route["reason_code"] == "cell_not_licensed", route
        assert route["reason_code"] in RUNTIME_CODES
        assert (ROOT / route["refusal_test"]).is_file(), route
        assert route["refusal_assertion"] in (ROOT / route["refusal_test"]).read_text(
            encoding="utf-8"
        )
    # The failed gate is recorded in the record, not silently dropped.
    assert "unmeasured" in record.get("inference_notes", "") or coverage, record["id"]
    for cell in CELLS_BY_ID.get(record["id"], []):
        assert "unmeasured" in cell["calibration"] or cell["calibration"].startswith("cov."), cell


@pytest.mark.parametrize("record", CARRIED, ids=ids(CARRIED))
def test_a_exit_ledger_carried_forward_routes_are_closed_with_a_registered_code(
    record: Record,
) -> None:
    assert record["routes"], record["id"]
    for route in record["routes"]:
        assert route["status"] == "closed", (record["id"], route["name"])
        assert route["reason_code"] in RUNTIME_CODES, route
    detail_codes = {refusal["code"] for refusal in record["refusals"]}
    assert "cell_not_licensed" in detail_codes, record["id"]
    frozen = [
        r["detail"]
        for r in record["refusals"]
        if r["code"] == "cell_not_licensed" and r["detail"].endswith(".route_frozen")
    ]
    assert frozen, f"{record['id']}: no registered route_frozen refusal detail"


# ------------------------------------------------------------------ real closed producers
# Request builders copied from test_closed_pilots.py, test_temporal_extensions.py and
# test_temporal_counterfactual.py.

NODES = ["X1", "X2", "X3", "X4"]
CELLS = [100.0 + index for index in range(16)]


def _joint_bayesian() -> object:
    return transport.joint_bayesian_transport(
        sources=[{"id": "s1"}, {"id": "s2"}],
        target={"x": [0.1, 0.2, 0.3]},
        features=["x"],
        draws=1000,
        seed=7,
    )


def _learned_joint() -> object:
    return transport.learned_joint_transport(
        sources=[{"id": "s1"}, {"id": "s2"}],
        target={"x": [0.1, 0.2, 0.3]},
        features=["x"],
        basis_degree=2,
        draws=1000,
        seed=7,
    )


def _nested_markov() -> object:
    graph = Admg.from_edges(NODES, [("X1", "X2"), ("X2", "X3"), ("X3", "X4")], [("X2", "X4")])
    return transport.binary_nested_markov(graph=graph, regimes=[{"counts": CELLS}])


def _nested_markov_fisher() -> object:
    graph = Admg.from_edges(NODES, [("X1", "X2"), ("X2", "X3"), ("X3", "X4")], [("X2", "X4")])
    return transport.binary_nested_markov_fisher_interval(graph=graph, regimes=[{"counts": CELLS}])


def _sampled_recovery() -> object:
    query = transport.ObservationRecoveryQuery(
        population="clinic",
        observed_regime="observed",
        partially_observed=[transport.PartiallyObservedVariable("X", "R", "X_star")],
    )
    rows = [(index, 1, (0, 1)[index % 2], 0) for index in range(40)]
    return transport.sampled_observation_recovery(
        stage=SimpleNamespace(outcome="recovered"),
        query=query,
        rows=rows,
        snapshot="snap-observed",
        replicates=100,
        seed=3,
    )


def _dependent_interval() -> object:
    rows = []
    for unit in range(24):
        time_id = 0
        for s0 in range(2):
            for level in range(2):
                for a2 in range(2):
                    y = ((unit * 7 + s0 * 3 + level * 5 + a2) % 10) / 10.0
                    rows.append((unit, time_id, s0, 0, level, a2, y))
                    time_id += 1
    panel = TemporalUnitPanel.from_rows("interval-panel", rows)
    return temporal_dependent_interval(panel, sequence=(0, 0), replicates=40)


def _transported_path_specific() -> object:
    return transported_path_specific(required_factors=["source:rct", "target:field"])


CLOSED_PRODUCERS: dict[str, Callable[[], object]] = {
    "antecedent.transport.joint_bayesian": _joint_bayesian,
    "antecedent.learned.joint_transport": _learned_joint,
    "antecedent.transport.binary_nested_markov": _nested_markov,
    "antecedent.transport.binary_nested_markov_fisher_interval": _nested_markov_fisher,
    "antecedent.transport.sampled_observation_recovery": _sampled_recovery,
    "antecedent.transport.temporal_dependent_interval": _dependent_interval,
    "antecedent.cross_world.transported_path_specific": _transported_path_specific,
}

REGISTERED_CLOSED_ROUTES = [
    (record, route)
    for record in CARRIED
    for route in record["routes"]
    if route["status"] == "closed"
]


@pytest.mark.parametrize(
    ("record", "route"),
    REGISTERED_CLOSED_ROUTES,
    ids=[route["name"] for _, route in REGISTERED_CLOSED_ROUTES],
)
def test_a_exit_ledger_closed_public_producer_refuses_with_its_typed_cell(
    record: Record, route: Record
) -> None:
    assert route["name"] in CLOSED_PRODUCERS, f"no producer call for closed route {route['name']}"
    with pytest.raises(CausalUnsupportedError) as caught:
        CLOSED_PRODUCERS[route["name"]]()
    error = caught.value
    assert error.reason_code == route["reason_code"] == "cell_not_licensed"
    assert error.reason_code in REGISTERED
    frozen = [
        r["detail"]
        for r in record["refusals"]
        if r["code"] == "cell_not_licensed" and r["detail"].endswith(".route_frozen")
    ]
    assert any(detail in str(error) for detail in frozen), (str(error), frozen)


def test_a_exit_ledger_registry_closed_routes_are_all_known_producers() -> None:
    names = {route["name"] for _, route in REGISTERED_CLOSED_ROUTES}
    assert names <= set(CLOSED_PRODUCERS), names - set(CLOSED_PRODUCERS)
    # Every A inferential claim that is not promoted contributes at least one closed route.
    for record in INFERENTIAL:
        if record["status"] != "promoted":
            assert any(route["status"] == "closed" for route in record["routes"])


def test_a_exit_ledger_no_public_python_consumer_for_a_closed_artifact() -> None:
    # Producer and consumer stay closed together: a closed route has no public Python
    # consumer name (the artifact consumers of the carried-forward cells are Rust-internal).
    stems = ("joint_bayesian", "nested_markov", "sampled_observation", "dependent_interval")
    public = list(getattr(transport, "__all__", [])) + list(
        getattr(temporal_counterfactual, "__all__", [])
    )
    closed_names = {route["name"] for _, route in REGISTERED_CLOSED_ROUTES}
    for name in public:
        lowered = name.lower()
        if "consume" in lowered and any(stem in lowered for stem in stems):
            # Allowed only when the registry has since opened the matching route.
            assert not closed_names, f"public consumer {name} beside closed routes {closed_names}"
    assert "consume_transported_counterfactual_artifact" not in public
