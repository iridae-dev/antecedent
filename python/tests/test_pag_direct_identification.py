"""The Rust/Python compiler surfaces on the published oracle PAG fixtures."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest
from antecedent import _native
from antecedent import composition_bundle as cb
from antecedent import pag_identification as pid
from antecedent.outcomes import require_success

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "conformance" / "pag_direct_reference.json").read_text()
)


def _compile(case):
    fixture = CASES[case]
    names = fixture["names"]
    graph = _native.Pag.from_marked_edges(
        names, [(names[a], names[b], x, y) for a, b, x, y in fixture["edges"]]
    )
    return require_success(
        pid.compile_pag_identification(
            graph,
            actions=[names[i] for i in fixture["actions"]],
            outcomes=[names[i] for i in fixture["outcomes"]],
            given=[names[i] for i in fixture["given"]],
        )
    )


def _mass(bit, probability):
    return (1 - probability, probability)[bit]


def _equation_14(x1, x2, y):
    y1, y2, y3, y4, y5 = y
    return (
        _mass(y1, 0.1 + 0.2 * x1 + 0.2 * y5)
        * _mass(y2, (0.2, 0.7)[x2])
        * _mass(y3, (0.2, 0.6)[y4])
        * _mass(y4, 0.4)
        * _mass(y5, (0.3, 0.8)[y4])
    )


def test_published_marginal_formula_runs_and_replays_in_fresh_process(tmp_path: Path) -> None:
    plan = _compile("marginal_beyond_adjustment")
    result = require_success(plan.execute())
    assert result.status == "identified"
    density = [
        _mass(x1, 0.4) * _mass(x2, (0.3, 0.55)[x1]) * _equation_14(x1, x2, y)
        for x1, x2, *y in itertools.product((0, 1), repeat=7)
    ]
    law = require_success(result.evaluate([[0, 1]] * 7, density, {"X1": 1, "X2": 0}))
    assert law.probabilities == pytest.approx([_equation_14(1, 0, y) for y in law.atoms], abs=1e-12)
    bundle = cb.Bundle.builder().add_artifact(result).build()
    assert require_success(
        cb.consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity)
    ).all_replayed
    path = tmp_path / "direct-pag.bin"
    from antecedent import load

    artifact = result.export()
    assert artifact.expectation is not None
    restored = require_success(load(artifact, expected=artifact.expectation))
    assert restored.identity == result.identity
    path.write_bytes(bytes(artifact))
    call = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys, pathlib
from antecedent.outcomes import require_success
from antecedent.pag_identification import consume_pag_identification
run = require_success(consume_pag_identification(
    pathlib.Path(sys.argv[1]).read_bytes(), expected_result=sys.argv[2], expected_program=sys.argv[3],
))
print(run.status)
""",
            str(path),
            result.identity,
            result.program_identity,
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert call.stdout.strip() == "identified"
    with pytest.raises(pid.PagIdentificationRefusal):
        require_success(
            pid.consume_pag_identification(
                result.export(),
                expected_result="0" * 64,
                expected_program=result.program_identity,
            )
        )


def test_cidp_exchanges_observation_and_intervention_and_reports_a_blocking_path() -> None:
    forward = require_success(_compile("conditional_exchange").execute())
    masses = [
        _mass(a, 0.35)
        * _mass(d, 0.5)
        * _mass(b, 0.1 + 0.3 * a + 0.2 * d)
        * _mass(c, 0.1 + 0.2 * b + 0.4 * d)
        for a, b, c, d in itertools.product((0, 1), repeat=4)
    ]
    law = require_success(forward.evaluate([[0, 1]] * 4, masses, {"A": 0, "B": 1, "D": 1}))
    assert law.probabilities == pytest.approx((0.3, 0.7))
    reverse = require_success(_compile("conditional_reverse_exchange").execute())
    assert reverse.status == "identified"
    plan = _compile("conditional_obstruction")
    witness = require_success(plan.execute())
    assert witness.status == "conditional_obstruction"
    assert witness.proof["path"] == [0, 3, 2]
    with pytest.raises(pid.PagIdentificationRefusal) as refused:
        require_success(plan.execute(steps=2))
    assert refused.value.detail == "pag_identification.resource_exhausted"


def test_visible_nonnull_effect_and_retained_search_stop() -> None:
    plan = _compile("visible_collider_parent")
    result = require_success(plan.execute())
    probabilities = [
        0.25 * _mass(b, 0.25 + 0.25 * (a + c)) * _mass(d, (0.125, 0.875)[b])
        for a, b, c, d in itertools.product((0, 1), repeat=4)
    ]
    for b, expected in enumerate((0.125, 0.875)):
        law = require_success(result.evaluate([[0, 1]] * 4, probabilities, {"B": b}))
        assert law.probabilities[1] == pytest.approx(expected)

    plan = _compile("conditional_obstruction")
    stopped = require_success(plan.execute(steps=2, retain_resource_stop=True))
    assert stopped.status == "resource_stop"
    assert stopped.proof == {"status": "resource_stop", "limit": "work", "work": 2, "work_limit": 2}
    consumed = require_success(
        pid.consume_pag_identification(
            stopped.export(),
            expected_result=stopped.identity,
            expected_program=stopped.program_identity,
        )
    )
    assert consumed.proof == stopped.proof
    assert require_success(plan.execute()).status == "conditional_obstruction"
