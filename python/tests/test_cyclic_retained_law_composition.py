"""New worlds retain the exact native experimental law and independent codec."""

import gc
import json
import os
import subprocess
import sys

import antecedent as ant
import pytest
from antecedent.cyclic import (
    AsymptoticNormal,
    CyclicResult,
    EvidenceSchema,
    EvidenceVariable,
    RecoveredCyclicLaw,
    RecoveryBootstrap,
    Set,
    Shift,
    World,
)
from antecedent.errors import CausalValueError
from antecedent.outcomes import require_success

from test_cyclic import _recovery


@pytest.mark.parametrize(
    "origin, design",
    [(AsymptoticNormal(4), None), (RecoveryBootstrap(4), "independent_subject_trajectories")],
)
@pytest.mark.parametrize("loaded", [False, True])
def test_law_prepares_worlds_without_converting_original_source(
    monkeypatch, origin, design, loaded
):
    source = require_success(_recovery(origin, sampling_design=design).execute())
    if loaded:
        artifact = source.export()
        source = require_success(ant.load(artifact, expected=artifact.expectation))
    law = source.law
    before = law.to_dict()

    def forbidden(*args, **kwargs):
        pytest.fail("retained-law composition materialized the source")

    with monkeypatch.context() as bounded:
        bounded.setattr(RecoveredCyclicLaw, "to_dict", forbidden)
        bounded.setattr(CyclicResult, "to_dict", forbidden)
        prepared = law.prepare([World(47, Set("A", 3.0))], variables=["A", "B"], outcome="B")
        assert prepared.inspect()["preparation_evaluations"] == 0
        assert prepared.inspect()["retained_law"] is True
        branch = prepared.reconfigure(worlds=[World(48, Set("A", 4.0))])
        first = require_success(prepared.execute())
        second = require_success(branch.execute())
        assert first.law.draw_status().items == law.draw_status().items
    assert first.law.to_dict() == before == second.law.to_dict() == law.to_dict()
    assert first.identity != second.identity
    assert first.law.inspect().details["law_meaning"] == law.inspect().details["law_meaning"]
    worlds = require_success(first.interventions())
    assert tuple(worlds) == (47,)
    selected = worlds[47].draw_values(0)
    assert selected is not None and selected.readonly
    assert selected[0] == pytest.approx(3.0)
    original_worlds = require_success(source.interventions())
    assert 47 not in original_worlds
    with pytest.raises(CausalValueError, match="no rebindable observations"):
        prepared.rebind({"samples": []})
    with pytest.raises(CausalValueError, match="inference"):
        prepared.reconfigure(uncertainty=AsymptoticNormal(8))
    del source, prepared, branch, first, second, worlds, original_worlds
    gc.collect()
    assert law.to_dict() == before
    assert selected[0] == pytest.approx(3.0)


def test_unresolved_law_preserves_failed_slots_and_refuses_unsupported_actions():
    source = require_success(_recovery(AsymptoticNormal(4), designs=(None, "A")).execute())
    law = source.law
    before = law.to_dict()
    prepared = law.prepare([World(47, Set("A", 3.0))], variables=["A", "B"], outcome="B")
    result = require_success(prepared.execute())
    assert result.law.to_dict() == before
    assert result.law.draw_status().items == law.draw_status().items
    assert result.law.unresolved(0).items == law.unresolved(0).items
    refused = law.prepare([World(48, Shift("A", 1.0))], variables=["A", "B"], outcome="B")
    assert refused.refusal.code == "route_not_supported"
    assert ant.estimate(refused).refusal.detail == "cyclic_recovery.hard_clamp_required"
    assert law.to_dict() == before


def test_continuation_exports_existing_codec_and_loads_in_fresh_process(tmp_path):
    source = require_success(_recovery(AsymptoticNormal(4)).execute())
    prepared = source.law.prepare([World(47, Set("A", 3.0))], variables=["A", "B"], outcome="B")
    result = require_success(prepared.execute())
    artifact = result.export()
    loaded = require_success(ant.load(bytes(artifact), expected=artifact.expectation))
    assert loaded.to_dict() == result.to_dict()
    assert loaded.law.to_dict() == source.law.to_dict()
    assert loaded.inspect().stage == "consumed"
    schema = EvidenceSchema(
        "B",
        "declared population",
        (
            EvidenceVariable("A", "meter"),
            EvidenceVariable("B", "meter"),
        ),
    )
    selected = require_success(loaded.interventions())[47]
    evidence = require_success(selected.evidence(schema, variable=1, port="response"))
    assert evidence.quantities[0].units == "meter"
    assert evidence.quantities[0].population_id == "declared population"
    encoded = tmp_path / "continued.ant"
    encoded.write_bytes(bytes(artifact))
    script = """
import json, pathlib, sys
from antecedent.cyclic import consume_family_artifact, Set, World
from antecedent.outcomes import require_success
result = require_success(consume_family_artifact(pathlib.Path(sys.argv[1]).read_bytes(), expected_identity=sys.argv[2]))
branch = result.law.prepare([World(51, Set('A', 5.0))], variables=['A', 'B'], outcome='B')
continued = require_success(branch.execute())
print(json.dumps({'source': result.identity, 'worlds': list(require_success(continued.interventions())), 'law': continued.law.to_dict()}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(encoded), result.identity],
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    replay = json.loads(completed.stdout)
    assert replay["source"] == result.identity
    assert replay["worlds"] == [51]
    assert replay["law"] == source.law.to_dict()
