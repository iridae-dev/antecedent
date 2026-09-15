"""Identity dictionary: every domain is inspectable where advertised."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tomllib

import antecedent as ant

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = tomllib.loads((ROOT / "parity" / "identity.toml").read_text())
NAMING = (ROOT / "docs" / "api_naming.md").read_text()
GRAPH = [("t", "y"), ("z", "y")]


def _data(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=128)
    t = (rng.uniform(size=128) < 0.5).astype(float)
    return {"t": t, "y": 2 * t + z + rng.normal(scale=0.1, size=128), "z": z}


def test_identity_domains_match_native_order():
    assert [row["domain"] for row in IDENTITY["identity"]] == [
        "target",
        "identification",
        "identification_product",
        "program",
        "inference_binding",
        "observation",
        "data_snapshot",
        "execution",
        "claim",
        "score_reuse",
        "target_weights",
    ]


def test_naming_rows_exist():
    for row in IDENTITY["identity"]:
        assert f"| {row['naming_row']} |" in NAMING
        assert row["python"] in NAMING


def test_prepared_and_loaded_availability():
    prepared = ant.prepare(
        _data(), graph=GRAPH, query=ant.AverageEffect("t", "y"), refute="none", bootstrap=0
    )
    slots = prepared.inspect()
    executed = prepared.estimate()
    loaded = ant.load(executed.export())
    loaded_slots = loaded.inspect()
    for row in IDENTITY["identity"]:
        attr = row["python"].removeprefix("inspect().")
        prepared_value = getattr(slots, attr, None)
        loaded_value = getattr(loaded_slots, attr, None)
        if row["availability"] == "loaded":
            assert prepared_value in (None, "")
            if attr != "target_weights_id":
                assert loaded_value is None or isinstance(loaded_value, str)
        elif "prepared" in row["availability"] and prepared_value:
            assert isinstance(prepared_value, str)
        keys = row["contract_key"].split(".")
        payload = executed.inspect().to_dict()
        cur = payload.get("contract") or payload
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        assert cur is not None or row["python"] in NAMING
