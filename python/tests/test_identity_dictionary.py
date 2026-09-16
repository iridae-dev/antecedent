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


HEX64 = re.compile(r"[0-9a-f]{64}")


def _contract_value(section: dict, key_path: str):
    cur = section
    for key in key_path.split("."):
        assert isinstance(cur, dict) and key in cur, f"{key_path} missing from the contract"
        cur = cur[key]
    return cur


def _hex(value) -> str:
    return value if isinstance(value, str) else bytes(value).hex()


def test_prepared_and_loaded_availability():
    # An AIPW prepared study holds a score table, so every advertised domain,
    # including score_reuse, has a value; a nonconstant retarget adds
    # target_weights.
    data = _data()
    prepared = ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect("t", "y"),
        estimator="aipw",
        refute="none",
        bootstrap=0,
    )
    slots = prepared.inspect()
    executed = prepared.estimate()
    retargeted = prepared.retarget(np.exp(data["z"] / 3), depends_on=["z"])
    exported = retargeted.export()
    loaded_slots = ant.load(exported).inspect()
    section = ant.artifacts.loads(exported).contract
    for row in IDENTITY["identity"]:
        attr = row["python"].removeprefix("inspect().")
        prepared_value = getattr(slots, attr)
        loaded_value = getattr(loaded_slots, attr)
        if "prepared" in row["availability"]:
            assert isinstance(prepared_value, str) and HEX64.fullmatch(prepared_value), attr
        else:
            assert prepared_value is None, attr
        assert "loaded" in row["availability"], attr
        assert isinstance(loaded_value, str) and HEX64.fullmatch(loaded_value), attr
        assert _hex(_contract_value(section, row["contract_key"])) == loaded_value, attr
    assert executed.inspect().target_weights_id is None
    assert HEX64.fullmatch(executed.inspect().score_reuse_id)
