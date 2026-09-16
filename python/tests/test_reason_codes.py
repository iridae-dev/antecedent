"""Closed reason-code vocabulary."""

from __future__ import annotations

from pathlib import Path

from antecedent.errors import CausalUnsupportedError

from _repo_text import load_toml, read_text

ROOT = Path(__file__).resolve().parents[2]
CODES = load_toml(ROOT / "parity" / "reason_codes.toml")


def test_reason_codes_are_closed():
    ids = [row["id"] for row in CODES["code"]]
    assert "not_executed" in ids
    assert "cancelled_no_claim" in ids
    assert "row_weights_bound_to_snapshot" in ids


def test_every_runtime_reason_code_is_in_the_vocabulary():
    import re
    from pathlib import Path

    vocab = {
        row["id"]
        for row in CODES["code"]
        if "runtime_refusal" in row.get("applies_to", [])
    }
    text = "\n".join(
        path.read_text() for path in Path(__file__).resolve().parents[1].joinpath("antecedent").rglob("*.py")
    )
    for code in re.findall(r'reason_code="([^"]+)"', text):
        assert code in vocab, code


def test_unsupported_error_carries_code():
    err = CausalUnsupportedError("no study", reason_code="not_executed")
    assert err.reason_code == "not_executed"
