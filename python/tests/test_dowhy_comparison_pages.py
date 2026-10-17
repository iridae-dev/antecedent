"""The comparison pages name the pinned DoWhy version, the overlap and the differences.

This is the part of the DoWhy comparison that needs no DoWhy: it checks that each capability guide
carries a "Differences from DoWhy" section tied to the version pinned in
``parity/baselines/dowhy.toml``, that every comparison recorded in that pin is checked by a test that
exists, and that the executable examples a page points to exist, assert something, and are
registered in the Rust example suite, the Cargo manifest and the example catalog. The values
themselves are compared in ``test_dowhy_comparison.py``.
"""

from __future__ import annotations

import ast
import re

import pytest

from _repo_text import REPO_ROOT, load_toml, read_text

PIN = load_toml(REPO_ROOT / "parity" / "baselines" / "dowhy.toml")
GUIDES = REPO_ROOT / "docs" / "guides"
COMPARISON_TEST = REPO_ROOT / "python" / "tests" / "test_dowhy_comparison.py"

# guide -> (DoWhy function it compares, example it runs)
COMPARED = {
    "intervention-programs": ("gcm.interventional_samples", "intervention_programs"),
    "mechanism-selection": ("gcm.auto.assign_causal_mechanisms", "mechanism_selection"),
    "intrinsic-influence": ("gcm.intrinsic_causal_influence", "intrinsic_influence"),
    "graph-falsification": ("gcm.falsify.falsify_graph", "graph_falsification"),
}
EXAMPLES = sorted({example for _, example in COMPARED.values()} | {"noise_replay"})


def _section(text: str, heading: str) -> str:
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"missing section {heading!r}"
    return match.group(1)


def _assert_example_is_registered(example):
    rust = REPO_ROOT / "examples" / "rust" / f"{example}.rs"
    python = REPO_ROOT / "examples" / "python" / f"{example}.py"
    assert "assert" in read_text(rust) and "pub fn run()" in read_text(rust)
    assert any(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(read_text(python))))

    cargo = read_text(REPO_ROOT / "crates" / "antecedent" / "Cargo.toml")
    assert f'name = "{example}"\npath = "../../examples/rust/{example}.rs"' in cargo
    suite = read_text(REPO_ROOT / "crates" / "antecedent" / "tests" / "examples_run.rs")
    assert f'#[path = "../../../examples/rust/{example}.rs"]' in suite
    assert f"{example}::run().unwrap();" in suite

    entries = {e["id"]: e for e in load_toml(REPO_ROOT / "examples" / "catalog.toml")["example"]}
    assert entries[example]["rust"] == f"examples/rust/{example}.rs"
    assert entries[example]["python"] == f"examples/python/{example}.py"


@pytest.mark.parametrize("guide", sorted(COMPARED))
def test_the_page_states_overlap_and_differences_at_the_pinned_version(guide):
    function, example = COMPARED[guide]
    text = read_text(GUIDES / f"{guide}.md")
    section = _section(text, "Differences from DoWhy")
    assert f"DoWhy {PIN['version']}" in section
    assert "parity/baselines/dowhy.toml" in section
    assert "python/tests/test_dowhy_comparison.py" in section
    assert f"`{function}" in section
    assert "Overlapping semantics" in section and "Where they differ" in section
    examples = _section(text, "Examples")
    assert f"examples/rust/{example}.rs" in examples
    assert f"examples/python/{example}.py" in examples
    # The examples the page points to run in both languages' example suites.
    _assert_example_is_registered(example)


def test_every_recorded_comparison_is_checked_by_an_existing_test():
    source = read_text(COMPARISON_TEST)
    defined = {n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)}
    recorded = PIN["comparison"]
    assert PIN["comparison_test"] == "python/tests/test_dowhy_comparison.py"
    assert len({c["id"] for c in recorded}) == len(recorded)
    for comparison in recorded:
        assert comparison["checked_by"] in defined, comparison["id"]
        assert comparison["function"].startswith("dowhy.gcm."), comparison["id"]
        assert comparison["tolerance"], comparison["id"]
    # Each guide's function is one a recorded comparison covers.
    covered = {c["function"].removeprefix("dowhy.") for c in recorded}
    for function, _ in COMPARED.values():
        assert function in covered, function


def test_the_comparison_module_skips_without_dowhy_and_computes_its_references():
    source = read_text(COMPARISON_TEST)
    assert 'pytest.importorskip("dowhy"' in source
    assert 'PIN["version"]' in source
    # No reference value is pasted in: every DoWhy number comes from a call at test time.
    for call in (
        "gcm.interventional_samples",
        "gcm.counterfactual_samples",
        "gcm.intrinsic_causal_influence",
        "falsify_graph",
        "gcm.auto.assign_causal_mechanisms",
    ):
        assert call in source, call


@pytest.mark.parametrize("example", EXAMPLES)
def test_each_example_exists_asserts_and_is_registered(example):
    _assert_example_is_registered(example)
