"""Every native a trade-off decision or a preference learner reaches runs through its compiled
program family, and none reaches a design-crate kernel directly.

The Python modules reach these natives through function-local imports, which the compiled-path
gate does not follow, so this reads the native sources: each deciding or learning native calls
its family's compile entry, and no native names a kernel entry or the retired hidden seam.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NATIVES = (
    ROOT / "python" / "src" / "preferences_api.rs",
    ROOT / "python" / "src" / "preferences_api" / "learning.rs",
)

# Native (or native method) -> the compile entry of the family it must run through, or the one
# helper that compiles and executes it.
ROUTED = {
    "decide_tradeoff": "compile_tradeoff_decision",
    "decide_preferences_joint": "decide_tradeoff",
    "decide_preferences_means": "decide_tradeoff",
    "export_preference_decision": "compile_tradeoff_decision",
    "decide_preferences_structures": "compile_tradeoff_structural_decision",
    "learn_preference_set": "compile_compatible_set",
    "accept_preferences_compatible_set": "compile_compatible_set",
    "relearn_compatible_set": "compile_compatible_set_relearn",
    "estimate_preference_choice_model": "compile_choice_model",
    "accept_preferences_choice_model": "compile_choice_model",
    "relearn_choice_model": "compile_choice_model_relearn",
}

# Kernel entries that decide or learn; a native reaches them only through a family.
KERNELS = (
    "decide_preferences_with_context",
    "decide_preferences_on_means_with_context",
    "decide_preferences_structural_with_context",
    "decide_across_structures_with_context",
    "decide_with_accepted_with_context",
    "learn_compatible_set_with_context",
    "learn_compatible_set_accepted_with_context",
    "estimate_choice_model_with_context",
    "estimate_accepted_with_context",
    "relearn_compatible_set_with_context",
    "relearn_estimate_with_context",
    "PreferenceDecisionArtifact::evaluate",
    "adapt_claims",
)


def _source() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in NATIVES)


def _body(source: str, name: str) -> str:
    """The text of `fn name` up to the next item at the same indentation."""
    match = re.search(rf"^(\s*)fn {name}\b", source, re.M)
    assert match is not None, f"no native named {name}"
    indent = match.group(1)
    rest = source[match.end() :]
    end = re.search(rf"^{indent}(?:#\[|fn |pub|impl|}})", rest, re.M)
    return rest[: end.start()] if end else rest


def test_the_hidden_seam_is_gone():
    assert "preference_decision::bridge" not in _source()


@pytest.mark.parametrize("kernel", KERNELS)
def test_no_native_reaches_a_kernel_directly(kernel):
    assert kernel not in _source(), f"{kernel} is reached without its compiled family"


@pytest.mark.parametrize(("native", "entry"), sorted(ROUTED.items()))
def test_each_deciding_or_learning_native_runs_its_family(native, entry):
    assert re.search(rf"\b{entry}\(", _body(_source(), native)), f"{native} does not call {entry}"
