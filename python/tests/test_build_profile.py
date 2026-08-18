"""The native extension under test must be an optimized (release) build.

A debug-profile ``antecedent._native`` returns bit-identical estimates at
roughly 50x the wall time, so no numeric test can catch it — this gate is the
only thing that fails loudly. ``[tool.maturin] profile = "release"`` makes
every standard build path (wheels, ``maturin develop``, uv editable rebuilds)
compile optimized; if this test fails, that pin was removed or bypassed.

Deliberately testing a debug build? Opt out with
``ANTECEDENT_ALLOW_DEBUG_NATIVE=1``.
"""

import os

import pytest
from antecedent import _native


def test_native_extension_is_an_optimized_build():
    if os.environ.get("ANTECEDENT_ALLOW_DEBUG_NATIVE") == "1":
        pytest.skip("ANTECEDENT_ALLOW_DEBUG_NATIVE=1: debug extension explicitly allowed")
    assert getattr(_native, "__build_optimized__", False) is True, (
        "antecedent._native was compiled in Cargo's debug profile (or predates "
        "the __build_optimized__ flag). Rebuild it optimized — e.g. "
        "`uv sync --reinstall-package antecedent` or `maturin develop --release` "
        "— or set ANTECEDENT_ALLOW_DEBUG_NATIVE=1 if a debug build is intended."
    )
