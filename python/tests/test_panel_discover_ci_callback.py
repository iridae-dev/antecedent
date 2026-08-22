"""`ci=` (a Python callable CI test) reaching the PanelFrame discovery path.

`analyze(panel, discovery=..., ...)` for `PanelFrame` routes through
`_handle_panel_frame` (`antecedent/_analyze.py`) to the native
`analyze_panel_discover`. That native function previously had no `ci` parameter
at all, so a caller-supplied CI test was accepted by `PCMCI`/`PCMCIPlus`/
`LPCMCI`/`JPCMCIPlus` (which all carry a `ci` field) and then silently dropped
before crossing into Rust — the panel path always ran the default `parcorr`
test regardless of what was requested.

These tests assert the callable actually runs during panel discovery (a call
counter is incremented), not merely that passing `ci=` is accepted without
error — the latter would also have passed against the broken code.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("antecedent")
import antecedent


def _lag1_unit(n: int = 100, seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = 0.55 * x[t - 1] + 0.2 * rng.normal()
    return {"x": x, "y": y}


def _counting_ci(calls: list[int]):
    """Batch CI callback: always-independent, records how many queries it saw."""

    def _ci(columns, queries):
        calls.append(len(queries))
        return [(0.0, 1.0) for _ in queries]

    return _ci


def test_panel_discover_pcmci_ci_callback_invoked():
    """PCMCI branch of `_handle_panel_frame` must forward `ci=` to the native call.

    The counting callback always reports independence, so discovery finds no
    edges at all; PCMCI's review is DAG-shaped (`accept_temporal_graph_review`
    in `python/src/lib.rs`) and accepts an empty graph unconditionally, but the
    requested x->y pulse can then legitimately be unidentifiable against that
    empty graph. Either a clean estimate or `CausalIdentifyError` proves the
    callback reached native discovery; any other exception is a real wiring
    break, not a "review/ID may fail closed" outcome.
    """
    calls: list[int] = []
    panel = antecedent.data.panel([_lag1_unit(seed=3), _lag1_unit(seed=4), _lag1_unit(seed=5)])
    try:
        antecedent.analyze(
            panel,
            discovery=antecedent.discovery.PCMCI(
                max_lag=1, alpha=0.2, fdr=False, ci=_counting_ci(calls)
            ),
            query=antecedent.PulseEffect(
                treatment="x",
                outcome="y",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            bootstrap=0,
            seed=1,
            refute=False,
        )
    except (
        antecedent.errors.CausalIdentifyError,
        antecedent.errors.CausalCompileError,
    ) as exc:
        assert str(exc)
    assert calls, "ci callback was never invoked on the PanelFrame + PCMCI discovery path"
    assert sum(calls) > 0


def test_panel_discover_jpcmci_plus_ci_callback_invoked():
    """JPCMCIPlus branch of `_handle_panel_frame` must forward `ci=` to the native call.

    The counting callback always reports independence, so discovery finds no
    edges at all. JPCMCIPlus's review is CPDAG-shaped
    (`accept_temporal_cpdag_review` in `python/src/lib.rs`); an empty skeleton has
    no undirected marks to block on, so it accepts unconditionally too, but the
    requested x->y pulse can then legitimately be unidentifiable against that
    empty graph. A clean estimate proves the callback reached native discovery;
    so does any of the four exceptions caught below, each a legitimate
    fail-closed outcome rather than a wiring break: `CausalReviewError` (a
    leftover undirected/circle mark blocked acceptance), `CausalIdentifyError`
    (the pulse is unidentifiable against the discovered graph),
    `CausalCompileError` (a compile-stage failure), or `CausalUnsupportedError`
    (this discovery/query/graph combination is a support-matrix refusal —
    `SupportRefusal::Refused` / `NotApplicable`, surfaced via `Support{id,
    message}` in `python/src/lib.rs`). Any other exception is a real wiring
    break.
    """
    calls: list[int] = []
    panel = antecedent.data.panel([_lag1_unit(seed=3), _lag1_unit(seed=4), _lag1_unit(seed=5)])
    try:
        antecedent.analyze(
            panel,
            discovery=antecedent.discovery.JPCMCIPlus(
                max_lag=1, alpha=0.2, fdr=False, ci=_counting_ci(calls)
            ),
            query=antecedent.PulseEffect(
                treatment="x",
                outcome="y",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            bootstrap=0,
            seed=1,
            refute=False,
        )
    except (
        antecedent.errors.CausalReviewError,
        antecedent.errors.CausalIdentifyError,
        antecedent.errors.CausalCompileError,
        antecedent.errors.CausalUnsupportedError,
    ) as exc:
        assert str(exc)
        if isinstance(exc, antecedent.errors.CausalReviewError):
            assert exc.kind
            assert exc.hint
    assert calls, "ci callback was never invoked on the PanelFrame + JPCMCIPlus discovery path"
    assert sum(calls) > 0
