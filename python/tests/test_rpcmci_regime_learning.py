"""Compiled regime discovery from simulated raw series, with verified replay."""

from __future__ import annotations

import numpy as np
import pytest

from antecedent import Execution
from antecedent.discovery import RPCMCI
from antecedent.errors import CausalValueError


def _series(n: int, change: int, coefficient: float, *, seed: int) -> dict[str, np.ndarray]:
    noise = np.random.default_rng(seed).normal(size=(n, 2))
    x, y = noise.copy().T
    for row in range(1, n):
        if row < change:
            y[row] = 1.15 * x[row - 1] + 0.19 * noise[row, 1]
        else:
            y[row] = coefficient * x[row - 1] + noise[row, 1]
    return {"x": x, "y": y}


def _lagged_link(edges: list[tuple[str, int, str, str, int, str]]) -> bool:
    return ("x", 1, "tail", "y", 0, "arrow") in edges or ("y", 0, "arrow", "x", 1, "tail") in edges


def test_rpcmci_learns_regimes_and_known_graph_then_verifies_raw_series_and_seed() -> None:
    n, change = 540, 248
    data = _series(n, change, 0.0, seed=312)
    # Add a level shift to the receiving equation while keeping the late graph empty.
    data["y"][change:] += 3.0
    config = RPCMCI(n_regimes=2, min_regime_len=55, starts=4, alpha=0.002, switch_penalty=18.0)
    fit = config.run(data, execution=Execution(seed=22, threads=1))
    labels = np.asarray(fit.assignments)
    agreement = float(np.mean(labels == (np.arange(n) >= change)))
    assert max(agreement, 1 - agreement) > 0.92
    early, late = labels[90], labels[475]
    assert early != late
    assert _lagged_link(fit.marked_edges[early])
    assert not _lagged_link(fit.marked_edges[late])
    assert fit.seed == 22
    assert fit.assignment_agreement is not None
    assert len(fit.graph_agreement) == 2 and all(x is not None for x in fit.graph_agreement)
    assert any(code == "rpcmci.post_selection" for code, _ in fit.diagnostics)
    assert config.replay(data, fit, execution=Execution(seed=22, threads=1)).marked_edges == (
        fit.marked_edges
    )
    with pytest.raises(CausalValueError, match="seed"):
        config.replay(data, fit, execution=Execution(seed=23, threads=1))
    changed = {"x": data["x"].copy(), "y": data["y"]}
    changed["x"][0] += 0.25
    with pytest.raises(CausalValueError, match="data_identity"):
        config.replay(changed, fit)


def test_rpcmci_bic_prefers_a_single_regime_with_weak_departure_and_skips_short_counts() -> None:
    # Full series is close to stationary. The later coefficient changes only slightly;
    # keep the innovation variance identical across the candidate boundary.
    n, change = 172, 111
    noise = np.random.default_rng(8092).normal(size=(n, 2))
    x = noise[:, 0]
    y = noise[:, 1].copy()
    y[1:] += np.where(np.arange(1, n) < change, 0.74, 0.75) * x[:-1]
    result = RPCMCI(max_regimes=4, min_regime_len=62, starts=4, switch_penalty=25.0).run(
        {"x": x, "y": y}, execution=Execution(seed=14, threads=1)
    )
    assert result.n_regimes == 1
    assert result.infeasible_counts == [3, 4]
    assert result.assignments == [0] * n
    assert result.converged
    assert result.graph_agreement == [None]


def test_portable_shared_series_replays_in_fresh_process_and_reviews_each_graph(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    from antecedent import ResourceLimits
    from antecedent import composition_bundle as cb
    from antecedent.outcomes import require_success
    from antecedent.rpcmci_portable import LearnedRpcmciRefusal, consume_learned_rpcmci

    fixture = json.loads(
        (Path(__file__).resolve().parents[2] / "conformance/discovery/rpcmci_shared_portable.json").read_text()
    )
    config = RPCMCI(n_regimes=2, min_regime_len=48, starts=4, alpha=0.01, fdr=False, switch_penalty=18.0)
    series = dict(zip(fixture["names"], fixture["columns"], strict=True))
    policy = Execution(seed=fixture["seed"], threads=1)
    fit = config.run_portable(series, execution=policy)
    report = fit.report
    labels = np.asarray(report["assignments"])
    agreement = np.mean(labels == (np.arange(len(labels)) >= fixture["split"]))
    assert max(agreement, 1 - agreement) >= 0.94
    assert report["seed"] == fixture["seed"] and report["selected_count"] == 2
    first, second = labels[48], labels[330]
    assert first != second
    raw = config.run(series, execution=policy)
    assert raw.program_identity == report["program_identity"]
    assert raw.data_identity == report["data_identity"]
    assert raw.assignments == report["assignments"]
    assert _lagged_link(raw.marked_edges[first])
    assert not _lagged_link(raw.marked_edges[second])
    assert ("y", 1, "tail", "x", 0, "arrow") in raw.marked_edges[second] or (
        "x", 0, "arrow", "y", 1, "tail"
    ) in raw.marked_edges[second]
    assert require_success(fit.accept_regime(int(first))).algorithm_id == "rpcmci"
    assert require_success(fit.accept_regime(int(second))).algorithm_id == "rpcmci"
    with pytest.raises(LearnedRpcmciRefusal, match="choose a regime"):
        require_success(fit.accept_regime(42))

    bundle = cb.Bundle.builder().add_artifact(fit).build()
    verified = require_success(cb.consume_bundle(bytes(bundle.export()), expected_identity=bundle.identity))
    assert verified.all_replayed
    path = tmp_path / "learned-rpcmci.bin"
    from antecedent import load

    artifact = fit.export()
    assert artifact.expectation is not None
    restored = require_success(load(artifact, expected=artifact.expectation))
    assert restored.identity == fit.identity
    path.write_bytes(bytes(artifact))
    fresh = subprocess.run(
        [
            sys.executable, "-c",
            """
import json
from pathlib import Path
import sys
from antecedent.outcomes import require_success
from antecedent.rpcmci_portable import consume_learned_rpcmci
result = require_success(consume_learned_rpcmci(Path(sys.argv[1]).read_bytes(), expected_identity=sys.argv[2]))
print(json.dumps(result.report))
""",
            str(path), fit.identity,
        ],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(fresh.stdout) == report
    with pytest.raises(LearnedRpcmciRefusal, match="resource"):
        require_success(consume_learned_rpcmci(
            fit.export(), expected_identity=fit.identity,
            execution=Execution(limits=ResourceLimits(memory_bytes=1024)),
        ))
