"""PanelFrame pooled PCMCI-family discovery (not JPCMCI+ multi-env)."""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent.outcomes import require_success


def _lag1_unit(n: int = 100, seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.empty(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = 0.55 * x[t - 1] + 0.2 * rng.normal()
    return {"x": x, "y": y}


def test_panel_pooled_pcmci_smoke():
    """Inline discovery on pooled panel data recovers the lag-one effect.

    At ``alpha=0.2`` pooled PCMCI keeps a lag-one self edge on ``x``, so the
    treatment's ancestry is unbounded and unfolding cannot certify. The single-step
    pulse is then identified by adjusting for the treatment's parents (``x[t-2]``)
    and the pooled fit recovers the generating effect 0.55; the exported contract
    names the ``temporal.parent_adjustment`` derivation.
    """
    panel = antecedent.data.panel([_lag1_unit(seed=3), _lag1_unit(seed=4), _lag1_unit(seed=5)])
    result = require_success(
        antecedent.analyze(
            panel,
            graph=antecedent.discovery.discover(
                panel,
                method=antecedent.discovery.PCMCI(max_lag=1, alpha=0.2, fdr=False),
                execution=antecedent.Execution(seed=1),
                accept_discovered=True,
            ),
            query=antecedent.PulseEffect(
                treatment="x",
                outcome="y",
                treatment_lag=1,
                horizon_steps=1,
                active_level=1.0,
            ),
            inference=antecedent.Frequentist(bootstrap=0),
            execution=antecedent.Execution(seed=1),
            refute=False,
        )
    )
    assert result.answer.kind == "point"
    assert result.answer.value == pytest.approx(0.55, abs=0.05)
    assert result.identification.adjustment_set == tuple(["x"])
    product = (
        require_success(antecedent.load(bytes(result.export())))
        .inspect()
        .to_dict()["contract"]["identification_product"]
    )
    assert product["derivation_rules"][0] == "temporal.parent_adjustment"


def test_panel_pooled_rejects_rpcmci():
    panel = antecedent.data.panel([_lag1_unit(seed=3), _lag1_unit(seed=4)])
    with pytest.raises(
        antecedent.errors.CausalUnsupportedError, match="single observation sequence"
    ):
        require_success(
            antecedent.analyze(
                panel,
                graph=antecedent.discovery.discover(
                    panel,
                    method=antecedent.discovery.RPCMCI(max_lag=1, alpha=0.2, fdr=False),
                    execution=antecedent.Execution(seed=1),
                    accept_discovered=True,
                ),
                query=antecedent.PulseEffect(
                    treatment="x",
                    outcome="y",
                    treatment_lag=1,
                    horizon_steps=1,
                    active_level=1.0,
                ),
                inference=antecedent.Frequentist(bootstrap=0),
                execution=antecedent.Execution(seed=1),
                refute=False,
            )
        )


def _pooled(units):
    names = list(units[0])
    columns = [np.concatenate([u[n] for u in units]) for n in names]
    return names, columns, [len(u[names[0]]) for u in units]


def test_native_panel_pcmci_builds_lag_windows_inside_each_unit():
    """The planted lag-one link is found from a panel of short units, and a wrong
    ``unit_lengths`` (rows that do not add up) is a typed refusal, not a guess."""
    from antecedent import _native

    units = [_lag1_unit(n=40, seed=s) for s in range(10, 16)]
    names, columns, lengths = _pooled(units)
    result = _native.discover_pcmci(
        names, columns, max_lag=1, alpha=0.05, fdr=False, seed=1, unit_lengths=lengths
    )
    links = {(link.source, link.source_lag, link.target) for link in result.links}
    assert ("x", 1, "y") in links
    with pytest.raises(antecedent.errors.CausalUnsupportedError) as info:
        _native.discover_pcmci(names, columns, unit_lengths=[len(columns[0]) + 1])
    assert info.value.reason_code == "invalid_argument"
    with pytest.raises(antecedent.errors.CausalUnsupportedError) as info:
        _native.discover_pcmci(
            names, columns, unit_lengths=lengths, weights=[1.0] * len(columns[0])
        )
    assert info.value.reason_code == "option_not_applicable"


def test_panel_dbn_posterior_pools_per_unit_rows():
    panel = antecedent.data.panel([_lag1_unit(n=60, seed=s) for s in range(20, 24)])
    posterior = antecedent.discovery.DbnPosterior(max_lag=1).run(
        panel, execution=antecedent.Execution(seed=1)
    )
    assert posterior.n_graphs >= 1
    # Lag-one edge marginals in (source, target) order x->x, x->y, y->x, y->y: the planted
    # x[t-1] -> y[t] link is certain and the three absent links are improbable.
    x_x, x_y, y_x, y_y = posterior.lagged_edge_marginals
    assert x_y > 0.99
    assert max(x_x, y_x, y_y) < 0.3
