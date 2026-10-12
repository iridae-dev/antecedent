"""Tests for the shared `estimator_config` dict parser (P5b).

`estimator_config` is a single table-driven kwarg that replaces the old pattern of adding a
loose Python kwarg per estimator setter. These tests exercise it directly against the native
`PreparedAnalysis.prepare` handle (through `_native_ate.native_ate`), which still takes the
loose `rd.sharp` triple that the facade refuses: a config that measurably changes a result,
unknown-key and wrong-estimator-key errors, a wrong-type error, the `estimator_config=None`
no-drift guarantee, and the `rd.sharp` triple expressed through `estimator_config`.
"""

from __future__ import annotations

import re

import antecedent
import numpy as np
import pytest
from antecedent.estimators import GlmOptions, PropensityWeighting

from _native_ate import native_ate
from _scm import confounded_columns as _confounded_data
from _scm import public_scm as _public_scm

# A sharp design as a graph: the running variable is the treatment's only cause.
_RD_GRAPH = [("r", "t"), ("t", "y"), ("r", "y")]


def _rd_data(seed: int = 25, n: int = 1500):
    """Sharp RD fixture: running variable `r`, cutoff at 0."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(-2.0, 2.0, size=n)
    t = (r >= 0.0).astype(np.float64)
    y = 1.0 + 2.0 * t + 0.3 * r + rng.normal(scale=0.2, size=n)
    names = ["t", "y", "r"]
    columns = [t, y, r]
    return names, columns


# --- A config that changes a result -------------------------------------------------------


def test_cluster_se_kind_changes_se_but_not_the_point_estimate():
    names, columns, edges = _confounded_data()
    n = len(columns[0])
    cluster_ids = [int(x) for x in np.random.default_rng(11).integers(0, 4, size=n)]

    baseline = native_ate(names, columns, edges, "t", "y", refute=False, bootstrap=0, seed=1)
    configured = native_ate(
        names,
        columns,
        edges,
        "t",
        "y",
        refute=False,
        bootstrap=0,
        seed=1,
        estimator_config={"se_kind": "cluster", "cluster_ids": cluster_ids},
    )

    # se_kind only changes the standard-error formula, never the point estimate.
    assert configured.ate == pytest.approx(baseline.ate)
    assert configured.se_analytic != pytest.approx(baseline.se_analytic)
    assert np.isfinite(configured.se_analytic)


def test_bootstrap_replicates_via_estimator_config_matches_ambient_bootstrap_kwarg():
    """Omitting `bootstrap_replicates` from estimator_config falls back to the ambient
    `bootstrap=` kwarg — configuring a *different* key (se_kind) shouldn't silently reset it
    to the estimator struct's own 200-replicate default."""
    names, columns, edges = _confounded_data()

    unconfigured = native_ate(names, columns, edges, "t", "y", refute=False, bootstrap=64, seed=5)
    configured = native_ate(
        names,
        columns,
        edges,
        "t",
        "y",
        refute=False,
        bootstrap=64,
        seed=5,
        estimator_config={"se_kind": "homoskedastic"},
    )

    assert configured.ate == pytest.approx(unconfigured.ate)
    assert configured.se_analytic == pytest.approx(unconfigured.se_analytic)
    # If the ambient `bootstrap=64` kwarg weren't honored as the configured estimator's
    # replicate count, this would silently fall back to the struct's own 200-replicate
    # default instead, and the two bootstrap SEs would very likely disagree.
    assert configured.se_bootstrap is not None
    assert unconfigured.se_bootstrap is not None
    assert configured.se_bootstrap == pytest.approx(unconfigured.se_bootstrap)


# --- Unknown key / wrong-estimator key ------------------------------------------------------


def test_unknown_key_names_the_key():
    names, columns, edges = _confounded_data()
    with pytest.raises(ValueError, match="unknown estimator_config key") as exc_info:
        native_ate(
            names,
            columns,
            edges,
            "t",
            "y",
            refute=False,
            bootstrap=0,
            estimator_config={"totally_bogus_key": 1},
        )
    assert "totally_bogus_key" in str(exc_info.value)


def test_key_belonging_to_a_different_estimator_names_it():
    names, columns, edges = _confounded_data()
    # `glm_options` is valid for propensity/AIPW/glm.adjustment estimators, not the default
    # linear.adjustment.ate — this should name linear.adjustment.ate and point at an owner.
    with pytest.raises(ValueError, match="belongs to estimator") as exc_info:
        native_ate(
            names,
            columns,
            edges,
            "t",
            "y",
            refute=False,
            bootstrap=0,
            estimator_config={"glm_options": {"max_iter": 10}},
        )
    message = str(exc_info.value)
    assert "glm_options" in message
    assert "linear.adjustment.ate" in message


def test_estimator_config_unsupported_for_estimators_with_no_config_surface():
    names, columns, edges = _confounded_data()
    with pytest.raises(ValueError, match="estimator_config is not supported for estimator"):
        native_ate(
            names,
            columns,
            edges,
            "t",
            "y",
            refute=False,
            bootstrap=0,
            estimator="bayesian.gcomp",
            estimator_config={"n_draws": 100},
        )


# --- Wrong value type ------------------------------------------------------------------------


def test_wrong_value_type_raises():
    names, columns, edges = _confounded_data()
    with pytest.raises(ValueError, match="must be a non-negative int"):
        native_ate(
            names,
            columns,
            edges,
            "t",
            "y",
            refute=False,
            bootstrap=0,
            estimator_config={"bootstrap_replicates": "not-an-int"},
        )


def test_wrong_cluster_ids_type_raises():
    names, columns, edges = _confounded_data()
    with pytest.raises(ValueError, match="must be a list of non-negative ints"):
        native_ate(
            names,
            columns,
            edges,
            "t",
            "y",
            refute=False,
            bootstrap=0,
            estimator_config={"se_kind": "cluster", "cluster_ids": "not-a-list"},
        )


# --- estimator_config=None is a strict no-op ------------------------------------------------


def test_estimator_config_none_matches_omitting_it_entirely():
    names, columns, edges = _confounded_data()

    omitted = native_ate(names, columns, edges, "t", "y", refute=False, bootstrap=0, seed=3)
    explicit_none = native_ate(
        names, columns, edges, "t", "y", refute=False, bootstrap=0, seed=3, estimator_config=None
    )
    empty_dict = native_ate(
        names, columns, edges, "t", "y", refute=False, bootstrap=0, seed=3, estimator_config={}
    )

    assert explicit_none.ate == omitted.ate
    assert explicit_none.se_analytic == omitted.se_analytic
    assert empty_dict.ate == omitted.ate
    assert empty_dict.se_analytic == omitted.se_analytic


# --- rd.sharp triple through estimator_config ------------------------------------------------


def test_rd_triple_via_estimator_config_matches_loose_kwargs():
    names, columns = _rd_data()

    via_loose_kwargs = native_ate(
        names,
        columns,
        _RD_GRAPH,
        "t",
        "y",
        estimator="rd.sharp",
        identifier="rd.sharp",
        running_variable="r",
        cutoff=0.0,
        bandwidth=1.5,
        refute=False,
        bootstrap=0,
        seed=26,
    )
    via_estimator_config = native_ate(
        names,
        columns,
        _RD_GRAPH,
        "t",
        "y",
        estimator="rd.sharp",
        identifier="rd.sharp",
        refute=False,
        bootstrap=0,
        seed=26,
        estimator_config={"running_variable": "r", "cutoff": 0.0, "bandwidth": 1.5},
    )

    assert via_estimator_config.ate == pytest.approx(via_loose_kwargs.ate)
    assert via_estimator_config.se_analytic == pytest.approx(via_loose_kwargs.se_analytic)
    assert abs(via_estimator_config.ate - 2.0) < 0.35


def test_rd_triple_conflict_between_loose_and_estimator_config_raises():
    names, columns = _rd_data()
    with pytest.raises(ValueError, match="conflicting rd.sharp"):
        native_ate(
            names,
            columns,
            _RD_GRAPH,
            "t",
            "y",
            estimator="rd.sharp",
            identifier="rd.sharp",
            running_variable="r",
            cutoff=0.0,
            bandwidth=1.5,
            refute=False,
            bootstrap=0,
            seed=26,
            estimator_config={"bandwidth": 2.0},
        )


def _rd_heteroskedastic_data(seed: int = 31, n: int = 1500):
    """Sharp RD fixture whose outcome noise grows along the running variable."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(-2.0, 2.0, size=n)
    t = (r >= 0.0).astype(np.float64)
    y = 1.0 + 2.0 * t + 0.3 * r + rng.normal(size=n) * (0.1 + 0.4 * np.abs(r))
    return ["t", "y", "r"], [t, y, r]


def _rd_fit(names, columns, se_kind=None):
    config = {"running_variable": "r", "cutoff": 0.0, "bandwidth": 1.5}
    if se_kind is not None:
        config["se_kind"] = se_kind
    return native_ate(
        names,
        columns,
        _RD_GRAPH,
        "t",
        "y",
        estimator="rd.sharp",
        identifier="rd.sharp",
        refute=False,
        bootstrap=0,
        seed=26,
        estimator_config=config,
    )


def test_rd_se_kind_defaults_to_hc1_and_selects_the_analytic_se():
    names, columns = _rd_heteroskedastic_data()
    default = _rd_fit(names, columns)
    fits = {
        kind: _rd_fit(names, columns, kind)
        for kind in ("hc1", "homoskedastic", "hc0", "hc2", "hc3")
    }
    # Omitting se_kind is the HC1 default, bit for bit.
    assert default.se_analytic == fits["hc1"].se_analytic
    # se_kind never moves the point estimate.
    for fit in fits.values():
        assert fit.ate == default.ate
    # Each kind reaches the estimator: the SEs are pairwise distinct.
    ses = [fit.se_analytic for fit in fits.values()]
    assert len({round(se, 12) for se in ses}) == len(ses)
    # Standard orderings of the residual sandwiches: HC0 < HC1 and HC2 < HC3.
    assert fits["hc0"].se_analytic < fits["hc1"].se_analytic
    assert fits["hc2"].se_analytic < fits["hc3"].se_analytic


@pytest.mark.parametrize("kind", ["cluster", "multiway", "newey_west", "panel_cluster_hac", "hc4"])
def test_rd_se_kind_rejects_kinds_that_do_not_apply(kind):
    names, columns = _rd_data()
    with pytest.raises(ValueError, match="not supported for rd.sharp"):
        _rd_fit(names, columns, kind)


def test_rd_key_used_without_rd_sharp_estimator_reports_the_owner():
    names, columns, edges = _confounded_data()
    # `running_variable` is only a valid estimator_config key for rd.sharp; against the
    # default linear.adjustment.ate it should report rd.sharp as the owning estimator.
    with pytest.raises(ValueError, match="belongs to estimator") as exc_info:
        native_ate(
            names,
            columns,
            edges,
            "t",
            "y",
            refute=False,
            bootstrap=0,
            estimator_config={"running_variable": "z"},
        )
    assert "rd.sharp" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Public-facade coverage. The tests above drive the native prepared handle directly;
# these pin that `estimator_config=` is actually reachable from `antecedent.analyze`,
# which is the only spelling documented to users.
# ---------------------------------------------------------------------------


def test_public_analyze_accepts_estimator_config_and_none_is_identical():
    data, graph = _public_scm()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    omitted = antecedent.analyze(data, graph=graph, query=query, seed=1)
    explicit_none = antecedent.analyze(
        data, graph=graph, query=query, seed=1, estimator_config=None
    )
    assert explicit_none.estimate.ate == omitted.estimate.ate
    assert explicit_none.estimate.se_analytic == omitted.estimate.se_analytic


def test_public_analyze_estimator_config_changes_the_standard_error():
    data, graph = _public_scm()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    default = antecedent.analyze(data, graph=graph, query=query, seed=1)
    clustered = antecedent.analyze(
        data,
        graph=graph,
        query=query,
        seed=1,
        estimator="linear.adjustment.ate",
        estimator_config={
            "se_kind": "cluster",
            "cluster_ids": [i % 20 for i in range(len(data["z"]))],
        },
    )
    assert clustered.estimate.se_analytic != default.estimate.se_analytic
    assert clustered.estimate.se_analytic > 0.0


@pytest.mark.parametrize(
    ("config", "needle"),
    [
        ({"nonsense": 1}, "nonsense"),
        ({"caliper": 0.1}, "caliper"),
        ({"se_kind": 5}, "se_kind"),
    ],
    ids=["unknown-key", "key-of-another-estimator", "wrong-value-type"],
)
def test_public_analyze_rejects_bad_estimator_config(config, needle):
    data, graph = _public_scm()
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    with pytest.raises(Exception, match=re.escape(needle)):
        antecedent.analyze(
            data,
            graph=graph,
            query=query,
            seed=1,
            estimator="linear.adjustment.ate",
            estimator_config=config,
        )


# ---------------------------------------------------------------------------
# estimator_config on the discovery= path (regression for the bug where the discovery
# route silently dropped `estimator_config`, so a configured estimator silently fell
# back to its defaults whenever `discovery=` was used instead of `graph=`).
#
# The reliable trick: `GlmOptions(max_iter=3)` starves propensity.weighting's
# GLM IRLS fit of iterations on `_confounded_data()` (a near-separable binary
# treatment threshold — exactly the pathological case IRLS struggles to
# converge on quickly). With the config honored this raises
# `CausalEstimateError` ("GLM IRLS did not converge"); with it silently
# dropped, the call falls back to the estimator's own default `max_iter` and
# succeeds, returning a plausible-but-wrong point estimate instead of failing
# loudly.
# ---------------------------------------------------------------------------


def test_estimator_config_reaches_estimator_on_discovery_path():
    names, columns, edges = _confounded_data()
    data = dict(zip(names, columns, strict=True))
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    cfg = PropensityWeighting(glm_options=GlmOptions(max_iter=3))

    # Reference case: on the graph= path, the config is already known to reach
    # the estimator and force non-convergence.
    with pytest.raises(antecedent.errors.CausalEstimateError, match="did not converge"):
        antecedent.analyze(
            data, graph=edges, query=query, estimator=cfg, refute=False, bootstrap=0, seed=1
        )

    # Regression case: the same config object, reached through discovery=
    # instead of graph=, must raise identically — not silently fall back to
    # estimator defaults and succeed.
    with pytest.raises(antecedent.errors.CausalEstimateError, match="did not converge"):
        antecedent.analyze(
            data,
            discovery=antecedent.discovery.LiNGAM(),
            accept_discovered=True,
            query=query,
            estimator=cfg,
            refute=False,
            bootstrap=0,
            seed=1,
        )


def test_estimator_config_dict_reaches_estimator_on_discovery_path():
    """Same regression as above, but through the raw `estimator_config=` dict
    spelling (rather than a typed `antecedent.estimators` dataclass), to pin the
    fix at the `_analyze.py` dispatch layer independent of `estimator._wire()`.
    """
    names, columns, edges = _confounded_data()
    data = dict(zip(names, columns, strict=True))
    query = antecedent.AverageEffect(treatment="t", outcome="y")

    with pytest.raises(antecedent.errors.CausalEstimateError, match="did not converge"):
        antecedent.analyze(
            data,
            discovery=antecedent.discovery.LiNGAM(),
            accept_discovered=True,
            query=query,
            estimator="propensity.weighting",
            estimator_config={"glm_options": {"max_iter": 3}},
            refute=False,
            bootstrap=0,
            seed=1,
        )
