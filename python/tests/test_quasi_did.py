from __future__ import annotations

import numpy as np
import pytest
from antecedent.errors import CausalValueError
from antecedent.quasi import (
    DifferenceInDifferences,
    PanelDifferenceInDifferences,
    StaggeredAdoption,
    estimate_did,
    estimate_group_time_att,
    estimate_panel_did,
    estimate_staggered_event_study,
)


def test_native_2x2_did_recovers_known_treatment_effect_and_discloses_scope():
    treated = np.array([False, False, True, True] * 4)
    post = np.array([False, True, False, True] * 4)
    outcome = 10.0 + 3.0 * post + 4.0 * (treated & post)
    result = estimate_did(
        {"y": outcome, "group": treated, "after": post},
        DifferenceInDifferences("y", "group", "after"),
    )
    assert result.estimate == pytest.approx(4.0)
    assert result.uncertainty == "point_only"
    assert result.support_status == "unlicensed_point_utility"
    assert "parallel_untreated_trends" in result.assumptions


def test_did_refuses_empty_cells_and_nonbinary_design_columns():
    query = DifferenceInDifferences("y", "group", "after")
    with pytest.raises(CausalValueError, match="all four"):
        estimate_did(
            {"y": [1.0, 3.0, 2.0], "group": [False, False, True], "after": [False, True, True]},
            query,
        )
    with pytest.raises(CausalValueError, match="bool or encoded as 0/1"):
        estimate_did(
            {"y": [1.0, 2.0, 3.0, 4.0], "group": [0, 0, 2, 2], "after": [0, 1, 0, 1]},
            query,
        )


def test_native_balanced_panel_did_recovers_known_treatment_effect():
    subjects = [f"s{i}" for i in range(8) for _ in range(2)]
    treated = [i < 4 for i in range(8) for _ in range(2)]
    post = [period for _ in range(8) for period in (False, True)]
    # Stable subject effects cancel in the within-subject change.
    outcome = [float(10 + int(subject[1:]) + (3 if is_post else 0) + (4 if is_treated and is_post else 0))
               for subject, is_treated, is_post in zip(subjects, treated, post, strict=True)]
    result = estimate_panel_did(
        {"y": outcome, "id": subjects, "group": treated, "after": post},
        PanelDifferenceInDifferences("y", "id", "group", "after"),
    )
    assert result.estimate == pytest.approx(4.0)
    assert result.standard_error == pytest.approx(0.0)
    assert result.treated_subjects == 4
    assert result.control_subjects == 4
    assert result.clusters == 8
    assert result.uncertainty == "cluster_robust_standard_error_no_interval"
    assert result.support_status == "unlicensed_point_utility"
    assert "complete_pre_post_panel" in result.assumptions
    import antecedent
    assert result == antecedent.analyze(
        {"y": outcome, "id": subjects, "group": treated, "after": post},
        query=PanelDifferenceInDifferences("y", "id", "group", "after"),
    ).panel_did


def test_panel_did_clustered_standard_error_matches_known_influence_variance():
    changes = [2., 4., 6., 8., 0., 2., 4., 6.]
    subjects = [f"s{i}" for i in range(8) for _ in range(2)]
    treated = [i < 4 for i in range(8) for _ in range(2)]
    post = [p for _ in range(8) for p in (False, True)]
    outcome = [float(i * 10 + (changes[i] if p else 0.)) for i in range(8) for p in (False, True)]
    result = estimate_panel_did(
        {"y": outcome, "id": subjects, "group": treated, "after": post},
        PanelDifferenceInDifferences("y", "id", "group", "after"),
    )
    # CR1 factor 8/7 times sum of squared cluster influence contributions.
    assert result.estimate == pytest.approx(2.0)
    assert result.standard_error == pytest.approx((20.0 / 7.0) ** 0.5)


def test_direct_panel_did_inherits_retained_interval_and_support_status():
    import antecedent

    ids, groups, periods, outcomes = [], [], [], []
    for group in (False, True):
        for cluster in range(30):
            for after in (False, True):
                ids.append(f"subject-{group}-{cluster}")
                groups.append(group)
                periods.append(after)
                outcomes.append(cluster * 0.1 + 2.0 * int(group and after) if after else 0.0)
    data = {"y": outcomes, "id": ids, "group": groups, "after": periods}
    query = PanelDifferenceInDifferences("y", "id", "group", "after")
    direct = estimate_panel_did(data, query)
    assert direct == antecedent.analyze(data, query=query).panel_did
    assert direct.interval_95 is not None
    assert direct.support_status == "licensed"


def test_panel_did_refuses_too_few_clusters_or_cluster_changes_within_subject():
    ids = [f"s{i}" for i in range(4) for _ in range(2)]
    treated = [i < 2 for i in range(4) for _ in range(2)]
    post = [p for _ in range(4) for p in (False, True)]
    outcome = [float(i + p) for i in range(4) for p in (0, 1)]
    query = PanelDifferenceInDifferences("y", "id", "group", "after")
    with pytest.raises(CausalValueError, match="at least two clusters in each treatment group"):
        estimate_panel_did(
            {"y": outcome, "id": ids, "group": treated, "after": post,
             "cluster": ["a", "a", "a", "a", "b", "b", "b", "b"]},
            query, cluster="cluster",
        )
    bad_clusters = ["a", "x", "b", "b", "c", "c", "d", "d"]
    with pytest.raises(CausalValueError, match="constant within each subject"):
        estimate_panel_did(
            {"y": outcome, "id": ids, "group": treated, "after": post,
             "cluster": bad_clusters}, query, cluster="cluster",
        )


def test_panel_did_runs_through_retained_public_prepare_and_analyze_routes():
    import antecedent

    ids = [f"s{i}" for i in range(8) for _ in range(2)]
    treated = [i < 4 for i in range(8) for _ in range(2)]
    post = [period for _ in range(8) for period in (False, True)]
    outcome = [float(10 + int(subject[1:]) + 3 * int(after) + 4 * int(group and after))
               for subject, group, after in zip(ids, treated, post, strict=True)]
    data = {"y": outcome, "id": ids, "group": treated, "after": post}
    query = PanelDifferenceInDifferences("y", "id", "group", "after")
    prepared = antecedent.prepare(data, query=query)
    estimate = prepared.estimate()
    assert estimate.panel_did is not None
    assert estimate.panel_did.estimate == pytest.approx(4.0)
    assert estimate.panel_did.standard_error == pytest.approx(0.0)
    assert estimate.panel_did.uncertainty == "cluster_robust_standard_error_no_interval"
    assert np.isnan(estimate.estimate.se_analytic)
    assert estimate.plan.estimator == "quasi.panel_change_score"
    assert "parallel_trends" in " ".join(estimate.assumptions or [])
    assert any(d.startswith("identification.quasi.parallel_trends_untestable_two_periods")
               for d in estimate.diagnostics)

    direct = antecedent.analyze(data, query=query)
    assert direct.panel_did == estimate.panel_did


def test_supported_panel_did_reports_interval_through_public_analyze():
    import antecedent

    ids, groups, periods, outcomes = [], [], [], []
    for group in (False, True):
        for cluster in range(30):
            for after in (False, True):
                ids.append(f"subject-{group}-{cluster}")
                groups.append(group)
                periods.append(after)
                outcomes.append(cluster * 0.1 + 2.0 * int(group and after) if after else 0.0)
    data = {"y": outcomes, "id": ids, "group": groups, "after": periods}
    query = PanelDifferenceInDifferences("y", "id", "group", "after")
    result = antecedent.analyze(data, query=query)
    did = result.panel_did
    assert did is not None
    assert did.estimate == pytest.approx(2.0)
    assert did.interval_95 is not None
    assert did.interval_95[0] < 2.0 < did.interval_95[1]
    assert did.uncertainty == "cluster_robust_normal_interval_independent_clusters"
    assert did.support_status == "licensed"
    assert result.estimate.se_analytic == pytest.approx(did.standard_error)


def test_supported_repeated_cross_section_did_has_exact_graphless_license():
    import antecedent

    ids, clusters, groups, periods, outcomes = [], [], [], [], []
    for group in (False, True):
        for after in (False, True):
            for cell_cluster in range(30):
                ids.append(f"subject-{group}-{after}-{cell_cluster}")
                clusters.append(f"cluster-{group}-{after}-{cell_cluster}")
                groups.append(group)
                periods.append(after)
                outcomes.append(float(group) + 0.5 * float(after)
                                + 2.0 * float(group and after) + 0.1 * cell_cluster)
    data = {"y": outcomes, "id": ids, "group": groups, "after": periods, "cluster": clusters}
    query = PanelDifferenceInDifferences.repeated_cross_section(
        "y", "id", "group", "after", cluster="cluster",
    )
    result = antecedent.analyze(data, query=query, refute="none")
    assert result.panel_did.estimate == pytest.approx(2.0)
    assert result.panel_did.interval_95 is not None
    assert result.panel_did.support_status == "licensed"
    loaded = antecedent.load(result.export(artifact_id="licensed-repeated-did"))
    assert loaded.answer.structured["graphless_support_status"] == "licensed"


def test_repeated_cross_section_did_runs_through_public_flow_and_direct_utility():
    import antecedent

    data = {
        "y": [0.0, 2.0, 2.0, 4.0, 10.0, 12.0, 14.0, 16.0],
        "id": [f"s{i}" for i in range(8)],
        "group": [False] * 4 + [True] * 4,
        "after": [False, False, True, True] * 2,
    }
    query = PanelDifferenceInDifferences.repeated_cross_section("y", "id", "group", "after")
    prepared = antecedent.prepare(data, query=query)
    result = prepared.estimate()
    assert result.panel_did is not None
    assert result.panel_did.estimate == pytest.approx(2.0)
    assert result.panel_did.standard_error == pytest.approx((16.0 / 7.0) ** 0.5)
    assert result.panel_did.treated_subjects == 4
    assert result.panel_did.control_subjects == 4
    assert result.panel_did.clusters == 8
    assert result.panel_did.uncertainty == "cluster_robust_standard_error_no_interval"
    assert np.isnan(result.estimate.se_analytic)
    assert result.panel_did.design == "repeated_cross_section_2x2"
    assert "repeated_cross_section" in " ".join(result.assumptions or [])
    assert antecedent.analyze(data, query=query).panel_did == result.panel_did
    assert estimate_panel_did(data, query) == result.panel_did


def test_repeated_cross_section_did_refuses_duplicate_subject_and_sparse_cell_clusters():
    import antecedent
    from antecedent.errors import CausalError

    data = {
        "y": [0.0, 2.0, 2.0, 4.0, 10.0, 12.0, 14.0, 16.0],
        "id": [f"s{i}" for i in range(8)],
        "group": [False] * 4 + [True] * 4,
        "after": [False, False, True, True] * 2,
        "cluster": [f"c{i}" for i in range(8)],
    }
    query = PanelDifferenceInDifferences.repeated_cross_section(
        "y", "id", "group", "after", cluster="cluster"
    )
    duplicated = {**data, "id": ["s0", "s0", *data["id"][2:]]}
    with pytest.raises(CausalError, match="one row per subject"):
        antecedent.analyze(duplicated, query=query)
    collapsed = {**data, "cluster": ["c0", "c0", *data["cluster"][2:]]}
    with pytest.raises(CausalError, match="at least two clusters in each group-period cell"):
        antecedent.analyze(collapsed, query=query)


def test_retained_did_refuses_cluster_shared_across_treatment_groups():
    import antecedent
    from antecedent.errors import CausalError

    for repeated in (False, True):
        ids = [f"s{i}" for i in range(8)]
        groups = [False] * 4 + [True] * 4
        periods = [False, False, True, True] * 2
        outcomes = [0., 2., 2., 4., 10., 12., 14., 16.]
        clusters = [f"c{i}" for i in range(8)]
        if not repeated:
            ids = [subject for subject in ids for _ in range(2)]
            groups = [group for group in groups for _ in range(2)]
            periods = [after for _ in range(8) for after in (False, True)]
            outcomes = [float(i + (2 if group and after else 0))
                        for i, group in enumerate([False] * 4 + [True] * 4)
                        for after in (False, True)]
            clusters = [cluster for cluster in clusters for _ in range(2)]
        clusters[8 if not repeated else 4] = clusters[0]
        if not repeated:
            clusters[9] = clusters[0]
        data = {"y": outcomes, "id": ids, "group": groups,
                "after": periods, "cluster": clusters}
        query = (PanelDifferenceInDifferences.repeated_cross_section(
            "y", "id", "group", "after", cluster="cluster") if repeated else
            PanelDifferenceInDifferences("y", "id", "group", "after", cluster="cluster"))
        with pytest.raises(CausalError, match="clusters nested within treatment groups"):
            antecedent.analyze(data, query=query)


def test_prepared_panel_did_refuses_too_few_rows():
    import antecedent
    from antecedent.errors import CausalError

    data = {"y": [1., 2., 3.], "id": ["a", "a", "b"],
            "group": [True, True, False], "after": [False, True, False]}
    with pytest.raises(CausalError, match="at least four aligned rows"):
        antecedent.analyze(data, query=PanelDifferenceInDifferences("y", "id", "group", "after"))


@pytest.mark.parametrize(
    "subjects,treated,post,match",
    [
        (["a", "a", "b"], [0, 0, 1], [0, 1, 1], "exactly one pre and one post"),
        (["a", "a", "a", "b", "b"], [0, 0, 0, 1, 1], [0, 1, 1, 0, 1], "exactly one pre and one post"),
        (["a", "a", "b", "b"], [0, 1, 1, 1], [0, 1, 0, 1], "stable within each subject"),
    ],
)
def test_panel_did_refuses_unbalanced_or_changing_assignment(subjects, treated, post, match):
    query = PanelDifferenceInDifferences("y", "id", "group", "after")
    with pytest.raises(CausalValueError, match=match):
        estimate_panel_did(
            {"y": list(range(len(subjects))), "id": subjects, "group": treated, "after": post},
            query,
        )
def test_selected_staggered_group_time_runs_through_retained_analyze():
    import antecedent

    ids = [f"s{i}" for i in range(10) for _ in range(4)]
    periods = [period for _ in range(10) for period in range(1, 5)]
    cohorts = [0 if i < 4 else 3 if i < 8 else 4 for i in range(10) for _ in range(4)]
    outcomes = [
        float(10 + int(subject[1:]) + 2 * period
              + (4 if cohort == 3 and period >= 3 else 0)
              + (7 if cohort == 4 and period >= 4 else 0))
        for subject, period, cohort in zip(ids, periods, cohorts, strict=True)
    ]
    data = {"y": outcomes, "id": ids, "period": periods, "cohort": cohorts}
    query = StaggeredAdoption("y", "id", "period", "cohort", target_cohort=3, target_period=4)
    prepared = antecedent.prepare(data, query=query)
    result = prepared.estimate()
    assert result.panel_did is not None
    assert result.panel_did.estimate == pytest.approx(4.0)
    assert result.panel_did.standard_error == pytest.approx(0.0)
    assert result.panel_did.design == "balanced_staggered_adoption_group_time_att"
    assert (result.panel_did.cohort, result.panel_did.period) == (3, 4)
    assert result.panel_did.uncertainty == "cluster_robust_standard_error_no_interval"
    assert np.isnan(result.estimate.se_analytic)
    assert result.panel_did.support_status == "unlicensed_point_utility"
    assert "cohort_specific_parallel_untreated_trends" in result.panel_did.assumptions
    assert result.plan.estimator == "quasi.staggered_group_time_never_treated"
    assert antecedent.analyze(data, query=query).panel_did == result.panel_did
    direct = estimate_group_time_att(data, query)
    assert next(effect.estimate for effect in direct.effects if (effect.cohort, effect.period) == (3, 4)) == pytest.approx(4.0)

    changed = dict(data, y=[value + 1.0 for value in outcomes])
    assert prepared.estimate(changed).panel_did == result.panel_did
    with pytest.raises(CausalValueError, match="target_cohort and target_period"):
        StaggeredAdoption("y", "id", "period", "cohort", target_cohort=3)
    with pytest.raises(Exception, match="never-treated"):
        antecedent.analyze({**data, "cohort": [3] * len(cohorts)}, query=query)


def test_native_group_time_att_recovers_staggered_known_effects_and_reports_support():
    subjects: list[str] = []
    periods: list[int] = []
    cohorts: list[int] = []
    outcome: list[float] = []
    cohort_by_subject = [0] * 4 + [2] * 3 + [3] * 3
    for unit, cohort in enumerate(cohort_by_subject):
        for period in (1, 2, 3, 4):
            subjects.append(f"s{unit}")
            periods.append(period)
            cohorts.append(cohort)
            outcome.append(float(100 + unit + 2 * period + (5 if cohort and period >= cohort else 0)))
    result = estimate_group_time_att(
        {"y": outcome, "id": subjects, "period": periods, "cohort": cohorts},
        StaggeredAdoption("y", "id", "period", "cohort"),
    )
    assert [(e.cohort, e.period) for e in result.effects] == [
        (2, 2), (2, 3), (2, 4), (3, 3), (3, 4)
    ]
    assert [effect.estimate for effect in result.effects] == pytest.approx([5.0] * 5)
    assert {(e.treated_subjects, e.control_subjects) for e in result.effects} == {(3, 4)}
    assert result.uncertainty == "cluster_robust_se_only_pointwise_cr1_unlicensed"
    assert all(effect.standard_error >= 0.0 and effect.clusters >= 4 for effect in result.effects)
    assert result.support_status == "unlicensed_point_utility"
    assert result.control_group == "never_treated_cohort_0"
    assert "cohort_specific_parallel_untreated_trends" in result.assumptions
    assert "pretrend_not_tested" in result.diagnostics


def test_group_time_att_refuses_missing_never_treated_or_unbalanced_panel():
    query = StaggeredAdoption("y", "id", "period", "cohort")
    rows = {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "id": ["a", "a", "b", "b", "c", "c"],
        "period": [1, 2, 1, 2, 1, 2],
        "cohort": [0, 0, 2, 2, 2, 2],
    }
    with pytest.raises(CausalValueError, match="never-treated controls"):
        estimate_group_time_att(
            {**rows, "cohort": [2, 2, 2, 2, 2, 2]},
            query,
        )
    supported = {**rows, "cohort": [0, 0, 2, 2, 0, 0]}
    with pytest.raises(CausalValueError, match="balanced panel"):
        estimate_group_time_att(
            {key: value[:-1] for key, value in supported.items()},
            query,
        )


def test_group_time_att_clustered_se_matches_known_variance_and_refuses_sparse_clusters():
    ids, periods, cohorts, outcomes = [], [], [], []
    changes = [0., 2., 4., 6., 7., 9., 11., 13.]
    for unit, change in enumerate(changes):
        cohort = 0 if unit < 4 else 2
        for period in (1, 2, 3):
            ids.append(f"s{unit}")
            periods.append(period)
            cohorts.append(cohort)
            outcomes.append(float(unit + (change if period >= 2 else 0.)))
    data = {"y": outcomes, "id": ids, "period": periods, "cohort": cohorts}
    query = StaggeredAdoption("y", "id", "period", "cohort")
    result = estimate_group_time_att(data, query)
    adoption = next(effect for effect in result.effects if effect.period == 2)
    assert adoption.estimate == pytest.approx(7.)
    assert adoption.standard_error == pytest.approx((20.0 / 7.0) ** 0.5)
    assert adoption.clusters == 8
    assert result.uncertainty == "cluster_robust_se_only_pointwise_cr1_unlicensed"
    with pytest.raises(CausalValueError, match="at least two clusters in each cohort/control group"):
        estimate_group_time_att(
            {**data, "cluster": [f"t{i}" if i >= 4 else "all-control" for i in range(8) for _ in (1, 2, 3)]},
            query,
            cluster="cluster",
        )


def test_staggered_event_study_reports_cohort_event_time_and_descriptive_preperiods():
    ids, ts, gs, ys = [], [], [], []
    for unit, g in enumerate([0, 0, 2, 2, 3, 3]):
        for t in range(1, 5):
            ids.append(f"u{unit}")
            ts.append(t)
            gs.append(g)
            rel = t - g
            effect = (2.0 if rel == 0 else 4.0 if rel == 1 else 4.0) if g == 2 else (3.0 if rel == 0 else 6.0 if rel == 1 else 0.0)
            ys.append(10.0 + unit + 2.0 * t + (effect if g and t >= g else 0.0))
    result = estimate_staggered_event_study(
        {"y": ys, "id": ids, "period": ts, "cohort": gs},
        StaggeredAdoption("y", "id", "period", "cohort"),
    )
    observed = {(e.cohort, e.event_time): e.estimate for e in result.effects}
    assert observed == pytest.approx({
        (2, 0): 2.0, (2, 1): 4.0, (2, 2): 4.0,
        (3, -2): 0.0, (3, 0): 3.0, (3, 1): 6.0,
    })
    assert (2, -1) not in observed and (3, -1) not in observed
    assert result.uncertainty == "cluster_robust_se_only_pointwise_cr1_unlicensed"
    assert all(effect.standard_error >= 0.0 and effect.clusters >= 4 for effect in result.effects)
    assert result.support_status == "unlicensed_point_utility"
    assert "no_anticipation" in result.assumptions
    assert "pre_adoption_estimates_are_descriptive_diagnostics_not_a_test" in result.diagnostics


def test_staggered_event_study_runs_through_retained_analyze_with_direct_parity():
    import antecedent

    ids = [f"s{i}" for i in range(8) for _ in range(4)]
    periods = [period for _ in range(8) for period in range(1, 5)]
    cohorts = [0 if i < 4 else 3 for i in range(8) for _ in range(4)]
    outcomes = [
        float(int(subject[1:]) + 2 * period + (4 if cohort == 3 and period >= 3 else 0))
        for subject, period, cohort in zip(ids, periods, cohorts, strict=True)
    ]
    data = {"y": outcomes, "id": ids, "period": periods, "cohort": cohorts}
    direct = estimate_staggered_event_study(data, StaggeredAdoption("y", "id", "period", "cohort"))
    query = StaggeredAdoption("y", "id", "period", "cohort", event_study=True)
    prepared = antecedent.prepare(data, query=query)
    result = prepared.estimate()
    assert result.panel_did == direct
    assert result.plan.estimator == "quasi.staggered_event_study_never_treated"
    assert np.isnan(result.estimate.se_analytic)
    assert result.panel_did.support_status == "unlicensed_point_utility"
    assert "pre_adoption_estimates_are_descriptive_diagnostics_not_a_test" in result.panel_did.diagnostics
    assert any(d.startswith("diagnostic.quasi.event_study.pretrend_joint_unavailable")
               for d in result.diagnostics)
    assert antecedent.analyze(data, query=query).panel_did == direct
    assert estimate_staggered_event_study(data, query) == direct
    assert prepared.estimate({**data, "y": [value + 1 for value in outcomes]}).panel_did == direct
    with pytest.raises(Exception, match="never-treated"):
        antecedent.analyze({**data, "cohort": [3] * len(cohorts)}, query=query)


def _staggered_event_interval_fixture(clusters_per_group: int):
    data = {"y": [], "id": [], "period": [], "cohort": [], "cluster": []}
    for group, cohort in (("control", 0), ("treated", 3)):
        for cluster in range(clusters_per_group):
            identifier = f"{group}-{cluster}"
            for period in range(1, 5):
                noise = ((cluster % 7) - 3) * ((period % 3) - 1) / 10
                data["y"].append(5.0 + 2.0 * period + noise
                                 + (4.0 if cohort == 3 and period >= 3 else 0.0))
                data["id"].append(identifier)
                data["period"].append(period)
                data["cohort"].append(cohort)
                data["cluster"].append(identifier)
    query = StaggeredAdoption("y", "id", "period", "cohort",
                             cluster="cluster", event_study=True)
    return data, query


def test_staggered_event_joint_pretrend_statistic_is_labeled_and_artifact_retained():
    import antecedent

    data = {"y": [], "id": [], "period": [], "cohort": []}
    for group, cohort in (("control", 0), ("treated", 4)):
        for cluster in range(8):
            identifier = f"{group}-{cluster}"
            for period in range(1, 6):
                noise = (cluster - 3.5) * ((period % 3) - 1) / 10.0
                data["y"].append(2.0 * period + noise
                                 + (2.0 if cohort == 4 and period == 1 else 0.0)
                                 + (3.0 if cohort == 4 and period >= 4 else 0.0))
                data["id"].append(identifier)
                data["period"].append(period)
                data["cohort"].append(cohort)
    query = StaggeredAdoption("y", "id", "period", "cohort", event_study=True)
    result = antecedent.analyze(data, query=query)
    diagnostic = next(d for d in result.diagnostics
                      if d.startswith("diagnostic.quasi.event_study.pretrend_joint_max_cluster_z"))
    assert "across 2 non-reference" in diagnostic
    assert "no calibrated p-value or cutoff" in diagnostic
    assert "cannot establish parallel untreated trends" in diagnostic
    artifact = antecedent.load(result.export(artifact_id="joint-pretrend"))
    assert any(item["code"] == "diagnostic.quasi.event_study.pretrend_joint_max_cluster_z"
               for item in artifact.artifact.payload["diagnostics"])


def test_staggered_event_known_truth_fixture_spans_thin_and_supported_clusters():
    import antecedent

    for clusters_per_group in (8, 24):
        data, query = _staggered_event_interval_fixture(clusters_per_group)
        result = antecedent.analyze(data, query=query)
        effects = result.panel_did.effects
        assert len(effects) == 3
        for effect, truth in zip(effects, (0.0, 4.0, 4.0), strict=True):
            assert effect.estimate == pytest.approx(truth)
            assert effect.standard_error > 0.0
            assert effect.clusters == 2 * clusters_per_group
            assert effect.treated_subjects == clusters_per_group
            assert effect.control_subjects == clusters_per_group
        assert effects[0].interval_95 is None
        if clusters_per_group == 24:
            for effect in effects[1:]:
                assert effect.interval_95[0] < 4.0 < effect.interval_95[1]
            assert result.estimate.se_analytic > 0.0
            assert result.panel_did.support_status == "off_axis_pointwise_95"
            assert result.answer.detail == "staggered_event_post_pointwise_intervals"
            assert "no simultaneous band" in result.claim()
        else:
            assert all(effect.interval_95 is None for effect in effects)
            assert np.isnan(result.estimate.se_analytic)
            assert result.panel_did.support_status == "unlicensed_point_utility"
            assert result.answer.detail == "staggered_event_point_only"
        assert estimate_staggered_event_study(data, query) == result.panel_did
        artifact = antecedent.load(result.export(artifact_id="staggered-event-boundary"))
        assert len(artifact.artifact.payload["panel_did"]["event_time_intervals_95"]) == 3


def test_staggered_event_study_refuses_no_never_treated_comparison_group():
    with pytest.raises(CausalValueError, match="never-treated controls"):
        estimate_staggered_event_study(
            {"y": [1., 2., 3., 4.], "id": ["a", "a", "b", "b"],
             "period": [1, 2, 1, 2], "cohort": [2, 2, 2, 2]},
            StaggeredAdoption("y", "id", "period", "cohort"),
        )


def test_event_study_clustered_se_matches_known_cohort_event_variance():
    ids, periods, cohorts, outcomes = [], [], [], []
    changes = [0., 2., 4., 6., 7., 9., 11., 13.]
    for unit, change in enumerate(changes):
        cohort = 0 if unit < 4 else 3
        for period in (1, 2, 3, 4):
            ids.append(f"s{unit}")
            periods.append(period)
            cohorts.append(cohort)
            outcomes.append(float(unit + (change if period >= 3 else 0.)))
    result = estimate_staggered_event_study(
        {"y": outcomes, "id": ids, "period": periods, "cohort": cohorts},
        StaggeredAdoption("y", "id", "period", "cohort"),
    )
    adoption = next(e for e in result.effects if (e.cohort, e.period) == (3, 3))
    assert adoption.estimate == pytest.approx(7.)
    assert adoption.standard_error == pytest.approx((20.0 / 7.0) ** 0.5)
