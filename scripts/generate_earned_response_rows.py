#!/usr/bin/env python3
"""Generate the licensed rows of the conditional and response cells earned by known-truth tests.

Each family below names the cells it licenses (a cartesian selector over the support axes), the
fixture whose closed-form or enumerated oracle is the truth, the executing test that consumes the
fixture and builds every cell of the family, and the checked-execution citation. The rows are
written between the generated markers at the end of parity/support_licensed.toml; the script is
deterministic and `--check` fails when the file differs from the table.

    python3 scripts/generate_earned_response_rows.py          # rewrite the generated block
    python3 scripts/generate_earned_response_rows.py --check  # fail on drift

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENSED = ROOT / "parity" / "support_licensed.toml"
BEGIN = "# generated:earned_response_rows:begin"
END = "# generated:earned_response_rows:end"

TESTS = "crates/antecedent/tests"

# Every family: `cells` is the cartesian selector, `evidence` the executing test with its fixture,
# `checked` the checked-execution citation per estimator.
FAMILIES: list[dict] = [
    {
        "query": ["InterventionResponse"],
        "graph_class": ["CoDetermined"],
        "structure": ["explicit"],
        "inference": ["Bayesian"],
        "validation": ["none"],
        "estimators": ["response.bayesian"],
        "fixture": "conformance/estimate/codetermined_joint_level",
        "test": f"{TESTS}/codetermined_bayesian_response_truth.rs",
        "assertion": "codetermined_bayesian_joint_level_matches_the_closed_form_additive_law",
        "limitations": (
            "Same-tier joint do(t1, t2) on the CoDetermined tier-closure ADMG via the Bayesian "
            "response.bayesian cell: the outcome regression on the treatments and the closure's "
            "generalized adjustment set (the tier identification the Frequentist cell.aipw cell "
            "uses) is additive in the treatments, so the published level is that model's joint "
            "level, not the saturated cell mean cell.aipw estimates. The posterior mean of the "
            "level is checked against the closed form of an additive law with a confounded "
            "design (exact cell counts in the fixture). A law with a treatment interaction is "
            "answered by the additive model, not by the saturated mean. No coverage record "
            "measures the credible interval and calibration is not claimed. Cheap/full, "
            "accepted and graph-posterior stay refused."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["InterventionResponse"],
        "graph_class": ["Dag"],
        "structure": ["explicit", "accepted", "graph_posterior"],
        "inference": ["Bayesian"],
        "validation": ["cheap", "full"],
        "estimators": ["response.bayesian"],
        "fixture": "conformance/estimate/bayesian_level_refuter_states",
        "test": f"{TESTS}/bayesian_intervention_response_refuter_truth.rs",
        "assertion": "bayesian_level_cheap_and_full_pass_on_the_clean_law_at_the_closed_form_level",
        "checked": {
            "response.bayesian": (
                f"{TESTS}/bayesian_intervention_response_refuter_truth.rs",
                "bayesian_level_refuters_execute_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "Bayesian InterventionResponse cheap and full on a static DAG (explicit, accepted, "
            "or a graph posterior of DAG atoms) run the Bayesian refuter state on the "
            "intervention level. Cheap runs the plugin-level overlap assessment on the "
            "posterior-mean level and the prior and posterior predictive checks of the outcome "
            "regression; full adds the overlap rule, level-native prior sensitivity (the "
            "response is refitted at each prior-grid point and its posterior-mean level read "
            "back, not a dose coefficient). A graph posterior runs the plugin-level suite on the "
            "pooled level and the posterior checks on its primary identified atom. Known-truth "
            "evidence: on the clean law the level is the closed form 3 + 1.5 mean(z) and every "
            "refuter passes; a planted positivity violation is detected by the overlap "
            "assessment and a planted curvature by the posterior predictive check. The verdicts "
            "are diagnostics of this fitted outcome regression, not a calibration claim, and no "
            "coverage record measures the credible interval."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["ConditionalEffect"],
        "graph_class": ["TemporalAdmg"],
        "structure": ["explicit", "accepted"],
        "inference": ["Frequentist", "Bayesian"],
        "validation": ["none", "cheap", "full"],
        "estimators": ["functional.effect"],
        "fixture": "conformance/estimate/admg_conditional_strata",
        "test": f"{TESTS}/conditional_effect_admg_validation_truth.rs",
        "assertion": "conditional_effect_admg_explicit_and_accepted_match_the_stratified_difference",
        "checked": {
            "functional.effect": (
                f"{TESTS}/conditional_effect_admg_validation_truth.rs",
                "conditional_effect_admg_cells_execute_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "A ConditionalEffect on a TemporalAdmg reads tabular rows, which carry no time "
            "index, so it is the conditional contrast of the contemporaneous projection of the "
            "template (each variable once, at lag 0, with the template's directed and "
            "bidirected edges), the same twin the licensed AverageEffect has: IDC at the two "
            "hard-set levels for a latent-confounded projection, the Dag conditional route for "
            "a latent-free one, evaluated by the discrete plug-in or its Dirichlet row "
            "posterior. A template whose variable repeats across lags has no projection and is "
            "refused. It is not the temporal conditional contrast, which is "
            "TemporalConditionalPulse or TemporalConditionalSustained on one series, "
            "identified by IDC on the unrolled window. Known-truth evidence: the stratified "
            "difference 7/27 of an exact eight-cell table (exact Frequentist plug-in, Dirichlet "
            "posterior mean within a stated band). Cheap and full run the functional refuter "
            "state on the projection: the placebo treatment (exact null 0), subset stability "
            "and, under full, the placebo outcome and bootstrap stability of the whole "
            "procedure (Bayesian: placebo treatment, prior sensitivity and, under full, the "
            "placebo outcome); the verdicts are diagnostics, not a calibration claim, and no "
            "coverage record measures the intervals."
        ),
        "calibration_reason": "estimator_grid_not_measured",
        "calibration_reason_by_inference": {"Frequentist": "no_interval_reported"},
        "checked_by_inference": {
            "Bayesian": {
                "functional.effect": (
                    f"{TESTS}/conditional_effect_admg_bayesian_interval.rs",
                    "conditional_effect_admg_bayesian_cells_execute_with_a_posterior_quantile_interval",
                )
            }
        },
    },
    {
        "query": ["ConditionalEffect"],
        "graph_class": ["Admg"],
        "structure": ["explicit", "accepted"],
        "inference": ["Frequentist", "Bayesian"],
        "validation": ["cheap", "full"],
        "estimators": ["functional.effect"],
        "fixture": "conformance/estimate/admg_conditional_strata",
        "test": f"{TESTS}/conditional_effect_admg_validation_truth.rs",
        "assertion": "conditional_effect_admg_explicit_and_accepted_match_the_stratified_difference",
        "checked": {
            "functional.effect": (
                f"{TESTS}/conditional_effect_admg_validation_truth.rs",
                "conditional_effect_admg_cells_execute_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "Cheap and full validation of a ConditionalEffect on an Admg run the functional "
            "refuter state. The IDC contrast over the modifier law executes as at validation "
            "none; the population value is then refuted by a placebo treatment (the permuted "
            "treatment removes the effect the identified functional reads; exact null 0), by "
            "subset stability and, under full, by the placebo outcome and bootstrap stability "
            "of the whole procedure, each a joint statistic over the refitted value (Bayesian: "
            "placebo treatment, prior sensitivity and, under full, the placebo outcome). "
            "Known-truth evidence: the stratified difference 7/27 of an exact eight-cell table "
            "with every refuter passing on it. The verdicts are diagnostics, not a calibration "
            "claim."
        ),
        "calibration_reason": "estimator_grid_not_measured",
        "calibration_reason_by_inference": {"Frequentist": "no_interval_reported"},
        "checked_by_inference": {
            "Bayesian": {
                "functional.effect": (
                    f"{TESTS}/conditional_effect_admg_bayesian_interval.rs",
                    "conditional_effect_admg_bayesian_cells_execute_with_a_posterior_quantile_interval",
                )
            }
        },
    },
    {
        "query": ["ConditionalEffect"],
        "graph_class": ["Admg", "TemporalAdmg"],
        "structure": ["graph_posterior"],
        "inference": ["Frequentist", "Bayesian"],
        "validation": ["none", "cheap", "full"],
        "estimators": ["functional.effect"],
        "fixture": "conformance/estimate/admg_conditional_strata",
        "test": f"{TESTS}/conditional_effect_admg_validation_truth.rs",
        "assertion": "conditional_effect_admg_graph_posterior_states_its_mass_and_matches_the_stratified_difference",
        "checked": {
            "functional.effect": (
                f"{TESTS}/conditional_effect_admg_validation_truth.rs",
                "conditional_effect_admg_cells_execute_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "A ConditionalEffect on a graph posterior of Admg atoms (or of TemporalAdmg atoms, "
            "each read as the contemporaneous projection of its template) is answered on every "
            "atom by the executor the query has on a supplied ADMG (the modifier-conditioned "
            "IDC contrast on a latent-confounded atom, the DAG estimator on a latent-free one) "
            "and the atoms mix by posterior probability. An atom the identifier refuses is "
            "unidentified posterior mass and an atom whose estimator cannot be evaluated is "
            "unevaluable mass; neither is redistributed, the result states both and resolves "
            "its status to graph-dependent when either exceeds the tolerance. Several atoms "
            "that report row influence publish the interval of the weighted influence on one "
            "shared sample, otherwise they are re-run on shared resampled tables (Frequentist) "
            "or pooled draw by draw (Bayesian); an interval that cannot be formed is withheld "
            "with the reason stated. Cheap and full run each atom's own refuter suite against "
            "that atom's own estimate and mix the reports by posterior weight; a mixed verdict "
            "passes only if every contributing atom passes. Known-truth evidence: identified "
            "atoms answer with the stratified difference 7/27 of an exact table and the "
            "refusing atom's mass 0.2 is stated, not redistributed. A latent-confounded atom "
            "must be encodable without a bow (a directed and a bidirected edge on one pair). "
            "The verdicts are diagnostics, not a calibration claim."
        ),
        "calibration_reason": "estimator_grid_not_measured",
        "calibration_reason_by_inference": {"Frequentist": "no_interval_reported"},
        "checked_by_inference": {
            "Bayesian": {
                "functional.effect": (
                    f"{TESTS}/conditional_effect_admg_bayesian_interval.rs",
                    "conditional_effect_admg_bayesian_cells_execute_with_a_posterior_quantile_interval",
                )
            }
        },
    },
    {
        "query": ["AverageEffect"],
        "graph_class": ["TemporalDag", "TemporalCpdag", "TemporalPag"],
        "structure": ["explicit", "accepted", "graph_posterior"],
        "inference": ["Frequentist"],
        "validation": ["none", "cheap", "full"],
        "estimators": ["temporal.linear.adjustment"],
        "fixture": "conformance/estimate/lag_zero_average_effect",
        "test": f"{TESTS}/lag_zero_average_effect_truth.rs",
        "assertion": "lag_zero_average_effect_matches_the_structural_coefficient_on_every_temporal_cell",
        "checked": {
            "temporal.linear.adjustment": (
                f"{TESTS}/lag_zero_average_effect_truth.rs",
                "lag_zero_average_effect_cells_execute_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "An AverageEffect on one series under a temporal graph denotes the lag-0 contrast "
            "E[Y_t | do(X_t = active)] - E[Y_t | do(X_t = control)]. The build compiles it as the "
            "pulse at offset 0 read at horizon 1 (the same program as the explicit lag-0 pulse, "
            "which the evidence checks bit for bit): on a TemporalDag the temporal backdoor, on a "
            "TemporalCpdag every completion adjusts for the parents of X_t, on a TemporalPag every "
            "MAG completion uses visibility-aware adjustment (a class Bayesian study publishes "
            "the identified set), and over a graph posterior the same pulse is compiled and the "
            "per-atom proofs mix as for any temporal pulse. Known-truth evidence: the structural "
            "coefficient of the stated linear law (2) is recovered within a stated band while the "
            "omitted-confounder slope (2.353) is not; cheap and full run the refuters the fixture "
            "names with the placebo treatment and dummy outcome at the null 0 and the random "
            "common cause, subset and bootstrap refuters at the original value. A class graph "
            "posterior under Bayesian inference mixes the frequentist-style suite (no predictive "
            "checks, no prior sensitivity). The e-value passes only through the confidence limit of the "
            "effect, so a TemporalDag Frequentist study without bootstrap replicates (no standard "
            "error) reports it uninformative and failed, correctly, while the class and "
            "graph-posterior studies pass it. Effect "
            "modifiers, a non-mean functional and a population other than AllObserved are "
            "refused (next attempt: TemporalConditionalPulse or a temporal ResponseCurve). The "
            "verdicts are diagnostics, not a calibration claim, and no coverage record measures "
            "the intervals."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["AverageEffect"],
        "graph_class": ["TemporalDag", "TemporalCpdag", "TemporalPag"],
        "structure": ["explicit", "accepted", "graph_posterior"],
        "inference": ["Bayesian"],
        "validation": ["none", "cheap", "full"],
        "estimators": ["bayesian.temporal.gcomp"],
        "fixture": "conformance/estimate/lag_zero_average_effect",
        "test": f"{TESTS}/lag_zero_average_effect_truth.rs",
        "assertion": "lag_zero_average_effect_matches_the_structural_coefficient_on_every_temporal_cell",
        "checked": {
            "bayesian.temporal.gcomp": (
                f"{TESTS}/lag_zero_average_effect_truth.rs",
                "lag_zero_average_effect_cells_execute_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "An AverageEffect on one series under a temporal graph denotes the lag-0 contrast "
            "E[Y_t | do(X_t = active)] - E[Y_t | do(X_t = control)]. The build compiles it as the "
            "pulse at offset 0 read at horizon 1 (the same program as the explicit lag-0 pulse, "
            "which the evidence checks bit for bit): on a TemporalDag the temporal backdoor, on a "
            "TemporalCpdag every completion adjusts for the parents of X_t, on a TemporalPag every "
            "MAG completion uses visibility-aware adjustment (a class Bayesian study publishes "
            "the identified set), and over a graph posterior the same pulse is compiled and the "
            "per-atom proofs mix as for any temporal pulse. Known-truth evidence: the structural "
            "coefficient of the stated linear law (2) is recovered within a stated band while the "
            "omitted-confounder slope (2.353) is not; cheap and full run the refuters the fixture "
            "names with the placebo treatment and dummy outcome at the null 0 and the random "
            "common cause, subset and bootstrap refuters at the original value. A class graph "
            "posterior under Bayesian inference mixes the frequentist-style suite (no predictive "
            "checks, no prior sensitivity). The e-value passes only through the confidence limit of the "
            "effect, so a TemporalDag Frequentist study without bootstrap replicates (no standard "
            "error) reports it uninformative and failed, correctly, while the class and "
            "graph-posterior studies pass it. Effect "
            "modifiers, a non-mean functional and a population other than AllObserved are "
            "refused (next attempt: TemporalConditionalPulse or a temporal ResponseCurve). The "
            "verdicts are diagnostics, not a calibration claim, and no coverage record measures "
            "the intervals."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["TemporalConditionalPulse", "TemporalConditionalSustained"],
        "graph_class": ["TemporalDag", "TemporalCpdag", "TemporalPag"],
        "structure": ["explicit", "accepted"],
        "inference": ["Frequentist", "Bayesian"],
        "validation": ["cheap", "full"],
        "estimators": ["temporal.linear.adjustment"],
        "fixture": "conformance/estimate/temporal_conditional_refuter_states",
        "test": f"{TESTS}/temporal_conditional_validation_truth.rs",
        "assertion": "temporal_conditional_cheap_and_full_pass_on_the_clean_law_at_the_marginal_effect",
        "checked": {
            "temporal.linear.adjustment": (
                f"{TESTS}/temporal_conditional_validation_truth.rs",
                "temporal_conditional_cells_execute_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "Cheap and full validation of a temporal conditional pulse or sustained effect on a "
            "TemporalDag, or on every completion of a TemporalCpdag or TemporalPag, run the "
            "functional refuter state; the conditional contrast executes as at validation none. "
            "Known-truth evidence: the marginal conditional effect of a stated law with a binary "
            "modifier (b + d p = 1.82) is recovered within a stated band while the unmoderated slope "
            "(1.5) is not. The functional refuter state runs the circular-shift placebo treatment, "
            "contiguous-window subset stability and, under full, the placebo outcome and the "
            "negative-control outcome, pre-trend and anticipation checks (Bayesian: placebo "
            "treatment, prior sensitivity and, under full, the placebo outcome, the posterior "
            "predictive check of a lag-aware reference model and the same control checks); the "
            "placebo and control checks sit at the exact null 0 and the stability checks at the "
            "original value, and a planted effect of the treatment of the pulse period is detected "
            "by the pre-trend check. The circular-block bootstrap stability of a conditional effect "
            "is refused, not run: the interaction design reads no sampling weights, and a "
            "weighted series is refused by the class path as it is by the TemporalDag path. The "
            "verdicts are diagnostics, not a calibration claim, and no coverage record measures "
            "the intervals."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["TemporalConditionalPulse", "TemporalConditionalSustained"],
        "graph_class": ["TemporalDag"],
        "structure": ["graph_posterior"],
        "inference": ["Frequentist", "Bayesian"],
        "validation": ["none", "cheap", "full"],
        "estimators": ["temporal.linear.adjustment"],
        "fixture": "conformance/estimate/temporal_conditional_refuter_states",
        "test": f"{TESTS}/temporal_conditional_validation_truth.rs",
        "assertion": "temporal_conditional_graph_posterior_cells_match_the_marginal_effect",
        "checked": {
            "temporal.linear.adjustment": (
                f"{TESTS}/temporal_conditional_validation_truth.rs",
                "temporal_conditional_graph_posterior_executes_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "A temporal conditional effect on a graph posterior of TemporalDag atoms is answered on "
            "every atom (a stationary template with its lag mask) by the temporal adjustment route "
            "of that template, and the atoms mix by posterior probability; an atom the identifier "
            "refuses is unidentified posterior mass and an atom whose estimator cannot be evaluated "
            "is unevaluable mass, neither redistributed. The result artifact declares the unfolded "
            "window of the template its identification names. Cheap and full refit the whole "
            "graph-posterior procedure under the refuters and mix each refuter's report by "
            "posterior weight. "
            "Known-truth evidence: the marginal conditional effect of a stated law with a binary "
            "modifier (b + d p = 1.82) is recovered within a stated band while the unmoderated slope "
            "(1.5) is not. The functional refuter state runs the circular-shift placebo treatment, "
            "contiguous-window subset stability and, under full, the placebo outcome and the "
            "negative-control outcome, pre-trend and anticipation checks (Bayesian: placebo "
            "treatment, prior sensitivity and, under full, the placebo outcome, the posterior "
            "predictive check of a lag-aware reference model and the same control checks); the "
            "placebo and control checks sit at the exact null 0 and the stability checks at the "
            "original value, and a planted effect of the treatment of the pulse period is detected "
            "by the pre-trend check. The circular-block bootstrap stability of a conditional effect "
            "is refused, not run: the interaction design reads no sampling weights, and a "
            "weighted series is refused by the class path as it is by the TemporalDag path. The "
            "verdicts are diagnostics, not a calibration claim, and no coverage record measures "
            "the intervals."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["TemporalConditionalPulse", "TemporalConditionalSustained"],
        "graph_class": ["TemporalCpdag", "TemporalPag"],
        "structure": ["graph_posterior"],
        "inference": ["Frequentist", "Bayesian"],
        "validation": ["cheap", "full"],
        "estimators": ["temporal.linear.adjustment"],
        "fixture": "conformance/estimate/temporal_conditional_refuter_states",
        "test": f"{TESTS}/temporal_conditional_validation_truth.rs",
        "assertion": "temporal_conditional_graph_posterior_cells_match_the_marginal_effect",
        "checked": {
            "temporal.linear.adjustment": (
                f"{TESTS}/temporal_conditional_validation_truth.rs",
                "temporal_conditional_graph_posterior_executes_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "Cheap and full validation of a graph-posterior temporal conditional effect on "
            "TemporalCpdag or TemporalPag atoms run the functional refuter state on the "
            "posterior-mixed value, refitting the whole graph-posterior procedure. "
            "Known-truth evidence: the marginal conditional effect of a stated law with a binary "
            "modifier (b + d p = 1.82) is recovered within a stated band while the unmoderated slope "
            "(1.5) is not. The functional refuter state runs the circular-shift placebo treatment, "
            "contiguous-window subset stability and, under full, the placebo outcome and the "
            "negative-control outcome, pre-trend and anticipation checks (Bayesian: placebo "
            "treatment, prior sensitivity and, under full, the placebo outcome, the posterior "
            "predictive check of a lag-aware reference model and the same control checks); the "
            "placebo and control checks sit at the exact null 0 and the stability checks at the "
            "original value, and a planted effect of the treatment of the pulse period is detected "
            "by the pre-trend check. The circular-block bootstrap stability of a conditional effect "
            "is refused, not run: the interaction design reads no sampling weights, and a "
            "weighted series is refused by the class path as it is by the TemporalDag path. The "
            "verdicts are diagnostics, not a calibration claim, and no coverage record measures "
            "the intervals."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["InterventionResponse"],
        "graph_class": ["Cpdag", "Pag"],
        "structure": ["explicit", "accepted"],
        "inference": ["Frequentist"],
        "validation": ["cheap", "full"],
        "estimators": ["response.intervention_gcomp"],
        "fixture": "conformance/estimate/intervention_response_class_refuter_states",
        "test": f"{TESTS}/intervention_response_class_validation_truth.rs",
        "assertion": "intervention_response_class_cheap_and_full_match_each_completions_closed_form",
        "checked": {
            "response.intervention_gcomp": (
                f"{TESTS}/intervention_response_class_validation_truth.rs",
                "intervention_response_class_cells_execute_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "Cheap and full InterventionResponse on a supplied or accepted Cpdag or Pag run the "
            "plugin-level suite (overlap; full adds the overlap rule) on every completion's own "
            "level and estimand, mix the reports only when every completion answers the same level, "
            "and report how far the completions disagree. Known-truth evidence: each completion's "
            "level equals the closed form (the completion that adjusts for the confounder 3, the "
            "one that does not the association 3.441), and every refuter reads a completion's own "
            "closed-form level. The verdicts are diagnostics, not a calibration claim, and no "
            "coverage record measures the intervals."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["InterventionResponse"],
        "graph_class": ["Cpdag", "Pag"],
        "structure": ["explicit", "accepted"],
        "inference": ["Bayesian"],
        "validation": ["cheap", "full"],
        "estimators": ["response.bayesian"],
        "fixture": "conformance/estimate/intervention_response_class_refuter_states",
        "test": f"{TESTS}/intervention_response_class_validation_truth.rs",
        "assertion": "intervention_response_class_cheap_and_full_match_each_completions_closed_form",
        "checked": {
            "response.bayesian": (
                f"{TESTS}/intervention_response_class_validation_truth.rs",
                "intervention_response_class_cells_execute_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "Cheap and full InterventionResponse on a supplied or accepted Cpdag or Pag run the "
            "plugin-level suite (overlap; full adds the overlap rule) on every completion's own "
            "level and estimand, mix the reports only when every completion answers the same level, "
            "and report how far the completions disagree. Known-truth evidence: each completion's "
            "level equals the closed form (the completion that adjusts for the confounder 3, the "
            "one that does not the association 3.441), and every refuter reads a completion's own "
            "closed-form level. The verdicts are diagnostics, not a calibration claim, and no "
            "coverage record measures the intervals."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["InterventionResponse"],
        "graph_class": ["TemporalDag", "TemporalCpdag", "TemporalPag"],
        "structure": ["explicit", "accepted"],
        "inference": ["Frequentist"],
        "validation": ["cheap", "full"],
        "estimators": ["temporal.response.gcomp"],
        "fixture": "conformance/estimate/intervention_response_class_refuter_states",
        "test": f"{TESTS}/intervention_response_class_validation_truth.rs",
        "assertion": "intervention_response_class_cheap_and_full_match_each_completions_closed_form",
        "checked": {
            "temporal.response.gcomp": (
                f"{TESTS}/intervention_response_class_validation_truth.rs",
                "intervention_response_class_cells_execute_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "Cheap and full InterventionResponse on a TemporalDag, TemporalCpdag or TemporalPag run "
            "the TemporalDag Pulse-native suite on each requested horizon's own Pulse witness (the "
            "unit contrast with the response's intervention as the active arm, under that "
            "completion's proof and lag indexer): overlap, the placebo and sensitivity checks and, "
            "under full, the subset and bootstrap checks of the dependent-series design; Bayesian "
            "studies add the predictive checks and prior sensitivity of each witness. Completions "
            "and horizons are not mixed; a disagreement diagnostic reports how far the completions' "
            "witness effects spread. Known-truth evidence: each completion's level and witness equal "
            "the closed form (adjusting: level 3, witness 2; omitted-confounder completion: 3.209, "
            "2.299), the placebo treatment and dummy outcome sit at the null 0, and the posterior "
            "predictive check passes on the clean law; the e-value, subset and prior-predictive "
            "verdicts depend on the sample, interval and prior scale and are not asserted. The "
            "verdicts are diagnostics, not a calibration claim, and no coverage record measures the "
            "intervals."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["InterventionResponse"],
        "graph_class": ["TemporalDag", "TemporalCpdag", "TemporalPag"],
        "structure": ["explicit", "accepted"],
        "inference": ["Bayesian"],
        "validation": ["cheap", "full"],
        "estimators": ["response.temporal.bayesian"],
        "fixture": "conformance/estimate/intervention_response_class_refuter_states",
        "test": f"{TESTS}/intervention_response_class_validation_truth.rs",
        "assertion": "intervention_response_class_cheap_and_full_match_each_completions_closed_form",
        "checked": {
            "response.temporal.bayesian": (
                f"{TESTS}/intervention_response_class_validation_truth.rs",
                "intervention_response_class_cells_execute_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "Cheap and full InterventionResponse on a TemporalDag, TemporalCpdag or TemporalPag run "
            "the TemporalDag Pulse-native suite on each requested horizon's own Pulse witness (the "
            "unit contrast with the response's intervention as the active arm, under that "
            "completion's proof and lag indexer): overlap, the placebo and sensitivity checks and, "
            "under full, the subset and bootstrap checks of the dependent-series design; Bayesian "
            "studies add the predictive checks and prior sensitivity of each witness. Completions "
            "and horizons are not mixed; a disagreement diagnostic reports how far the completions' "
            "witness effects spread. Known-truth evidence: each completion's level and witness equal "
            "the closed form (adjusting: level 3, witness 2; omitted-confounder completion: 3.209, "
            "2.299), the placebo treatment and dummy outcome sit at the null 0, and the posterior "
            "predictive check passes on the clean law; the e-value, subset and prior-predictive "
            "verdicts depend on the sample, interval and prior scale and are not asserted. The "
            "verdicts are diagnostics, not a calibration claim, and no coverage record measures the "
            "intervals."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["ResponseCurve"],
        "graph_class": ["Admg", "TemporalAdmg"],
        "structure": ["graph_posterior"],
        "inference": ["Frequentist"],
        "validation": ["cheap", "full"],
        "estimators": ["functional.effect"],
        "fixture": "conformance/estimate/response_curve_posterior_refuter_states",
        "test": f"{TESTS}/response_curve_posterior_validation_truth.rs",
        "assertion": "response_curve_graph_posterior_cheap_and_full_match_the_identified_curves",
        "checked": {
            "functional.effect": (
                f"{TESTS}/response_curve_posterior_validation_truth.rs",
                "response_curve_graph_posterior_executes_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "Cheap and full validation of a graph-posterior ResponseCurve over Admg, TemporalAdmg, "
            "Cpdag or Pag atoms run the functional refuter state on the posterior-mixed curve, "
            "refitting the whole graph-posterior procedure on perturbed data: the placebo "
            "treatment (which removes the effect the identified curve reads, so the refitted curve "
            "is flat at the mean outcome), the placebo outcome and bootstrap stability of the "
            "whole mixed curve under full, and subset stability (Bayesian: prior sensitivity and "
            "the posterior predictive check where the route has them); each is one joint "
            "statistic over the grid, and a refuter the curve cannot honour is recorded as "
            "refused. A curve conditional on the atoms that identify it (lost mass to atoms the "
            "identifier refuses) is refuted as that same conditional curve. Beside them the "
            "support of the curve over the posterior atoms is reported (curve.atom_support): the "
            "identified mass equals the posterior weight of the accepted atoms. Known-truth "
            "evidence: the front-door curve [0.314, 0.596] of an exact table (exact Frequentist "
            "plug-in, Dirichlet mean within a band) and the sampled Cpdag and PAG curves; every "
            "refuter passes on the clean law. The verdicts are diagnostics, not a calibration "
            "claim."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["ResponseCurve"],
        "graph_class": ["Admg", "TemporalAdmg"],
        "structure": ["graph_posterior"],
        "inference": ["Bayesian"],
        "validation": ["cheap", "full"],
        "estimators": ["functional.effect"],
        "fixture": "conformance/estimate/response_curve_posterior_refuter_states",
        "test": f"{TESTS}/response_curve_posterior_validation_truth.rs",
        "assertion": "response_curve_graph_posterior_cheap_and_full_match_the_identified_curves",
        "checked": {
            "functional.effect": (
                f"{TESTS}/response_curve_posterior_validation_truth.rs",
                "response_curve_graph_posterior_executes_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "Cheap and full validation of a graph-posterior ResponseCurve over Admg, TemporalAdmg, "
            "Cpdag or Pag atoms run the functional refuter state on the posterior-mixed curve, "
            "refitting the whole graph-posterior procedure on perturbed data: the placebo "
            "treatment (which removes the effect the identified curve reads, so the refitted curve "
            "is flat at the mean outcome), the placebo outcome and bootstrap stability of the "
            "whole mixed curve under full, and subset stability (Bayesian: prior sensitivity and "
            "the posterior predictive check where the route has them); each is one joint "
            "statistic over the grid, and a refuter the curve cannot honour is recorded as "
            "refused. A curve conditional on the atoms that identify it (lost mass to atoms the "
            "identifier refuses) is refuted as that same conditional curve. Beside them the "
            "support of the curve over the posterior atoms is reported (curve.atom_support): the "
            "identified mass equals the posterior weight of the accepted atoms. Known-truth "
            "evidence: the front-door curve [0.314, 0.596] of an exact table (exact Frequentist "
            "plug-in, Dirichlet mean within a band) and the sampled Cpdag and PAG curves; every "
            "refuter passes on the clean law. The verdicts are diagnostics, not a calibration "
            "claim."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["ResponseCurve"],
        "graph_class": ["Cpdag", "Pag"],
        "structure": ["graph_posterior"],
        "inference": ["Frequentist"],
        "validation": ["cheap", "full"],
        "estimators": ["response.kennedy_dr"],
        "fixture": "conformance/estimate/response_curve_posterior_refuter_states",
        "test": f"{TESTS}/response_curve_posterior_validation_truth.rs",
        "assertion": "response_curve_graph_posterior_cheap_and_full_match_the_identified_curves",
        "checked": {
            "response.kennedy_dr": (
                f"{TESTS}/response_curve_posterior_validation_truth.rs",
                "response_curve_graph_posterior_executes_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "Cheap and full validation of a graph-posterior ResponseCurve over Admg, TemporalAdmg, "
            "Cpdag or Pag atoms run the functional refuter state on the posterior-mixed curve, "
            "refitting the whole graph-posterior procedure on perturbed data: the placebo "
            "treatment (which removes the effect the identified curve reads, so the refitted curve "
            "is flat at the mean outcome), the placebo outcome and bootstrap stability of the "
            "whole mixed curve under full, and subset stability (Bayesian: prior sensitivity and "
            "the posterior predictive check where the route has them); each is one joint "
            "statistic over the grid, and a refuter the curve cannot honour is recorded as "
            "refused. A curve conditional on the atoms that identify it (lost mass to atoms the "
            "identifier refuses) is refuted as that same conditional curve. Beside them the "
            "support of the curve over the posterior atoms is reported (curve.atom_support): the "
            "identified mass equals the posterior weight of the accepted atoms. Known-truth "
            "evidence: the front-door curve [0.314, 0.596] of an exact table (exact Frequentist "
            "plug-in, Dirichlet mean within a band) and the sampled Cpdag and PAG curves; every "
            "refuter passes on the clean law. The verdicts are diagnostics, not a calibration "
            "claim."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
    {
        "query": ["ResponseCurve"],
        "graph_class": ["Cpdag", "Pag"],
        "structure": ["graph_posterior"],
        "inference": ["Bayesian"],
        "validation": ["cheap", "full"],
        "estimators": ["response.bayesian"],
        "fixture": "conformance/estimate/response_curve_posterior_refuter_states",
        "test": f"{TESTS}/response_curve_posterior_validation_truth.rs",
        "assertion": "response_curve_graph_posterior_cheap_and_full_match_the_identified_curves",
        "checked": {
            "response.bayesian": (
                f"{TESTS}/response_curve_posterior_validation_truth.rs",
                "response_curve_graph_posterior_executes_from_the_retained_plan_with_the_builder_dropped",
            )
        },
        "limitations": (
            "Cheap and full validation of a graph-posterior ResponseCurve over Admg, TemporalAdmg, "
            "Cpdag or Pag atoms run the functional refuter state on the posterior-mixed curve, "
            "refitting the whole graph-posterior procedure on perturbed data: the placebo "
            "treatment (which removes the effect the identified curve reads, so the refitted curve "
            "is flat at the mean outcome), the placebo outcome and bootstrap stability of the "
            "whole mixed curve under full, and subset stability (Bayesian: prior sensitivity and "
            "the posterior predictive check where the route has them); each is one joint "
            "statistic over the grid, and a refuter the curve cannot honour is recorded as "
            "refused. A curve conditional on the atoms that identify it (lost mass to atoms the "
            "identifier refuses) is refuted as that same conditional curve. Beside them the "
            "support of the curve over the posterior atoms is reported (curve.atom_support): the "
            "identified mass equals the posterior weight of the accepted atoms. Known-truth "
            "evidence: the front-door curve [0.314, 0.596] of an exact table (exact Frequentist "
            "plug-in, Dirichlet mean within a band) and the sampled Cpdag and PAG curves; every "
            "refuter passes on the clean law. The verdicts are diagnostics, not a calibration "
            "claim."
        ),
        "calibration_reason": "estimator_grid_not_measured",
    },
]


def quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_row(family: dict, cell: tuple[str, str, str, str, str]) -> str:
    query, graph_class, structure, inference, validation = cell
    checked = family.get("checked_by_inference", {}).get(inference) or family.get("checked") or {
        estimator: (family["test"], family["assertion"]) for estimator in family["estimators"]
    }
    lines = [
        "[[cell]]",
        f"query = {quote(query)}",
        f"graph_class = {quote(graph_class)}",
        f"structure = {quote(structure)}",
        f"inference = {quote(inference)}",
        f"validation = {quote(validation)}",
        "estimators = [" + ", ".join(quote(e) for e in family["estimators"]) + "]",
        "staged = true",
        f"evidence_kind = {quote(family.get('evidence_kind', 'internal_known_truth'))}",
    ]
    if family.get("fixture"):
        lines.append(f"known_truth_fixture = {quote(family['fixture'])}")
    lines += [
        f"evidence_test = {quote(family['test'])}",
        f"evidence_assertion = {quote(family['assertion'])}",
        "checked_execution = [",
    ]
    for estimator in family["estimators"]:
        test, assertion = checked[estimator]
        lines.append(
            f"  {{ estimator = {quote(estimator)}, test = {quote(test)}, "
            f"assertion = {quote(assertion)} }},"
        )
    lines += [
        "]",
        f"limitations = {quote(family['limitations'])}",
        "calibration_reason = "
        + quote(family.get("calibration_reason_by_inference", {}).get(inference, family["calibration_reason"])),
    ]
    return "\n".join(lines)


def render_block() -> str:
    rows = []
    for family in FAMILIES:
        for cell in product(
            family["query"],
            family["graph_class"],
            family["structure"],
            family["inference"],
            family["validation"],
        ):
            rows.append(render_row(family, cell))
    return BEGIN + "\n\n" + "\n\n".join(rows) + "\n\n" + END + "\n"


def splice(text: str, block: str) -> str:
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        return head.rstrip("\n") + "\n\n" + block + tail.lstrip("\n")
    return text.rstrip("\n") + "\n\n" + block


def main(argv: list[str]) -> int:
    current = LICENSED.read_text(encoding="utf-8")
    updated = splice(current, render_block())
    if argv[:1] == ["--check"]:
        if updated != current:
            print("parity/support_licensed.toml: generated earned-response block drifted")
            return 1
        print("earned response rows OK")
        return 0
    if updated != current:
        LICENSED.write_text(updated, encoding="utf-8", newline="\n")
    print(f"wrote {sum(len(f['query']) * len(f['graph_class']) * len(f['structure']) * len(f['inference']) * len(f['validation']) for f in FAMILIES)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
