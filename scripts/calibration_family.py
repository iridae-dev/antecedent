"""Freeze the complete gate roster and reviewed source-fixed coverage-label family.

The per-group cap is a fail-closed guard, not a justification for data-adaptive
hypothesis selection. The reviewed source footprint is hashed separately; changed
label-producing code requires another explicit review before a new family is frozen.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import tempfile
from functools import cache
from pathlib import Path

import calibration_allocations as allocations
import calibration_decision as decision
from _repo import REPO_ROOT as ROOT
from _rust_source import lex

MANIFEST_PATH = "scripts/calibration_family_manifest.json"
AUDIT_PATH = "scripts/calibration_family_labels.json"
# The gate's group table; a commit from before it held the groups in a shell driver,
# under one of the earlier names (newest first).
GATE_PATH = "parity/calibration_groups.toml"
EARLIER_GATE_PATHS = ("scripts/measure_calibration_groups.sh", "scripts/gate_calibration.sh")
ACCEPTANCE_PATH = "parity/acceptance_manifest.toml"
SCHEMA = "source_fixed_calibration_family.v1"

# Reviewed producer provenance, not an automated proof from a runtime cell cap.
LABEL_REVIEW = {
    "discovered_transport_calibration.rs": "Literal protocol versions; fixed requested discovery/transport construction, never measured coverage.",
    "calibration_backlog.rs": "Prespecified mediation contrast labels and dose GRID/curve-dose arrays; helper labels passed from fixed query constructors.",
    "nested_markov_bayesian_calibration.rs": "Caller-declared functional labels from fixed nominal/boundary test configurations.",
    "sample_split_parent_set_effect_calibration.rs": "Literal producer-version labels; parent discovery outcomes do not choose a coverage hypothesis.",
    "temporal_response_calibration.rs": "Fixed test/horizon/dose procedures. Atom tallies are created after simulation from observed adjustment sets, but all labels belong to the fixed graph subset universe: CPDAG at most2 sets x3 cells x3 tests, PAG at most4 sets x3 cells x3 tests. Source-possible whole-file bound162, including empty/unobserved allowed subsets; observed historical135 is not the family definition.",
    "calibration_bayesian_static.rs": "Optional labels supplied by declared static query coordinates; finite source case/dose arrays.",
    "temporal_frequentist.rs": "Prespecified functional contrast labels at two source-fixed confidence levels.",
    "static_calibration.rs": "Constant dose GRID, fixed contrast tables, and literal linear-outcome labels.",
    "weighted_interval_calibration.rs": "Literal interval-construction versions; no labels selected using calibration outcomes.",
    "panel_calibration.rs": "Two fixed dose labels and deterministic sample-grid unit count.",
    "calibration_temporal_class.rs": "Optional labels from finite caller-declared query contrast cases.",
    "bayesian_temporal.rs": "Finite declared component-name tuples, deterministically lowercased before simulation.",
    "derivative_calibration.rs": "Fixed derivative-coordinate indices from the declared finite target dimension.",
    "calibration_response.rs": "Dose/shift/horizon labels from fixed caller query grids, constructed before replicate mapping.",
    "calibration_temporal.rs": "Caller-supplied declared temporal-query labels; no observed-statistic label generation.",
    "mixture_interval_calibration.rs": "Literal x=1, fixed dose grids, and finite direct/mediated/total contrast tuples.",
    "decision_reliability_interval_calibration.rs": "Finite declared coupling labels and literal shared-rho construction; selected action remains one fixed procedure target.",
    "calibration_counterfactual.rs": "Declared counterfactual-world labels from fixed query construction.",
    "practitioner_scale_calibration.rs": "Source-fixed named dose/horizon coordinates and two declared confidence levels.",
    "adaptive_whole_procedure_calibration.rs": "Literal selected_contrast scores the whole selection algorithm, rather than choosing a hypothesis from observed coverage.",
    "nested_markov_fisher_calibration.rs": "Debug rendering of a finite caller-declared functional enum.",
    "mnar_interval_calibration.rs": "Declared-shift and arm-missingness band labels from a finite source grid; a replicate whose realized cell differs from the measured one is redrawn, so observed data never choose a label.",
    "categorical_posterior_calibration.rs": "Level-count and row-multiplier labels k2_r1 to k8_r16 from finite source loops; the three imbalances cycled inside a record are fixed settings, not outcomes.",
    "grid_functional_effect_calibration.rs": "One test per group of supported functional-effect coordinates that report one interval, through the shared sibling driver: the coordinates sharing a measured base are a source-fixed list, each checked to report the base interval, truth and scope on the first replicates before it is scored on the base; the scored targets (labels and closed-form truths) are a caller-declared list handed to the driver before any replicate runs, at most three per coordinate, at the two source-fixed levels; the sibling variants are the fixed validation modes and the accepted twin of each coordinate, never chosen from outcomes.",
    "grid_functional_posterior_calibration.rs": "One test per graph-posterior PAG coordinate scoring the identified set the route publishes, at the coordinate's own source-fixed level (0.95 frequentist, 0.90 Bayesian): one tally per fixed validation variant, named by its variant index, each checked to report the base identified set bit for bit; the coordinate list, truths and variants are source-fixed before any replicate runs, never chosen from outcomes.",
    "grid_temporal_effect_calibration.rs": "One test per supported temporal-effect coordinate through the shared sibling driver: the scored targets (labels and closed-form truths) are a caller-declared list handed to the driver before any replicate runs, at most three per test, at the two source-fixed levels; the sibling variants are the fixed validation modes and the accepted twin of the coordinate, never chosen from outcomes.",
    "grid_response_calibration.rs": "One test per group of supported static response coordinates that report one interval, through the shared sibling driver: the coordinates sharing a measured base are a source-fixed list, each checked to report the base interval, truth and scope on the first replicates before it is scored on the base; the scored targets (labels and closed-form truths) are a caller-declared list handed to the driver before any replicate runs, at most three per coordinate, at the two source-fixed levels; the sibling variants are the fixed validation modes and the accepted twin of each coordinate, never chosen from outcomes.",
    "grid_temporal_response_calibration.rs": "One test per supported temporal response coordinate through the shared sibling driver: the scored targets (labels and closed-form truths) are a caller-declared list handed to the driver before any replicate runs, at most three per test, at the two source-fixed levels; the sibling variants are the fixed validation modes and the accepted twin of the coordinate, never chosen from outcomes.",
    "grid_temporal_response_bayesian_calibration.rs": "One test per supported Bayesian temporal response coordinate through the shared sibling driver: the scored targets (labels and closed-form truths) are a caller-declared list handed to the driver before any replicate runs, at most three per test, at the two source-fixed levels; the sibling variants are the fixed validation modes and the accepted twin of the coordinate, never chosen from outcomes.",
    "grid_static_calibration.rs": "One test per supported static estimator coordinate through the shared sibling driver: the scored targets (labels and closed-form truths) are a caller-declared list handed to the driver before any replicate runs, at most three per test, at the two source-fixed levels; the sibling variants are the fixed validation modes and the accepted twin of the coordinate, never chosen from outcomes.",
    "grid_mechanism_design_calibration.rs": "One test per group of supported mechanism, transport and design coordinates that report one interval, through the shared sibling driver: the coordinates sharing a measured base are a source-fixed list, each checked to report the base interval, truth and scope on the first replicates before it is scored on the base; the scored targets (labels and closed-form truths) are a caller-declared list handed to the driver before any replicate runs, at most three per coordinate, at the two source-fixed levels; the sibling variants are the fixed validation modes and the accepted twin of each coordinate, never chosen from outcomes.",
    "fitted_model_calibration.rs": "Nineteen source-fixed designs of the fitted-model routes (change, root-cause, path, feature-relevance, anomaly and intrinsic-influence attributions, policy values, a temporal change and rollout, the posterior probability of a mechanism change and two structure posteriors) on linear-Gaussian laws with closed-form truths; labels are literal names of the published quantities fixed in source (total_change, mechanism_x, mechanism_y, the path, feature, anomaly and noise-source names, the policy quantities, the rollout steps), each scored at the reported 0.95 level and at 0.90 re-formed from the same draws; no label, level or coordinate is chosen from outcomes.",
    "dose_grid_statements_calibration.rs": "Five source-fixed bias-aware dose-grid designs (exact quadratic and cubic means with declared third-derivative bounds of zero and M) at the fixed bandwidth: each pointwise label is a literal quantity name (level, derivative) joined to a dose of the constant grid 1, 1.5, 2, 2.5, 3, and the one joint label is the literal band, each scored at the reported 0.95 level against a closed-form truth; no label, level or coordinate is chosen from outcomes.",
    "refuter_error_rates_calibration.rs": "Eleven source-fixed refuter error-rate designs (static and series, frequentist and Bayesian predictive check; sound procedure, spurious effect, misreported target): each label is the string id of a refuter kind taken from the design's literal kind list, and every record is a non-rejection rate at the source-fixed 1 - 0.05 level against a known null; the detection rates are printed, not asserted, and no kind, violation or level is chosen from outcomes.",
    "panel_did_statements_calibration.rs": "Eight source-fixed difference-in-differences designs (dynamic, group and calendar rollout aggregations, group-time and pre-trend bands, three unbalanced-panel estimators, an augmented panel): each label is the literal name carried by a declared target of the design's fixed target list (summary, each component, band) scored at the reported 0.95 level against a closed-form cell effect; no target, label or level is chosen from outcomes.",
    "fitted_model_statements_calibration.rs": "Twenty source-fixed designs of the fitted-model attribution and simulation routes (mechanism-refit bootstrap and Dirichlet or block-weight posteriors of path, feature, anomaly, unit-change, structure-change, robust-change, temporal component and influence targets, and of interventional samples, unit effects and counterfactuals) on linear-Gaussian laws with closed-form truths; labels are literal names of the published quantities in each test's fixed label list, each scored at the reported 0.95 level and at 0.90 re-formed from the same draws; no label, level or coordinate is chosen from outcomes.",
    "conjugate_prior_calibration.rs": "Seven source-fixed conjugate-prior designs (stated and misspecified Beta-Binomial and Gamma-Poisson priors; the select-then-estimate power-prior uniform interval at source displacements of 0, 1, 2, 4 and 8 target standard errors, its bias-aware interval at 0, 0.5 and 1 under a declared bound of one standard error, and the Beta-Binomial interval at five fixed truth and bound cells): each tally name is a literal design name joined to a displacement or truth taken from the design's fixed constant list, scored at the reported 0.95 level against a closed-form truth; no name, displacement, bound or level is chosen from outcomes.",
    "graph_posterior_weighted_target_calibration.rs": "Three source-fixed designs (a moderate and a thin overlap regime of the frequentist mixture, and a moderate regime of the Bayesian mixture) of the structure-posterior mixture for a kernel-weighted target: each label is a literal kernel centre and bandwidth from the design's fixed list, scored at the reported 0.95 level against the closed-form kernel-weighted effect of the planted modified effect; the two posterior structures and their masses are source-fixed, and no label, level or bandwidth is chosen from outcomes.",
    "temporal_design_statements_calibration.rs": "Twenty-five source-fixed temporal-design panels (pointwise coordinates, sup-t bands, Holm family errors, quartile effects, dose-grid derivative and bias-aware intervals, longer-horizon survival): each label is a literal coordinate name paired with its closed-form truth in the test's declared target list, or the literal band, scored at the reported 0.95 level; no coordinate, label or level is chosen from outcomes.",
}


# Every additional producer/consumer receives a named review; there is no generic
# approval for a new unlabelled constructor file.
SOURCE_REVIEW = {
    "longitudinal_schedule_calibration.rs": "Three source-fixed two-decision fair-coin laws, four binary schedules, two fixed candidates for the optimum multiplicity, and 199 subject-level draws. The fixed family scores coverage of all three contrasts to never-treat over the 400/800/1600 source grid. The other laws train on alternating rows and independently evaluate the chosen schedule and all six class contrasts. The selected-value target is the known population value of the realized trained schedule, paired with a separately predeclared nominal lower bound on inclusion of every true optimum. Neither the family, labels nor score construction is chosen from outcomes.",
    "longitudinal_regime_calibration.rs": "Source-fixed fair-coin always-treat designs. The four-decision study holds conditional censoring survival at 0.95, freezes external fold-indexed Q predictions, and scores the compiled 0.95 normal interval against the additive Gaussian outcome's value of six over 2400/4800/9600 subjects. The two uncensored two-decision studies use a binary baseline and additive Gaussian terminal outcome with population value 3.75 over 400/800/1600 subjects. Each scores the compiled 0.95 percentile interval of 199 whole-subject refits, with fixed internal fold exclusion or a separate external training fold. The targets, labels, interval levels and hypotheses are fixed before any measurement.",
    "fitted_dependent_measurement_calibration.rs": "One source-fixed labelled joint-indicator validation design with two construct rows and four fixed joint cells per row. The 150/300/600 sample-size grid, Dirichlet prior, equal-tail 0.95 interval, outcome coordinate and analytic truth are fixed before sampling; the runtime calibration basis is checked against the declared record. No label or target is chosen from outcomes.",
    "graph_narrow_policy_calibration.rs": "One source-fixed graph-checked held-out policy design with binary policy information x, wider nuisance adjustment x,w, known assignment probabilities and a 500/1000/2000-row grid. The executed action in each x stratum determines the exact conditional target before interval scoring; the 0.95 Wald method and runtime calibration basis are fixed. No label or target is chosen by observed coverage.",
    "graph_imputation_transport_calibration.rs": "One source-fixed root-treatment, binary pretreatment modifier design with source MAR outcomes and an independently sampled target covariate frame. The 60/120/240 source-and-target grid redoes imputation, graph scoring, child target standardization, fit and utility in every replicate. The 0.95 target-effect interval (Rubin's rules with Barnard-Rubin degrees of freedom across the completed tables) is taken from the compiled route and scored against analytic truth conditional on its target frame; the runtime basis is bound before coverage is seen. The pooled interval replaced order statistics of the two-table draw mixture, and the target standardization dropped a source bootstrap of frame means its weights already reproduce, after measured under-coverage; the construction is keyed separately.",
    "joint_transport_predictive_calibration.rs": "One source-fixed Gaussian trial with the same declared baseline and known noise in the target. The 60/120/240 source-and-target grid refits the producer, draws paired target outcomes through the compiler and scores the 0.95 nearest-rank treated-outcome interval against an independent future target unit. The executed model and target-noise basis is bound before coverage is seen; this is predictive coverage, not action-choice reliability.",
    "iid_sampling_assumption_union_calibration.rs": "One source-fixed IID Gaussian mean and fixed assumption range [0,1]. The 60/120/240 raw-unit grid reruns compiler-generated joint-row resampling and every chain for 200 bootstrap replicates per outer sample. The event is simultaneous coverage of the known identified-set endpoints [1,2], scored at 0.95 through a scalar-equivalent transform; the route's exact basis is bound before coverage is seen.",
    "graph_narrow_cross_fitted_policy_calibration.rs": "One source-fixed graph-checked three-fold policy learner uses binary x while nuisance adjustment uses x,w and known row-level assignment. The 500/1000/2000-row grid reruns the whole cross-fitting schedule and scores the distinct 0.95 candidate against the route's stated target, the population value of each realized fold policy weighted by its fold's share of the scored rows, including unstable fold policies. The target replaced the row-wise conditional means of the scored rows after a measured coverage of one showed those were not the quantity the influence-score interval estimates; the interval itself is unchanged. No interval or hypothesis is chosen from observed coverage.",
    "cyclic_posterior_predictive_calibration.rs": "One source-fixed two-equation Gaussian feedback system with known covariance, one fitted cyclic weight and two hard-clamp worlds. The 120/240/480 training-row grid refits the joint MCMC posterior, draws population disturbances, solves both worlds and compiles utility each replicate; the ordinary-world 0.95 outcome interval is scored against an independent future equilibrium under the same model. Diagnostics and runtime basis are checked before coverage is seen. The warmup-tuned random walk with dispersed chain starts replaced a fixed-step walk from one start after 9 of 400 replicates at the largest grid point failed the convergence gate.",
    "conjugate_prior_calibration.rs": "Fixed conjugate-posterior and select-then-estimate power-prior procedures, closed-form or declared truths and one reported confidence level; the discount selection is part of each scored procedure, never chosen from empirical coverage.",
    "calibration_coverage.rs": "Runtime coverage consumer/tests only; no Monte Carlo hypothesis producer.",
    "adaptive_inference_calibration.rs": "Fixed declared adaptive procedure, arm truths, stopping rules and source-selected confidence levels; whole-procedure target is fixed.",
    "adaptive_procedure_prior_predictive_calibration.rs": "Fixed prior-predictive generators and procedure configurations; calibration outcomes do not select the prior or hypothesis.",
    "cohort_linear_effect_calibration.rs": "Fixed cohort test names, source-declared effect/truth and confidence levels.",
    "inference_metrics.rs": "Error-rate tallies named from the caller's fixed test, covariance-kind/direction loop and sample size; each is a non-rejection rate at the source-fixed 1 - 0.05 level, never chosen from outcomes.",
    "candidate_calibration.rs": "Shared candidate tally constructor takes caller static test/DGP/method identity and declared confidence; callers are source-fixed gate cases.",
    "count_fitted_posterior_calibration.rs": "Fixed count-family posterior construction, truths and test confidence.",
    "counterfactual_transport_sampled.rs": "Fixed sampled counterfactual transport test configurations and static record keys.",
    "coverage_determinism.rs": "Small explicit deterministic test constructors; no measured outcome selects a family.",
    "joint_bayesian_transport_calibration.rs": "Fixed declared joint Bayesian transport cases/levels and static producer keys.",
    "knn_matching_calibration.rs": "One fixed matching interval whole procedure, source-named tally and confidence.",
    "learned_joint_calibration.rs": "Fixed learned-joint/query cases and declared confidence; the learning algorithm is part of each fixed procedure.",
    "observational_effects_calibration.rs": "Fixed source-named effect/policy procedures; policy selection is inside the scored fixed procedure, not selected from empirical coverage.",
    "quadratic_dose_calibration.rs": "Fixed quadratic-dose contrasts and test-declared confidence.",
    "recovery_whole_method_calibration.rs": "Fixed whole-method recovery cases including refusals; static truths/producer identities.",
    "temporal_unit_calibration.rs": "Fixed temporal units, query configurations and confidence levels.",
    "coverage_records_data.rs": "Embedded registry consumer/tests only; no calibration hypothesis constructor.",
    "batch_retarget_calibration.rs": "Finite prespecified retarget rows and batch contrasts with static test keys.",
    "prepared_adjusted_retarget_calibration.rs": "Two fixed adjusted-retarget interval procedures with source-declared Fisher/HC1 keys, fixed four-cell target and truth, and one 0.95 confidence level; observed coverage selects no hypothesis.",
    "calibration_admg.rs": "Fixed ADMG estimators, generative models and caller static test/level arguments.",
    "calibration_anomaly_bayesian.rs": "Fixed anomaly queries and Bayesian posterior constructions; static record identities.",
    "calibration_attribution_bayesian.rs": "Fixed attribution functionals and declared posterior confidence.",
    "calibration_design.rs": "Fixed randomized-design procedures and named confidence targets.",
    "calibration_design_bayesian.rs": "Fixed randomized Bayesian-design procedures/priors and confidence targets.",
    "calibration_distribution_admg.rs": "Finite fixed distribution-query configurations and static record keys.",
    "calibration_estimate.rs": "Prespecified estimator/DGP/test tuples; constructor levels are declared constants.",
    "calibration_posterior_draws.rs": "Source-fixed posterior draw-count/row-count design cases and confidence.",
    "class_posterior_calibration.rs": "Fixed class-posterior procedures and declared source test configurations.",
    "cluster_dml_calibration.rs": "Fixed whole-cluster DML procedure configurations/targets; no empirical label selection.",
    "dml_outcome_prior_calibration.rs": "Three fixed prior-strength constructions (weak, informative, misspecified) with a literal functional label; no labels selected using calibration outcomes.",
    "count_conjugate_calibration.rs": "Two fixed conjugate-count designs (Beta exceedance, Gamma rate) with literal test-named tallies at the declared level; truths are drawn from the stated prior and no outcome selects a label.",
    "calibration_bind.rs": "Shared immutable construction/record-key binding, no calibration outcomes choose hypotheses.",
    "reported.rs": "Shared per-test reporting wrapper; one source-fixed test execution context, no empirical hypothesis selection.",
    "cross_world_interval_calibration.rs": "Fixed cross-world interval/query cases and static record identifiers.",
    "evidence_synthesis_calibration.rs": "Fixed6 study counts across4 declared heterogeneity regimes; distinct source-fixed legacy/new pooling methods and confidence.",
    "joint_law_composed_interval_calibration.rs": "Finite prespecified joint-law composed query constructions and confidence.",
    "joint_sensitivity_calibration.rs": "Fixed sensitivity parameters, model/grid families and named confidence targets.",
    "learned_continuous_calibration.rs": "Fixed learned continuous procedures and generative configurations; learning is inside each fixed scoring procedure.",
    "missing_outcome_calibration.rs": "Prespecified missingness/outcome constructors and static test identities.",
    "multilevel_interval_calibration.rs": "Fixed multilevel interval/query cases and source-declared confidence.",
    "mz_transport_calibration.rs": "Fixed limited-experiment transport queries/configurations and static producer identities.",
    "predictive_score_interval_calibration.rs": "Prespecified predictive-score methods/truths/levels and source-fixed test names.",
    "randomized_interval_calibration.rs": "Fixed randomized procedures and source-declared interval levels.",
    "sample_split_effect_calibration.rs": "Fixed sample-split procedures and named interval targets.",
    "sample_split_weighted_effect_calibration.rs": "Fixed legacy half-split and current three-quarter discovery procedures; current treatment-invariant conflict rule is a fixed method, not selected from coverage outcomes. Producer version labels are finite, with one declared confidence level.",
    "scientific_bridge_interval_calibration.rs": "Finite prespecified scientific bridge queries, models and confidence. The two trial-pairs additions fix shared vs independent source law, equal sample counts, 399 refits and nominal 0.95 on the 40/80/160 grid. They use two literal hypotheses and do not choose them from outcomes.",
    "smoothed_dose_calibration.rs": "Source-fixed smoothing/dose query cases and static interval identifiers.",
    "static_envelope_calibration.rs": "Fixed envelope query/functionals; endpoint selection is inside the scored fixed algorithm.",
    "static_mixture_calibration.rs": "Prespecified static mixture/posterior constructions and source-fixed confidence.",
    "temporal_class_calibration.rs": "Fixed temporal graph-class procedures, declared horizons and confidence.",
    "temporal_recalc_calibration.rs": "Fixed temporal recalculation scenarios/queries and static keys.",
    "transport_statistical_calibration.rs": "Finite source-declared transport construction/target cases and confidence.",
    "z_transport_surrogate_calibration.rs": "Fixed surrogate/transport procedures and static producer test identities.",
    "choice_interval_calibration.rs": "Fixed pooled and hierarchical choice-share cells: option counts, respondents on the sample grid and declared hyperpriors; the scored share is drawn from the declared prior.",
    "graph_imputation_interval_calibration.rs": "Fixed graph-imputation prior-predictive cells with declared priors and effects; candidate measurements, no label chosen from outcomes.",
    "recovery_held_out_regimes.rs": "Fixed held-out clamp regimes of a known cyclic linear truth; coverage designs are source-declared and nothing records a coverage number from outcomes.",
    "calibration_binding.rs": "Facade-key binding check of the estimator-level record constructions; runs no Monte Carlo hypothesis and selects no label.",
    "learned_continuous_bootstrap_calibration.rs": "Fixed learned-continuous joint outer bootstrap procedure, the Good scenario of the declared DGP under both sampling designs and static record keys.",
    "randomized_quantile_interval_calibration.rs": "Fixed randomized-quantile median-effect design with a known truth and static record key; the quantile and arms are source-declared.",
    "matched_odds_ratio_interval_calibration.rs": "Fixed 1:1 matched-pair design with a declared conditional odds ratio and static record key.",
    "descriptive_comparison_interval_calibration.rs": "Fixed two-arm proportion design scored on two source-declared reporting scales with static record keys.",
    "nonlinear_mediation_interval_calibration.rs": "Fixed Gaussian-mediator quadratic-outcome design with closed-form natural effects; the three effect labels are source-declared and the record keys are the estimate's own calibration basis.",
    "mechanism_composition_interval_calibration.rs": "Fixed recursive Gaussian chain with closed-form interventional means, composed and transported variants, and static record keys.",
    "transport_weighted_target_calibration.rs": "Fixed planted-neighbourhood design with a declared kernel family and a source-declared bandwidth grid per test; the bandwidth labels and the record key are fixed in source.",
    "weighted_target_effect_calibration.rs": "Fixed planted heterogeneous-effect design with a Gaussian kernel target at a source-declared centre and a source-declared bandwidth grid per test; the centre and bandwidth labels, the closed-form truth and the record key are fixed in source.",
    "weighted_target_transport_calibration.rs": "Fixed planted effect-modification design (trial N(0,1), target N(1,1)) with a Gaussian kernel target at a source-declared centre and a source-declared bandwidth grid per test; the centre and bandwidth labels, the closed-form truth and the record key are fixed in source.",
    "typed_transported_mechanism_interval_calibration.rs": "Fixed Poisson source and shifted target design with closed-form interventional mean and a static record key.",
    "sequential_off_policy_interval_calibration.rs": "Fixed two-action off-policy design with declared logging and target policies, known target value and a static record key.",
    "bayesian_randomized_calibration.rs": "Twelve source-fixed designs of the Bayesian design posteriors (finite-population and superpopulation arm contrasts, switchback, complier effect, piecewise survival, dose policy, the few-cluster reference posterior and the model-based arm effect), each scoring one named coordinate's equal-tailed 0.95 posterior interval against its analytic truth with static record keys; no label, level or coordinate is chosen from outcomes.",
    "bayesian_dose_grid_calibration.rs": "Three source-fixed designs of the Bayesian randomized dose grid on one exactly quadratic law (level, derivative, contrast), each at one 0.95 level with a static record key and a prior fixed in source.",
    "temporal_routes_calibration.rs": "Six source-fixed designs of the sequential contrast, the sequential regime and binary policy posteriors and the within-period history routes (the cross-fit Wald interval and the modular posterior of the always-treat against never-treat contrast at five folds, the Bayesian-bootstrap regime value with frozen Q predictions under a fair coin, the Bayesian-bootstrap incremental policy value, the fixed-b period interval of the history match against the population effect on the treated, and the history outcome model's posterior against its design-conditional effect), each scoring the one interval the route publishes at one 0.95 level against a closed-form truth with static record keys; no label, level, fold count or coordinate is chosen from outcomes.",
    "temporal_wider_range_calibration.rs": "Fifty-four source-fixed designs repeating base temporal-design, sequential-contrast and regime-survival panels with one axis moved (decision lags two and three of every lag-taking design, assignment probabilities on each row's floor, a sequential panel whose true action probabilities reach 0.07 and 0.93, regime survival with thin treatment overlap and with heavy censoring), each scoring the one interval the route publishes at one 0.95 level against the base design's closed-form or simulated truth with static record keys; no lag, probability, censoring rate or coordinate is chosen from outcomes. In each wide sequential group reserve a second hypothesis slot for coverage conditional on publication, with fixed independent panel streams 0x5E20201 (Wald) and 0x5E20202 (Bayesian absolute-deviation radius). The first slot and its all-panel outcome remain historical. Only a typed fitted-overlap refusal redraws; a refusal count creates no additional hypothesis.",
    "temporal_design_calibration.rs": "Ten source-fixed temporal-design panels (two-arm, multi-arm, categorical, factorial, interference, three dose targets, quantile, regime survival), each scored on one named coordinate against its known truth in frequentist and Bayesian form at one 0.95 level with static record keys; no label and no coordinate is chosen from outcomes.",
    "measurement_composition_calibration.rs": "Sixteen source-fixed designs of the measurement corrections and the aligned posterior composition (binary prevalence and risk difference, classical error with declared and estimated variance, repeated measurements, the latent-posterior substitution and imputation routes, pooled differential transport, the two-arm temporal misclassification correction in both modes, and the contrast of two aligned posterior means), each scoring one named coordinate's published 0.95 interval against its closed-form truth with static record keys; the homogeneous pooled case is held one-sided because its interval is conservative at the variance boundary by construction.",
    "measured_exposure_calibration.rs": "Five source-fixed designs of the measured-exposure route (shared and outcome-differential misclassification, calibrated soft probabilities, covariate-dependent detection, repeat codings of a latent facet), each scoring the Rubin pooled 0.95 interval of the planted exposure effect against its closed-form truth; every estimator, functional, dependence and interval label is a literal at the test's call site and no label depends on a simulated outcome.",
    "measurement_exposure_calibration.rs": "Eight source-fixed designs of the exposure-error corrections (outcome-specific and shared validated exposure misclassification, soft exposures by substitution and by imputation, covariate-dependent detection declared exactly and with estimated parameters, binary and ordinal repeat codings), each scoring one named coordinate's published 0.95 interval against its closed-form truth; every estimator, functional and interval label is a literal in the source and no label depends on a simulated outcome.",
    "measurement_composition_horizon_calibration.rs": "Four source-fixed designs of the horizon composition interval on one two-lag linear law (cumulative effect and horizon contrast, joint block bootstrap and block-weight draws), each at the 0.95 level against the closed-form composed truth with static record keys.",
    "local_designs_calibration.rs": "Nine source-fixed local-ratio designs (fuzzy jump and kink at the MSE-optimal bandwidth, fuzzy jump at a declared bandwidth with the density test, sharp jump and kink in calendar time under AR(1) errors in bootstrap and block-posterior form, fixed-bandwidth Bayesian jump and kink on the model's own local-linear likelihood), each scoring the route's published 0.95 interval (and the 0.90 quantiles of the same posterior draws) against the generated ratio with static record keys; no label, bandwidth or level is chosen from outcomes.",
    "joint_response_calibration.rs": "Sixteen source-fixed joint-response designs (nine native designs in frequentist and Bayesian form, three held-out randomized selections at exact ties, near ties and a separated winner, one continuous structural selection, the reference decomposition against static and two-period worlds, and the experiment update), each scoring the interval the result publishes for one prespecified target at one 0.95 level with static record keys; the conformal contrast and both decomposition regions are held one-sided because they are Bonferroni constructions; a replicate whose runtime key leaves the cell's support condition is a skip, never a relabelled record.",
    "bounds_association_calibration.rs": "Ten source-fixed designs of the binary-instrument bounds and of the descriptive association: the Imbens-Manski row-bootstrap interval and the endpoint posterior quantiles of the bounds, each held one-sided against the true sharp lower and the true sharp upper bound of a known response-type mixture (an interval built for the whole identified set over-covers by construction), and the slope of a linear regression in six forms (standard error at the normal quantile and resampling spread for rows and for one autoregressive series, the conjugate marginal and the block Bayesian bootstrap), each scoring the one interval the route states at one 0.95 level against the population slope with static record keys on the design axes; no label, level or endpoint is chosen from outcomes.",
    "panel_did_calibration.rs": "Ten source-fixed difference-in-differences designs (the staggered rollout's simple aggregate under three estimators, both comparisons and both resampling schemes; the covariate-adjusted two-period effect under three nuisance methods; the augmented panel with the true predictions supplied; the rollout's Bayesian bootstrap and the two-period Normal-Inverse-Gamma posterior), each scoring the route's published 0.95 interval against the generated effect with static record keys; no label, estimator or level is chosen from outcomes.",
    "synthetic_control_calibration.rs": "Eight source-fixed synthetic-control designs (the donor-weight posterior's average effect; the predictive path's average, first and last post period; the conformal set of the first and last post period and of a constant effect at permutation counts that are multiples of twenty; the placebo-variance interval of three staggered adoptions among exchangeable units), each scoring the route's published 0.95 interval against the generated effect with static record keys; no label, period or level is chosen from outcomes. For the Bayesian predictive average reserve a second construction slot for the within-path average on the independent fixed stream xor 0x5041544800000000. The original marginal-average slot and observations are retained. Donors and post horizon are fixed inside each group, pre-treatment periods index its three planned grid looks, and returned-interval counts add no hypothesis.",
    "joint_panel_calibration.rs": "Source-fixed joint-panel designs (reviewed and broader Michaelis-Menten and Hill families, finite-lag and time-varying specifications, prepared joint response) whose record names are the literal test names fixed by the two test macros, each scored at the one declared 0.95 level against a closed-form or prior-drawn truth; no outcome selects a design, test or level.",
    "discovery_statements_calibration.rs": "Ten source-fixed structure-search designs (order-MCMC, structure-MCMC, CI-screened, ancestral-graph class, exact edge-marginal and bounded-lag template posteriors) on fixed linear laws with a known true structure class or a prior-drawn indicator, one literal test-named credible-set tally at the declared level each; no outcome selects a search, structure or level.",
    "panel_did_range_calibration.rs": "Two source-fixed difference-in-differences range designs (the two-period posterior at its row floor and the augmented panel with in-sample nuisance fits) with a closed-form truth and one declared 0.95 level; the design axes and test names are literal.",
    "synthetic_control_range_calibration.rs": "Ten source-fixed synthetic-control range designs (predictive-path intervals and conformal sets at one and three post periods) on the base design's panel with closed-form effects and literal pre-period counts chosen so the permutation size is exact; the design axes and test names are literal.",
    "bayesian_randomized_statements_calibration.rs": "Three source-fixed Bayesian randomized designs (finite-population Bernoulli arms under the sharp null and varying unit effects, and the cluster Bayesian bootstrap at twenty-five to forty-nine clusters per arm) with closed-form truths and the one declared 0.95 level; the design axes and test names are literal.",
    "bounds_association_range_calibration.rs": "Two source-fixed conjugate association designs at the literal prior scales 6 and 1000 on the base linear design with the population slope 2 and one declared 0.95 level; the scales are fixed before any run and no outcome selects a design or level.",
    "measurement_composition_statements_calibration.rs": "Twenty-five source-fixed measurement-correction and composition designs (categorical correction, regression calibration, SIMEX, temporal misclassification beyond two arms, composed sums, weighted sums, products and ratios) with closed-form truths and one declared 0.95 level; the design axes and test names are literal.",
    "temporal_routes_range_calibration.rs": "Four source-fixed sequential-contrast designs (ten balanced and eight unbalanced cross-fit folds, Wald and posterior intervals) on the base panel with the closed-form contrast 2.82 and one declared 0.95 level; the fold splits are literal cycles and the test names are literal.",
    "temporal_design_range_calibration.rs": "Four source-fixed regime-survival designs (horizons one and two, bootstrap and posterior intervals) on the base survival law with a truth simulated from that law and one declared 0.95 level; the horizons and test names are literal.",
    "delayed_entry_incidence_calibration.rs": "Six source-fixed two-cause randomized designs with independent delayed entry and either unweighted or fixed known censoring survival. Two score the pointwise 0.95 target-cause interval at horizon three, and four score the simultaneous 0.95 whole-grid bands for survival and incidence. A seventh source-fixed design scores the whole pre-entry standardized survival contrast under baseline-dependent entry and censoring, using exact source mixture weights and refitting the same-subject censoring fit. A prespecified closed-form curve at all observed grid points defines each binary whole-grid coverage event; no outcome selects a cause, horizon, method or record.",
    "bounds_association_statements_calibration.rs": "Fifteen source-fixed association designs (stratum mean and probability, logistic and Poisson slopes, Pearson and distance correlation, row and block bootstrap spread, Bayesian bootstrap, temporal stratum mean) with closed-form truths and one declared 0.95 level; the design axes and test names are literal.",
    "local_designs_statements_calibration.rs": "Seven source-fixed local-ratio designs (declared-bandwidth kink with diagnostics, temporal fuzzy jump, kink and step at the selected bandwidth, and their Bayesian counterparts) with a closed-form ratio truth and one declared 0.95 level; the design axes and test names are literal.",
    "temporal_routes_statements_calibration.rs": "One source-fixed sequential-regime design (the Bayesian-bootstrap posterior with cross-fitted Q predictions) on the base two-decision trial with the closed-form value 3.82 and one declared 0.95 level; the design axes and test name are literal.",
    "conjugate_routes_calibration.rs": "Nine source-fixed designs of the conjugate count routes and the select-then-estimate power priors (exceedance and rate under a stated or source-count prior drawn from itself; the Beta-Binomial, Gamma-Poisson and Gaussian-linear selections at a displaced truth for the uniform interval and a truth inside the declared bound for the bias-aware one), each scoring the one interval the route reports at the one declared 0.95 level against the drawn or fixed truth with literal record keys; no label, level, bound or truth is chosen from outcomes.",
    "graphless_wider_range_calibration.rs": "Source-fixed designs that repeat base designs of the cluster Bayesian bootstrap, synthetic control, panel difference-in-differences, binary-instrument bounds and association routes at support ranges the base grids do not reach (more clusters per arm, more donors, other post-period counts, pre-period counts that are not multiples of twenty, and sizes four times the base grid's largest); each scores the route's published 0.95 interval against the generated truth with static record keys, and no label, size or level is chosen from outcomes. For the Bayesian predictive average reserve a second construction slot for the within-path average on the independent fixed stream xor 0x5041544800000000. The original marginal-average slot and observations are retained. Donors and post horizon are fixed inside each group, pre-treatment periods index its three planned grid looks, and returned-interval counts add no hypothesis.",
    "bridge_transport_premise_calibration.rs": "One source-fixed pair of independent Gaussian source and target trial lines on a fixed surrogate design, differing by a known intercept. The 60/120/240-trial-per-scope grid refits both lines through the compiler and scores simultaneous Bonferroni 0.95 Wald discrepancy bands on five prespecified overlap points against the known target-minus-source line; the runtime design basis is bound before coverage is seen. No label or target is chosen from outcomes.",
    "cluster_sampling_assumption_union_calibration.rs": "One source-fixed design of independent Gaussian clusters, each contributing two correlated raw unit rows, with fixed assumption range [0,1]. The 60/120/240-row grid regenerates clusters, reruns 200 compiler-generated complete-cluster bootstrap resamples and recompiles the whole chain; the event is joint containment of the known set [1,2] at 0.95 under the bound cluster-design basis.",
    "continuous_chain_transport_calibration.rs": "One source-fixed randomized linear sequential DAG with independent structural errors and independent target baselines. The 120/240/480-unit grid scores the 200-resample paired arm-stratified 0.95 percentile whole-contrast interval against analytic truth 2; the runtime basis is bound and refusals are counted. No label or target is chosen from outcomes.",
    "event_count_calibration.rs": "One source-fixed exponential-kernel Hawkes cluster process simulated independently of the fitted model. The history windows are 128, 256 and 512 time units; the record scopes each point by its measured event counts. Each replicate refits the forecast (16 refits, 256 paths) and scores the 0.90 smoothed-rank count interval on a fixed two-unit window against the simulated future count; the runtime basis is bound before coverage is seen. The smoothed-rank construction, separately keyed, replaced inclusive nearest-rank quantiles after a measured over-coverage traced to their always-included tied boundary counts.",
    "graph_imputation_binary_cluster_calibration.rs": "Two source-fixed binary-target graph-imputation designs (randomized and pretreatment-confounded assignment) with MAR source outcomes and independent two-row target clusters. The 60/120/240-row grid reruns 40 whole-cluster target resamples of the full graph, imputation, transport, fit and decision schedule, each on its own imputation and posterior streams, and scores the 0.95 replicate order-statistic effect interval against known target-population truth; the runtime basis is bound before coverage is seen. The replicate order statistics over per-replicate streams replaced a forty-draw nearest-rank interval over one shared stream after measured under-coverage; the construction is keyed separately.",
    "graph_imputation_kernel_cluster_calibration.rs": "Two source-fixed continuous-target graph-imputation designs (randomized and pretreatment-confounded assignment) with MAR source outcomes and independent two-row target clusters. The 60/120/240-row grid reruns 40 whole-cluster target resamples of the full graph, imputation, mean-calibrated kernel transport, fit and decision schedule, each on its own imputation and posterior streams, and scores the 0.95 replicate order-statistic effect interval against known target-population truth; the runtime basis is bound before coverage is seen. The weakly informative prior (scale 10) replaced a scale-1 prior that shrank the effect, and the replicate order statistics over per-replicate streams replaced a forty-draw nearest-rank interval over one shared stream, after a measured coverage near one half; the construction is keyed separately.",
    "graph_imputation_kernel_transport_calibration.rs": "One source-fixed design with continuous source and target covariate frames, randomized binary treatment, MAR Gaussian outcomes and known linear effect modification. The 60/120/240-row grid reruns imputation, graph selection, kernel target standardization calibrated to the frame's covariate means, child posterior fit and utility, and scores the 0.95 target-frame interval (Rubin's rules with Barnard-Rubin degrees of freedom across the completed tables) against analytic truth conditional on the frame; the runtime basis is bound before coverage is seen. The weakly informative prior (scale 10) replaced a scale-1 prior that shrank the effect toward zero, and the mean calibration and pooled interval replaced the fixed-bandwidth ratio and mixture order statistics, after a measured coverage near one half; the construction is keyed separately.",
    "graph_imputation_mar_cluster_calibration.rs": "Four source-fixed declared-MAR target-cluster designs (binary or kernel target, randomized or confounded assignment) with a partially observed modifier whose known observation probability depends on an always-observed covariate. Each replicate refits the 40-resample graph-imputation transport and utility schedule, each resample on its own imputation and posterior streams, and scores the 0.95 replicate order-statistic interval against the analytic target-population effect under the bound runtime scope; labels come from the four fixed test functions. The selection targets covering every pretreatment covariate and the 100/200/400-row grid replaced a declaration every replicate refused as not S-admissible and a 60-row base point where a stratum could miss an arm; no coverage was scored under the old declaration.",
    "graph_imputation_predictive_calibration.rs": "Four source-fixed individual-outcome designs (Bernoulli or Gaussian outcome, control or active world) with MAR source outcomes and an independent binary target frame. Each replicate recompiles graph selection, imputation, transport, predictive worlds and the joint decision, checks the world-specific runtime basis and scores the 0.95 interval against a future individual drawn from the frame (its generated outcome when Gaussian, its known success probability when Bernoulli, pooled across the completed tables by Rubin's rules on the logit scale); labels come from the four fixed test functions. Scoring the success probability replaced scoring a future 0/1 outcome, which a [0,1] interval always covered, after a measured coverage of one, and a residual stream keyed by the execution seed replaced one fixed stream after measured under-coverage of the active world; the construction is keyed separately.",
    "graph_imputation_predictive_cluster_calibration.rs": "Four source-fixed predictive designs (Bernoulli or Gaussian outcome, control or active world) under independent two-row target clusters and a known response law. Graph, imputation and target predictive fit rerun across 40 whole-cluster resamples, each on its own imputation and posterior streams; a future population individual (its outcome when Gaussian, its success probability when Bernoulli) and the known-optimum decision are scored under their bound scopes; labels come from the four fixed test functions. Scoring the success probability replaced scoring a future 0/1 outcome after a measured coverage of one, and per-replicate streams replaced one shared stream; the construction is keyed separately.",
    "graph_narrow_cross_fitted_subsampling_policy_calibration.rs": "The source-fixed graph-checked three-fold policy design of graph_narrow_cross_fitted_policy_calibration on a near-tie outcome law. The 500/1000/2000-row grid reruns all folds with 100 half-sample subsamples and scores the basic subsampling 0.95 interval, with the finite-population sqrt(m/(n-m)) deviation scaling and (B + 1) p quantile positions, against the route's stated target, the population value of each realized fold policy weighted by its fold's share of the scored rows, after an independent replay. The target replaced the row-wise conditional means of the scored rows, and the scaling and positions replaced sqrt(m/n) and type-7 quantiles, after a measured coverage of one; the construction is keyed separately. No label or target is chosen by observed coverage.",
    "sampled_temporal_path_transport_calibration.rs": "One source-fixed finite-SCM mediation design with covariate shift; every cited source and target regime is sampled independently. The 20000/40000/80000-unit-per-regime grid scores the clamped paired-delta 0.95 whole-contrast interval against the finite oracle and binds the runtime basis; refusals are counted. One uniform draw per sampled unit replaced a sampler that redrew inside its binary search after most replicates were refused for unseen cells; no coverage was scored under the old sampler's law.",
    "trial_ipw_predictive_calibration.rs": "One source-fixed known-randomization trial and independent target units sharing a Gaussian outcome law with a known treatment effect. Each replicate fits the normalized IPW predictive decision, scores the 0.95 nearest-rank treated-outcome interval against a fresh target outcome, checks the exact interval basis and action-choice scope, and reports the true-optimum choice rate.",
}
PATH_REVIEW = {
    "crates/antecedent-io-core/src/calibration.rs": "Runtime calibration lookup and its test fixtures; no Monte Carlo family producer.",
    "crates/antecedent/tests/common/calibration_design_counts.rs": "Reports auxiliary counts and tests nominal coverage at four exact 40-donor panel designs on an independent stream. Their final look, precision and simultaneous equivalence bounds are fixed before validation. The second sequential tally sums refusals and has no independent statistical assertion or coverage record.",
    "crates/antecedent/tests/common/calibration.rs": "Shared tally definition plus bounded unit fixtures; actual named hypothesis labels/levels come from audited caller sources.",
    "crates/antecedent-estimate/src/bayesian/tests.rs": "Four fixed conjugate temporal coverage designs with literal tally names at the source-fixed 90% level; the equal-tailed interval rule is fixed before simulation.",
    "crates/antecedent-estimate/src/dml.rs": "Four fixed learner methods with literal tally names at the fixture's declared nominal level; one source-fixed DGP and truth.",
    "crates/antecedent/tests/common/grid_cell.rs": "Shared sibling driver with no hypothesis of its own: the caller hands it the coordinate (test, DGP, interval, estimator, variant count) and a declared target list (label and closed-form truth) before any replicate runs. It makes one record tally per variant, target and level at the two source-fixed levels (reported 0.95 and gate 0.90), so a test asserts at most variants x targets x 2 cells per grid point, all fixed by the caller's source. A label is the target's declared label. The variants are the caller's fixed validation modes and accepted twin; each sibling runs on the first two replicates, must report the base interval bit for bit, and its record is bound from its own execution (runtime match key and validation mode) and scored on those base intervals. No observed coverage chooses a label, target or variant.",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def producer_calls(text: str, labels: bool = True) -> list[dict]:
    source = lex(text)
    out = []
    pattern = r"\.labelled\s*\(" if labels else r"\bCoverageTally\s*::\s*(?:new|for_record)\s*\("
    for match in re.finditer(pattern, source.skeleton):
        start = match.end()
        depth = 1
        end = start
        while end < len(source.skeleton) and depth:
            char = source.skeleton[end]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            end += 1
        if depth:
            raise ValueError("unclosed label producer")
        argument = source.kept[start : end - 1].strip()
        out.append(
            {
                "line": text.count("\n", 0, match.start()) + 1,
                "call": source.kept[match.start() : start].strip(),
                "argument": argument,
            }
        )
    return out


def labelled_calls(text: str) -> list[dict]:
    return producer_calls(text)


def _source_paths(root: Path, sha: str | None) -> list[str]:
    if sha:
        result = subprocess.run(
            ["git", "grep", "-l", "CoverageTally", sha, "--", "crates"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise ValueError(result.stderr.strip())
        return sorted(
            line.partition(":")[2] for line in result.stdout.splitlines() if line.endswith(".rs")
        )
    return sorted(
        path.relative_to(root).as_posix()
        for path in (root / "crates").rglob("*.rs")
        if "CoverageTally" in path.read_text(encoding="utf-8")
    )


@cache
def _immutable_source(root: Path, path: str, sha: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{sha}:{path}"], cwd=root)


def read_source(root: Path, path: str, sha: str | None = None) -> bytes:
    if sha and path == GATE_PATH:
        for candidate in (path, *EARLIER_GATE_PATHS):
            try:
                return _read_source_at(root, candidate, sha)
            except subprocess.CalledProcessError:
                if candidate == EARLIER_GATE_PATHS[-1]:
                    raise
    return _read_source_at(root, path, sha)


def _read_source_at(root: Path, path: str, sha: str | None) -> bytes:
    if sha:
        # Cache complete object identities; mutable refs and live files stay fresh.
        if re.fullmatch(r"[0-9a-f]{40}", sha):
            return _immutable_source(root.resolve(), path, sha)
        return subprocess.check_output(["git", "show", f"{sha}:{path}"], cwd=root)
    return (root / path).read_bytes()


def reviewed_label_audit(root: Path = ROOT) -> dict:
    sources = []
    for path in _source_paths(root, None):
        data = read_source(root, path)
        calls = labelled_calls(data.decode())
        review = (
            PATH_REVIEW.get(path)
            or LABEL_REVIEW.get(Path(path).name)
            or SOURCE_REVIEW.get(Path(path).name)
        )
        if not review:
            raise ValueError(
                f"new tally producer/consumer requires explicit source-fixed-family review: {path}"
            )
        sources.append(
            {
                "path": path,
                "sha256": _sha(data),
                "labelled_calls": calls,
                "constructor_calls": producer_calls(data.decode(), labels=False),
                "review": review,
            }
        )
    return {
        "schema": SCHEMA,
        "reviewed_cell_cap_per_group": decision.CELL_CAP,
        "review_scope": "Frozen coverage compatibility hypotheses; every level/label is a separate cell. Labels and procedures are fixed before their current prospective validation outcomes; method revisions informed by older data require an independent registered domain and a distinct cumulative hypothesis slot.",
        "source_possible_cap_invariant": "Every group/grid has at most256 source-possible asserted cells across all branch outcomes and registered looks, including recordless, conditionally empty and alternate-confidence tallies. Runtime emitted-count enforcement only complements this source review.",
        "transitive_provider_review": "Caller-provided labels, target names and levels were reviewed at fixed test/query construction sites. Imported functional enums and numeric helper values describe fixed algorithms; none select a coverage hypothesis from the current prospective validation outcomes. Older development evidence may motivate a separately allocated method revision before its independent validation. The direct constructor/label footprint and named reviews are an audited invariant, not an automated transitive dataflow proof.",
        "prospective_validation_domain": decision.DOMAIN,
        "historical_cardinality_audit": {
            "source_sha": "c30e10304f4bd85a01ad81dd133a360ddc95c2f1",
            "group": "temporal_response_calibration",
            "maximum_distinct_tally_ids": 135,
            "source_possible_upper_bound": 162,
            "source_possible_derivation": "42 curve +24 persistent-treatment +6 intervention +6 sequence +6 Bayesian-intervention +8 selected-observation +6 observation-sequence +10 horizon-dependent +18 CPDAG atom +36 PAG atom cells; adjustment labels range over all fixed-graph eligible subsets, including unobserved subsets.",
            "basis": "Source-defined construction cardinality only; observed coverage values were not used to choose the cap.",
        },
        "sources": sources,
    }


def source_validation_domain(root: Path = ROOT, sha: str | None = None) -> str:
    """Bind historical metadata to its measured source's literal domain declaration."""
    if sha is None:
        return decision.DOMAIN
    tree = ast.parse(read_source(root, "scripts/calibration_decision.py", sha))
    declarations = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DOMAIN" for target in node.targets)
    ]
    if len(declarations) != 1 or not isinstance(declarations[0], ast.Constant):
        raise ValueError("measured source lacks one literal validation domain")
    domain = declarations[0].value
    if not isinstance(domain, str) or domain not in decision.DOMAINS:
        raise ValueError("measured source declares an unregistered validation domain")
    return domain


def validate_label_audit(audit: dict, root: Path = ROOT, sha: str | None = None) -> None:
    if (
        audit.get("schema") != SCHEMA
        or audit.get("reviewed_cell_cap_per_group") != decision.CELL_CAP
        or audit.get("prospective_validation_domain") != source_validation_domain(root, sha)
    ):
        raise ValueError("unknown source-fixed label audit")
    sources = audit.get("sources")
    if not isinstance(sources, list) or [row.get("path") for row in sources] != _source_paths(
        root, sha
    ):
        raise ValueError(
            "tally producer footprint changed; explicitly review the complete source family"
        )
    for row in sources:
        data = read_source(root, row["path"], sha)
        if (
            row.get("sha256") != _sha(data)
            or row.get("labelled_calls") != labelled_calls(data.decode())
            or row.get("constructor_calls") != producer_calls(data.decode(), labels=False)
            or not row.get("review")
        ):
            raise ValueError(f"source-fixed label review is stale: {row['path']}")


def group_slots(table: dict):
    """Yield each permanent family slot; reserved slots have no runnable study."""
    rows = table["group"]
    reserved = table.get("reserved_group_indices", [])
    size = len(rows) + len(reserved)
    if (any(type(i) is not int or not 1 <= i <= size for i in reserved)
            or len(set(reserved)) != len(reserved)):
        raise ValueError("invalid reserved calibration group indices")
    unused = set(reserved)
    active = iter(rows)
    for index in range(1, size + 1):
        yield index, None if index in unused else next(active)


def gate_roster(gate: bytes, acceptance: bytes) -> list[dict]:
    """The complete roster: the group table's rows, or an earlier shell driver's dry run."""
    if not gate.startswith(b"#!"):
        import tomllib
        slots = group_slots(tomllib.loads(gate.decode("utf-8")))
        groups = [{"index": i, "label": row["label"], "grid": bool(row["grid"])} if row is not None
                  else {"index": i, "label": f"reserved slot {i}", "grid": False} for i, row in slots]
        return _complete_roster(groups)
    # Execute only the earlier driver's dry-run path in a scratch source tree.
    # This obtains the complete roster even when the caller selected a shard.
    with tempfile.TemporaryDirectory(prefix="calibration-family-") as directory:
        root = Path(directory)
        (root / "scripts").mkdir()
        (root / "parity").mkdir()
        (root / EARLIER_GATE_PATHS[0]).write_bytes(gate)
        (root / ACCEPTANCE_PATH).write_bytes(acceptance)
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("ANTECEDENT_CALIBRATION_")
        }
        env.update(
            ANTECEDENT_CALIBRATION_DRY_RUN="1",
            ANTECEDENT_CALIBRATION_NSIM="400",
            ANTECEDENT_CALIBRATION_RECHECK_NSIM="2000",
        )
        result = subprocess.run(
            ["bash", str(root / EARLIER_GATE_PATHS[0])],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
    grid = {int(m.group(1)) for m in re.finditer(r"^grid (\d+)$", result.stdout, re.MULTILINE)}
    groups = [
        {"index": int(m.group(1)), "label": m.group(2), "grid": int(m.group(1)) in grid}
        for m in re.finditer(r"^group (\d+): (.+)$", result.stdout, re.MULTILINE)
    ]
    return _complete_roster(groups)


def _complete_roster(groups: list[dict]) -> list[dict]:
    if (
        not groups
        or [g["index"] for g in groups] != list(range(1, len(groups) + 1))
        or len({g["label"] for g in groups}) != len(groups)
        or len({g["label"].translate(str.maketrans(" /:", "___")) for g in groups}) != len(groups)
    ):
        raise ValueError("the gate did not give a complete unique source roster")
    return groups


def build_manifest(root: Path = ROOT, audit: dict | None = None) -> dict:
    audit = audit or json.loads((root / AUDIT_PATH).read_text(encoding="utf-8"))
    validate_label_audit(audit, root)
    gate = read_source(root, GATE_PATH)
    acceptance = read_source(root, ACCEPTANCE_PATH)
    manifest = {
        "schema": SCHEMA,
        "decision_protocol": decision.ID,
        "validation_domain": decision.DOMAIN,
        "family_alpha": decision.ALPHA,
        "family_grid_points": decision.GRID_POINTS,
        "cell_cap_per_group": decision.CELL_CAP,
        "looks": [dict(x) for x in decision.LOOKS],
        "gate_sha256": _sha(gate),
        "acceptance_sha256": _sha(acceptance),
        "label_audit_sha256": decision.digest(audit),
        "groups": gate_roster(gate, acceptance),
    }

    if decision.DOMAIN != decision.LEGACY_DOMAIN:
        manifest["hypothesis_allocations_sha256"] = allocation_digest(
            root, None, manifest["groups"]
        )
    return manifest


def original_roster(root: Path = ROOT) -> list[dict]:
    """The complete roster of the original frozen family, read from its immutable source."""
    manifest = json.loads(read_source(root, MANIFEST_PATH, allocations.ORIGINAL_SOURCE))
    return manifest["groups"]


def allocation_digest(root: Path, sha: str | None, groups: list[dict]) -> str:
    """Digest of the cumulative allocation receipt, validated against the original roster.

    The receipt's risk budget, slot owners and indices belong to the original family
    (`allocations.ROSTER_DIGEST` is the digest of that roster, so it never moves when the
    gate gains a group). `groups`, the roster of the tree being frozen, only has to still
    contain every allocation owner by label."""
    raw = read_source(root, allocations.PATH, sha)
    allocations.validate(json.loads(raw), original_roster(root), source_validation_domain(root, sha))
    owners = {allocations.GROUP, allocations.PAG_EXTENSION["group"]["label"]}
    if not owners <= {group["label"] for group in groups}:
        raise ValueError("an allocation owner group is absent from the gate roster")
    return _sha(raw)


def validate_manifest(
    manifest: dict, audit: dict, root: Path = ROOT, sha: str | None = None
) -> dict:
    validate_label_audit(audit, root, sha)
    gate = read_source(root, GATE_PATH, sha)
    acceptance = read_source(root, ACCEPTANCE_PATH, sha)
    expected = {
        "schema": SCHEMA,
        "decision_protocol": decision.ID,
        "validation_domain": source_validation_domain(root, sha),
        "family_alpha": decision.ALPHA,
        "family_grid_points": decision.GRID_POINTS,
        "cell_cap_per_group": decision.CELL_CAP,
        "looks": [dict(x) for x in decision.LOOKS],
        "gate_sha256": _sha(gate),
        "acceptance_sha256": _sha(acceptance),
        "label_audit_sha256": decision.digest(audit),
        "groups": gate_roster(gate, acceptance),
    }
    if expected["validation_domain"] != decision.LEGACY_DOMAIN:
        expected["hypothesis_allocations_sha256"] = allocation_digest(root, sha, expected["groups"])
    if manifest != expected:
        raise ValueError("family manifest differs from its complete frozen source roster/audit")
    return decision.config_for_manifest(manifest)


@cache
def config_at_sha(sha: str, root: Path = ROOT) -> dict:
    if not re.fullmatch("[0-9a-f]{40}", sha):
        raise ValueError("family provenance requires a full source SHA")
    manifest = json.loads(read_source(root, MANIFEST_PATH, sha))
    audit = json.loads(read_source(root, AUDIT_PATH, sha))
    return validate_manifest(manifest, audit, root, sha)


def followup_allocation(config: dict) -> tuple[str, dict] | None:
    """A new random domain spends only its explicitly reserved construction slot."""
    decision.validate_config(config)
    domain = config["validation_domain"]
    if domain in (decision.LEGACY_DOMAIN, decision.MEASUREMENT_DOMAIN):
        return None
    if domain == decision.HC3_DOMAIN:
        return allocations.GROUP, {
            "record_id": allocations.RECORD + "_variance_hc3",
            "functional": allocations.FUNCTIONAL + ".variance_hc3",
            "interval_method": "student_t_sandwich_effective_n",
            "se_kind": "hc3_sampling_weighted",
            "nominal": 0.95,
        }
    if domain == decision.PAG_DOMAIN:
        entry = allocations.PAG_EXTENSION
    elif domain == decision.PAG_CONFIDENCE_SET_DOMAIN:
        entry = allocations.PAG_CONFIDENCE_SET_EXTENSION
    else:
        raise ValueError("unallocated followup validation domain")
    return entry["group"]["label"], entry["new_slot"]


def validate_measurement_selection(config: dict, labels: list[str]) -> None:
    """Reject an unallocated new-domain group before build or numerical dispatch."""
    allocation = followup_allocation(config)
    if allocation is not None:
        if not labels or set(labels) != {allocation[0]}:
            raise ValueError("followup validation domain permits only its allocated group")
        if (config["family_groups"], config["family_cells"]) != (757, 581376):
            raise ValueError("followup validation changed the original risk budget")
        if (
            config["validation_domain"] == decision.HC3_DOMAIN
            and os.environ.get("ANTECEDENT_WEIGHTED_SPLIT_PROTOCOL", "3") != "3"
        ):
            raise ValueError("HC3 validation requires its allocated protocol3 construction")
        return
    if config["validation_domain"] == decision.MEASUREMENT_DOMAIN:
        # An ordinary measurement admits every group of the frozen current roster.
        current = json.loads(read_source(ROOT, MANIFEST_PATH))
        if not labels or not set(labels) <= {group["label"] for group in current["groups"]}:
            raise ValueError("measurement group is absent from the frozen current roster")
        return
    original = json.loads(read_source(ROOT, MANIFEST_PATH, allocations.ORIGINAL_SOURCE))
    roster = {group["label"] for group in original["groups"]}
    if not set(labels) <= roster:
        raise ValueError("legacy validation group is absent from the original roster")


def validate_measurement_construction(config: dict, label: str, record: dict) -> None:
    """No emitted foreign construction can inherit a reserved domain's evidence."""
    allocation = followup_allocation(config)
    if allocation is None:
        return
    expected_label, expected = allocation
    if label != expected_label or any(
        record.get("id" if key == "record_id" else key) != expected[key]
        for key in ("record_id", "functional", "interval_method", "se_kind", "nominal")
    ):
        raise ValueError("unallocated construction in followup validation domain")


def current_config(root: Path = ROOT) -> dict:
    return validate_manifest(
        json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8")),
        json.loads((root / AUDIT_PATH).read_text(encoding="utf-8")),
        root,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--ack-source-fixed-label-review", action="store_true")
    args = parser.parse_args()
    if args.refresh:
        if not args.ack_source_fixed_label_review:
            raise SystemExit(
                "refresh requires explicit --ack-source-fixed-label-review after reviewing finite label/target producers"
            )
        audit = reviewed_label_audit()
        manifest = build_manifest(audit=audit)
        (ROOT / AUDIT_PATH).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        (ROOT / MANIFEST_PATH).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    config = current_config()
    print(decision.canonical_json(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
