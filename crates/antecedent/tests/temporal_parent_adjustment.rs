//! Single-step Pulse on a `TemporalDag` whose treatment is autoregressive.
//!
//! Unfolding cannot certify an autoregressive treatment (its ancestry never
//! ends), so the analysis identifies the pulse by adjusting for the treatment's
//! own parents `pa(T[t])` (derivation rule `temporal.parent_adjustment`). These
//! tests pin: the known effect is recovered where the analysis used to refuse;
//! parent adjustment and unfolding agree where both certify; multi-step
//! schedules, unoriented treatment edges and parents outside the permitted
//! history still refuse; the contract records the rule and set, and they
//! survive export and consumption.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::doc_markdown)]

mod common;

use antecedent::{AcceptedGraph, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    Assumption, CausalQuery, ExecutionContext, IdentificationStatus, Lag, TemporalEffectQuery,
    TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_estimate::{EstimationWorkspace, TemporalLinearAdjustment};
use antecedent_graph::{TemporalCpdag, TemporalDag};
use antecedent_identify::{
    IdentificationError, PARENT_ADJUSTMENT_RULE, TemporalBackdoorIdentifier,
    TemporalIdentificationResult,
};
use common::persistent_dgp::{
    AR_BETA, AR_PHI, autoregressive_pulse_dag, autoregressive_pulse_series, pulse,
};

const N: usize = 2000;

fn study(data: TimeSeriesData, graph: AcceptedGraph, query: TemporalEffectQuery) -> Study {
    Study::series(data)
        .graph(graph)
        .query(CausalQuery::TemporalEffect(query))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn rules(result: &StudyResult) -> Vec<String> {
    result.identification.derivation.steps.iter().map(|step| step.rule.to_string()).collect()
}

#[test]
fn autoregressive_pulse_was_refused_and_is_now_identified_by_parent_adjustment() {
    let graph = autoregressive_pulse_dag(true, false);
    // Unfolding alone (the behaviour before the fallback) cannot certify.
    let Err(IdentificationError::NotCertified { message }) =
        TemporalBackdoorIdentifier::new().identify_temporal(&graph, &pulse())
    else {
        panic!("unfolding must not certify an autoregressive treatment");
    };
    assert!(message.contains("lagged cycle"), "{message}");

    let data = autoregressive_pulse_series(N, AR_PHI, false, 7);
    let ctx = ExecutionContext::for_tests(7);
    for accepted in [false, true] {
        let graph = if accepted {
            AcceptedGraph::temporal_dag(graph.clone())
        } else {
            graph.clone().into()
        };
        let result = study(data.clone(), graph, pulse()).run(&ctx).expect("identified");
        assert_eq!(result.identification.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(
            rules(&result).iter().any(|rule| rule == PARENT_ADJUSTMENT_RULE),
            "{:?}",
            rules(&result)
        );
        // pa(t[-1]) = {t[-2], z[-2]}.
        assert_eq!(result.estimand.adjustment_set.len(), 2);
        assert!(
            (result.estimate.ate - AR_BETA).abs() < 0.05,
            "accepted={accepted}: ate={} truth={AR_BETA}",
            result.estimate.ate
        );
    }
}

/// Every licensed `PulseEffect × TemporalDag` coordinate on fixed structure
/// takes the same parent-adjusted identification: Frequentist and Bayesian,
/// explicit and accepted, validation none / cheap / full, through the staged
/// prepare → estimate handle.
#[test]
fn every_fixed_structure_pulse_coordinate_uses_parent_adjustment() {
    let data = autoregressive_pulse_series(1200, AR_PHI, false, 14);
    let ctx = ExecutionContext::for_tests(14);
    let graph = autoregressive_pulse_dag(true, false);
    for bayesian in [false, true] {
        for accepted in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let structure = if accepted {
                    AcceptedGraph::temporal_dag(graph.clone())
                } else {
                    graph.clone().into()
                };
                let inference = if bayesian {
                    InferenceMode::Bayesian(
                        antecedent::BayesianConfig::conjugate().n_draws(400).prior_scale(100.0),
                    )
                } else {
                    InferenceMode::Frequentist
                };
                let prepared = Study::series(data.clone())
                    .graph(structure)
                    .query(CausalQuery::TemporalEffect(pulse()))
                    .inference(inference)
                    .refute(suite)
                    .bootstrap_replicates(20)
                    .build()
                    .unwrap()
                    .prepare(&ctx)
                    .unwrap();
                if bayesian {
                    assert!(
                        prepared.checked_bayesian_temporal_dag_effect_info().is_some(),
                        "the parent-adjustment Bayesian route must retain its checked temporal plan"
                    );
                }
                let result = prepared.estimate_series(&data, &ctx).unwrap_or_else(|error| {
                    panic!("bayesian={bayesian} accepted={accepted} {suite:?}: {error}")
                });
                let what = format!("bayesian={bayesian} accepted={accepted} {suite:?}");
                assert!(rules(&result).iter().any(|rule| rule == PARENT_ADJUSTMENT_RULE), "{what}");
                let value = if bayesian {
                    let posterior = result.posterior.as_ref().expect("posterior");
                    posterior.summaries.mean[posterior.effect_column().unwrap()]
                } else {
                    result.estimate.ate
                };
                assert!((value - AR_BETA).abs() < 0.08, "{what}: {value}");
            }
        }
    }
}

/// Pooled panel data uses the same identification owner and fallback.
#[test]
fn pooled_panel_pulse_uses_parent_adjustment() {
    let units: Vec<antecedent_data::PanelUnit> = (0..3)
        .map(|unit| antecedent_data::PanelUnit {
            unit_id: unit,
            series: autoregressive_pulse_series(800, AR_PHI, false, 20 + u64::from(unit)),
        })
        .collect();
    let panel = antecedent_data::PanelData::try_new(std::sync::Arc::from(units)).unwrap();
    let result = Study::panel(panel)
        .graph(autoregressive_pulse_dag(true, false))
        .query(CausalQuery::TemporalEffect(pulse()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(20))
        .unwrap();
    assert!(rules(&result).iter().any(|rule| rule == PARENT_ADJUSTMENT_RULE));
    assert!((result.estimate.ate - AR_BETA).abs() < 0.05, "ate={}", result.estimate.ate);
}

/// The same data fitted without the confounder shows the adjustment matters:
/// the `z[t-2]` path biases the lag-one regression.
#[test]
fn parent_adjustment_removes_the_confounding_the_naive_regression_keeps() {
    let data = autoregressive_pulse_series(N, AR_PHI, false, 8);
    let ctx = ExecutionContext::for_tests(8);
    let mut naive = TemporalDag::empty();
    let t1 = naive.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = naive.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    naive.insert_directed(t1, y0).unwrap();
    let biased = study(data.clone(), naive.into(), pulse()).run(&ctx).unwrap();
    let adjusted =
        study(data, autoregressive_pulse_dag(true, false).into(), pulse()).run(&ctx).unwrap();
    assert!((biased.estimate.ate - AR_BETA).abs() > 0.2, "naive ate={}", biased.estimate.ate);
    assert!((adjusted.estimate.ate - AR_BETA).abs() < 0.05, "ate={}", adjusted.estimate.ate);
}

fn fit(data: &TimeSeriesData, identified: &TemporalIdentificationResult) -> f64 {
    let ctx = ExecutionContext::for_tests(1);
    let mut estimator = TemporalLinearAdjustment::new();
    estimator.inner.bootstrap_replicates = 0;
    let prepared = estimator
        .prepare(
            data,
            &identified.result.estimands[0],
            &pulse(),
            &identified.indexer,
            None,
            &ctx.kernel_policy,
        )
        .unwrap();
    estimator
        .fit(
            &prepared,
            &mut EstimationWorkspace::default(),
            &ctx,
            identified.result.required_assumptions.clone(),
        )
        .unwrap()
        .ate
}

/// Where unfolding certifies (no autoregression), the unfolding-derived set
/// `{z[-2]}` and the parent set `{z[-2], w[-2]}` are different adjustments of
/// the same effect; fitted by the one temporal linear estimator on the same
/// data they agree, so a dense-id or window mistake in the parent path would
/// show here.
#[test]
fn parent_and_unfolding_adjustment_agree_where_both_certify() {
    let graph = autoregressive_pulse_dag(false, true);
    let identifier = TemporalBackdoorIdentifier::new().with_parent_adjustment_fallback();
    let unfolded = identifier.identify_temporal(&graph, &pulse()).unwrap();
    let parents = identifier.identify_pulse_by_parent_adjustment(&graph, &pulse()).unwrap();
    let rule_of = |identified: &TemporalIdentificationResult| {
        identified
            .result
            .derivation
            .steps
            .iter()
            .any(|step| step.rule.as_ref() == PARENT_ADJUSTMENT_RULE)
    };
    assert!(!rule_of(&unfolded) && rule_of(&parents));
    assert_ne!(
        unfolded.result.estimands[0].adjustment_set.len(),
        parents.result.estimands[0].adjustment_set.len(),
        "the two derivations must adjust different sets for this check to mean anything"
    );
    let data = autoregressive_pulse_series(N, 0.0, true, 9);
    let (by_unfolding, by_parents) = (fit(&data, &unfolded), fit(&data, &parents));
    assert!((by_unfolding - by_parents).abs() < 0.03, "{by_unfolding} vs {by_parents}");
    assert!((by_parents - AR_BETA).abs() < 0.05, "{by_parents}");
}

#[test]
fn multi_step_sustained_on_the_autoregressive_graph_still_refuses() {
    let data = autoregressive_pulse_series(400, AR_PHI, false, 10);
    let sustained =
        TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), -2, 1.0)
            .with_policy(TemporalPolicy::sustained(-2, -1))
            .with_horizon_steps(1);
    let error = study(data, autoregressive_pulse_dag(true, false).into(), sustained)
        .run(&ExecutionContext::for_tests(1))
        .expect_err("multi-step sustained must not use parent adjustment");
    let message = error.to_string();
    assert!(message.contains("lagged cycle"), "{message}");
}

#[test]
fn treatment_parent_outside_max_history_lag_still_refuses() {
    let data = autoregressive_pulse_series(400, AR_PHI, false, 11);
    let error = study(
        data,
        autoregressive_pulse_dag(true, false).into(),
        pulse().with_max_history_lag(Some(1)),
    )
    .run(&ExecutionContext::for_tests(1))
    .expect_err("a parent two slices back is not observed within a one-slice history");
    assert!(error.to_string().contains("beyond max_history_lag"), "{error}");
}

/// `t@1 → t@0`, `t@1 → y@0`, `z@1 → y@0`, and `z@0 — t@0` unoriented: the
/// treatment's parents are not determined, so no completion is adjusted by
/// its parents and the pulse is not point-identified.
#[test]
fn temporal_cpdag_with_an_unoriented_treatment_edge_still_refuses() {
    let mut g = TemporalCpdag::empty();
    let lagged = |g: &mut TemporalCpdag, variable: u32, lag: u32| {
        g.add_lagged(VariableId::from_raw(variable), Lag::from_raw(lag)).unwrap()
    };
    let t1 = lagged(&mut g, 0, 1);
    let t0 = lagged(&mut g, 0, 0);
    let y0 = lagged(&mut g, 1, 0);
    let z1 = lagged(&mut g, 2, 1);
    let z0 = lagged(&mut g, 2, 0);
    g.insert_directed(t1, t0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_undirected(z0, t0).unwrap();
    let data = autoregressive_pulse_series(400, AR_PHI, false, 12);
    let error = Study::series(data)
        .graph(g)
        .query(CausalQuery::TemporalEffect(pulse()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(1)))
        .expect_err("an unoriented treatment edge must not identify by parent adjustment");
    let message = error.to_string();
    assert!(message.contains("lagged cycle"), "{message}");
}

/// Coverage record whose construction is exactly `key` over `n_min..=n_max`.
fn record_for(
    key: &antecedent_io::calibration::CalibrationKeyWire,
    id: &'static str,
    n_min: u64,
    n_max: u64,
) -> antecedent_io::coverage_records_data::CoverageRecord {
    let leak = |value: &str| -> &'static str { Box::leak(value.to_owned().into_boxed_str()) };
    antecedent_io::coverage_records_data::CoverageRecord {
        id,
        query: leak(&key.query),
        graph_class: leak(&key.graph_class),
        structure: leak(&key.structure),
        modality: leak(&key.modality),
        inference: leak(&key.inference),
        estimator: leak(&key.estimator),
        interval_method: leak(&key.interval_method),
        se_kind: leak(&key.se_kind),
        dependence: leak(&key.dependence),
        posterior: leak(&key.posterior),
        functional: leak(&key.functional),
        identification: leak(&key.identification),
        nominal: key.level,
        n_min,
        n_max,
        replicates_min: 0,
        posterior_draws_min: 0,
        unidentified_mass_max: 0.0,
        observed: 0.95,
        mcse: 0.01,
        replicates: 400,
        boundary: false,
        grid: &[],
        dgp: "crates/antecedent/tests/temporal_parent_adjustment.rs::fixture",
        test: "crates/antecedent/tests/temporal_parent_adjustment.rs::fixture",
        calibration_sha: "0123456789abcdef0123456789abcdef01234567",
    }
}

/// A parent-adjusted pulse and an unfolding-identified pulse report the same
/// estimator and interval, but their calibration keys differ in the
/// identification construction: the parent-adjusted interval binds only to a
/// record measured for parent adjustment, never to an unfolding record.
#[test]
fn parent_adjusted_interval_binds_only_to_parent_adjustment_records() {
    use antecedent_io::calibration::{IDENTIFICATION_NOT_MEASURED, calibration_slot_in};
    let ctx = ExecutionContext::for_tests(31);
    let run = |graph: TemporalDag, data: TimeSeriesData| {
        let study = Study::series(data)
            .graph(graph)
            .query(CausalQuery::TemporalEffect(pulse()))
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let result = study.run(&ctx).unwrap();
        let contract = study.inspect().unwrap();
        result.calibration_bases(&contract).unwrap().remove(0)
    };
    let parent = run(
        autoregressive_pulse_dag(true, false),
        autoregressive_pulse_series(400, AR_PHI, false, 31),
    );
    let unfolding = run(
        autoregressive_pulse_dag(false, false),
        autoregressive_pulse_series(400, 0.0, false, 31),
    );
    assert_eq!(unfolding.key.identification, "point");
    assert_eq!(parent.key.identification, "point+temporal.parent_adjustment");
    let mut same_otherwise = parent.key.clone();
    same_otherwise.identification = unfolding.key.identification.clone();
    assert_eq!(same_otherwise, unfolding.key, "only the identification construction differs");

    let unfolding_record = record_for(&unfolding.key, "cov.unfolding", 100, 1000);
    let parent_record = record_for(&parent.key, "cov.parent", 100, 1000);
    let slot = calibration_slot_in(&parent, &[unfolding_record]);
    assert_eq!(slot.status, "unavailable");
    assert_eq!(slot.reason.as_deref(), Some(IDENTIFICATION_NOT_MEASURED));
    let slot = calibration_slot_in(&parent, &[unfolding_record, parent_record]);
    assert_eq!(slot.status, "calibrated", "{slot:?}");
    assert_eq!(slot.record_id.as_deref(), Some("cov.parent"));
    let slot = calibration_slot_in(&unfolding, &[parent_record, unfolding_record]);
    assert_eq!(slot.record_id.as_deref(), Some("cov.unfolding"));
}

#[test]
fn contract_records_the_parent_adjustment_and_round_trips_it() {
    let data = autoregressive_pulse_series(600, AR_PHI, false, 13);
    let ctx = ExecutionContext::for_tests(13);
    let prepared = study(data.clone(), autoregressive_pulse_dag(true, false).into(), pulse())
        .prepare(&ctx)
        .unwrap();
    let contract = prepared.contract().unwrap();
    let result = prepared.estimate_series(&data, &ctx).unwrap();
    let product = contract.identities.identification_product.expect("identification product");

    // The rule id and its premises are identity: renaming the rule to the
    // unfolding derivation's, or dropping the sufficiency premise, changes
    // the product digest.
    let digest = |identification: &antecedent_identify::IdentificationResult| {
        antecedent_io::identification_product_digest(identification, false).unwrap()
    };
    let own = digest(&result.identification);
    let mut renamed = result.identification.clone();
    for step in &mut renamed.derivation.steps {
        if step.rule.as_ref() == PARENT_ADJUSTMENT_RULE {
            step.rule = "backdoor.criterion".into();
        }
    }
    assert_ne!(own, digest(&renamed));
    let mut without_premise = result.identification.clone();
    without_premise
        .required_assumptions
        .entries
        .retain(|record| !matches!(record.assumption, Assumption::CausalSufficiency));
    assert_ne!(own, digest(&without_premise));
    assert_eq!(*own.as_bytes(), *product.as_bytes(), "the contract binds this product");

    let bytes = prepared.encode_contracted_result(&result, "parent-adjustment", &ctx).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    assert!(
        consumed.acceptance.unresolved.iter().any(|reason| {
            reason.as_ref() == "dependencies.checked_temporal_dag_effect_operation"
        })
    );
    assert!(!consumed.acceptance.accepts_as_verified_program());
    let body = &consumed.body.identification;
    assert!(body.derivation.iter().any(|step| step.rule == PARENT_ADJUSTMENT_RULE));
    let section = consumed.contract.as_ref().expect("contract");
    let wire = section.identification_product.as_ref().expect("product payload");
    assert!(wire.derivation_rules.iter().any(|rule| rule == PARENT_ADJUSTMENT_RULE));
    assert_eq!(wire.estimands[0].adjustment_set.len(), 2);
}
