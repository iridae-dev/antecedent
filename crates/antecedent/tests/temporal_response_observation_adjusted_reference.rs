//! The observation-adjusted temporal response against an exact reference, through the facade.
//!
//! `oracle_data/temporal_response_observation_adjusted_reference.py` (frozen in the JSON beside it)
//! builds, from one 100-row series with a lag-1 pulse and horizons 1 and 2, the quantities the
//! composition owns and compares them with the product:
//!
//! * the lag-aligned observation table (conditioning columns read one row back, the outcome and
//!   indicator at the outcome time) with contiguous time-block folds, the logistic selection fit,
//!   the least-squares outcome regression on the conditioning set plus the downstream design
//!   columns x and z at lag 2, the pseudo-values `m + R (y - m) / p` (or `R y / p`), the weight
//!   diagnostic, and for a right-censored series the Cox inverse-censoring pseudo-values;
//! * the per-horizon g-computation of the pseudo-outcome series (the arithmetic of
//!   `temporal_response_gcomp_reference`), dose-major, horizon-minor;
//! * the outer observation bootstrap: the circular-block anchors are drawn by the library, regenerated
//!   here from the disclosed block length and compared with the frozen anchors, and the reference
//!   refits the observation nuisance and both horizons on exactly those anchors in exact arithmetic;
//!   the published pointwise band must equal `surface +/- z * dispersion * kernel * sd`, with the
//!   dispersion and per-cell kernel factors read from the product's own diagnostics (they belong to
//!   the response family's block-bootstrap records and are not recomputed here).
//!
//! The declared options are exercised on one series: selected correction (IPW against AIPW), the
//! cross-fitting folds (5 against 3), the declared probability floor (0.3, which clamps), the
//! bootstrap replicate count, and the refusals of a conditioning set that omits the treatment or the
//! adjustment variable and of a delayed entry.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Lag, ObservationAssumption,
    ObservationSpec, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, StreamDomain, TemporalEffectQuery, TemporalNodeKey,
    TemporalPolicy, TemporalResponseSpec, VariableId, derive_stream,
};
use antecedent_data::TimeSeriesData;
use antecedent_estimate::{
    ObservationEstimatorOptions, ObservationMechanismEstimator, SelectedOutcomeCorrection,
    circular_block_positions_into, temporal_curve_outcome_regressors,
};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::TemporalBackdoorIdentifier;
use serde_json::Value as Json;

const X: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;
/// The bootstrap stream base of the observation-adjusted curve surface.
const STREAM: u64 = 0x0B5E_0000;
/// Two-sided 95% normal quantile (`normal_ppf(0.975)` to binary64).
const Z_95: f64 = 1.959_963_984_540_054;
const SEED: u64 = 3;

fn reference() -> Json {
    serde_json::from_str(include_str!(
        "oracle_data/temporal_response_observation_adjusted_reference.json"
    ))
    .unwrap()
}

fn streams() -> Json {
    serde_json::from_str(include_str!(
        "oracle_data/temporal_response_observation_adjusted_streams.json"
    ))
    .unwrap()
}

fn floats(v: &Json) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn series_of(doc: &Json, kind: &str) -> TimeSeriesData {
    let s = &doc["series"][kind];
    let columns: Vec<(&str, Vec<f64>)> = if kind == "selected" {
        ["x", "y", "z", "r"].iter().map(|k| (*k, floats(&s[*k]))).collect()
    } else {
        ["x", "y", "z", "c", "e"].iter().map(|k| (*k, floats(&s[*k]))).collect()
    };
    let borrowed: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (*name, values.as_slice())).collect();
    TimeSeriesData::from_f64_columns(borrowed, 1).unwrap()
}

/// z_s -> x_s, z_{s-1} -> x_s, x_{s-1} -> y_s, x_{s-2} -> y_s, z_{s-1} -> y_s, z_{s-2} -> y_s.
fn graph() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let (x, y, z) = (VariableId::from_raw(X), VariableId::from_raw(Y), VariableId::from_raw(Z));
    let x1 = ensure_lagged(&mut g, x, Lag::from_raw(1)).unwrap();
    let x2 = ensure_lagged(&mut g, x, Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut g, y, Lag::CONTEMPORANEOUS).unwrap();
    let z1 = ensure_lagged(&mut g, z, Lag::from_raw(1)).unwrap();
    let z2 = ensure_lagged(&mut g, z, Lag::from_raw(2)).unwrap();
    for (from, to) in [(x1, y0), (x2, y0), (z1, x1), (z2, x1), (z1, y0), (z2, y0)] {
        g.insert_directed(from, to).unwrap();
    }
    g
}

fn identify(
    g: &TemporalDag,
    horizon: u32,
) -> (antecedent_expr::IdentifiedEstimand, antecedent_data::TemporalIndexer) {
    let query = TemporalEffectQuery::pulse(VariableId::from_raw(X), VariableId::from_raw(Y), 1.0)
        .with_horizon_steps(horizon)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_max_history_lag(Some(6));
    let id = TemporalBackdoorIdentifier::new().identify_temporal(g, &query).unwrap();
    (id.result.estimands.first().cloned().expect("identified estimand"), id.indexer)
}

struct Case<'a> {
    doc: &'a Json,
    entry: &'a Json,
    options: ObservationEstimatorOptions,
    query: ResponseQuery,
    series: TimeSeriesData,
}

fn query_for(doc: &Json, kind: &str, conditioning: &[u32]) -> ResponseQuery {
    let grid = floats(&doc["doses"]);
    let ids: Arc<[VariableId]> = conditioning.iter().map(|i| VariableId::from_raw(*i)).collect();
    let (spec, assumption) = if kind == "selected" {
        (
            ObservationSpec::Selected {
                latent: VariableId::from_raw(Y),
                observed: VariableId::from_raw(Y),
                indicator: VariableId::from_raw(3),
            },
            ObservationAssumption::OutcomeIndependentGiven(ids),
        )
    } else {
        (
            ObservationSpec::RightCensored {
                latent: VariableId::from_raw(Y),
                observed: VariableId::from_raw(Y),
                censoring: VariableId::from_raw(3),
                event: VariableId::from_raw(4),
            },
            ObservationAssumption::IndependentGiven(ids),
        )
    };
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(Y),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(X),
            GridSpec::Values(Arc::from(grid)),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1, 2], TemporalPolicy::pulse(-1), None).unwrap())
    .with_observation(spec, [assumption])
}

fn options_for(entry: &Json) -> ObservationEstimatorOptions {
    let mut options = ObservationEstimatorOptions::default();
    match entry["mode"].as_str().unwrap() {
        "ipw" => options.selected_correction = SelectedOutcomeCorrection::Ipw,
        _ => options.selected_correction = SelectedOutcomeCorrection::Aipw,
    }
    if let Some(folds) = entry["folds"].as_u64().filter(|f| *f > 0) {
        options.crossfit_folds = usize::try_from(folds).unwrap();
    }
    options.observation_probability_floor = entry["floor"].as_f64();
    options
}

fn study(case: &Case<'_>, replicates: u32) -> Study {
    Study::series(case.series.clone())
        .graph(graph())
        .query(CausalQuery::Response(case.query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .observation_options(case.options)
        .build()
        .unwrap()
}

fn cases(doc: &Json) -> Vec<Case<'_>> {
    doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            let kind = entry["series"].as_str().unwrap();
            Case {
                doc,
                entry,
                options: options_for(entry),
                query: query_for(doc, kind, &[X, Z]),
                series: series_of(doc, kind),
            }
        })
        .collect()
}

/// The downstream design columns and causal adjustment variables the facade hands the correction,
/// derived from the identified horizons exactly as the execution path does.
fn regressors_and_adjustment(query: &ResponseQuery) -> (Vec<TemporalNodeKey>, Vec<VariableId>) {
    let g = graph();
    let ids: Vec<_> = [1, 2].iter().map(|&h| identify(&g, h)).collect();
    let pairs: Vec<_> = ids.iter().map(|(e, i)| (e, i)).collect();
    let regressors = temporal_curve_outcome_regressors(query, &pairs).unwrap();
    let mut adjustment = Vec::new();
    for (estimand, indexer) in &ids {
        for dense in estimand.adjustment_set.iter() {
            let variable = indexer.key_of(dense.raw()).unwrap().variable;
            if !adjustment.contains(&variable) {
                adjustment.push(variable);
            }
        }
    }
    (regressors, adjustment)
}

fn surface(result: &antecedent::result::StudyResult) -> Vec<f64> {
    let response = result.response().expect("response");
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected a point-identified surface");
    };
    mean.to_vec()
}

fn diagnostic(result: &antecedent::result::StudyResult, id: &str) -> Vec<f64> {
    result
        .response()
        .unwrap()
        .support
        .diagnostics
        .iter()
        .find(|d| d.id.as_ref() == id)
        .unwrap_or_else(|| panic!("diagnostic {id}"))
        .values
        .to_vec()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * (1.0 + want.abs()), "{what}: {got} vs {want}");
}

#[test]
fn the_hand_derived_design_columns_are_what_the_identified_horizons_give() {
    let doc = reference();
    let query = query_for(&doc, "selected", &[X, Z]);
    let (regressors, adjustment) = regressors_and_adjustment(&query);
    let key = |v: u32, offset: i32| TemporalNodeKey { variable: VariableId::from_raw(v), offset };
    // Horizon 1 reads x at -1 and z at -1, -2; horizon 2 (outcome one step later) reads x and z at -2.
    assert_eq!(regressors, vec![key(X, -2), key(X, -1), key(Z, -2), key(Z, -1)]);
    assert_eq!(adjustment, vec![VariableId::from_raw(Z)]);
}

#[test]
fn pseudo_outcomes_weights_and_the_composed_surface_match_the_exact_reference() {
    let doc = reference();
    let query = query_for(&doc, "selected", &[X, Z]);
    let (regressors, adjustment) = regressors_and_adjustment(&query);
    let mut surfaces: Vec<(String, Vec<f64>)> = Vec::new();
    for case in cases(&doc) {
        let name = case.entry["name"].as_str().unwrap();
        let (regressors, adjustment) = if case.entry["series"] == "selected" {
            (regressors.clone(), adjustment.clone())
        } else {
            regressors_and_adjustment(&case.query)
        };
        let estimator = ObservationMechanismEstimator::new(case.options);

        // Row by row, the pseudo-outcome series and its weights.
        let (_, adjusted) = estimator
            .adjust_temporal_series(&case.series, &case.query, &adjustment, &regressors)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = floats(&case.entry["pseudo"]);
        let want_weights = floats(&case.entry["weights"]);
        assert_eq!(adjusted.values.len(), want.len(), "{name}");
        assert_eq!(adjusted.values[0], want[0], "{name}: the first row keeps its recorded outcome");
        for (i, (got, w)) in adjusted.values.iter().zip(&want).enumerate() {
            close(*got, *w, 2e-6, &format!("{name} pseudo {i}"));
        }
        for (i, (got, w)) in adjusted.weights.iter().zip(&want_weights).enumerate() {
            close(*got, *w, 2e-6, &format!("{name} weight {i}"));
        }

        // Through the facade: provenance, method, weight diagnostic and the composed levels.
        let result = study(&case, 0).run(&ExecutionContext::for_tests(SEED)).unwrap();
        let response = result.response().unwrap();
        assert_eq!(
            response.provenance_id.as_ref(),
            "estimate.temporal_response.observation_adjusted",
            "{name}"
        );
        assert!(matches!(response.uncertainty, ResponseUncertainty::None), "{name}");
        let method = match case.entry["mode"].as_str().unwrap() {
            "ipw" => "observation.selected.logistic_ipw.v1",
            "aipw" => "observation.selected.crossfit_logistic_aipw.v1",
            _ => "observation.right_censored.cox_ipcw.v1",
        };
        assert!(
            response
                .support
                .warnings
                .iter()
                .any(|w| w.code.as_ref() == "response.observation_adjustment_method"
                    && w.message.as_ref() == method),
            "{name}: method {method}"
        );
        let got_diag = diagnostic(&result, "response.observation_adjustment_weights");
        for (g, w) in got_diag.iter().zip(floats(&case.entry["weight_diagnostic"])) {
            close(*g, w, 2e-6, &format!("{name} weight diagnostic"));
        }
        let got = surface(&result);
        let want_surface = floats(&case.entry["surface"]);
        assert_eq!(got.len(), want_surface.len(), "{name}");
        for (cell, (g, w)) in got.iter().zip(&want_surface).enumerate() {
            close(*g, *w, 2e-5, &format!("{name} cell {cell}"));
        }
        surfaces.push((name.to_owned(), got));
        let _ = case.doc;
    }
    // The declared options are read: on one series each changes the surface.
    let by = |n: &str| surfaces.iter().find(|(name, _)| name == n).unwrap().1.clone();
    let apart =
        |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let base = by("aipw_default_five_folds");
    assert!(apart(&base, &by("aipw_three_folds")) > 1e-3, "crossfit_folds is read");
    assert!(apart(&base, &by("ipw_in_sample")) > 1e-3, "selected_correction is read");
    assert!(
        apart(&base, &by("aipw_declared_floor")) > 1e-3,
        "observation_probability_floor is read"
    );
    let floor_case = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "aipw_declared_floor")
        .unwrap();
    assert!(floor_case["clamped_probabilities"].as_u64().unwrap() >= 5, "the floor binds");
}

#[test]
fn the_outer_bootstrap_regenerates_the_library_streams_and_recomputes_the_band_exactly() {
    let doc = reference();
    let frozen = streams();
    let n = usize::try_from(doc["n"].as_u64().unwrap()).unwrap();
    for case in cases(&doc) {
        let name = case.entry["name"].as_str().unwrap();
        let boot = &case.entry["bootstrap"];
        let anchors_frozen = frozen["cases"][name]["anchors"].as_array().unwrap();
        let replicates = u32::try_from(anchors_frozen.len()).unwrap();
        let ctx = ExecutionContext::for_tests(SEED);
        let result = study(&case, replicates).run(&ctx).unwrap();
        let response = result.response().unwrap();

        // The disclosed block rule; the anchors are the library's circular-block draws.
        let block_diag = diagnostic(&result, "response.temporal.block_length");
        let (block, rows, dispersion) =
            (block_diag[0] as usize, block_diag[3] as usize, block_diag[4]);
        let first_anchor = n - rows;
        assert_eq!(first_anchor, usize::try_from(boot["first_anchor"].as_u64().unwrap()).unwrap());
        assert_eq!(block, usize::try_from(boot["block"].as_u64().unwrap()).unwrap(), "{name}");
        let mut positions = Vec::new();
        let regenerated: Vec<Vec<usize>> = (0..replicates)
            .map(|r| {
                let mut rng =
                    ctx.rng.stream_for(StreamDomain::Execute, derive_stream(STREAM, u64::from(r)));
                circular_block_positions_into(rows, block, &mut rng, &mut positions);
                positions.iter().map(|p| first_anchor + p).collect()
            })
            .collect();
        for (r, (got, want)) in regenerated.iter().zip(anchors_frozen).enumerate() {
            let want: Vec<usize> =
                want.as_array().unwrap().iter().map(|a| a.as_u64().unwrap() as usize).collect();
            assert_eq!(got, &want, "{name}: replicate {r} anchors");
        }

        // Each replicate's pseudo-outcomes, refit on exactly those anchors.
        let (regressors, adjustment) = regressors_and_adjustment(&case.query);
        let estimator = ObservationMechanismEstimator::new(case.options);
        for (r, anchors) in regenerated.iter().enumerate() {
            let got = estimator
                .adjust_temporal_anchors(
                    &case.series,
                    &case.query,
                    &adjustment,
                    &regressors,
                    anchors,
                )
                .unwrap_or_else(|e| panic!("{name} replicate {r}: {e}"));
            let want = floats(&boot["replicate_pseudo"][r]);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                close(*g, *w, 2e-6, &format!("{name} replicate {r} pseudo {i}"));
            }
        }

        // The published band: surface +/- z * dispersion * kernel * sd of the raw replicate surfaces.
        let counts = diagnostic(&result, "response.observation_block_bootstrap");
        assert_eq!(counts[0] as u32, replicates, "{name}: requested");
        assert_eq!(counts[1] as u32, replicates, "{name}: every replicate completed");
        let kernel = diagnostic(&result, "response.temporal.kernel_bias_factor");
        let point = surface(&result);
        let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &response.uncertainty else {
            panic!("{name}: the outer bootstrap publishes a pointwise band");
        };
        let sd = floats(&boot["raw_sd"]);
        assert_eq!(lower.len(), point.len());
        assert_eq!(kernel.len(), point.len(), "{name}: one kernel factor per cell");
        for c in 0..point.len() {
            let half = Z_95 * dispersion * kernel[c] * sd[c];
            close(0.5 * (lower[c] + upper[c]), point[c], 1e-12, &format!("{name} centre {c}"));
            let got_half = 0.5 * (upper[c] - lower[c]);
            assert!(
                (got_half - half).abs() <= 1e-4 * half.abs() + 1e-9,
                "{name} cell {c}: half-width {got_half} vs {half}"
            );
        }
    }
}

#[test]
fn incomplete_conditioning_sets_and_delayed_entry_are_refused() {
    let doc = reference();
    for (conditioning, label) in [(vec![X], "adjustment variable"), (vec![Z], "treatment")] {
        let case = Case {
            doc: &doc,
            entry: &doc["cases"][0],
            options: ObservationEstimatorOptions::default(),
            query: query_for(&doc, "selected", &conditioning),
            series: series_of(&doc, "selected"),
        };
        let error = Study::series(case.series.clone())
            .graph(graph())
            .query(CausalQuery::Response(case.query.clone()))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .map_err(|e| e.to_string())
            .and_then(|s| s.run(&ExecutionContext::for_tests(SEED)).map_err(|e| e.to_string()))
            .unwrap_err();
        assert!(
            error.contains("OutcomeIndependentGiven to include the treatment and every causal"),
            "{label}: {error}"
        );
    }
    let case = Case {
        doc: &doc,
        entry: &doc["cases"][0],
        options: ObservationEstimatorOptions::default(),
        query: query_for(&doc, "selected", &[X, Z]),
        series: series_of(&doc, "selected"),
    };
    let error = Study::series(case.series.clone())
        .graph(graph())
        .query(CausalQuery::Response(case.query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .observation_delayed_entry(VariableId::from_raw(3))
        .build()
        .map_err(|e| e.to_string())
        .and_then(|s| s.run(&ExecutionContext::for_tests(SEED)).map_err(|e| e.to_string()))
        .unwrap_err();
    assert!(error.contains("delayed entry"), "{error}");
}

/// `censoring_survival_floor` is read by the censoring branch: a floor above the estimated
/// censoring survival of the early rows is refused rather than ignored.
#[test]
fn the_censoring_survival_floor_is_read_by_the_censored_route() {
    let doc = reference();
    let mut options = ObservationEstimatorOptions::default();
    options.censoring_survival_floor = 0.999;
    let case = Case {
        doc: &doc,
        entry: &doc["cases"][4],
        options,
        query: query_for(&doc, "censored", &[X, Z]),
        series: series_of(&doc, "censored"),
    };
    let error = Study::series(case.series.clone())
        .graph(graph())
        .query(CausalQuery::Response(case.query.clone()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .observation_options(case.options)
        .build()
        .map_err(|e| e.to_string())
        .and_then(|s| s.run(&ExecutionContext::for_tests(SEED)).map_err(|e| e.to_string()))
        .unwrap_err();
    assert!(
        error.contains("censoring survival is below the configured positivity floor"),
        "{error}"
    );
}

/// Prints the library-drawn anchors for the reference script; run with `--ignored --nocapture`
/// when the series or the replicate count changes (the reference JSON then only needs its series
/// and case options, which the script prints before the streams exist) and save the output as the
/// streams JSON.
#[test]
#[ignore = "regenerates a frozen input of the reference script"]
fn print_the_library_streams() {
    let doc = reference();
    let n = usize::try_from(doc["n"].as_u64().unwrap()).unwrap();
    let mut out = serde_json::Map::new();
    for case in cases(&doc) {
        let name = case.entry["name"].as_str().unwrap().to_owned();
        let ctx = ExecutionContext::for_tests(SEED);
        let result = study(&case, 10).run(&ctx).unwrap();
        let block_diag = diagnostic(&result, "response.temporal.block_length");
        let (block, rows) = (block_diag[0] as usize, block_diag[3] as usize);
        let first_anchor = n - rows;
        let mut positions = Vec::new();
        let anchors: Vec<Vec<usize>> = (0..10u64)
            .map(|r| {
                let mut rng = ctx.rng.stream_for(StreamDomain::Execute, derive_stream(STREAM, r));
                circular_block_positions_into(rows, block, &mut rng, &mut positions);
                positions.iter().map(|p| first_anchor + p).collect()
            })
            .collect();
        out.insert(
            name,
            serde_json::json!({"first_anchor": first_anchor, "block": block, "anchors": anchors}),
        );
    }
    println!("{}", serde_json::json!({ "cases": out }));
}
