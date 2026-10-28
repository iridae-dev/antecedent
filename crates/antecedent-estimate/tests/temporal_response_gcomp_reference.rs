//! The temporal response surface and policy-path levels against an exact rational reference.
//!
//! `oracle_data/temporal_response_gcomp_reference.py` (frozen in the JSON beside it) fits, per
//! horizon h, the regression of the outcome at t on an intercept, the treatment at t - h and the
//! confounder at the lags of that horizon's adjustment set, over that horizon's own windows, in
//! exact rational arithmetic, and evaluates the g-computed level `beta' cbar(a)` at each dose and
//! at the additive shift `mean(treatment) + shift`. The product builds the same designs from a
//! temporal graph, per-horizon identification and a `ResponseQuery`. Cases cover an ordinary
//! series, an outcome offset of 1e6 with a treatment offset of 1e3, confounder lags correlated at
//! 0.9995, a nine-row series and 400 rows. The recorded adjustment set of each horizon is checked
//! against the one derived by hand for the graph (the identifier itself is reviewed elsewhere).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, CausalSchemaBuilder, ContinuousDomain, ExecutionContext, GridSpec,
    IdentificationStatus, Intervention, Lag, MeasurementSpec, MechanismOverride,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue, RoleHint,
    SmallRoleSet, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, Value, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::TemporalResponseEstimator;
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::TemporalBackdoorIdentifier;
use serde_json::Value as Json;

const X: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;

fn floats(v: &Json) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn series(case: &Json) -> TimeSeriesData {
    let mut b = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("x", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(hint),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let n = floats(&case["y"]).len();
    let column = |id: u32, key: &str| {
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(id),
                Arc::from(floats(&case[key])),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        )
    };
    let storage = OwnedColumnarStorage::try_new(
        schema,
        vec![column(X, "x"), column(Y, "y"), column(Z, "z")],
        None,
        None,
    )
    .unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
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

fn surface(response: &antecedent_core::CausalResponse) -> Vec<f64> {
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected a point-identified surface");
    };
    mean.to_vec()
}

fn adjustment_lags(response: &antecedent_core::CausalResponse, horizon: u32) -> Vec<i64> {
    let ident = response.horizon_identification.as_ref().unwrap();
    let entry = ident.iter().find(|h| h.horizon == horizon).unwrap();
    let mut lags: Vec<i64> = entry
        .adjustment
        .iter()
        .inspect(|k| assert_eq!(k.variable, VariableId::from_raw(Z)))
        .map(|k| i64::from(horizon) - 1 - i64::from(k.offset))
        .collect();
    lags.sort_unstable();
    lags
}

#[test]
fn the_surface_and_policy_levels_match_the_exact_rational_reference() {
    let reference: Json =
        serde_json::from_str(include_str!("oracle_data/temporal_response_gcomp_reference.json"))
            .unwrap();
    let g = graph();
    for case in reference.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let data = series(case);
        let horizons: Vec<u32> = case["horizons"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h.as_u64().unwrap() as u32)
            .collect();
        let ids: Vec<_> = horizons.iter().map(|&h| identify(&g, h)).collect();
        let pairs: Vec<_> = ids.iter().map(|(e, i)| (e, i)).collect();
        let spec = || {
            TemporalResponseSpec::new(horizons.clone(), TemporalPolicy::pulse(-1), Some(6)).unwrap()
        };
        let est = TemporalResponseEstimator::new();
        let ctx = ExecutionContext::for_tests(3);
        let status = IdentificationStatus::NonparametricallyIdentified;

        // mean curve over the dose grid
        let doses = floats(&case["doses"]);
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(Y),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(X),
                GridSpec::Values(Arc::from(doses.clone())),
            ),
        })
        .with_temporal(spec());
        let response =
            est.estimate(&data, &pairs, &query, status, AssumptionSet::new(), &ctx).unwrap();
        for &h in &horizons {
            let want: Vec<i64> = case["adjustment_lags"][h.to_string()]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| l.as_i64().unwrap())
                .collect();
            // the outcome sits at offset horizon - 1; an adjustment node's lag is the distance to it
            let got = adjustment_lags(&response, h);
            assert_eq!(got, want, "{name} h={h}: adjustment lags");
        }
        let mean = surface(&response);
        let (want, scale) = (floats(&case["surface"]), floats(&case["surface_scale"]));
        assert_eq!(mean.len(), want.len(), "{name}");
        let rel = if name.contains("collinear") { 1e-6 } else { 1e-9 };
        for i in 0..mean.len() {
            let tol = rel * scale[i].max(1.0);
            assert!(
                (mean[i] - want[i]).abs() <= tol,
                "{name} cell {i}: {} vs {}",
                mean[i],
                want[i]
            );
        }

        // a hard level is the mean-curve cell at that dose for every horizon
        let level = doses[doses.len() - 1];
        let level_query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(Y),
            interventions: Arc::from(vec![Intervention::set(
                VariableId::from_raw(X),
                Value::f64(level),
            )]),
        })
        .with_temporal(spec());
        let by_level = surface(
            &est.estimate(&data, &pairs, &level_query, status, AssumptionSet::new(), &ctx).unwrap(),
        );
        let nh = horizons.len();
        for (h, got) in by_level.iter().enumerate() {
            let idx = (doses.len() - 1) * nh + h;
            let tol = rel * scale[idx].max(1.0);
            assert!((got - want[idx]).abs() <= tol, "{name} level h{h}: {got} vs {}", want[idx]);
        }

        // an additive shift is evaluated at each horizon's own treatment mean plus the shift
        let shift = case["shift"].as_f64().unwrap();
        let shift_query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(Y),
            interventions: Arc::from(vec![Intervention::soft(
                VariableId::from_raw(X),
                MechanismOverride::additive_shift(shift),
            )]),
        })
        .with_temporal(spec());
        let shifted = surface(
            &est.estimate(&data, &pairs, &shift_query, status, AssumptionSet::new(), &ctx).unwrap(),
        );
        let want_shift = floats(&case["shifted"]);
        for h in 0..nh {
            let tol = rel * want_shift[h].abs().max(1.0);
            assert!(
                (shifted[h] - want_shift[h]).abs() <= tol,
                "{name} shift h{h}: {} vs {}",
                shifted[h],
                want_shift[h]
            );
        }
    }
}
