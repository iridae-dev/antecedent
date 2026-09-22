#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::manual_map,
    clippy::match_wildcard_for_single_variants,
    clippy::doc_markdown,
    clippy::map_unwrap_or
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]
//! Temporal dose × horizon ``ResponseCurve`` on a ``TemporalDag``.
//!
//! Run: `cargo run -p antecedent --example temporal_response_curve`

use std::sync::Arc;

use antecedent::RefuteSuite;
use antecedent::prelude::*;
use antecedent_core::{
    ContinuousDomain, GridSpec, Lag, MeasurementSpec, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, RoleHint, SmallRoleSet, TemporalPolicy,
    TemporalResponseSpec, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{TemporalDag, ensure_lagged};

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), CausalError> {
    let n = 400usize;
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    // A slowly varying `sin(0.04t)` makes pressure[t-1] and pressure[t-2]
    // almost identical (their sample correlation is ~0.999, since a phase
    // shift of 0.04 rad barely moves the sine). The lag-1 and lag-2
    // response cells below each regress `defect` on *only* its own lag,
    // with no adjustment for the other lag (there is no edge between them,
    // so none is owed graphically) — but under near-collinearity that
    // single-lag regression absorbs the other lag's coefficient too via
    // classic omitted-variable bias: beta ≈ 0.9 + 0.1 * corr(p1, p2) ≈ 1.0
    // for lag 1, and symmetrically ≈ 1.0 for lag 2, which is exactly the
    // wrong, dose-independent-of-horizon surface a stale fixture produced
    // here. Using `sin((pi/2) t)` instead makes pressure[t-1] and
    // pressure[t-2] exactly phase-quadrature (their sample correlation is
    // ~1e-5, effectively zero — the pi/2 phase shift makes them a sine and
    // a cosine of the same argument), and moreover pressure[t] is exactly 0
    // on every other step, so in every row exactly one of the two lags is
    // nonzero: the single-lag regressions cleanly recover 0.9 and 0.1
    // without picking up the other lag's coefficient.
    let omega = std::f64::consts::FRAC_PI_2;
    for t in 0..n {
        pressure[t] = ((t as f64) * omega).sin();
        if t > 0 {
            defect[t] = 0.9 * pressure[t - 1];
        }
        if t > 1 {
            defect[t] += 0.1 * pressure[t - 2];
        }
    }

    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "pressure",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "defect",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build()?;
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(pressure),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(defect),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None)?;
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex {
            regularity: SamplingRegularity::Regular { interval_ns: 3_600_000_000_000 },
            length: n,
        },
    )?;

    let mut g = TemporalDag::empty();
    let p1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1))?;
    let p2 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(2))?;
    let d0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS)?;
    g.insert_directed(p1, d0)?;
    g.insert_directed(p2, d0)?;

    let temporal = TemporalResponseSpec::new(vec![1, 2], TemporalPolicy::pulse(-1), None).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(vec![0.0, 0.5, 1.0])),
        ),
    })
    .with_temporal(temporal);

    let result = Study::series(series)
        .graph(g)
        .query(CausalQuery::Response(query))
        // The band comes from joint circular-block replicates of the whole surface;
        // zero replicates publish the point surface with no band.
        .bootstrap_replicates(100)
        .refute(RefuteSuite::None)
        .build()?
        .run(&ExecutionContext::for_tests(42))?;

    let response = result.response.as_ref().expect("response payload");
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected surface");
    };
    let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &response.uncertainty else {
        panic!("expected pointwise bands");
    };
    println!("dose × horizon surface (mean, lower, upper):");
    for i in 0..mean.len() {
        println!("  cell {i}: mean={:.4}  [{:.4}, {:.4}]", mean[i], lower[i], upper[i]);
    }
    // defect[t] = 0.9 pressure[t-1] + 0.1 pressure[t-2]: at doses 0, 0.5, 1 the two horizons
    // read 0.9 * dose and 0.1 * dose, whatever order the surface lists its cells in.
    let mut sorted = mean.to_vec();
    sorted.sort_by(f64::total_cmp);
    let truth = [0.0, 0.0, 0.05, 0.1, 0.45, 0.9];
    assert_eq!(sorted.len(), truth.len());
    for (got, want) in sorted.iter().zip(truth) {
        assert!((got - want).abs() < 0.05, "surface {sorted:?} vs {truth:?}");
    }
    for i in 0..mean.len() {
        assert!(lower[i] <= mean[i] && mean[i] <= upper[i], "band must contain its own mean");
    }
    Ok(())
}

fn main() -> Result<(), CausalError> {
    run()
}
