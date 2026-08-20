#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::manual_map,
    clippy::match_wildcard_for_single_variants,
    clippy::doc_markdown,
    clippy::map_unwrap_or
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

fn main() -> Result<(), CausalError> {
    let n = 400usize;
    let mut pressure = vec![0.0; n];
    let mut defect = vec![0.0; n];
    for t in 0..n {
        pressure[t] = ((t as f64) * 0.04).sin();
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
        .bootstrap_replicates(0)
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
    Ok(())
}
