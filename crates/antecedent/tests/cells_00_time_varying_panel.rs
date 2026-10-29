// SPDX-License-Identifier: MIT OR Apache-2.0
//! Declarations the time-varying joint panel cannot honour are refused, never ignored.
//!
//! The panel reads its rows as independent geographic groups of contiguous periods with
//! continuous nonnegative channel assignments. It has no observation weights, no missing-data
//! policy, no categorical treatment coding and no extrapolation licence, so each such
//! declaration reaches a typed refusal (or a typed unsupported world) instead of changing the
//! population or the estimand silently.
#[path = "../../../examples/rust/joint_panel_fixture.rs"]
mod fixture;
use antecedent::{
    Study,
    analysis::joint_panel::{JointPanelDesignOutcome, JointPanelError, JointPanelRequest},
};
use antecedent_core::{
    CategoryDomainId, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, Float64Column, OwnedColumn,
    OwnedColumnarStorage, PanelData, PanelUnit, SamplingRegularity, TableView, TimeIndex,
    TimeSeriesData, UnknownCategoryPolicy, ValidityBitmap,
};
use antecedent_estimate::joint_panel::{
    PanelInference,
    time_varying::{PanelMeanParameter, TemporalPanelSpec},
};
use std::sync::Arc;

const INTERVAL_NS: u64 = 604_800_000_000_000;
const NAMES: [&str; 5] = ["channel_a", "channel_b", "outcome", "z", "season"];

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(20_261_100)
}

fn temporal_spec() -> TemporalPanelSpec {
    TemporalPanelSpec {
        varying: vec![PanelMeanParameter::Interaction { id: "ab".into() }],
        change_points: vec![8, 12, 16],
        minimum_segment: 4,
        weak_change_gap: 4.0,
        smooth_knots: vec![0, 8, 16, 23],
        smoothing: vec![1.0],
        path_prior_sd: 1.0,
    }
}

fn temporal_request(data: &PanelData) -> JointPanelRequest {
    let mut request = fixture::request(data, false, PanelInference::Frequentist { draws: 0 });
    request.temporal = Some(temporal_spec());
    request
}

fn columns_of(series: &TimeSeriesData) -> Vec<Vec<f64>> {
    (0..5_u32)
        .map(|id| series.float64_cow(VariableId::from_raw(id)).unwrap().into_owned())
        .collect()
}

fn series_from(columns: &[Vec<f64>]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns(
        NAMES.into_iter().zip(columns.iter().map(Vec::as_slice)),
        INTERVAL_NS,
    )
    .unwrap()
}

/// The fixture panel with every unit's columns rewritten by `edit(unit, columns)`.
fn edited(data: &PanelData, edit: impl Fn(usize, &mut Vec<Vec<f64>>)) -> PanelData {
    let mut result = data.clone();
    for (index, unit) in data.units().iter().enumerate() {
        let mut columns = columns_of(&unit.series);
        edit(index, &mut columns);
        result = result.with_unit_series(index, series_from(&columns)).unwrap();
    }
    result
}

fn refusal(result: Result<impl Sized, JointPanelError>) -> JointPanelError {
    match result {
        Ok(_) => panic!("the declaration must be refused, not executed"),
        Err(error) => error,
    }
}

#[test]
fn cells_00_time_varying_panel_refuses_sampling_weights_and_analysis_masks() {
    let data = fixture::source(false);
    let request = temporal_request(&data);
    // The undeclared panel prepares: the refusals below are of a route that works.
    Study::prepare_joint_panel(data.clone(), request.clone(), &ctx()).unwrap();
    let rows = data.units()[0].series.row_count();
    // A weight carried by one unit only declares no inclusion probability for the others.
    let weighted = data
        .with_unit_series(0, data.units()[0].series.with_weights(vec![2.0; rows].into()).unwrap())
        .unwrap();
    let error = refusal(Study::prepare_joint_panel(weighted, request.clone(), &ctx()));
    assert!(
        matches!(&error, JointPanelError::Compile(message) if message.to_lowercase().contains("weight")),
        "{error}"
    );
}

#[test]
fn cells_00_time_varying_panel_declares_no_missing_policy_and_refuses_missing_cells() {
    let data = fixture::source(false);
    let request = temporal_request(&data);
    // A missing outcome cell, a missing assignment cell and a missing confounder cell.
    for column in [2_usize, 0, 3] {
        let missing = edited(&data, |unit, columns| {
            if unit == 2 {
                columns[column][10] = f64::NAN;
            }
        });
        let error = refusal(Study::prepare_joint_panel(missing, request.clone(), &ctx()));
        assert!(
            matches!(&error, JointPanelError::Engine(_) | JointPanelError::Compile(_)),
            "column {column}: {error}"
        );
        assert!(error.to_string().to_lowercase().contains("finite"), "column {column}: {error}");
    }
}

fn categorical_panel(data: &PanelData) -> PanelData {
    let units: Vec<PanelUnit> = data
        .units()
        .iter()
        .map(|unit| {
            let columns = columns_of(&unit.series);
            let n = columns[0].len();
            let mut builder = CausalSchemaBuilder::new();
            let mut built = Vec::new();
            for (index, name) in NAMES.iter().enumerate() {
                let id = VariableId::from_raw(u32::try_from(index).unwrap());
                if index == 0 {
                    builder
                        .add_variable(
                            *name,
                            ValueType::Categorical,
                            SmallRoleSet::from_hint(RoleHint::Context),
                            None,
                            Some(CategoryDomainId::from_raw(1)),
                            MeasurementSpec::default(),
                        )
                        .unwrap();
                    let domain = CategoryDomain::try_new(
                        CategoryDomainId::from_raw(1),
                        ["none", "low", "high"]
                            .into_iter()
                            .map(|label| CategoryLevel { label: Arc::from(label) })
                            .collect::<Vec<_>>(),
                        false,
                        Some(CategoryCode::from_raw(0)),
                        UnknownCategoryPolicy::Fail,
                    )
                    .unwrap();
                    let codes: Vec<CategoryCode> = columns[0]
                        .iter()
                        .map(|&value| CategoryCode::from_raw(if value > 0.0 { 2 } else { 0 }))
                        .collect();
                    built.push(OwnedColumn::Categorical(
                        CategoricalColumn::try_new(
                            id,
                            codes,
                            ValidityBitmap::all_valid(n),
                            Arc::new(domain),
                        )
                        .unwrap(),
                    ));
                } else {
                    builder
                        .add_variable(
                            *name,
                            ValueType::Continuous,
                            SmallRoleSet::from_hint(RoleHint::Context),
                            None,
                            None,
                            MeasurementSpec::default(),
                        )
                        .unwrap();
                    built.push(OwnedColumn::Float64(
                        Float64Column::new(
                            id,
                            Arc::<[f64]>::from(columns[index].as_slice()),
                            ValidityBitmap::all_valid(n),
                        )
                        .unwrap(),
                    ));
                }
            }
            let storage =
                OwnedColumnarStorage::try_new(builder.build().unwrap(), built, None, None).unwrap();
            let series = TimeSeriesData::try_new(
                storage,
                TimeIndex {
                    regularity: SamplingRegularity::Regular { interval_ns: INTERVAL_NS },
                    length: n,
                },
            )
            .unwrap();
            PanelUnit { unit_id: unit.unit_id, series }
        })
        .collect();
    PanelData::try_new(Arc::from(units)).unwrap()
}

#[test]
fn cells_00_time_varying_panel_refuses_a_categorical_treatment_channel() {
    let data = fixture::source(false);
    let request = temporal_request(&data);
    let categorical = categorical_panel(&data);
    let error = refusal(Study::prepare_joint_panel(categorical, request, &ctx()));
    assert!(matches!(&error, JointPanelError::Compile(_)), "{error}");
}

#[test]
fn cells_00_time_varying_panel_refuses_constant_and_collinear_treatment_channels() {
    let data = fixture::source(false);
    let request = temporal_request(&data);
    let constant = edited(&data, |_, columns| columns[0] = vec![1.5; columns[0].len()]);
    let proportional = edited(&data, |_, columns| {
        columns[1] = columns[0].iter().map(|value| 2.0 * value).collect();
    });
    let affine = edited(&data, |_, columns| {
        columns[1] = columns[0].iter().map(|value| 0.5 * value + 3.0).collect();
    });
    for (name, panel, needle) in [
        ("constant", constant, "constant"),
        ("proportional", proportional, "collinear"),
        ("affine", affine, "collinear"),
    ] {
        let error = refusal(Study::prepare_joint_panel(panel, request.clone(), &ctx()));
        assert!(
            matches!(&error, JointPanelError::Engine(antecedent_estimate::joint_panel::PanelError::Invalid(message))
                if message.contains(needle)),
            "{name}: {error}"
        );
    }
}

#[test]
fn cells_00_time_varying_panel_refuses_degenerate_path_declarations() {
    let data = fixture::source(false);
    let mut duplicate = temporal_request(&data);
    duplicate.temporal.as_mut().unwrap().varying = vec![
        PanelMeanParameter::Interaction { id: "ab".into() },
        PanelMeanParameter::Interaction { id: "ab".into() },
    ];
    let mut empty = temporal_request(&data);
    empty.temporal.as_mut().unwrap().varying.clear();
    let mut reference_season = temporal_request(&data);
    reference_season.temporal.as_mut().unwrap().varying =
        vec![PanelMeanParameter::Season { level: 0 }];
    let mut one_window = temporal_request(&data);
    one_window.temporal.as_mut().unwrap().change_points = vec![22];
    for (name, request) in [
        ("duplicate coordinate", duplicate),
        ("no varying coordinate", empty),
        ("reference season level", reference_season),
        ("change window past the data", one_window),
    ] {
        let error = refusal(Study::prepare_joint_panel(data.clone(), request, &ctx()));
        assert!(
            matches!(
                &error,
                JointPanelError::Engine(antecedent_estimate::joint_panel::PanelError::Invalid(_))
            ),
            "{name}: {error}"
        );
    }
}

#[test]
fn cells_00_time_varying_panel_withholds_support_from_a_held_out_program() {
    let data = fixture::source(false);
    let prepared =
        Study::prepare_joint_panel(data.clone(), temporal_request(&data), &ctx()).unwrap();
    let inspection = prepared.inspect();
    // The declared programs follow no observed group trajectory: no design attempt supports a
    // world, and the fitted path is not extrapolated into a causal claim.
    assert!(!inspection.identification.is_identified_claim());
    assert!(!inspection.design_attempts.is_empty());
    for attempt in &inspection.design_attempts {
        assert!(!matches!(attempt.outcome, JointPanelDesignOutcome::Identified), "{}", attempt.id);
        assert_eq!(attempt.worlds.len(), 4);
        assert!(
            attempt.worlds.iter().all(|world| world.matched_groups.is_empty() && !world.identified),
            "{}",
            attempt.id
        );
    }
    let result = prepared.estimate(&ctx()).unwrap();
    let antecedent::analysis::joint_panel::JointPanelFit::Varying(fit) = &result.fit else {
        panic!("time-varying fit")
    };
    assert_eq!(fit.claim, "model_conditional_temporal_mechanism_only");
    assert_eq!(fit.calibration, "unmeasured");
    // Control: the same route on a panel whose programs are the observed whole-program
    // trajectories, with the known randomized assignment, earns the world support.
    let whole = fixture::whole_program_source();
    let mut request = temporal_request(&whole);
    request.target =
        antecedent::analysis::joint_panel::declare_joint_panel_exogenous_context(&request.target);
    request.assignment = Some(fixture::whole_program_randomization());
    let supported = Study::prepare_joint_panel(whole, request, &ctx()).unwrap();
    let attempt = supported
        .inspect()
        .design_attempts
        .iter()
        .find(|attempt| attempt.id == "randomized.multi_arm_itt.whole_program")
        .cloned()
        .unwrap();
    assert!(matches!(attempt.outcome, JointPanelDesignOutcome::Identified));
    assert!(attempt.worlds.iter().all(|world| world.matched_groups.len() == 1 && world.identified));
}
