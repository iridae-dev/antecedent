//! The independent joint panel against the inputs it cannot read: observation weights, missing
//! cells and analysis masks, a categorical channel, degenerate channels, and a world whose
//! program no observed group received.
//!
//! The public request declares channels as nonnegative finite numeric doses, an outcome, a
//! confounder and an integer season, and it has no weight, missing-data or level declaration.
//! Every input that would need one is therefore refused with a typed error at preparation
//! instead of being dropped, imputed or coerced. The degenerate cases follow their known
//! answers: a constant or exactly collinear channel is refused by name, and a constant outcome is
//! reproduced exactly by every world. A world whose program no group received keeps its fitted
//! numbers but is withheld from the identified claim.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::Study;
use antecedent::analysis::joint_panel::{
    JointPanelColumns, JointPanelFunctional, JointPanelRequest, JointPanelWorld, joint_panel_target,
};
use antecedent_core::{
    CategoryDomainId, CausalSchemaBuilder, MeasurementSpec, SmallRoleSet, ValueType,
};
use antecedent_core::{
    ExecutionContext, InferenceSelection, ScientificWorldId, ValueExpr, VariableId,
};
use antecedent_data::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, Float64Column, OwnedColumn,
    OwnedColumnarStorage, PanelData, PanelUnit, SamplingRegularity, TimeIndex, TimeSeriesData,
    UnknownCategoryPolicy, ValidityBitmap,
};
use antecedent_estimate::joint_panel::{
    PanelChannelHistory, PanelChannelProgram, PanelFitOptions, PanelInference,
    PanelInteractionSpec, PanelOptimizerOptions, PanelProgram, PanelScenario,
};

#[path = "../../../examples/rust/joint_panel_fixture.rs"]
mod fixture;

#[derive(Clone)]
struct Cols {
    a: Vec<f64>,
    b: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    season: Vec<f64>,
}

fn base() -> Cols {
    Cols {
        a: vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0],
        b: vec![0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 2.0],
        y: vec![f64::NAN, f64::NAN, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0],
        z: vec![f64::NAN, f64::NAN, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        season: vec![0.0; 8],
    }
}

fn series(c: &Cols) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns(
        [
            ("a", c.a.as_slice()),
            ("b", c.b.as_slice()),
            ("y", c.y.as_slice()),
            ("z", c.z.as_slice()),
            ("season", c.season.as_slice()),
        ],
        100,
    )
    .unwrap()
}

fn panel(units: Vec<TimeSeriesData>) -> PanelData {
    let units = units
        .into_iter()
        .enumerate()
        .map(|(i, series)| PanelUnit { unit_id: i as u32, series })
        .collect::<Vec<_>>();
    PanelData::try_new(Arc::from(units)).unwrap()
}

fn request(source: &PanelData, dose: f64) -> JointPanelRequest {
    let columns = JointPanelColumns {
        channels: vec![VariableId::from_raw(0), VariableId::from_raw(1)],
        outcome: VariableId::from_raw(2),
        confounder: VariableId::from_raw(3),
        season: VariableId::from_raw(4),
    };
    let scenarios = (0..4)
        .map(|id| PanelScenario {
            group: id.to_string(),
            history: vec![
                PanelChannelHistory { id: "0".into(), values: [0.0; 2] },
                PanelChannelHistory { id: "1".into(), values: [0.0; 2] },
            ],
            confounder: vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            season: vec![0; 6],
        })
        .collect::<Vec<_>>();
    let worlds = [(0.0, 0.0), (dose, 0.0), (0.0, dose), (dose, dose)]
        .into_iter()
        .enumerate()
        .map(|(index, (a, b))| JointPanelWorld {
            world: ScientificWorldId::from_raw(u32::try_from(index).unwrap()),
            reference: index == 0,
            program: PanelProgram {
                id: format!("p{index}"),
                channels: vec![
                    PanelChannelProgram { id: "0".into(), values: vec![a; 6] },
                    PanelChannelProgram { id: "1".into(), values: vec![b; 6] },
                ],
                scenarios: scenarios.clone(),
            },
        })
        .collect::<Vec<_>>();
    let target = joint_panel_target(
        source.schema(),
        &columns,
        2,
        &worlds,
        JointPanelFunctional::Interaction {
            joint: ScientificWorldId::from_raw(3),
            a_at_reference: ScientificWorldId::from_raw(1),
            b_at_reference: ScientificWorldId::from_raw(2),
            reference: ScientificWorldId::from_raw(0),
        },
        InferenceSelection::Frequentist,
    )
    .unwrap();
    JointPanelRequest {
        columns,
        interactions: vec![
            PanelInteractionSpec::from_saturated_features(
                "ab",
                &ValueExpr::var(VariableId::from_raw(0))
                    .mul(ValueExpr::var(VariableId::from_raw(1))),
            )
            .unwrap(),
        ],
        seasons: 1,
        lag_order: 2,
        options: PanelFitOptions {
            optimizer: PanelOptimizerOptions { max_iterations: 60, gradient_tolerance: 1e-4 },
            inference: PanelInference::Frequentist { draws: 0 },
            initial: None,
        },
        family: antecedent_estimate::joint_panel::PanelFamily::default(),
        worlds,
        target,
        assignment: None,
        temporal: None,
    }
}

fn four(columns: &Cols) -> Vec<TimeSeriesData> {
    (0..4).map(|_| series(columns)).collect()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

fn prepare_error(units: Vec<TimeSeriesData>) -> String {
    let source = panel(units);
    let request = request(&source, 1.0);
    match Study::prepare_joint_panel(source, request, &ctx()) {
        Ok(_) => panic!("the panel was prepared although an input it cannot read was declared"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn the_panel_fits_the_clean_fixture_that_every_refusal_below_departs_from() {
    let source = panel(four(&base()));
    let request = request(&source, 1.0);
    let prepared = Study::prepare_joint_panel(source, request, &ctx()).unwrap();
    let result = prepared.estimate(&ctx()).unwrap();
    assert!(result.point.iter().all(|response| response.means.iter().all(|m| m.is_finite())));
}

#[test]
fn observation_weights_and_analysis_masks_are_refused_not_dropped() {
    let mut weighted = four(&base());
    weighted[0] = weighted[0].with_weights(Arc::from(vec![1.0; 8])).unwrap();
    let text = prepare_error(weighted);
    assert!(text.contains("weighted or masked panel laws require a matching engine"), "{text}");
    let mut masked = four(&base());
    let mask = ValidityBitmap::all_valid(8);
    masked[2] = masked[2].with_analysis_mask(mask).unwrap();
    let text = prepare_error(masked);
    assert!(text.contains("weighted or masked panel laws require a matching engine"), "{text}");
}

#[test]
fn a_missing_cell_in_any_channel_outcome_or_confounder_is_refused_not_imputed() {
    for column in ["channel", "outcome", "confounder"] {
        let mut changed = base();
        match column {
            "channel" => changed.a[4] = f64::NAN,
            "outcome" => changed.y[4] = f64::NAN,
            _ => changed.z[4] = f64::NAN,
        }
        let mut units = four(&base());
        units[1] = series(&changed);
        let text = prepare_error(units);
        assert!(
            text.contains("group IDs, finite values, cadence or vector dimensions"),
            "{column}: {text}"
        );
    }
}

fn categorical_channel_series() -> TimeSeriesData {
    let plain = base();
    let id = CategoryDomainId::from_raw(3);
    let mut builder = CausalSchemaBuilder::new();
    for (name, categorical) in
        [("a", true), ("b", false), ("y", false), ("z", false), ("season", false)]
    {
        builder
            .add_variable(
                name,
                if categorical { ValueType::Categorical } else { ValueType::Continuous },
                SmallRoleSet::empty(),
                None,
                categorical.then_some(id),
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let levels: Arc<[CategoryLevel]> = ["low", "mid", "high", "peak"]
        .into_iter()
        .map(|label| CategoryLevel { label: Arc::from(label) })
        .collect();
    let domain =
        CategoryDomain::try_new(id, levels, false, None, UnknownCategoryPolicy::Fail).unwrap();
    let codes: Vec<CategoryCode> =
        plain.a.iter().map(|&value| CategoryCode::from_raw(value as u32)).collect();
    let mut columns = vec![OwnedColumn::Categorical(
        CategoricalColumn::try_new(
            VariableId::from_raw(0),
            Arc::<[CategoryCode]>::from(codes),
            ValidityBitmap::all_valid(8),
            Arc::new(domain),
        )
        .unwrap(),
    )];
    for (index, values) in [&plain.b, &plain.y, &plain.z, &plain.season].into_iter().enumerate() {
        columns.push(OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(index as u32 + 1),
                Arc::<[f64]>::from(values.as_slice()),
                ValidityBitmap::all_valid(8),
            )
            .unwrap(),
        ));
    }
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 100 }, length: 8 },
    )
    .unwrap()
}

#[test]
fn a_categorical_channel_is_refused_not_read_as_a_magnitude() {
    let units: Vec<_> = (0..4).map(|_| categorical_channel_series()).collect();
    let source = panel(units);
    let request = request(&source, 1.0);
    let error = Study::prepare_joint_panel(source, request, &ctx())
        .err()
        .expect("a dictionary-coded channel was read as a dose");
    let text = error.to_string();
    assert!(text.contains("joint panel compilation refused"), "{text}");
    assert!(text.contains("variable V0 is not of type float64"), "{text}");
}

#[test]
fn a_constant_or_collinear_channel_is_refused_and_a_constant_outcome_is_reproduced() {
    let mut constant = base();
    constant.b = vec![1.0; 8];
    let text = prepare_error(four(&constant));
    assert!(text.contains("channel 1 is constant across every group and period"), "{text}");
    assert!(text.contains("not identified"), "{text}");
    let mut collinear = base();
    collinear.b = collinear.a.clone();
    let text = prepare_error(four(&collinear));
    assert!(text.contains("channel 1 is exactly collinear with the earlier channels"), "{text}");
    // A constant outcome has every world's mean equal to that constant, and the fitted
    // interaction contrast is zero.
    let mut flat = base();
    flat.y = vec![f64::NAN, f64::NAN, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    let source = panel(four(&flat));
    let request = request(&source, 1.0);
    let result =
        Study::prepare_joint_panel(source, request, &ctx()).unwrap().estimate(&ctx()).unwrap();
    assert_eq!(result.point.len(), 4);
    for response in &result.point {
        for mean in &response.means {
            assert!((mean - 2.0).abs() < 1e-3, "{mean}");
        }
    }
}

#[test]
fn a_world_no_group_received_is_withheld_from_the_identified_claim() {
    use antecedent::analysis::joint_panel::declare_joint_panel_exogenous_context;
    let build = |held_out: bool| {
        let data = fixture::whole_program_source();
        let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
        if held_out {
            // Worlds 1 and 3 move the first channel to a dose no group was ever given.
            for world in [1, 3] {
                for value in &mut request.worlds[world].program.channels[0].values {
                    *value = 7.5;
                }
            }
            request.target = joint_panel_target(
                data.schema(),
                &request.columns,
                2,
                &request.worlds,
                JointPanelFunctional::Interaction {
                    joint: ScientificWorldId::from_raw(3),
                    a_at_reference: ScientificWorldId::from_raw(1),
                    b_at_reference: ScientificWorldId::from_raw(2),
                    reference: ScientificWorldId::from_raw(0),
                },
                InferenceSelection::Frequentist,
            )
            .unwrap();
        }
        request.target = declare_joint_panel_exogenous_context(&request.target);
        request.assignment = Some(fixture::whole_program_randomization());
        Study::prepare_joint_panel(data, request, &ctx()).unwrap()
    };
    let supported = build(false);
    assert!(supported.inspect().identification.is_identified_claim());
    let held_out = build(true);
    let inspection = held_out.inspect();
    assert!(!inspection.identification.is_identified_claim());
    let attempt = inspection
        .design_attempts
        .iter()
        .find(|attempt| attempt.id == "randomized.multi_arm_itt.whole_program")
        .unwrap();
    assert!(matches!(
        attempt.outcome,
        antecedent::analysis::joint_panel::JointPanelDesignOutcome::Rejected
    ));
    assert!(attempt.detail.contains("did not receive its assigned complete program"));
    for world in &attempt.worlds {
        assert!(!world.identified, "{:?}", world.world);
        let held = world.world == ScientificWorldId::from_raw(1)
            || world.world == ScientificWorldId::from_raw(3);
        assert_eq!(world.matched_groups.is_empty(), held, "{:?}", world.world);
    }
    assert_eq!(inspection.fits_performed, 0);
}
