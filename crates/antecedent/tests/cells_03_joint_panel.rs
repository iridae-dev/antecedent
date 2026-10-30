//! Declarations the static joint panel cannot honour are refused, never ignored.
//!
//! The panel reads its rows as independent geographic groups of contiguous periods with
//! continuous nonnegative channel assignments. It has no observation weights, no missing-data
//! policy, no categorical treatment coding, and it identifies a channel's effect only when the
//! channel varies and is not a multiple of an earlier one. Each such declaration reaches a typed
//! refusal instead of changing the population or the estimand silently. The undeclared panel is
//! first prepared and fitted, so every refusal below is of a route that works.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#[path = "../../../examples/rust/joint_panel_fixture.rs"]
mod fixture;
use antecedent::analysis::joint_panel::JointPanelFit;
use antecedent::{
    Study,
    analysis::joint_panel::{
        JointPanelError, JointPanelFunctional, JointPanelRequest, PanelMissingPolicy,
        PanelUnitDrop, joint_panel_target,
    },
};
use antecedent_core::{
    CategoryDomainId, CausalSchemaBuilder, ExecutionContext, InferenceSelection, MeasurementSpec,
    RoleHint, ScientificWorldId, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::TableView;
use antecedent_data::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, Float64Column, OwnedColumn,
    OwnedColumnarStorage, PanelData, PanelUnit, SamplingRegularity, TimeIndex, TimeSeriesData,
    UnknownCategoryPolicy, ValidityBitmap,
};
use antecedent_estimate::joint_panel::PanelInference;
use std::sync::Arc;

const INTERVAL_NS: u64 = 604_800_000_000_000;
const NAMES: [&str; 5] = ["channel_a", "channel_b", "outcome", "z", "season"];

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(20_261_100)
}

fn plain_request(data: &PanelData) -> JointPanelRequest {
    fixture::request(data, false, PanelInference::Frequentist { draws: 0 })
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

/// Every unit rebuilt on one explicit schema, with one outcome cell of one unit carrying a
/// cleared validity bit and the finite stored value `stored`.
fn invalid_cell_panel(data: &PanelData, unit: usize, row: usize, stored: f64) -> PanelData {
    let units: Vec<PanelUnit> = data
        .units()
        .iter()
        .enumerate()
        .map(|(position, source)| {
            let columns = columns_of(&source.series);
            let n = columns[0].len();
            let mut builder = CausalSchemaBuilder::new();
            let mut built = Vec::new();
            for (index, name) in NAMES.iter().enumerate() {
                let id = VariableId::from_raw(u32::try_from(index).unwrap());
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
                let mut values = columns[index].clone();
                let validity = if index == 2 && position == unit {
                    values[row] = stored;
                    let mut bytes = vec![0xFF_u8; n.div_ceil(8)];
                    bytes[row / 8] &= !(1 << (row % 8));
                    ValidityBitmap::from_bytes(bytes, n).unwrap()
                } else {
                    ValidityBitmap::all_valid(n)
                };
                built.push(OwnedColumn::Float64(
                    Float64Column::new(id, Arc::<[f64]>::from(values.as_slice()), validity)
                        .unwrap(),
                ));
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
            PanelUnit { unit_id: source.unit_id, series }
        })
        .collect();
    PanelData::try_new(Arc::from(units)).unwrap()
}

fn terminal_means(data: PanelData, request: JointPanelRequest) -> Vec<f64> {
    let prepared = Study::prepare_joint_panel(data, request, &ctx()).unwrap();
    prepared.estimate(&ctx()).unwrap().point.iter().map(|world| world.means[23]).collect()
}

fn weighted(data: &PanelData, weights: &[f64]) -> PanelData {
    let mut result = data.clone();
    for (index, weight) in weights.iter().enumerate() {
        let series = &data.units()[index].series;
        let rows = series.row_count();
        result =
            result.with_unit_series(index, series.with_weights(vec![*weight; rows].into()).unwrap())
                .unwrap();
    }
    result
}

fn masked(data: &PanelData, unit: usize, hidden: &[usize]) -> PanelData {
    let series = &data.units()[unit].series;
    let rows = series.row_count();
    let keep: Vec<bool> = (0..rows).map(|row| !hidden.contains(&row)).collect();
    data.with_unit_series(
        unit,
        series.with_analysis_mask(ValidityBitmap::from_bools(&keep)).unwrap(),
    )
    .unwrap()
}

/// The panel with its units listed by source position, each a fresh unit, and every world's
/// scenarios following; `sources = [0, 1, 1]` is unit 1 twice.
fn replicated(
    data: &PanelData,
    request: &JointPanelRequest,
    sources: &[usize],
) -> (PanelData, JointPanelRequest) {
    let units: Vec<PanelUnit> = sources
        .iter()
        .enumerate()
        .map(|(new, &old)| PanelUnit {
            unit_id: u32::try_from(new).unwrap(),
            series: data.units()[old].series.clone(),
        })
        .collect();
    let copy = PanelData::try_new(Arc::from(units)).unwrap();
    let mut request = request.clone();
    for world in &mut request.worlds {
        let scenarios = world.program.scenarios.clone();
        world.program.scenarios = sources
            .iter()
            .enumerate()
            .map(|(new, &old)| {
                let mut scenario =
                    scenarios.iter().find(|s| s.group == old.to_string()).unwrap().clone();
                scenario.group = new.to_string();
                scenario
            })
            .collect();
    }
    request.target = joint_panel_target(
        copy.schema(),
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
    (copy, request)
}

fn all_means(
    prepared: &antecedent::PreparedStudy<antecedent::analysis::joint_panel::JointPanelState>,
) -> Vec<Vec<f64>> {
    prepared.estimate(&ctx()).unwrap().point.iter().map(|world| world.means.clone()).collect()
}

#[test]
fn cells_03_joint_panel_reads_sampling_weights_and_analysis_masks() {
    let data = fixture::source(false);
    let request = plain_request(&data);
    let plain = Study::prepare_joint_panel(data.clone(), request.clone(), &ctx()).unwrap();
    assert_eq!(plain.data_accounting().units_analysed(), 4);
    assert!(plain.data_accounting().is_complete() && !plain.data_accounting().weighted);

    // Equal weights are the unweighted panel: the engine receives the same data.
    let equal = Study::prepare_joint_panel(weighted(&data, &[2.5; 4]), request.clone(), &ctx()).unwrap();
    assert_eq!(equal.model_data(), plain.model_data());
    assert!(!equal.data_accounting().weighted);
    // An analysis mask that hides nothing is the same panel.
    let all_valid = data
        .with_unit_series(
            0,
            data.units()[0]
                .series
                .with_analysis_mask(ValidityBitmap::all_valid(data.units()[0].series.row_count()))
                .unwrap(),
        )
        .unwrap();
    let hidden_nothing = Study::prepare_joint_panel(all_valid, request.clone(), &ctx()).unwrap();
    assert_eq!(hidden_nothing.model_data(), plain.model_data());

    // Known answer for unequal weights: weights (1, 2, 1, 3) are the panel in which unit 1 is
    // present twice and unit 3 three times. The weighted pseudo-likelihood, its maximizer and
    // the weighted response average are those of the replicated panel.
    let weights = [1.0, 2.0, 1.0, 3.0];
    let weighted_panel =
        Study::prepare_joint_panel(weighted(&data, &weights), request.clone(), &ctx()).unwrap();
    assert!(weighted_panel.data_accounting().weighted);
    let normalized: Vec<f64> = weighted_panel.model_data().groups.iter().map(|g| g.weight).collect();
    let mean_weight = 7.0 / 4.0;
    for (got, want) in normalized.iter().zip(weights) {
        assert!((got - want / mean_weight).abs() < 1e-15, "{normalized:?}");
    }
    let (copy, copy_request) = replicated(&data, &request, &[0, 1, 1, 2, 3, 3, 3]);
    let copies = Study::prepare_joint_panel(copy, copy_request, &ctx()).unwrap();
    let (a, b) = (all_means(&weighted_panel), all_means(&copies));
    let mut worst = 0.0_f64;
    for (wa, wb) in a.iter().zip(&b) {
        for (x, y) in wa.iter().zip(wb) {
            worst = worst.max((x - y).abs() / (1.0 + y.abs()));
        }
    }
    assert!(worst < 2e-4, "weighted vs replicated worst relative gap {worst}");
    // The weights move the answer: the unweighted panel is a different population.
    let unweighted = all_means(&plain);
    assert!(
        unweighted[3].iter().zip(&a[3]).any(|(x, y)| (x - y).abs() > 1e-4),
        "unequal weights must change the weighted population mean"
    );

    // A hidden row removes the periods that read it; the accounting is exact and the fit is
    // that of the same panel with those outcomes declared missing under the period policy.
    let hidden = masked(&data, 1, &[10]);
    let by_mask = Study::prepare_joint_panel(hidden, request.clone(), &ctx()).unwrap();
    let account = &by_mask.data_accounting().units[1];
    assert_eq!(account.masked_rows, vec![10]);
    assert_eq!(account.excluded_periods, vec![8, 9, 10]);
    assert_eq!((account.rows, account.periods, account.analysed_periods), (26, 24, 21));
    assert!(account.missing_cells.is_empty() && account.dropped.is_none());
    assert_eq!(by_mask.data_accounting().periods_analysed(), 4 * 24 - 3);
    let by_cells = Study::prepare_joint_panel_with_policy(
        edited(&data, |unit, columns| {
            if unit == 1 {
                for row in [10, 11, 12] {
                    columns[2][row] = f64::NAN;
                }
            }
        }),
        request.clone(),
        PanelMissingPolicy::DropPeriods,
        &ctx(),
    )
    .unwrap();
    assert_eq!(by_cells.data_accounting().units[1].excluded_periods, vec![8, 9, 10]);
    let (a, b) = (all_means(&by_mask), all_means(&by_cells));
    for (wa, wb) in a.iter().zip(&b) {
        for (x, y) in wa.iter().zip(wb) {
            assert!((x - y).abs() < 1e-9, "{x} vs {y}");
        }
    }
}

#[test]
fn cells_03_joint_panel_refuses_weights_it_has_no_likelihood_for() {
    let data = fixture::source(false);
    let request = plain_request(&data);
    let rows = data.units()[0].series.row_count();
    // A weight that varies over a serially dependent unit has no pseudo-likelihood term.
    let mut varying = vec![2.0; rows];
    varying[5] = 3.0;
    let varying = weighted(&data, &[2.0; 4])
        .with_unit_series(0, data.units()[0].series.with_weights(varying.into()).unwrap())
        .unwrap();
    let error = refusal(Study::prepare_joint_panel(varying, request.clone(), &ctx()));
    assert!(
        matches!(&error, JointPanelError::Compile(message)
            if message.contains("one strictly positive finite value over the unit")),
        "{error}"
    );
    // A unit with no weight among weighted units has no declared inclusion probability.
    let one = data
        .with_unit_series(0, data.units()[0].series.with_weights(vec![2.0; rows].into()).unwrap())
        .unwrap();
    let error = refusal(Study::prepare_joint_panel(one, request.clone(), &ctx()));
    assert!(
        matches!(&error, JointPanelError::Compile(message)
            if message.contains("every unit or by none")),
        "{error}"
    );
    // The weighted objective is a pseudo-likelihood: only its weighted cluster sandwich is a
    // covariance. A parametric bootstrap or a posterior would misstate its precision.
    let unequal = weighted(&data, &[1.0, 2.0, 1.0, 3.0]);
    for inference in [
        PanelInference::FrequentistBootstrap { draws: 2 },
        PanelInference::Bayesian {
            prior: antecedent_estimate::joint_panel::PanelPrior::default(),
            sampler: fixture::posterior_sampling(),
        },
    ] {
        let request = fixture::request(&data, false, inference);
        let error = refusal(Study::prepare_joint_panel(unequal.clone(), request, &ctx()));
        assert!(error.to_string().contains("weighted group-cluster sandwich"), "{error}");
    }
    // A time-varying mechanism is selected on complete, equally weighted series.
    let mut temporal = plain_request(&data);
    temporal.temporal = Some(antecedent_estimate::joint_panel::time_varying::TemporalPanelSpec {
        varying: vec![
            antecedent_estimate::joint_panel::time_varying::PanelMeanParameter::Interaction {
                id: "ab".into(),
            },
        ],
        change_points: vec![8, 12, 16],
        minimum_segment: 4,
        weak_change_gap: 4.0,
        smooth_knots: vec![0, 8, 16, 23],
        smoothing: vec![1.0],
        path_prior_sd: 1.0,
    });
    let error = refusal(Study::prepare_joint_panel(unequal, temporal, &ctx()));
    assert!(error.to_string().contains("complete, equally weighted series"), "{error}");
}

#[test]
fn cells_03_joint_panel_declares_a_missing_policy_and_accounts_for_every_missing_cell() {
    let data = fixture::source(false);
    let request = plain_request(&data);
    // Undeclared, a missing cell in any of the five roles is refused, as before.
    for column in 0..5_usize {
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
    let invalid = invalid_cell_panel(&data, 2, 10, 0.0);
    let error = refusal(Study::prepare_joint_panel(invalid, request.clone(), &ctx()));
    assert!(error.to_string().to_lowercase().contains("finite"), "{error}");

    // Period-level complete case. A missing outcome removes its own period; a missing
    // assignment removes the three periods whose exposure window reads it; a missing
    // confounder removes its own period. The bookkeeping is exact.
    for (column, row, excluded) in [
        (2_usize, 10_usize, vec![8_usize]),
        (0, 10, vec![8, 9, 10]),
        (1, 10, vec![8, 9, 10]),
        (3, 10, vec![8]),
        (4, 10, vec![8]),
    ] {
        let missing = edited(&data, |unit, columns| {
            if unit == 2 {
                columns[column][row] = f64::NAN;
            }
        });
        let prepared = Study::prepare_joint_panel_with_policy(
            missing,
            request.clone(),
            PanelMissingPolicy::DropPeriods,
            &ctx(),
        )
        .unwrap_or_else(|error| panic!("column {column}: {error}"));
        let accounting = prepared.data_accounting();
        assert_eq!(accounting.units_analysed(), 4);
        assert_eq!(accounting.units[2].excluded_periods, excluded, "column {column}");
        assert_eq!(accounting.units[2].missing_cells.len(), 1);
        assert_eq!(accounting.periods_total() - accounting.periods_analysed(), excluded.len());
        assert!(!accounting.is_complete());
        let group = &prepared.model_data().groups[2];
        let hidden: Vec<usize> =
            (0..24).filter(|&period| group.outcome[period].is_nan()).collect();
        assert_eq!(hidden, excluded, "column {column}");
        assert!(group.outcome.iter().filter(|value| value.is_finite()).count() == 24 - excluded.len());
        assert!(prepared.data_accounting().policy == PanelMissingPolicy::DropPeriods);
    }
    // The fit of a panel with a missing outcome is finite and the policy is part of the
    // source identity.
    let missing = edited(&data, |unit, columns| {
        if unit == 2 {
            columns[2][10] = f64::NAN;
        }
    });
    let prepared = Study::prepare_joint_panel_with_policy(
        missing.clone(),
        request.clone(),
        PanelMissingPolicy::DropPeriods,
        &ctx(),
    )
    .unwrap();
    assert!(all_means(&prepared).iter().flatten().all(|mean| mean.is_finite()));
    let dropped = Study::prepare_joint_panel_with_policy(
        missing.clone(),
        {
            let mut request = request.clone();
            for world in &mut request.worlds {
                world.program.scenarios.retain(|scenario| scenario.group != "2");
            }
            request.target = joint_panel_target(
                missing.schema(),
                &request.columns,
                2,
                &request.worlds,
                antecedent::analysis::joint_panel::JointPanelFunctional::Interaction {
                    joint: antecedent_core::ScientificWorldId::from_raw(3),
                    a_at_reference: antecedent_core::ScientificWorldId::from_raw(1),
                    b_at_reference: antecedent_core::ScientificWorldId::from_raw(2),
                    reference: antecedent_core::ScientificWorldId::from_raw(0),
                },
                antecedent_core::InferenceSelection::Frequentist,
            )
            .unwrap();
            request
        },
        PanelMissingPolicy::DropUnits,
        &ctx(),
    )
    .unwrap();
    // Unit-level complete case: the unit with a missing cell leaves, the others are whole.
    let accounting = dropped.data_accounting();
    assert_eq!(accounting.units_total(), 4);
    assert_eq!(accounting.units_analysed(), 3);
    assert_eq!(accounting.units[2].dropped, Some(PanelUnitDrop::MissingCell));
    assert_eq!(dropped.model_data().groups.len(), 3);
    assert!(dropped.model_data().groups.iter().all(|group| group.outcome.iter().all(|v| v.is_finite())));
    assert_ne!(prepared.inspect().source_identity, dropped.inspect().source_identity);
}

#[test]
fn cells_03_joint_panel_drops_a_unit_with_no_analysable_period_and_records_it() {
    let data = fixture::source(false);
    let request = plain_request(&data);
    // Hiding every other row leaves no period whose exposure window is available.
    let hidden: Vec<usize> = (0..26).step_by(2).collect();
    let panel = masked(&data, 3, &hidden);
    let mut request = request;
    for world in &mut request.worlds {
        world.program.scenarios.retain(|scenario| scenario.group != "3");
    }
    request.target = joint_panel_target(
        panel.schema(),
        &request.columns,
        2,
        &request.worlds,
        JointPanelFunctional::Interaction {
            joint: ScientificWorldId::from_raw(3),
            a_at_reference: ScientificWorldId::from_raw(1),
            b_at_reference: ScientificWorldId::from_raw(2),
            reference: ScientificWorldId::from_raw(0),
        },
        antecedent_core::InferenceSelection::Frequentist,
    )
    .unwrap();
    let prepared = Study::prepare_joint_panel(panel, request, &ctx()).unwrap();
    assert_eq!(
        prepared.data_accounting().units[3].dropped,
        Some(PanelUnitDrop::TooFewAnalysedPeriods { analysed: 0 })
    );
    assert_eq!(prepared.model_data().groups.len(), 3);
}

/// Level code of a dose at a row: no dose is `none`, a dose on an even row is `low`, on an odd
/// row `high` (the fixture's doses are 0 or 4, so the split is by row).
fn level_of(value: f64, row: usize) -> u32 {
    if value <= 0.0 { 0 } else if row % 2 == 0 { 1 } else { 2 }
}

fn rebuilt_levels(data: &PanelData, levels: &[&str]) -> PanelData {
    // Channel 0 recoded into the declared levels.
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
                        levels
                            .iter()
                            .map(|label| CategoryLevel { label: Arc::from(*label) })
                            .collect::<Vec<_>>(),
                        false,
                        Some(CategoryCode::from_raw(0)),
                        UnknownCategoryPolicy::Fail,
                    )
                    .unwrap();
                    let codes: Vec<CategoryCode> = columns[0]
                        .iter()
                        .enumerate()
                        .map(|(row, &value)| CategoryCode::from_raw(level_of(value, row)))
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
fn cells_03_joint_panel_dummy_codes_a_categorical_channel_with_a_declared_level_set() {
    use antecedent::analysis::joint_panel::{PanelCategoricalChannel, dummy_code_panel};
    let data = fixture::source(false);
    let levels = ["none", "low", "high"];
    let categorical = rebuilt_levels(&data, &levels);
    let declared = PanelCategoricalChannel {
        variable: VariableId::from_raw(0),
        levels: levels.iter().map(ToString::to_string).collect(),
        reference: "none".into(),
    };
    let coded = dummy_code_panel(&categorical, &fixture::columns(), &[declared.clone()]).unwrap();
    // Two indicator channels for the non-reference levels, then the untouched continuous one.
    assert_eq!(coded.indicators.len(), 2);
    assert_eq!(
        coded.indicators.iter().map(|d| d.level.as_str()).collect::<Vec<_>>(),
        ["low", "high"]
    );
    assert_eq!(coded.columns.channels.len(), 3);
    for (unit, source) in coded.data.units().iter().zip(data.units()) {
        let original = source.series.float64_cow(VariableId::from_raw(0)).unwrap();
        let low = unit.series.float64_cow(coded.columns.channels[0]).unwrap();
        let high = unit.series.float64_cow(coded.columns.channels[1]).unwrap();
        for (row, ((value, low), high)) in original.iter().zip(low.iter()).zip(high.iter()).enumerate() {
            assert_eq!(*low, f64::from(u8::from(level_of(*value, row) == 1)));
            assert_eq!(*high, f64::from(u8::from(level_of(*value, row) == 2)));
        }
        // The continuous channel, outcome, confounder and season carry over unchanged.
        let b = source.series.float64_cow(VariableId::from_raw(1)).unwrap();
        assert_eq!(unit.series.float64_cow(coded.columns.channels[2]).unwrap(), b);
        let bits = |values: std::borrow::Cow<'_, [f64]>| -> Vec<u64> {
            values.iter().map(|value| value.to_bits()).collect()
        };
        assert_eq!(
            bits(unit.series.float64_cow(coded.columns.outcome).unwrap()),
            bits(source.series.float64_cow(VariableId::from_raw(2)).unwrap())
        );
    }
    // A fit on the dummy-coded panel is the fit on the same indicators supplied as numbers,
    // and the high level shifts the response by its own coefficient.
    let request =
        |panel: &PanelData, columns: &antecedent::analysis::joint_panel::JointPanelColumns| {
            let scenarios: Vec<_> = panel
                .units()
                .iter()
                .map(|unit| {
                    let chan = |id: VariableId| unit.series.float64_cow(id).unwrap().into_owned();
                    antecedent_estimate::joint_panel::PanelScenario {
                        group: unit.unit_id.to_string(),
                        history: columns
                            .channels
                            .iter()
                            .map(|&id| antecedent_estimate::joint_panel::PanelChannelHistory {
                                id: id.raw().to_string(),
                                values: [chan(id)[0], chan(id)[1]],
                            })
                            .collect(),
                        confounder: chan(columns.confounder)[2..].to_vec(),
                        season: chan(columns.season)[2..].iter().map(|v| *v as usize).collect(),
                    }
                })
                .collect();
            let programs = [
                ("reference", [0.0, 0.0, 0.0]),
                ("low", [1.0, 0.0, 0.0]),
                ("high", [0.0, 1.0, 0.0]),
            ];
            let worlds: Vec<_> = programs
                .into_iter()
                .enumerate()
                .map(|(index, (id, doses))| antecedent::analysis::joint_panel::JointPanelWorld {
                    world: ScientificWorldId::from_raw(u32::try_from(index).unwrap()),
                    reference: index == 0,
                    program: antecedent_estimate::joint_panel::PanelProgram {
                        id: id.into(),
                        channels: columns
                            .channels
                            .iter()
                            .zip(doses)
                            .map(|(id, dose)| {
                                antecedent_estimate::joint_panel::PanelChannelProgram {
                                    id: id.raw().to_string(),
                                    values: vec![dose; 24],
                                }
                            })
                            .collect(),
                        scenarios: scenarios.clone(),
                    },
                })
                .collect();
            let target = joint_panel_target(
                panel.schema(),
                columns,
                2,
                &worlds,
                JointPanelFunctional::Worlds,
                antecedent_core::InferenceSelection::Frequentist,
            )
            .unwrap();
            JointPanelRequest {
                columns: columns.clone(),
                interactions: vec![],
                seasons: 4,
                lag_order: 2,
                options: antecedent_estimate::joint_panel::PanelFitOptions {
                    optimizer: antecedent_estimate::joint_panel::PanelOptimizerOptions {
                        max_iterations: 3000,
                        gradient_tolerance: 1e-5,
                    },
                    inference: PanelInference::Frequentist { draws: 0 },
                    initial: None,
                },
                family: antecedent_estimate::joint_panel::PanelFamily::default(),
                worlds,
                target,
                assignment: None,
                temporal: None,
            }
        };
    let coded_request = request(&coded.data, &coded.columns);
    let by_levels = Study::prepare_joint_panel(coded.data.clone(), coded_request, &ctx()).unwrap();
    // The same indicators as hand-coded numeric channels.
    let hand_units: Vec<PanelUnit> = data
        .units()
        .iter()
        .map(|unit| {
            let a = unit.series.float64_cow(VariableId::from_raw(0)).unwrap();
            let low: Vec<f64> =
                a.iter().enumerate().map(|(row, v)| f64::from(u8::from(level_of(*v, row) == 1))).collect();
            let high: Vec<f64> =
                a.iter().enumerate().map(|(row, v)| f64::from(u8::from(level_of(*v, row) == 2))).collect();
            let rest: Vec<Vec<f64>> = (1..5_u32)
                .map(|id| unit.series.float64_cow(VariableId::from_raw(id)).unwrap().into_owned())
                .collect();
            let series = TimeSeriesData::from_f64_columns(
                [
                    ("channel_a=low", low.as_slice()),
                    ("channel_a=high", high.as_slice()),
                    ("channel_b", rest[0].as_slice()),
                    ("outcome", rest[1].as_slice()),
                    ("z", rest[2].as_slice()),
                    ("season", rest[3].as_slice()),
                ],
                INTERVAL_NS,
            )
            .unwrap();
            PanelUnit { unit_id: unit.unit_id, series }
        })
        .collect();
    let hand = PanelData::try_new(Arc::from(hand_units)).unwrap();
    let hand_request = request(&hand, &coded.columns);
    let by_hand = Study::prepare_joint_panel(hand, hand_request, &ctx()).unwrap();
    assert_eq!(by_levels.model_data(), by_hand.model_data());
    let (a, b) = (all_means(&by_levels), all_means(&by_hand));
    assert_eq!(a, b);
    assert!(a.iter().flatten().all(|mean| mean.is_finite()));

    // Refusals: every one names its cause and none recodes anything silently.
    let refuse = |panel: &PanelData, categorical: Vec<PanelCategoricalChannel>| {
        dummy_code_panel(panel, &fixture::columns(), &categorical).err().expect("refused").to_string()
    };
    let mut wrong_set = declared.clone();
    wrong_set.levels = vec!["none".into(), "low".into()];
    assert!(refuse(&categorical, vec![wrong_set]).contains("differs from the domain"));
    let mut outside = declared.clone();
    outside.reference = "mid".into();
    assert!(refuse(&categorical, vec![outside]).contains("one of them the reference"));
    assert!(refuse(&data, vec![declared.clone()]).contains("is declared categorical but its column is not"));
    let mut not_a_channel = declared.clone();
    not_a_channel.variable = VariableId::from_raw(2);
    assert!(refuse(&categorical, vec![not_a_channel]).contains("not one of the declared channels"));
    assert!(refuse(&categorical, vec![declared.clone(), declared.clone()]).contains("declared twice"));
    // A declared level no unit assigns has no identified effect; here `low` is never assigned.
    let two_levels = categorical_panel(&data);
    assert!(
        refuse(&two_levels, vec![declared]).contains("is assigned in no unit; its effect is not identified")
    );
    // The categorical column still cannot be read as a dose without the declaration.
    let error = refusal(Study::prepare_joint_panel(categorical, plain_request(&data), &ctx()));
    assert!(error.to_string().contains("is not of type float64"), "{error}");
}

#[test]
fn cells_03_joint_panel_refuses_constant_collinear_and_flat_outcome_designs() {
    let data = fixture::source(false);
    let request = plain_request(&data);
    assert_eq!(terminal_means(data.clone(), request.clone()).len(), 4);
    let constant = edited(&data, |_, columns| columns[0] = vec![1.5; columns[0].len()]);
    let zero = edited(&data, |_, columns| columns[0] = vec![0.0; columns[0].len()]);
    for panel in [constant, zero] {
        let error = refusal(Study::prepare_joint_panel(panel, request.clone(), &ctx()));
        assert!(
            matches!(&error, JointPanelError::Engine(_))
                && error
                    .to_string()
                    .contains("channel 0 is constant across every group and period"),
            "{error}"
        );
    }
    let collinear = edited(&data, |_, columns| {
        columns[1] = columns[0].iter().map(|value| 2.0 * value).collect();
    });
    let error = refusal(Study::prepare_joint_panel(collinear, request.clone(), &ctx()));
    assert!(
        matches!(&error, JointPanelError::Engine(_))
            && error
                .to_string()
                .contains("channel 1 is exactly collinear with the earlier channels"),
        "{error}"
    );
    // A flat outcome has no innovation scale: the fit is refused as numerical, not returned.
    let flat = edited(&data, |_, columns| columns[2] = vec![3.0; columns[2].len()]);
    let prepared = Study::prepare_joint_panel(flat, request, &ctx()).unwrap();
    let error = refusal(prepared.estimate(&ctx()).map(|_| ()));
    assert!(matches!(&error, JointPanelError::Engine(_)), "{error}");
    assert!(error.to_string().contains("nonfinite local likelihood score"), "{error}");
}

const ORACLE: &str =
    include_str!("../../antecedent-estimate/tests/oracle_data/joint_panel_weights_gaps_reference.json");

fn number(value: &serde_json::Value) -> f64 {
    value.as_f64().unwrap_or(f64::NAN)
}

/// One oracle case as a public panel: two pre-period rows of assignment history, then the
/// follow-up rows. A `null` outcome is a missing cell. `hidden` rows are masked, `weighted`
/// attaches each unit's weight.
fn oracle_panel(case: &serde_json::Value, hidden: &[(usize, usize)], weighted: bool) -> PanelData {
    let weights: Vec<f64> =
        case["weights"].as_array().unwrap().iter().map(|w| w.as_f64().unwrap()).collect();
    let units: Vec<PanelUnit> = case["groups"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, group)| {
            let history: Vec<f64> = group["history"].as_array().unwrap().iter().map(number).collect();
            let channel: Vec<f64> = history
                .iter()
                .copied()
                .chain(group["channel"].as_array().unwrap().iter().map(number))
                .collect();
            let nan_pad = |values: &serde_json::Value| -> Vec<f64> {
                [f64::NAN, f64::NAN]
                    .into_iter()
                    .chain(values.as_array().unwrap().iter().map(number))
                    .collect()
            };
            let outcome = nan_pad(&group["outcome"]);
            let confounder = nan_pad(&group["confounder"]);
            let season = vec![0.0; channel.len()];
            let mut series = TimeSeriesData::from_f64_columns(
                [
                    ("channel", channel.as_slice()),
                    ("outcome", outcome.as_slice()),
                    ("z", confounder.as_slice()),
                    ("season", season.as_slice()),
                ],
                INTERVAL_NS,
            )
            .unwrap();
            let rows = channel.len();
            if weighted {
                series = series.with_weights(vec![weights[index]; rows].into()).unwrap();
            }
            let hide: Vec<usize> =
                hidden.iter().filter(|(unit, _)| *unit == index).map(|(_, row)| *row).collect();
            if !hide.is_empty() {
                let keep: Vec<bool> = (0..rows).map(|row| !hide.contains(&row)).collect();
                series = series.with_analysis_mask(ValidityBitmap::from_bools(&keep)).unwrap();
            }
            PanelUnit { unit_id: u32::try_from(index).unwrap(), series }
        })
        .collect();
    PanelData::try_new(Arc::from(units)).unwrap()
}

fn oracle_request(case: &serde_json::Value, data: &PanelData) -> JointPanelRequest {
    let columns = antecedent::analysis::joint_panel::JointPanelColumns {
        channels: vec![VariableId::from_raw(0)],
        outcome: VariableId::from_raw(1),
        confounder: VariableId::from_raw(2),
        season: VariableId::from_raw(3),
    };
    let groups = case["groups"].as_array().unwrap().len();
    let worlds: Vec<_> = ["reference", "treated"]
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let program = &case["programs"][name];
            let values: Vec<f64> = program["dose"].as_array().unwrap().iter().map(number).collect();
            let history: Vec<f64> =
                program["history"].as_array().unwrap().iter().map(number).collect();
            let confounder: Vec<f64> =
                program["confounder"].as_array().unwrap().iter().map(number).collect();
            antecedent::analysis::joint_panel::JointPanelWorld {
                world: ScientificWorldId::from_raw(u32::try_from(index).unwrap()),
                reference: index == 0,
                program: antecedent_estimate::joint_panel::PanelProgram {
                    id: name.into(),
                    channels: vec![antecedent_estimate::joint_panel::PanelChannelProgram {
                        id: "0".into(),
                        values: values.clone(),
                    }],
                    scenarios: (0..groups)
                        .map(|group| antecedent_estimate::joint_panel::PanelScenario {
                            group: group.to_string(),
                            history: vec![antecedent_estimate::joint_panel::PanelChannelHistory {
                                id: "0".into(),
                                values: [history[0], history[1]],
                            }],
                            confounder: confounder.clone(),
                            season: vec![0; values.len()],
                        })
                        .collect(),
                },
            }
        })
        .collect();
    let target = joint_panel_target(
        data.schema(),
        &columns,
        2,
        &worlds,
        JointPanelFunctional::Worlds,
        antecedent_core::InferenceSelection::Frequentist,
    )
    .unwrap();
    JointPanelRequest {
        columns,
        interactions: vec![],
        seasons: 1,
        lag_order: 2,
        options: antecedent_estimate::joint_panel::PanelFitOptions {
            optimizer: antecedent_estimate::joint_panel::PanelOptimizerOptions {
                max_iterations: 50_000,
                gradient_tolerance: 1e-6,
            },
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

#[test]
fn cells_03_joint_panel_weighted_masked_and_missing_fits_match_the_dense_covariance_maximum() {
    // `joint_panel_weights_gaps_reference.py` maximizes the weighted dense-covariance
    // likelihood of a five-group unbalanced panel by Newton iteration in 60-digit arithmetic,
    // for sampling weights, missing outcome cells, both, and rows hidden by an analysis mask,
    // and evaluates the weighted response of two programs at the maximizer. The public route
    // reaches the same responses from the series' weights, mask and missing cells.
    let oracle: serde_json::Value = serde_json::from_str(ORACLE).unwrap();
    let cases = oracle["model_cases"].as_array().unwrap();
    let by_name = |name: &str| cases.iter().find(|case| case["name"] == name).unwrap();
    for (name, hidden, weights, policy, source) in [
        ("weighted", vec![], true, PanelMissingPolicy::Refuse, "weighted"),
        ("gapped", vec![], false, PanelMissingPolicy::DropPeriods, "gapped"),
        (
            "weighted_and_gapped",
            vec![],
            true,
            PanelMissingPolicy::DropPeriods,
            "weighted_and_gapped",
        ),
        ("masked_rows", vec![(2, 7), (4, 2)], false, PanelMissingPolicy::Refuse, "weighted"),
    ] {
        let case = by_name(name);
        let data = oracle_panel(by_name(source), &hidden, weights);
        let request = oracle_request(case, &data);
        let prepared = Study::prepare_joint_panel_with_policy(data, request, policy, &ctx())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        let result = prepared.estimate(&ctx()).unwrap();
        let JointPanelFit::Invariant(fit) = &result.fit else { panic!("invariant fit") };
        assert!(fit.diagnostics.optimizer_converged, "{name}: {:?}", fit.diagnostics);
        // The maximized weighted log-likelihood is the dense one.
        // The route scales the weights to mean one; only their ratios enter the estimate.
        let mean_weight = if weights {
            let w: Vec<f64> = case["weights"].as_array().unwrap().iter().map(number).collect();
            w.iter().sum::<f64>() / w.len() as f64
        } else {
            1.0
        };
        let want = case["mle_log_likelihood"].as_f64().unwrap() / mean_weight;
        let got = fit.diagnostics.log_likelihood;
        assert!(
            (got - want).abs() <= 2e-4 * want.abs().max(1.0),
            "{name}: maximized log-likelihood {got} vs dense {want}"
        );
        for (world, program) in ["reference", "treated"].into_iter().enumerate() {
            let expected: Vec<f64> =
                case["responses"][program].as_array().unwrap().iter().map(number).collect();
            for (period, (got, want)) in result.point[world].means.iter().zip(&expected).enumerate() {
                assert!(
                    (got - want).abs() <= 2e-3 * (1.0 + want.abs()),
                    "{name} {program} period {period}: {got} vs dense {want}"
                );
            }
        }
    }
}

#[test]
fn cells_03_joint_panel_pools_multiply_imputed_responses_with_rubins_rules() {
    use antecedent_data::ImputationSpec;
    use antecedent_estimate::multiple_imputation::PooledVectorEstimate;
    let ctx = ExecutionContext::for_tests(20_261_101);
    let data = fixture::source(false);
    let request = fixture::request(&data, false, PanelInference::Frequentist { draws: 40 });
    let spec = ImputationSpec::new().with_imputations(4).with_iterations(3).with_seed(11);
    let missing = edited(&data, |unit, columns| {
        if unit == 2 {
            for row in [10, 15, 20] {
                columns[2][row] = f64::NAN;
            }
        }
        if unit == 1 {
            columns[3][12] = f64::NAN;
        }
    });
    let domains = [(Arc::<str>::from("confounder"), antecedent_data::ImputationDomain::Continuous)];
    let prepared = Study::prepare_joint_panel_imputed_with_domains(
        missing.clone(),
        request.clone(),
        spec,
        &domains,
        &ctx,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let result = prepared.estimate(&ctx).unwrap();
    assert_eq!(result.imputation_assumptions.domains.as_slice(), &domains);
    assert!(result.imputation_assumptions.binary_boundary_variables.is_empty());
    assert_eq!(result.imputation_assumptions.digest().len(), 64);
    assert_eq!(result.programs.len(), 4);
    assert_eq!(result.horizons, vec![24; 4]);
    assert_eq!(result.fits.len(), 4);
    assert_eq!(
        result.imputed_cells.iter().map(|(name, count)| (name.as_ref(), *count)).collect::<Vec<_>>(),
        vec![("outcome", 3), ("confounder", 1)]
    );
    let PooledVectorEstimate::Rubin(pooled) = &result.pooled else {
        panic!("frequentist fits are pooled by Rubin's rules")
    };
    assert_eq!((pooled.imputations, pooled.dim), (4, 96));
    // Rubin's rules, coordinate by coordinate, from the per-imputation fits and the pooled
    // moments: the point is their mean, the between variance their sample variance, and the
    // total variance is within + (1 + 1/m) between.
    let m = 4.0_f64;
    for coordinate in 0..96 {
        let values: Vec<f64> = result.fits.iter().map(|fit| fit[coordinate]).collect();
        let mean = values.iter().sum::<f64>() / m;
        let between = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (m - 1.0);
        let pooled_coordinate = &pooled.coordinates[coordinate];
        assert!((pooled.point[coordinate] - mean).abs() < 1e-12);
        assert!((pooled_coordinate.point - mean).abs() < 1e-12);
        assert!((pooled_coordinate.between - between).abs() < 1e-12 * (1.0 + between));
        assert!(pooled_coordinate.within > 0.0, "the fit's own uncertainty enters");
        let total = pooled_coordinate.within + (1.0 + 1.0 / m) * between;
        assert!((pooled_coordinate.total - total).abs() < 1e-12 * (1.0 + total));
    }
    // The imputed cells matter: the imputations differ from one another.
    assert!(pooled.between.iter().any(|value| *value > 0.0));

    // A panel with nothing missing has nothing to impute: every imputation is the observed
    // panel, so the pooled point is the ordinary estimate and there is no between variance.
    let complete = Study::prepare_joint_panel_imputed(data.clone(), request.clone(), spec, &ctx);
    match complete {
        Ok(prepared) => {
            let result = prepared.estimate(&ctx).unwrap();
            let ordinary = Study::prepare_joint_panel(data.clone(), request.clone(), &ctx)
                .unwrap()
                .estimate(&ctx)
                .unwrap();
            let flat: Vec<f64> =
                ordinary.point.iter().flat_map(|world| world.means.iter().copied()).collect();
            let point = result.pooled.point();
            for (pooled, plain) in point.iter().zip(&flat) {
                assert!((pooled - plain).abs() < 1e-9, "{pooled} vs {plain}");
            }
            assert!(result.imputed_cells.is_empty());
        }
        Err(error) => assert!(error.to_string().contains("impute"), "{error}"),
    }

    // Refusals, each by name: an assignment or season level is not imputed, a mask is not
    // imputed around, and pooling needs each fit's own uncertainty.
    let refuse = |data: PanelData, request: JointPanelRequest| {
        refusal(Study::prepare_joint_panel_imputed(data, request, spec, &ctx)).to_string()
    };
    let hole = edited(&data, |unit, columns| {
        if unit == 0 {
            columns[0][12] = f64::NAN;
        }
    });
    assert!(refuse(hole, request.clone()).contains("intervention itself"));
    let hole = edited(&data, |unit, columns| {
        if unit == 0 {
            columns[4][12] = f64::NAN;
        }
    });
    assert!(refuse(hole, request.clone()).contains("design label"));
    assert!(refuse(masked(&missing, 0, &[3]), request.clone()).contains("analysis mask"));
    let mut no_draws = request;
    no_draws.options.inference = PanelInference::Frequentist { draws: 0 };
    assert!(refuse(missing, no_draws).contains("uncertainty draws"));
}
