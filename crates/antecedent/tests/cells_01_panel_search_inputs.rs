// SPDX-License-Identifier: MIT OR Apache-2.0
//! What the panel program-search route does with declarations the response fit cannot honour:
//! missing cells under no declared policy, a categorical channel,
//! negative or fractional codes, a constant outcome, and an identified causal reading of a
//! response that only the fitted model supports. Each is refused with a typed error naming the
//! cause; none is ignored, imputed or dropped. A channel that never varies is accepted, but only
//! as model-conditional: its identified reading is withheld.
#[path = "../../../examples/rust/joint_panel_fixture.rs"]
mod fixture;
use antecedent::Study;
use antecedent::analysis::joint_panel::{
    JointPanelError, JointPanelFit, JointPanelResponseFunctional, JointPanelScientificScope,
    JointPanelSearchRequest, JointPanelSearchUncertainty,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{PanelData, PanelUnit, TableView, TimeSeriesData};
use antecedent_design::decision_contract::{
    ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr,
};
use antecedent_design::program_search::{FiniteProgramDomain, ProgramControl};
use antecedent_estimate::joint_panel::{PanelError, PanelInference};
use std::sync::Arc;

fn rebuilt(edit: impl Fn(u32, &mut Vec<Vec<f64>>)) -> PanelData {
    let base = fixture::source(false);
    let units: Vec<PanelUnit> = base
        .units()
        .iter()
        .map(|unit| {
            let mut cols: Vec<Vec<f64>> = (0..5u32)
                .map(|i| unit.series.float64_cow(VariableId::from_raw(i)).unwrap().to_vec())
                .collect();
            edit(unit.unit_id, &mut cols);
            let series = TimeSeriesData::from_f64_columns(
                ["channel_a", "channel_b", "outcome", "z", "season"]
                    .into_iter()
                    .zip(cols.iter().map(Vec::as_slice)),
                604_800_000_000_000,
            )
            .unwrap();
            PanelUnit { unit_id: unit.unit_id, series }
        })
        .collect();
    PanelData::try_new(Arc::from(units)).unwrap()
}

fn prepare(
    data: PanelData,
    draws: usize,
) -> Result<antecedent::analysis::joint_panel::JointPanelResult, JointPanelError> {
    let ctx = ExecutionContext::for_tests(5);
    let request = fixture::request(&data, false, PanelInference::Frequentist { draws });
    Study::prepare_joint_panel(data, request, &ctx)?.estimate(&ctx)
}

fn weighted_units(weights: [f64; 4]) -> PanelData {
    let base = fixture::source(false);
    let units: Vec<PanelUnit> = base
        .units()
        .iter()
        .zip(weights)
        .map(|(unit, weight)| {
            let n = unit.series.row_count();
            PanelUnit {
                unit_id: unit.unit_id,
                series: unit.series.with_weights(vec![weight; n].into()).unwrap(),
            }
        })
        .collect();
    PanelData::try_new(Arc::from(units)).unwrap()
}

#[test]
fn cells_01_weighted_or_masked_panel_units_are_read_not_ignored() {
    let plain = prepare(fixture::source(false), 0).unwrap();
    let terminal = |result: &antecedent::analysis::joint_panel::JointPanelResult| -> Vec<f64> {
        result.point.iter().map(|world| *world.means.last().unwrap()).collect()
    };
    // Equal weights, and a mask that hides nothing, are the unweighted panel: same answer.
    let equal = prepare(weighted_units([3.0; 4]), 0).unwrap();
    assert_eq!(terminal(&equal), terminal(&plain));
    let masked: Vec<PanelUnit> = fixture::source(false)
        .units()
        .iter()
        .map(|unit| {
            let n = unit.series.row_count();
            let mask = antecedent_data::ValidityBitmap::all_valid(n);
            PanelUnit {
                unit_id: unit.unit_id,
                series: unit.series.with_analysis_mask(mask).unwrap(),
            }
        })
        .collect();
    let hidden_nothing = prepare(PanelData::try_new(Arc::from(masked)).unwrap(), 0).unwrap();
    assert_eq!(terminal(&hidden_nothing), terminal(&plain));
    // Unequal weights change the population the response averages describe, and a search over
    // the weighted fit is still a search over a converged mechanism.
    let unequal = prepare(weighted_units([1.0, 4.0, 1.0, 2.0]), 0).unwrap();
    assert!(
        terminal(&unequal).iter().zip(terminal(&plain)).any(|(a, b)| (a - b).abs() > 1e-6),
        "unequal weights must move the weighted mean"
    );
}

#[test]
fn cells_01_missing_cells_and_invalid_codes_are_refused_before_any_fit() {
    // A missing outcome cell, a missing channel cell and a negative channel code (a categorical
    // code is not a dose) are all invalid input to the engine, naming the finite-value rule.
    for (label, data) in [
        ("missing outcome", rebuilt(|_, c| c[2][10] = f64::NAN)),
        ("missing channel", rebuilt(|_, c| c[0][10] = f64::NAN)),
        ("negative channel code", rebuilt(|_, c| c[0][10] = -1.0)),
    ] {
        let error = prepare(data, 0).unwrap_err();
        assert!(
            matches!(&error, JointPanelError::Engine(PanelError::Invalid(detail)) if detail.contains("finite values")),
            "{label}: {error}"
        );
    }
    // The seasonal level is the categorical covariate: a non-integer or out-of-range code is
    // refused by name instead of being rounded or clipped into a neighbouring level.
    for (label, data) in [
        ("fractional season", rebuilt(|_, c| c[4][10] = 1.5)),
        ("season beyond the declared levels", rebuilt(|_, c| c[4][10] = 9.0)),
    ] {
        let error = prepare(data, 0).unwrap_err();
        assert!(
            matches!(&error, JointPanelError::Compile(detail) if detail.contains("season must be a finite declared integer level")),
            "{label}: {error}"
        );
    }
}

#[test]
fn cells_01_a_constant_outcome_is_a_typed_numerical_failure_not_an_estimate() {
    let error = prepare(rebuilt(|_, c| c[2].iter_mut().for_each(|v| *v = 3.0)), 0).unwrap_err();
    assert!(
        matches!(&error, JointPanelError::Engine(PanelError::Numerical(detail)) if detail.contains("nonfinite local likelihood score")),
        "{error}"
    );
}

fn search_request(
    result: &antecedent::analysis::joint_panel::JointPanelResult,
    scope: JointPanelScientificScope,
) -> JointPanelSearchRequest {
    let horizon = result.request().worlds[0].horizon();
    let controls = (0..horizon)
        .flat_map(|offset| {
            result.request().columns.channels.iter().map(move |&channel| ProgramControl {
                variable: channel,
                offset: i32::try_from(offset).unwrap(),
                lower: 0.0,
                upper: 4.0,
                unit_cost: 0.0,
            })
        })
        .collect();
    JointPanelSearchRequest {
        scope,
        functional: JointPanelResponseFunctional::TerminalMean,
        uncertainty: JointPanelSearchUncertainty::PointMeans,
        contract: ProgramSearchContract {
            space: ProgramSearchSpace {
                id: "panel_response".into(),
                variables: vec!["response".into()],
            },
            objective: ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)),
            constraints: vec![],
        },
        domain: FiniteProgramDomain { controls, budget: None, restrictions: vec![] },
        utility_units: "response".into(),
        max_candidates: 8,
    }
}

#[test]
fn cells_01_a_model_only_response_is_searched_as_conditional_and_its_identified_reading_is_refused()
{
    let ctx = ExecutionContext::for_tests(5);
    // The declared worlds move both channels to 4, a whole program the data never realized as a
    // unit; without a whole-program design the response is the fitted model's alone.
    let result = prepare(fixture::source(false), 0).unwrap();
    assert!(!result.identification().is_identified_claim());
    let conditional = result
        .compile_search(&search_request(&result, JointPanelScientificScope::ModelConditional), &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    assert!(conditional.result.exhaustive);
    assert_eq!(conditional.result.continuous_gap, None);
    assert_eq!(conditional.result.reliability, "unmeasured");
    assert_eq!(conditional.result.rows.len(), 4);
    let error = result
        .compile_search(&search_request(&result, JointPanelScientificScope::IdentifiedCausal), &ctx)
        .map(|_| ())
        .unwrap_err();
    assert!(
        matches!(&error, JointPanelError::Unsupported(detail) if detail.contains("does not identify this target")),
        "{error}"
    );
}

#[test]
fn cells_01_constant_and_exactly_collinear_channels_are_refused_as_unidentified() {
    let error = prepare(rebuilt(|_, c| c[1].iter_mut().for_each(|v| *v = 1.0)), 0).unwrap_err();
    assert!(
        matches!(&error, JointPanelError::Engine(PanelError::Invalid(detail)) if detail.contains("is constant across every group and period")),
        "{error}"
    );
    let error = prepare(
        rebuilt(|_, c| {
            let a = c[0].clone();
            c[1] = a.iter().map(|v| 2.0 * v).collect();
        }),
        0,
    )
    .unwrap_err();
    assert!(
        matches!(&error, JointPanelError::Engine(PanelError::Invalid(detail)) if detail.contains("exactly collinear")),
        "{error}"
    );
}

#[test]
fn cells_01_point_mean_search_refuses_a_fit_without_a_converged_optimizer_and_curvature() {
    use antecedent::analysis::joint_panel::JointPanelContinuousSearchRequest;
    use antecedent_design::program_search::continuous::ContinuousSearchOptions;
    let ctx = ExecutionContext::for_tests(5);
    let fit = |iterations: usize| {
        let data = fixture::source(false);
        let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
        request.options.optimizer.max_iterations = iterations;
        Study::prepare_joint_panel(data, request, &ctx).unwrap().estimate(&ctx).unwrap()
    };
    let continuous = |result: &antecedent::analysis::joint_panel::JointPanelResult| {
        let finite = search_request(result, JointPanelScientificScope::ModelConditional);
        JointPanelContinuousSearchRequest {
            base_world: result.request().worlds[0].world,
            scope: JointPanelScientificScope::ModelConditional,
            functional: JointPanelResponseFunctional::TerminalMean,
            uncertainty: JointPanelSearchUncertainty::PointMeans,
            contract: finite.contract.clone(),
            domain: finite.domain.clone(),
            utility_units: "response".into(),
            options: ContinuousSearchOptions {
                max_evaluations: 8,
                max_iterations: 2,
                initial_step: 0.25,
                step_tolerance: 1e-3,
                objective_tolerance: 1e-9,
            },
            initial: vec![1.0; finite.domain.controls.len()],
        }
    };
    // Two accepted iterations stop far from a stationary point: the point is not a fitted
    // mechanism, so neither the finite nor the continuous point-mean search reads it.
    let unconverged = fit(2);
    let JointPanelFit::Invariant(partial) = &unconverged.fit else { panic!("invariant fit") };
    assert!(!partial.diagnostics.optimizer_converged);
    let error = unconverged
        .compile_search(&search_request(&unconverged, JointPanelScientificScope::ModelConditional), &ctx)
        .map(|_| ())
        .unwrap_err();
    assert!(
        matches!(&error, JointPanelError::Unsupported(detail)
            if detail.contains("converged optimizer with positive-definite curvature")),
        "{error}"
    );
    let error = unconverged.compile_continuous_search(continuous(&unconverged), &ctx).map(|_| ()).unwrap_err();
    assert!(
        matches!(&error, JointPanelError::Unsupported(detail)
            if detail.contains("converged optimizer with positive-definite curvature")),
        "{error}"
    );
    // The converged fit of the same panel passes both gates.
    let converged = fit(2000);
    let JointPanelFit::Invariant(full) = &converged.fit else { panic!("invariant fit") };
    assert!(full.diagnostics.optimizer_converged && full.diagnostics.curvature_available);
    converged
        .compile_search(&search_request(&converged, JointPanelScientificScope::ModelConditional), &ctx)
        .unwrap();
    converged.compile_continuous_search(continuous(&converged), &ctx).unwrap();
}
