//! Identity-site tests of a joint-panel program search's declaration, source and domain
//! digests (`formal/identity_sites.toml`).
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::*;
use crate::identity_laws::{all_distinct, bit_exact, ordered_laws};
use antecedent_core::VariableId;
use antecedent_design::decision_contract::{ProgramObjective, ProgramSearchSpace, UtilityExpr};
use antecedent_design::program_search::continuous::ContinuousSearchOptions;
use antecedent_design::program_search::{ProgramControl, ProgramRestriction};

fn control(variable: u32, offset: i32) -> ProgramControl {
    ProgramControl {
        variable: VariableId::from_raw(variable),
        offset,
        lower: 0.0,
        upper: 2.0,
        unit_cost: 1.0,
    }
}

fn restriction(id: &str, bound: f64) -> ProgramRestriction {
    ProgramRestriction { id: id.into(), coefficients: vec![1.0, 0.5, 0.25], bound }
}

fn domain() -> FiniteProgramDomain {
    FiniteProgramDomain {
        controls: vec![control(0, 0), control(1, 0), control(0, 1)],
        budget: Some(3.0),
        restrictions: vec![restriction("r1", 1.0), restriction("r2", 2.0)],
    }
}

fn digest(
    functional: &JointPanelResponseFunctional,
    uncertainty: JointPanelSearchUncertainty,
    domain: &FiniteProgramDomain,
    units: &str,
) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.joint_panel.search.site_test.v1");
    write_search(&mut writer, functional, uncertainty, domain, units);
    writer.finish()
}

#[test]
fn site_joint_panel_search_domain_controls_and_restrictions_are_ordered() {
    use JointPanelSearchUncertainty::{AlignedLaw, PointMeans};
    let base = domain();
    let terminal = JointPanelResponseFunctional::TerminalMean;
    // Controls are positional: every candidate value, initial level and restriction
    // coefficient is aligned with its control's index.
    ordered_laws(&base.controls, |controls| {
        let mut declared = base.clone();
        declared.controls = controls.to_vec();
        Some(digest(&terminal, PointMeans, &declared, "u"))
    });
    // The first violated restriction is the reported reason, so their order is meaning.
    ordered_laws(&base.restrictions, |restrictions| {
        let mut declared = base.clone();
        declared.restrictions = restrictions.to_vec();
        Some(digest(&terminal, PointMeans, &declared, "u"))
    });
    bit_exact(|value| {
        let mut declared = base.clone();
        declared.budget = Some(value);
        Some(digest(&terminal, PointMeans, &declared, "u"))
    });
    let mut no_budget = base.clone();
    no_budget.budget = None;
    let mut zero_budget = base.clone();
    zero_budget.budget = Some(0.0);
    all_distinct(&[
        digest(&terminal, PointMeans, &base, "u"),
        digest(&terminal, AlignedLaw, &base, "u"),
        digest(&terminal, PointMeans, &no_budget, "u"),
        digest(&terminal, PointMeans, &zero_budget, "u"),
        digest(
            &JointPanelResponseFunctional::WindowAverage { from: 0, until: 1 },
            PointMeans,
            &base,
            "u",
        ),
        digest(
            &JointPanelResponseFunctional::WindowChange { before: (0, 0), after: (1, 1) },
            PointMeans,
            &base,
            "u",
        ),
        digest(
            &JointPanelResponseFunctional::WeightedSum { coefficients: vec![0.0, 1.0] },
            PointMeans,
            &base,
            "u",
        ),
        digest(&terminal, PointMeans, &base, "v"),
    ]);
}

#[test]
fn site_joint_panel_search_source_and_continuous_declaration_bind_their_parts() {
    use JointPanelSearchUncertainty::{AlignedLaw, PointMeans};
    let ctx = ExecutionContext::for_tests(1);
    let (source, request) = super::super::tests::fixture();
    let result =
        crate::Study::prepare_joint_panel(source, request, &ctx).unwrap().estimate(&ctx).unwrap();
    all_distinct(&[search_source(&result, PointMeans), search_source(&result, AlignedLaw)]);
    let contract = |space: &str| ProgramSearchContract {
        space: ProgramSearchSpace { id: space.into(), variables: vec!["response".into()] },
        objective: ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)),
        constraints: vec![],
    };
    let request = |initial: &[f64], scope| JointPanelContinuousSearchRequest {
        base_world: ScientificWorldId::from_raw(0),
        scope,
        functional: JointPanelResponseFunctional::TerminalMean,
        uncertainty: PointMeans,
        contract: contract("panel_response"),
        domain: domain(),
        utility_units: "response".into(),
        options: ContinuousSearchOptions {
            max_evaluations: 30,
            max_iterations: 2,
            initial_step: 0.35,
            step_tolerance: 1e-4,
            objective_tolerance: 1e-10,
        },
        initial: initial.to_vec(),
    };
    let declaration = |initial: &[f64], scope, operation| {
        SearchSpec::continuous(&result, &request(initial, scope), operation).declaration
    };
    // One initial level per control, by position.
    ordered_laws(&[0.0, 1.0, 2.0], |initial| {
        Some(declaration(
            initial,
            JointPanelScientificScope::ModelConditional,
            SearchOperation::Continuous,
        ))
    });
    let initial = [0.0, 1.0, 2.0];
    all_distinct(&[
        declaration(
            &initial,
            JointPanelScientificScope::ModelConditional,
            SearchOperation::Continuous,
        ),
        declaration(
            &initial,
            JointPanelScientificScope::ModelConditional,
            SearchOperation::Certified,
        ),
        declaration(
            &initial,
            JointPanelScientificScope::IdentifiedCausal,
            SearchOperation::Continuous,
        ),
    ]);
    // The lowered program binds the declaration and the objective contract.
    let lowered = |space: &str| {
        SearchSpec::continuous(
            &result,
            &request(&initial, JointPanelScientificScope::ModelConditional),
            SearchOperation::Continuous,
        )
        .lower(&contract(space))
        .unwrap()
        .identity()
    };
    all_distinct(&[lowered("panel_response"), lowered("panel_response_b")]);
    // A finite search binds its scope, enumeration bound and declared domain.
    let finite = |scope, max_candidates: usize, units: &str| {
        let request = JointPanelSearchRequest {
            scope,
            functional: JointPanelResponseFunctional::TerminalMean,
            uncertainty: PointMeans,
            contract: contract("panel_response"),
            domain: domain(),
            utility_units: units.into(),
            max_candidates,
        };
        SearchSpec::finite(&result, &request, 0).declaration
    };
    all_distinct(&[
        finite(JointPanelScientificScope::ModelConditional, 16, "response"),
        finite(JointPanelScientificScope::IdentifiedCausal, 16, "response"),
        finite(JointPanelScientificScope::ModelConditional, 17, "response"),
        finite(JointPanelScientificScope::ModelConditional, 16, "other"),
    ]);
}
