//! Tests of the clamp-experiment recovery recipe.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AssembledCoefficientRecovery, CLAMP_EXPERIMENT_RECIPE, CLAMP_EXPERIMENT_RECIPE_ATTEMPT,
    ClampExogeneity, ClampExperimentSet, DeclarationRecovery, DeclaredClampExperiment,
    DeclaredEquationRank, DeclaredNoiseLaw, DesignFault, DirectCoefficient, DrawAdmissibility,
    ExperimentDesign, ExperimentalRecoveryDecision, FiniteSampleConditioning,
    InadmissibleCoefficientDraw, JointRecoveryDraw, LATENT_VARIABLE_EXPANSION_ATTEMPT,
    NotThisRecipe, PRIOR_REGULARIZATION_ATTEMPT, PhysicalDynamicalStability,
    RecoveredCoefficientPoint, RecoveredCoefficientUncertainty, RecoveryAttempt, RecoveryDrawInput,
    RecoveryWarning, RefusalReason, RegimeAssumption, UncertaintyFault, UniquenessRestriction,
    UnresolvedReason, WeakStability, as_recovered_mechanism, decide_experimental_recovery,
    joint_draw_from_blocks, recover_assembled_coefficients, recover_declared_coefficients,
    regime_is_identified,
};
use crate::experiment_moments::{
    ConstraintRow, RegimeSample, TargetConstraintBlock, UnitColumn, assemble_experiment_constraints,
};
use crate::experiment_regime::{
    DependencePremise, ExperimentRegime, PopulationId, RegimeContract, RegimeContractDecision,
    RegimeIntervention, SourceIdentity, UnitMask, validate_regime_contract,
};
use antecedent_core::{Value, VariableId};

fn node(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn coefficient(into: u32, from: u32) -> DirectCoefficient {
    DirectCoefficient { into: node(into), from: node(from) }
}

fn all_but_one(nodes: &[VariableId]) -> Vec<DeclaredClampExperiment> {
    nodes
        .iter()
        .copied()
        .map(|passive| DeclaredClampExperiment {
            clamped: nodes.iter().copied().filter(|other| *other != passive).collect(),
            passive: vec![passive],
            exogeneity: ClampExogeneity::RandomizedIntervention,
        })
        .collect()
}

fn clamp_design(
    nodes: Vec<VariableId>,
    experiments: Vec<DeclaredClampExperiment>,
    equation_ranks: Vec<DeclaredEquationRank>,
) -> ExperimentDesign {
    ExperimentDesign {
        nodes,
        attempt: RecoveryAttempt::ClampExperiments(ClampExperimentSet {
            experiments,
            noise: DeclaredNoiseLaw::IndependentGaussian,
            regimes: RegimeAssumption::Independent,
            weak_stability: WeakStability::InvertibilityDeclared,
            physical_dynamical_stability: PhysicalDynamicalStability::NotDeclared,
            finite_sample_conditioning: FiniteSampleConditioning::NotDeclared,
            uniqueness_restriction: UniquenessRestriction::NotOffered,
            equation_ranks,
        }),
    }
}

fn three() -> Vec<VariableId> {
    vec![node(0), node(1), node(2)]
}

#[test]
fn accepts_all_but_one_randomized_clamps_under_the_minimum_recipe() {
    let nodes = three();
    let design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
    let decision = decide_experimental_recovery(&design);
    let ExperimentalRecoveryDecision::Recovered(recovery) = decision else {
        panic!("expected a closed coefficient recovery");
    };
    assert_eq!(recovery.recipe, CLAMP_EXPERIMENT_RECIPE);
    assert!(recovery.goal_closed());
    assert!(recovery.generic);
    assert!(recovery.warnings.is_empty());
    assert!(recovery.unresolved.is_empty());
    assert_eq!(
        recovery.recoverable,
        vec![
            coefficient(0, 1),
            coefficient(0, 2),
            coefficient(1, 0),
            coefficient(1, 2),
            coefficient(2, 0),
            coefficient(2, 1),
        ]
    );
}

#[test]
fn refuses_observational_equilibrium_ols_and_names_the_clamp_experiment_recipe() {
    let design =
        ExperimentDesign { nodes: three(), attempt: RecoveryAttempt::ObservationalEquilibriumOls };
    let decision = decide_experimental_recovery(&design);
    assert!(!decision.goal_closed());
    assert!(decision.coefficient_recovery().is_none());
    let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
        panic!("expected a refusal");
    };
    assert_eq!(refusal.reason, RefusalReason::ObservationalEquilibriumOls);
    assert_eq!(refusal.next_attempt, CLAMP_EXPERIMENT_RECIPE_ATTEMPT);
    assert_eq!(refusal.next_attempt.name(), CLAMP_EXPERIMENT_RECIPE);
}

#[test]
fn insufficient_coverage_returns_partial_recovery_without_a_mechanism_matrix() {
    let nodes = three();
    let experiments = vec![DeclaredClampExperiment {
        clamped: vec![node(1), node(2)],
        passive: vec![node(0)],
        exogeneity: ClampExogeneity::RandomizedIntervention,
    }];
    let decision = decide_experimental_recovery(&clamp_design(nodes, experiments, Vec::new()));
    assert!(!decision.goal_closed());
    let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
        panic!("expected a partial recovery");
    };
    assert_eq!(recovery.recipe, CLAMP_EXPERIMENT_RECIPE);
    assert_eq!(recovery.recoverable, vec![coefficient(0, 1), coefficient(0, 2)]);
    assert_eq!(recovery.unresolved.len(), 2);
    assert_eq!(recovery.unresolved[0].support, vec![coefficient(1, 0), coefficient(1, 2)]);
    assert_eq!(
        recovery.unresolved[0].reason,
        UnresolvedReason::IncompletePairCoverage {
            target: node(1),
            missing_clamps: vec![node(0), node(2)],
        }
    );
    assert_eq!(recovery.unresolved[1].support, vec![coefficient(2, 0), coefficient(2, 1)]);
    assert_eq!(
        recovery.unresolved[1].reason,
        UnresolvedReason::IncompletePairCoverage {
            target: node(2),
            missing_clamps: vec![node(0), node(1)],
        }
    );
}

#[test]
fn rank_deficiency_names_unresolved_directions_and_leaves_the_goal_open() {
    let nodes = three();
    let ranks = vec![DeclaredEquationRank {
        target: node(0),
        column_rank: 1,
        nullspace: vec![vec![coefficient(0, 1)]],
    }];
    let decision =
        decide_experimental_recovery(&clamp_design(nodes.clone(), all_but_one(&nodes), ranks));
    assert!(!decision.goal_closed());
    let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
        panic!("expected the goal to stay open");
    };
    assert_eq!(
        recovery.recoverable,
        vec![
            coefficient(0, 2),
            coefficient(1, 0),
            coefficient(1, 2),
            coefficient(2, 0),
            coefficient(2, 1),
        ]
    );
    assert_eq!(recovery.unresolved.len(), 1);
    assert_eq!(recovery.unresolved[0].support, vec![coefficient(0, 1)]);
    assert_eq!(
        recovery.unresolved[0].reason,
        UnresolvedReason::DeficientConstraintRank { target: node(0), column_rank: 1, width: 2 }
    );
}

#[test]
fn records_latent_variable_expansion_as_not_this_recipe() {
    let as_attempt =
        ExperimentDesign { nodes: three(), attempt: RecoveryAttempt::LatentVariableExpansion };
    let decision = decide_experimental_recovery(&as_attempt);
    assert!(!decision.goal_closed());
    assert!(decision.coefficient_recovery().is_none());
    let ExperimentalRecoveryDecision::NotThisRecipe(attempt) = decision else {
        panic!("expected not_this_recipe");
    };
    assert_eq!(attempt, NotThisRecipe::LatentVariableExpansion);
    assert_eq!(attempt.name(), LATENT_VARIABLE_EXPANSION_ATTEMPT);

    let nodes = three();
    let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
    clamp_set_mut(&mut design).noise = DeclaredNoiseLaw::LatentConfounding;
    let decision = decide_experimental_recovery(&design);
    assert!(matches!(
        decision,
        ExperimentalRecoveryDecision::NotThisRecipe(NotThisRecipe::LatentVariableExpansion)
    ));
}

#[test]
fn records_prior_regularization_as_not_this_recipe() {
    let design = ExperimentDesign { nodes: three(), attempt: RecoveryAttempt::PriorRegularization };
    let decision = decide_experimental_recovery(&design);
    assert!(!decision.goal_closed());
    assert!(decision.coefficient_recovery().is_none());
    let ExperimentalRecoveryDecision::NotThisRecipe(attempt) = decision else {
        panic!("expected not_this_recipe");
    };
    assert_eq!(attempt, NotThisRecipe::PriorRegularization);
    assert_eq!(attempt.name(), PRIOR_REGULARIZATION_ATTEMPT);
}

#[test]
fn refuses_a_clamp_conditioned_on_the_factual_disturbance() {
    let nodes = three();
    let mut experiments = all_but_one(&nodes);
    experiments[0].exogeneity = ClampExogeneity::ConditionedOnFactualDisturbance;
    let decision = decide_experimental_recovery(&clamp_design(nodes, experiments, Vec::new()));
    assert!(decision.coefficient_recovery().is_none());
    let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
        panic!("expected an exogeneity refusal");
    };
    assert_eq!(refusal.reason, RefusalReason::ClampNotExogenous { experiment: 0 });
    assert_eq!(refusal.next_attempt.name(), CLAMP_EXPERIMENT_RECIPE);
}

#[test]
fn keeps_weak_stability_distinct_from_physical_dynamical_stability() {
    let nodes = three();
    let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
    {
        let set = clamp_set_mut(&mut design);
        set.weak_stability = WeakStability::NotDeclared;
        set.physical_dynamical_stability = PhysicalDynamicalStability::Declared;
    }
    let decision = decide_experimental_recovery(&design);
    let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
        panic!("physical dynamical stability must not stand in for invertibility");
    };
    assert_eq!(refusal.reason, RefusalReason::PhysicalDynamicalStabilityIsNotWeakStability);

    let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
    clamp_set_mut(&mut design).weak_stability = WeakStability::NotDeclared;
    let decision = decide_experimental_recovery(&design);
    let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
        panic!("missing invertibility must be refused");
    };
    assert_eq!(refusal.reason, RefusalReason::WeakStabilityNotDeclared);

    let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
    clamp_set_mut(&mut design).physical_dynamical_stability = PhysicalDynamicalStability::Declared;
    let decision = decide_experimental_recovery(&design);
    assert!(matches!(decision, ExperimentalRecoveryDecision::Recovered(_)));
    assert!(decision.goal_closed());
}

#[test]
fn refuses_dependent_regimes() {
    let nodes = three();
    let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
    clamp_set_mut(&mut design).regimes = RegimeAssumption::Dependent;
    let decision = decide_experimental_recovery(&design);
    let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
        panic!("expected a regime refusal");
    };
    assert_eq!(refusal.reason, RefusalReason::DependentRegimes);
    assert_eq!(refusal.next_attempt, CLAMP_EXPERIMENT_RECIPE_ATTEMPT);
}

#[test]
fn refuses_an_undeclared_noise_law() {
    let nodes = three();
    let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
    clamp_set_mut(&mut design).noise = DeclaredNoiseLaw::Undeclared;
    let decision = decide_experimental_recovery(&design);
    let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
        panic!("expected a noise-law refusal");
    };
    assert_eq!(refusal.reason, RefusalReason::NoiseLawUndeclared);
}

#[test]
fn null_experiment_does_not_satisfy_pair_coverage() {
    let nodes = three();
    let experiments = vec![DeclaredClampExperiment {
        clamped: Vec::new(),
        passive: nodes.clone(),
        exogeneity: ClampExogeneity::RandomizedIntervention,
    }];
    let decision = decide_experimental_recovery(&clamp_design(nodes, experiments, Vec::new()));
    let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
        panic!("a null experiment is under-covered, not an OLS refusal");
    };
    assert!(!recovery.goal_closed());
    assert!(recovery.recoverable.is_empty());
    assert_eq!(recovery.unresolved.len(), 3);
}

#[test]
fn refuses_a_clamp_set_that_does_not_partition_the_nodes() {
    let nodes = three();
    let experiments = vec![DeclaredClampExperiment {
        clamped: vec![node(0), node(1)],
        passive: vec![node(1)],
        exogeneity: ClampExogeneity::RandomizedIntervention,
    }];
    let decision = decide_experimental_recovery(&clamp_design(nodes, experiments, Vec::new()));
    let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
        panic!("expected a design fault");
    };
    assert_eq!(
        refusal.reason,
        RefusalReason::DesignFault(DesignFault::ExperimentDoesNotPartition { experiment: 0 })
    );
}

fn deficient_into_zero() -> DeclaredEquationRank {
    DeclaredEquationRank {
        target: node(0),
        column_rank: 0,
        nullspace: vec![vec![coefficient(0, 1)], vec![coefficient(0, 2)]],
    }
}

fn one_passive_experiment() -> Vec<DeclaredClampExperiment> {
    vec![DeclaredClampExperiment {
        clamped: vec![node(1), node(2)],
        passive: vec![node(0)],
        exogeneity: ClampExogeneity::RandomizedIntervention,
    }]
}

#[test]
fn sample_full_column_rank_does_not_recover_a_weak_design() {
    let nodes = three();
    let mut design = clamp_design(nodes, one_passive_experiment(), Vec::new());
    clamp_set_mut(&mut design).finite_sample_conditioning =
        FiniteSampleConditioning::SampleFullColumnRank;
    let decision = decide_experimental_recovery(&design);
    let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
        panic!("a sample full rank must not close an under-covered row");
    };
    assert!(!recovery.goal_closed());
    assert_eq!(recovery.finite_sample_conditioning, FiniteSampleConditioning::SampleFullColumnRank);
    assert_eq!(recovery.uniqueness_restriction, UniquenessRestriction::NotOffered);
    assert_eq!(recovery.recoverable, vec![coefficient(0, 1), coefficient(0, 2)]);
    assert_eq!(recovery.unresolved.len(), 2);
    assert!(recovery.unresolved.iter().all(|direction| {
        matches!(direction.reason, UnresolvedReason::IncompletePairCoverage { .. })
    }));
    assert!(!recovery.recoverable.contains(&coefficient(1, 0)));
    assert!(!recovery.recoverable.contains(&coefficient(2, 0)));
}

#[test]
fn sample_full_column_rank_does_not_license_a_rank_on_an_uncovered_row() {
    let nodes = three();
    let ranks =
        vec![DeclaredEquationRank { target: node(1), column_rank: 2, nullspace: Vec::new() }];
    let mut design = clamp_design(nodes, one_passive_experiment(), ranks);
    clamp_set_mut(&mut design).finite_sample_conditioning =
        FiniteSampleConditioning::SampleFullColumnRank;
    let decision = decide_experimental_recovery(&design);
    let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
        panic!("an uncovered rank declaration is a partial report, not a refusal");
    };
    assert!(!recovery.goal_closed());
    assert!(!recovery.generic);
    assert_eq!(
        recovery.warnings,
        vec![RecoveryWarning::RankDeclarationOnUncoveredTarget { target: node(1) }]
    );
    assert_eq!(recovery.warnings[0].design_fault(), DesignFault::RankDeclarationOnUncoveredTarget);
    assert!(recovery.recoverable.iter().all(|coefficient| coefficient.into == node(0)));
    assert!(recovery.unresolved.iter().any(|direction| {
        matches!(
            direction.reason,
            UnresolvedReason::IncompletePairCoverage { target, .. } if target == node(1)
        )
    }));
}

#[test]
fn finite_condition_number_does_not_fill_a_rank_deficient_row() {
    let nodes = three();
    let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), vec![deficient_into_zero()]);
    clamp_set_mut(&mut design).finite_sample_conditioning =
        FiniteSampleConditioning::FiniteConditionNumber;
    let decision = decide_experimental_recovery(&design);
    let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
        panic!("a finite condition number must not fill a nullspace");
    };
    assert!(!recovery.goal_closed());
    assert_eq!(
        recovery.finite_sample_conditioning,
        FiniteSampleConditioning::FiniteConditionNumber
    );
    assert_eq!(
        recovery.recoverable,
        vec![coefficient(1, 0), coefficient(1, 2), coefficient(2, 0), coefficient(2, 1)]
    );
    assert_eq!(recovery.unresolved.len(), 2);
    assert_eq!(recovery.unresolved[0].support, vec![coefficient(0, 1)]);
    assert_eq!(recovery.unresolved[1].support, vec![coefficient(0, 2)]);
    assert!(recovery.unresolved.iter().all(|direction| {
        direction.reason
            == UnresolvedReason::DeficientConstraintRank {
                target: node(0),
                column_rank: 0,
                width: 2,
            }
    }));
}

#[test]
fn finite_condition_number_does_not_open_a_covered_full_rank_design() {
    let nodes = three();
    let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
    clamp_set_mut(&mut design).finite_sample_conditioning =
        FiniteSampleConditioning::FiniteConditionNumber;
    let decision = decide_experimental_recovery(&design);
    let ExperimentalRecoveryDecision::Recovered(recovery) = decision else {
        panic!("a finite condition number is not a rank deficiency");
    };
    assert!(recovery.goal_closed());
    assert!(recovery.unresolved.is_empty());
    assert_eq!(recovery.recoverable.len(), 6);
    assert_eq!(
        recovery.finite_sample_conditioning,
        FiniteSampleConditioning::FiniteConditionNumber
    );
}

#[test]
fn physical_stability_does_not_resolve_rank_deficiency() {
    let nodes = three();
    let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), vec![deficient_into_zero()]);
    clamp_set_mut(&mut design).physical_dynamical_stability = PhysicalDynamicalStability::Declared;
    let decision = decide_experimental_recovery(&design);
    let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
        panic!("physical stability must not select the nullspace");
    };
    assert!(!recovery.goal_closed());
    assert_eq!(recovery.uniqueness_restriction, UniquenessRestriction::NotOffered);
    assert_eq!(recovery.unresolved.len(), 2);
    assert!(!recovery.recoverable.contains(&coefficient(0, 1)));
    assert!(!recovery.recoverable.contains(&coefficient(0, 2)));
}

#[test]
fn uniqueness_restriction_is_not_a_stability_certificate() {
    for restriction in [
        UniquenessRestriction::MinimumNormCompletion,
        UniquenessRestriction::BackgroundZero,
        UniquenessRestriction::Faithfulness,
    ] {
        let nodes = three();
        let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
        {
            let set = clamp_set_mut(&mut design);
            set.weak_stability = WeakStability::NotDeclared;
            set.uniqueness_restriction = restriction;
        }
        let decision = decide_experimental_recovery(&design);
        assert!(decision.coefficient_recovery().is_none());
        let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
            panic!("a uniqueness restriction must not stand in for invertibility");
        };
        assert_eq!(refusal.reason, RefusalReason::UniquenessRestrictionIsNotAStabilityCertificate);
        assert_eq!(refusal.next_attempt, CLAMP_EXPERIMENT_RECIPE_ATTEMPT);
    }

    let nodes = three();
    let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
    {
        let set = clamp_set_mut(&mut design);
        set.weak_stability = WeakStability::NotDeclared;
        set.physical_dynamical_stability = PhysicalDynamicalStability::Declared;
        set.uniqueness_restriction = UniquenessRestriction::MinimumNormCompletion;
    }
    let decision = decide_experimental_recovery(&design);
    let ExperimentalRecoveryDecision::Refused(refusal) = decision else {
        panic!("physical stability stays a distinct refusal");
    };
    assert_eq!(refusal.reason, RefusalReason::PhysicalDynamicalStabilityIsNotWeakStability);

    let nodes = three();
    let mut design = clamp_design(nodes.clone(), all_but_one(&nodes), Vec::new());
    clamp_set_mut(&mut design).uniqueness_restriction =
        UniquenessRestriction::MinimumNormCompletion;
    let decision = decide_experimental_recovery(&design);
    let ExperimentalRecoveryDecision::Recovered(recovery) = decision else {
        panic!("a restriction on a closed design is not a stability certificate");
    };
    assert!(recovery.goal_closed());
    assert_eq!(recovery.uniqueness_restriction, UniquenessRestriction::MinimumNormCompletion);
    assert_eq!(
        clamp_set_mut(&mut design).physical_dynamical_stability,
        PhysicalDynamicalStability::NotDeclared
    );
}

#[test]
fn minimum_norm_and_background_zero_do_not_fill_rank_deficiency() {
    for restriction in [
        UniquenessRestriction::MinimumNormCompletion,
        UniquenessRestriction::BackgroundZero,
        UniquenessRestriction::Faithfulness,
    ] {
        let nodes = three();
        let mut design =
            clamp_design(nodes.clone(), all_but_one(&nodes), vec![deficient_into_zero()]);
        clamp_set_mut(&mut design).uniqueness_restriction = restriction;
        let decision = decide_experimental_recovery(&design);
        let ExperimentalRecoveryDecision::Partial(recovery) = decision else {
            panic!("a uniqueness restriction must not fill the nullspace");
        };
        assert!(!recovery.goal_closed());
        assert_eq!(recovery.uniqueness_restriction, restriction);
        assert_eq!(
            recovery.recoverable,
            vec![coefficient(1, 0), coefficient(1, 2), coefficient(2, 0), coefficient(2, 1)]
        );
        assert_eq!(recovery.unresolved[0].support, vec![coefficient(0, 1)]);
        assert_eq!(recovery.unresolved[1].support, vec![coefficient(0, 2)]);
    }
}

fn clamp_set_mut(design: &mut ExperimentDesign) -> &mut ClampExperimentSet {
    match &mut design.attempt {
        RecoveryAttempt::ClampExperiments(set) => set,
        RecoveryAttempt::ObservationalEquilibriumOls
        | RecoveryAttempt::PriorRegularization
        | RecoveryAttempt::LatentVariableExpansion => {
            panic!("expected a clamp-experiment attempt")
        }
    }
}

fn parallel_constraint_into_z() -> TargetConstraintBlock {
    let independent = DependencePremise::Independence;
    let contract = {
        let decision = validate_regime_contract(&RegimeContract {
            regimes: vec![
                clamp_regime(&[0, 1, 2], &[0], independent, 1),
                clamp_regime(&[0, 1, 2], &[1], independent, 2),
            ],
            mask: UnitMask { dropped: Vec::new() },
        });
        let RegimeContractDecision::Accepted(contract) = decision else {
            panic!("the fixture regime contract was refused");
        };
        contract
    };
    let (x, y, z) = ([1.0, -1.0], [1.0, -1.0], [1.0, -1.0]);
    let columns = [
        UnitColumn { unit: node(0), values: &x },
        UnitColumn { unit: node(1), values: &y },
        UnitColumn { unit: node(2), values: &z },
    ];
    let samples = [
        RegimeSample { columns: &columns, source: SourceIdentity::from_raw(1) },
        RegimeSample { columns: &columns, source: SourceIdentity::from_raw(2) },
    ];
    let assembled =
        assemble_experiment_constraints(&contract, &samples, u64::MAX).expect("z is covered");
    assembled.blocks.into_iter().find(|block| block.target == node(2)).expect("the block for z")
}

fn clamp_regime(
    units: &[u32],
    targets: &[u32],
    premise: DependencePremise,
    source: u64,
) -> ExperimentRegime {
    ExperimentRegime {
        targets: targets
            .iter()
            .map(|&raw| RegimeIntervention { unit: node(raw), value: Value::Int64(0) })
            .collect(),
        units: units.iter().copied().map(node).collect(),
        population: PopulationId::from_raw(1),
        premise,
        source: SourceIdentity::from_raw(source),
    }
}

#[test]
fn declared_rank_and_invertibility_do_not_return_coefficients() {
    let full = DeclaredEquationRank { target: node(2), column_rank: 2, nullspace: Vec::new() };
    assert_eq!(
        recover_declared_coefficients(Some(&full), WeakStability::InvertibilityDeclared),
        DeclarationRecovery::NoCoefficients
    );
    assert_eq!(
        recover_declared_coefficients(None, WeakStability::InvertibilityDeclared),
        DeclarationRecovery::NoCoefficients
    );
    let block = TargetConstraintBlock {
        target: node(2),
        sources: vec![node(0), node(1)],
        rows: vec![
            ConstraintRow { regime: 0, probe: node(0) },
            ConstraintRow { regime: 1, probe: node(1) },
        ],
        coefficients: vec![1.0, 0.0, 0.0, 1.0],
        rhs: vec![0.25, 0.5],
    };
    let from_matrix =
        recover_assembled_coefficients(&block, Some(&full), WeakStability::InvertibilityDeclared);
    assert_eq!(from_matrix.coefficients(), Some(&[0.25, 0.5][..]));
}

#[test]
fn caller_asserted_full_rank_on_a_deficient_matrix_does_not_recover_coefficients() {
    let block = parallel_constraint_into_z();
    assert_eq!(block.coefficients, vec![1.0, 1.0, 1.0, 1.0]);
    // Full column rank, with a support that is not the matrix null direction.
    let asserted = DeclaredEquationRank {
        target: block.target,
        column_rank: block.sources.len(),
        nullspace: vec![vec![coefficient(2, 0)]],
    };
    let recovery = recover_assembled_coefficients(
        &block,
        Some(&asserted),
        WeakStability::InvertibilityDeclared,
    );
    assert!(recovery.coefficients().is_none());
    let AssembledCoefficientRecovery::NullDirections {
        column_rank,
        directions,
        full_rank_was_asserted,
        ..
    } = recovery
    else {
        panic!("a full-rank claim must not replace the matrix nullspace");
    };
    assert!(full_rank_was_asserted);
    assert_eq!(column_rank, 1);
    assert_eq!(directions, vec![vec![-1.0, 1.0]]);
    let without_invertibility =
        recover_assembled_coefficients(&block, Some(&asserted), WeakStability::NotDeclared);
    assert_eq!(without_invertibility.coefficients(), None);
    assert!(matches!(
        without_invertibility,
        AssembledCoefficientRecovery::NullDirections { full_rank_was_asserted: true, .. }
    ));
}

fn two_nodes() -> Vec<VariableId> {
    vec![node(0), node(1)]
}

fn paired_noise(variance: f64) -> (Vec<f64>, Vec<f64>) {
    (vec![0.0, 0.0], vec![1.0, 0.0, 0.0, variance])
}

fn complete_draw(variance: f64, weight: f64, marked_inadmissible: bool) -> RecoveryDrawInput {
    let (noise_mean, noise_covariance) = paired_noise(variance);
    RecoveryDrawInput {
        coefficients: vec![(coefficient(0, 1), 0.4), (coefficient(1, 0), 0.5)],
        intercept: vec![0.0, 0.0],
        noise_mean,
        noise_covariance,
        unresolved: Vec::new(),
        weight,
        marked_inadmissible,
    }
}

#[test]
fn inadmissible_draw_stays_inadmissible_and_uncertainty_is_not_a_point() {
    let nodes = two_nodes();
    let point = RecoveredCoefficientPoint::plug_in(
        vec![(coefficient(0, 1), 0.4), (coefficient(1, 0), 0.5)],
        paired_noise(1.0).1,
    );
    let uncertainty = RecoveredCoefficientUncertainty::retain(
        &nodes,
        vec![complete_draw(1.0, 0.75, false), complete_draw(4.0, 0.25, true)],
    )
    .expect("two draws");
    let refused = uncertainty.try_collapse().expect_err("a law is not a plug-in");
    assert_eq!(refused.draws, 2);
    assert!((refused.inadmissible_mass - 0.25).abs() < 1e-12);
    assert!((uncertainty.inadmissible_mass() - 0.25).abs() < 1e-12);
    assert!((uncertainty.total_mass() - 1.0).abs() < 1e-12);
    assert_eq!(uncertainty.draws().len(), 2);
    assert_eq!(
        uncertainty.draws()[1].admissibility(),
        DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::Marked)
    );
    assert_eq!(uncertainty.draws()[1].noise_covariance(), paired_noise(4.0).1.as_slice());
    assert_ne!(uncertainty.draws()[1].noise_covariance(), point.noise_covariance());
    assert!(uncertainty.draws().iter().any(JointRecoveryDraw::is_inadmissible));
}

#[test]
fn rank_deficient_block_keeps_unresolved_directions_without_a_mechanism_matrix() {
    let nodes = two_nodes();
    let (noise_mean, noise_covariance) = paired_noise(1.0);
    let block = AssembledCoefficientRecovery::NullDirections {
        target: node(1),
        sources: vec![node(0)],
        column_rank: 0,
        directions: vec![vec![1.0]],
        full_rank_was_asserted: false,
    };
    let draw = joint_draw_from_blocks(
        &nodes,
        &[block],
        vec![0.0, 0.0],
        noise_mean,
        noise_covariance.clone(),
        1.0,
    )
    .expect("null directions are a draw");
    assert_eq!(
        draw.admissibility(),
        DrawAdmissibility::Inadmissible(InadmissibleCoefficientDraw::UnresolvedDirections)
    );
    assert!(draw.coefficients().is_empty());
    assert_eq!(draw.unresolved().len(), 1);
    assert_eq!(draw.unresolved()[0].support, vec![coefficient(1, 0)]);
    assert_eq!(draw.noise_covariance(), noise_covariance.as_slice());
    assert!(draw.coefficients().len() < nodes.len() * nodes.len());
    assert!(as_recovered_mechanism(&draw).is_none());
}

fn draw_from(block: AssembledCoefficientRecovery) -> Result<JointRecoveryDraw, UncertaintyFault> {
    let (noise_mean, noise_covariance) = paired_noise(1.0);
    joint_draw_from_blocks(
        &two_nodes(),
        &[block],
        vec![0.0, 0.0],
        noise_mean,
        noise_covariance,
        1.0,
    )
}

#[test]
fn block_with_mismatched_values_is_refused_and_named() {
    let refused = draw_from(AssembledCoefficientRecovery::Coefficients {
        target: node(1),
        sources: vec![node(0)],
        values: vec![0.5, 0.6],
    })
    .expect_err("a value list that does not match its sources");
    assert_eq!(
        refused,
        UncertaintyFault::BlockShape { block: 0, target: node(1), sources: 1, values: 2 }
    );
}

#[test]
fn inconsistent_and_non_constraint_blocks_are_refused_and_named() {
    let solved = AssembledCoefficientRecovery::Coefficients {
        target: node(0),
        sources: vec![node(1)],
        values: vec![0.4],
    };
    let (noise_mean, noise_covariance) = paired_noise(1.0);
    let refused = joint_draw_from_blocks(
        &two_nodes(),
        &[solved.clone(), AssembledCoefficientRecovery::Inconsistent { column_rank: 1 }],
        vec![0.0, 0.0],
        noise_mean.clone(),
        noise_covariance.clone(),
        1.0,
    )
    .expect_err("rows that disagree");
    assert_eq!(refused, UncertaintyFault::BlockInconsistent { block: 1, column_rank: 1 });
    let refused = joint_draw_from_blocks(
        &two_nodes(),
        &[solved, AssembledCoefficientRecovery::NotAConstraint],
        vec![0.0, 0.0],
        noise_mean,
        noise_covariance,
        1.0,
    )
    .expect_err("not a coefficient matrix");
    assert_eq!(refused, UncertaintyFault::BlockNotAConstraint { block: 1 });
    assert!(refused.to_string().contains("block 1"));
}

/// `x1 = 0.5 x0 + e1`, `x2 = 0.25 x0 + 0.4 x1 + e2`, `x0 = e0`.
fn simulate_chain(seed: u64, rows: usize, clamped: &[u32]) -> [Vec<f64>; 3] {
    let mut rng = Lcg(seed);
    let mut columns = [vec![0.0; rows], vec![0.0; rows], vec![0.0; rows]];
    for row in 0..rows {
        let x0 = rng.normal();
        let x1 = if clamped.contains(&1) { rng.normal() } else { 0.5 * x0 + rng.normal() };
        let x2 =
            if clamped.contains(&2) { rng.normal() } else { 0.25 * x0 + 0.4 * x1 + rng.normal() };
        columns[0][row] = x0;
        columns[1][row] = x1;
        columns[2][row] = x2;
    }
    columns
}

fn fixed_joint_holdout(rows: usize) -> [Vec<f64>; 3] {
    let mut rng = Lcg(90);
    let mut columns = [vec![2.0; rows], vec![1.25; rows], vec![0.0; rows]];
    for row in 0..rows {
        columns[2][row] = 0.25 * 2.0 + 0.4 * 1.25 + rng.normal();
    }
    columns
}

struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        self.0
    }

    fn uniform(&mut self) -> f64 {
        let bits = (self.next_u64() >> 11) as f64;
        bits / ((1_u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1.0e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::PI * 2.0 * u2).cos()
    }
}

fn single_clamp(clamped: u32) -> DeclaredClampExperiment {
    let nodes = [0_u32, 1, 2];
    DeclaredClampExperiment {
        clamped: vec![node(clamped)],
        passive: nodes.into_iter().filter(|other| *other != clamped).map(node).collect(),
        exogeneity: ClampExogeneity::RandomizedIntervention,
    }
}

fn effect(coefficients: &[(DirectCoefficient, f64)], into: u32, from: u32) -> f64 {
    coefficients
        .iter()
        .find(|(coefficient, _)| coefficient.into == node(into) && coefficient.from == node(from))
        .map(|(_, value)| *value)
        .expect("recovered coefficient")
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one held-out regime fixture and its independent coefficient uncertainty comparison"
)]
fn held_out_joint_regime_uses_coefficient_uncertainty() {
    let rows = 2_000_u16;
    let row_count = usize::from(rows);
    let training = [
        simulate_chain(1, row_count, &[0]),
        simulate_chain(2, row_count, &[1]),
        simulate_chain(3, row_count, &[2]),
    ];
    let experiments = vec![single_clamp(0), single_clamp(1), single_clamp(2)];
    let decision =
        decide_experimental_recovery(&clamp_design(three(), experiments.clone(), Vec::new()));
    let ExperimentalRecoveryDecision::Recovered(report) = &decision else {
        panic!("single-node clamps are generic identification");
    };
    assert!(report.generic);
    assert!(regime_is_identified(&decision, &experiments, &experiments[0]));
    let joint = DeclaredClampExperiment {
        clamped: vec![node(0), node(1)],
        passive: vec![node(2)],
        exogeneity: ClampExogeneity::RandomizedIntervention,
    };
    assert!(!regime_is_identified(&decision, &experiments, &joint));
    let null = DeclaredClampExperiment {
        clamped: Vec::new(),
        passive: three(),
        exogeneity: ClampExogeneity::RandomizedIntervention,
    };
    assert!(!regime_is_identified(&decision, &experiments, &null));

    let regimes = vec![
        clamp_regime(&[0, 1, 2], &[0], DependencePremise::Independence, 11),
        clamp_regime(&[0, 1, 2], &[1], DependencePremise::Independence, 12),
        clamp_regime(&[0, 1, 2], &[2], DependencePremise::Independence, 13),
    ];
    let contract = {
        let accepted = validate_regime_contract(&RegimeContract {
            regimes,
            mask: UnitMask { dropped: Vec::new() },
        });
        let RegimeContractDecision::Accepted(contract) = accepted else {
            panic!("training regimes");
        };
        contract
    };
    let c0 = [
        UnitColumn { unit: node(0), values: &training[0][0] },
        UnitColumn { unit: node(1), values: &training[0][1] },
        UnitColumn { unit: node(2), values: &training[0][2] },
    ];
    let c1 = [
        UnitColumn { unit: node(0), values: &training[1][0] },
        UnitColumn { unit: node(1), values: &training[1][1] },
        UnitColumn { unit: node(2), values: &training[1][2] },
    ];
    let c2 = [
        UnitColumn { unit: node(0), values: &training[2][0] },
        UnitColumn { unit: node(1), values: &training[2][1] },
        UnitColumn { unit: node(2), values: &training[2][2] },
    ];
    let samples = [
        RegimeSample { columns: &c0, source: SourceIdentity::from_raw(11) },
        RegimeSample { columns: &c1, source: SourceIdentity::from_raw(12) },
        RegimeSample { columns: &c2, source: SourceIdentity::from_raw(13) },
    ];
    let assembled = assemble_experiment_constraints(&contract, &samples, u64::MAX)
        .expect("single clamps cover every target");
    let mut coefficients = Vec::new();
    for block in &assembled.blocks {
        let solved = recover_assembled_coefficients(block, None, WeakStability::NotDeclared);
        let values = solved.coefficients().expect("a covered block is a mechanism row");
        for (source, value) in block.sources.iter().zip(values) {
            coefficients.push((DirectCoefficient { into: block.target, from: *source }, *value));
        }
    }
    let close_to = |into: u32, from: u32, truth: f64| {
        let got = effect(&coefficients, into, from);
        // 2,000 rows per regime: observed errors are at most 0.039, so 0.06 still rejects a
        // swapped or mis-scaled coefficient (the smallest nonzero truth is 0.25).
        assert!((got - truth).abs() < 0.06, "K[{into},{from}] = {got}, truth {truth}");
    };
    close_to(1, 0, 0.5);
    close_to(2, 0, 0.25);
    close_to(2, 1, 0.4);
    close_to(0, 1, 0.0);
    close_to(0, 2, 0.0);
    close_to(1, 2, 0.0);

    let perturb = |scale: f64, marked: bool, weight: f64| {
        let shifted = coefficients
            .iter()
            .map(|(coefficient, value)| {
                let delta = if coefficient.into == node(2) && coefficient.from == node(0) {
                    scale
                } else {
                    0.0
                };
                (*coefficient, value + delta)
            })
            .collect();
        RecoveryDrawInput {
            coefficients: shifted,
            intercept: vec![0.0, 0.0, 0.0],
            noise_mean: vec![0.0, 0.0, 0.0],
            noise_covariance: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            unresolved: Vec::new(),
            weight,
            marked_inadmissible: marked,
        }
    };
    let nodes = three();
    let uncertainty = RecoveredCoefficientUncertainty::retain(
        &nodes,
        vec![
            perturb(0.0, false, 0.3),
            perturb(0.05, false, 0.3),
            perturb(-0.05, false, 0.3),
            perturb(20.0, true, 0.1),
        ],
    )
    .expect("coefficient law");
    let held = fixed_joint_holdout(row_count);
    let held_mean = held[2].iter().sum::<f64>() / f64::from(rows);
    let predict = |mechanism: &[(DirectCoefficient, f64)]| {
        effect(mechanism, 2, 0) * 2.0 + effect(mechanism, 2, 1) * 1.25
    };
    let admissible: Vec<_> =
        uncertainty.draws().iter().filter_map(as_recovered_mechanism).collect();
    assert_eq!(admissible.len(), 3);
    assert!(uncertainty.draws().iter().any(|draw| as_recovered_mechanism(draw).is_none()));
    let center = admissible.iter().map(|mechanism| predict(mechanism)).sum::<f64>() / 3.0;
    let spread =
        admissible.iter().map(|mechanism| (predict(mechanism) - center).abs()).fold(0.0, f64::max);
    assert!(spread > 0.05, "held-out check must use the coefficient draws");
    // The held-out regime fixes x0 = 2, x1 = 1.25, so the true mean of x2 is 0.25 * 2 + 0.4 * 1.25.
    assert!((center - 1.0).abs() < 0.06, "draw center {center} vs true 1.0");
    assert!(
        (held_mean - center).abs() < 0.05,
        "held-out joint regime mean {held_mean} vs draw center {center}"
    );
}

#[test]
fn an_unclamped_regime_and_a_deficient_design_are_not_recovered() {
    let nodes = three();
    let only_zero = vec![single_clamp(0)];
    let decision =
        decide_experimental_recovery(&clamp_design(nodes, only_zero.clone(), Vec::new()));
    assert!(matches!(decision, ExperimentalRecoveryDecision::Partial(_)));
    assert!(!decision.coefficient_recovery().is_some_and(|recovery| recovery.generic));
    let never = single_clamp(1);
    assert!(!regime_is_identified(&decision, &only_zero, &never));
    assert!(!regime_is_identified(&decision, &only_zero, &only_zero[0]));
}
