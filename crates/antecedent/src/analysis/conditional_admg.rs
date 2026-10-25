//! Compile arm for a conditional effect on a static ADMG.
//!
//! The effect is the mean contrast of two conditional interventional laws,
//! `E[Y | do(active), Z] − E[Y | do(control), Z]`, identified by IDC at the two
//! hard-set levels. The modifiers are free variables of the contrast: the
//! evaluator reports the contrast averaged over the observed modifier law
//! (`Σ_z P(z) τ(z)` over the supported strata), the same quantity the
//! Dag/Cpdag/Pag conditional routes report. The frequentist object is the
//! plug-in of that contrast. The Bayesian object is the Dirichlet row-law
//! posterior of the same contrast. A modifier that is a descendant of the
//! treatment is refused: its observed law is not the law it has under the
//! intervention. A conditional effect on a temporal ADMG read from tabular rows is the effect on
//! its contemporaneous projection; the temporal conditional contrast is a `TemporalEffect` with
//! modifiers, identified by IDC on the unrolled window (`analysis::temporal_admg_contrast`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "the compiled objects carry typed ids that the routes read selectively; tests read all of them"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, CausalQuery, ConditionalEffectQuery, IdentificationStatus, Intervention,
    OutcomeFunctional, TargetPopulation, VariableId,
};
use antecedent_expr::{
    CausalExprArena, ContrastOp, DerivationMeta, EstimandMethod, ExprNode, IdentifiedEstimand,
    OutcomeExprId,
};
use antecedent_graph::{Admg, TemporalAdmg};
use antecedent_identify::{
    DerivationTrace, IdcIdentifier, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationWorkspace, PreparedAdmg,
};

use crate::{CausalError, EstimatorId, IdentifierId};

/// IDC contrast of the active and control conditional laws.
#[derive(Clone, Debug)]
pub(crate) struct IdentifiedAdmgConditional {
    /// Conditional effect the two laws realise.
    pub(crate) query: ConditionalEffectQuery,
    /// Mean contrast, method `general.id`.
    pub(crate) estimand: IdentifiedEstimand,
    /// Arena that owns [`Self::estimand`]'s functional.
    pub(crate) arena: CausalExprArena,
    /// Union of the assumptions the two IDC calls recorded.
    pub(crate) assumptions: AssumptionSet,
    /// Both laws were nonparametrically identified.
    pub(crate) status: IdentificationStatus,
}

/// Frequentist compile: plug-in evaluation of the identified contrast.
#[derive(Clone, Debug)]
#[must_use = "a compiled conditional effect binds an estimator"]
pub(crate) struct FrequentistAdmgConditional {
    /// Shared IDC contrast.
    pub(crate) identified: IdentifiedAdmgConditional,
}

impl FrequentistAdmgConditional {
    /// Identifier of the IDC laws.
    pub(crate) const IDENTIFIER: IdentifierId = IdentifierId::GeneralId;
    /// Discrete plug-in of the contrast. No posterior is attached.
    pub(crate) const ESTIMATOR: EstimatorId = EstimatorId::FunctionalEffect;

    /// `general.id`.
    #[allow(clippy::unused_self, reason = "the compiled object is the binding; the id is fixed")]
    #[must_use]
    pub(crate) const fn identifier(&self) -> IdentifierId {
        Self::IDENTIFIER
    }

    /// `functional.effect`.
    #[allow(
        clippy::unused_self,
        reason = "the compiled object is the binding; the estimator is fixed"
    )]
    #[must_use]
    pub(crate) const fn estimator(&self) -> EstimatorId {
        Self::ESTIMATOR
    }
}

/// Dirichlet(1, …, 1) posterior of one identified contrast.
///
/// Draws reweight the empirical rows once and push the same weights through
/// every factor of the contrast. Unobserved joint cells keep zero mass. This
/// is the functional posterior, not `conditional.bayesian`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirichletRowPosterior;

impl DirichletRowPosterior {
    /// Procedure id of [`antecedent_estimate::FunctionalEffect::estimate_bayesian`].
    pub(crate) const PROCEDURE: &'static str = "functional.dirichlet_row_posterior";

    /// Stable procedure id.
    #[allow(
        clippy::unused_self,
        reason = "the posterior type is the binding; the procedure id is fixed"
    )]
    #[must_use]
    pub(crate) const fn procedure(self) -> &'static str {
        Self::PROCEDURE
    }
}

/// Bayesian compile: Dirichlet row-law posterior of the identified contrast.
#[derive(Clone, Debug)]
#[must_use = "a compiled conditional posterior binds a procedure"]
pub(crate) struct BayesianAdmgConditional {
    /// Shared IDC contrast.
    pub(crate) identified: IdentifiedAdmgConditional,
    /// Posterior procedure. Absent from the frequentist object.
    pub(crate) posterior: DirichletRowPosterior,
}

impl BayesianAdmgConditional {
    /// Identifier of the IDC laws. The posterior is not a second identifier.
    pub(crate) const IDENTIFIER: IdentifierId = IdentifierId::GeneralId;

    /// `general.id`.
    #[allow(clippy::unused_self, reason = "the compiled object is the binding; the id is fixed")]
    #[must_use]
    pub(crate) const fn identifier(&self) -> IdentifierId {
        Self::IDENTIFIER
    }

    /// Dirichlet procedure id.
    #[must_use]
    pub(crate) const fn procedure(&self) -> &'static str {
        self.posterior.procedure()
    }
}

/// The IDC contrast of a conditional effect as the identification result the prepared and
/// the one-shot routes read. The query on the result is the conditional effect itself.
///
/// # Errors
///
/// The query is not a conditional mean over the observed population, a modifier is repeated or
/// is not pre-treatment, or IDC does not identify either conditional law.
pub(crate) fn admg_conditional_identification(
    graph: &Admg,
    query: &ConditionalEffectQuery,
) -> Result<IdentificationResult, CausalError> {
    let identified = identify_admg_conditional(graph, query)?;
    let mut derivation = DerivationTrace::default();
    derivation.push(
        "idc.conditional_contrast",
        "difference of the active and control conditional interventional laws, each identified \
         by IDC at its hard-set level; the modifiers stay free variables averaged over their \
         observed law",
    );
    Ok(IdentificationResult::identified(
        CausalQuery::ConditionalEffect(identified.query),
        vec![identified.estimand],
        identified.arena,
        derivation,
        identified.assumptions,
        IdentificationPerformanceRecord { candidates_examined: 2, sets_returned: 1 },
    ))
}

/// Compile the frequentist plug-in of a conditional effect on `graph`.
///
/// # Errors
///
/// The query is not a conditional mean, a modifier is repeated, or IDC does
/// not identify either conditional law.
pub(crate) fn compile_frequentist_admg_conditional(
    graph: &Admg,
    query: &ConditionalEffectQuery,
) -> Result<FrequentistAdmgConditional, CausalError> {
    Ok(FrequentistAdmgConditional { identified: identify_admg_conditional(graph, query)? })
}

/// Compile the Dirichlet posterior of a conditional effect on `graph`.
///
/// # Errors
///
/// Same as [`compile_frequentist_admg_conditional`]. A prior does not identify
/// a law IDC refused.
pub(crate) fn compile_bayesian_admg_conditional(
    graph: &Admg,
    query: &ConditionalEffectQuery,
) -> Result<BayesianAdmgConditional, CausalError> {
    Ok(BayesianAdmgConditional {
        identified: identify_admg_conditional(graph, query)?,
        posterior: DirichletRowPosterior,
    })
}

/// Compile the plug-in of a conditional effect on the contemporaneous projection of a
/// temporal ADMG.
///
/// Tabular rows carry no time index, so the static conditional query on a temporal ADMG is the
/// conditional contrast of the projection (each variable once, at lag 0, with the template's
/// directed and bidirected edges), the same twin the licensed `AverageEffect` has. It is not
/// the temporal conditional contrast: that one is `TemporalConditionalPulse` or
/// `TemporalConditionalSustained` on one series, identified by IDC on the unrolled window.
///
/// # Errors
///
/// A template whose variables repeat across lags (no contemporaneous projection exists), or
/// anything [`compile_frequentist_admg_conditional`] refuses on the projection.
pub(crate) fn compile_temporal_admg_conditional_frequentist(
    graph: &TemporalAdmg,
    query: &ConditionalEffectQuery,
) -> Result<FrequentistAdmgConditional, CausalError> {
    let projected = graph.project_preserving_functional().map_err(CausalError::rejected)?;
    compile_frequentist_admg_conditional(&projected, query)
}

/// Compile the Dirichlet posterior of a conditional effect on the contemporaneous projection of
/// a temporal ADMG.
///
/// # Errors
///
/// As [`compile_temporal_admg_conditional_frequentist`].
pub(crate) fn compile_temporal_admg_conditional_bayesian(
    graph: &TemporalAdmg,
    query: &ConditionalEffectQuery,
) -> Result<BayesianAdmgConditional, CausalError> {
    let projected = graph.project_preserving_functional().map_err(CausalError::rejected)?;
    compile_bayesian_admg_conditional(&projected, query)
}

fn identify_admg_conditional(
    graph: &Admg,
    query: &ConditionalEffectQuery,
) -> Result<IdentifiedAdmgConditional, CausalError> {
    query.validate()?;
    if !matches!(query.inner.outcome_functional, OutcomeFunctional::Mean) {
        return Err(CausalError::Compile {
            message: "the ADMG conditional compile arm is the mean contrast of the two \
                      conditional interventional laws"
                .into(),
        });
    }
    if !matches!(query.inner.target_population, TargetPopulation::AllObserved) {
        return Err(CausalError::Compile {
            message: "the ADMG conditional compile arm answers for the observed population; \
                      a reweighted or local target population is a different estimand"
                .into(),
        });
    }
    let modifiers = unique_modifiers(query)?;
    require_pre_treatment_modifiers(graph, query, &modifiers)?;
    let idc = IdcIdentifier::new();
    let prepared = idc.prepare(graph)?;
    let mut workspace = IdentificationWorkspace::default();
    let active = conditional_law(
        &idc,
        &prepared,
        query,
        &modifiers,
        &query.inner.active,
        "active",
        &mut workspace,
    )?;
    let control = conditional_law(
        &idc,
        &prepared,
        query,
        &modifiers,
        &query.inner.control,
        "control",
        &mut workspace,
    )?;
    let (arena, estimand, assumptions) = assemble_contrast(query.inner.outcome, &active, &control)?;
    Ok(IdentifiedAdmgConditional {
        query: query.clone(),
        estimand,
        arena,
        assumptions,
        status: IdentificationStatus::NonparametricallyIdentified,
    })
}

fn unique_modifiers(query: &ConditionalEffectQuery) -> Result<Vec<VariableId>, CausalError> {
    let mut modifiers: Vec<VariableId> = query.inner.effect_modifiers.iter().copied().collect();
    modifiers.sort_unstable();
    if modifiers.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(CausalError::Compile {
            message: "effect modifiers on an ADMG conditional effect must be unique".into(),
        });
    }
    Ok(modifiers)
}

/// The modifiers are averaged over their observed law, which is the law they have under
/// the intervention only when the treatment cannot change them.
fn require_pre_treatment_modifiers(
    graph: &Admg,
    query: &ConditionalEffectQuery,
    modifiers: &[VariableId],
) -> Result<(), CausalError> {
    let treatment = graph.dense_of_static(query.inner.treatment);
    for &modifier in modifiers {
        let node = graph.dense_of_static(modifier);
        let (Some(treatment), Some(node)) = (treatment, node) else {
            return Err(CausalError::Compile {
                message: "an effect modifier or the treatment is not a variable of the ADMG".into(),
            });
        };
        if modifier == query.inner.treatment
            || modifier == query.inner.outcome
            || graph.reaches(treatment, node)
        {
            return Err(CausalError::Compile {
                message:
                    "an effect modifier of an ADMG conditional effect must be a pre-treatment \
                          variable (not the treatment, the outcome, or a descendant of the \
                          treatment). Condition on a post-treatment variable with an \
                          InterventionalDistribution query that names the conditioning values"
                        .into(),
            });
        }
    }
    Ok(())
}

fn conditional_law(
    idc: &IdcIdentifier,
    prepared: &PreparedAdmg,
    query: &ConditionalEffectQuery,
    modifiers: &[VariableId],
    intervention: &Intervention,
    arm: &str,
    workspace: &mut IdentificationWorkspace,
) -> Result<IdentificationResult, CausalError> {
    let identified = idc.identify_conditional_at_levels(
        prepared,
        &[query.inner.outcome],
        std::slice::from_ref(intervention),
        modifiers,
        workspace,
    )?;
    require_one_law(&identified, arm)?;
    Ok(identified)
}

fn require_one_law(result: &IdentificationResult, arm: &str) -> Result<(), CausalError> {
    if result.status == IdentificationStatus::NonparametricallyIdentified
        && result.estimands.len() == 1
    {
        return Ok(());
    }
    let search_capped = result
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code.as_ref().ends_with(".search_bounded"));
    Err(CausalError::not_identified(
        result.status,
        search_capped,
        &format!("IDC did not identify the {arm} conditional law on the ADMG"),
    ))
}

fn assemble_contrast(
    outcome: VariableId,
    active: &IdentificationResult,
    control: &IdentificationResult,
) -> Result<(CausalExprArena, IdentifiedEstimand, AssumptionSet), CausalError> {
    let active_law = &active.estimands[0];
    let control_law = &control.estimands[0];
    let mut arena = CausalExprArena::new();
    let active_root = arena.import(&active.arena, active_law.functional);
    let control_root = arena.import(&control.arena, control_law.functional);
    let left = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(outcome),
        distribution: active_root,
    });
    let right = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(outcome),
        distribution: control_root,
    });
    let contrast = arena.intern(ExprNode::Contrast { left, right, op: ContrastOp::Difference });
    let functional = arena.simplify(contrast).map_err(|error| CausalError::Compile {
        message: format!("the conditional contrast on the ADMG is ill-formed: {error}"),
    })?;
    arena.set_derivation(
        functional,
        DerivationMeta::rule(
            "idc.conditional_contrast",
            Some(Arc::from("difference of the active and control conditional interventional laws")),
        ),
    );
    let estimand = IdentifiedEstimand::new(
        EstimandMethod::GeneralId.as_str(),
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut assumptions = active.required_assumptions.clone();
    assumptions.extend_unique(&control.required_assumptions.entries);
    Ok((arena, estimand, assumptions))
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        Assumption, AverageEffectQuery, ConditionalEffectQuery, OutcomeFunctional, VariableId,
    };
    use antecedent_expr::{ContrastOp, ExprNode};
    use antecedent_graph::{Admg, DenseNodeId, TemporalAdmg};

    use crate::{CausalError, EstimatorId, IdentifierId};

    use super::{
        BayesianAdmgConditional, DirichletRowPosterior, FrequentistAdmgConditional,
        admg_conditional_identification, compile_bayesian_admg_conditional,
        compile_frequentist_admg_conditional, compile_temporal_admg_conditional_bayesian,
        compile_temporal_admg_conditional_frequentist,
    };

    fn latent_parent_chain() -> (Admg, ConditionalEffectQuery) {
        let mut graph = Admg::with_variables(3);
        let modifier = DenseNodeId::from_raw(0);
        let treatment = DenseNodeId::from_raw(1);
        let outcome = DenseNodeId::from_raw(2);
        graph.insert_directed(modifier, treatment).unwrap();
        graph.insert_directed(modifier, outcome).unwrap();
        graph.insert_directed(treatment, outcome).unwrap();
        graph.insert_bidirected(modifier, treatment).unwrap();
        let query = ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2))
                .with_effect_modifiers([VariableId::from_raw(0)]),
        )
        .unwrap();
        (graph, query)
    }

    fn bow_with_modifier() -> (Admg, ConditionalEffectQuery) {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let query = ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_effect_modifiers([VariableId::from_raw(2)]),
        )
        .unwrap();
        (graph, query)
    }

    #[test]
    fn conditional_effect_on_a_latent_admg_compiles_a_mean_contrast() {
        let (graph, query) = latent_parent_chain();
        assert!(!graph.bidirected_edge_pairs_sorted().is_empty());
        let frequentist = compile_frequentist_admg_conditional(&graph, &query).unwrap();
        let bayesian = compile_bayesian_admg_conditional(&graph, &query).unwrap();
        assert_eq!(frequentist.identifier(), IdentifierId::GeneralId);
        assert_eq!(bayesian.identifier(), IdentifierId::GeneralId);
        assert_eq!(frequentist.estimator(), EstimatorId::FunctionalEffect);
        assert_ne!(frequentist.estimator().as_str(), bayesian.procedure());
        assert_eq!(bayesian.procedure(), DirichletRowPosterior::PROCEDURE);
        assert_ne!(bayesian.procedure(), EstimatorId::BayesianConditional.as_str());
        assert_ne!(bayesian.procedure(), EstimatorId::ConditionalLinearAdjustment.as_str());
        assert!(!frequentist.identified.estimand.is_adjustment_shaped());
        assert!(
            frequentist
                .identified
                .assumptions
                .entries
                .iter()
                .any(|record| { matches!(record.assumption, Assumption::CausalMarkov) })
        );
        assert_eq!(
            frequentist.identified.estimand.functional,
            bayesian.identified.estimand.functional
        );
        assert_eq!(frequentist.identified.status, bayesian.identified.status);
    }

    #[test]
    fn contrast_is_the_difference_of_two_conditional_laws() {
        let (graph, query) = latent_parent_chain();
        let compiled = compile_frequentist_admg_conditional(&graph, &query).unwrap();
        let arena = &compiled.identified.arena;
        let functional = compiled.identified.estimand.functional;
        let ExprNode::Contrast { left, right, op: ContrastOp::Difference } = arena.node(functional)
        else {
            panic!("the conditional effect is a difference of two laws");
        };
        assert!(matches!(arena.node(*left), ExprNode::Expectation { .. }));
        assert!(matches!(arena.node(*right), ExprNode::Expectation { .. }));
        let derivation = arena.derivation(functional).unwrap();
        assert_eq!(derivation.rule.as_ref(), "idc.conditional_contrast");
    }

    #[test]
    fn a_prior_does_not_identify_the_bow() {
        let (graph, query) = bow_with_modifier();
        let frequentist = compile_frequentist_admg_conditional(&graph, &query).unwrap_err();
        let bayesian = compile_bayesian_admg_conditional(&graph, &query).unwrap_err();
        assert!(frequentist.to_string().contains("effect_not_identified"), "{frequentist}");
        assert!(bayesian.to_string().contains("effect_not_identified"), "{bayesian}");
    }

    #[test]
    fn a_post_treatment_modifier_is_refused_with_its_next_attempt() {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let query = ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(2))
                .with_effect_modifiers([VariableId::from_raw(1)]),
        )
        .unwrap();
        let refusal = compile_frequentist_admg_conditional(&graph, &query).unwrap_err();
        let text = refusal.to_string();
        assert!(text.contains("pre-treatment"), "{text}");
        assert!(text.contains("InterventionalDistribution"), "{text}");
        assert!(compile_bayesian_admg_conditional(&graph, &query).is_err());
    }

    #[test]
    fn an_identification_result_carries_the_conditional_query_and_one_estimand() {
        let (graph, query) = latent_parent_chain();
        let result = admg_conditional_identification(&graph, &query).unwrap();
        assert_eq!(result.query, antecedent_core::CausalQuery::ConditionalEffect(query));
        assert_eq!(
            result.status,
            antecedent_core::IdentificationStatus::NonparametricallyIdentified
        );
        assert_eq!(result.estimands.len(), 1);
        assert_eq!(
            result.estimands[0].method_kind(),
            Ok(antecedent_expr::EstimandMethod::GeneralId)
        );
        // The two arms are two functionals: the baked levels differ.
        let ExprNode::Contrast { left, right, .. } =
            result.arena.node(result.estimands[0].functional)
        else {
            panic!("a conditional contrast");
        };
        assert_ne!(left, right);
    }

    #[test]
    fn repeated_modifiers_and_non_mean_functionals_are_refused() {
        let (graph, query) = latent_parent_chain();
        let repeated = ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2))
                .with_effect_modifiers([VariableId::from_raw(0), VariableId::from_raw(0)]),
        )
        .unwrap();
        let repeated = compile_frequentist_admg_conditional(&graph, &repeated).unwrap_err();
        assert!(repeated.to_string().contains("unique"), "{repeated}");
        let exceedance = ConditionalEffectQuery::try_new(
            query.inner.clone().with_outcome_functional(OutcomeFunctional::exceedance(0.5)),
        )
        .unwrap();
        let exceedance = compile_bayesian_admg_conditional(&graph, &exceedance).unwrap_err();
        assert!(exceedance.to_string().contains("mean contrast"), "{exceedance}");
    }

    #[test]
    fn temporal_admg_conditional_is_the_contemporaneous_projection_conditional() {
        let (graph, query) = latent_parent_chain();
        let temporal = TemporalAdmg::from_static_contemporaneous(&graph).unwrap();
        let frequentist = compile_temporal_admg_conditional_frequentist(&temporal, &query).unwrap();
        let bayesian = compile_temporal_admg_conditional_bayesian(&temporal, &query).unwrap();
        let direct = compile_frequentist_admg_conditional(&graph, &query).unwrap();
        // The projection has the same edges, so IDC derives the same identified contrast.
        assert_eq!(frequentist.estimator().as_str(), "functional.effect");
        assert_eq!(frequentist.identified.status, direct.identified.status);
        assert_eq!(bayesian.identified.status, direct.identified.status);
        assert_eq!(
            frequentist.identified.estimand.method,
            direct.identified.estimand.method
        );
        assert_eq!(bayesian.posterior.procedure(), "functional.dirichlet_row_posterior");
        // A template whose variable repeats across lags has no contemporaneous projection.
        let mut repeated = TemporalAdmg::empty();
        let a = repeated
            .add_lagged(antecedent_core::VariableId::from_raw(0), antecedent_core::Lag::from_raw(1))
            .unwrap();
        let b = repeated
            .add_lagged(
                antecedent_core::VariableId::from_raw(0),
                antecedent_core::Lag::CONTEMPORANEOUS,
            )
            .unwrap();
        repeated.insert_directed(a, b).unwrap();
        assert!(compile_temporal_admg_conditional_frequentist(&repeated, &query).is_err());
    }

    #[test]
    fn frequentist_and_bayesian_compiles_are_different_types() {
        fn plug_in(arm: &FrequentistAdmgConditional) -> &'static str {
            arm.estimator().as_str()
        }
        fn posterior(arm: &BayesianAdmgConditional) -> &'static str {
            arm.procedure()
        }
        let (graph, query) = latent_parent_chain();
        let frequentist = compile_frequentist_admg_conditional(&graph, &query).unwrap();
        let bayesian = compile_bayesian_admg_conditional(&graph, &query).unwrap();
        assert_eq!(plug_in(&frequentist), "functional.effect");
        assert_eq!(posterior(&bayesian), "functional.dirichlet_row_posterior");
    }
}
