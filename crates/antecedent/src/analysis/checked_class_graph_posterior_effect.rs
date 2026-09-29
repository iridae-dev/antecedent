//! Checked effect operation for CPDAG/PAG graph-posterior atoms.
//!
//! Frequentist and Bayesian average and conditional effects retain the same
//! frozen samples and completion envelopes; only the per-completion procedure
//! differs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AverageEffectQuery, PopulationRegistry};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_estimate::OverlapPolicy;
use antecedent_prob::GraphIdentFlag;

use crate::{CausalError, EstimatorSpec, InferenceMode, RefuteSuite};

use super::GraphPosteriorEffectTarget;

use super::latency::LatencyMode;
use super::prepared::CachedGraphPosteriorIdentification;

/// A fixed effect procedure over posterior-weighted CPDAG/PAG atoms. Each
/// outer posterior sample, including unidentified samples, remains in
/// `identification.graphs`; each identified atom retains its full completion
/// envelope and failed completion cases in `class_atoms`. The target records
/// whether the sealed click estimates the average or a conditional effect.
#[derive(Clone, Debug)]
pub(crate) struct CheckedClassGraphPosteriorEffect {
    posterior: Arc<GraphPosterior>,
    target: GraphPosteriorEffectTarget,
    identification: Arc<CachedGraphPosteriorIdentification>,
    inference: InferenceMode,
    procedure: EstimatorSpec,
    bootstrap_replicates: u32,
    overlap: OverlapPolicy,
    population_registry: Option<PopulationRegistry>,
    latency_mode: Option<LatencyMode>,
    validation: RefuteSuite,
}

impl CheckedClassGraphPosteriorEffect {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        posterior: GraphPosterior,
        target: GraphPosteriorEffectTarget,
        identification: CachedGraphPosteriorIdentification,
        inference: InferenceMode,
        procedure: EstimatorSpec,
        bootstrap_replicates: u32,
        overlap: OverlapPolicy,
        population_registry: Option<PopulationRegistry>,
        latency_mode: Option<LatencyMode>,
        validation: RefuteSuite,
    ) -> Result<Self, CausalError> {
        if !matches!(
            posterior.atom_kind,
            GraphPosteriorAtomKind::Cpdag | GraphPosteriorAtomKind::Pag
        ) {
            return Err(CausalError::Unsupported {
                message: "checked class graph-posterior effect requires CPDAG or PAG atoms",
            });
        }
        let licensed = match inference {
            InferenceMode::Frequentist => target.frequentist_estimator(),
            InferenceMode::Bayesian(_) => target.bayesian_estimator(),
        };
        if procedure.id() != licensed {
            return Err(CausalError::Unsupported {
                message: "checked class graph-posterior effect procedure does not match its inference and query kind",
            });
        }
        let query = target.inner();
        if query.treatment == query.outcome
            || usize::try_from(query.treatment.raw()).map_or(true, |i| i >= posterior.n_vars)
            || usize::try_from(query.outcome.raw()).map_or(true, |i| i >= posterior.n_vars)
        {
            return Err(CausalError::Unsupported {
                message: "class graph-posterior query variables must be distinct and in range",
            });
        }
        if !matches!(
            validation,
            RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full
        ) {
            return Err(CausalError::Unsupported {
                message: "class graph-posterior effect has an unsupported validation suite",
            });
        }
        validate_identification_binding(&posterior, &target, &identification)?;
        Ok(Self {
            posterior: Arc::new(posterior),
            target,
            identification: Arc::new(identification),
            inference,
            procedure,
            bootstrap_replicates,
            overlap,
            population_registry,
            latency_mode,
            validation,
        })
    }

    #[must_use]
    pub(crate) fn posterior(&self) -> &GraphPosterior {
        &self.posterior
    }

    /// The average-effect query every posterior atom was identified for.
    #[must_use]
    pub(crate) const fn query(&self) -> &AverageEffectQuery {
        self.target.inner()
    }

    /// The sealed query kind, including conditional modifiers.
    #[must_use]
    pub(crate) const fn target(&self) -> &GraphPosteriorEffectTarget {
        &self.target
    }

    #[must_use]
    pub(crate) const fn inference(&self) -> &InferenceMode {
        &self.inference
    }

    #[must_use]
    pub(crate) fn identification(&self) -> &CachedGraphPosteriorIdentification {
        &self.identification
    }

    #[must_use]
    pub(crate) fn procedure(&self) -> &EstimatorSpec {
        &self.procedure
    }

    #[must_use]
    pub(crate) const fn bootstrap_replicates(&self) -> u32 {
        self.bootstrap_replicates
    }

    #[must_use]
    pub(crate) const fn overlap(&self) -> OverlapPolicy {
        self.overlap
    }

    #[must_use]
    pub(crate) fn population_registry(&self) -> Option<&PopulationRegistry> {
        self.population_registry.as_ref()
    }

    #[must_use]
    pub(crate) const fn latency_mode(&self) -> Option<LatencyMode> {
        self.latency_mode
    }

    #[must_use]
    pub(crate) const fn validation(&self) -> RefuteSuite {
        self.validation
    }

    /// Posterior sample keys, weights, and identification flags retain the
    /// sample-level mass, including failed/unidentified graph atoms.
    #[must_use]
    pub(crate) fn sample_mass(&self) -> (&[u64], &[f64], &[GraphIdentFlag]) {
        (
            &self.posterior.graph_keys,
            &self.posterior.weights,
            &self.identification.graphs.identified,
        )
    }

    /// Per-key completion envelopes retain identified, failed, and truncated
    /// completions for each identified graph-class posterior atom.
    #[must_use]
    pub(crate) fn class_atoms(
        &self,
    ) -> &[crate::analysis::prepared::CachedClassPosteriorAtomIdentification] {
        &self.identification.class_atoms
    }
}

fn validate_identification_binding(
    posterior: &GraphPosterior,
    target: &GraphPosteriorEffectTarget,
    identification: &CachedGraphPosteriorIdentification,
) -> Result<(), CausalError> {
    if identification.graphs.weights.as_ref() != posterior.weights.as_ref()
        || identification.graphs.graph_keys.as_ref() != posterior.graph_keys.as_ref()
        || identification.graphs.n_samples != posterior.n_graphs
        || identification.graphs.identified.len() != posterior.n_graphs
        || !identification.atoms.is_empty()
    {
        return Err(CausalError::Unsupported {
            message: "class graph-posterior cache does not match frozen samples or contains DAG atoms",
        });
    }
    let atom_query = target.atom_identification_query();
    if identification.class_atoms.iter().any(|atom| atom.identification.query != atom_query) {
        return Err(CausalError::Unsupported {
            message: "class graph-posterior cache identifies a different ATE query",
        });
    }
    let mut class_keys = std::collections::HashSet::with_capacity(identification.class_atoms.len());
    for atom in identification.class_atoms.iter() {
        if !posterior.graph_keys.iter().any(|key| *key == atom.key)
            || !class_keys.insert(atom.key)
            || !atom.identified_weight.is_finite()
            || atom.identified_weight <= 0.0
            || atom.cases.is_empty()
        {
            return Err(CausalError::Unsupported {
                message: "class graph-posterior cache has an invalid completion atom binding",
            });
        }
        let completion_mass = atom.cases.iter().map(|case| case.weight).sum::<f64>();
        let identified_completion_mass = atom
            .cases
            .iter()
            .filter(|case| identified_case_status(case.status))
            .map(|case| case.weight)
            .sum::<f64>();
        if atom.cases.iter().any(|case| !case.weight.is_finite() || case.weight < 0.0)
            || !completion_mass.is_finite()
            || completion_mass <= 0.0
            || atom.identified_weight > completion_mass + 1e-10
            || (identified_completion_mass - atom.identified_weight).abs() > 1e-10
        {
            return Err(CausalError::Unsupported {
                message: "class graph-posterior completion or identified masses are invalid",
            });
        }
        if !posterior
            .graph_keys
            .iter()
            .zip(identification.graphs.identified.iter())
            .any(|(key, flag)| *key == atom.key && matches!(flag, GraphIdentFlag::Identified))
        {
            return Err(CausalError::Unsupported {
                message: "retained class envelope has no identified posterior sample mass",
            });
        }
    }
    for (key, flag) in posterior.graph_keys.iter().zip(identification.graphs.identified.iter()) {
        if matches!(flag, GraphIdentFlag::Identified) && !class_keys.contains(key) {
            return Err(CausalError::Unsupported {
                message: "identified posterior sample has no retained class completion envelope",
            });
        }
    }
    Ok(())
}

fn identified_case_status(status: antecedent_core::IdentificationStatus) -> bool {
    matches!(
        status,
        antecedent_core::IdentificationStatus::NonparametricallyIdentified
            | antecedent_core::IdentificationStatus::PartiallyIdentified
            | antecedent_core::IdentificationStatus::IdentifiedUnderParametricRestrictions
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EstimatorId;
    use antecedent_core::VariableId;
    use antecedent_prob::{InferenceDiagnostics, WeightedGraphSamples};

    fn posterior(kind: GraphPosteriorAtomKind) -> GraphPosterior {
        GraphPosterior::new(
            2,
            vec![0.4, 0.6],
            vec![0, 0],
            vec![0.0; 4],
            vec![0.0; 4],
            1.923_076_923_076_923_1,
            InferenceDiagnostics::analytic("checked_class_posterior_test"),
            0,
        )
        .unwrap()
        .with_atom_kind(kind)
    }

    fn query() -> AverageEffectQuery {
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
    }

    fn unidentified_cache(graphs: &GraphPosterior) -> CachedGraphPosteriorIdentification {
        CachedGraphPosteriorIdentification {
            graphs: WeightedGraphSamples::new(
                graphs.weights.to_vec(),
                vec![GraphIdentFlag::Unidentified; graphs.n_graphs],
                graphs.graph_keys.to_vec(),
            )
            .unwrap(),
            atoms: Arc::from([]),
            class_atoms: Arc::from([]),
        }
    }

    fn identified_cpdag_cache(graphs: &GraphPosterior) -> CachedGraphPosteriorIdentification {
        let target = query();
        let envelope = crate::strategy_table::identify_cpdag(
            crate::strategy_table::IdentifierId::GeneralizedAdjustment,
            &antecedent_graph::Cpdag::with_variables(2),
            &target,
        )
        .expect("empty CPDAG ATE identification fixture");
        let atom = crate::analysis::prepared::CachedClassPosteriorAtomIdentification {
            key: graphs.graph_keys[0],
            identification: envelope.cases[0].result.clone(),
            invariant: envelope.invariant.clone(),
            cases: Arc::from(
                envelope
                    .cases
                    .iter()
                    .map(|case| crate::analysis::prepared::CachedClassPosteriorCase {
                        weight: case.weight.0,
                        status: case.result.status,
                        estimand: case.result.estimands.first().cloned(),
                    })
                    .collect::<Vec<_>>(),
            ),
            identified_weight: envelope.identified_weight.0,
            truncated_completions: envelope.truncated_completions,
        };
        CachedGraphPosteriorIdentification {
            graphs: WeightedGraphSamples::new(
                graphs.weights.to_vec(),
                vec![GraphIdentFlag::Identified; graphs.n_graphs],
                graphs.graph_keys.to_vec(),
            )
            .unwrap(),
            atoms: Arc::from([]),
            class_atoms: Arc::from([atom]),
        }
    }

    fn prepare(
        graphs: GraphPosterior,
        cache: CachedGraphPosteriorIdentification,
        procedure: EstimatorSpec,
    ) -> Result<CheckedClassGraphPosteriorEffect, CausalError> {
        prepare_with(graphs, GraphPosteriorEffectTarget::Average(query()), cache, procedure)
    }

    fn prepare_with(
        graphs: GraphPosterior,
        target: GraphPosteriorEffectTarget,
        cache: CachedGraphPosteriorIdentification,
        procedure: EstimatorSpec,
    ) -> Result<CheckedClassGraphPosteriorEffect, CausalError> {
        let inference = match procedure.id() {
            EstimatorId::BayesianGcomp | EstimatorId::BayesianConditional => {
                InferenceMode::Bayesian(crate::BayesianConfig::conjugate())
            }
            _ => InferenceMode::Frequentist,
        };
        CheckedClassGraphPosteriorEffect::prepare(
            graphs,
            target,
            cache,
            inference,
            procedure,
            0,
            OverlapPolicy::ExplicitOverride,
            None,
            Some(LatencyMode::Standard),
            RefuteSuite::None,
        )
    }

    #[test]
    fn binds_cpdag_and_pag_atoms_while_retaining_unidentified_sample_mass() {
        for kind in [GraphPosteriorAtomKind::Cpdag, GraphPosteriorAtomKind::Pag] {
            let graphs = posterior(kind);
            let cache = unidentified_cache(&graphs);
            let operation = prepare(
                graphs.clone(),
                cache,
                EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte),
            )
            .unwrap();
            assert_eq!(operation.query(), &query());
            assert_eq!(operation.sample_mass().0, &[0, 0]);
            assert_eq!(operation.sample_mass().1, &[0.4, 0.6]);
            assert_eq!(operation.sample_mass().2, &[GraphIdentFlag::Unidentified; 2]);
            assert!(operation.class_atoms().is_empty());
        }
    }

    #[test]
    fn retains_the_full_completion_envelope_for_an_identified_class_atom() {
        let graphs = posterior(GraphPosteriorAtomKind::Cpdag);
        let cache = identified_cpdag_cache(&graphs);
        let operation =
            prepare(graphs, cache, EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte))
                .unwrap();
        assert_eq!(operation.class_atoms().len(), 1);
        let atom = &operation.class_atoms()[0];
        assert_eq!(atom.key, 0);
        assert!(!atom.cases.is_empty());
        assert!((atom.cases.iter().map(|case| case.weight).sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(atom.identified_weight > 0.0);
        assert_eq!(operation.sample_mass().2, &[GraphIdentFlag::Identified; 2]);
    }

    #[test]
    fn accepts_count_weighted_completion_envelopes() {
        let graphs = posterior(GraphPosteriorAtomKind::Cpdag);
        let mut cache = identified_cpdag_cache(&graphs);
        let mut atom = cache.class_atoms[0].clone();
        let case = atom.cases[0].clone();
        atom.cases = Arc::from([case.clone(), case]);
        atom.identified_weight = 2.0;
        cache.class_atoms = Arc::from([atom]);

        let operation =
            prepare(graphs, cache, EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte))
                .expect("enumerated completion counts need not be normalized probabilities");
        let completion_mass =
            operation.class_atoms()[0].cases.iter().map(|case| case.weight).sum::<f64>();
        assert!((completion_mass - 2.0).abs() < 1e-12);
        assert!((operation.class_atoms()[0].identified_weight - 2.0).abs() < 1e-12);
    }

    #[test]
    fn retains_conditional_and_bayesian_targets_with_matching_procedures() {
        let graphs = posterior(GraphPosteriorAtomKind::Cpdag);
        let conditional = GraphPosteriorEffectTarget::Conditional(
            antecedent_core::ConditionalEffectQuery::try_new(
                query().with_effect_modifiers([VariableId::from_raw(2)]),
            )
            .unwrap(),
        );
        let operation = prepare_with(
            graphs.clone(),
            conditional.clone(),
            unidentified_cache(&graphs),
            EstimatorSpec::Default(EstimatorId::ConditionalLinearAdjustment),
        )
        .unwrap();
        assert!(
            prepare_with(
                graphs.clone(),
                conditional.clone(),
                identified_cpdag_cache(&graphs),
                EstimatorSpec::Default(EstimatorId::ConditionalLinearAdjustment),
            )
            .is_err(),
            "a cache identified for the modifier-free ATE must not bind a conditional target"
        );
        assert!(operation.target().is_conditional());
        assert_eq!(operation.query(), conditional.inner());
        assert!(matches!(operation.inference(), InferenceMode::Frequentist));

        let bayesian = prepare_with(
            graphs.clone(),
            GraphPosteriorEffectTarget::Average(query()),
            unidentified_cache(&graphs),
            EstimatorSpec::Default(EstimatorId::BayesianGcomp),
        )
        .unwrap();
        assert!(matches!(bayesian.inference(), InferenceMode::Bayesian(_)));
        assert!(
            prepare_with(
                graphs.clone(),
                conditional.clone(),
                unidentified_cache(&graphs),
                EstimatorSpec::Default(EstimatorId::BayesianConditional),
            )
            .is_ok()
        );

        assert!(
            prepare_with(
                graphs.clone(),
                conditional,
                unidentified_cache(&graphs),
                EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte),
            )
            .is_err(),
            "a conditional target must not retain the average-effect procedure"
        );
        assert!(
            CheckedClassGraphPosteriorEffect::prepare(
                graphs.clone(),
                GraphPosteriorEffectTarget::Average(query()),
                unidentified_cache(&graphs),
                InferenceMode::Frequentist,
                EstimatorSpec::Default(EstimatorId::BayesianGcomp),
                0,
                OverlapPolicy::ExplicitOverride,
                None,
                Some(LatencyMode::Standard),
                RefuteSuite::None,
            )
            .is_err(),
            "frequentist inference must not retain a Bayesian procedure"
        );
    }

    #[test]
    fn refuses_wrong_atom_kind_procedure_and_cache_weight_binding() {
        let graphs = posterior(GraphPosteriorAtomKind::Dag);
        assert!(
            prepare(
                graphs.clone(),
                unidentified_cache(&graphs),
                EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte),
            )
            .is_err()
        );

        let graphs = posterior(GraphPosteriorAtomKind::Cpdag);
        assert!(
            prepare(
                graphs.clone(),
                unidentified_cache(&graphs),
                EstimatorSpec::Default(EstimatorId::Aipw),
            )
            .is_err()
        );

        let mut cache = unidentified_cache(&graphs);
        cache.graphs.weights = Arc::from([0.5, 0.5]);
        assert!(
            prepare(graphs, cache, EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte),)
                .is_err()
        );
    }

    #[test]
    fn refuses_an_identified_sample_without_a_retained_completion_envelope() {
        let graphs = posterior(GraphPosteriorAtomKind::Pag);
        let cache = CachedGraphPosteriorIdentification {
            graphs: WeightedGraphSamples::new(
                graphs.weights.to_vec(),
                vec![GraphIdentFlag::Identified, GraphIdentFlag::Unidentified],
                graphs.graph_keys.to_vec(),
            )
            .unwrap(),
            atoms: Arc::from([]),
            class_atoms: Arc::from([]),
        };
        assert!(
            prepare(graphs, cache, EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte),)
                .is_err()
        );
    }
}
