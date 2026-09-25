//! Checked frequentist ATE operation for CPDAG/PAG graph-posterior atoms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AverageEffectQuery, PopulationRegistry};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_estimate::OverlapPolicy;
use antecedent_prob::GraphIdentFlag;

use crate::{CausalError, EstimatorId, EstimatorSpec, RefuteSuite};

use super::latency::LatencyMode;
use super::prepared::CachedGraphPosteriorIdentification;

/// A fixed frequentist effect procedure over posterior-weighted CPDAG/PAG
/// atoms. Each outer posterior sample, including unidentified samples, remains
/// in `identification.graphs`; each identified atom retains its full
/// completion envelope and failed completion cases in `class_atoms`.
#[derive(Clone, Debug)]
pub(crate) struct CheckedClassGraphPosteriorEffect {
    posterior: Arc<GraphPosterior>,
    query: AverageEffectQuery,
    identification: Arc<CachedGraphPosteriorIdentification>,
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
        query: AverageEffectQuery,
        identification: CachedGraphPosteriorIdentification,
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
        if procedure.id() != EstimatorId::LinearAdjustmentAte {
            return Err(CausalError::Unsupported {
                message: "checked class graph-posterior effect requires linear.adjustment.ate",
            });
        }
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
        validate_identification_binding(&posterior, &query, &identification)?;
        Ok(Self {
            posterior: Arc::new(posterior),
            query,
            identification: Arc::new(identification),
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

    #[must_use]
    pub(crate) fn query(&self) -> &AverageEffectQuery {
        &self.query
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
    query: &AverageEffectQuery,
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
    if identification.class_atoms.iter().any(|atom| {
        atom.identification.query != antecedent_core::CausalQuery::AverageEffect(query.clone())
    }) {
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
            || atom.identified_weight > 1.0
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
            || (completion_mass - 1.0).abs() > 1e-10
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
    use antecedent_core::VariableId;
    use antecedent_prob::{InferenceDiagnostics, WeightedGraphSamples};

    fn posterior(kind: GraphPosteriorAtomKind) -> GraphPosterior {
        GraphPosterior::new(
            2,
            vec![0.4, 0.6],
            vec![0, 0],
            vec![0.0; 4],
            vec![0.0; 4],
            1.9230769230769231,
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
        CheckedClassGraphPosteriorEffect::prepare(
            graphs,
            query(),
            cache,
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
        assert_eq!(atom.cases.iter().map(|case| case.weight).sum::<f64>(), 1.0);
        assert!(atom.identified_weight > 0.0);
        assert_eq!(operation.sample_mass().2, &[GraphIdentFlag::Identified; 2]);
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
