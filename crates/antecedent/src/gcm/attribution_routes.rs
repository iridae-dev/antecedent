use super::{
    AnomalyAttributionQuery, AnomalyScores, Arc, CausalError, CausalRng, ChangeAttributionQuery,
    ChangeAttributionResult, CompiledCausalModel, DistributionChangeOptions, ExecutionContext,
    FeatureRelevance, InterventionalDistributionQuery, IntrinsicInfluenceQuery,
    IntrinsicInfluenceResult, MechanismChangeMethod, MechanismChangeQuery, NoiseIndependence,
    PathSpecificEffectQuery, RobustChangeOptions, RootCauseRank, StructureChangeOptions,
    TabularData, TargetPopulation, UnitChangeQuery, UnitChangeResult, ValueBatch, VariableId,
    detect_mechanism_changes, distribution_change, distribution_change_robust, feature_relevance,
    intrinsic_influence, path_decompose, root_cause_rank, sample_do, score_anomalies,
    score_anomalies_with, structure_change, unit_change,
};

/// The anomaly query over `outcomes` with the default reference.
fn anomaly_query(
    outcomes: impl IntoIterator<Item = VariableId>,
    max_units: usize,
) -> AnomalyAttributionQuery {
    AnomalyAttributionQuery::new(outcomes.into_iter().collect::<Arc<[VariableId]>>(), max_units)
}

internal_engine! {
    /// Score anomalies for listed outcome variables.
    ///
    /// # Errors
    ///
    /// Attribution failures.
    #[doc(hidden)]
    pub fn anomaly_attribution(
        model: &CompiledCausalModel,
        data: &TabularData,
        outcomes: impl IntoIterator<Item = VariableId>,
        max_units: usize,
    ) -> Result<Vec<AnomalyScores>, CausalError> {
        let q = anomaly_query(outcomes, max_units);
        score_anomalies(model, data, &q).map_err(CausalError::from)
    }
}

/// Forwards sub-crate attribution entry points through the one error mapping the facade owns:
/// every forwarded function keeps its arguments and returns the sub-crate's value with the
/// failure converted by `CausalError::from`.
macro_rules! forward_mapped {
    ($(
        $(#[$meta:meta])*
        $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty => $target:path;
    )+) => {
        $(
            internal_engine! {
                $(#[$meta])*
                #[doc(hidden)]
                pub fn $name($($arg: $ty),*) -> Result<$ret, CausalError> {
                    $target($($arg),*).map_err(CausalError::from)
                }
            }
        )+
    };
}

forward_mapped! {
    /// [`anomaly_attribution`] against a full [`AnomalyAttributionQuery`] under an execution
    /// context: rows are scored in parallel up to its thread budget, a cancelled context stops
    /// the run, and the query's reference (`Empirical` / `Fixed`) is honored rather than
    /// reconstructed.
    ///
    /// # Errors
    ///
    /// Attribution failures.
    anomaly_attribution_query_with(
        model: &CompiledCausalModel,
        data: &TabularData,
        query: &AnomalyAttributionQuery,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Vec<AnomalyScores> => score_anomalies_with;

    /// Intrinsic causal influence: the Shapley split of `query.target`'s variance among the
    /// exogenous noise terms of its ancestors and itself, with permutation-sampling, simulation
    /// and mechanism-fit uncertainty reported as separate components.
    ///
    /// `noise` must declare the noise terms independent (or pass a structure without bidirected
    /// edges); `data`, when supplied with a requested bootstrap, is resampled to quantify
    /// mechanism-fitting uncertainty, which is otherwise marked unquantified.
    ///
    /// # Errors
    ///
    /// Coded refusals for undeclared or correlated noise, unsupported mechanisms, oversized
    /// plans and invalid settings; model or replay failures; cancellation.
    attribute_intrinsic_influence(
        model: &CompiledCausalModel,
        noise: NoiseIndependence<'_>,
        data: Option<&TabularData>,
        query: &IntrinsicInfluenceQuery,
        ctx: &ExecutionContext,
    ) -> IntrinsicInfluenceResult => intrinsic_influence;

    /// Distribution-change attribution (pinned baseline-GCM parity).
    ///
    /// # Errors
    ///
    /// Attribution failures.
    attribute_distribution_change(
        model: &CompiledCausalModel,
        data: &TabularData,
        query: &ChangeAttributionQuery,
        options: &DistributionChangeOptions,
        ctx: &ExecutionContext,
    ) -> ChangeAttributionResult => distribution_change;

    /// Robust distribution-change attribution.
    ///
    /// # Errors
    ///
    /// Attribution failures.
    attribute_distribution_change_robust(
        model: &CompiledCausalModel,
        data: &TabularData,
        query: &ChangeAttributionQuery,
        options: &RobustChangeOptions,
        ctx: &ExecutionContext,
    ) -> ChangeAttributionResult => distribution_change_robust;

    /// Structure-change attribution between two DAGs (parent-set Shapley).
    ///
    /// # Errors
    ///
    /// Attribution failures.
    attribute_structure_change(
        baseline_model: &CompiledCausalModel,
        comparison_model: &CompiledCausalModel,
        data: &TabularData,
        query: &ChangeAttributionQuery,
        options: &StructureChangeOptions,
        ctx: &ExecutionContext,
    ) -> ChangeAttributionResult => structure_change;

    /// Mechanism-change detection (not attribution).
    ///
    /// # Errors
    ///
    /// Detection failures.
    mechanism_change_detection(
        model: &CompiledCausalModel,
        data: &TabularData,
        query: &MechanismChangeQuery,
        method: MechanismChangeMethod,
        ctx: &ExecutionContext,
    ) -> Vec<antecedent_attribution::MechanismChangeDetection> => detect_mechanism_changes;

    /// Unit-change attribution.
    ///
    /// # Errors
    ///
    /// Attribution failures.
    attribute_unit_change(
        model: &CompiledCausalModel,
        data: &TabularData,
        query: &UnitChangeQuery,
        ctx: &ExecutionContext,
    ) -> UnitChangeResult => unit_change;

    /// Path decomposition.
    ///
    /// # Errors
    ///
    /// Path / model failures.
    attribute_paths(
        model: &CompiledCausalModel,
        sources: &[VariableId],
        outcome: VariableId,
        max_paths: usize,
        max_len: usize,
        ctx: &ExecutionContext,
    ) -> ChangeAttributionResult => path_decompose;

    /// Feature relevance under interventions, with the sampling options as one value.
    ///
    /// # Errors
    ///
    /// Sampling failures.
    attribute_feature_relevance_with(
        model: &CompiledCausalModel,
        data: &TabularData,
        outcome: VariableId,
        features: &[VariableId],
        options: &antecedent_attribution::FeatureRelevanceOptions,
        ctx: &ExecutionContext,
    ) -> Vec<FeatureRelevance> => feature_relevance;
}

internal_engine! {
    /// Sample an interventional distribution under [`InterventionalDistributionQuery`].
    ///
    /// Thin wrapper over [`sample_do`]. Only [`TargetPopulation::AllObserved`] is supported.
    ///
    /// # Errors
    ///
    /// Query validation, unsupported target population, or sampling failures.
    #[doc(hidden)]
    pub fn sample_interventional_distribution(
        model: &CompiledCausalModel,
        query: &InterventionalDistributionQuery,
        n: usize,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<ValueBatch, CausalError> {
        query.validate()?;
        if query.target_population != TargetPopulation::AllObserved {
            return Err(crate::unsupported_reason!(
                "population_not_estimable",
                "sample_interventional_distribution only supports TargetPopulation::AllObserved"
            ));
        }
        sample_do(model, &query.interventions, n, rng, ctx)
    }
}

internal_engine! {
    /// Path-specific contribution via [`PathSpecificEffectQuery`].
    ///
    /// Thin wrapper over [`path_decompose`]. Only [`TargetPopulation::AllObserved`] is supported.
    /// When `path_nodes` is non-empty, keeps paths that visit every listed intermediate node.
    ///
    /// # Errors
    ///
    /// Query validation, unsupported target population, or path decomposition failures.
    #[doc(hidden)]
    pub fn attribute_path_specific(
        model: &CompiledCausalModel,
        query: &PathSpecificEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<ChangeAttributionResult, CausalError> {
        query.validate()?;
        if query.target_population != TargetPopulation::AllObserved {
            return Err(crate::unsupported_reason!(
                "population_not_estimable",
                "attribute_path_specific only supports TargetPopulation::AllObserved"
            ));
        }
        let mut result = path_decompose(
            model,
            &[query.treatment],
            query.outcome,
            query.max_paths,
            query.max_len,
            ctx,
        )?;
        if !query.path_nodes.is_empty() {
            let filtered: Vec<_> = result
                .path_breakdown
                .iter()
                .filter(|p| query.path_nodes.iter().all(|n| p.path.iter().any(|v| v == n)))
                .cloned()
                .collect();
            result.total_change = filtered.iter().map(|p| p.contribution).sum();
            result.path_breakdown = Arc::from(filtered);
        }
        Ok(result)
    }
}

internal_engine! {
    /// Feature relevance under interventions.
    ///
    /// Positional form of [`attribute_feature_relevance_with`], kept for the Python binding's
    /// keyword arguments.
    ///
    /// # Errors
    ///
    /// Sampling failures.
    #[allow(
        clippy::too_many_arguments,
        reason = "positional form of attribute_feature_relevance_with, which the Python binding calls with its keyword arguments"
    )]
    #[doc(hidden)]
    pub fn attribute_feature_relevance(
        model: &CompiledCausalModel,
        data: &TabularData,
        outcome: VariableId,
        features: &[VariableId],
        delta: f64,
        n_samples: usize,
        max_features: usize,
        ctx: &ExecutionContext,
    ) -> Result<Vec<FeatureRelevance>, CausalError> {
        let options =
            antecedent_attribution::FeatureRelevanceOptions { delta, n_samples, max_features };
        attribute_feature_relevance_with(model, data, outcome, features, &options, ctx)
    }
}

/// Rank root causes from an attribution result.
///
/// # Errors
///
/// Ranking failures.
pub fn rank_root_causes(
    attribution: &ChangeAttributionResult,
    ctx: &ExecutionContext,
) -> Result<Vec<RootCauseRank>, CausalError> {
    root_cause_rank(attribution, None, None, ctx).map_err(CausalError::from)
}
