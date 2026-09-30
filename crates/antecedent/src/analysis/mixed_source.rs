//! Prepared execution of a mixed-source proof-search formula.
//!
//! Preparation freezes one checked, catalog-bound derivation and compiles one
//! evaluation plan per request. Estimation evaluates those plans and never
//! searches again. The route is point-only: exact laws carry no sampling
//! uncertainty, so no interval is computed, stored or accepted. Refresh replaces
//! laws only for the snapshots the frozen catalog binds, so the proof is unchanged;
//! evidence under another snapshot, regime or catalog needs a new decision.
use super::StudyBuilder;
use super::transport_common::{err, estimate_err};
use antecedent_core::{ExecutionContext, SearchLimits};
use antecedent_expr::{
    Assignment, ExactDistribution, ExactEvaluationLimits, ExactEvaluationPlan, ExactTransportData,
};
use antecedent_identify::BoundMixedSourceFunctional;
use antecedent_io::IoError;
use antecedent_io::mixed_source_artifact::{
    MixedSourceArtifactInput, MixedSourceArtifactWire, MixedSourceConsumeLimits,
};

/// A prepared mixed-source formula: frozen proof, bound catalog, laws and compiled plans.
#[derive(Clone, Debug)]
pub struct PreparedMixedSource {
    graph: antecedent_graph::Admg,
    functional: BoundMixedSourceFunctional,
    search: SearchLimits,
    data: ExactTransportData,
    requests: Vec<Assignment>,
    limits: ExactEvaluationLimits,
    /// One compiled plan per request, in request order.
    plans: Vec<ExactEvaluationPlan>,
}

/// One execution: the point of every request.
#[derive(Clone, Debug)]
pub struct MixedSourceResult {
    distributions: Vec<ExactDistribution>,
}

impl MixedSourceResult {
    /// The point distribution of request 0.
    #[must_use]
    pub fn distribution(&self) -> &ExactDistribution {
        &self.distributions[0]
    }

    /// The point distribution of every request, in request order.
    #[must_use]
    pub fn distributions(&self) -> &[ExactDistribution] {
        &self.distributions
    }

    /// Export this result with its checked proof, source-named bindings, laws and
    /// requests, for an independent consumer.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn export(&self, prepared: &PreparedMixedSource) -> Result<Vec<u8>, IoError> {
        self.export_named(prepared, &[])
    }

    /// [`Self::export`], binding the name of every graph coordinate into the
    /// artifact's verified identity, so a consumer can refuse relabelled names.
    ///
    /// # Errors
    /// As [`Self::export`], or names that do not cover the graph.
    pub fn export_named(
        &self,
        prepared: &PreparedMixedSource,
        variable_names: &[String],
    ) -> Result<Vec<u8>, IoError> {
        MixedSourceArtifactWire::checked(&MixedSourceArtifactInput {
            graph: &prepared.graph,
            functional: &prepared.functional,
            search: prepared.search,
            data: &prepared.data,
            requests: &prepared.requests,
            limits: prepared.limits,
            variable_names,
            results: &self.distributions,
        })?
        .export()
    }
}

/// Independently consume a mixed-source artifact: re-decide the query, re-check the
/// proof, re-bind every source-named leaf, and recompute every point bit for bit,
/// without fetching data or providers.
///
/// # Errors
/// Any reconstruction or replay failure.
pub fn consume_mixed_source_artifact(
    bytes: &[u8],
    limits: MixedSourceConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<MixedSourceResult, IoError> {
    let consumed = MixedSourceArtifactWire::consume_with_limits(bytes, limits, ctx)?;
    Ok(MixedSourceResult { distributions: consumed.distributions })
}

impl StudyBuilder {
    /// Prepare exact-law evaluation of a checked mixed-source functional for
    /// `requests`. `search` records the limits it was decided under, so a consumer
    /// can refuse larger ones.
    ///
    /// # Errors
    /// No request, a functional that was not decided on `graph`, or laws, requests
    /// or resources that do not compile.
    pub fn mixed_source(
        graph: antecedent_graph::Admg,
        functional: BoundMixedSourceFunctional,
        search: SearchLimits,
        data: ExactTransportData,
        requests: Vec<Assignment>,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMixedSource, IoError> {
        if requests.is_empty() {
            return Err(err("prepare at least one target request"));
        }
        antecedent_identify::bind_mixed_source_catalog(
            &graph,
            functional.derivation(),
            functional.catalog(),
        )
        .map_err(|error| IoError::Refused {
            code: antecedent_core::reason_code!("transport_not_certified"),
            message: error.to_string(),
        })?;
        let plans = compile(&functional, &data, &requests, limits, ctx)?;
        Ok(PreparedMixedSource { graph, functional, search, data, requests, limits, plans })
    }
}

fn compile(
    functional: &BoundMixedSourceFunctional,
    data: &ExactTransportData,
    requests: &[Assignment],
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<Vec<ExactEvaluationPlan>, IoError> {
    requests
        .iter()
        .map(|request| {
            antecedent_estimate::prepare_exact_mixed_source(
                functional,
                data.clone(),
                request.clone(),
                limits,
                ctx,
            )
            .map_err(|e| estimate_err(antecedent_estimate::refuse_eval(&e)))
        })
        .collect()
}

impl PreparedMixedSource {
    /// Evaluate every retained plan. Exact laws are point-only.
    ///
    /// # Errors
    /// Cancellation or an evaluation refusal.
    pub fn estimate(&self, ctx: &ExecutionContext) -> Result<MixedSourceResult, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "mixed-source estimate")
            .map_err(estimate_err)?;
        let distributions = self
            .plans
            .iter()
            .map(|plan| {
                plan.evaluate(ctx).map_err(|e| estimate_err(antecedent_estimate::refuse_eval(&e)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MixedSourceResult { distributions })
    }

    /// Replace the laws for the snapshots the frozen catalog binds and recompile.
    /// The proof, bindings and requests are unchanged; laws under any other
    /// snapshot or regime are refused, which forces a new preparation.
    ///
    /// # Errors
    /// A law that does not match the frozen catalog, or cancellation.
    pub fn refresh(
        &self,
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "mixed-source refresh").map_err(estimate_err)?;
        let plans = compile(&self.functional, &data, &self.requests, self.limits, ctx)?;
        Ok(Self { data, plans, ..self.clone() })
    }

    /// The frozen, catalog-bound functional.
    #[must_use]
    pub const fn functional(&self) -> &BoundMixedSourceFunctional {
        &self.functional
    }

    /// Retained laws.
    #[must_use]
    pub const fn data(&self) -> &ExactTransportData {
        &self.data
    }

    /// Retained requests.
    #[must_use]
    pub fn requests(&self) -> &[Assignment] {
        &self.requests
    }

    /// The compiled evaluation plan of every request, in request order.
    #[must_use]
    pub fn plans(&self) -> &[ExactEvaluationPlan] {
        &self.plans
    }
}
