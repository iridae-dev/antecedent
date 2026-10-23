//! Graph-posterior mixtures compiled into a prepared study.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::discovery::{BayesianDiscoverParams, GraphMcmcSchedule};
use antecedent::{CausalContract, DiscoveredStructure, DiscoverySpec, Study};
use antecedent_core::{CausalQuery, CausalSchema, MediationQuery, ResponseQuery};
use antecedent_data::{TableView, TabularData, TimeSeriesData};
use pyo3::prelude::*;

use super::{PrepareRequest, PyPreparedAnalysis, apply_estimator, with_response_options};

pub(crate) fn contract_to_map(
    contract: &CausalContract,
    schema: &CausalSchema,
    query: &CausalQuery,
    registry: Option<&antecedent_core::PopulationRegistry>,
) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    out.insert("target".into(), contract.identities.target.to_hex());
    out.insert("identification".into(), contract.identities.identification.to_hex());
    if let Some(product) = contract.identities.identification_product {
        out.insert("identification_product".into(), product.to_hex());
    }
    if let Some(program) = contract.identities.program {
        out.insert("program".into(), program.to_hex());
    }
    out.insert("inference_binding".into(), contract.identities.inference_binding.to_hex());
    out.insert("observation".into(), contract.identities.observation.to_hex());
    out.insert("data_snapshot".into(), contract.identities.data_snapshot.to_hex());
    out.insert("graph_class".into(), contract.graph_class.as_str().to_string());
    out.insert("structure_source".into(), contract.structure_source.as_str().to_string());
    out.insert("accepted_version".into(), contract.accepted_version.to_string());
    if let Some(algorithm) = &contract.discovery_algorithm {
        out.insert("discovery_algorithm".into(), algorithm.to_string());
    }
    out.insert(
        "accepted_variable_binding".into(),
        if contract.accepted_variable_names.is_some() { "explicit" } else { "unbound" }.into(),
    );
    if let Some(status) = contract.support_status {
        out.insert("matrix_status".into(), status.as_str().to_string());
    } else if let Some(support) = contract.reasoning.support.as_ref() {
        // Graphless families leave the license enum unset and record `off_axis`
        // on the support slot. The sealed artifact already publishes that
        // string; the live contract has to carry the same one.
        out.insert("matrix_status".into(), support.matrix_status.to_string());
    }
    if let Some(support) = contract.reasoning.support.as_ref() {
        if let Some(coordinate) = &support.matrix_coordinate {
            out.insert("matrix_coordinate".into(), coordinate.to_string());
        }
        out.insert(
            "empirical_support".into(),
            support.empirical.label(std::string::ToString::to_string),
        );
    }
    out.insert(
        "assumptions".into(),
        contract.reasoning.assumptions.label(|slot| {
            slot.obligations.iter().map(|o| o.id.to_string()).collect::<Vec<_>>().join(",")
        }),
    );
    out.insert(
        "identification_status".into(),
        contract.reasoning.identification.label(|slot| slot.status.as_str().to_string()),
    );
    if let Some(slot) = contract.reasoning.identification.as_ref() {
        out.insert("identified_mass".into(), slot.identified_mass.to_string());
        out.insert("unidentified_mass".into(), slot.unidentified_mass.to_string());
        out.insert("unevaluable_mass".into(), slot.unevaluable_mass.to_string());
        out.insert("incomplete_search_mass".into(), slot.incomplete_search_mass.to_string());
        out.insert("full_mass_scope".into(), slot.full_mass_scope.to_string());
        out.insert("search_capped".into(), slot.search_capped.to_string());
    }
    if let Some(slot) = contract.reasoning.uncertainty.as_ref() {
        let components: Vec<_> = slot.components.iter().map(|c| serde_json::json!({
            "source": c.source.as_str(), "target": c.target.to_string(), "omitted": c.omitted
        })).collect();
        out.insert("uncertainty_components".into(), serde_json::json!(components).to_string());
    }
    if let Some(slot) = contract.reasoning.assumptions.as_ref() {
        let obligations: Vec<_> = slot.obligations.iter().map(|o| serde_json::json!({
            "id": o.id.to_string(), "scope": o.scope.as_str(), "kind": o.kind.as_str(), "status": o.status.as_str()
        })).collect();
        out.insert("assumption_obligations".into(), serde_json::json!(obligations).to_string());
    }
    out.insert("uncertainty".into(), contract.reasoning.uncertainty.label(|_| "available".into()));
    out.insert(
        "variable_names".into(),
        schema
            .variables()
            .iter()
            .map(|variable| variable.name.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    if let Ok(wire) = antecedent_io::causal_query_to_wire_with_registry(query, registry) {
        for (key, value) in antecedent_io::executed_functional_labels(&wire) {
            out.insert(key, value);
        }
    }
    out
}

/// Flat inspect keys for the score-reuse and target-weights identities.
pub(super) fn insert_reuse_identities(
    report: &mut std::collections::HashMap<String, String>,
    score_reuse: Option<antecedent_core::SemanticDigest>,
    target_weights: Option<antecedent_core::SemanticDigest>,
) {
    if let Some(digest) = score_reuse {
        report.insert("score_reuse".into(), digest.to_hex());
    }
    if let Some(digest) = target_weights {
        report.insert("target_weights".into(), digest.to_hex());
    }
}

pub(super) fn require_prepared_names(
    expected: &[String],
    names: &[String],
    op: &str,
) -> PyResult<()> {
    if names != expected {
        return Err(crate::with_reason_code(
            crate::value_err(format!(
                "prepared {op} requires the same column names (order) as prepare"
            )),
            antecedent_core::reason_code!("schema_mismatch"),
        ));
    }
    Ok(())
}

pub(super) fn take_supplied_posterior(
    posterior: Option<Bound<'_, crate::bayesian::PyGraphPosterior>>,
    names: &[String],
) -> PyResult<Option<antecedent::discovery::GraphPosterior>> {
    posterior
        .map(|bound| {
            let posterior = bound.borrow();
            posterior.require_bound_to(names)?;
            posterior.to_rust()
        })
        .transpose()
}

pub(super) fn exact_dag_posterior(
    supplied: Option<antecedent::discovery::GraphPosterior>,
    data: &TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<antecedent::discovery::GraphPosterior> {
    if let Some(gp) = supplied {
        return Ok(gp);
    }
    let vars: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
    compiled_posterior(
        DiscoverySpec::exact_dag_posterior(BayesianDiscoverParams::default()),
        crate::compiled_route_api::DiscoverInput::Tabular(data.clone()),
        &vars,
        ctx,
    )
}

/// The posterior a compiled structure-posterior search returns.
fn compiled_posterior(
    spec: DiscoverySpec,
    input: crate::compiled_route_api::DiscoverInput,
    variables: &[antecedent_core::VariableId],
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<antecedent::discovery::GraphPosterior> {
    let (posterior, _route) =
        crate::discovery_api::compiled_search(spec, input, variables, ctx, |structure| {
            match structure {
                DiscoveredStructure::Posterior(posterior) => Some(posterior),
                _ => None,
            }
        })?;
    Ok(posterior)
}

pub(super) fn dbn_posterior(
    supplied: Option<antecedent::discovery::GraphPosterior>,
    series: &TimeSeriesData,
    max_lag: u32,
    force_mcmc: bool,
    n_chains: u32,
    n_warmup: u32,
    mcmc_draws: u32,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<antecedent::discovery::GraphPosterior> {
    if let Some(gp) = supplied {
        return Ok(gp);
    }
    let vars: Vec<_> = series.schema().variables().iter().map(|v| v.id).collect();
    let schedule = GraphMcmcSchedule { n_chains, n_warmup, n_draws: mcmc_draws, thin: 1 };
    compiled_posterior(
        DiscoverySpec::dbn_posterior(
            BayesianDiscoverParams::default(),
            max_lag,
            force_mcmc,
            schedule,
        ),
        crate::compiled_route_api::DiscoverInput::Series(series.clone()),
        &vars,
        ctx,
    )
}

pub(super) fn finish_static_graph_posterior(
    data: TabularData,
    mut req: PrepareRequest,
    gp: antecedent::discovery::GraphPosterior,
    query: CausalQuery,
    spec: Option<antecedent::EstimatorSpec>,
    response_options: Option<antecedent_estimate::ContinuousResponseOptions>,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<PyPreparedAnalysis> {
    req.refuse_prior("a graph-posterior mixture")?;
    if spec.is_some() && req.opts.inference.is_bayesian() {
        return Err(crate::refusal(
            antecedent_core::reason_code!("option_not_applicable"),
            "a Bayesian graph-posterior mixture fits \
             Bayesian g-computation per atom; a configured Frequentist estimator does not apply",
        ));
    }
    let builder = with_response_options(
        Study::tabular(data).graph_posterior(gp).query(query),
        response_options,
    );
    let builder = apply_estimator(req.opts.apply_budget_for(builder, spec.is_some()), None, spec)?;
    let builder = req.opts.apply_inference(builder)?;
    req.compile_with(builder, false, ctx)
}

pub(super) enum SeriesGraphQuery {
    Temporal(antecedent_core::TemporalEffectQuery),
    Mediation(MediationQuery),
    Response(ResponseQuery),
}

pub(super) fn finish_series_graph_posterior(
    series: TimeSeriesData,
    mut req: PrepareRequest,
    gp: antecedent::discovery::GraphPosterior,
    query: SeriesGraphQuery,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<PyPreparedAnalysis> {
    req.refuse_prior("a DBN graph-posterior mixture")?;
    let mut builder = Study::series(series).graph_posterior(gp);
    builder = match query {
        SeriesGraphQuery::Temporal(q) => builder.temporal_query(q),
        SeriesGraphQuery::Mediation(q) => builder.query(CausalQuery::Mediation(q)),
        SeriesGraphQuery::Response(q) => builder.query(CausalQuery::Response(q)),
    };
    let builder = req.opts.apply(builder);
    let builder = req.opts.apply_inference(builder)?;
    req.compile_with(builder, true, ctx)
}
