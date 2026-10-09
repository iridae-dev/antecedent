//! Feature-only candidate Gaussian/learned joint transport public lifecycle.
#[cfg(feature = "calibration-internal")]
mod enabled {
    use crate::transport_interference_api::TransportIdentificationResult;
    use antecedent_core::reason_code;
    use antecedent_estimate::joint_bayesian_transport::{
        DataIdentity, GaussianPrior, JointPriors, JointTransportModel, JointTransportOptions,
        PriorProvenance, SourceData, SourceDependence, SourceSharing, TargetData,
        TransportGraphClass, VaryingBlock,
    };
    use antecedent_estimate::learned_joint_transport::LearnedJointModel;
    use antecedent_identify::{TransportFormula, TransportIdentification};
    use antecedent_io::joint_bayesian_transport_artifact::{
        DataIdentityWire, GaussianPriorWire, JointBayesianArtifactWire, JointBayesianConsumeLimits,
        JointBayesianExpectation, PriorProvenanceWire,
    };
    use antecedent_io::learned_joint_transport_artifact::{
        LearnedJointArtifactWire, LearnedJointConsumeLimits, LearnedJointExpectation,
    };
    use pyo3::prelude::*;
    use serde::Deserialize;

    const MAX_BYTES: usize = 16 * 1024 * 1024;
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct SourceWire {
        id: String,
        population: String,
        identity: DataIdentityWire,
        treatment: Vec<bool>,
        outcome: Vec<f64>,
        covariates: Vec<Vec<f64>>,
        noise_variance: f64,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct TargetWire {
        population: String,
        identity: DataIdentityWire,
        rows: usize,
        covariates: Vec<Vec<f64>>,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Request {
        features: Vec<String>,
        treatment: String,
        outcome: String,
        varying: String,
        sharing: String,
        dependence: String,
        invariant: GaussianPriorWire,
        varying_prior: GaussianPriorWire,
        sources: Vec<SourceWire>,
        target: TargetWire,
        draws: usize,
        seed: u64,
        basis_degree: usize,
        max_unsupported_mass: f64,
        conflict_z_threshold: f64,
    }
    #[derive(Default, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Expectation {
        premises_digest: Option<String>,
        data_digest: Option<String>,
    }
    fn identity(wire: DataIdentityWire) -> DataIdentity {
        DataIdentity { snapshot_digest: wire.snapshot_digest, datum_ids: wire.datum_ids }
    }
    fn prior(wire: GaussianPriorWire) -> GaussianPrior {
        GaussianPrior {
            mean: wire.mean,
            covariance: wire.covariance,
            provenance: match wire.provenance {
                PriorProvenanceWire::Declared => PriorProvenance::Declared,
                PriorProvenanceWire::Bank { bank_id, consumed } => PriorProvenance::Bank {
                    bank_id,
                    consumed: consumed.into_iter().map(identity).collect(),
                },
            },
        }
    }
    fn invalid(detail: &str) -> PyErr {
        crate::with_reason_code(crate::value_err(detail), reason_code!("invalid_argument"))
    }
    fn io(error: antecedent_io::IoError) -> PyErr {
        crate::transport_common::error(error)
    }

    fn check_identity(identity: &DataIdentityWire, rows: usize) -> PyResult<()> {
        let ids: std::collections::HashSet<_> = identity.datum_ids.iter().collect();
        if identity.snapshot_digest.trim().is_empty()
            || identity.datum_ids.len() != rows
            || ids.len() != rows
            || identity.datum_ids.iter().any(|id| id.trim().is_empty())
        {
            return Err(invalid("joint_transport.observation_identity_mismatch"));
        }
        Ok(())
    }
    fn check_bank(prior: &GaussianPriorWire) -> PyResult<()> {
        if let PriorProvenanceWire::Bank { bank_id, consumed } = &prior.provenance {
            if bank_id.trim().is_empty() || consumed.is_empty() {
                return Err(invalid("joint_transport.prior_bank_lineage_missing"));
            }
            for identity in consumed {
                check_identity(identity, identity.datum_ids.len())?;
            }
        }
        Ok(())
    }
    #[pyfunction]
    fn joint_transport_candidate(
        py: Python<'_>,
        proof: PyRef<'_, TransportIdentificationResult>,
        request_json: &str,
        learned: bool,
    ) -> PyResult<(String, Vec<u8>)> {
        if request_json.len() > MAX_BYTES {
            return Err(invalid("joint_transport.request_too_large"));
        }
        let request: Request = serde_json::from_str(request_json)
            .map_err(|e| invalid(&format!("joint_transport.invalid_request: {e}")))?;
        for source in &request.sources {
            check_identity(&source.identity, source.outcome.len())?;
        }
        check_identity(&request.target.identity, request.target.rows)?;
        check_bank(&request.invariant)?;
        check_bank(&request.varying_prior)?;
        let (id, graph, populations) = proof.original_joint_inputs();
        // A display-level graph_class cannot relabel the original certified graph.
        if (0..graph.admg.node_count()).any(|i| {
            antecedent_graph::DenseNodeId::try_from_usize(i)
                .is_ok_and(|node| !graph.admg.bidirected_neighbors(node).is_empty())
        }) {
            return Err(crate::refusal(
                reason_code!("route_not_supported"),
                "joint_transport.unsupported_graph",
            ));
        }
        let (response, target_factor) = match id {
            TransportIdentification::Transportable {
                formula: TransportFormula::Direct(response),
                ..
            } => (response, None),
            TransportIdentification::Transportable {
                formula: TransportFormula::Standardize { source_response, target_law, .. },
                ..
            } => (source_response, Some(target_law)),
            _ => {
                return Err(crate::refusal(
                    reason_code!("transport_not_certified"),
                    "joint_transport.identification_required",
                ));
            }
        };
        let resolve = |name: &str| {
            graph
                .names
                .iter()
                .position(|n| n == name)
                .map(|i| {
                    antecedent_core::VariableId::from_raw(
                        u32::try_from(i).expect("native graph variable indices fit u32"),
                    )
                })
                .ok_or_else(|| invalid("joint_transport.schema_mismatch"))
        };
        let treatment = resolve(&request.treatment)?;
        let outcome = resolve(&request.outcome)?;
        if response.variables.as_ref() != [outcome]
            || response.interventions.as_ref() != [treatment]
        {
            return Err(invalid("joint_transport.query_mismatch"));
        }
        if request.sources.iter().any(|s| s.population != response.population.as_ref())
            || target_factor.is_some_and(|f| request.target.population != f.population.as_ref())
        {
            return Err(invalid("joint_transport.population_mismatch"));
        }
        if request.target.population != populations.1
            || request.sources.iter().any(|s| s.population != populations.0)
        {
            return Err(invalid("joint_transport.population_mismatch"));
        }
        let features: Vec<u32> = request
            .features
            .iter()
            .map(|name| resolve(name).map(antecedent_core::VariableId::raw))
            .collect::<PyResult<_>>()?;
        let varying = match request.varying.as_str() {
            "intercept" => VaryingBlock::Intercept,
            "intercept_and_covariates" => VaryingBlock::InterceptAndCovariates,
            _ => return Err(invalid("joint_transport.varying_invalid")),
        };
        let sharing = match request.sharing.as_str() {
            "independent_varying_blocks" => SourceSharing::IndependentVaryingBlocks,
            "shared_varying_block" => SourceSharing::SharedVaryingBlock,
            _ => return Err(invalid("joint_transport.sharing_invalid")),
        };
        let dependence = match request.dependence.as_str() {
            "independent_samples" => SourceDependence::IndependentSamples,
            "overlapping_units" => SourceDependence::OverlappingUnits,
            "unknown" => SourceDependence::Unknown,
            _ => return Err(invalid("joint_transport.dependence_invalid")),
        };
        let source_rows: usize = request.sources.iter().map(|s| s.outcome.len()).sum();
        let limits = JointBayesianConsumeLimits::default();
        let basis = features.len().saturating_mul(if learned { request.basis_degree } else { 1 });
        let varying_dim = if varying == VaryingBlock::Intercept { 1 } else { 1 + basis };
        let invariant_dim =
            if varying == VaryingBlock::Intercept { 1 + 2 * basis } else { 1 + basis };
        let varying_blocks =
            if sharing == SourceSharing::SharedVaryingBlock { 1 } else { request.sources.len() };
        let dimension = invariant_dim.saturating_add(varying_dim.saturating_mul(varying_blocks));
        let draw_width = dimension.saturating_add(request.sources.len()).saturating_add(1);
        if request.sources.len() > limits.max_sources
            || source_rows > limits.max_source_rows
            || request.target.rows > limits.max_target_rows
            || features.len() > limits.max_features
            || request.draws.saturating_mul(draw_width) > limits.max_draw_cells
        {
            return Err(invalid("joint_transport.consumer_bounds_exceeded"));
        }
        let sources: Vec<_> = request
            .sources
            .into_iter()
            .map(|s| SourceData {
                id: s.id,
                identity: identity(s.identity),
                treatment: s.treatment,
                outcome: s.outcome,
                covariates: s.covariates,
                noise_variance: s.noise_variance,
            })
            .collect();
        let target = TargetData {
            identity: identity(request.target.identity),
            rows: request.target.rows,
            covariates: request.target.covariates,
        };
        let model = JointTransportModel {
            graph: TransportGraphClass::FixedDag,
            features,
            varying,
            sharing,
            dependence,
            priors: JointPriors {
                invariant: prior(request.invariant),
                varying: prior(request.varying_prior),
            },
            max_unsupported_mass: request.max_unsupported_mass,
            conflict_z_threshold: request.conflict_z_threshold,
        };
        let id = id.clone();
        let options = JointTransportOptions { draws: request.draws, seed: request.seed };
        let ctx = antecedent_core::ExecutionContext::production_default(request.seed);
        crate::detach_catch(py, move || {
            if learned {
                let model = LearnedJointModel {
                    graph: model.graph,
                    features: model.features,
                    basis_degree: request.basis_degree,
                    varying: model.varying,
                    sharing: model.sharing,
                    dependence: model.dependence,
                    priors: model.priors,
                    max_unsupported_mass: model.max_unsupported_mass,
                    conflict_z_threshold: model.conflict_z_threshold,
                };
                let (wire, _) =
                    LearnedJointArtifactWire::build(&id, &model, &sources, &target, options, &ctx)
                        .map_err(io)?;
                let data = wire.export().map_err(io)?;
                Ok((serde_json::to_string(&wire).map_err(|e| invalid(&e.to_string()))?, data))
            } else {
                let (wire, _) =
                    JointBayesianArtifactWire::build(&id, &model, &sources, &target, options, &ctx)
                        .map_err(io)?;
                let data = wire.export().map_err(io)?;
                Ok((serde_json::to_string(&wire).map_err(|e| invalid(&e.to_string()))?, data))
            }
        })
    }
    #[pyfunction]
    fn consume_joint_transport_candidate(
        py: Python<'_>,
        data: &[u8],
        expectation_json: &str,
        learned: bool,
    ) -> PyResult<String> {
        if data.len() > 128 * 1024 * 1024 || expectation_json.len() > MAX_BYTES {
            return Err(invalid("joint_transport.consumer_bounds_exceeded"));
        }
        let expected: Expectation =
            serde_json::from_str(expectation_json).map_err(|e| invalid(&e.to_string()))?;
        let ctx = antecedent_core::ExecutionContext::production_default(0);
        py.detach(|| {
            if learned {
                let (wire, _) = LearnedJointArtifactWire::consume_expecting(
                    data,
                    &LearnedJointExpectation {
                        premises_digest: expected.premises_digest,
                        data_digest: expected.data_digest,
                    },
                    LearnedJointConsumeLimits::default(),
                    &ctx,
                )
                .map_err(io)?;
                for source in &wire.sources {
                    check_identity(&source.identity, source.outcome.len())?;
                }
                check_identity(&wire.target.identity, wire.target.rows)?;
                check_bank(&wire.model.invariant_prior)?;
                check_bank(&wire.model.varying_prior)?;
                serde_json::to_string(&wire).map_err(|e| invalid(&e.to_string()))
            } else {
                let (wire, _) = JointBayesianArtifactWire::consume_expecting(
                    data,
                    &JointBayesianExpectation {
                        premises_digest: expected.premises_digest,
                        data_digest: expected.data_digest,
                    },
                    JointBayesianConsumeLimits::default(),
                    &ctx,
                )
                .map_err(io)?;
                for source in &wire.sources {
                    check_identity(&source.identity, source.outcome.len())?;
                }
                check_identity(&wire.target.identity, wire.target.rows)?;
                check_bank(&wire.model.invariant_prior)?;
                check_bank(&wire.model.varying_prior)?;
                serde_json::to_string(&wire).map_err(|e| invalid(&e.to_string()))
            }
        })
    }
    pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(joint_transport_candidate, m)?)?;
        m.add_function(wrap_pyfunction!(consume_joint_transport_candidate, m)?)
    }
}
pub(crate) fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    #[cfg(feature = "calibration-internal")]
    return enabled::register(m);
    #[cfg(not(feature = "calibration-internal"))]
    {
        let _ = m;
        Ok(())
    }
}
