//! Graph interchange and durable artifacts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::CausalError;

/// Parse a DOT digraph into a [`causal_graph::Dag`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed DOT or invalid DAG structure.
pub fn dag_from_dot(dot: &str) -> Result<causal_graph::Dag, CausalError> {
    causal_io::dag_from_dot(dot).map_err(CausalError::from)
}

/// Serialize a DAG to DOT.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn dag_to_dot(
    dag: &causal_graph::Dag,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::dag_to_dot(dag, names).map_err(CausalError::from)
}

/// Parse a JSON DAG document into a [`causal_graph::Dag`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed JSON or invalid DAG structure.
pub fn dag_from_json(json: &str) -> Result<causal_graph::Dag, CausalError> {
    causal_io::dag_from_json(json).map_err(CausalError::from)
}

/// Serialize a DAG to JSON.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn dag_to_json(
    dag: &causal_graph::Dag,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::dag_to_json(dag, names).map_err(CausalError::from)
}

/// Parse a GML digraph into a [`causal_graph::Dag`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed GML or invalid DAG structure.
pub fn dag_from_gml(gml: &str) -> Result<causal_graph::Dag, CausalError> {
    causal_io::dag_from_gml(gml).map_err(CausalError::from)
}

/// Serialize a DAG to GML.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn dag_to_gml(
    dag: &causal_graph::Dag,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::dag_to_gml(dag, names).map_err(CausalError::from)
}

/// Parse `NetworkX` `node_link_data` JSON into a [`causal_graph::Dag`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed JSON or invalid DAG structure.
pub fn dag_from_networkx_node_link(json: &str) -> Result<causal_graph::Dag, CausalError> {
    causal_io::dag_from_networkx_node_link(json).map_err(CausalError::from)
}

/// Serialize a DAG to `NetworkX` `node_link_data` JSON.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn dag_to_networkx_node_link(
    dag: &causal_graph::Dag,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::dag_to_networkx_node_link(dag, names).map_err(CausalError::from)
}

/// Parse `NetworkX` `adjacency_data` JSON into a [`causal_graph::Dag`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed JSON or invalid DAG structure.
pub fn dag_from_networkx_adjacency(json: &str) -> Result<causal_graph::Dag, CausalError> {
    causal_io::dag_from_networkx_adjacency(json).map_err(CausalError::from)
}

/// Serialize a DAG to `NetworkX` `adjacency_data` JSON.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn dag_to_networkx_adjacency(
    dag: &causal_graph::Dag,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::dag_to_networkx_adjacency(dag, names).map_err(CausalError::from)
}

/// Parse DOT into a [`causal_graph::Pag`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed input.
pub fn pag_from_dot(dot: &str) -> Result<causal_graph::Pag, CausalError> {
    causal_io::pag_from_dot(dot).map_err(CausalError::from)
}
/// Serialize a PAG to DOT.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn pag_to_dot(
    pag: &causal_graph::Pag,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::pag_to_dot(pag, names).map_err(CausalError::from)
}
/// Parse JSON into a [`causal_graph::Pag`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed input.
pub fn pag_from_json(json: &str) -> Result<causal_graph::Pag, CausalError> {
    causal_io::pag_from_json(json).map_err(CausalError::from)
}
/// Serialize a PAG to JSON.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn pag_to_json(
    pag: &causal_graph::Pag,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::pag_to_json(pag, names).map_err(CausalError::from)
}
/// Parse GML into a [`causal_graph::Pag`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed input.
pub fn pag_from_gml(gml: &str) -> Result<causal_graph::Pag, CausalError> {
    causal_io::pag_from_gml(gml).map_err(CausalError::from)
}
/// Serialize a PAG to GML.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn pag_to_gml(
    pag: &causal_graph::Pag,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::pag_to_gml(pag, names).map_err(CausalError::from)
}
/// Parse `NetworkX` node-link JSON into a [`causal_graph::Pag`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed input.
pub fn pag_from_networkx_node_link(json: &str) -> Result<causal_graph::Pag, CausalError> {
    causal_io::pag_from_networkx_node_link(json).map_err(CausalError::from)
}
/// Serialize a PAG to `NetworkX` node-link JSON.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn pag_to_networkx_node_link(
    pag: &causal_graph::Pag,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::pag_to_networkx_node_link(pag, names).map_err(CausalError::from)
}

/// Parse DOT into a [`causal_graph::Cpdag`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed input.
pub fn cpdag_from_dot(dot: &str) -> Result<causal_graph::Cpdag, CausalError> {
    causal_io::cpdag_from_dot(dot).map_err(CausalError::from)
}
/// Serialize a CPDAG to DOT.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn cpdag_to_dot(
    cpdag: &causal_graph::Cpdag,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::cpdag_to_dot(cpdag, names).map_err(CausalError::from)
}
/// Parse JSON into a [`causal_graph::Cpdag`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed input.
pub fn cpdag_from_json(json: &str) -> Result<causal_graph::Cpdag, CausalError> {
    causal_io::cpdag_from_json(json).map_err(CausalError::from)
}
/// Serialize a CPDAG to JSON.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn cpdag_to_json(
    cpdag: &causal_graph::Cpdag,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::cpdag_to_json(cpdag, names).map_err(CausalError::from)
}
/// Parse GML into a [`causal_graph::Cpdag`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed input.
pub fn cpdag_from_gml(gml: &str) -> Result<causal_graph::Cpdag, CausalError> {
    causal_io::cpdag_from_gml(gml).map_err(CausalError::from)
}
/// Serialize a CPDAG to GML.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn cpdag_to_gml(
    cpdag: &causal_graph::Cpdag,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::cpdag_to_gml(cpdag, names).map_err(CausalError::from)
}
/// Parse `NetworkX` node-link JSON into a [`causal_graph::Cpdag`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed input.
pub fn cpdag_from_networkx_node_link(json: &str) -> Result<causal_graph::Cpdag, CausalError> {
    causal_io::cpdag_from_networkx_node_link(json).map_err(CausalError::from)
}
/// Serialize a CPDAG to `NetworkX` node-link JSON.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn cpdag_to_networkx_node_link(
    cpdag: &causal_graph::Cpdag,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::cpdag_to_networkx_node_link(cpdag, names).map_err(CausalError::from)
}

/// Parse DOT into a [`causal_graph::Admg`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed input.
pub fn admg_from_dot(dot: &str) -> Result<causal_graph::Admg, CausalError> {
    causal_io::admg_from_dot(dot).map_err(CausalError::from)
}
/// Serialize an ADMG to DOT.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn admg_to_dot(
    admg: &causal_graph::Admg,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::admg_to_dot(admg, names).map_err(CausalError::from)
}
/// Parse JSON into a [`causal_graph::Admg`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed input.
pub fn admg_from_json(json: &str) -> Result<causal_graph::Admg, CausalError> {
    causal_io::admg_from_json(json).map_err(CausalError::from)
}
/// Serialize an ADMG to JSON.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn admg_to_json(
    admg: &causal_graph::Admg,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::admg_to_json(admg, names).map_err(CausalError::from)
}
/// Parse GML into a [`causal_graph::Admg`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed input.
pub fn admg_from_gml(gml: &str) -> Result<causal_graph::Admg, CausalError> {
    causal_io::admg_from_gml(gml).map_err(CausalError::from)
}
/// Serialize an ADMG to GML.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn admg_to_gml(
    admg: &causal_graph::Admg,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::admg_to_gml(admg, names).map_err(CausalError::from)
}
/// Parse `NetworkX` node-link JSON into a [`causal_graph::Admg`].
///
/// # Errors
///
/// [`CausalError::Serialization`] on malformed input.
pub fn admg_from_networkx_node_link(json: &str) -> Result<causal_graph::Admg, CausalError> {
    causal_io::admg_from_networkx_node_link(json).map_err(CausalError::from)
}
/// Serialize an ADMG to `NetworkX` node-link JSON.
///
/// # Errors
///
/// [`CausalError::Serialization`] on conversion failure.
pub fn admg_to_networkx_node_link(
    admg: &causal_graph::Admg,
    names: Option<&[String]>,
) -> Result<String, CausalError> {
    causal_io::admg_to_networkx_node_link(admg, names).map_err(CausalError::from)
}

/// Encode a model bundle to durable bytes.
///
/// # Errors
///
/// [`CausalError::Serialization`] on IO failures.
pub fn encode_model_bundle_bytes(
    input: &causal_io::ModelBundleEncode<'_>,
) -> Result<Vec<u8>, CausalError> {
    let art = causal_io::encode_model_bundle(input).map_err(CausalError::from)?;
    let mut buf = Vec::new();
    art.write_to(&mut buf).map_err(CausalError::from)?;
    Ok(buf)
}

/// Decode a model bundle from durable bytes (migrates format if needed).
///
/// # Errors
///
/// [`CausalError::Serialization`] on IO failures.
pub fn decode_model_bundle_bytes(bytes: &[u8]) -> Result<causal_io::ModelBundle, CausalError> {
    let art = causal_io::read_and_migrate(bytes).map_err(CausalError::from)?;
    causal_io::decode_model_bundle(&art).map_err(CausalError::from)
}

/// Encode a [`causal_estimate::CausalPosterior`] to durable bytes.
///
/// # Errors
///
/// [`CausalError::Serialization`] on IO failures.
pub fn encode_causal_posterior_bytes(
    posterior: &causal_estimate::CausalPosterior,
    artifact_id: &str,
) -> Result<Vec<u8>, CausalError> {
    causal_io::encode_causal_posterior_bytes(posterior, artifact_id).map_err(CausalError::from)
}

/// Encode a [`causal_estimate::CausalPosterior`] to a durable artifact.
///
/// # Errors
///
/// [`CausalError::Serialization`] on IO failures.
pub fn encode_causal_posterior(
    posterior: &causal_estimate::CausalPosterior,
    artifact_id: &str,
) -> Result<causal_io::EncodedArtifact, CausalError> {
    causal_io::encode_causal_posterior(posterior, artifact_id).map_err(CausalError::from)
}

/// Decode posterior wire metadata + draws.
///
/// # Errors
///
/// [`CausalError::Serialization`] on IO failures.
pub fn decode_causal_posterior_bytes(
    bytes: &[u8],
) -> Result<(causal_io::CausalPosteriorWire, Vec<f64>), CausalError> {
    causal_io::decode_causal_posterior_bytes(bytes).map_err(CausalError::from)
}

/// Hydrate a coefficient [`causal_prob::PriorSet`] from posterior artifact bytes.
///
/// Uses per-coefficient posterior means and SDs (identical-subspace mapping).
/// Effect columns are ignored. Prefer [`hydrate_prior_from_posterior_bytes`] when
/// a heterogeneous mapping is required.
///
/// # Errors
///
/// Decode failures or hydrate failures (no coefficients / non-finite summaries).
pub fn prior_set_from_posterior_bytes(bytes: &[u8]) -> Result<causal_prob::PriorSet, CausalError> {
    use std::sync::Arc;

    use causal_estimate::hydrate_prior_from_quantity_summaries;
    use causal_io::PosteriorQuantityWire;
    use causal_prob::PosteriorQuantityKind;

    let (wire, _) = decode_causal_posterior_bytes(bytes)?;
    let quantities: Vec<PosteriorQuantityKind> = wire
        .quantities
        .iter()
        .map(|q| match q {
            PosteriorQuantityWire::Coefficient { index, name } => {
                PosteriorQuantityKind::Coefficient {
                    index: *index as usize,
                    name: name.as_ref().map(|s| Arc::<str>::from(s.as_str())),
                }
            }
            PosteriorQuantityWire::ResidualVariance => PosteriorQuantityKind::ResidualVariance,
            PosteriorQuantityWire::Effect { name } => {
                PosteriorQuantityKind::Effect { name: Arc::from(name.as_str()) }
            }
            PosteriorQuantityWire::Scalar { name } => {
                PosteriorQuantityKind::Scalar { name: Arc::from(name.as_str()) }
            }
        })
        .collect();
    hydrate_prior_from_quantity_summaries(&quantities, &wire.mean, &wire.sd, None)
        .map_err(CausalError::from)
}
