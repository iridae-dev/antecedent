//! Seals over the identification products a lowered program retains.
//!
//! Compile identifies once and retains the products with the program. Each product that a
//! checked operation proves its claim from is sealed there: a digest over the full
//! identification result (status, estimands, expression arena, derivation rules, required
//! assumptions, hedge, and the query it answers) and the selected estimand. An operation that
//! holds one of those products verifies it against the seal instead of identifying again, so
//! identification runs once and a retained product that changed afterwards is refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::IdentityDomain;
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::result::IdentificationResult;

use super::prepared::{CachedTemporalIdentification, IdentificationCache};
use crate::error::CausalError;

const STATIC_DOMAIN: &str = "antecedent.lowered.identification.seal.static.v1";
const TEMPORAL_DOMAIN: &str = "antecedent.lowered.identification.seal.temporal.v1";
const RESULT_DOMAIN: &str = "antecedent.lowered.identification.seal.result.v1";

fn word(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn text(hasher: &mut blake3::Hasher, value: &str) {
    word(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

fn variables(hasher: &mut blake3::Hasher, set: &[antecedent_core::VariableId]) {
    word(hasher, set.len() as u64);
    for variable in set {
        word(hasher, u64::from(variable.raw()));
    }
}

fn estimand_into(hasher: &mut blake3::Hasher, estimand: &IdentifiedEstimand) {
    text(hasher, &estimand.method);
    variables(hasher, &estimand.adjustment_set);
    variables(hasher, &estimand.instruments);
    variables(hasher, &estimand.mediators);
    word(hasher, u64::from(estimand.functional.raw()));
    match estimand.rd_design {
        None => word(hasher, 0),
        Some(design) => {
            word(hasher, 1);
            word(hasher, u64::from(design.running_variable.raw()));
            word(hasher, design.cutoff.to_bits());
            word(hasher, design.bandwidth.to_bits());
        }
    }
}

/// Digest of the whole result: the product payload (everything but search effort and prose)
/// and the query it answers.
fn result_into(
    hasher: &mut blake3::Hasher,
    result: &IdentificationResult,
) -> Result<(), CausalError> {
    let product = antecedent_io_core::identification_product_digest(result, false)?;
    let query = antecedent_io_core::digest_wire(
        IdentityDomain::IdentificationProduct,
        &antecedent_io_core::causal_query_to_wire(&result.query)?,
    )?;
    hasher.update(product.as_bytes());
    hasher.update(query.as_bytes());
    Ok(())
}

/// Seal of a static identification and the estimand selected from it.
///
/// # Errors
///
/// The result or its query cannot be encoded.
pub(crate) fn seal_static(
    result: &IdentificationResult,
    estimand: &IdentifiedEstimand,
) -> Result<[u8; 32], CausalError> {
    let mut hasher = blake3::Hasher::new_derive_key(STATIC_DOMAIN);
    result_into(&mut hasher, result)?;
    estimand_into(&mut hasher, estimand);
    Ok(*hasher.finalize().as_bytes())
}

/// Seal of an aggregate identification result that carries no single selected estimand.
///
/// # Errors
///
/// The result or its query cannot be encoded.
pub(crate) fn seal_result(result: &IdentificationResult) -> Result<[u8; 32], CausalError> {
    let mut hasher = blake3::Hasher::new_derive_key(RESULT_DOMAIN);
    result_into(&mut hasher, result)?;
    Ok(*hasher.finalize().as_bytes())
}

/// Seal of a temporal identification: every horizon's result, selected estimand and lag
/// indexer shape, in order.
///
/// # Errors
///
/// A horizon's result or query cannot be encoded.
pub(crate) fn seal_temporal(cache: &CachedTemporalIdentification) -> Result<[u8; 32], CausalError> {
    let mut hasher = blake3::Hasher::new_derive_key(TEMPORAL_DOMAIN);
    word(&mut hasher, cache.by_horizon.len() as u64);
    for entry in cache.by_horizon.iter() {
        word(&mut hasher, u64::from(entry.horizon));
        result_into(&mut hasher, &entry.identification)?;
        estimand_into(&mut hasher, &entry.estimand);
        word(&mut hasher, u64::from(entry.indexer.variable_count()));
        word(&mut hasher, u64::from(entry.indexer.history()));
        word(&mut hasher, u64::from(entry.indexer.horizon()));
    }
    Ok(*hasher.finalize().as_bytes())
}

/// Seal of one retained product, for the products a checked operation proves from. `None`
/// for a product no checked operation verifies.
///
/// # Errors
///
/// The product cannot be encoded.
pub(crate) fn seal_cache(cache: &IdentificationCache) -> Result<Option<[u8; 32]>, CausalError> {
    Ok(match cache {
        IdentificationCache::Static(cache) => {
            Some(seal_static(&cache.identification, &cache.estimand)?)
        }
        IdentificationCache::Temporal(cache) => Some(seal_temporal(cache)?),
        IdentificationCache::Pag(cache) => Some(seal_result(&cache.identification)?),
        IdentificationCache::Cpdag(cache) => Some(seal_result(&cache.identification)?),
        _ => None,
    })
}
