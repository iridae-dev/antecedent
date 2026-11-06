//! Parsing and binding cluster-DML independence units.
// SPDX-License-Identifier: MIT OR Apache-2.0

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{get_string, get_u32, get_u32_list, invalid, type_name};

/// Valid `cluster_dml` sub-dict keys (see [`build_cluster_dml`]).
const CLUSTER_DML_KEYS: &[&str] = &[
    "cluster_ids",
    "second_cluster_ids",
    "source_cluster_ids",
    "source_second_cluster_ids",
    "min_clusters",
    "min_components_per_fold",
    "unit",
];

/// Declared cluster-DML independence unit of an `aipw` estimator and its row labels.
pub(super) struct ClusterDmlConfig {
    pub(super) spec: antecedent::estimate::ClusterDml,
    pub(super) first: Vec<u32>,
    pub(super) second: Option<Vec<u32>>,
    /// The labels are those of every row of the source table, not of the analysed rows.
    pub(super) per_source_row: bool,
}

/// Build the declared cluster-DML independence unit of `aipw` and its row labels from
/// `cluster_dml` (`{"cluster_ids": [int], "min_clusters": int, "unit": "cluster" | "dyad",
/// "second_cluster_ids": [int], "min_components_per_fold": int}`).
///
/// A `dyad` unit takes the second endpoint labels (`second_cluster_ids`, required) and a
/// minimum components per fold; a `cluster` unit takes neither.
pub(super) fn build_cluster_dml(dict: &Bound<'_, PyDict>) -> PyResult<Option<ClusterDmlConfig>> {
    use antecedent::estimate::{ClusterDml, DEFAULT_MIN_CLUSTERS, DEFAULT_MIN_COMPONENTS_PER_FOLD};
    let Some(value) = dict.get_item("cluster_dml")? else {
        return Ok(None);
    };
    let sub = value.cast::<PyDict>().map_err(|_| {
        invalid(format!(
            "estimator_config['cluster_dml'] must be a dict, got {}",
            type_name(&value)
        ))
    })?;
    for (key_obj, _val) in sub.iter() {
        let key: String = key_obj
            .extract()
            .map_err(|_| invalid("estimator_config['cluster_dml'] keys must be str".to_string()))?;
        if !CLUSTER_DML_KEYS.contains(&key.as_str()) {
            return Err(invalid(format!(
                "unknown estimator_config['cluster_dml'] key {key:?}; valid keys are: {}",
                CLUSTER_DML_KEYS.join(", "),
            )));
        }
    }
    let retained = get_u32_list(sub, "cluster_ids")?;
    let source = get_u32_list(sub, "source_cluster_ids")?;
    let (ids, per_source_row) = match (retained, source) {
        (Some(ids), None) => (ids, false),
        (None, Some(ids)) => (ids, true),
        (Some(_), Some(_)) => {
            return Err(invalid(
                "estimator_config['cluster_dml'] takes cluster_ids or source_cluster_ids, not both"
                    .to_string(),
            ));
        }
        (None, None) => {
            return Err(invalid(
                "estimator_config['cluster_dml'] requires cluster_ids".to_string(),
            ));
        }
    };
    let second = match (
        get_u32_list(sub, "second_cluster_ids")?,
        get_u32_list(sub, "source_second_cluster_ids")?,
    ) {
        (second, None) if !per_source_row => second,
        (None, second) if per_source_row => second,
        (None, None) => None,
        _ => {
            return Err(invalid(
                "estimator_config['cluster_dml'] declares both endpoints per analysed row \
                 (cluster_ids, second_cluster_ids) or per source row (source_cluster_ids, \
                 source_second_cluster_ids), not a mixture"
                    .to_string(),
            ));
        }
    };
    let min_clusters = get_u32(sub, "min_clusters")?.map_or(DEFAULT_MIN_CLUSTERS, |m| m as usize);
    let min_components = get_u32(sub, "min_components_per_fold")?;
    let refused = |error: antecedent::estimate::EstimationError| {
        let message = error.to_string();
        let tail = antecedent::reason::reason_code::split_prefix(&message)
            .map_or(message.as_str(), |(_, tail)| tail);
        invalid(tail.to_string())
    };
    let spec = match get_string(sub, "unit")?.as_deref().unwrap_or("cluster") {
        "cluster" => {
            if second.is_some() || min_components.is_some() {
                return Err(invalid(
                    "estimator_config['cluster_dml'] second_cluster_ids and \
                     min_components_per_fold belong to unit='dyad'"
                        .to_string(),
                ));
            }
            ClusterDml::new(min_clusters).map_err(refused)?
        }
        "dyad" => {
            match &second {
                None => {
                    return Err(invalid(
                        "estimator_config['cluster_dml'] unit='dyad' requires second_cluster_ids"
                            .to_string(),
                    ));
                }
                Some(second) if second.len() != ids.len() => {
                    return Err(invalid(format!(
                        "estimator_config['cluster_dml'] second_cluster_ids has {} labels but \
                         cluster_ids has {}",
                        second.len(),
                        ids.len()
                    )));
                }
                Some(_) => {}
            }
            let per_fold = min_components.map_or(DEFAULT_MIN_COMPONENTS_PER_FOLD, |m| m as usize);
            ClusterDml::dyadic(min_clusters, per_fold).map_err(refused)?
        }
        other => {
            return Err(invalid(format!(
                "estimator_config['cluster_dml']['unit'] {other:?} is not recognized; \
                 use cluster|dyad"
            )));
        }
    };
    Ok(Some(ClusterDmlConfig { spec, first: ids, second, per_source_row }))
}

/// Store a parsed cluster-DML declaration on an estimator's [`DeclaredUnits`](antecedent::estimate::DeclaredUnits).
pub(super) fn declare_units(
    units: &mut antecedent::estimate::DeclaredUnits,
    config: Option<ClusterDmlConfig>,
) {
    let Some(ClusterDmlConfig { spec, first, second, per_source_row }) = config else {
        return;
    };
    units.spec = Some(spec);
    let (first_slot, second_slot) = if per_source_row {
        (&mut units.first_source, &mut units.second_source)
    } else {
        (&mut units.first, &mut units.second)
    };
    *first_slot = Some(first);
    *second_slot = second;
}
