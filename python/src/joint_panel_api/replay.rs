//! Historical replay of exported experimental updates, without current authority.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{context, facade, panel_error};
use antecedent_io::distribution_artifact::DistributionArtifact;
use pyo3::prelude::*;

/// Posterior weights and weighted coordinate means of a replayed numerical law.
fn law_summary(law: &DistributionArtifact) -> serde_json::Value {
    let [rows, width] = law.shape();
    let weights =
        law.metadata().weights.clone().unwrap_or_else(|| vec![1.0 / rows.max(1) as f64; rows]);
    let draws = law.draws();
    let means = (0..width)
        .map(|column| (0..rows).map(|row| weights[row] * draws[row * width + column]).sum::<f64>())
        .collect::<Vec<_>>();
    serde_json::json!({
        "draws": rows,
        "coordinates": width,
        "posterior_weights": weights,
        "posterior_means": means,
        "provider_id": law.metadata().identity.provider_id,
        "dependency": law.metadata().identity.causal_contract_id,
    })
}

/// Replay an exported population-design experiment from its retained inputs alone.
#[pyfunction]
#[pyo3(signature=(artifact,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn replay_joint_law_experiment(
    py: Python<'_>,
    artifact: &[u8],
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    crate::detach_catch(py, || {
        let historical =
            facade::JointLawExperimentResult::replay(artifact, &ctx).map_err(panel_error)?;
        let mut summary = law_summary(historical.numerical_law());
        summary["replays_original_design"] = historical.replays_original_design().into();
        summary["grants_current_authority"] = historical.grants_current_authority().into();
        Ok(summary.to_string())
    })
}

/// Replay an exported method-native experiment, re-executing its original identification.
#[pyfunction]
#[pyo3(signature=(artifact,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn replay_native_joint_experiment(
    py: Python<'_>,
    artifact: &[u8],
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    crate::detach_catch(py, || {
        let historical =
            facade::NativeJointExperimentResult::replay(artifact, &ctx).map_err(panel_error)?;
        let mut summary = law_summary(historical.numerical_law());
        summary["replays_original_design"] = true.into();
        summary["grants_current_authority"] = historical.grants_current_authority().into();
        Ok(summary.to_string())
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(replay_joint_law_experiment, module)?)?;
    module.add_function(wrap_pyfunction!(replay_native_joint_experiment, module)?)?;
    Ok(())
}
