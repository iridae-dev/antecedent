//! Python bridge for scoring a decision demand from what an answer published.
//!
//! Python names the demand (an affine utility read from a parameter posterior or from the
//! mean, or a demand no mean can carry) and the carriers the answer published. Which carrier
//! scores the demand, the affine arithmetic and the typed missing-carrier refusal are Rust's.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::{DecisionDemand, PublishedCarriers, score_decision};
use pyo3::prelude::*;

use crate::value_err;

/// `(value, carrier)` when scored, or the name of the missing carrier when refused.
type Scored = (Option<(f64, String)>, Option<String>);

fn affine(slope: Option<f64>, intercept: Option<f64>, demand: &str) -> PyResult<(f64, f64)> {
    match (slope, intercept) {
        (Some(slope), Some(intercept)) if slope.is_finite() && intercept.is_finite() => {
            Ok((slope, intercept))
        }
        _ => Err(value_err(format!(
            "the {demand} demand needs a finite slope and a finite intercept"
        ))),
    }
}

/// Score `demand` (`affine_posterior`, `affine_mean`, `nonlinear` or `risk`) from the published
/// point answer `mean` and, when the answer is a posterior, its `posterior_mean`.
#[pyfunction]
#[pyo3(signature = (demand, mean, posterior_mean=None, slope=None, intercept=None))]
fn score_published_decision(
    demand: &str,
    mean: f64,
    posterior_mean: Option<f64>,
    slope: Option<f64>,
    intercept: Option<f64>,
) -> PyResult<Scored> {
    let demand = match demand {
        "affine_posterior" => {
            let (slope, intercept) = affine(slope, intercept, "affine_posterior")?;
            DecisionDemand::AffinePosterior { slope, intercept }
        }
        "affine_mean" => {
            let (slope, intercept) = affine(slope, intercept, "affine_mean")?;
            DecisionDemand::AffineMean { slope, intercept }
        }
        "nonlinear" => DecisionDemand::Nonlinear,
        "risk" => DecisionDemand::Risk,
        other => {
            return Err(value_err(format!(
                "unknown decision demand {other:?}; use affine_posterior, affine_mean, \
                 nonlinear or risk"
            )));
        }
    };
    match score_decision(demand, PublishedCarriers { mean, posterior_mean }) {
        Ok(scored) => Ok((Some((scored.value, scored.carrier.as_str().to_owned())), None)),
        Err(missing) => Ok((None, Some(missing.as_str().to_owned()))),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(score_published_decision, module)?)?;
    Ok(())
}
