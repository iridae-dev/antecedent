//! Analytic univariate intervention laws, conditional on composed mechanism parameters.
//! Outcome variance is retained; no parameter posterior or calibrated sampling interval is implied.

#![allow(clippy::float_cmp, reason = "exact zeros select polynomial degree and degenerate laws")]

use crate::decision_contract::{DecisionContract, DecisionFunctional, Tail, UtilityExpr};
use crate::decision_eval::{DecisionEvalError, SourceReceipt};
use antecedent_contract::ScientificQuantity;

/// A univariate Gaussian outcome law with its producing snapshot.
/// It supplies no cross-action or cross-outcome coupling.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaussianForwardLaw {
    /// Analytic evaluator and wire version; this reader accepts only one.
    pub version: u16,
    /// Parameter uncertainty standing; must be `not_propagated`.
    pub parameter_uncertainty: String,
    /// Outcome coordinate, never a mean-contrast coordinate.
    #[serde(with = "quantity_serde")]
    pub coordinate: ScientificQuantity,
    /// Conditional outcome mean.
    pub mean: f64,
    /// Conditional outcome variance, including residual mechanism noise.
    pub variance: f64,
    /// Identity of the exact compiled composition query and result.
    pub query_identity: String,
    /// Provider identity.
    pub provider_id: String,
    /// Verified producing premises.
    pub snapshot_id: String,
    /// Causal contract identity.
    pub causal_contract_id: String,
}

impl GaussianForwardLaw {
    /// Validate a law before use or replay. No parameter uncertainty is inferred.
    #[must_use]
    pub fn validate(&self) -> bool {
        self.version == 1
            && self.parameter_uncertainty == "not_propagated"
            && self.mean.is_finite()
            && self.variance.is_finite()
            && self.variance >= 0.0
            && self.coordinate.functional_id == "outcome"
            && self.query_identity.len() == 64
            && self.query_identity.bytes().all(|b| b.is_ascii_hexdigit())
            && [&self.provider_id, &self.snapshot_id, &self.causal_contract_id]
                .iter()
                .all(|s| !s.trim().is_empty())
    }

    pub(crate) fn receipt(&self) -> SourceReceipt {
        SourceReceipt {
            provider_id: self.provider_id.clone(),
            snapshot_id: self.snapshot_id.clone(),
            causal_contract_id: self.causal_contract_id.clone(),
            rng_id: "none:analytic_gaussian_v1".into(),
        }
    }

    /// Evaluate an affine probability/quantile or a quadratic expected utility analytically.
    /// Higher-degree and piecewise utilities and joint inputs refuse; they require a different
    /// law evaluator. Standard error is unavailable: outcome variance is not estimator error.
    ///
    /// # Errors
    /// Invalid law, missing coordinate, unsupported expression, or non-finite arithmetic.
    pub fn evaluate(
        &self,
        contract: &DecisionContract,
        action_id: &str,
        functional: DecisionFunctional,
    ) -> Result<(f64, Option<f64>), DecisionEvalError> {
        contract.validate().map_err(DecisionEvalError::Contract)?;
        let unsupported = || {
            DecisionEvalError::AnalyticLawUnsupported(
                "analytic_gaussian_v1 requires one outcome input, a quadratic expectation or an affine risk functional",
            )
        };
        if !self.validate() {
            return Err(unsupported());
        }
        let action = contract.actions.iter().find(|a| a.id == action_id).ok_or_else(unsupported)?;
        if action.inputs.len() != 1 {
            return Err(unsupported());
        }
        if self.coordinate.require_same_coordinate(&action.inputs[0]).is_err() {
            return Err(DecisionEvalError::QuantityNotFound { action: action_id.into(), input: 0 });
        }
        let [c, b, a] = polynomial(&action.utility).ok_or_else(unsupported)?;
        let mean = self.mean;
        let var = self.variance;
        let value = match functional {
            DecisionFunctional::Expectation | DecisionFunctional::ExpectedUtility => {
                (a * mean + b).mul_add(mean, c) + a * var
            }
            DecisionFunctional::Variance => {
                (b + 2.0 * a * mean).powi(2) * var + 2.0 * a * a * var * var
            }
            _ if a != 0.0 => return Err(unsupported()),
            DecisionFunctional::Probability { threshold, tail } if threshold.is_finite() => {
                let mu = c + b * mean;
                let sd = b.abs() * var.sqrt();
                if !mu.is_finite() || !sd.is_finite() || (sd == 0.0 && b != 0.0 && var > 0.0) {
                    return Err(DecisionEvalError::NonFiniteUtility { action: action_id.into() });
                }
                if sd == 0.0 {
                    f64::from(match tail {
                        Tail::Lower => mu <= threshold,
                        Tail::Upper => mu >= threshold,
                    })
                } else {
                    let z = (threshold - mu) / sd;
                    match tail {
                        Tail::Lower => antecedent_kernels::norm_cdf(z),
                        Tail::Upper => antecedent_kernels::norm_sf(z),
                    }
                }
            }
            DecisionFunctional::Quantile { p } if p > 0.0 && p < 1.0 => {
                c + b * mean + b.abs() * var.sqrt() * antecedent_kernels::norm_inv(p)
            }
            _ => return Err(unsupported()),
        };
        if !value.is_finite() {
            return Err(DecisionEvalError::NonFiniteUtility { action: action_id.into() });
        }
        Ok((value, None))
    }
}

fn polynomial(expr: &UtilityExpr) -> Option<[f64; 3]> {
    use UtilityExpr as U;
    let out = match expr {
        U::Const(c) => [*c, 0.0, 0.0],
        U::Input(0) => [0.0, 1.0, 0.0],
        U::Neg(a) => polynomial(a)?.map(|x| -x),
        U::Add(a, b) | U::Sub(a, b) => {
            let a = polynomial(a)?;
            let b = polynomial(b)?;
            let sign = if matches!(expr, U::Add(..)) { 1.0 } else { -1.0 };
            std::array::from_fn(|i| a[i] + sign * b[i])
        }
        U::Mul(a, b) => {
            let a = polynomial(a)?;
            let b = polynomial(b)?;
            if (a[2] != 0.0 && (b[1] != 0.0 || b[2] != 0.0))
                || (b[2] != 0.0 && (a[1] != 0.0 || a[2] != 0.0))
            {
                return None;
            }
            [a[0] * b[0], a[0] * b[1] + a[1] * b[0], a[0] * b[2] + a[1] * b[1] + a[2] * b[0]]
        }
        _ => return None,
    };
    out.iter().all(|v| v.is_finite()).then_some(out)
}

mod quantity_serde {
    use antecedent_contract::ScientificQuantity;
    use antecedent_io_core::quantity_wire::ScientificQuantityWire;
    use serde::{Deserialize, Serialize};
    pub fn serialize<S: serde::Serializer>(
        q: &ScientificQuantity,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        ScientificQuantityWire::from(q).serialize(s)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<ScientificQuantity, D::Error> {
        ScientificQuantity::try_from(ScientificQuantityWire::deserialize(d)?)
            .map_err(serde::de::Error::custom)
    }
}
