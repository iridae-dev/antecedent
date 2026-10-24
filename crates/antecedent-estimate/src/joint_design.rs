//! Design-native joint response functionals. Joint arms label complete programs;
//! a marginal treatment range never supplies joint support.
// SPDX-License-Identifier: MIT OR Apache-2.0

pub mod capabilities;
pub mod compiled;
pub mod gaussian_longitudinal;
pub mod instrumented;
pub mod longitudinal;
pub mod local_instrument;
pub mod randomized;
pub mod rollout;
pub mod switchback;
pub mod synthetic;

/// Meaning of a joint response's uncertainty, separate from causal identification.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum JointUncertainty {
    /// Joint independent-unit sandwich; asymptotic and not release calibrated.
    IndependentUnitSandwich,
    /// Entire independent episodes deleted and the complete functional re-estimated.
    DeleteEpisodeJackknife,
    /// Complete independent-episode resampling refits every selected mechanism and support model.
    FullProcedureBootstrap,
    /// Explicit posterior model, whose assumptions are retained verbatim.
    Posterior {
        /// Statistical model and prior.
        model: String,
    },
}

/// Common joint-world estimate; covariance is in world order and survives contrast assembly.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JointDesignEstimate {
    /// Stable complete-program world keys.
    pub worlds: Vec<String>,
    /// One identified functional estimate per world.
    pub means: Vec<f64>,
    /// Row-major joint covariance, not separately simulated marginal intervals.
    pub covariance: Vec<f64>,
    /// Independent sampling unit.
    pub sampling_unit: String,
    /// Exact statistical uncertainty meaning.
    pub uncertainty: JointUncertainty,
}
impl JointDesignEstimate {
    /// Assemble any prespecified main, interaction or rollout contrast using joint covariance.
    ///
    /// # Errors
    /// Wrong dimensions, invalid coefficients or a nonfinite result.
    pub fn contrast(&self, coefficients: &[f64]) -> Result<(f64, f64), String> {
        let k = self.means.len();
        if coefficients.len() != k
            || self.covariance.len() != k * k
            || coefficients.iter().any(|x| !x.is_finite())
        {
            return Err("joint contrast has invalid dimensions or coefficients".into());
        }
        let mean = self.means.iter().zip(coefficients).map(|(m, c)| m * c).sum::<f64>();
        let variance = (0..k)
            .flat_map(|j| (0..k).map(move |l| (j, l)))
            .map(|(j, l)| coefficients[j] * coefficients[l] * self.covariance[j * k + l])
            .sum::<f64>();
        if !mean.is_finite() || !variance.is_finite() || variance < -1e-10 {
            return Err("joint contrast overflow or invalid covariance".into());
        }
        Ok((mean, variance.max(0.0)))
    }
}

/// Aligned world-functional draws from an explicitly retained likelihood and prior.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JointDesignPosterior {
    /// Posterior moments, with exact model meaning in the uncertainty field.
    pub estimate: JointDesignEstimate,
    /// Realization-major joint world expectations; all worlds share each mechanism draw.
    pub draws: Vec<Vec<f64>>,
}
impl JointDesignPosterior {
    /// Summarize aligned mechanism draws without treating Monte Carlo error as sampling uncertainty.
    /// # Errors
    /// Too few draws, invalid axes, nonfinite draws or missing likelihood/prior declaration.
    pub fn from_draws(
        worlds: Vec<String>,
        draws: Vec<Vec<f64>>,
        sampling_unit: String,
        model: String,
    ) -> Result<Self, String> {
        let n = draws.len();
        let k = worlds.len();
        if n < 2
            || k == 0
            || model.trim().is_empty()
            || draws.iter().any(|d| d.len() != k || d.iter().any(|v| !v.is_finite()))
        {
            return Err("invalid aligned Bayesian mechanism draws or model declaration".into());
        }
        let means =
            (0..k).map(|j| draws.iter().map(|d| d[j]).sum::<f64>() / n as f64).collect::<Vec<_>>();
        let mut covariance = vec![0.0; k * k];
        for d in &draws {
            for j in 0..k {
                for l in 0..k {
                    covariance[j * k + l] += (d[j] - means[j]) * (d[l] - means[l]) / (n - 1) as f64;
                }
            }
        }
        Ok(Self {
            estimate: JointDesignEstimate {
                worlds,
                means,
                covariance,
                sampling_unit,
                uncertainty: JointUncertainty::Posterior { model },
            },
            draws,
        })
    }
}
