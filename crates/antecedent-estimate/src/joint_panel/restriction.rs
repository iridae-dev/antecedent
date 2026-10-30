// SPDX-License-Identifier: MIT OR Apache-2.0
//! Checks of the declared Gaussian restriction on a fitted panel mechanism.
//!
//! The panel mechanism is a mean (saturated lag-weighted channel exposures, their declared
//! products, a covariate and a season effect) plus a Gaussian group intercept and stationary
//! Gaussian AR(1) errors. A causal statement about the fitted panel holds only if the observed
//! conditional laws are of that form. This module checks the two parts of that form the data can
//! speak to, on the fitted point mechanism:
//!
//! - **Normality** of the AR(1)-whitened innovations, by the Jarque-Bera statistic. The
//!   innovations are `e_t = (r_t - rho r_{t-1}) / sigma` (the first analysed period of a group
//!   `e = r sqrt(1 - rho^2) / sigma`), with `r` the outcome minus the fitted mean minus the
//!   group's conditional intercept. Under the mechanism they are independent standard
//!   normals, so `JB = n (S^2 / 6 + K^2 / 24)` is asymptotically chi-square with two degrees
//!   of freedom, whose survival function is exactly `exp(-JB / 2)`.
//! - **Linearity** of the mean in each fitted feature: the whitened innovations are cut into
//!   equal-count bins of each channel's lag-weighted exposure and of the covariate; under the
//!   mechanism every bin mean is zero with variance `1 / n_bin`, so `sum_b n_b mean_b^2` is
//!   (conservatively, because parameters were fitted) chi-square with the number of bins
//!   degrees of freedom. A mean the saturation family cannot represent leaves a systematic
//!   pattern across bins.
//!
//! Both are diagnostics at a stated level with a Bonferroni split over the tests run; failing
//! to flag is not proof that the restriction holds, and no calibration is measured here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::likelihood::{CompiledPanel, exposure, observed_response};
use super::{PanelDraw, PanelError, PanelData, invalid};

/// Fewest whitened innovations for which a normality check is reported.
pub const MIN_DIAGNOSTIC_INNOVATIONS: usize = 30;
/// Equal-count bins per feature in the linearity check.
pub const LINEARITY_BINS: usize = 5;

/// Normality of the whitened innovations.
#[derive(Clone, Debug, PartialEq)]
pub struct PanelNormalityCheck {
    /// Whitened innovations used.
    pub count: usize,
    /// Sample skewness.
    pub skewness: f64,
    /// Sample excess kurtosis.
    pub excess_kurtosis: f64,
    /// Jarque-Bera statistic.
    pub jarque_bera: f64,
    /// Asymptotic p-value `exp(-JB / 2)`.
    pub p_value: f64,
}

/// Lack of fit of the mean against one feature.
#[derive(Clone, Debug, PartialEq)]
pub struct PanelLinearityCheck {
    /// Feature name: `channel:<id>` or `covariate`.
    pub feature: String,
    /// Bins used.
    pub bins: usize,
    /// `sum_b n_b mean_b^2`.
    pub statistic: f64,
    /// Chi-square survival with `bins` degrees of freedom.
    pub p_value: f64,
}

/// Checks of the declared restriction on one fitted point mechanism.
#[derive(Clone, Debug, PartialEq)]
pub struct PanelRestrictionDiagnostics {
    /// Family-wise level split evenly over the tests actually run.
    pub level: f64,
    /// Normality check, absent when fewer than [`MIN_DIAGNOSTIC_INNOVATIONS`] innovations exist.
    pub normality: Option<PanelNormalityCheck>,
    /// One linearity check per channel and one for the covariate.
    pub linearity: Vec<PanelLinearityCheck>,
}

impl PanelRestrictionDiagnostics {
    fn tests(&self) -> usize {
        usize::from(self.normality.is_some()) + self.linearity.len()
    }
    /// Whether the innovations contradict normality at the per-test level.
    #[must_use]
    pub fn normality_flagged(&self) -> bool {
        let per_test = self.level / self.tests().max(1) as f64;
        self.normality.as_ref().is_some_and(|check| check.p_value < per_test)
    }
    /// Names of the features whose mean the fitted mechanism does not represent.
    #[must_use]
    pub fn linearity_flags(&self) -> Vec<&str> {
        let per_test = self.level / self.tests().max(1) as f64;
        self.linearity
            .iter()
            .filter(|check| check.p_value < per_test)
            .map(|check| check.feature.as_str())
            .collect()
    }
    /// Whether any check flags the restriction.
    #[must_use]
    pub fn flagged(&self) -> bool {
        self.normality_flagged() || !self.linearity_flags().is_empty()
    }
}

/// Check the declared Gaussian restriction on a fitted point mechanism.
///
/// # Errors
/// A non-positive or non-finite level, a mechanism that does not match the panel, or a fitted
/// parameter outside its domain.
pub fn panel_restriction_diagnostics(
    data: &PanelData,
    point: &PanelDraw,
    level: f64,
) -> Result<PanelRestrictionDiagnostics, PanelError> {
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(invalid("diagnostic level must lie in (0, 1)"));
    }
    let compiled = CompiledPanel::new(data)?;
    let p = &point.parameters;
    if point.group_intercepts.len() != data.groups.len()
        || p.channels.len() != data.channels.len()
        || !(p.innovation_sd.is_finite() && p.innovation_sd > 0.0)
        || !(p.rho.is_finite() && p.rho.abs() < 1.0)
    {
        return Err(invalid("fitted mechanism does not match the panel"));
    }
    let scale = (1.0 - p.rho * p.rho).sqrt();
    let features = p.channels.len() + 1;
    // (innovation, [feature values]) over every analysed period.
    let mut rows: Vec<(f64, Vec<f64>)> = Vec::new();
    for (index, group) in data.groups.iter().enumerate() {
        let mut previous: Option<f64> = None;
        for time in 0..group.outcome.len() {
            let outcome = group.outcome[time];
            if !outcome.is_finite() {
                previous = None;
                continue;
            }
            let fitted = observed_response(&compiled, p, group, index, time);
            let residual = outcome - fitted - point.group_intercepts[index];
            if !residual.is_finite() {
                return Err(invalid("non-finite residual at an analysed period"));
            }
            let innovation = match previous {
                Some(before) => (residual - p.rho * before) / p.innovation_sd,
                None => residual * scale / p.innovation_sd,
            };
            previous = Some(residual);
            let mut covariates = Vec::with_capacity(features);
            for ((values, history), parameters) in
                group.channels.iter().zip(&group.history).zip(&p.channels)
            {
                covariates.push(exposure(values, *history, parameters.lag, time));
            }
            covariates.push(group.confounder[time]);
            rows.push((innovation, covariates));
        }
    }
    let count = rows.len();
    let normality = (count >= MIN_DIAGNOSTIC_INNOVATIONS).then(|| {
        let n = count as f64;
        let mean = rows.iter().map(|row| row.0).sum::<f64>() / n;
        let central = |power: i32| rows.iter().map(|row| (row.0 - mean).powi(power)).sum::<f64>() / n;
        let (m2, m3, m4) = (central(2), central(3), central(4));
        let skewness = m3 / m2.powf(1.5);
        let excess_kurtosis = m4 / (m2 * m2) - 3.0;
        let jarque_bera = n * (skewness * skewness / 6.0 + excess_kurtosis * excess_kurtosis / 24.0);
        PanelNormalityCheck {
            count,
            skewness,
            excess_kurtosis,
            jarque_bera,
            p_value: (-jarque_bera / 2.0).exp(),
        }
    });
    let mut linearity = Vec::new();
    if count >= LINEARITY_BINS * 4 {
        for feature in 0..features {
            let mut order: Vec<usize> = (0..count).collect();
            order.sort_by(|&a, &b| rows[a].1[feature].total_cmp(&rows[b].1[feature]));
            let mut statistic = 0.0;
            for bin in 0..LINEARITY_BINS {
                let members = &order[bin * count / LINEARITY_BINS..(bin + 1) * count / LINEARITY_BINS];
                let sum = members.iter().map(|&row| rows[row].0).sum::<f64>();
                statistic += sum * sum / members.len() as f64;
            }
            linearity.push(PanelLinearityCheck {
                feature: if feature < p.channels.len() {
                    format!("channel:{}", p.channels[feature].id)
                } else {
                    "covariate".into()
                },
                bins: LINEARITY_BINS,
                statistic,
                p_value: antecedent_stats::chi_square_sf(statistic, LINEARITY_BINS),
            });
        }
    }
    Ok(PanelRestrictionDiagnostics { level, normality, linearity })
}
