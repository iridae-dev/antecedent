//! Selectable mechanism families for the panel fit.
//!
//! The reviewed family is a three-period simplex adstock, one Michaelis-Menten saturation
//! with a single half-saturation shared by every channel and group, and independent
//! Gaussian group intercepts. Each broader family changes only the numerical mechanism
//! or residual law; none carries an identification claim. Whether a fitted surface supports
//! a causal contrast remains the checked compiler's decision, under its own design proof
//! (for example a known randomized assignment).
//!
//! Inference follows the family. Frequentist uncertainty is the group-cluster sandwich over
//! independent groups, or, when a spatial residual law couples the groups, the inverse
//! observed information of the full spatially dependent likelihood (a cluster sandwich is
//! not valid there), plus conditional random-effect draws; the parametric bootstrap repeats
//! the complete fit, including temporal change selection. The Bayesian realization is a
//! posterior under explicit priors on every added coordinate, including the logit and log
//! Jacobians of the transformed parameters.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::PanelError;
use serde::{Deserialize, Serialize};

/// Largest group count for which the dense spatial intercept covariance is factored.
pub const MAX_SPATIAL_GROUPS: usize = 256;
/// Diagonal regularization added to the spatial correlation so a smooth kernel stays factorable.
pub(super) const SPATIAL_JITTER: f64 = 1e-9;

/// Carryover weights over the current and two earlier periods.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PanelAdstock {
    /// Free weights on the three-period simplex for every channel (reviewed family).
    #[default]
    Simplex,
    /// Geometric decay `w_l proportional to theta^l` for l = 0..=2 with one decay
    /// `theta` in (0,1) per channel, normalized over the observable window.
    Geometric,
}

/// Saturation of the lag-weighted exposure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PanelSaturation {
    /// `x / (x + k)` (reviewed family).
    #[default]
    MichaelisMenten,
    /// `x^s / (x^s + k^s)` with one free positive shape `s` per channel.
    Hill,
}

/// Which groups share a half-saturation.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PanelSaturationScope {
    /// One half-saturation for every group (reviewed family).
    #[default]
    Shared,
    /// A separate half-saturation for each class of groups. `assignment[g]` is the class
    /// of the g-th group in the data's group order; classes are `0..m`, each used.
    Classes {
        /// Class of every group, in group order.
        assignment: Vec<usize>,
    },
}

/// Correlation decay of the spatial group-intercept kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PanelSpatialKernel {
    /// `exp(-d / range)`.
    #[default]
    Exponential,
    /// `exp(-(d / range)^2)`.
    Gaussian,
}

/// Spatially correlated group intercepts: `Cov(u) = group_sd^2 * K(range)`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelSpatial {
    /// Kernel family; its range is fitted.
    pub kernel: PanelSpatialKernel,
    /// Group identities in the data's group order; alignment is by identity, not position.
    pub groups: Vec<String>,
    /// Row-major symmetric nonnegative distance matrix with a zero diagonal.
    pub distances: Vec<f64>,
}

/// The selected mechanism family; the default reproduces the reviewed family exactly.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelFamily {
    /// Carryover shape.
    #[serde(default)]
    pub adstock: PanelAdstock,
    /// Saturation shape.
    #[serde(default)]
    pub saturation: PanelSaturation,
    /// Sharing of the half-saturation across groups.
    #[serde(default)]
    pub scope: PanelSaturationScope,
    /// Optional spatial residual structure on the group intercepts.
    #[serde(default)]
    pub spatial: Option<PanelSpatial>,
}

impl PanelFamily {
    /// Whether this is exactly the reviewed family (serialized as absent).
    #[must_use]
    pub fn is_reviewed(&self) -> bool {
        *self == Self::default()
    }
    /// A separate half-saturation for each of `groups` groups.
    #[must_use]
    pub fn per_group_saturation(groups: usize) -> PanelSaturationScope {
        PanelSaturationScope::Classes { assignment: (0..groups).collect() }
    }
    /// Number of distinct half-saturation classes.
    #[must_use]
    pub fn classes(&self) -> usize {
        match &self.scope {
            PanelSaturationScope::Shared => 1,
            PanelSaturationScope::Classes { assignment } => {
                assignment.iter().copied().max().map_or(1, |max| max + 1)
            }
        }
    }
    /// Class of a group by position.
    #[must_use]
    pub fn class_of(&self, group: usize) -> usize {
        match &self.scope {
            PanelSaturationScope::Shared => 0,
            PanelSaturationScope::Classes { assignment } => assignment[group],
        }
    }
    pub(super) fn is_hill(&self) -> bool {
        matches!(self.saturation, PanelSaturation::Hill)
    }
    pub(super) fn is_geometric(&self) -> bool {
        matches!(self.adstock, PanelAdstock::Geometric)
    }
    /// Check the family against the declared group axis and carryover order.
    ///
    /// # Errors
    /// Geometric adstock without carryover, malformed class assignment, or a spatial
    /// structure that is not aligned by identity to the groups.
    pub(super) fn validate(&self, group_ids: &[&str], lag_order: usize) -> Result<(), PanelError> {
        let invalid = |message: &str| PanelError::Invalid(message.into());
        if self.is_geometric() && lag_order != 2 {
            return Err(invalid("geometric adstock requires carryover lag order two"));
        }
        if let PanelSaturationScope::Classes { assignment } = &self.scope {
            let classes = self.classes();
            let mut used = vec![false; classes];
            for &class in assignment {
                used[class] = true;
            }
            if assignment.len() != group_ids.len() || used.iter().any(|used| !used) {
                return Err(invalid(
                    "saturation classes must assign every group and use classes 0..m contiguously",
                ));
            }
        }
        if let Some(spatial) = &self.spatial {
            let groups = group_ids.len();
            let scale = spatial.distances.iter().copied().fold(0.0_f64, f64::max);
            if groups > MAX_SPATIAL_GROUPS
                || spatial.groups.len() != groups
                || spatial.groups.iter().map(String::as_str).ne(group_ids.iter().copied())
                || spatial.distances.len() != groups * groups
                || spatial.distances.iter().any(|d| !d.is_finite() || *d < 0.0)
                || (0..groups).any(|i| {
                    spatial.distances[i * groups + i] != 0.0
                        || (0..i).any(|j| {
                            (spatial.distances[i * groups + j] - spatial.distances[j * groups + i])
                                .abs()
                                > 1e-12 * scale.max(1.0)
                        })
                })
                || scale <= 0.0
            {
                return Err(invalid(
                    "spatial structure needs identity-aligned groups (at most 256), a symmetric \
                     nonnegative distance matrix with zero diagonal and a positive distance",
                ));
            }
        }
        Ok(())
    }
    /// Median of the positive pairwise distances, or one when there is no spatial structure.
    pub(super) fn median_distance(&self) -> f64 {
        let Some(spatial) = &self.spatial else {
            return 1.0;
        };
        let mut positive: Vec<f64> =
            spatial.distances.iter().copied().filter(|d| *d > 0.0).collect();
        positive.sort_by(f64::total_cmp);
        positive.get(positive.len() / 2).copied().unwrap_or(1.0)
    }
}

/// Family-specific mechanism parameters; all empty or zero for the reviewed family.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelFamilyParameters {
    /// Hill shape per channel in channel order; empty for Michaelis-Menten.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hill_shape: Vec<f64>,
    /// Half-saturations of classes `1..m` (class zero is `PanelParameters::half_saturation`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub class_half_saturation: Vec<f64>,
    /// Spatial kernel range; zero when there is no spatial structure.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub spatial_range: f64,
}
#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if requires a borrowed field"
)]
fn is_zero(value: &f64) -> bool {
    *value == 0.0
}
impl PanelFamilyParameters {
    /// Whether every family-specific parameter is absent.
    #[must_use]
    pub fn is_reviewed(&self) -> bool {
        *self == Self::default()
    }
}

/// Priors on the family-specific coordinates; ignored by coordinates the family lacks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelFamilyPrior {
    /// Beta(a, b) prior on every geometric decay.
    pub geometric_decay: [f64; 2],
    /// Mean of log(Hill shape); zero centers on the Michaelis-Menten shape.
    pub log_hill_shape_mean: f64,
    /// Standard deviation of log(Hill shape).
    pub log_hill_shape_sd: f64,
    /// Standard deviation of `log k_class - log k_0` (partial pooling across classes).
    pub class_log_saturation_sd: f64,
    /// Mean of log(spatial range); `None` centers on the median positive distance.
    pub log_spatial_range_mean: Option<f64>,
    /// Standard deviation of log(spatial range).
    pub log_spatial_range_sd: f64,
}
impl Default for PanelFamilyPrior {
    fn default() -> Self {
        Self {
            geometric_decay: [1.0, 1.0],
            log_hill_shape_mean: 0.0,
            log_hill_shape_sd: 0.5,
            class_log_saturation_sd: 0.5,
            log_spatial_range_mean: None,
            log_spatial_range_sd: 1.0,
        }
    }
}
impl PanelFamilyPrior {
    /// Whether every entry is the default.
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
    pub(super) fn validate(&self) -> Result<(), PanelError> {
        let finite_positive = self
            .geometric_decay
            .iter()
            .chain([
                &self.log_hill_shape_sd,
                &self.class_log_saturation_sd,
                &self.log_spatial_range_sd,
            ])
            .all(|value| value.is_finite() && *value > 0.0);
        if !finite_positive
            || !self.log_hill_shape_mean.is_finite()
            || self.log_spatial_range_mean.is_some_and(|value| !value.is_finite())
        {
            return Err(PanelError::Invalid(
                "positive finite Beta, Hill-shape, class-saturation and spatial-range prior parameters"
                    .into(),
            ));
        }
        Ok(())
    }
}
