//! Categorical treatment declarations and their wire forms.

use crate::vector_treatment_artifact::AdjustmentColumnRequest;
use antecedent_estimate::categorical_treatment::{LevelScale, MonotonicityDirection};
use antecedent_estimate::vector_treatment::VectorCovariance;
use serde::{Deserialize, Serialize};

/// The declared design of the categorical regime on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CategoricalSpecWire {
    /// Declared level set (the scale, when ordered).
    pub declared_levels: Vec<String>,
    /// Whether the levels are ordered.
    pub scale: LevelScaleWire,
    /// Reference level.
    pub reference: String,
    /// Fewest rows a declared level may have.
    pub min_level_rows: u64,
    /// Requested pairwise contrasts `(from, to)`.
    pub pairwise: Vec<(String, String)>,
    /// Declared monotonicity direction, when a monotonicity test is requested.
    pub monotonicity: Option<MonotonicityDirectionWire>,
    /// Covariance estimator.
    pub covariance: CovarianceKindWire,
}

/// Level scale of a categorical treatment on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LevelScaleWire {
    /// Levels are ordered as declared.
    Ordered,
    /// Levels are unordered.
    Unordered,
}

impl LevelScaleWire {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ordered => "ordered",
            Self::Unordered => "unordered",
        }
    }
}

impl From<LevelScale> for LevelScaleWire {
    fn from(scale: LevelScale) -> Self {
        match scale {
            LevelScale::Ordered => Self::Ordered,
            LevelScale::Unordered => Self::Unordered,
        }
    }
}

impl From<LevelScaleWire> for LevelScale {
    fn from(scale: LevelScaleWire) -> Self {
        match scale {
            LevelScaleWire::Ordered => Self::Ordered,
            LevelScaleWire::Unordered => Self::Unordered,
        }
    }
}

impl PartialEq<&str> for LevelScaleWire {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// Direction of a declared monotonicity test on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MonotonicityDirectionWire {
    /// Effects do not decrease with the level order.
    NonDecreasing,
    /// Effects do not increase with the level order.
    NonIncreasing,
}

impl MonotonicityDirectionWire {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NonDecreasing => "non_decreasing",
            Self::NonIncreasing => "non_increasing",
        }
    }
}

impl From<MonotonicityDirection> for MonotonicityDirectionWire {
    fn from(direction: MonotonicityDirection) -> Self {
        match direction {
            MonotonicityDirection::NonDecreasing => Self::NonDecreasing,
            MonotonicityDirection::NonIncreasing => Self::NonIncreasing,
        }
    }
}

impl From<MonotonicityDirectionWire> for MonotonicityDirection {
    fn from(direction: MonotonicityDirectionWire) -> Self {
        match direction {
            MonotonicityDirectionWire::NonDecreasing => Self::NonDecreasing,
            MonotonicityDirectionWire::NonIncreasing => Self::NonIncreasing,
        }
    }
}

impl PartialEq<&str> for MonotonicityDirectionWire {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// Covariance estimator of a categorical treatment fit on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CovarianceKindWire {
    /// Model-based covariance.
    ModelBased,
    /// Heteroskedasticity-consistent HC0.
    Hc0,
    /// HC1.
    Hc1,
    /// HC2.
    Hc2,
    /// HC3.
    Hc3,
    /// One-way cluster-robust; needs a `cluster` id per row.
    ClusterRobust,
    /// Newey-West (Bartlett) HAC; needs a `serial_lag` in the design.
    NeweyWest,
}

impl CovarianceKindWire {
    /// Stable `snake_case` name, the one [`VectorCovariance::as_str`] gives.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModelBased => "model_based",
            Self::Hc0 => "hc0",
            Self::Hc1 => "hc1",
            Self::Hc2 => "hc2",
            Self::Hc3 => "hc3",
            Self::ClusterRobust => "cluster_robust",
            Self::NeweyWest => "newey_west",
        }
    }
}

impl From<VectorCovariance> for CovarianceKindWire {
    fn from(kind: VectorCovariance) -> Self {
        match kind {
            VectorCovariance::ModelBased => Self::ModelBased,
            VectorCovariance::Hc0 => Self::Hc0,
            VectorCovariance::Hc1 => Self::Hc1,
            VectorCovariance::Hc2 => Self::Hc2,
            VectorCovariance::Hc3 => Self::Hc3,
            VectorCovariance::Cluster => Self::ClusterRobust,
            VectorCovariance::NeweyWest => Self::NeweyWest,
        }
    }
}

impl From<CovarianceKindWire> for VectorCovariance {
    fn from(kind: CovarianceKindWire) -> Self {
        match kind {
            CovarianceKindWire::ModelBased => Self::ModelBased,
            CovarianceKindWire::Hc0 => Self::Hc0,
            CovarianceKindWire::Hc1 => Self::Hc1,
            CovarianceKindWire::Hc2 => Self::Hc2,
            CovarianceKindWire::Hc3 => Self::Hc3,
            CovarianceKindWire::ClusterRobust => Self::Cluster,
            CovarianceKindWire::NeweyWest => Self::NeweyWest,
        }
    }
}

impl PartialEq<&str> for CovarianceKindWire {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// What a producer declares: rows with level labels, the shared adjustment block and the design.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalTreatmentRequest {
    /// Outcome, one value per row.
    pub outcome: Vec<f64>,
    /// Identity of the row snapshot.
    pub row_snapshot: String,
    /// Shared adjustment columns, in design order.
    pub adjustment: Vec<AdjustmentColumnRequest>,
    /// Level label of each row.
    pub levels: Vec<String>,
    /// The declared design.
    pub spec: CategoricalSpecWire,
}

/// Declarations of a categorical request beside the plain [`CategoricalTreatmentRequest`]:
/// sampling weights and a cluster id, each one entry per row, and the Newey-West lag.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CategoricalRowsRequest {
    /// Sampling weights (finite, non-negative, at least one positive). Rows of weight zero are
    /// deleted; the rest are normalized to mean one. Need a robust covariance.
    pub sampling_weights: Option<Vec<f64>>,
    /// Cluster id of each row; read by, and required for, the `cluster_robust` covariance.
    pub cluster: Option<Vec<u32>>,
    /// Newey-West (Bartlett) lag for rows in time order at unit spacing; read by, and required
    /// for, the `newey_west` covariance.
    pub serial_lag: Option<u64>,
}

impl CategoricalRowsRequest {
    /// Whether nothing is declared.
    #[must_use]
    pub fn is_plain(&self) -> bool {
        self.sampling_weights.is_none() && self.cluster.is_none() && self.serial_lag.is_none()
    }
}

/// What the declared rows did, stored beside the unchanged result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredRowsWire {
    /// Format marker of this section.
    pub format: String,
    /// Whether sampling weights were declared.
    pub weights_declared: bool,
    /// Rows used after zero-weight rows were deleted.
    pub rows_used: u64,
    /// Rows of weight zero that were deleted.
    pub zero_weight_rows: u64,
    /// Kish effective size of the weights of the rows used, when weights were declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_effective_sample_size: Option<f64>,
    /// Sum of the declared weights of the rows used before normalization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_sum: Option<f64>,
    /// Clusters among the rows used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clusters: Option<u64>,
    /// Newey-West lag.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serial_lag: Option<u64>,
}

/// The format marker of [`DeclaredRowsWire`].
pub const CATEGORICAL_DECLARED_ROWS_FORMAT: &str = "categorical_declared_rows_v1";
