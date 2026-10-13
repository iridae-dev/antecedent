//! Closed model-declaration codes stored as the historical `snake_case` strings.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::joint_bayesian_transport::{
    SourceDependence, SourceSharing, TransportGraphClass, VaryingBlock,
};
use serde::{Deserialize, Serialize};

/// Graph class on a joint or learned transport declaration.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GraphClassCode {
    /// `fixed_dag`.
    FixedDag,
    /// `admg`.
    Admg,
    /// `graph_posterior`.
    GraphPosterior,
}

/// Varying block on a joint or learned transport declaration.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VaryingBlockCode {
    /// `intercept`.
    Intercept,
    /// `intercept_and_covariates`.
    InterceptAndCovariates,
}

/// Source sharing on a joint or learned transport declaration.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceSharingCode {
    /// `independent_varying_blocks`.
    IndependentVaryingBlocks,
    /// `shared_varying_block`.
    SharedVaryingBlock,
}

/// Source dependence on a joint or learned transport declaration.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceDependenceCode {
    /// `independent_samples`.
    IndependentSamples,
    /// `overlapping_units`.
    OverlappingUnits,
    /// `unknown`.
    Unknown,
}

impl GraphClassCode {
    /// Historical wire text.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FixedDag => "fixed_dag",
            Self::Admg => "admg",
            Self::GraphPosterior => "graph_posterior",
        }
    }

    /// From the engine class.
    #[must_use]
    pub const fn from_class(class: TransportGraphClass) -> Self {
        match class {
            TransportGraphClass::FixedDag => Self::FixedDag,
            TransportGraphClass::Admg => Self::Admg,
            TransportGraphClass::GraphPosterior => Self::GraphPosterior,
        }
    }

    /// Engine class.
    #[must_use]
    pub const fn to_class(self) -> TransportGraphClass {
        match self {
            Self::FixedDag => TransportGraphClass::FixedDag,
            Self::Admg => TransportGraphClass::Admg,
            Self::GraphPosterior => TransportGraphClass::GraphPosterior,
        }
    }
}

impl VaryingBlockCode {
    /// Historical wire text.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Intercept => "intercept",
            Self::InterceptAndCovariates => "intercept_and_covariates",
        }
    }

    /// From the engine block.
    #[must_use]
    pub const fn from_block(block: VaryingBlock) -> Self {
        match block {
            VaryingBlock::Intercept => Self::Intercept,
            VaryingBlock::InterceptAndCovariates => Self::InterceptAndCovariates,
        }
    }

    /// Engine block.
    #[must_use]
    pub const fn to_block(self) -> VaryingBlock {
        match self {
            Self::Intercept => VaryingBlock::Intercept,
            Self::InterceptAndCovariates => VaryingBlock::InterceptAndCovariates,
        }
    }
}

impl SourceSharingCode {
    /// Historical wire text.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IndependentVaryingBlocks => "independent_varying_blocks",
            Self::SharedVaryingBlock => "shared_varying_block",
        }
    }

    /// From the engine sharing.
    #[must_use]
    pub const fn from_sharing(sharing: SourceSharing) -> Self {
        match sharing {
            SourceSharing::IndependentVaryingBlocks => Self::IndependentVaryingBlocks,
            SourceSharing::SharedVaryingBlock => Self::SharedVaryingBlock,
        }
    }

    /// Engine sharing.
    #[must_use]
    pub const fn to_sharing(self) -> SourceSharing {
        match self {
            Self::IndependentVaryingBlocks => SourceSharing::IndependentVaryingBlocks,
            Self::SharedVaryingBlock => SourceSharing::SharedVaryingBlock,
        }
    }
}

impl SourceDependenceCode {
    /// Historical wire text.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::IndependentSamples => "independent_samples",
            Self::OverlappingUnits => "overlapping_units",
            Self::Unknown => "unknown",
        }
    }

    /// From the engine dependence.
    #[must_use]
    pub const fn from_dependence(dependence: SourceDependence) -> Self {
        match dependence {
            SourceDependence::IndependentSamples => Self::IndependentSamples,
            SourceDependence::OverlappingUnits => Self::OverlappingUnits,
            SourceDependence::Unknown => Self::Unknown,
        }
    }

    /// Engine dependence.
    #[must_use]
    pub const fn to_dependence(self) -> SourceDependence {
        match self {
            Self::IndependentSamples => SourceDependence::IndependentSamples,
            Self::OverlappingUnits => SourceDependence::OverlappingUnits,
            Self::Unknown => SourceDependence::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::de::DeserializeOwned;

    fn assert_historical_text<T>(text: &str, value: &T)
    where
        T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        #[derive(Serialize)]
        struct AsText<'a> {
            code: &'a str,
        }
        #[derive(Serialize)]
        struct AsEnum<'a, T> {
            code: &'a T,
        }
        #[derive(Deserialize, Debug)]
        struct Owned<T> {
            code: T,
        }
        let text_bytes = crate::to_cbor(&AsText { code: text }).expect("text cbor");
        let enum_bytes = crate::to_cbor(&AsEnum { code: value }).expect("enum cbor");
        assert_eq!(text_bytes, enum_bytes, "{text}");
        let back: Owned<T> = crate::from_cbor(&text_bytes).expect("decode historical text");
        assert_eq!(&back.code, value);
        assert_eq!(serde_json::json!(text), serde_json::json!(value));
    }

    #[test]
    fn model_codes_keep_historical_cbor_text() {
        for value in
            [GraphClassCode::FixedDag, GraphClassCode::Admg, GraphClassCode::GraphPosterior]
        {
            assert_historical_text(value.as_str(), &value);
        }
        for value in [VaryingBlockCode::Intercept, VaryingBlockCode::InterceptAndCovariates] {
            assert_historical_text(value.as_str(), &value);
        }
        for value in
            [SourceSharingCode::IndependentVaryingBlocks, SourceSharingCode::SharedVaryingBlock]
        {
            assert_historical_text(value.as_str(), &value);
        }
        for value in [
            SourceDependenceCode::IndependentSamples,
            SourceDependenceCode::OverlappingUnits,
            SourceDependenceCode::Unknown,
        ] {
            assert_historical_text(value.as_str(), &value);
        }
    }
}
