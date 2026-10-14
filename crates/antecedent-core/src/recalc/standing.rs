//! Typed standing inputs of a recalculation workflow.
//!
//! A standing input states what a claim's uncertainty and provenance rest on beyond its data:
//! the calibration record behind an interval or posterior, the representation (model family
//! and version) an artifact was built with, the coupling between sampled streams a joint
//! statement relies on, and the verification standing of an external provider. Each input has
//! one canonical, versioned, domain-separated identity (never `Debug`, `repr` or JSON text), so a
//! change in any of them changes the effective identity of everything that reads it.
//!
//! Absence is a value: "not measured" and "no provider" digest differently from every held
//! record, so a record appearing, moving or being withdrawn is always a visible change.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Branch, StageIdentity};

/// Version word written into every standing input identity. Bumping it changes every
/// identity derived here.
pub const STANDING_INPUT_VERSION: u64 = 1;

fn tag_word(version_label: &str, variant: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(version_label.len() + variant.len() + 16);
    out.extend_from_slice(&STANDING_INPUT_VERSION.to_le_bytes());
    out.extend_from_slice(&(variant.len() as u64).to_le_bytes());
    out.extend_from_slice(variant.as_bytes());
    out
}

/// The calibration standing an uncertainty statement rests on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CalibrationInput {
    /// No calibration record exists or has been measured. Interval or posterior claims have no
    /// backing; point values are unaffected.
    Unmeasured,
    /// The claim states a point value only.
    PointOnly,
    /// A calibration record established at exactly this coordinate (method, target population,
    /// regime, quantity).
    Held {
        /// Identity of the calibration record.
        record: StageIdentity,
        /// Identity of the coordinate the record was established at.
        coordinate: StageIdentity,
    },
    /// A record that existed and no longer holds (retired, replaced or out of coordinate).
    Withdrawn {
        /// Identity of the record that no longer holds.
        record: StageIdentity,
    },
}

impl CalibrationInput {
    /// Canonical identity of this input.
    #[must_use]
    pub fn identity(&self) -> StageIdentity {
        const LABEL: &str = "recalc.calibration_input.v1";
        match self {
            Self::Unmeasured => StageIdentity::of(LABEL, &[&tag_word(LABEL, "unmeasured")]),
            Self::PointOnly => StageIdentity::of(LABEL, &[&tag_word(LABEL, "point_only")]),
            Self::Held { record, coordinate } => StageIdentity::of(
                LABEL,
                &[&tag_word(LABEL, "held"), record.as_bytes(), coordinate.as_bytes()],
            ),
            Self::Withdrawn { record } => {
                StageIdentity::of(LABEL, &[&tag_word(LABEL, "withdrawn"), record.as_bytes()])
            }
        }
    }
}

/// The representation (model family and its version) artifacts were built with.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepresentationInput {
    /// Family label, for example `linear_gaussian_scm`.
    pub family: String,
    /// Version label of the family's implementation or fitted representation.
    pub version: String,
}

impl RepresentationInput {
    /// A representation at a version.
    #[must_use]
    pub fn new(family: impl Into<String>, version: impl Into<String>) -> Self {
        Self { family: family.into(), version: version.into() }
    }

    /// Canonical identity of this input.
    #[must_use]
    pub fn identity(&self) -> StageIdentity {
        const LABEL: &str = "recalc.representation_input.v1";
        StageIdentity::of(
            LABEL,
            &[
                &tag_word(LABEL, "representation"),
                self.family.as_bytes(),
                self.version.as_bytes(),
            ],
        )
    }
}

/// The coupling between sampled streams a joint statement relies on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CouplingInput {
    /// The workflow forms no joint statement across streams.
    None,
    /// Streams are declared independent marginals.
    DeclaredIndependent,
    /// Streams share one exogenous source (one abduction, one common random stream).
    SharedExogenous {
        /// Identity of the shared source, by semantic coordinates.
        source: StageIdentity,
    },
    /// Draws are rows of one aligned joint law.
    AlignedJoint {
        /// Identity of the alignment (source, stream and draw count).
        alignment: StageIdentity,
    },
    /// The coupling is not known. A joint statement under it is not licensed.
    Unknown,
}

impl CouplingInput {
    /// Canonical identity of this input.
    #[must_use]
    pub fn identity(&self) -> StageIdentity {
        const LABEL: &str = "recalc.coupling_input.v1";
        match self {
            Self::None => StageIdentity::of(LABEL, &[&tag_word(LABEL, "none")]),
            Self::DeclaredIndependent => {
                StageIdentity::of(LABEL, &[&tag_word(LABEL, "declared_independent")])
            }
            Self::SharedExogenous { source } => StageIdentity::of(
                LABEL,
                &[&tag_word(LABEL, "shared_exogenous"), source.as_bytes()],
            ),
            Self::AlignedJoint { alignment } => StageIdentity::of(
                LABEL,
                &[&tag_word(LABEL, "aligned_joint"), alignment.as_bytes()],
            ),
            Self::Unknown => StageIdentity::of(LABEL, &[&tag_word(LABEL, "unknown")]),
        }
    }
}

/// The verification standing of one external provider.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ProviderStandingInput {
    /// The provider is not verified at its current identity.
    Unverified,
    /// Verified at a named level by a receipt.
    Verified {
        /// Verification level label (the producer's own vocabulary).
        level: String,
        /// Identity of the verification receipt.
        receipt: StageIdentity,
    },
    /// Trust was revoked or the provider replaced.
    Revoked,
}

impl ProviderStandingInput {
    /// Canonical identity of this input for the provider behind `branch`.
    #[must_use]
    pub fn identity(&self, branch: Branch) -> StageIdentity {
        const LABEL: &str = "recalc.provider_standing_input.v1";
        let branch = [branch.index()];
        match self {
            Self::Unverified => {
                StageIdentity::of(LABEL, &[&tag_word(LABEL, "unverified"), &branch])
            }
            Self::Verified { level, receipt } => StageIdentity::of(
                LABEL,
                &[&tag_word(LABEL, "verified"), &branch, level.as_bytes(), receipt.as_bytes()],
            ),
            Self::Revoked => StageIdentity::of(LABEL, &[&tag_word(LABEL, "revoked"), &branch]),
        }
    }
}
