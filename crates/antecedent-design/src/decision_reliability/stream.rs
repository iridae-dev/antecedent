//! Stream identities for decision draws, by semantic coordinates.
//!
//! A stream is named by `(run seed, purpose, replicate, label)`, never by an object
//! pointer or an index into a list. Two candidate evaluations use common random numbers
//! exactly when they carry the same [`DrawSegment`]; two segments are independent when
//! their streams differ in any coordinate, or when they are disjoint ranges of the same
//! stream. Anything else (a shifted overlap) is neither paired nor independent, and the
//! caller must refuse rather than guess.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, CausalRng, RngFactory, StreamDomain, mix64};
use serde::{Deserialize, Serialize};

/// Stream domain of decision-reliability simulation. Streams are keyed by the semantic
/// coordinates of a [`StreamId`], so one stream's draws do not depend on how many other
/// streams exist.
pub const DECISION_RELIABILITY_STREAM_DOMAIN: StreamDomain =
    StreamDomain::external(StreamDomain::EXTERNAL_BASE + 0x4452);

/// What a stream is used for. The purpose is a coordinate of the stream's identity: a
/// selection stream and a validation stream never share noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamPurpose {
    /// Draws of the source law itself (a posterior, predictive or sampling simulation).
    Source,
    /// Draws on which candidates were evaluated and one was selected.
    Selection,
    /// Draws on which the selected decision was re-evaluated.
    Validation,
    /// Draws of an evaluation that did not select (a pre-specified decision).
    Evaluation,
}

impl StreamPurpose {
    /// Stable lowercase label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Selection => "selection",
            Self::Validation => "validation",
            Self::Evaluation => "evaluation",
        }
    }

    const fn word(self) -> u64 {
        match self {
            Self::Source => 1,
            Self::Selection => 2,
            Self::Validation => 3,
            Self::Evaluation => 4,
        }
    }
}

/// Semantic coordinates of one random stream.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamId {
    /// Run seed the stream is derived from.
    pub run_seed: u64,
    /// What the stream is used for.
    pub purpose: StreamPurpose,
    /// Replicate coordinate (calibration replicate, repeated evaluation number).
    pub replicate: u64,
    /// Semantic label (producer-defined: unit, world, fold or source name).
    pub label: String,
}

impl StreamId {
    /// A stream identity.
    #[must_use]
    pub fn new(
        run_seed: u64,
        purpose: StreamPurpose,
        replicate: u64,
        label: impl Into<String>,
    ) -> Self {
        Self { run_seed, purpose, replicate, label: label.into() }
    }

    /// Stream index within [`DECISION_RELIABILITY_STREAM_DOMAIN`]: a function of the
    /// semantic coordinates only.
    #[must_use]
    pub fn index(&self) -> u64 {
        let mut h = mix64(self.purpose.word() ^ 0x4452_5354);
        h = mix64(h ^ self.replicate);
        h = mix64(h ^ (self.label.len() as u64));
        for byte in self.label.bytes() {
            h = mix64(h ^ u64::from(byte));
        }
        h
    }

    /// The generator of this stream.
    #[must_use]
    pub fn rng(&self) -> CausalRng {
        RngFactory::from_seed(self.run_seed)
            .stream_for(DECISION_RELIABILITY_STREAM_DOMAIN, self.index())
    }

    /// Typed length-delimited digest of the coordinates.
    #[must_use]
    pub fn digest_hex(&self) -> String {
        CanonicalWriter::new("decision_reliability.stream_id.v1")
            .word(self.run_seed)
            .text(self.purpose.as_str())
            .word(self.replicate)
            .text(&self.label)
            .finish_hex()
    }

    /// Human-readable coordinates for refusals and receipts.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "seed={}/purpose={}/replicate={}/label={}",
            self.run_seed,
            self.purpose.as_str(),
            self.replicate,
            self.label
        )
    }
}

/// A contiguous range of draws of one stream: draw `i` of the segment is draw
/// `start + i` of the stream.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DrawSegment {
    /// The stream the draws belong to.
    pub stream: StreamId,
    /// First draw of the stream in the segment.
    pub start: u64,
    /// Number of draws.
    pub len: u64,
}

impl DrawSegment {
    /// Draws `[0, len)` of a stream.
    #[must_use]
    pub fn whole(stream: StreamId, len: usize) -> Self {
        Self { stream, start: 0, len: len as u64 }
    }

    /// One past the last draw index.
    #[must_use]
    pub fn end(&self) -> u64 {
        self.start.saturating_add(self.len)
    }

    /// The sub-range `[from, from + len)` of this segment, in stream coordinates.
    #[must_use]
    pub fn sub(&self, from: u64, len: u64) -> Self {
        Self { stream: self.stream.clone(), start: self.start.saturating_add(from), len }
    }

    /// Whether draw `i` of both segments is the same random state (common random
    /// numbers): same stream, same start and same length.
    #[must_use]
    pub fn same_draws(&self, other: &Self) -> bool {
        self == other
    }

    /// Whether the two segments share any draw of one stream.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.stream == other.stream && self.start < other.end() && other.start < self.end()
    }

    /// Whether the segments are statistically independent: different streams, or
    /// disjoint ranges of one stream separated by at least `gap` draws.
    ///
    /// Streams are independent by construction (distinct semantic coordinates give
    /// distinct mixed seeds). Ranges of one stream are independent only for serially
    /// independent draws or when `gap` exceeds the serial dependence; the caller owns
    /// that choice.
    #[must_use]
    pub fn independent_of(&self, other: &Self, gap: u64) -> bool {
        if self.stream != other.stream {
            return true;
        }
        let separation = if self.end() <= other.start {
            other.start - self.end()
        } else if other.end() <= self.start {
            self.start - other.end()
        } else {
            return false;
        };
        separation >= gap
    }

    /// Human-readable coordinates.
    #[must_use]
    pub fn describe(&self) -> String {
        format!("{}/draws=[{}, {})", self.stream.describe(), self.start, self.end())
    }

    /// Typed length-delimited digest.
    #[must_use]
    pub fn digest_hex(&self) -> String {
        CanonicalWriter::new("decision_reliability.draw_segment.v1")
            .text(&self.stream.digest_hex())
            .word(self.start)
            .word(self.len)
            .finish_hex()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(purpose: StreamPurpose, replicate: u64) -> StreamId {
        StreamId::new(7, purpose, replicate, "candidate_noise")
    }

    #[test]
    fn purpose_and_replicate_are_coordinates_of_the_stream() {
        let a = stream(StreamPurpose::Selection, 0);
        let b = stream(StreamPurpose::Validation, 0);
        let c = stream(StreamPurpose::Selection, 1);
        assert_ne!(a.index(), b.index());
        assert_ne!(a.index(), c.index());
        assert_ne!(a.digest_hex(), b.digest_hex());
        assert_eq!(a.index(), stream(StreamPurpose::Selection, 0).index());
        let (mut ra, mut rb) = (a.rng(), b.rng());
        assert_ne!(ra.next_u64(), rb.next_u64());
    }

    #[test]
    fn segments_distinguish_paired_independent_and_ambiguous() {
        let s = stream(StreamPurpose::Selection, 0);
        let whole = DrawSegment::whole(s.clone(), 100);
        assert!(whole.same_draws(&whole.clone()));
        let other = DrawSegment::whole(stream(StreamPurpose::Validation, 0), 100);
        assert!(whole.independent_of(&other, 0) && !whole.overlaps(&other));
        // Disjoint ranges of one stream are independent only past the gap.
        let head = whole.sub(0, 40);
        let tail = whole.sub(50, 50);
        assert!(head.independent_of(&tail, 10) && !head.independent_of(&tail, 11));
        // A shifted overlap is neither paired nor independent.
        let shifted = whole.sub(10, 100);
        assert!(whole.overlaps(&shifted));
        assert!(!whole.same_draws(&shifted) && !whole.independent_of(&shifted, 0));
    }
}
