// SPDX-License-Identifier: MIT OR Apache-2.0

//! Legality of physical rewrites against the scientific frame.
//!
//! A column or filter pushdown, a shared row and design preparation, a
//! nuisance reuse, a sufficient-statistics lowering, and a tiled evaluation
//! are one [`Rewrite`]. The effect enums name what the caller attributes to
//! that rewrite. [`Rewrite::check`] admits it only when the effect alters no
//! axis and an executed fingerprint of every frame matches: selected rows,
//! folds, population, history, dependence, and uncertainty. A fingerprint is
//! built from the rows an executor actually carried ([`FrameContents`], or
//! [`FrameLedger`] while rows stream past), never from an identity string.
//! Production rewrites call [`check_executed`], which takes only the original
//! and rewritten fingerprints and has no description to forge. The check does
//! not execute the rewrite or number rows; the executor supplies the ids it
//! used.
//!
//! An altered axis is a counterexample. The [`RewriteRefusal`] names that axis
//! and the next attempt that keeps it. Copying the same identity text onto an
//! altering effect, or onto a frame whose fingerprint differs, does not make
//! the rewrite legal. When several axes would change, the earliest axis in
//! [`PreservationAxis`] order is the refusal.

use std::{collections::BTreeSet, error::Error, fmt, sync::Arc};

use antecedent_core::{
    CanonicalWriter, NextAttempt, SemanticDigest, named_variants, text::is_blank,
};

const INVALID_CODE: &str = antecedent_core::reason_code!("invalid_argument");
const BLANK_DETAIL: &str = "rewrite_legality.blank_identity";
const TAG_SELECTED_ROWS: &str = "rewrite_legality.selected_rows";
const TAG_FOLDS: &str = "rewrite_legality.fold_assignment";
const TAG_POPULATION: &str = "rewrite_legality.population_id";
const TAG_HISTORY: &str = "rewrite_legality.history";
const TAG_DEPENDENCE: &str = "rewrite_legality.dependence_factors";
const TAG_UNCERTAINTY: &str = "rewrite_legality.uncertainty";
const NO_FINGERPRINT: &str = "it carries no fingerprint of the selected rows, folds, population, \
     history, dependence, and uncertainty";
/// Dependence recorded for a frame that declared none.
pub const UNDECLARED_DEPENDENCE: &str = "dependence.undeclared";
/// Uncertainty recorded for a frame that declared none.
pub const UNDECLARED_UNCERTAINTY: &str = "uncertainty.undeclared";
/// Sampling law recorded for a frame that declared none.
pub const UNDECLARED_SAMPLING_LAW: &str = "sampling_law.undeclared";
const FRAME_BINDING: &str = "rewrite_legality.frame_binding";

/// An axis a legal rewrite leaves unchanged.
///
/// Declaration order is the order [`Rewrite::check`] reports alterations.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum PreservationAxis {
    /// Which rows the estimator sees.
    SampleSelection,
    /// Cross-fitting or split assignment.
    Folds,
    /// Target population.
    Population,
    /// History window, horizon, or initial state.
    History,
    /// Shared factors, clusters, and covariance cross terms.
    Dependence,
    /// Weights, sampling law, and failed-draw mass.
    Uncertainty,
}

named_variants! {
    PreservationAxis {
        SampleSelection => "sample_selection",
        Folds => "folds",
        Population => "population",
        History => "history",
        Dependence => "dependence",
        Uncertainty => "uncertainty",
    }
}

impl PreservationAxis {
    /// Prose name used in a refusal message.
    #[must_use]
    pub const fn phrase(self) -> &'static str {
        match self {
            Self::SampleSelection => "sample selection",
            Self::Folds => "folds",
            Self::Population => "population",
            Self::History => "history",
            Self::Dependence => "dependence",
            Self::Uncertainty => "uncertainty",
        }
    }

    /// Next attempt that keeps this axis.
    #[must_use]
    pub const fn next_attempt(self) -> NextAttempt {
        match self {
            Self::SampleSelection => RETAIN_SAMPLE_SELECTION,
            Self::Folds => RETAIN_FOLDS,
            Self::Population => RETAIN_POPULATION,
            Self::History => RETAIN_HISTORY,
            Self::Dependence => RETAIN_DEPENDENCE,
            Self::Uncertainty => RETAIN_UNCERTAINTY,
        }
    }

    const fn detail(self) -> &'static str {
        match self {
            Self::SampleSelection => "rewrite_legality.sample_selection",
            Self::Folds => "rewrite_legality.folds",
            Self::Population => "rewrite_legality.population",
            Self::History => "rewrite_legality.history",
            Self::Dependence => "rewrite_legality.dependence",
            Self::Uncertainty => "rewrite_legality.uncertainty",
        }
    }
}

impl fmt::Display for PreservationAxis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Keep the rows the unrewritten frame selected.
pub const RETAIN_SAMPLE_SELECTION: NextAttempt =
    NextAttempt::new("retain_sample_selection", "retain the original sample selection");

/// Keep the declared fold assignment.
pub const RETAIN_FOLDS: NextAttempt = NextAttempt::new("retain_folds", "retain the declared folds");

/// Keep the declared population.
pub const RETAIN_POPULATION: NextAttempt =
    NextAttempt::new("retain_population", "retain the declared population");

/// Keep the declared history.
pub const RETAIN_HISTORY: NextAttempt =
    NextAttempt::new("retain_history", "retain the declared history");

/// Keep the declared dependence.
pub const RETAIN_DEPENDENCE: NextAttempt =
    NextAttempt::new("retain_dependence", "retain the declared dependence");

/// Keep the declared uncertainty.
pub const RETAIN_UNCERTAINTY: NextAttempt =
    NextAttempt::new("retain_uncertainty", "retain the declared uncertainty");

/// Replace a blank or over-long identity before checking a rewrite.
pub const NAME_PRESERVED_IDENTITY: NextAttempt =
    NextAttempt::new("name_preserved_identity", "name the preserved identity");

/// Every attempt, in refusal order, then the identity attempt.
pub const NEXT_ATTEMPTS: [NextAttempt; 7] = [
    RETAIN_SAMPLE_SELECTION,
    RETAIN_FOLDS,
    RETAIN_POPULATION,
    RETAIN_HISTORY,
    RETAIN_DEPENDENCE,
    RETAIN_UNCERTAINTY,
    NAME_PRESERVED_IDENTITY,
];

/// Identity of one preserved axis. Not a row number.
#[derive(Clone, Debug, Eq, PartialEq)]
struct AxisId(Arc<str>);

impl AxisId {
    fn try_new(text: &str, axis: PreservationAxis) -> Result<Self, RewriteRefusal> {
        if is_blank(text) { Err(blank(axis)) } else { Ok(Self(Arc::from(text))) }
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

/// The six identities a rewrite must carry unchanged.
///
/// Each identity is a name the caller supplies. Matching names are not a
/// fingerprint of the frame, and they do not admit a rewrite. This contract
/// does not assign row numbers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreservedIdentities {
    sample_selection: AxisId,
    folds: AxisId,
    population: AxisId,
    history: AxisId,
    dependence: AxisId,
    uncertainty: AxisId,
}

impl PreservedIdentities {
    /// Identities in [`PreservationAxis`] order.
    ///
    /// Whitespace around a non-blank name is part of the identity.
    ///
    /// # Errors
    ///
    /// A blank identity. The axis
    /// is the first such one in [`PreservationAxis`] order, and the next
    /// attempt is [`NAME_PRESERVED_IDENTITY`].
    pub fn try_new(
        sample_selection: &str,
        folds: &str,
        population: &str,
        history: &str,
        dependence: &str,
        uncertainty: &str,
    ) -> Result<Self, RewriteRefusal> {
        Ok(Self {
            sample_selection: AxisId::try_new(sample_selection, PreservationAxis::SampleSelection)?,
            folds: AxisId::try_new(folds, PreservationAxis::Folds)?,
            population: AxisId::try_new(population, PreservationAxis::Population)?,
            history: AxisId::try_new(history, PreservationAxis::History)?,
            dependence: AxisId::try_new(dependence, PreservationAxis::Dependence)?,
            uncertainty: AxisId::try_new(uncertainty, PreservationAxis::Uncertainty)?,
        })
    }

    /// Identity text of `axis`.
    #[must_use]
    pub fn identity(&self, axis: PreservationAxis) -> &str {
        match axis {
            PreservationAxis::SampleSelection => self.sample_selection.as_str(),
            PreservationAxis::Folds => self.folds.as_str(),
            PreservationAxis::Population => self.population.as_str(),
            PreservationAxis::History => self.history.as_str(),
            PreservationAxis::Dependence => self.dependence.as_str(),
            PreservationAxis::Uncertainty => self.uncertainty.as_str(),
        }
    }
}

/// One executed row of a frame, with what the executor attached to it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameRow<'a> {
    /// Id of the selected row.
    pub row: &'a str,
    /// Fold the row was assigned to.
    pub fold: &'a str,
    /// Shared factor or cluster the row belongs to.
    pub dependence: &'a str,
    /// Weight or failed-draw state the row carries into the uncertainty.
    pub uncertainty: &'a str,
}

/// Contents of one scientific frame.
///
/// [`Self::fingerprint`] hashes these contents. It is not an identity string
/// copied from [`PreservedIdentities`]. Row ids are stored in order, each with
/// the fold id, dependence factor, and uncertainty state that row carried.
/// Whitespace around a non-blank id is part of the content.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameContents {
    row_ids: Vec<Arc<str>>,
    fold_ids: Vec<Arc<str>>,
    dependence_ids: Vec<Arc<str>>,
    uncertainty_ids: Vec<Arc<str>>,
    population_id: Arc<str>,
    history: Arc<str>,
    sampling_law: Arc<str>,
}

impl FrameContents {
    /// Frame whose selected rows are `row_ids`, each assigned `fold_ids` in
    /// the same order, in `population_id`, with `history`.
    ///
    /// Dependence and uncertainty are recorded as undeclared. A frame that
    /// carries either differs from this one on that axis. Use
    /// [`Self::try_new_executed`] to record them.
    ///
    /// # Errors
    ///
    /// A blank id, a repeated row
    /// id, or a fold list whose length is not the number of selected rows. The
    /// axis is the earliest one in [`PreservationAxis`] order.
    pub fn try_new(
        row_ids: &[&str],
        fold_ids: &[&str],
        population_id: &str,
        history: &str,
    ) -> Result<Self, RewriteRefusal> {
        Self::validate(row_ids, fold_ids, population_id, history)?;
        let undeclared_dependence = vec![UNDECLARED_DEPENDENCE; row_ids.len()];
        let undeclared_uncertainty = vec![UNDECLARED_UNCERTAINTY; row_ids.len()];
        Ok(Self::store(
            row_ids,
            fold_ids,
            &undeclared_dependence,
            &undeclared_uncertainty,
            population_id,
            history,
            UNDECLARED_SAMPLING_LAW,
        ))
    }

    /// Frame whose rows each carry their fold, dependence factor, and
    /// uncertainty state, with the frame's `sampling_law`.
    ///
    /// # Errors
    ///
    /// The refusals of [`Self::try_new`], and a blank or over-long dependence
    /// id, uncertainty state, or sampling law.
    pub fn try_new_executed(
        rows: &[FrameRow<'_>],
        population_id: &str,
        history: &str,
        sampling_law: &str,
    ) -> Result<Self, RewriteRefusal> {
        let row_ids: Vec<&str> = rows.iter().map(|row| row.row).collect();
        let fold_ids: Vec<&str> = rows.iter().map(|row| row.fold).collect();
        let dependence: Vec<&str> = rows.iter().map(|row| row.dependence).collect();
        let uncertainty: Vec<&str> = rows.iter().map(|row| row.uncertainty).collect();
        Self::validate(&row_ids, &fold_ids, population_id, history)?;
        for id in &dependence {
            check_text(PreservationAxis::Dependence, "a dependence id", id)?;
        }
        for id in &uncertainty {
            check_text(PreservationAxis::Uncertainty, "an uncertainty state", id)?;
        }
        check_text(PreservationAxis::Uncertainty, "the sampling law", sampling_law)?;
        Ok(Self::store(
            &row_ids,
            &fold_ids,
            &dependence,
            &uncertainty,
            population_id,
            history,
            sampling_law,
        ))
    }

    fn validate(
        row_ids: &[&str],
        fold_ids: &[&str],
        population_id: &str,
        history: &str,
    ) -> Result<(), RewriteRefusal> {
        let mut seen = BTreeSet::new();
        for row in row_ids {
            check_text(PreservationAxis::SampleSelection, "a selected row id", row)?;
            if !seen.insert(*row) {
                return Err(content_refusal(
                    PreservationAxis::SampleSelection,
                    "a selected row id is repeated",
                ));
            }
        }
        for fold in fold_ids {
            check_text(PreservationAxis::Folds, "a fold id", fold)?;
        }
        if row_ids.len() != fold_ids.len() {
            return Err(content_refusal(
                PreservationAxis::Folds,
                "the fold assignment does not cover the selected rows",
            ));
        }
        check_text(PreservationAxis::Population, "the population id", population_id)?;
        check_text(PreservationAxis::History, "the history", history)
    }

    fn store(
        row_ids: &[&str],
        fold_ids: &[&str],
        dependence: &[&str],
        uncertainty: &[&str],
        population_id: &str,
        history: &str,
        sampling_law: &str,
    ) -> Self {
        let mut order: Vec<usize> = (0..row_ids.len()).collect();
        order.sort_by_key(|&index| row_ids[index]);
        let pick = |texts: &[&str]| -> Vec<Arc<str>> {
            order.iter().map(|&index| Arc::from(texts[index])).collect()
        };
        Self {
            row_ids: pick(row_ids),
            fold_ids: pick(fold_ids),
            dependence_ids: pick(dependence),
            uncertainty_ids: pick(uncertainty),
            population_id: Arc::from(population_id),
            history: Arc::from(history),
            sampling_law: Arc::from(sampling_law),
        }
    }

    /// Selected row ids, in canonical order.
    #[must_use = "the selected row ids are the fingerprint input"]
    pub fn row_ids(&self) -> impl Iterator<Item = &str> {
        self.row_ids.iter().map(AsRef::as_ref)
    }

    /// Fold id of each [`Self::row_ids`] entry, in the same order.
    #[must_use = "the fold ids are the fingerprint input"]
    pub fn fold_ids(&self) -> impl Iterator<Item = &str> {
        self.fold_ids.iter().map(AsRef::as_ref)
    }

    /// Dependence factor of each [`Self::row_ids`] entry, in the same order.
    #[must_use = "the dependence ids are the fingerprint input"]
    pub fn dependence_ids(&self) -> impl Iterator<Item = &str> {
        self.dependence_ids.iter().map(AsRef::as_ref)
    }

    /// Uncertainty state of each [`Self::row_ids`] entry, in the same order.
    #[must_use = "the uncertainty states are the fingerprint input"]
    pub fn uncertainty_ids(&self) -> impl Iterator<Item = &str> {
        self.uncertainty_ids.iter().map(AsRef::as_ref)
    }

    /// Population id.
    #[must_use]
    pub fn population_id(&self) -> &str {
        &self.population_id
    }

    /// History window, horizon, or initial state.
    #[must_use]
    pub fn history(&self) -> &str {
        &self.history
    }

    /// Sampling law of the frame.
    #[must_use]
    pub fn sampling_law(&self) -> &str {
        &self.sampling_law
    }

    /// Fingerprint of these contents.
    #[must_use]
    pub fn fingerprint(&self) -> FrameFingerprint {
        let mut ledger = FrameLedger::seeded(
            Arc::clone(&self.population_id),
            Arc::clone(&self.history),
            Arc::clone(&self.sampling_law),
        );
        for index in 0..self.row_ids.len() {
            ledger.record(
                &self.row_ids[index],
                &self.fold_ids[index],
                &self.dependence_ids[index],
                &self.uncertainty_ids[index],
            );
        }
        ledger.fingerprint()
    }
}

/// Four 64-bit lanes that add one 256-bit row digest at a time.
///
/// The sum does not depend on the order rows arrive in, so an executor that
/// streams rows in file order and a frame listed in id order agree. A repeated
/// row is added twice and changes the sum.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Lanes([u64; 4]);

impl Lanes {
    fn add(&mut self, digest: [u8; 32]) {
        for (lane, chunk) in self.0.iter_mut().zip(digest.chunks_exact(8)) {
            let mut word = [0_u8; 8];
            word.copy_from_slice(chunk);
            *lane = lane.wrapping_add(u64::from_le_bytes(word));
        }
    }

    /// Add a row from its numeric ids with a fast, non-cryptographic mixer.
    fn add_mixed(&mut self, tag: u64, first: u64, second: u64) {
        const SALT: [u64; 4] = [
            0x243F_6A88_85A3_08D3,
            0x1319_8A2E_0370_7344,
            0xA409_3822_299F_31D0,
            0x082E_FA98_EC4E_6C89,
        ];
        let mut hash = mix64(tag);
        hash = mix64(hash ^ first);
        hash = mix64(hash ^ second.rotate_left(29) ^ 0xC2B2_AE3D_27D4_EB4F);
        for (lane, salt) in self.0.iter_mut().zip(SALT) {
            *lane = lane.wrapping_add(mix64(hash ^ salt));
        }
    }

    fn digest(&self, tag: &str, rows: u64, tail: Option<&str>) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(tag);
        writer.word(rows);
        for lane in self.0 {
            writer.word(lane);
        }
        if let Some(tail) = tail {
            writer.text(tail);
        }
        SemanticDigest::from_bytes(writer.finish())
    }
}

const fn mix64(value: u64) -> u64 {
    let mut x = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// One id for a tuple of numeric ids, for [`FrameLedger::record_numeric`].
///
/// Order matters and a different tuple gives a different id.
#[must_use]
pub fn combine_ids(parts: &[u64]) -> u64 {
    let mut hash = mix64(parts.len() as u64);
    for part in parts {
        hash = mix64(hash ^ part);
    }
    hash
}

/// Row id that includes the row's values, for [`FrameLedger::record`].
///
/// A row whose values changed is a different selected row, so a rewrite that
/// keeps every row id but alters a value does not share a fingerprint.
#[must_use]
pub fn value_row_id(row: u64, values: impl IntoIterator<Item = f64>) -> String {
    let mut writer = CanonicalWriter::new("rewrite_legality.row_values");
    for value in values {
        writer.word(value.to_bits());
    }
    let digest = writer.finish();
    let mut head = [0_u8; 8];
    head.copy_from_slice(&digest[..8]);
    format!("{row}:{:016x}", u64::from_le_bytes(head))
}

fn row_digest(tag: &str, parts: &[&str]) -> [u8; 32] {
    let mut writer = CanonicalWriter::new(tag);
    for part in parts {
        writer.text(part);
    }
    writer.finish()
}

/// Streaming account of the rows a rewrite carried.
///
/// An executor records each row it selects, assigns, links, and weights while
/// it runs, then reads [`Self::fingerprint`]. Rows recorded with [`Self::record`]
/// give the value [`FrameContents::fingerprint`] gives for the same rows, so a streaming
/// pass over a file can be compared with a frame listed in memory. Nothing in
/// the ledger is an identity string supplied to describe a match.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameLedger {
    population: Arc<str>,
    history: Arc<str>,
    sampling_law: Arc<str>,
    rows: u64,
    selected: Lanes,
    folds: Lanes,
    dependence: Lanes,
    uncertainty: Lanes,
}

impl FrameLedger {
    /// Empty ledger for `population_id`, `history`, and `sampling_law`.
    ///
    /// # Errors
    ///
    /// A blank or over-long id. The axis is the earliest in
    /// [`PreservationAxis`] order.
    pub fn try_new(
        population_id: &str,
        history: &str,
        sampling_law: &str,
    ) -> Result<Self, RewriteRefusal> {
        check_text(PreservationAxis::Population, "the population id", population_id)?;
        check_text(PreservationAxis::History, "the history", history)?;
        check_text(PreservationAxis::Uncertainty, "the sampling law", sampling_law)?;
        Ok(Self::seeded(Arc::from(population_id), Arc::from(history), Arc::from(sampling_law)))
    }

    fn seeded(population: Arc<str>, history: Arc<str>, sampling_law: Arc<str>) -> Self {
        Self {
            population,
            history,
            sampling_law,
            rows: 0,
            selected: Lanes::default(),
            folds: Lanes::default(),
            dependence: Lanes::default(),
            uncertainty: Lanes::default(),
        }
    }

    /// Record one executed row.
    pub fn record(&mut self, row: &str, fold: &str, dependence: &str, uncertainty: &str) {
        self.rows += 1;
        self.selected.add(row_digest(TAG_SELECTED_ROWS, &[row]));
        self.folds.add(row_digest(TAG_FOLDS, &[row, fold]));
        self.dependence.add(row_digest(TAG_DEPENDENCE, &[row, dependence]));
        self.uncertainty.add(row_digest(TAG_UNCERTAINTY, &[row, uncertainty]));
    }

    /// Record one executed row whose ids are numbers.
    ///
    /// This is the fast path for streaming hot loops. Its fingerprints are
    /// comparable with other [`Self::record_numeric`] ledgers, not with
    /// [`Self::record`] ledgers or [`FrameContents::fingerprint`].
    pub fn record_numeric(&mut self, row: u64, fold: u64, dependence: u64, uncertainty: u64) {
        self.rows += 1;
        self.selected.add_mixed(1, row, 0);
        self.folds.add_mixed(2, row, fold);
        self.dependence.add_mixed(3, row, dependence);
        self.uncertainty.add_mixed(4, row, uncertainty);
    }

    /// Rows recorded so far.
    #[must_use]
    pub const fn rows(&self) -> u64 {
        self.rows
    }

    /// Fingerprint of the rows recorded so far.
    #[must_use]
    pub fn fingerprint(&self) -> FrameFingerprint {
        FrameFingerprint {
            selected_rows: self.selected.digest(TAG_SELECTED_ROWS, self.rows, None),
            folds: self.folds.digest(TAG_FOLDS, self.rows, None),
            population: digest_text(TAG_POPULATION, &self.population),
            history: digest_text(TAG_HISTORY, &self.history),
            dependence: self.dependence.digest(TAG_DEPENDENCE, self.rows, None),
            uncertainty: self.uncertainty.digest(
                TAG_UNCERTAINTY,
                self.rows,
                Some(&self.sampling_law),
            ),
        }
    }
}

/// Fingerprint of one frame's selected rows, folds, population, history,
/// dependence, and uncertainty.
///
/// Each part is a digest of the contents. Two frames with the same identity
/// strings and different contents do not share a fingerprint.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FrameFingerprint {
    selected_rows: SemanticDigest,
    folds: SemanticDigest,
    population: SemanticDigest,
    history: SemanticDigest,
    dependence: SemanticDigest,
    uncertainty: SemanticDigest,
}

impl FrameFingerprint {
    /// Digest of the selected row ids.
    #[must_use]
    pub const fn selected_rows(&self) -> SemanticDigest {
        self.selected_rows
    }

    /// Digest of each selected row id paired with its fold id.
    #[must_use]
    pub const fn folds(&self) -> SemanticDigest {
        self.folds
    }

    /// Digest of the population id.
    #[must_use]
    pub const fn population(&self) -> SemanticDigest {
        self.population
    }

    /// Digest of the history.
    #[must_use]
    pub const fn history(&self) -> SemanticDigest {
        self.history
    }

    /// Digest of each selected row id paired with its dependence factor.
    #[must_use]
    pub const fn dependence(&self) -> SemanticDigest {
        self.dependence
    }

    /// Digest of each selected row id paired with its uncertainty state, and
    /// of the sampling law.
    #[must_use]
    pub const fn uncertainty(&self) -> SemanticDigest {
        self.uncertainty
    }

    const fn on_axis(self, axis: PreservationAxis) -> SemanticDigest {
        match axis {
            PreservationAxis::SampleSelection => self.selected_rows,
            PreservationAxis::Folds => self.folds,
            PreservationAxis::Population => self.population,
            PreservationAxis::History => self.history,
            PreservationAxis::Dependence => self.dependence,
            PreservationAxis::Uncertainty => self.uncertainty,
        }
    }
}

/// Admit a rewrite from the fingerprints of the frames it actually ran on.
///
/// `original` is the frame before the rewrite. `rewritten` is each frame after
/// it: one for a pushdown, a summary, a tile, or a nuisance consumer, and the
/// left and right consumers of a shared preparation. There is no description
/// or effect to supply, so nothing can claim a match the rows do not show.
/// The earliest altered axis in [`PreservationAxis`] order is the refusal.
///
/// # Errors
///
/// No rewritten frame, or the first axis whose digest differs.
#[must_use = "a refusal is the counterexample"]
pub fn check_executed(
    kind: &'static str,
    original: &FrameFingerprint,
    rewritten: &[FrameFingerprint],
) -> Result<(), RewriteRefusal> {
    if rewritten.is_empty() {
        return Err(frame_binding(kind));
    }
    let frames: Vec<(&FrameFingerprint, &FrameFingerprint)> =
        rewritten.iter().map(|frame| (original, frame)).collect();
    admit(kind, &[], None, &frames)
}

/// Shared defect table.
///
/// Index 0 is the preserving declaration and admits no axis by itself. Each
/// later index is one [`PreservationAxis`], in refusal order, and the reason
/// the effect gives for altering it. Dependence and uncertainty are rows of
/// this table. A row here does not supply the executed fingerprint; the
/// frames' own dependence and uncertainty digests do.
const fn defect_at<const N: usize>(
    index: usize,
    reasons: &[&'static str; N],
) -> Option<(PreservationAxis, &'static str)> {
    if index == 0 {
        return None;
    }
    let axis_index = index - 1;
    if axis_index >= PreservationAxis::ALL.len() {
        return Some((PreservationAxis::SampleSelection, MISSING_DEFECT));
    }
    if axis_index >= N {
        return Some((PreservationAxis::ALL[axis_index], MISSING_DEFECT));
    }
    Some((PreservationAxis::ALL[axis_index], reasons[axis_index]))
}

const MISSING_DEFECT: &str = "the effect table has no row for this variant";

macro_rules! effect_enum {
    (
        $(#[$enum_meta:meta])*
        $name:ident {
            $(#[$preserve_meta:meta])*
            $preserve:ident => $preserve_name:literal,
            $(
                $(#[$variant_meta:meta])*
                $variant:ident => $variant_name:literal, $because:literal
            ),+ $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
        #[non_exhaustive]
        pub enum $name {
            $(#[$preserve_meta])*
            $preserve,
            $(
                $(#[$variant_meta])*
                $variant,
            )+
        }

        impl $name {
            /// Every effect: the preserving declaration, then one alteration per
            /// [`PreservationAxis`], in refusal order.
            ///
            /// [`Self::as_str`] and [`Self::from_name`] both use this list. A
            /// new variant has to be added here; a hand-written match is not
            /// the list.
            pub const ALL: [Self; { PreservationAxis::ALL.len() + 1 }] =
                [Self::$preserve, $( Self::$variant, )+];

            /// Stable `snake_case` name.
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    Self::$preserve => $preserve_name,
                    $( Self::$variant => $variant_name, )+
                }
            }

            /// Parse the spelling of [`Self::as_str`].
            #[must_use = "an unknown name is not an effect"]
            pub fn from_name(name: &str) -> Option<Self> {
                Self::ALL.into_iter().find(|effect| effect.as_str() == name)
            }

            /// Axis this effect alters, with the reason, when it alters one.
            ///
            /// The preserving declaration has no defect. Dependence and
            /// uncertainty are declarations in the shared defect table. They
            /// are not an executed fingerprint.
            #[must_use]
            pub const fn defect(self) -> Option<(PreservationAxis, &'static str)> {
                const REASONS: [&str; { PreservationAxis::ALL.len() }] = [ $( $because, )+ ];
                defect_at(self.index(), &REASONS)
            }

            const fn index(self) -> usize {
                self as usize
            }
        }
    };
}

effect_enum! {
    /// What a column or filter pushdown would do to the frame.
    ///
    /// [`Self::Declared`] names a pushdown that alters no axis. Naming it does not
    /// admit the rewrite. [`Rewrite::check`] admits it only with an executed
    /// fingerprint of the rewritten frame: selected rows, folds, population,
    /// history, dependence, and uncertainty. Equal identity strings are not that
    /// fingerprint. Each other effect alters one axis even when the identity
    /// strings match.
    PushdownEffect {
        /// Required columns stay, and the filter is the query's declared predicate.
        Declared => "declared",
        /// The projection omits a column that decides which rows remain.
        OmitsRequiredColumn => "omits_required_column",
            "the projection omits a column the cohort closure requires",
        /// The pushdown omits or rewrites fold membership.
        ReassignsFolds => "reassigns_folds",
            "the pushdown reassigns fold membership",
        /// The filter is a population cut the query did not declare.
        UndeclaredPopulationCut => "undeclared_population_cut",
            "the filter is a population cut the query did not declare",
        /// The filter cuts the history window.
        CutsHistory => "cuts_history",
            "the filter cuts the history window",
        /// The projection drops a cluster or other dependence factor.
        DropsDependenceFactor => "drops_dependence_factor",
            "the projection drops a dependence factor",
        /// The selected rows stay, but a weight or uncertainty column is dropped.
        DropsUncertainty => "drops_uncertainty",
            "the pushdown drops a weight or uncertainty column",
    }
}

effect_enum! {
    /// What sharing one row and design preparation would do.
    ///
    /// [`Self::Shared`] names a preparation that alters no axis. Naming it does not
    /// admit the share. [`Rewrite::check`] admits it only with an executed
    /// fingerprint of the preparation and of each consumer: selected rows, folds,
    /// population, history, dependence, and uncertainty. Equal identity strings
    /// are not that fingerprint. Each other effect mixes one axis even when the
    /// strings match.
    PreparationEffect {
        /// Both consumers use the preparation's sample, folds, population, history,
        /// dependence, and uncertainty.
        Shared => "shared",
        /// The preparation mixes two sample selections.
        MixesSamples => "mixes_samples",
            "the shared preparation mixes sample selections",
        /// The preparation mixes two fold assignments.
        MixesFolds => "mixes_folds",
            "the shared preparation mixes fold assignments",
        /// The preparation mixes two populations.
        MixesPopulations => "mixes_populations",
            "the shared preparation mixes populations",
        /// The preparation mixes two histories.
        MixesHistories => "mixes_histories",
            "the shared preparation mixes histories",
        /// The preparation mixes two dependence declarations.
        MixesDependence => "mixes_dependence",
            "the shared preparation mixes dependence",
        /// The preparation mixes two uncertainty procedures.
        MixesUncertainty => "mixes_uncertainty",
            "the shared preparation mixes uncertainty",
    }
}

effect_enum! {
    /// What reusing a nuisance fit would do.
    ///
    /// [`Self::Certified`] names reuse that alters no axis, with the evaluation fold
    /// held out of training. Naming it does not admit the reuse. [`Rewrite::check`]
    /// admits it only with an executed fingerprint of the fit and of the consumer:
    /// selected rows, folds, population, history, dependence, and uncertainty.
    /// Equal identity strings are not that fingerprint. Each other effect is a
    /// counterexample.
    NuisanceEffect {
        /// The fit and the consumer name one frame, and training excludes the
        /// evaluation fold.
        Certified => "certified",
        /// Reuse adds or drops rows from the estimation sample.
        ChangesEstimationSample => "changes_estimation_sample",
            "reuse changes which rows the estimation sample contains",
        /// Predictions trained on a fold are reused on that same fold.
        TrainsOnEvaluationFold => "trains_on_evaluation_fold",
            "the nuisance was trained on the evaluation fold",
        /// The nuisance was fit on a different population.
        FitOnOtherPopulation => "fit_on_other_population",
            "the nuisance was fit on another population",
        /// The nuisance was fit on a different history.
        FitOnOtherHistory => "fit_on_other_history",
            "the nuisance was fit on another history",
        /// Reuse drops a shared dependence factor.
        DropsSharedFactor => "drops_shared_factor",
            "reuse drops a shared dependence factor",
        /// Reuse replaces the consumer's uncertainty procedure.
        ReplacesUncertainty => "replaces_uncertainty",
            "reuse replaces the uncertainty procedure",
    }
}

effect_enum! {
    /// What lowering a frame to sufficient statistics would do.
    ///
    /// [`Self::Sufficient`] names a summary that alters no axis. Naming it does not
    /// admit the lowering. [`Rewrite::check`] admits it only with an executed
    /// fingerprint of the rows and of the summary: selected rows, folds, population,
    /// history, dependence, and uncertainty. Equal identity strings are not that
    /// fingerprint. A summary that evicts a row, pools folds, or drops the sampling law is a
    /// counterexample, including when the summary still carries the original text.
    SummaryEffect {
        /// The summary is sufficient for the declared likelihood and names every axis.
        ///
        /// The executed fingerprint still has to match. This variant does not admit
        /// the lowering by itself.
        Sufficient => "sufficient",
        /// The summary evicts a row or inserts a column the rows did not have.
        EvictsRow => "evicts_row",
            "the summary evicts a row from the sample",
        /// The summary pools distinct folds into one fit.
        PoolsFolds => "pools_folds",
            "the summary pools distinct folds",
        /// The summary marginalizes to a different population.
        MarginalizesPopulation => "marginalizes_population",
            "the summary marginalizes to another population",
        /// The summary collapses history.
        CollapsesHistory => "collapses_history",
            "the summary collapses history",
        /// The summary drops a cluster or other dependence factor.
        DropsCluster => "drops_cluster",
            "the summary drops a cluster from the dependence",
        /// The summary keeps a point and drops the sampling law.
        DropsSamplingLaw => "drops_sampling_law",
            "the summary drops the sampling law",
    }
}

effect_enum! {
    /// What evaluating one tile would do to the full frame.
    ///
    /// [`Self::Faithful`] names a tile that alters no axis and keeps failed draws.
    /// Naming it does not admit the tile. [`Rewrite::check`] admits it only with
    /// an executed fingerprint of the full frame and of the tile: selected rows,
    /// folds, population, history, dependence, and uncertainty. Equal identity
    /// strings are not that fingerprint. A tile that drops rows, splits a fold, or drops
    /// failed draws is a counterexample.
    TileEffect {
        /// The tile covers the full frame and keeps joint alignment and failed draws.
        Faithful => "faithful",
        /// The tile drops rows from the sample.
        DropsRows => "drops_rows",
            "a tile drops rows from the sample",
        /// The tile splits a fold or reorders fold membership.
        SplitsFold => "splits_fold",
            "a tile splits a fold",
        /// The tile evaluates a different population.
        RetargetsPopulation => "retargets_population",
            "a tile evaluates another population",
        /// The tile cuts the history horizon.
        CutsHorizon => "cuts_horizon",
            "a tile cuts the history horizon",
        /// The tile drops dependence cross terms.
        DropsCrossTerms => "drops_cross_terms",
            "a tile drops dependence cross terms",
        /// The tile drops failed draws, so uncertainty mass changes.
        DropsFailedDraws => "drops_failed_draws",
            "a tile drops failed draws from the uncertainty",
    }
}

/// Effect a [`Rewrite`] carries into the shared defect table.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum AttributedEffect {
    Pushdown(PushdownEffect),
    Preparation(PreparationEffect),
    Nuisance(NuisanceEffect),
    Summary(SummaryEffect),
    Tile(TileEffect),
}

impl AttributedEffect {
    const fn defect(self) -> Option<(PreservationAxis, &'static str)> {
        match self {
            Self::Pushdown(effect) => effect.defect(),
            Self::Preparation(effect) => effect.defect(),
            Self::Nuisance(effect) => effect.defect(),
            Self::Summary(effect) => effect.defect(),
            Self::Tile(effect) => effect.defect(),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Pushdown(effect) => effect.as_str(),
            Self::Preparation(effect) => effect.as_str(),
            Self::Nuisance(effect) => effect.as_str(),
            Self::Summary(effect) => effect.as_str(),
            Self::Tile(effect) => effect.as_str(),
        }
    }
}

/// Identities a rewrite compares, after the source frame.
#[derive(Clone, Debug, Eq, PartialEq)]
enum Claims {
    /// One rewritten frame: pushed, consumer, summary, or tile.
    One(PreservedIdentities),
    /// Left consumer, then right consumer, of a shared preparation.
    Two(PreservedIdentities, PreservedIdentities),
}

/// Executed fingerprints of the source and of each claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExecutedFrames {
    /// Source and the one rewritten frame.
    Pair { source: FrameFingerprint, rewritten: FrameFingerprint },
    /// Preparation and its left and right consumers.
    Share { source: FrameFingerprint, left: FrameFingerprint, right: FrameFingerprint },
}

/// A physical rewrite checked against the scientific frame.
///
/// Column or filter pushdown, shared row and design preparation, nuisance
/// reuse, sufficient statistics, and tiled evaluation share this shape. The
/// effect is one of the effect enums. [`Self::check`] reads that effect from
/// the shared defect table. The rewrite is not admitted without an executed
/// fingerprint of selected rows, folds, population, history, dependence, and
/// uncertainty.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rewrite {
    name: &'static str,
    source: PreservedIdentities,
    claims: Claims,
    effect: AttributedEffect,
    frames: Option<ExecutedFrames>,
}

impl Rewrite {
    /// Stable name of a column or filter pushdown.
    pub const COLUMN_FILTER_PUSHDOWN: &'static str = "column_filter_pushdown";

    /// Stable name of a shared row and design preparation.
    pub const SHARED_ROW_DESIGN_PREPARATION: &'static str = "shared_row_design_preparation";

    /// Stable name of nuisance reuse.
    pub const NUISANCE_REUSE: &'static str = "nuisance_reuse";

    /// Stable name of a sufficient-statistics lowering.
    pub const SUFFICIENT_STATISTICS: &'static str = "sufficient_statistics";

    /// Stable name of a tiled evaluation.
    pub const TILED_EVALUATION: &'static str = "tiled_evaluation";

    /// Push `cohort` down to `pushed` with `effect`.
    ///
    /// [`Self::check`] compares `cohort` with `pushed`.
    #[must_use]
    pub const fn column_filter_pushdown(
        cohort: PreservedIdentities,
        pushed: PreservedIdentities,
        effect: PushdownEffect,
    ) -> Self {
        Self::new(
            Self::COLUMN_FILTER_PUSHDOWN,
            cohort,
            Claims::One(pushed),
            AttributedEffect::Pushdown(effect),
        )
    }

    /// Share `preparation` with `left` and `right` under `effect`.
    ///
    /// [`Self::check`] compares `preparation` with `left`, then with `right`.
    #[must_use]
    pub const fn shared_row_design_preparation(
        preparation: PreservedIdentities,
        left: PreservedIdentities,
        right: PreservedIdentities,
        effect: PreparationEffect,
    ) -> Self {
        Self::new(
            Self::SHARED_ROW_DESIGN_PREPARATION,
            preparation,
            Claims::Two(left, right),
            AttributedEffect::Preparation(effect),
        )
    }

    /// Reuse a nuisance fit on `fit` for `consumer` under `effect`.
    ///
    /// [`Self::check`] compares `fit` with `consumer`.
    #[must_use]
    pub const fn nuisance_reuse(
        fit: PreservedIdentities,
        consumer: PreservedIdentities,
        effect: NuisanceEffect,
    ) -> Self {
        Self::new(
            Self::NUISANCE_REUSE,
            fit,
            Claims::One(consumer),
            AttributedEffect::Nuisance(effect),
        )
    }

    /// Lower `rows` to `summary` under `effect`.
    ///
    /// [`Self::check`] compares `rows` with `summary`.
    #[must_use]
    pub const fn sufficient_statistics(
        rows: PreservedIdentities,
        summary: PreservedIdentities,
        effect: SummaryEffect,
    ) -> Self {
        Self::new(
            Self::SUFFICIENT_STATISTICS,
            rows,
            Claims::One(summary),
            AttributedEffect::Summary(effect),
        )
    }

    /// Evaluate `tile` as a piece of `full` under `effect`.
    ///
    /// [`Self::check`] compares `full` with `tile`. A faithful tile claims the
    /// full frame.
    #[must_use]
    pub const fn tiled_evaluation(
        full: PreservedIdentities,
        tile: PreservedIdentities,
        effect: TileEffect,
    ) -> Self {
        Self::new(Self::TILED_EVALUATION, full, Claims::One(tile), AttributedEffect::Tile(effect))
    }

    const fn new(
        name: &'static str,
        source: PreservedIdentities,
        claims: Claims,
        effect: AttributedEffect,
    ) -> Self {
        Self { name, source, claims, effect, frames: None }
    }

    /// Bind the executed contents of the source and of each claim.
    ///
    /// `source` is the cohort, preparation, fit, rows, or full frame. `claims`
    /// follow comparison order and are one rewritten frame, or the left and
    /// right consumers of a shared preparation. [`Self::check`] compares
    /// fingerprints of selected rows, folds, population, and history.
    /// Dependence and uncertainty are not in that fingerprint. Matching
    /// identity strings are not those fingerprints.
    ///
    /// # Errors
    ///
    /// `claims` does not have one frame for each compared identity. The detail
    /// is `rewrite_legality.frame_binding`.
    #[must_use = "a mismatched frame binding is a refusal"]
    pub fn with_frames(
        mut self,
        source: &FrameContents,
        claims: &[&FrameContents],
    ) -> Result<Self, RewriteRefusal> {
        self.frames = Some(match (&self.claims, claims) {
            (Claims::One(_), [rewritten]) => ExecutedFrames::Pair {
                source: source.fingerprint(),
                rewritten: rewritten.fingerprint(),
            },
            (Claims::Two(_, _), [left, right]) => ExecutedFrames::Share {
                source: source.fingerprint(),
                left: left.fingerprint(),
                right: right.fingerprint(),
            },
            _ => return Err(frame_binding(self.name)),
        });
        Ok(self)
    }

    /// Stable name of this rewrite.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Frame the rewrite starts from.
    ///
    /// The cohort, the shared preparation, the nuisance fit, the rows, or the
    /// full evaluation.
    #[must_use]
    pub const fn source(&self) -> &PreservedIdentities {
        &self.source
    }

    /// First claim [`Self::check`] compares with [`Self::source`].
    ///
    /// The pushed frame, the left consumer, the nuisance consumer, the
    /// summary, or the tile.
    #[must_use]
    pub const fn primary_claim(&self) -> &PreservedIdentities {
        match &self.claims {
            Claims::One(claim) | Claims::Two(claim, _) => claim,
        }
    }

    /// Right consumer of a shared preparation.
    ///
    /// [`None`] for a rewrite that compares one claim.
    #[must_use]
    pub const fn secondary_claim(&self) -> Option<&PreservedIdentities> {
        match &self.claims {
            Claims::One(_) => None,
            Claims::Two(_, claim) => Some(claim),
        }
    }

    /// Stable name of the effect the caller attributed to this rewrite.
    #[must_use]
    pub const fn effect_name(&self) -> &'static str {
        self.effect.as_str()
    }

    /// Defect of the attributed effect, from the shared defect table.
    ///
    /// [`None`] is the preserving declaration. That declaration does not admit
    /// the rewrite.
    #[must_use]
    pub const fn defect(&self) -> Option<(PreservationAxis, &'static str)> {
        self.effect.defect()
    }

    /// Admit this rewrite when every preservation axis is unchanged.
    ///
    /// Axes are examined in [`PreservationAxis`] order. Within one axis, the
    /// source is compared with each claim in claim order. An effect that names
    /// the axis is a refusal even when every identity string matches. A
    /// fingerprint of the frame contents that differs is a refusal even when
    /// the identity strings match. Equal identity strings without a
    /// fingerprint are a refusal. Dependence and uncertainty are compared as
    /// executed digests like the other four axes.
    ///
    /// # Errors
    ///
    /// The first altered axis. [`RewriteRefusal::next_attempt`] keeps that
    /// axis. Stored fingerprints that do not match the claims are
    /// `rewrite_legality.frame_binding`.
    #[must_use = "a refusal is the counterexample"]
    pub fn check(&self) -> Result<(), RewriteRefusal> {
        let defect = self.effect.defect();
        match (&self.claims, &self.frames) {
            (Claims::One(claim), None) => admit(self.name, &[(&self.source, claim)], defect, &[]),
            (Claims::One(claim), Some(ExecutedFrames::Pair { source, rewritten })) => {
                admit(self.name, &[(&self.source, claim)], defect, &[(source, rewritten)])
            }
            (Claims::Two(left, right), None) => {
                admit(self.name, &[(&self.source, left), (&self.source, right)], defect, &[])
            }
            (
                Claims::Two(left, right),
                Some(ExecutedFrames::Share { source, left: left_frame, right: right_frame }),
            ) => admit(
                self.name,
                &[(&self.source, left), (&self.source, right)],
                defect,
                &[(source, left_frame), (source, right_frame)],
            ),
            (Claims::One(_), Some(ExecutedFrames::Share { .. }))
            | (Claims::Two(_, _), Some(ExecutedFrames::Pair { .. })) => {
                Err(frame_binding(self.name))
            }
        }
    }
}

fn frame_binding(kind: &'static str) -> RewriteRefusal {
    RewriteRefusal {
        code: INVALID_CODE,
        detail: FRAME_BINDING,
        message: format!("{kind} needs one executed fingerprint for each compared frame"),
        kind: Some(kind),
        axis: PreservationAxis::SampleSelection,
        next: RETAIN_SAMPLE_SELECTION,
    }
}

/// A refused rewrite, or a blank identity.
///
/// [`Self::next_attempt`] is the named attempt. [`Display`] prints it after
/// the detail and the message. Library paths return this value; they do not panic.
///
/// [`Display`]: fmt::Display
#[derive(Clone, Debug, Eq, PartialEq)]
#[must_use = "the next attempt keeps the axis the rewrite would change"]
pub struct RewriteRefusal {
    code: &'static str,
    detail: &'static str,
    message: String,
    kind: Option<&'static str>,
    axis: PreservationAxis,
    next: NextAttempt,
}

impl RewriteRefusal {
    /// Registered reason code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }

    /// Namespaced detail, `rewrite_legality.<rule>`.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        self.detail
    }

    /// What was refused, without the next-attempt sentence.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Rewrite that was refused. [`None`] when the identity itself is blank.
    #[must_use]
    pub const fn kind(&self) -> Option<&'static str> {
        self.kind
    }

    /// Axis that is blank or that the rewrite would alter.
    #[must_use]
    pub const fn axis(&self) -> PreservationAxis {
        self.axis
    }

    /// Named attempt after this refusal.
    #[must_use = "the next attempt names how the rewrite stays legal"]
    pub const fn next_attempt(&self) -> NextAttempt {
        self.next
    }
}

impl fmt::Display for RewriteRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let detail = self.detail;
        let sentence = self.next.sentence();
        write!(formatter, "{detail}: {message}; next attempt: {sentence}", message = self.message)
    }
}

impl Error for RewriteRefusal {}

fn altered(kind: &'static str, axis: PreservationAxis, because: &str) -> RewriteRefusal {
    RewriteRefusal {
        code: INVALID_CODE,
        detail: axis.detail(),
        message: format!("{kind} would alter {}: {because}", axis.phrase()),
        kind: Some(kind),
        axis,
        next: axis.next_attempt(),
    }
}

fn blank(axis: PreservationAxis) -> RewriteRefusal {
    RewriteRefusal {
        code: INVALID_CODE,
        detail: BLANK_DETAIL,
        message: format!("the {} identity is blank", axis.phrase()),
        kind: None,
        axis,
        next: NAME_PRESERVED_IDENTITY,
    }
}

fn admit(
    kind: &'static str,
    pairs: &[(&PreservedIdentities, &PreservedIdentities)],
    defect: Option<(PreservationAxis, &str)>,
    frames: &[(&FrameFingerprint, &FrameFingerprint)],
) -> Result<(), RewriteRefusal> {
    for axis in PreservationAxis::ALL {
        if let Some((forced, because)) = defect {
            if forced == axis {
                return Err(altered(kind, axis, because));
            }
        }
        for (original, rewritten) in pairs {
            let original_id = original.identity(axis);
            let rewritten_id = rewritten.identity(axis);
            if original_id != rewritten_id {
                return Err(altered(
                    kind,
                    axis,
                    &format!("`{original_id}` is not `{rewritten_id}`"),
                ));
            }
        }
        for (original, rewritten) in frames {
            let original_digest = original.on_axis(axis);
            let rewritten_digest = rewritten.on_axis(axis);
            if original_digest != rewritten_digest {
                return Err(altered(
                    kind,
                    axis,
                    &format!("`{original_digest}` is not `{rewritten_digest}`"),
                ));
            }
        }
    }
    // The preserving effect is the caller's declaration. Equal identity strings
    // do not supply the executed fingerprint, so they do not admit the rewrite.
    if frames.is_empty() {
        return Err(altered(kind, PreservationAxis::SampleSelection, NO_FINGERPRINT));
    }
    Ok(())
}

fn content_refusal(axis: PreservationAxis, message: &str) -> RewriteRefusal {
    RewriteRefusal {
        code: INVALID_CODE,
        detail: axis.detail(),
        message: message.to_owned(),
        kind: None,
        axis,
        next: axis.next_attempt(),
    }
}

fn check_text(axis: PreservationAxis, what: &str, text: &str) -> Result<(), RewriteRefusal> {
    if is_blank(text) {
        return Err(content_refusal(axis, &format!("{what} is blank")));
    }
    Ok(())
}

fn digest_text(tag: &str, text: &str) -> SemanticDigest {
    let mut writer = CanonicalWriter::new(tag);
    writer.text(text);
    SemanticDigest::from_bytes(writer.finish())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::error::Error;

    use super::{
        FrameContents, FrameLedger, FrameRow, NAME_PRESERVED_IDENTITY, NEXT_ATTEMPTS,
        NuisanceEffect, PreparationEffect, PreservationAxis, PreservedIdentities, PushdownEffect,
        RETAIN_DEPENDENCE, RETAIN_SAMPLE_SELECTION, RETAIN_UNCERTAINTY, Rewrite, SummaryEffect,
        TileEffect, check_executed, combine_ids, value_row_id,
    };
    use antecedent_core::NextAttempt;

    fn identities() -> PreservedIdentities {
        PreservedIdentities::try_new(
            "sample",
            "folds",
            "population",
            "history",
            "dependence",
            "uncertainty",
        )
        .expect("preserved identities")
    }

    fn identities_with(axis: PreservationAxis, value: &str) -> PreservedIdentities {
        let mut sample_selection = "sample";
        let mut folds = "folds";
        let mut population = "population";
        let mut history = "history";
        let mut dependence = "dependence";
        let mut uncertainty = "uncertainty";
        match axis {
            PreservationAxis::SampleSelection => sample_selection = value,
            PreservationAxis::Folds => folds = value,
            PreservationAxis::Population => population = value,
            PreservationAxis::History => history = value,
            PreservationAxis::Dependence => dependence = value,
            PreservationAxis::Uncertainty => uncertainty = value,
        }
        PreservedIdentities::try_new(
            sample_selection,
            folds,
            population,
            history,
            dependence,
            uncertainty,
        )
        .expect("moved identity")
    }

    fn contents() -> FrameContents {
        FrameContents::try_new(&["row-a", "row-b"], &["fold-1", "fold-2"], "pop-1", "hist-1")
            .expect("frame contents")
    }

    fn bind_pair(rewrite: Rewrite, rows: &FrameContents) -> Rewrite {
        rewrite.with_frames(rows, &[rows]).expect("executed frame")
    }

    fn bind_share(rewrite: Rewrite, rows: &FrameContents) -> Rewrite {
        rewrite.with_frames(rows, &[rows, rows]).expect("executed frame")
    }

    #[track_caller]
    fn expect_refusal(rewrite: &Rewrite, axis: PreservationAxis, because: &str) {
        let err = rewrite.check().expect_err(rewrite.name());
        assert_eq!(err.kind(), Some(rewrite.name()));
        assert_eq!(err.axis(), axis);
        assert_eq!(err.next_attempt(), axis.next_attempt());
        assert_eq!(err.code(), "invalid_argument");
        assert_eq!(err.detail(), format!("rewrite_legality.{axis}"));
        assert!(err.message().contains(rewrite.name()), "{}", err.message());
        assert!(err.message().contains(axis.phrase()), "{}", err.message());
        assert!(err.message().contains(because), "{}", err.message());
        let text = err.to_string();
        assert!(text.starts_with(err.detail()));
        assert!(text.contains("next attempt:"));
        assert!(text.contains(err.next_attempt().sentence()));
        let _: &dyn Error = &err;
    }

    macro_rules! assert_effect_table {
        ($ty:ident) => {{
            assert!($ty::from_name("not_an_effect").is_none());
            let mut effects = $ty::ALL.into_iter();
            let preserving = effects.next().expect("preserving declaration");
            assert!(preserving.defect().is_none());
            assert_eq!($ty::from_name(preserving.as_str()), Some(preserving));
            let mut names = BTreeSet::new();
            assert!(names.insert(preserving.as_str()));
            for axis in PreservationAxis::ALL {
                let effect = effects.next().expect("alteration");
                let (found, because) = effect.defect().expect("defect");
                assert_eq!(found, axis);
                assert!(!because.is_empty());
                assert_eq!($ty::from_name(effect.as_str()), Some(effect));
                assert!(names.insert(effect.as_str()), "{}", effect.as_str());
            }
            assert!(effects.next().is_none());
        }};
    }

    #[test]
    fn effect_tables_follow_the_preservation_axes() {
        assert_effect_table!(PushdownEffect);
        assert_effect_table!(PreparationEffect);
        assert_effect_table!(NuisanceEffect);
        assert_effect_table!(SummaryEffect);
        assert_effect_table!(TileEffect);
    }

    #[test]
    fn preserved_frame_is_legal_for_every_rewrite() {
        let frame = identities();
        let rows = contents();
        let pushdown = bind_pair(
            Rewrite::column_filter_pushdown(frame.clone(), frame.clone(), PushdownEffect::Declared),
            &rows,
        );
        let preparation = bind_share(
            Rewrite::shared_row_design_preparation(
                frame.clone(),
                frame.clone(),
                frame.clone(),
                PreparationEffect::Shared,
            ),
            &rows,
        );
        let nuisance = bind_pair(
            Rewrite::nuisance_reuse(frame.clone(), frame.clone(), NuisanceEffect::Certified),
            &rows,
        );
        let summary = bind_pair(
            Rewrite::sufficient_statistics(frame.clone(), frame.clone(), SummaryEffect::Sufficient),
            &rows,
        );
        let tiled =
            bind_pair(Rewrite::tiled_evaluation(frame.clone(), frame, TileEffect::Faithful), &rows);
        assert!(pushdown.defect().is_none());
        assert!(preparation.defect().is_none());
        assert!(nuisance.defect().is_none());
        assert!(summary.defect().is_none());
        assert!(tiled.defect().is_none());
        assert!(matches!(pushdown.check(), Ok(())));
        assert!(matches!(preparation.check(), Ok(())));
        assert!(matches!(nuisance.check(), Ok(())));
        assert!(matches!(summary.check(), Ok(())));
        assert!(matches!(tiled.check(), Ok(())));
        assert_eq!(pushdown.name(), Rewrite::COLUMN_FILTER_PUSHDOWN);
        assert_eq!(preparation.name(), Rewrite::SHARED_ROW_DESIGN_PREPARATION);
        assert_eq!(nuisance.name(), Rewrite::NUISANCE_REUSE);
        assert_eq!(summary.name(), Rewrite::SUFFICIENT_STATISTICS);
        assert_eq!(tiled.name(), Rewrite::TILED_EVALUATION);
        assert_eq!(pushdown.effect_name(), PushdownEffect::Declared.as_str());
        assert_eq!(preparation.effect_name(), PreparationEffect::Shared.as_str());
        assert_eq!(nuisance.effect_name(), NuisanceEffect::Certified.as_str());
        assert_eq!(summary.effect_name(), SummaryEffect::Sufficient.as_str());
        assert_eq!(tiled.effect_name(), TileEffect::Faithful.as_str());
        assert_eq!(pushdown.source().identity(PreservationAxis::Folds), "folds");
        assert_eq!(pushdown.primary_claim().identity(PreservationAxis::Folds), "folds");
        assert_eq!(
            preparation.primary_claim().identity(PreservationAxis::Population),
            preparation
                .secondary_claim()
                .expect("right consumer")
                .identity(PreservationAxis::Population),
        );
        assert_eq!(
            nuisance.source().identity(PreservationAxis::History),
            nuisance.primary_claim().identity(PreservationAxis::History),
        );
        assert_eq!(
            summary.source().identity(PreservationAxis::Dependence),
            summary.primary_claim().identity(PreservationAxis::Dependence),
        );
        assert_eq!(
            tiled.source().identity(PreservationAxis::Uncertainty),
            tiled.primary_claim().identity(PreservationAxis::Uncertainty),
        );
    }

    #[test]
    fn matching_identities_do_not_admit_an_altering_effect() {
        let frame = identities();
        for effect in PushdownEffect::ALL {
            let Some((axis, because)) = effect.defect() else { continue };
            expect_refusal(
                &Rewrite::column_filter_pushdown(frame.clone(), frame.clone(), effect),
                axis,
                because,
            );
        }
        for effect in PreparationEffect::ALL {
            let Some((axis, because)) = effect.defect() else { continue };
            expect_refusal(
                &Rewrite::shared_row_design_preparation(
                    frame.clone(),
                    frame.clone(),
                    frame.clone(),
                    effect,
                ),
                axis,
                because,
            );
        }
        for effect in NuisanceEffect::ALL {
            let Some((axis, because)) = effect.defect() else { continue };
            expect_refusal(
                &Rewrite::nuisance_reuse(frame.clone(), frame.clone(), effect),
                axis,
                because,
            );
        }
        for effect in SummaryEffect::ALL {
            let Some((axis, because)) = effect.defect() else { continue };
            expect_refusal(
                &Rewrite::sufficient_statistics(frame.clone(), frame.clone(), effect),
                axis,
                because,
            );
        }
        for effect in TileEffect::ALL {
            let Some((axis, because)) = effect.defect() else { continue };
            expect_refusal(
                &Rewrite::tiled_evaluation(frame.clone(), frame.clone(), effect),
                axis,
                because,
            );
        }
    }

    #[test]
    fn faithful_effect_refuses_a_moved_identity() {
        let original = identities();
        for axis in PreservationAxis::ALL {
            let token = format!("{axis}-other");
            let moved = identities_with(axis, &token);
            expect_refusal(
                &Rewrite::column_filter_pushdown(
                    original.clone(),
                    moved.clone(),
                    PushdownEffect::Declared,
                ),
                axis,
                &token,
            );
            expect_refusal(
                &Rewrite::shared_row_design_preparation(
                    original.clone(),
                    original.clone(),
                    moved.clone(),
                    PreparationEffect::Shared,
                ),
                axis,
                &token,
            );
            expect_refusal(
                &Rewrite::nuisance_reuse(
                    original.clone(),
                    moved.clone(),
                    NuisanceEffect::Certified,
                ),
                axis,
                &token,
            );
            expect_refusal(
                &Rewrite::sufficient_statistics(
                    original.clone(),
                    moved.clone(),
                    SummaryEffect::Sufficient,
                ),
                axis,
                &token,
            );
            expect_refusal(
                &Rewrite::tiled_evaluation(original.clone(), moved, TileEffect::Faithful),
                axis,
                &token,
            );
        }
    }

    #[test]
    fn earlier_axis_is_the_refusal() {
        let original = identities();
        let pushed = PreservedIdentities::try_new(
            "sample",
            "folds",
            "population-other",
            "history",
            "dependence",
            "uncertainty",
        )
        .expect("population");
        let pushdown = Rewrite::column_filter_pushdown(
            original.clone(),
            pushed,
            PushdownEffect::OmitsRequiredColumn,
        );
        let err = pushdown.check().expect_err("sample selection is earlier");
        assert_eq!(err.axis(), PreservationAxis::SampleSelection);
        assert_eq!(err.next_attempt(), RETAIN_SAMPLE_SELECTION);
        assert!(err.message().contains("cohort closure"));
        assert!(!err.message().contains("population-other"));

        let both = PreservedIdentities::try_new(
            "sample-other",
            "folds",
            "population",
            "history",
            "dependence",
            "uncertainty-other",
        )
        .expect("two axes");
        let declared = Rewrite::column_filter_pushdown(original, both, PushdownEffect::Declared);
        let err = declared.check().expect_err("sample selection is earlier than uncertainty");
        assert_eq!(err.axis(), PreservationAxis::SampleSelection);
        assert!(err.message().contains("sample-other"));
        assert!(!err.message().contains("uncertainty-other"));
    }

    #[test]
    fn shared_preparation_names_the_first_consumer_on_one_axis() {
        let preparation = identities();
        let left = identities_with(PreservationAxis::Dependence, "left-dependence");
        let right = identities_with(PreservationAxis::Dependence, "right-dependence");
        let shared = Rewrite::shared_row_design_preparation(
            preparation,
            left,
            right,
            PreparationEffect::Shared,
        );
        let err = shared.check().expect_err("left consumer");
        assert_eq!(err.axis(), PreservationAxis::Dependence);
        assert_eq!(err.next_attempt(), RETAIN_DEPENDENCE);
        assert!(err.message().contains("left-dependence"));
        assert!(!err.message().contains("right-dependence"));
    }

    #[test]
    fn blank_identity_names_the_missing_axis() {
        let base = ["sample", "folds", "population", "history", "dependence", "uncertainty"];
        for (index, axis) in PreservationAxis::ALL.into_iter().enumerate() {
            let mut parts = base;
            parts[index] = " \t";
            let err = PreservedIdentities::try_new(
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
            )
            .expect_err("blank");
            assert_eq!(err.axis(), axis);
            assert_eq!(err.kind(), None);
            assert_eq!(err.code(), "invalid_argument");
            assert_eq!(err.detail(), "rewrite_legality.blank_identity");
            assert_eq!(err.next_attempt(), NAME_PRESERVED_IDENTITY);
            assert_eq!(err.next_attempt().as_str(), "name_preserved_identity");
            assert!(err.message().contains(axis.phrase()));
            assert!(err.to_string().contains(err.next_attempt().sentence()));
            let _: &dyn Error = &err;
        }
    }

    #[test]
    fn axis_and_attempt_names_round_trip() {
        assert_eq!(PreservationAxis::ALL.len(), 6);
        assert_eq!(NEXT_ATTEMPTS.len(), 7);
        assert_eq!(PreservationAxis::from_name("row_index"), None);
        assert_eq!(NextAttempt::from_name(&NEXT_ATTEMPTS, "drop_rows"), None);
        for axis in PreservationAxis::ALL {
            assert_eq!(PreservationAxis::from_name(axis.as_str()), Some(axis));
            assert_eq!(axis.to_string(), axis.as_str());
            let attempt = axis.next_attempt();
            assert_ne!(attempt, NAME_PRESERVED_IDENTITY);
            assert_eq!(NextAttempt::from_name(&NEXT_ATTEMPTS, attempt.as_str()), Some(attempt));
            assert_eq!(attempt.to_string(), attempt.sentence());
        }
        assert_eq!(
            NextAttempt::from_name(&NEXT_ATTEMPTS, "name_preserved_identity"),
            Some(NAME_PRESERVED_IDENTITY)
        );
    }

    #[test]
    fn equal_identity_strings_with_a_different_selected_row_fingerprint_are_refused() {
        let frame = identities();
        let original = contents();
        let bare =
            Rewrite::column_filter_pushdown(frame.clone(), frame.clone(), PushdownEffect::Declared);
        expect_refusal(&bare, PreservationAxis::SampleSelection, "fingerprint");
        assert_eq!(
            bare.source().identity(PreservationAxis::SampleSelection),
            bare.primary_claim().identity(PreservationAxis::SampleSelection),
        );

        let cases = [
            (
                &["row-a", "row-c"][..],
                &["fold-1", "fold-2"][..],
                "pop-1",
                "hist-1",
                PreservationAxis::SampleSelection,
            ),
            (
                &["row-a", "row-b"][..],
                &["fold-1", "fold-9"][..],
                "pop-1",
                "hist-1",
                PreservationAxis::Folds,
            ),
            (
                &["row-a", "row-b"][..],
                &["fold-1", "fold-2"][..],
                "pop-other",
                "hist-1",
                PreservationAxis::Population,
            ),
            (
                &["row-a", "row-b"][..],
                &["fold-1", "fold-2"][..],
                "pop-1",
                "hist-other",
                PreservationAxis::History,
            ),
        ];
        for (rows, folds, population, history, axis) in cases {
            let rewritten = FrameContents::try_new(rows, folds, population, history)
                .expect("rewritten contents");
            let digest = rewritten.fingerprint().on_axis(axis);
            assert_ne!(original.fingerprint().on_axis(axis), digest);
            let declared = Rewrite::column_filter_pushdown(
                frame.clone(),
                frame.clone(),
                PushdownEffect::Declared,
            )
            .with_frames(&original, &[&rewritten])
            .expect("executed frame");
            assert_eq!(declared.source().identity(axis), declared.primary_claim().identity(axis));
            expect_refusal(&declared, axis, &digest.to_string());
        }
    }

    #[test]
    fn matching_frame_fingerprint_is_admitted() {
        let frame = identities();
        let original =
            FrameContents::try_new(&["row-b", "row-a"], &["fold-1", "fold-2"], "pop-1", "hist-1")
                .expect("crossed assignment");
        let same =
            FrameContents::try_new(&["row-a", "row-b"], &["fold-2", "fold-1"], "pop-1", "hist-1")
                .expect("listed assignment");
        assert_eq!(original.fingerprint(), same.fingerprint());
        assert_eq!(
            original.fingerprint().dependence(),
            original.fingerprint().on_axis(PreservationAxis::Dependence),
        );
        assert_eq!(original.row_ids().collect::<Vec<&str>>(), ["row-a", "row-b"]);
        assert_eq!(original.fold_ids().collect::<Vec<&str>>(), ["fold-2", "fold-1"]);
        let pushdown =
            Rewrite::column_filter_pushdown(frame.clone(), frame, PushdownEffect::Declared)
                .with_frames(&original, &[&same])
                .expect("executed frame");
        assert!(matches!(pushdown.check(), Ok(())));
    }

    #[test]
    fn equal_declaration_strings_without_an_executed_fingerprint_are_refused() {
        let frame = identities();
        let pushdown =
            Rewrite::column_filter_pushdown(frame.clone(), frame.clone(), PushdownEffect::Declared);
        let preparation = Rewrite::shared_row_design_preparation(
            frame.clone(),
            frame.clone(),
            frame.clone(),
            PreparationEffect::Shared,
        );
        let nuisance =
            Rewrite::nuisance_reuse(frame.clone(), frame.clone(), NuisanceEffect::Certified);
        let summary =
            Rewrite::sufficient_statistics(frame.clone(), frame.clone(), SummaryEffect::Sufficient);
        let tiled = Rewrite::tiled_evaluation(frame.clone(), frame.clone(), TileEffect::Faithful);
        for axis in PreservationAxis::ALL {
            assert_eq!(pushdown.source().identity(axis), pushdown.primary_claim().identity(axis));
            assert_eq!(
                preparation.source().identity(axis),
                preparation.primary_claim().identity(axis),
            );
            assert_eq!(
                preparation.source().identity(axis),
                preparation.secondary_claim().expect("right consumer").identity(axis),
            );
            assert_eq!(nuisance.source().identity(axis), nuisance.primary_claim().identity(axis));
            assert_eq!(summary.source().identity(axis), summary.primary_claim().identity(axis));
            assert_eq!(tiled.source().identity(axis), tiled.primary_claim().identity(axis));
        }
        assert!(pushdown.defect().is_none());
        assert!(preparation.defect().is_none());
        assert!(nuisance.defect().is_none());
        assert!(summary.defect().is_none());
        assert!(tiled.defect().is_none());
        expect_refusal(&pushdown, PreservationAxis::SampleSelection, "fingerprint");
        expect_refusal(&preparation, PreservationAxis::SampleSelection, "fingerprint");
        expect_refusal(&nuisance, PreservationAxis::SampleSelection, "fingerprint");
        expect_refusal(&summary, PreservationAxis::SampleSelection, "fingerprint");
        expect_refusal(&tiled, PreservationAxis::SampleSelection, "fingerprint");
    }

    #[test]
    fn dependence_and_uncertainty_declarations_do_not_replace_the_fingerprint() {
        let rows = contents();
        let original = identities();
        for axis in [PreservationAxis::Dependence, PreservationAxis::Uncertainty] {
            let moved = identities_with(axis, "other-declaration");
            let declared = [
                bind_pair(
                    Rewrite::column_filter_pushdown(
                        original.clone(),
                        moved.clone(),
                        PushdownEffect::Declared,
                    ),
                    &rows,
                ),
                bind_share(
                    Rewrite::shared_row_design_preparation(
                        original.clone(),
                        original.clone(),
                        moved.clone(),
                        PreparationEffect::Shared,
                    ),
                    &rows,
                ),
                bind_pair(
                    Rewrite::nuisance_reuse(
                        original.clone(),
                        moved.clone(),
                        NuisanceEffect::Certified,
                    ),
                    &rows,
                ),
                bind_pair(
                    Rewrite::sufficient_statistics(
                        original.clone(),
                        moved.clone(),
                        SummaryEffect::Sufficient,
                    ),
                    &rows,
                ),
                bind_pair(
                    Rewrite::tiled_evaluation(original.clone(), moved, TileEffect::Faithful),
                    &rows,
                ),
            ];
            for rewrite in &declared {
                assert!(rewrite.defect().is_none());
                assert_eq!(
                    rewrite.source().identity(PreservationAxis::SampleSelection),
                    rewrite.primary_claim().identity(PreservationAxis::SampleSelection),
                );
                expect_refusal(rewrite, axis, "other-declaration");
            }
        }
    }

    #[test]
    fn a_fingerprint_must_cover_every_compared_frame() {
        let frame = identities();
        let rows = contents();
        let pushdown =
            Rewrite::column_filter_pushdown(frame.clone(), frame.clone(), PushdownEffect::Declared);
        let err = pushdown.with_frames(&rows, &[&rows, &rows]).expect_err("two claims");
        assert_eq!(err.detail(), "rewrite_legality.frame_binding");
        assert_eq!(err.kind(), Some(Rewrite::COLUMN_FILTER_PUSHDOWN));
        assert_eq!(err.axis(), PreservationAxis::SampleSelection);
        assert_eq!(err.next_attempt(), RETAIN_SAMPLE_SELECTION);
        assert!(err.message().contains("fingerprint"));

        let preparation = Rewrite::shared_row_design_preparation(
            frame.clone(),
            frame.clone(),
            frame,
            PreparationEffect::Shared,
        );
        let err = preparation.with_frames(&rows, &[&rows]).expect_err("one claim");
        assert_eq!(err.detail(), "rewrite_legality.frame_binding");
        assert_eq!(err.kind(), Some(Rewrite::SHARED_ROW_DESIGN_PREPARATION));
    }

    #[test]
    fn identity_text_is_not_length_capped() {
        let long = "x".repeat(10_000);
        assert!(PreservedIdentities::try_new("a", "b", "c", &long, "e", "f").is_ok());
        assert!(FrameContents::try_new(&[long.as_str()], &["f"], "p", "h").is_ok());
        assert!(FrameContents::try_new(&["r"], &["f"], &long, "h").is_ok());
    }

    fn executed(dependence: [&str; 3], uncertainty: [&str; 3], law: &str) -> FrameContents {
        let rows = ["r1", "r2", "r3"];
        let frame_rows: Vec<FrameRow<'_>> = (0..3)
            .map(|i| FrameRow {
                row: rows[i],
                fold: if i == 2 { "f2" } else { "f1" },
                dependence: dependence[i],
                uncertainty: uncertainty[i],
            })
            .collect();
        FrameContents::try_new_executed(&frame_rows, "pop", "hist", law).expect("executed frame")
    }

    #[test]
    fn ledger_streaming_agrees_with_listed_contents_in_any_order() {
        let listed = executed(["c1", "c1", "c2"], ["w1", "w2", "w3"], "law");
        let mut ledger = FrameLedger::try_new("pop", "hist", "law").expect("ledger");
        ledger.record("r3", "f2", "c2", "w3");
        ledger.record("r1", "f1", "c1", "w1");
        ledger.record("r2", "f1", "c1", "w2");
        assert_eq!(ledger.rows(), 3);
        assert_eq!(ledger.fingerprint(), listed.fingerprint());

        let mut forward = FrameLedger::try_new("pop", "hist", "law").expect("ledger");
        let mut reverse = FrameLedger::try_new("pop", "hist", "law").expect("ledger");
        for row in 0..50 {
            forward.record_numeric(row, row % 3, row / 5, 1);
            reverse.record_numeric(49 - row, (49 - row) % 3, (49 - row) / 5, 1);
        }
        assert_eq!(forward.fingerprint(), reverse.fingerprint());
        let mut moved = forward.clone();
        moved.record_numeric(50, 0, 0, 1);
        assert_ne!(moved.fingerprint(), forward.fingerprint());
        let mut reweighted = FrameLedger::try_new("pop", "hist", "law").expect("ledger");
        for row in 0..50 {
            reweighted.record_numeric(row, row % 3, row / 5, u64::from(row == 7) + 1);
        }
        let err =
            check_executed("tiled_evaluation", &forward.fingerprint(), &[reweighted.fingerprint()])
                .expect_err("one state changed");
        assert_eq!(err.axis(), PreservationAxis::Uncertainty);
        assert_ne!(value_row_id(1, [1.0, 2.0]), value_row_id(1, [1.0, 2.5]));
        assert_ne!(combine_ids(&[1, 2]), combine_ids(&[2, 1]));
    }

    #[test]
    fn a_repeated_executed_row_changes_the_fingerprint() {
        let mut once = FrameLedger::try_new("pop", "hist", "law").expect("ledger");
        let mut twice = FrameLedger::try_new("pop", "hist", "law").expect("ledger");
        once.record("r1", "f", "c", "w");
        twice.record("r1", "f", "c", "w");
        twice.record("r1", "f", "c", "w");
        assert_ne!(once.fingerprint(), twice.fingerprint());
        let err = check_executed("tiled_evaluation", &once.fingerprint(), &[twice.fingerprint()])
            .expect_err("duplicated row");
        assert_eq!(err.axis(), PreservationAxis::SampleSelection);
    }

    #[test]
    fn a_rewrite_that_drops_dependence_is_refused_whatever_it_is_called() {
        let original = executed(["c1", "c1", "c2"], ["w1", "w2", "w3"], "law");
        let collapsed = executed(["c1", "c1", "c1"], ["w1", "w2", "w3"], "law");
        let undeclared =
            FrameContents::try_new(&["r1", "r2", "r3"], &["f1", "f1", "f2"], "pop", "hist")
                .expect("undeclared dependence");
        let frame = identities();
        for rewritten in [&collapsed, &undeclared] {
            let err = check_executed(
                Rewrite::SUFFICIENT_STATISTICS,
                &original.fingerprint(),
                &[rewritten.fingerprint()],
            )
            .expect_err("dependence dropped");
            assert_eq!(err.axis(), PreservationAxis::Dependence);
            assert_eq!(err.next_attempt(), RETAIN_DEPENDENCE);

            let described = Rewrite::sufficient_statistics(
                frame.clone(),
                frame.clone(),
                SummaryEffect::Sufficient,
            )
            .with_frames(&original, &[rewritten])
            .expect("executed frames");
            assert_eq!(
                described.source().identity(PreservationAxis::Dependence),
                described.primary_claim().identity(PreservationAxis::Dependence),
            );
            expect_refusal(&described, PreservationAxis::Dependence, "is not");
        }
    }

    #[test]
    fn a_rewrite_that_drops_uncertainty_is_refused_whatever_it_is_called() {
        let original = executed(["c1", "c1", "c2"], ["w1", "w2", "w3"], "law");
        let reweighted = executed(["c1", "c1", "c2"], ["w1", "w2", "w9"], "law");
        let other_law = executed(["c1", "c1", "c2"], ["w1", "w2", "w3"], "point_only");
        let frame = identities();
        for rewritten in [&reweighted, &other_law] {
            let err = check_executed(
                Rewrite::TILED_EVALUATION,
                &original.fingerprint(),
                &[rewritten.fingerprint()],
            )
            .expect_err("uncertainty dropped");
            assert_eq!(err.axis(), PreservationAxis::Uncertainty);
            assert_eq!(err.next_attempt(), RETAIN_UNCERTAINTY);

            let described =
                Rewrite::tiled_evaluation(frame.clone(), frame.clone(), TileEffect::Faithful)
                    .with_frames(&original, &[rewritten])
                    .expect("executed frames");
            expect_refusal(&described, PreservationAxis::Uncertainty, "is not");
        }
        assert!(
            check_executed(
                Rewrite::TILED_EVALUATION,
                &original.fingerprint(),
                &[executed(["c1", "c1", "c2"], ["w1", "w2", "w3"], "law").fingerprint()],
            )
            .is_ok()
        );
    }

    #[test]
    fn a_forged_matching_description_cannot_admit_changed_rows() {
        let original = executed(["c1", "c1", "c2"], ["w1", "w2", "w3"], "law");
        let narrowed = FrameContents::try_new_executed(
            &[
                FrameRow { row: "r1", fold: "f1", dependence: "c1", uncertainty: "w1" },
                FrameRow { row: "r2", fold: "f1", dependence: "c1", uncertainty: "w2" },
            ],
            "pop",
            "hist",
            "law",
        )
        .expect("narrowed frame");
        let frame = identities();
        let forged =
            Rewrite::column_filter_pushdown(frame.clone(), frame, PushdownEffect::Declared)
                .with_frames(&original, &[&narrowed])
                .expect("executed frames");
        expect_refusal(&forged, PreservationAxis::SampleSelection, "is not");
        let err = check_executed(
            Rewrite::COLUMN_FILTER_PUSHDOWN,
            &original.fingerprint(),
            &[narrowed.fingerprint()],
        )
        .expect_err("rows dropped");
        assert_eq!(err.axis(), PreservationAxis::SampleSelection);
    }

    #[test]
    fn a_shared_preparation_is_refused_when_either_consumer_differs() {
        let original = executed(["c1", "c1", "c2"], ["w1", "w2", "w3"], "law");
        let same = executed(["c1", "c1", "c2"], ["w1", "w2", "w3"], "law");
        let mixed = executed(["c1", "c2", "c2"], ["w1", "w2", "w3"], "law");
        assert!(
            check_executed(
                Rewrite::SHARED_ROW_DESIGN_PREPARATION,
                &original.fingerprint(),
                &[same.fingerprint(), same.fingerprint()],
            )
            .is_ok()
        );
        let err = check_executed(
            Rewrite::SHARED_ROW_DESIGN_PREPARATION,
            &original.fingerprint(),
            &[same.fingerprint(), mixed.fingerprint()],
        )
        .expect_err("right consumer mixes dependence");
        assert_eq!(err.axis(), PreservationAxis::Dependence);
        assert_eq!(err.kind(), Some(Rewrite::SHARED_ROW_DESIGN_PREPARATION));
    }

    #[test]
    fn an_executed_check_needs_a_rewritten_frame_and_valid_ids() {
        let original = executed(["c1", "c1", "c2"], ["w1", "w2", "w3"], "law");
        let err = check_executed(Rewrite::NUISANCE_REUSE, &original.fingerprint(), &[])
            .expect_err("no rewritten frame");
        assert_eq!(err.detail(), "rewrite_legality.frame_binding");

        let blank = FrameContents::try_new_executed(
            &[FrameRow { row: "r1", fold: "f", dependence: " ", uncertainty: "w" }],
            "pop",
            "hist",
            "law",
        )
        .expect_err("blank dependence");
        assert_eq!(blank.axis(), PreservationAxis::Dependence);
        let blank = FrameContents::try_new_executed(
            &[FrameRow { row: "r1", fold: "f", dependence: "c", uncertainty: "w" }],
            "pop",
            "hist",
            " ",
        )
        .expect_err("blank sampling law");
        assert_eq!(blank.axis(), PreservationAxis::Uncertainty);
        let err = FrameLedger::try_new("pop", "", "law").expect_err("blank history");
        assert_eq!(err.axis(), PreservationAxis::History);
    }
}
