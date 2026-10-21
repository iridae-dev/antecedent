// SPDX-License-Identifier: MIT OR Apache-2.0

//! Legality of physical rewrites against the scientific frame.
//!
//! A column or filter pushdown, a shared row and design preparation, a
//! nuisance reuse, a sufficient-statistics lowering, and a tiled evaluation
//! are separate types. [`RewriteKind::check`] admits a rewrite only when it
//! preserves sample selection, folds, population, history, dependence, and
//! uncertainty. The check compares a fingerprint of the frame contents and
//! the effect the caller attributes to the rewrite. The fingerprint is the
//! selected row ids, each row's fold id, the population id, and the history.
//! An identity string the caller copies onto both sides is not that
//! fingerprint. The check does not execute the rewrite, scan rows, or number
//! them.
//!
//! An altered axis is a counterexample. The [`RewriteRefusal`] names that axis
//! and the next attempt that keeps it. Copying the same identity text onto an
//! altering effect, or onto a frame whose fingerprint differs, does not make
//! the rewrite legal. When several axes would change, the earliest axis in
//! [`PreservationAxis`] order is the refusal.

use std::{collections::BTreeSet, error::Error, fmt, sync::Arc};

use antecedent_core::{CanonicalWriter, SemanticDigest, text::is_blank};

const INVALID_CODE: &str = antecedent_core::reason_code!("invalid_argument");
const BLANK_DETAIL: &str = "rewrite_legality.blank_identity";
const TAG_SELECTED_ROWS: &str = "rewrite_legality.selected_rows";
const TAG_FOLDS: &str = "rewrite_legality.fold_assignment";
const TAG_POPULATION: &str = "rewrite_legality.population_id";
const TAG_HISTORY: &str = "rewrite_legality.history";
const NO_FINGERPRINT: &str =
    "it carries no fingerprint of the selected rows, folds, population, and history";

/// An axis a legal rewrite leaves unchanged.
///
/// Declaration order is the order [`RewriteKind::check`] reports alterations.
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

impl PreservationAxis {
    /// Every axis, in refusal order.
    pub const ALL: [Self; 6] = [
        Self::SampleSelection,
        Self::Folds,
        Self::Population,
        Self::History,
        Self::Dependence,
        Self::Uncertainty,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SampleSelection => "sample_selection",
            Self::Folds => "folds",
            Self::Population => "population",
            Self::History => "history",
            Self::Dependence => "dependence",
            Self::Uncertainty => "uncertainty",
        }
    }

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
            Self::SampleSelection => NextAttempt::RetainSampleSelection,
            Self::Folds => NextAttempt::RetainFolds,
            Self::Population => NextAttempt::RetainPopulation,
            Self::History => NextAttempt::RetainHistory,
            Self::Dependence => NextAttempt::RetainDependence,
            Self::Uncertainty => NextAttempt::RetainUncertainty,
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use = "an unknown name is not an axis"]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|axis| axis.as_str() == name)
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

/// What to do after a rewrite is refused.
///
/// [`Self::sentence`] is the attempt. [`Self::as_str`] is its stable name.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum NextAttempt {
    /// Keep the rows the unrewritten frame selected.
    RetainSampleSelection,
    /// Keep the declared fold assignment.
    RetainFolds,
    /// Keep the declared population.
    RetainPopulation,
    /// Keep the declared history.
    RetainHistory,
    /// Keep the declared dependence.
    RetainDependence,
    /// Keep the declared uncertainty.
    RetainUncertainty,
    /// Replace a blank identity before checking a rewrite.
    NamePreservedIdentity,
}

impl NextAttempt {
    /// Every attempt, in refusal order, then the blank-identity attempt.
    pub const ALL: [Self; 7] = [
        Self::RetainSampleSelection,
        Self::RetainFolds,
        Self::RetainPopulation,
        Self::RetainHistory,
        Self::RetainDependence,
        Self::RetainUncertainty,
        Self::NamePreservedIdentity,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RetainSampleSelection => "retain_sample_selection",
            Self::RetainFolds => "retain_folds",
            Self::RetainPopulation => "retain_population",
            Self::RetainHistory => "retain_history",
            Self::RetainDependence => "retain_dependence",
            Self::RetainUncertainty => "retain_uncertainty",
            Self::NamePreservedIdentity => "name_preserved_identity",
        }
    }

    /// Prose printed after `next attempt:`.
    #[must_use]
    pub const fn sentence(self) -> &'static str {
        match self {
            Self::RetainSampleSelection => "retain the original sample selection",
            Self::RetainFolds => "retain the declared folds",
            Self::RetainPopulation => "retain the declared population",
            Self::RetainHistory => "retain the declared history",
            Self::RetainDependence => "retain the declared dependence",
            Self::RetainUncertainty => "retain the declared uncertainty",
            Self::NamePreservedIdentity => "name the preserved identity",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use = "an unknown name is not a next attempt"]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|attempt| attempt.as_str() == name)
    }
}

impl fmt::Display for NextAttempt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.sentence())
    }
}

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
    /// A blank identity. The axis is the first blank one in
    /// [`PreservationAxis`] order, and the next attempt is
    /// [`NextAttempt::NamePreservedIdentity`].
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

/// Contents of one scientific frame.
///
/// [`Self::fingerprint`] hashes these contents. It is not an identity string
/// copied from [`PreservedIdentities`]. Row ids are stored in order, each with
/// the fold id that assigns that row. Whitespace around a non-blank id is part
/// of the content.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameContents {
    row_ids: Vec<Arc<str>>,
    fold_ids: Vec<Arc<str>>,
    population_id: Arc<str>,
    history: Arc<str>,
}

impl FrameContents {
    /// Frame whose selected rows are `row_ids`, each assigned `fold_ids` in
    /// the same order, in `population_id`, with `history`.
    ///
    /// # Errors
    ///
    /// A blank id, a repeated row id, or a fold list whose length is not the
    /// number of selected rows. The axis is the earliest one in
    /// [`PreservationAxis`] order.
    pub fn try_new(
        row_ids: &[&str],
        fold_ids: &[&str],
        population_id: &str,
        history: &str,
    ) -> Result<Self, RewriteRefusal> {
        let mut seen = BTreeSet::new();
        for row in row_ids {
            if is_blank(row) {
                return Err(content_refusal(
                    PreservationAxis::SampleSelection,
                    "a selected row id is blank",
                ));
            }
            if !seen.insert(*row) {
                return Err(content_refusal(
                    PreservationAxis::SampleSelection,
                    "a selected row id is repeated",
                ));
            }
        }
        for fold in fold_ids {
            if is_blank(fold) {
                return Err(content_refusal(PreservationAxis::Folds, "a fold id is blank"));
            }
        }
        if row_ids.len() != fold_ids.len() {
            return Err(content_refusal(
                PreservationAxis::Folds,
                "the fold assignment does not cover the selected rows",
            ));
        }
        if is_blank(population_id) {
            return Err(content_refusal(PreservationAxis::Population, "the population id is blank"));
        }
        if is_blank(history) {
            return Err(content_refusal(PreservationAxis::History, "the history is blank"));
        }
        let mut order: Vec<usize> = (0..row_ids.len()).collect();
        order.sort_by_key(|&index| row_ids[index]);
        let mut stored_rows = Vec::with_capacity(order.len());
        let mut stored_folds = Vec::with_capacity(order.len());
        for index in order {
            stored_rows.push(Arc::from(row_ids[index]));
            stored_folds.push(Arc::from(fold_ids[index]));
        }
        Ok(Self {
            row_ids: stored_rows,
            fold_ids: stored_folds,
            population_id: Arc::from(population_id),
            history: Arc::from(history),
        })
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

    /// Fingerprint of these contents.
    #[must_use]
    pub fn fingerprint(&self) -> FrameFingerprint {
        FrameFingerprint {
            selected_rows: digest_texts(TAG_SELECTED_ROWS, &self.row_ids),
            folds: digest_assignment(&self.row_ids, &self.fold_ids),
            population: digest_text(TAG_POPULATION, &self.population_id),
            history: digest_text(TAG_HISTORY, &self.history),
        }
    }
}

/// Fingerprint of one frame's selected rows, fold assignment, population, and history.
///
/// Each part is a digest of the contents. Two frames with the same identity
/// strings and different contents do not share a fingerprint.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct FrameFingerprint {
    selected_rows: SemanticDigest,
    folds: SemanticDigest,
    population: SemanticDigest,
    history: SemanticDigest,
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

    const fn on_axis(self, axis: PreservationAxis) -> Option<SemanticDigest> {
        match axis {
            PreservationAxis::SampleSelection => Some(self.selected_rows),
            PreservationAxis::Folds => Some(self.folds),
            PreservationAxis::Population => Some(self.population),
            PreservationAxis::History => Some(self.history),
            PreservationAxis::Dependence | PreservationAxis::Uncertainty => None,
        }
    }
}

/// What a column or filter pushdown would do to the frame.
///
/// [`Self::Declared`] names a pushdown that alters no axis. Naming it does not
/// admit the rewrite. [`RewriteKind::check`] admits it only with an executed
/// fingerprint of the rewritten frame: selected rows, folds, population, and
/// history. Equal identity strings are not that fingerprint. Dependence and
/// uncertainty stay caller-supplied identity declarations; the fingerprint does
/// not cover them, and those declarations do not admit the rewrite alone. Each
/// other effect alters one axis even when the identity strings match.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum PushdownEffect {
    /// Required columns stay, and the filter is the query's declared predicate.
    Declared,
    /// The projection omits a column that decides which rows remain.
    OmitsRequiredColumn,
    /// The pushdown omits or rewrites fold membership.
    ReassignsFolds,
    /// The filter is a population cut the query did not declare.
    UndeclaredPopulationCut,
    /// The filter cuts the history window.
    CutsHistory,
    /// The projection drops a cluster or other dependence factor.
    DropsDependenceFactor,
    /// The selected rows stay, but a weight or uncertainty column is dropped.
    DropsUncertainty,
}

impl PushdownEffect {
    /// Axis this effect alters, with the reason, when it alters one.
    #[must_use]
    pub const fn defect(self) -> Option<(PreservationAxis, &'static str)> {
        match self {
            Self::Declared => None,
            Self::OmitsRequiredColumn => Some((
                PreservationAxis::SampleSelection,
                "the projection omits a column the cohort closure requires",
            )),
            Self::ReassignsFolds => {
                Some((PreservationAxis::Folds, "the pushdown reassigns fold membership"))
            }
            Self::UndeclaredPopulationCut => Some((
                PreservationAxis::Population,
                "the filter is a population cut the query did not declare",
            )),
            Self::CutsHistory => {
                Some((PreservationAxis::History, "the filter cuts the history window"))
            }
            Self::DropsDependenceFactor => {
                Some((PreservationAxis::Dependence, "the projection drops a dependence factor"))
            }
            Self::DropsUncertainty => Some((
                PreservationAxis::Uncertainty,
                "the pushdown drops a weight or uncertainty column",
            )),
        }
    }
}

/// What sharing one row and design preparation would do.
///
/// [`Self::Shared`] names a preparation that alters no axis. Naming it does not
/// admit the share. [`RewriteKind::check`] admits it only with an executed
/// fingerprint of the preparation and of each consumer: selected rows, folds,
/// population, and history. Equal identity strings are not that fingerprint.
/// Dependence and uncertainty stay caller-supplied identity declarations; the
/// fingerprint does not cover them, and those declarations do not admit the
/// share alone. Each other effect mixes one axis even when the strings match.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum PreparationEffect {
    /// Both consumers use the preparation's sample, folds, population, history,
    /// dependence, and uncertainty.
    Shared,
    /// The preparation mixes two sample selections.
    MixesSamples,
    /// The preparation mixes two fold assignments.
    MixesFolds,
    /// The preparation mixes two populations.
    MixesPopulations,
    /// The preparation mixes two histories.
    MixesHistories,
    /// The preparation mixes two dependence declarations.
    MixesDependence,
    /// The preparation mixes two uncertainty procedures.
    MixesUncertainty,
}

impl PreparationEffect {
    /// Axis this effect alters, with the reason, when it alters one.
    #[must_use]
    pub const fn defect(self) -> Option<(PreservationAxis, &'static str)> {
        match self {
            Self::Shared => None,
            Self::MixesSamples => Some((
                PreservationAxis::SampleSelection,
                "the shared preparation mixes sample selections",
            )),
            Self::MixesFolds => {
                Some((PreservationAxis::Folds, "the shared preparation mixes fold assignments"))
            }
            Self::MixesPopulations => {
                Some((PreservationAxis::Population, "the shared preparation mixes populations"))
            }
            Self::MixesHistories => {
                Some((PreservationAxis::History, "the shared preparation mixes histories"))
            }
            Self::MixesDependence => {
                Some((PreservationAxis::Dependence, "the shared preparation mixes dependence"))
            }
            Self::MixesUncertainty => {
                Some((PreservationAxis::Uncertainty, "the shared preparation mixes uncertainty"))
            }
        }
    }
}

/// What reusing a nuisance fit would do.
///
/// [`Self::Certified`] names reuse that alters no axis, with the evaluation fold
/// held out of training. Naming it does not admit the reuse. [`RewriteKind::check`]
/// admits it only with an executed fingerprint of the fit and of the consumer:
/// selected rows, folds, population, and history. Equal identity strings are not
/// that fingerprint. Dependence and uncertainty stay caller-supplied identity
/// declarations; the fingerprint does not cover them, and those declarations do
/// not admit the reuse alone. Each other effect is a counterexample.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum NuisanceEffect {
    /// The fit and the consumer name one frame, and training excludes the
    /// evaluation fold.
    Certified,
    /// Reuse adds or drops rows from the estimation sample.
    ChangesEstimationSample,
    /// Predictions trained on a fold are reused on that same fold.
    TrainsOnEvaluationFold,
    /// The nuisance was fit on a different population.
    FitOnOtherPopulation,
    /// The nuisance was fit on a different history.
    FitOnOtherHistory,
    /// Reuse drops a shared dependence factor.
    DropsSharedFactor,
    /// Reuse replaces the consumer's uncertainty procedure.
    ReplacesUncertainty,
}

impl NuisanceEffect {
    /// Axis this effect alters, with the reason, when it alters one.
    #[must_use]
    pub const fn defect(self) -> Option<(PreservationAxis, &'static str)> {
        match self {
            Self::Certified => None,
            Self::ChangesEstimationSample => Some((
                PreservationAxis::SampleSelection,
                "reuse changes which rows the estimation sample contains",
            )),
            Self::TrainsOnEvaluationFold => {
                Some((PreservationAxis::Folds, "the nuisance was trained on the evaluation fold"))
            }
            Self::FitOnOtherPopulation => {
                Some((PreservationAxis::Population, "the nuisance was fit on another population"))
            }
            Self::FitOnOtherHistory => {
                Some((PreservationAxis::History, "the nuisance was fit on another history"))
            }
            Self::DropsSharedFactor => {
                Some((PreservationAxis::Dependence, "reuse drops a shared dependence factor"))
            }
            Self::ReplacesUncertainty => {
                Some((PreservationAxis::Uncertainty, "reuse replaces the uncertainty procedure"))
            }
        }
    }
}

/// What lowering a frame to sufficient statistics would do.
///
/// [`Self::Sufficient`] names a summary that alters no axis. Naming it does not
/// admit the lowering. [`RewriteKind::check`] admits it only with an executed
/// fingerprint of the rows and of the summary: selected rows, folds, population,
/// and history. Equal identity strings are not that fingerprint. Dependence and
/// uncertainty stay caller-supplied identity declarations; the fingerprint does
/// not cover them, and those declarations do not admit the lowering alone. A
/// summary that evicts a row, pools folds, or drops the sampling law is a
/// counterexample, including when the summary still carries the original text.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum SummaryEffect {
    /// The summary is sufficient for the declared likelihood and names every axis.
    ///
    /// The executed fingerprint still has to match. This variant does not admit
    /// the lowering by itself.
    Sufficient,
    /// The summary evicts a row or inserts a column the rows did not have.
    EvictsRow,
    /// The summary pools distinct folds into one fit.
    PoolsFolds,
    /// The summary marginalizes to a different population.
    MarginalizesPopulation,
    /// The summary collapses history.
    CollapsesHistory,
    /// The summary drops a cluster or other dependence factor.
    DropsCluster,
    /// The summary keeps a point and drops the sampling law.
    DropsSamplingLaw,
}

impl SummaryEffect {
    /// Axis this effect alters, with the reason, when it alters one.
    #[must_use]
    pub const fn defect(self) -> Option<(PreservationAxis, &'static str)> {
        match self {
            Self::Sufficient => None,
            Self::EvictsRow => Some((
                PreservationAxis::SampleSelection,
                "the summary evicts a row from the sample",
            )),
            Self::PoolsFolds => Some((PreservationAxis::Folds, "the summary pools distinct folds")),
            Self::MarginalizesPopulation => Some((
                PreservationAxis::Population,
                "the summary marginalizes to another population",
            )),
            Self::CollapsesHistory => {
                Some((PreservationAxis::History, "the summary collapses history"))
            }
            Self::DropsCluster => Some((
                PreservationAxis::Dependence,
                "the summary drops a cluster from the dependence",
            )),
            Self::DropsSamplingLaw => {
                Some((PreservationAxis::Uncertainty, "the summary drops the sampling law"))
            }
        }
    }
}

/// What evaluating one tile would do to the full frame.
///
/// [`Self::Faithful`] names a tile that alters no axis and keeps failed draws.
/// Naming it does not admit the tile. [`RewriteKind::check`] admits it only with
/// an executed fingerprint of the full frame and of the tile: selected rows,
/// folds, population, and history. Equal identity strings are not that
/// fingerprint. Dependence and uncertainty stay caller-supplied identity
/// declarations; the fingerprint does not cover them, and those declarations do
/// not admit the tile alone. A tile that drops rows, splits a fold, or drops
/// failed draws is a counterexample.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum TileEffect {
    /// The tile covers the full frame and keeps joint alignment and failed draws.
    Faithful,
    /// The tile drops rows from the sample.
    DropsRows,
    /// The tile splits a fold or reorders fold membership.
    SplitsFold,
    /// The tile evaluates a different population.
    RetargetsPopulation,
    /// The tile cuts the history horizon.
    CutsHorizon,
    /// The tile drops dependence cross terms.
    DropsCrossTerms,
    /// The tile drops failed draws, so uncertainty mass changes.
    DropsFailedDraws,
}

impl TileEffect {
    /// Axis this effect alters, with the reason, when it alters one.
    #[must_use]
    pub const fn defect(self) -> Option<(PreservationAxis, &'static str)> {
        match self {
            Self::Faithful => None,
            Self::DropsRows => {
                Some((PreservationAxis::SampleSelection, "a tile drops rows from the sample"))
            }
            Self::SplitsFold => Some((PreservationAxis::Folds, "a tile splits a fold")),
            Self::RetargetsPopulation => {
                Some((PreservationAxis::Population, "a tile evaluates another population"))
            }
            Self::CutsHorizon => {
                Some((PreservationAxis::History, "a tile cuts the history horizon"))
            }
            Self::DropsCrossTerms => {
                Some((PreservationAxis::Dependence, "a tile drops dependence cross terms"))
            }
            Self::DropsFailedDraws => Some((
                PreservationAxis::Uncertainty,
                "a tile drops failed draws from the uncertainty",
            )),
        }
    }
}

/// Column projection or declared filter pushdown.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColumnFilterPushdown {
    cohort: PreservedIdentities,
    pushed: PreservedIdentities,
    effect: PushdownEffect,
    frames: Option<Box<BoundFrames>>,
}

impl ColumnFilterPushdown {
    /// Stable name of this rewrite.
    pub const NAME: &'static str = "column_filter_pushdown";

    /// Push `cohort` down to `pushed` with `effect`.
    #[must_use]
    pub const fn new(
        cohort: PreservedIdentities,
        pushed: PreservedIdentities,
        effect: PushdownEffect,
    ) -> Self {
        Self { cohort, pushed, effect, frames: None }
    }

    /// Bind the contents of the cohort and of the pushed frame.
    ///
    /// [`RewriteKind::check`] compares their fingerprints. Matching identity
    /// strings are not those fingerprints.
    #[must_use]
    pub fn with_frames(mut self, cohort: &FrameContents, pushed: &FrameContents) -> Self {
        self.frames = Some(Box::new(BoundFrames {
            original: cohort.fingerprint(),
            rewritten: pushed.fingerprint(),
        }));
        self
    }

    /// Frame before the pushdown.
    #[must_use]
    pub const fn cohort(&self) -> &PreservedIdentities {
        &self.cohort
    }

    /// Frame the pushdown would leave.
    #[must_use]
    pub const fn pushed(&self) -> &PreservedIdentities {
        &self.pushed
    }

    /// Effect the caller attributes to the pushdown.
    #[must_use]
    pub const fn effect(&self) -> PushdownEffect {
        self.effect
    }
}

/// One row and design preparation shared by two consumers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedRowDesignPreparation {
    preparation: PreservedIdentities,
    left: PreservedIdentities,
    right: PreservedIdentities,
    effect: PreparationEffect,
    frames: Option<Box<BoundPreparation>>,
}

impl SharedRowDesignPreparation {
    /// Stable name of this rewrite.
    pub const NAME: &'static str = "shared_row_design_preparation";

    /// Share `preparation` with `left` and `right` under `effect`.
    #[must_use]
    pub const fn new(
        preparation: PreservedIdentities,
        left: PreservedIdentities,
        right: PreservedIdentities,
        effect: PreparationEffect,
    ) -> Self {
        Self { preparation, left, right, effect, frames: None }
    }

    /// Bind the contents of the preparation and of both consumers.
    ///
    /// [`RewriteKind::check`] compares the preparation's fingerprint with each
    /// consumer. Matching identity strings are not those fingerprints.
    #[must_use]
    pub fn with_frames(
        mut self,
        preparation: &FrameContents,
        left: &FrameContents,
        right: &FrameContents,
    ) -> Self {
        self.frames = Some(Box::new(BoundPreparation {
            preparation: preparation.fingerprint(),
            left: left.fingerprint(),
            right: right.fingerprint(),
        }));
        self
    }

    /// Identities of the shared preparation.
    #[must_use]
    pub const fn preparation(&self) -> &PreservedIdentities {
        &self.preparation
    }

    /// First consumer. Compared before [`Self::right`] on a shared axis.
    #[must_use]
    pub const fn left(&self) -> &PreservedIdentities {
        &self.left
    }

    /// Second consumer.
    #[must_use]
    pub const fn right(&self) -> &PreservedIdentities {
        &self.right
    }

    /// Effect the caller attributes to the share.
    #[must_use]
    pub const fn effect(&self) -> PreparationEffect {
        self.effect
    }
}

/// Reuse of a fitted nuisance on a consumer frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NuisanceReuse {
    fit: PreservedIdentities,
    consumer: PreservedIdentities,
    effect: NuisanceEffect,
    frames: Option<Box<BoundFrames>>,
}

impl NuisanceReuse {
    /// Stable name of this rewrite.
    pub const NAME: &'static str = "nuisance_reuse";

    /// Reuse a nuisance fit on `fit` for `consumer` under `effect`.
    #[must_use]
    pub const fn new(
        fit: PreservedIdentities,
        consumer: PreservedIdentities,
        effect: NuisanceEffect,
    ) -> Self {
        Self { fit, consumer, effect, frames: None }
    }

    /// Bind the contents of the fit and of the consumer.
    ///
    /// [`RewriteKind::check`] compares their fingerprints. Matching identity
    /// strings are not those fingerprints.
    #[must_use]
    pub fn with_frames(mut self, fit: &FrameContents, consumer: &FrameContents) -> Self {
        self.frames = Some(Box::new(BoundFrames {
            original: fit.fingerprint(),
            rewritten: consumer.fingerprint(),
        }));
        self
    }

    /// Frame the nuisance was fit on.
    #[must_use]
    pub const fn fit(&self) -> &PreservedIdentities {
        &self.fit
    }

    /// Frame that would reuse the fit.
    #[must_use]
    pub const fn consumer(&self) -> &PreservedIdentities {
        &self.consumer
    }

    /// Effect the caller attributes to the reuse.
    #[must_use]
    pub const fn effect(&self) -> NuisanceEffect {
        self.effect
    }
}

/// Lowering a row frame to a sufficient summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SufficientStatistics {
    rows: PreservedIdentities,
    summary: PreservedIdentities,
    effect: SummaryEffect,
    frames: Option<Box<BoundFrames>>,
}

impl SufficientStatistics {
    /// Stable name of this rewrite.
    pub const NAME: &'static str = "sufficient_statistics";

    /// Lower `rows` to `summary` under `effect`.
    #[must_use]
    pub const fn new(
        rows: PreservedIdentities,
        summary: PreservedIdentities,
        effect: SummaryEffect,
    ) -> Self {
        Self { rows, summary, effect, frames: None }
    }

    /// Bind the contents of the rows and of the summary.
    ///
    /// [`RewriteKind::check`] compares their fingerprints. Matching identity
    /// strings are not those fingerprints.
    #[must_use]
    pub fn with_frames(mut self, rows: &FrameContents, summary: &FrameContents) -> Self {
        self.frames = Some(Box::new(BoundFrames {
            original: rows.fingerprint(),
            rewritten: summary.fingerprint(),
        }));
        self
    }

    /// Frame of the rows.
    #[must_use]
    pub const fn rows(&self) -> &PreservedIdentities {
        &self.rows
    }

    /// Frame the summary claims.
    #[must_use]
    pub const fn summary(&self) -> &PreservedIdentities {
        &self.summary
    }

    /// Effect the caller attributes to the lowering.
    #[must_use]
    pub const fn effect(&self) -> SummaryEffect {
        self.effect
    }
}

/// One tile of a row, draw, grid, or horizon evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TiledEvaluation {
    full: PreservedIdentities,
    tile: PreservedIdentities,
    effect: TileEffect,
    frames: Option<Box<BoundFrames>>,
}

impl TiledEvaluation {
    /// Stable name of this rewrite.
    pub const NAME: &'static str = "tiled_evaluation";

    /// Evaluate `tile` as a piece of `full` under `effect`.
    #[must_use]
    pub const fn new(
        full: PreservedIdentities,
        tile: PreservedIdentities,
        effect: TileEffect,
    ) -> Self {
        Self { full, tile, effect, frames: None }
    }

    /// Bind the contents of the full frame and of the tile.
    ///
    /// [`RewriteKind::check`] compares their fingerprints. Matching identity
    /// strings are not those fingerprints.
    #[must_use]
    pub fn with_frames(mut self, full: &FrameContents, tile: &FrameContents) -> Self {
        self.frames = Some(Box::new(BoundFrames {
            original: full.fingerprint(),
            rewritten: tile.fingerprint(),
        }));
        self
    }

    /// Frame of the full evaluation.
    #[must_use]
    pub const fn full(&self) -> &PreservedIdentities {
        &self.full
    }

    /// Frame the tile claims. A faithful tile claims the full frame.
    #[must_use]
    pub const fn tile(&self) -> &PreservedIdentities {
        &self.tile
    }

    /// Effect the caller attributes to the tile.
    #[must_use]
    pub const fn effect(&self) -> TileEffect {
        self.effect
    }
}

mod seal {
    /// Marker for the five rewrite types this crate defines.
    pub trait Sealed {}
}

impl seal::Sealed for ColumnFilterPushdown {}
impl seal::Sealed for SharedRowDesignPreparation {}
impl seal::Sealed for NuisanceReuse {}
impl seal::Sealed for SufficientStatistics {}
impl seal::Sealed for TiledEvaluation {}

/// A rewrite whose legality [`Self::check`] can decide.
///
/// Only the five rewrite types in this module implement the trait.
pub trait RewriteKind: seal::Sealed {
    /// Stable `snake_case` name of this rewrite.
    #[must_use]
    fn name(&self) -> &'static str;

    /// Admit this rewrite when every preservation axis is unchanged.
    ///
    /// Axes are examined in [`PreservationAxis`] order. Within one axis, pairs
    /// are examined in the order the rewrite type documents. An effect that
    /// names the axis is a refusal even when every identity string matches. A
    /// fingerprint of the frame contents that differs is a refusal even when
    /// the identity strings match. Equal identity strings without a fingerprint
    /// are a refusal.
    ///
    /// # Errors
    ///
    /// The first altered axis. [`RewriteRefusal::next_attempt`] keeps that axis.
    #[must_use = "a refusal is the counterexample"]
    fn check(&self) -> Result<(), RewriteRefusal>;
}

impl RewriteKind for ColumnFilterPushdown {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn check(&self) -> Result<(), RewriteRefusal> {
        let identities = &[(&self.cohort, &self.pushed)];
        let defect = self.effect.defect();
        match &self.frames {
            Some(frames) => {
                admit(Self::NAME, identities, defect, &[(&frames.original, &frames.rewritten)])
            }
            None => admit(Self::NAME, identities, defect, &[]),
        }
    }
}

impl RewriteKind for SharedRowDesignPreparation {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn check(&self) -> Result<(), RewriteRefusal> {
        let identities = &[(&self.preparation, &self.left), (&self.preparation, &self.right)];
        let defect = self.effect.defect();
        match &self.frames {
            Some(frames) => admit(
                Self::NAME,
                identities,
                defect,
                &[(&frames.preparation, &frames.left), (&frames.preparation, &frames.right)],
            ),
            None => admit(Self::NAME, identities, defect, &[]),
        }
    }
}

impl RewriteKind for NuisanceReuse {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn check(&self) -> Result<(), RewriteRefusal> {
        let identities = &[(&self.fit, &self.consumer)];
        let defect = self.effect.defect();
        match &self.frames {
            Some(frames) => {
                admit(Self::NAME, identities, defect, &[(&frames.original, &frames.rewritten)])
            }
            None => admit(Self::NAME, identities, defect, &[]),
        }
    }
}

impl RewriteKind for SufficientStatistics {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn check(&self) -> Result<(), RewriteRefusal> {
        let identities = &[(&self.rows, &self.summary)];
        let defect = self.effect.defect();
        match &self.frames {
            Some(frames) => {
                admit(Self::NAME, identities, defect, &[(&frames.original, &frames.rewritten)])
            }
            None => admit(Self::NAME, identities, defect, &[]),
        }
    }
}

impl RewriteKind for TiledEvaluation {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn check(&self) -> Result<(), RewriteRefusal> {
        let identities = &[(&self.full, &self.tile)];
        let defect = self.effect.defect();
        match &self.frames {
            Some(frames) => {
                admit(Self::NAME, identities, defect, &[(&frames.original, &frames.rewritten)])
            }
            None => admit(Self::NAME, identities, defect, &[]),
        }
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
        next: NextAttempt::NamePreservedIdentity,
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
            let Some(original_digest) = original.on_axis(axis) else {
                continue;
            };
            let Some(rewritten_digest) = rewritten.on_axis(axis) else {
                continue;
            };
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
    // Dependence and uncertainty have no content digest; they stay declarations
    // and still require this fingerprint before the declaration can pass.
    if frames.is_empty() {
        return Err(altered(kind, PreservationAxis::SampleSelection, NO_FINGERPRINT));
    }
    Ok(())
}

/// Fingerprints of the two frames one rewrite compares.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BoundFrames {
    original: FrameFingerprint,
    rewritten: FrameFingerprint,
}

/// Fingerprints of a shared preparation and its two consumers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BoundPreparation {
    preparation: FrameFingerprint,
    left: FrameFingerprint,
    right: FrameFingerprint,
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

fn digest_text(tag: &str, text: &str) -> SemanticDigest {
    let mut writer = CanonicalWriter::new(tag);
    writer.text(text);
    SemanticDigest::from_bytes(writer.finish())
}

fn digest_texts(tag: &str, texts: &[Arc<str>]) -> SemanticDigest {
    let mut writer = CanonicalWriter::new(tag);
    writer.texts(texts);
    SemanticDigest::from_bytes(writer.finish())
}

fn digest_assignment(rows: &[Arc<str>], folds: &[Arc<str>]) -> SemanticDigest {
    let mut writer = CanonicalWriter::new(TAG_FOLDS);
    writer.word(rows.len() as u64);
    for (row, fold) in rows.iter().zip(folds) {
        writer.text(row).text(fold);
    }
    SemanticDigest::from_bytes(writer.finish())
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::{
        ColumnFilterPushdown, FrameContents, NextAttempt, NuisanceEffect, NuisanceReuse,
        PreparationEffect, PreservationAxis, PreservedIdentities, PushdownEffect, RewriteKind,
        SharedRowDesignPreparation, SufficientStatistics, SummaryEffect, TileEffect,
        TiledEvaluation,
    };

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

    #[track_caller]
    fn expect_refusal(rewrite: &impl RewriteKind, axis: PreservationAxis, because: &str) {
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

    #[test]
    fn preserved_frame_is_legal_for_every_rewrite() {
        let frame = identities();
        let rows = contents();
        let pushdown =
            ColumnFilterPushdown::new(frame.clone(), frame.clone(), PushdownEffect::Declared)
                .with_frames(&rows, &rows);
        let preparation = SharedRowDesignPreparation::new(
            frame.clone(),
            frame.clone(),
            frame.clone(),
            PreparationEffect::Shared,
        )
        .with_frames(&rows, &rows, &rows);
        let nuisance = NuisanceReuse::new(frame.clone(), frame.clone(), NuisanceEffect::Certified)
            .with_frames(&rows, &rows);
        let summary =
            SufficientStatistics::new(frame.clone(), frame.clone(), SummaryEffect::Sufficient)
                .with_frames(&rows, &rows);
        let tiled = TiledEvaluation::new(frame.clone(), frame, TileEffect::Faithful)
            .with_frames(&rows, &rows);
        assert!(pushdown.effect().defect().is_none());
        assert!(preparation.effect().defect().is_none());
        assert!(nuisance.effect().defect().is_none());
        assert!(summary.effect().defect().is_none());
        assert!(tiled.effect().defect().is_none());
        assert!(matches!(pushdown.check(), Ok(())));
        assert!(matches!(preparation.check(), Ok(())));
        assert!(matches!(nuisance.check(), Ok(())));
        assert!(matches!(summary.check(), Ok(())));
        assert!(matches!(tiled.check(), Ok(())));
        assert_eq!(pushdown.name(), ColumnFilterPushdown::NAME);
        assert_eq!(preparation.name(), SharedRowDesignPreparation::NAME);
        assert_eq!(nuisance.name(), NuisanceReuse::NAME);
        assert_eq!(summary.name(), SufficientStatistics::NAME);
        assert_eq!(tiled.name(), TiledEvaluation::NAME);
        assert_eq!(pushdown.cohort().identity(PreservationAxis::Folds), "folds");
        assert_eq!(
            preparation.left().identity(PreservationAxis::Population),
            preparation.right().identity(PreservationAxis::Population),
        );
        assert_eq!(
            nuisance.fit().identity(PreservationAxis::History),
            nuisance.consumer().identity(PreservationAxis::History),
        );
        assert_eq!(
            summary.rows().identity(PreservationAxis::Dependence),
            summary.summary().identity(PreservationAxis::Dependence),
        );
        assert_eq!(
            tiled.full().identity(PreservationAxis::Uncertainty),
            tiled.tile().identity(PreservationAxis::Uncertainty),
        );
    }

    #[test]
    fn matching_identities_do_not_admit_an_altering_effect() {
        let frame = identities();
        let pushdown = [
            (PushdownEffect::OmitsRequiredColumn, PreservationAxis::SampleSelection),
            (PushdownEffect::ReassignsFolds, PreservationAxis::Folds),
            (PushdownEffect::UndeclaredPopulationCut, PreservationAxis::Population),
            (PushdownEffect::CutsHistory, PreservationAxis::History),
            (PushdownEffect::DropsDependenceFactor, PreservationAxis::Dependence),
            (PushdownEffect::DropsUncertainty, PreservationAxis::Uncertainty),
        ];
        for (effect, axis) in pushdown {
            let because = effect.defect().expect("altering pushdown").1;
            expect_refusal(
                &ColumnFilterPushdown::new(frame.clone(), frame.clone(), effect),
                axis,
                because,
            );
        }
        let preparation = [
            (PreparationEffect::MixesSamples, PreservationAxis::SampleSelection),
            (PreparationEffect::MixesFolds, PreservationAxis::Folds),
            (PreparationEffect::MixesPopulations, PreservationAxis::Population),
            (PreparationEffect::MixesHistories, PreservationAxis::History),
            (PreparationEffect::MixesDependence, PreservationAxis::Dependence),
            (PreparationEffect::MixesUncertainty, PreservationAxis::Uncertainty),
        ];
        for (effect, axis) in preparation {
            let because = effect.defect().expect("altering preparation").1;
            expect_refusal(
                &SharedRowDesignPreparation::new(
                    frame.clone(),
                    frame.clone(),
                    frame.clone(),
                    effect,
                ),
                axis,
                because,
            );
        }
        let nuisance = [
            (NuisanceEffect::ChangesEstimationSample, PreservationAxis::SampleSelection),
            (NuisanceEffect::TrainsOnEvaluationFold, PreservationAxis::Folds),
            (NuisanceEffect::FitOnOtherPopulation, PreservationAxis::Population),
            (NuisanceEffect::FitOnOtherHistory, PreservationAxis::History),
            (NuisanceEffect::DropsSharedFactor, PreservationAxis::Dependence),
            (NuisanceEffect::ReplacesUncertainty, PreservationAxis::Uncertainty),
        ];
        for (effect, axis) in nuisance {
            let because = effect.defect().expect("altering nuisance").1;
            expect_refusal(
                &NuisanceReuse::new(frame.clone(), frame.clone(), effect),
                axis,
                because,
            );
        }
        let summary = [
            (SummaryEffect::EvictsRow, PreservationAxis::SampleSelection),
            (SummaryEffect::PoolsFolds, PreservationAxis::Folds),
            (SummaryEffect::MarginalizesPopulation, PreservationAxis::Population),
            (SummaryEffect::CollapsesHistory, PreservationAxis::History),
            (SummaryEffect::DropsCluster, PreservationAxis::Dependence),
            (SummaryEffect::DropsSamplingLaw, PreservationAxis::Uncertainty),
        ];
        for (effect, axis) in summary {
            let because = effect.defect().expect("altering summary").1;
            expect_refusal(
                &SufficientStatistics::new(frame.clone(), frame.clone(), effect),
                axis,
                because,
            );
        }
        let tiled = [
            (TileEffect::DropsRows, PreservationAxis::SampleSelection),
            (TileEffect::SplitsFold, PreservationAxis::Folds),
            (TileEffect::RetargetsPopulation, PreservationAxis::Population),
            (TileEffect::CutsHorizon, PreservationAxis::History),
            (TileEffect::DropsCrossTerms, PreservationAxis::Dependence),
            (TileEffect::DropsFailedDraws, PreservationAxis::Uncertainty),
        ];
        for (effect, axis) in tiled {
            let because = effect.defect().expect("altering tile").1;
            expect_refusal(
                &TiledEvaluation::new(frame.clone(), frame.clone(), effect),
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
                &ColumnFilterPushdown::new(
                    original.clone(),
                    moved.clone(),
                    PushdownEffect::Declared,
                ),
                axis,
                &token,
            );
            expect_refusal(
                &SharedRowDesignPreparation::new(
                    original.clone(),
                    original.clone(),
                    moved.clone(),
                    PreparationEffect::Shared,
                ),
                axis,
                &token,
            );
            expect_refusal(
                &NuisanceReuse::new(original.clone(), moved.clone(), NuisanceEffect::Certified),
                axis,
                &token,
            );
            expect_refusal(
                &SufficientStatistics::new(
                    original.clone(),
                    moved.clone(),
                    SummaryEffect::Sufficient,
                ),
                axis,
                &token,
            );
            expect_refusal(
                &TiledEvaluation::new(original.clone(), moved, TileEffect::Faithful),
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
        let pushdown = ColumnFilterPushdown::new(
            original.clone(),
            pushed,
            PushdownEffect::OmitsRequiredColumn,
        );
        let err = pushdown.check().expect_err("sample selection is earlier");
        assert_eq!(err.axis(), PreservationAxis::SampleSelection);
        assert_eq!(err.next_attempt(), NextAttempt::RetainSampleSelection);
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
        let declared = ColumnFilterPushdown::new(original, both, PushdownEffect::Declared);
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
        let shared =
            SharedRowDesignPreparation::new(preparation, left, right, PreparationEffect::Shared);
        let err = shared.check().expect_err("left consumer");
        assert_eq!(err.axis(), PreservationAxis::Dependence);
        assert_eq!(err.next_attempt(), NextAttempt::RetainDependence);
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
            assert_eq!(err.next_attempt(), NextAttempt::NamePreservedIdentity);
            assert_eq!(err.next_attempt().as_str(), "name_preserved_identity");
            assert!(err.message().contains(axis.phrase()));
            assert!(err.to_string().contains(err.next_attempt().sentence()));
            let _: &dyn Error = &err;
        }
    }

    #[test]
    fn axis_and_attempt_names_round_trip() {
        assert_eq!(PreservationAxis::ALL.len(), 6);
        assert_eq!(NextAttempt::ALL.len(), 7);
        assert_eq!(PreservationAxis::from_name("row_index"), None);
        assert_eq!(NextAttempt::from_name("drop_rows"), None);
        for axis in PreservationAxis::ALL {
            assert_eq!(PreservationAxis::from_name(axis.as_str()), Some(axis));
            assert_eq!(axis.to_string(), axis.as_str());
            let attempt = axis.next_attempt();
            assert_ne!(attempt, NextAttempt::NamePreservedIdentity);
            assert_eq!(NextAttempt::from_name(attempt.as_str()), Some(attempt));
            assert_eq!(attempt.to_string(), attempt.sentence());
        }
        assert_eq!(
            NextAttempt::from_name("name_preserved_identity"),
            Some(NextAttempt::NamePreservedIdentity)
        );
    }

    #[test]
    fn equal_identity_strings_with_a_different_selected_row_fingerprint_are_refused() {
        let frame = identities();
        let original = contents();
        let bare =
            ColumnFilterPushdown::new(frame.clone(), frame.clone(), PushdownEffect::Declared);
        expect_refusal(&bare, PreservationAxis::SampleSelection, "fingerprint");
        assert_eq!(
            bare.cohort().identity(PreservationAxis::SampleSelection),
            bare.pushed().identity(PreservationAxis::SampleSelection),
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
            let digest = rewritten.fingerprint().on_axis(axis).expect("content axis");
            assert_ne!(original.fingerprint().on_axis(axis), Some(digest));
            let declared = ColumnFilterPushdown::new(
                frame.clone(),
                frame.clone(),
                PushdownEffect::Declared,
            )
            .with_frames(&original, &rewritten);
            assert_eq!(
                declared.cohort().identity(axis),
                declared.pushed().identity(axis),
            );
            expect_refusal(&declared, axis, &digest.to_string());
        }
    }

    #[test]
    fn matching_frame_fingerprint_is_admitted() {
        let frame = identities();
        let original = FrameContents::try_new(
            &["row-b", "row-a"],
            &["fold-1", "fold-2"],
            "pop-1",
            "hist-1",
        )
        .expect("crossed assignment");
        let same = FrameContents::try_new(
            &["row-a", "row-b"],
            &["fold-2", "fold-1"],
            "pop-1",
            "hist-1",
        )
        .expect("listed assignment");
        assert_eq!(original.fingerprint(), same.fingerprint());
        assert_eq!(original.row_ids().collect::<Vec<&str>>(), ["row-a", "row-b"]);
        assert_eq!(original.fold_ids().collect::<Vec<&str>>(), ["fold-2", "fold-1"]);
        let pushdown =
            ColumnFilterPushdown::new(frame.clone(), frame, PushdownEffect::Declared)
                .with_frames(&original, &same);
        assert!(matches!(pushdown.check(), Ok(())));
    }

    #[test]
    fn equal_declaration_strings_without_an_executed_fingerprint_are_refused() {
        let frame = identities();
        let pushdown =
            ColumnFilterPushdown::new(frame.clone(), frame.clone(), PushdownEffect::Declared);
        let preparation = SharedRowDesignPreparation::new(
            frame.clone(),
            frame.clone(),
            frame.clone(),
            PreparationEffect::Shared,
        );
        let nuisance = NuisanceReuse::new(frame.clone(), frame.clone(), NuisanceEffect::Certified);
        let summary =
            SufficientStatistics::new(frame.clone(), frame.clone(), SummaryEffect::Sufficient);
        let tiled = TiledEvaluation::new(frame.clone(), frame.clone(), TileEffect::Faithful);
        for axis in PreservationAxis::ALL {
            assert_eq!(pushdown.cohort().identity(axis), pushdown.pushed().identity(axis));
            assert_eq!(
                preparation.preparation().identity(axis),
                preparation.left().identity(axis),
            );
            assert_eq!(
                preparation.preparation().identity(axis),
                preparation.right().identity(axis),
            );
            assert_eq!(nuisance.fit().identity(axis), nuisance.consumer().identity(axis));
            assert_eq!(summary.rows().identity(axis), summary.summary().identity(axis));
            assert_eq!(tiled.full().identity(axis), tiled.tile().identity(axis));
        }
        assert!(pushdown.effect().defect().is_none());
        assert!(preparation.effect().defect().is_none());
        assert!(nuisance.effect().defect().is_none());
        assert!(summary.effect().defect().is_none());
        assert!(tiled.effect().defect().is_none());
        expect_refusal(&pushdown, PreservationAxis::SampleSelection, "fingerprint");
        expect_refusal(&preparation, PreservationAxis::SampleSelection, "fingerprint");
        expect_refusal(&nuisance, PreservationAxis::SampleSelection, "fingerprint");
        expect_refusal(&summary, PreservationAxis::SampleSelection, "fingerprint");
        expect_refusal(&tiled, PreservationAxis::SampleSelection, "fingerprint");
    }
}
