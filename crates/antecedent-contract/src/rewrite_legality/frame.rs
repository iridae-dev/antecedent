use super::{
    Arc, BTreeSet, CanonicalWriter, PreservationAxis, RewriteRefusal, SemanticDigest,
    TAG_DEPENDENCE, TAG_FOLDS, TAG_HISTORY, TAG_POPULATION, TAG_SELECTED_ROWS, TAG_UNCERTAINTY,
    UNDECLARED_DEPENDENCE, UNDECLARED_SAMPLING_LAW, UNDECLARED_UNCERTAINTY, check_text,
    content_refusal, digest_text,
};

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
/// copied from [`PreservedIdentities`](super::PreservedIdentities). Row ids are stored in order, each with
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

    pub(super) const fn on_axis(self, axis: PreservationAxis) -> SemanticDigest {
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
