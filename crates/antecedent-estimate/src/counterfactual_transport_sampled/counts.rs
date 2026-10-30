//! Sampled finite-discrete regime evidence: count tables, the declared sampling
//! design of each, and the rule that turns counts into compatible finite laws.
//!
//! A count table is one population's joint counts of one regime (an
//! observational or hard-intervention world) over a declared variable set, as a
//! dense row-major table (last axis fastest). Nothing here fills a cell
//! silently: every unseen cell is either refused, given a declared pseudo-count
//! (and the law is labelled smoothed), or kept as an explicitly unseen cell for a
//! structural envelope. A cell the caller declares structurally zero must have
//! count zero and is never smoothed or filled.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, RegimeId, Value, VariableId};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, InterventionAssignment, LawTolerance};

use super::{
    DETAIL_CANCELLED, DETAIL_INVALID_COUNTS, DETAIL_SAMPLE_SIZE, DETAIL_SPARSE, DETAIL_UNSEEN,
    SAMPLED_TRANSPORT_MAX_TABLE_CELLS, SampledWork, invalid_counts, refused,
};
use crate::error::EstimationError;

/// Domain tag of the count-table content identity.
pub const SAMPLED_COUNT_IDENTITY_TAG: &str =
    "antecedent.estimate.counterfactual_transport_sampled.counts.v1";

/// Reference axis count of a count table; not enforced (the cell count is bounded).
pub const SAMPLED_TRANSPORT_MAX_AXES: usize = 16;

/// One measured variable of a count table and its declared levels.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledAxis {
    /// The variable.
    pub variable: VariableId,
    /// Declared levels in table order (finite, distinct).
    pub levels: Vec<f64>,
}

/// How the units of one count table were drawn.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SamplingDesign {
    /// Units drawn independently from the population's regime law.
    IndependentUnits,
    /// The totals of the strata given by the joint levels of `variables` were
    /// fixed by design; units are drawn independently within a stratum. The
    /// stratum proportions are then not sampled.
    Stratified {
        /// Table variables that define the strata.
        variables: Vec<VariableId>,
    },
}

/// Identity of one count table within one request: population, regime and
/// concrete intervention levels (as IEEE bit patterns).
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SampledTableKey {
    /// Population of the law.
    pub population: Arc<str>,
    /// Catalog regime.
    pub regime: RegimeId,
    /// `(intervened variable, level bits)` ascending by variable.
    pub interventions: Vec<(VariableId, u64)>,
}

impl SampledTableKey {
    pub(crate) fn checked_clone(
        &self,
        work: &mut SampledWork<'_>,
    ) -> Result<Self, EstimationError> {
        work.charge(self.interventions.len())?;
        let mut interventions = Vec::with_capacity(self.interventions.len());
        for chunk in self.interventions.chunks(512) {
            work.charge(0)?;
            interventions.extend_from_slice(chunk);
        }
        work.charge(0)?;
        Ok(Self { population: Arc::clone(&self.population), regime: self.regime, interventions })
    }
}

/// Unvalidated parts of a [`SampledRegimeCounts`].
#[derive(Clone, Debug)]
pub struct SampledRegimeCountsParts {
    /// Population of the law.
    pub population: Arc<str>,
    /// Catalog regime.
    pub regime: RegimeId,
    /// `(intervened variable, level)` of the regime's concrete world.
    pub interventions: Vec<(VariableId, f64)>,
    /// Measured axes in table order.
    pub axes: Vec<SampledAxis>,
    /// Dense counts, last axis fastest.
    pub counts: Vec<u64>,
    /// Physical snapshot identity of the sample (matches a catalog binding when
    /// the catalog carries one).
    pub snapshot_identity: Arc<str>,
    /// How the units were drawn.
    pub design: SamplingDesign,
    /// Cell indices declared structurally zero (never smoothed or filled).
    pub structural_zeros: Vec<usize>,
}

/// A validated sampled count table of one population and regime.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledRegimeCounts {
    population: Arc<str>,
    regime: RegimeId,
    interventions: Vec<(VariableId, f64)>,
    axes: Vec<SampledAxis>,
    counts: Vec<u64>,
    snapshot_identity: Arc<str>,
    design: SamplingDesign,
    structural_zeros: Vec<usize>,
}

fn level_bits(level: f64) -> u64 {
    (level + 0.0).to_bits()
}

impl SampledRegimeCounts {
    /// Validate a count table.
    ///
    /// # Errors
    ///
    /// `invalid_argument` (`counterfactual_transport_sampled.invalid_counts`)
    /// for an empty name, no axes, a non-finite or repeated level, a repeated
    /// variable, a shape that does not match the counts, no observations, a
    /// count total that overflows, a structural zero that is out of range or
    /// has observations, or a stratification variable the table does not
    /// measure.
    pub fn try_new(parts: SampledRegimeCountsParts) -> Result<Self, EstimationError> {
        let SampledRegimeCountsParts {
            population,
            regime,
            mut interventions,
            axes,
            counts,
            snapshot_identity,
            design,
            mut structural_zeros,
        } = parts;
        if population.is_empty() || snapshot_identity.is_empty() {
            return Err(invalid_counts("a count table names its population and its snapshot"));
        }
        if axes.is_empty() {
            return Err(invalid_counts("a count table measures at least one variable"));
        }
        let mut seen: Vec<VariableId> = Vec::new();
        let mut cells: usize = 1;
        for axis in &axes {
            if seen.contains(&axis.variable) {
                return Err(invalid_counts("a variable is measured on two axes"));
            }
            seen.push(axis.variable);
            if axis.levels.is_empty() || axis.levels.iter().any(|l| !l.is_finite()) {
                return Err(invalid_counts("an axis has no levels or a non-finite level"));
            }
            let mut bits: Vec<u64> = axis.levels.iter().map(|l| level_bits(*l)).collect();
            bits.sort_unstable();
            if bits.windows(2).any(|w| w[0] == w[1]) {
                return Err(invalid_counts("an axis repeats a level"));
            }
            cells = cells
                .checked_mul(axis.levels.len())
                .filter(|c| *c <= SAMPLED_TRANSPORT_MAX_TABLE_CELLS)
                .ok_or_else(|| {
                    invalid_counts(format!(
                        "a count table has at most {SAMPLED_TRANSPORT_MAX_TABLE_CELLS} cells"
                    ))
                })?;
        }
        interventions.sort_by_key(|(v, _)| *v);
        if interventions.windows(2).any(|w| w[0].0 == w[1].0)
            || interventions.iter().any(|(v, l)| !l.is_finite() || seen.contains(v))
        {
            return Err(invalid_counts(
                "an intervention repeats a variable, is non-finite, or intervenes on a measured axis",
            ));
        }
        if counts.len() != cells {
            return Err(invalid_counts(format!(
                "the table has {cells} cells but {} counts",
                counts.len()
            )));
        }
        let total = counts
            .iter()
            .try_fold(0u64, |n, k| n.checked_add(*k))
            .ok_or_else(|| invalid_counts("the count total overflows"))?;
        if total == 0 {
            return Err(invalid_counts("a count table with no observations carries no evidence"));
        }
        structural_zeros.sort_unstable();
        structural_zeros.dedup();
        for &cell in &structural_zeros {
            if cell >= cells || counts[cell] != 0 {
                return Err(invalid_counts(
                    "a declared structural zero is out of range or has observations",
                ));
            }
        }
        if let SamplingDesign::Stratified { variables } = &design {
            let distinct = variables.iter().enumerate().all(|(i, v)| !variables[..i].contains(v));
            if variables.is_empty() || !distinct || variables.iter().any(|v| !seen.contains(v)) {
                return Err(invalid_counts(
                    "stratification variables are distinct measured variables of the table",
                ));
            }
        }
        Ok(Self {
            population,
            regime,
            interventions,
            axes,
            counts,
            snapshot_identity,
            design,
            structural_zeros,
        })
    }

    /// Population of the law.
    #[must_use]
    pub fn population(&self) -> &str {
        &self.population
    }

    /// Catalog regime.
    #[must_use]
    pub const fn regime(&self) -> RegimeId {
        self.regime
    }

    /// `(intervened variable, level)` of the concrete intervention world.
    #[must_use]
    pub fn interventions(&self) -> &[(VariableId, f64)] {
        &self.interventions
    }

    /// Measured axes in table order.
    #[must_use]
    pub fn axes(&self) -> &[SampledAxis] {
        &self.axes
    }

    /// Dense counts, last axis fastest.
    #[must_use]
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Snapshot identity of the sample.
    #[must_use]
    pub fn snapshot_identity(&self) -> &str {
        &self.snapshot_identity
    }

    /// Declared sampling design.
    #[must_use]
    pub const fn design(&self) -> &SamplingDesign {
        &self.design
    }

    /// Declared structural zero cells, ascending.
    #[must_use]
    pub fn structural_zeros(&self) -> &[usize] {
        &self.structural_zeros
    }

    /// Clone a previously validated table, polling during caller-sized vector copies.
    pub(crate) fn checked_clone(
        &self,
        work: &mut SampledWork<'_>,
    ) -> Result<Self, EstimationError> {
        fn copied<T: Clone>(
            values: &[T],
            work: &mut SampledWork<'_>,
        ) -> Result<Vec<T>, EstimationError> {
            let mut out = Vec::with_capacity(values.len());
            for chunk in values.chunks(512) {
                work.charge(0)?;
                out.extend_from_slice(chunk);
            }
            Ok(out)
        }
        let mut axes = Vec::with_capacity(self.axes.len());
        for axis in &self.axes {
            work.charge(0)?;
            axes.push(SampledAxis { variable: axis.variable, levels: copied(&axis.levels, work)? });
        }
        let design = match &self.design {
            SamplingDesign::IndependentUnits => SamplingDesign::IndependentUnits,
            SamplingDesign::Stratified { variables } => {
                SamplingDesign::Stratified { variables: copied(variables, work)? }
            }
        };
        Ok(Self {
            population: Arc::clone(&self.population),
            regime: self.regime,
            interventions: copied(&self.interventions, work)?,
            axes,
            counts: copied(&self.counts, work)?,
            snapshot_identity: Arc::clone(&self.snapshot_identity),
            design,
            structural_zeros: copied(&self.structural_zeros, work)?,
        })
    }

    pub(crate) fn checked_unseen_cells(
        &self,
        work: &mut SampledWork<'_>,
    ) -> Result<Vec<usize>, EstimationError> {
        let mut unseen = Vec::new();
        for (block, counts) in self.counts.chunks(512).enumerate() {
            work.charge(0)?;
            for (offset, &count) in counts.iter().enumerate() {
                let cell = block * 512 + offset;
                if count == 0 && self.structural_zeros.binary_search(&cell).is_err() {
                    unseen.push(cell);
                }
            }
        }
        work.charge(0)?;
        Ok(unseen)
    }

    /// Total observations.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Number of cells.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.counts.len()
    }

    /// Cells with a zero count that are not declared structural zeros.
    #[must_use]
    pub fn unseen_cells(&self) -> Vec<usize> {
        (0..self.counts.len())
            .filter(|c| self.counts[*c] == 0 && self.structural_zeros.binary_search(c).is_err())
            .collect()
    }

    /// Cells with a nonzero count.
    #[must_use]
    pub fn observed_cells(&self) -> usize {
        self.counts.iter().filter(|c| **c > 0).count()
    }

    pub(crate) fn checked_report_counts(
        &self,
        work: &mut SampledWork<'_>,
    ) -> Result<(u64, usize, usize), EstimationError> {
        let mut total = 0u64;
        let mut observed = 0usize;
        let mut unseen = 0usize;
        let per_cell = 3usize
            .saturating_add((usize::BITS - self.structural_zeros.len().leading_zeros()) as usize);
        for (block, counts) in self.counts.chunks(512).enumerate() {
            work.charge(counts.len().saturating_mul(per_cell))?;
            for (offset, &count) in counts.iter().enumerate() {
                total += count;
                observed += usize::from(count > 0);
                let cell = block * 512 + offset;
                unseen +=
                    usize::from(count == 0 && self.structural_zeros.binary_search(&cell).is_err());
            }
        }
        work.charge(1)?;
        Ok((total, observed, unseen))
    }

    /// The table's identity within a request.
    #[must_use]
    pub fn key(&self) -> SampledTableKey {
        SampledTableKey {
            population: Arc::clone(&self.population),
            regime: self.regime,
            interventions: self.interventions.iter().map(|(v, l)| (*v, level_bits(*l))).collect(),
        }
    }

    pub(crate) fn checked_key(
        &self,
        work: &mut SampledWork<'_>,
    ) -> Result<SampledTableKey, EstimationError> {
        let mut interventions = Vec::with_capacity(self.interventions.len());
        for chunk in self.interventions.chunks(512) {
            work.charge(chunk.len())?;
            interventions.extend(chunk.iter().map(|(v, l)| (*v, level_bits(*l))));
        }
        work.charge(1)?;
        Ok(SampledTableKey {
            population: Arc::clone(&self.population),
            regime: self.regime,
            interventions,
        })
    }

    /// Whether two tables have identical summaries: equal axes, counts, design
    /// and structural zeros (their labels and snapshot identities may differ).
    /// This alone does not establish that they contain the same physical units.
    #[must_use]
    pub fn same_observations(&self, other: &Self) -> bool {
        self.axes == other.axes
            && self.counts == other.counts
            && self.design == other.design
            && self.structural_zeros == other.structural_zeros
    }

    pub(crate) fn same_observations_checked(
        &self,
        other: &Self,
        work: &mut SampledWork<'_>,
    ) -> Result<bool, EstimationError> {
        fn same<T: PartialEq>(
            a: &[T],
            b: &[T],
            work: &mut SampledWork<'_>,
        ) -> Result<bool, EstimationError> {
            if a.len() != b.len() {
                return Ok(false);
            }
            for (left, right) in a.chunks(512).zip(b.chunks(512)) {
                work.charge(0)?;
                if left != right {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        if self.axes.len() != other.axes.len() {
            return Ok(false);
        }
        match (&self.design, &other.design) {
            (SamplingDesign::IndependentUnits, SamplingDesign::IndependentUnits) => {}
            (
                SamplingDesign::Stratified { variables: left },
                SamplingDesign::Stratified { variables: right },
            ) if same(left, right, work)? => {}
            _ => return Ok(false),
        }
        for (left, right) in self.axes.iter().zip(&other.axes) {
            work.charge(0)?;
            if left.variable != right.variable || !same(&left.levels, &right.levels, work)? {
                return Ok(false);
            }
        }
        Ok(same(&self.counts, &other.counts, work)?
            && same(&self.structural_zeros, &other.structural_zeros, work)?)
    }

    /// Content identity (64 hex characters) of the whole table: labels, axes,
    /// levels, every count, snapshot, design and structural zeros. A changed
    /// observation changes it even when labels and shape match.
    #[must_use]
    pub fn content_identity(&self) -> String {
        let mut w = CanonicalWriter::new(SAMPLED_COUNT_IDENTITY_TAG);
        self.write_identity(&mut w);
        w.finish_hex()
    }

    pub(crate) fn checked_content_identity(
        &self,
        work: &mut SampledWork<'_>,
    ) -> Result<String, EstimationError> {
        let mut w = CanonicalWriter::new(SAMPLED_COUNT_IDENTITY_TAG);
        self.write_identity_checked(&mut w, work)?;
        work.charge(1)?;
        Ok(w.finish_hex())
    }

    pub(crate) fn write_identity_checked(
        &self,
        w: &mut CanonicalWriter,
        work: &mut SampledWork<'_>,
    ) -> Result<(), EstimationError> {
        work.charge(self.population.len())?;
        w.try_bytes(self.population.as_bytes(), || work.charge(0))?
            .word(u64::from(self.regime.raw()));
        w.word(self.interventions.len() as u64);
        for chunk in self.interventions.chunks(512) {
            work.charge(chunk.len().saturating_mul(2))?;
            for (v, l) in chunk {
                w.word(u64::from(v.raw())).word(level_bits(*l));
            }
        }
        w.word(self.axes.len() as u64);
        for axis in &self.axes {
            work.charge(2)?;
            w.word(u64::from(axis.variable.raw())).word(axis.levels.len() as u64);
            for chunk in axis.levels.chunks(512) {
                work.charge(chunk.len())?;
                for level in chunk {
                    w.word(level_bits(*level));
                }
            }
        }
        w.word(self.counts.len() as u64);
        for chunk in self.counts.chunks(512) {
            work.charge(chunk.len())?;
            for count in chunk {
                w.word(*count);
            }
        }
        work.charge(self.snapshot_identity.len())?;
        w.try_bytes(self.snapshot_identity.as_bytes(), || work.charge(0))?;
        match &self.design {
            SamplingDesign::IndependentUnits => {
                w.word(0);
            }
            SamplingDesign::Stratified { variables } => {
                w.word(1).word(variables.len() as u64);
                for chunk in variables.chunks(512) {
                    work.charge(chunk.len())?;
                    for v in chunk {
                        w.word(u64::from(v.raw()));
                    }
                }
            }
        }
        w.word(self.structural_zeros.len() as u64);
        for chunk in self.structural_zeros.chunks(512) {
            work.charge(chunk.len())?;
            for cell in chunk {
                w.word(*cell as u64);
            }
        }
        work.charge(1)
    }

    pub(crate) fn write_identity(&self, w: &mut CanonicalWriter) {
        w.text(&self.population).word(u64::from(self.regime.raw()));
        w.word(self.interventions.len() as u64);
        for (v, l) in &self.interventions {
            w.word(u64::from(v.raw())).word(level_bits(*l));
        }
        w.word(self.axes.len() as u64);
        for axis in &self.axes {
            w.word(u64::from(axis.variable.raw()));
            w.word(axis.levels.len() as u64);
            for level in &axis.levels {
                w.word(level_bits(*level));
            }
        }
        w.word(self.counts.len() as u64);
        for count in &self.counts {
            w.word(*count);
        }
        w.text(&self.snapshot_identity);
        match &self.design {
            SamplingDesign::IndependentUnits => {
                w.word(0);
            }
            SamplingDesign::Stratified { variables } => {
                w.word(1).word(variables.len() as u64);
                for v in variables {
                    w.word(u64::from(v.raw()));
                }
            }
        }
        w.word(self.structural_zeros.len() as u64);
        for cell in &self.structural_zeros {
            w.word(*cell as u64);
        }
    }

    /// Stratum index of every cell for a stratified design (all cells in stratum
    /// zero for independent units), with the number of strata.
    pub(crate) fn strata(
        &self,
        work: &mut SampledWork<'_>,
    ) -> Result<(Vec<usize>, usize), EstimationError> {
        let SamplingDesign::Stratified { variables } = &self.design else {
            let mut out = Vec::with_capacity(self.counts.len());
            for chunk in self.counts.chunks(512) {
                work.charge(0)?;
                out.resize(out.len() + chunk.len(), 0);
            }
            return Ok((out, 1));
        };
        let n_axes = self.axes.len();
        let mut strides = vec![1usize; n_axes];
        for a in (0..n_axes.saturating_sub(1)).rev() {
            work.charge(0)?;
            strides[a] = strides[a + 1] * self.axes[a + 1].levels.len();
        }
        let mut chosen = Vec::new();
        let mut strata = 1usize;
        for a in 0..n_axes {
            work.charge(variables.len())?;
            let mut contains = false;
            for chunk in variables.chunks(512) {
                work.charge(0)?;
                if chunk.contains(&self.axes[a].variable) {
                    contains = true;
                    break;
                }
            }
            if contains {
                chosen.push(a);
                strata *= self.axes[a].levels.len();
            }
        }
        let mut out = Vec::with_capacity(self.counts.len());
        for cell in 0..self.counts.len() {
            if cell % 512 == 0 {
                work.charge(0)?;
            }
            let mut stratum = 0usize;
            for (index, &a) in chosen.iter().enumerate() {
                if index % 512 == 0 {
                    work.charge(0)?;
                }
                let level = (cell / strides[a]) % self.axes[a].levels.len();
                stratum = stratum * self.axes[a].levels.len() + level;
            }
            out.push(stratum);
        }
        work.charge(0)?;
        Ok((out, strata))
    }
}

/// How an unseen cell (a zero count that is not a declared structural zero) is
/// handled. There is no silent zero fill: every route either refuses, declares
/// a smoothing rule that is recorded and labelled, or reports a structural
/// envelope.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnseenCellRule {
    /// Refuse any table with an unseen cell.
    Refuse,
    /// Additive (Dirichlet) smoothing: every non-structural cell receives
    /// `pseudo_count` before normalization, `p = (n + a) / (N + K a)` where `K`
    /// is the number of non-structural cells. The law is labelled smoothed and
    /// the pseudo-count is part of the result and the artifact.
    AdditiveSmoothing {
        /// Pseudo-count per cell, finite and positive.
        pseudo_count: f64,
    },
    /// Partial identification: `unseen_mass` of each table is declared to sit on
    /// unseen cells; the result is the envelope of the finite family of fillings
    /// that put all of it on one unseen cell (or none), evaluated through the
    /// stored derivation. The envelope is a structural range over that family,
    /// never an interval and never a point.
    UnseenMassEnvelope {
        /// Declared mass in `(0, 1)` assigned to the unseen cells of a table.
        unseen_mass: f64,
    },
}

impl UnseenCellRule {
    /// Stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Refuse => "refuse",
            Self::AdditiveSmoothing { .. } => "additive_smoothing",
            Self::UnseenMassEnvelope { .. } => "unseen_mass_envelope",
        }
    }

    /// The rule's numeric parameter, if any.
    #[must_use]
    pub const fn parameter(self) -> Option<f64> {
        match self {
            Self::Refuse => None,
            Self::AdditiveSmoothing { pseudo_count } => Some(pseudo_count),
            Self::UnseenMassEnvelope { unseen_mass } => Some(unseen_mass),
        }
    }

    pub(crate) fn validate(self) -> Result<(), EstimationError> {
        match self {
            Self::Refuse => Ok(()),
            Self::AdditiveSmoothing { pseudo_count }
                if pseudo_count.is_finite() && pseudo_count > 0.0 =>
            {
                Ok(())
            }
            Self::UnseenMassEnvelope { unseen_mass }
                if unseen_mass.is_finite() && unseen_mass > 0.0 && unseen_mass < 1.0 =>
            {
                Ok(())
            }
            _ => Err(refused(
                antecedent_core::reason_code!("invalid_argument"),
                DETAIL_INVALID_COUNTS,
                "the unseen-cell rule's parameter is not a finite positive pseudo-count or a \
                 mass in (0, 1)",
            )),
        }
    }
}

/// Sample-size and sparsity limits of a sampled request.
#[derive(Clone, Copy, Debug)]
pub struct SampledTransportLimits {
    /// Fewest observations a table may carry.
    pub min_table_total: u64,
    /// Largest share of a table's non-structural cells that may be unseen under
    /// a smoothing or envelope rule (a refuse rule refuses any unseen cell).
    pub max_unseen_fraction: f64,
    /// Most members of an envelope family evaluated.
    pub max_envelope_evaluations: u32,
    /// Shared operation allowance for point preparation and evaluation and all
    /// gradient preparation and evaluations. Envelope members each use their
    /// own evaluation allowance.
    pub evaluation: antecedent_expr::ExactEvaluationLimits,
}

impl Default for SampledTransportLimits {
    fn default() -> Self {
        Self {
            min_table_total: 20,
            max_unseen_fraction: 0.5,
            max_envelope_evaluations: 4096,
            evaluation: antecedent_expr::ExactEvaluationLimits::default(),
        }
    }
}

/// How a table's law was produced; fixes its provenance label on the law.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LawFlavor {
    /// Probabilities equal count shares.
    Empirical,
    /// Model-based probabilities (smoothed or perturbed): counts only retain
    /// the observed support.
    Model,
}

/// Probabilities of a table under a rule (envelope: the unsmoothed shares) and
/// whether they are smoothed.
///
/// # Errors
///
/// Sample-size and sparsity refusals, and a refuse rule's unseen-cell refusal.
pub(crate) fn table_probabilities(
    table: &SampledRegimeCounts,
    rule: UnseenCellRule,
    limits: &SampledTransportLimits,
    work: &mut SampledWork<'_>,
) -> Result<(Vec<f64>, bool), EstimationError> {
    let mut total = 0u64;
    for chunk in table.counts.chunks(512) {
        work.charge(0)?;
        total += chunk.iter().sum::<u64>();
    }
    work.charge(0)?;
    if total < limits.min_table_total {
        return Err(refused(
            antecedent_core::reason_code!("transport_support_failure"),
            DETAIL_SAMPLE_SIZE,
            format!(
                "population {} regime {} has {total} observations; the declared minimum is {}",
                super::name_preview(&table.population),
                table.regime.raw(),
                limits.min_table_total
            ),
        ));
    }
    let structural = table.structural_zeros.len();
    let live = table.counts.len() - structural;
    let unseen = table.checked_unseen_cells(work)?;
    if !unseen.is_empty() {
        match rule {
            UnseenCellRule::Refuse => {
                return Err(refused(
                    antecedent_core::reason_code!("transport_support_failure"),
                    DETAIL_UNSEEN,
                    format!(
                        "population {} regime {} has {} unseen cells of {live}; declare them \
                         structural zeros, or choose additive smoothing or an unseen-mass \
                         envelope (an unseen cell is never filled with zero silently)",
                        super::name_preview(&table.population),
                        table.regime.raw(),
                        unseen.len()
                    ),
                ));
            }
            UnseenCellRule::AdditiveSmoothing { .. }
            | UnseenCellRule::UnseenMassEnvelope { .. } => {
                let fraction = unseen.len() as f64 / live.max(1) as f64;
                if fraction > limits.max_unseen_fraction {
                    return Err(refused(
                        antecedent_core::reason_code!("transport_support_failure"),
                        DETAIL_SPARSE,
                        format!(
                            "population {} regime {} has {} of {live} cells unseen, above the \
                             declared sparsity limit {}",
                            super::name_preview(&table.population),
                            table.regime.raw(),
                            unseen.len(),
                            limits.max_unseen_fraction
                        ),
                    ));
                }
            }
        }
    }
    let n = total as f64;
    match rule {
        UnseenCellRule::AdditiveSmoothing { pseudo_count } if !unseen.is_empty() => {
            let denominator = n + live as f64 * pseudo_count;
            let mut probabilities = Vec::with_capacity(table.counts.len());
            for (block, chunk) in table.counts.chunks(512).enumerate() {
                work.charge(0)?;
                for (offset, &count) in chunk.iter().enumerate() {
                    let c = block * 512 + offset;
                    probabilities.push(if table.structural_zeros.binary_search(&c).is_ok() {
                        0.0
                    } else {
                        (count as f64 + pseudo_count) / denominator
                    });
                }
            }
            work.charge(0)?;
            Ok((probabilities, true))
        }
        _ => {
            let mut probabilities = Vec::with_capacity(table.counts.len());
            for chunk in table.counts.chunks(512) {
                work.charge(0)?;
                probabilities.extend(chunk.iter().map(|c| *c as f64 / n));
            }
            work.charge(0)?;
            Ok((probabilities, false))
        }
    }
}

/// The dense law of a table with the given probabilities.
///
/// # Errors
///
/// `invalid_argument` when the law does not validate (a probability outside
/// `[0, 1]` or a table that does not sum to one).
pub(crate) fn law_of(
    table: &SampledRegimeCounts,
    probabilities: &[f64],
    flavor: LawFlavor,
    work: &mut SampledWork<'_>,
) -> Result<ExactDiscreteLaw, EstimationError> {
    let mut axes = Vec::with_capacity(table.axes.len());
    for axis in &table.axes {
        let mut values = Vec::with_capacity(axis.levels.len());
        for chunk in axis.levels.chunks(512) {
            work.charge(0)?;
            values.extend(chunk.iter().map(|l| Value::f64(*l)));
        }
        axes.push(DiscreteAxis { variable: axis.variable, values: values.into() });
    }
    let mut interventions = Vec::with_capacity(table.interventions.len());
    for chunk in table.interventions.chunks(512) {
        work.charge(0)?;
        interventions.extend(
            chunk.iter().map(|(v, l)| InterventionAssignment::concrete(*v, Value::f64(*l))),
        );
    }
    let mut copied_probabilities = Vec::with_capacity(probabilities.len());
    for chunk in probabilities.chunks(512) {
        work.charge(0)?;
        copied_probabilities.extend_from_slice(chunk);
    }
    work.charge(0)?;
    let law = ExactDiscreteLaw::try_new(
        Arc::clone(&table.population),
        table.regime,
        interventions,
        axes,
        copied_probabilities,
        Arc::clone(&table.snapshot_identity),
        LawTolerance::default(),
    )
    .map_err(|e| map_law_error(e, "the estimated table is not a finite law"))?;
    let mut counts = Vec::with_capacity(table.counts.len());
    for chunk in table.counts.chunks(512) {
        work.charge(0)?;
        counts.extend_from_slice(chunk);
    }
    work.charge(0)?;
    let law = match flavor {
        LawFlavor::Empirical => law.with_empirical_counts(counts),
        LawFlavor::Model => law.with_learned_support(counts),
    };
    law.map_err(|e| map_law_error(e, "the estimated table does not reconcile with its counts"))
}

pub(crate) fn map_law_error(e: antecedent_expr::ExactLawError, message: &str) -> EstimationError {
    if e.kind == "cancelled" {
        refused(
            antecedent_core::reason_code!("transport_budget_cancel"),
            DETAIL_CANCELLED,
            "the sampled evaluation was cancelled",
        )
    } else {
        refused(
            antecedent_core::reason_code!("invalid_argument"),
            DETAIL_INVALID_COUNTS,
            format!("{message}: {e}"),
        )
    }
}
