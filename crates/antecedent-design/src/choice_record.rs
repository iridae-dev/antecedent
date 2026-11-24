//! Recorded past choices and stated comparisons, and the expected linear-form terms of every
//! available option.
//!
//! A [`ChoiceRecord`] states which alternatives a decision-maker had, what was expected of each
//! (a [`OptionBelief`]: attribute means or aligned joint draws), which one was chosen, where the
//! expectations came from ([`BeliefSource`]) and which evidence-ledger observation rows the record
//! rests on. Realized outcomes are never read as what the decision-maker expected unless that
//! assumption is named, and pooling the choices of several decision-makers into one set of
//! preferences is itself a named assumption ([`Pooling::Pooled`]).
//!
//! The same observation row under two records would count one piece of evidence twice, so it is
//! refused. Rows are compared as evidence-ledger row identities are: by their exact spelling,
//! after surrounding whitespace is trimmed (no other normalisation is applied).
//!
//! A stated comparison ([`BeliefSource::Stated`]) is a record of what someone said they would
//! choose among hypothetical outcome bundles in an elicitation exercise, not of what they chose.
//! It is read as evidence about preferences only under a named assumption that the statement
//! reflects the preferences the person would act on. A hypothetical bundle is known exactly, so
//! every option of a stated record is given by its attribute values ([`OptionBelief::Means`]).
//!
//! An option can be listed as [`OptionBelief::Unavailable`]: it existed at the time of the choice
//! but could not be chosen, because a named constraint (a budget cap, a regulation, a policy)
//! excluded it. Choosing another option over it says nothing about preferences, so an unavailable
//! option teaches nothing: it takes part in the record's digest and is reported by
//! [`choice_features`], but it contributes no features and no comparison with the chosen option.
//! The chosen option must be available. A record lists at least two options; one with a single
//! available option is valid and teaches nothing.
//!
//! [`choice_features`] reads a [`ValueModel`] whose value is `fixed + sum_j theta_j * terms[j]`
//! and gives, for every available option of every record, the expectation of the fixed part and
//! of each term under the option's belief. Every fixed parameter (`lower == upper`) is substituted by its
//! value before the linear form is taken, so its column of [`ChoiceSetFeatures::terms`] is zero
//! and its contribution is folded into [`ChoiceSetFeatures::fixed`]; the nonzero columns are
//! exactly the free parameters. A term's expectation from attribute means is its value at the
//! means, which is exact only for a term affine in the attributes; any other term needs draws.
//!
//! A value that is not affine in some free parameters (shape parameters: a risk attitude inside
//! an exponential, a breakpoint inside a kink) has no linear form, and [`choice_features`] refuses
//! it by name. [`choice_features_at`] takes the features at a given value of the shape
//! parameters instead: the shape values are substituted, and the value is then
//! `fixed + sum_j theta_j * terms[j]` over the remaining free parameters, so the columns of the
//! shape parameters are zero and their effect at the given point is in the fixed part.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use antecedent_contract::ExternalRefusal;
use antecedent_core::{CanonicalWriter, is_blank};

use crate::preference_model::{
    Attribute, LinearForm, PreferenceExpr, PreferenceModelError, ShapeSplit, ValueModel,
};
use crate::preference_refusal::{DECLARE, at, refusal};
use crate::signal::at_stage;

/// Where the expected outcomes of an option came from. Realized outcomes are never assumed to be
/// what the decision-maker expected unless that assumption is named.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BeliefSource {
    /// A forecast recorded before the choice.
    RecordedForecast {
        /// Identity of the recorded forecast; non-blank.
        source_id: String,
    },
    /// The result of an analysis as it stood at a date.
    AnalysisAsOf {
        /// Identity of the analysis result; non-blank.
        result_identity: String,
        /// The date the result is read as of; non-blank.
        as_of: String,
    },
    /// The outcomes that were later realized, read as the expectations under a named assumption.
    RealizedOutcome {
        /// Identity of the assumption that realized outcomes stand for expectations; non-blank.
        assumption_id: String,
    },
    /// A stated comparison: what the decider said they would choose among hypothetical outcome
    /// bundles, not what they chose. The options are the bundles, each given exactly by its
    /// attribute values ([`OptionBelief::Means`]).
    Stated {
        /// Identity of the elicitation exercise the statement came from; non-blank.
        elicitation_id: String,
        /// Identity of the assumption that a stated comparison reflects the preferences the
        /// person would act on; non-blank.
        assumption_id: String,
    },
}

impl BeliefSource {
    /// Whether the record is a stated comparison rather than a recorded choice.
    #[must_use]
    pub const fn is_stated(&self) -> bool {
        matches!(self, Self::Stated { .. })
    }
}

/// What was expected of one option.
#[derive(Clone, Debug, PartialEq)]
pub enum OptionBelief {
    /// Expected value of each attribute, one per attribute in attribute order.
    Means(Vec<f64>),
    /// Aligned joint draws of the attributes.
    Draws {
        /// Row-major `n x attributes` table, one row per joint draw; `n >= 1`.
        rows: Vec<f64>,
        /// Optional non-negative weight per row with a positive sum; uniform when absent.
        weights: Option<Vec<f64>>,
    },
    /// The option existed but could not be chosen: a named constraint excluded it. Nothing was
    /// expected of it for the purpose of the choice, and it teaches nothing about preferences.
    Unavailable {
        /// Identity of the constraint that excluded the option; non-blank.
        constraint_id: String,
    },
}

impl OptionBelief {
    /// Whether the option could be chosen (it is not [`OptionBelief::Unavailable`]).
    #[must_use]
    pub const fn is_available(&self) -> bool {
        !matches!(self, Self::Unavailable { .. })
    }
}

/// One alternative listed at a recorded choice.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceOption {
    /// Identity of the option within its record; non-blank and distinct.
    pub id: String,
    /// What was expected of the option, or that it was unavailable.
    pub belief: OptionBelief,
}

/// One recorded choice.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceRecord {
    /// Identity of the record; non-blank and distinct within a set.
    pub id: String,
    /// Who chose, or for a stated comparison who made the statement; non-blank.
    pub decider: String,
    /// The alternatives listed at the choice; at least two. An option whose belief is
    /// [`OptionBelief::Unavailable`] existed but could not be chosen and teaches nothing.
    pub options: Vec<ChoiceOption>,
    /// Identity of the chosen option (for a stated comparison, the bundle the decider said they
    /// would take); it must be available.
    pub chosen: String,
    /// Where the beliefs came from.
    pub belief_source: BeliefSource,
    /// Evidence-ledger row identities this record rests on (for a stated comparison, the
    /// identity of the response); non-empty, non-blank, distinct.
    pub observation_rows: Vec<String>,
}

/// How the choices of several decision-makers are read.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Pooling {
    /// Each decision-maker's choices are read separately.
    PerDecider,
    /// All choices are read as coming from one set of preferences; the assumption is named.
    Pooled {
        /// Identity of the pooling assumption; non-blank.
        assumption_id: String,
    },
}

/// A set of recorded choices over declared attributes.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceRecordSet {
    /// Outcome attributes every belief is stated over; must equal the value model's attributes
    /// (ids and units, in order).
    pub attributes: Vec<Attribute>,
    /// Population the decision-makers belong to; non-blank.
    pub population: String,
    /// The recorded choices; at least one.
    pub records: Vec<ChoiceRecord>,
    /// How the decision-makers' choices are read.
    pub pooling: Pooling,
}

/// Why a choice record set or a feature extraction is refused.
#[derive(Clone, Debug, PartialEq)]
pub enum ChoiceRecordError {
    /// The value model is invalid.
    Model(PreferenceModelError),
    /// No attributes are declared.
    AttributesMissing,
    /// An attribute identity is blank.
    AttributeIdBlank {
        /// Position of the attribute.
        index: usize,
    },
    /// An attribute's units are blank.
    AttributeUnitsBlank {
        /// Identity of the attribute.
        attribute: String,
    },
    /// Two attributes share an identity.
    AttributeDuplicate {
        /// The repeated identity.
        attribute: String,
    },
    /// The population is blank.
    PopulationBlank,
    /// Pooling is declared without naming its assumption.
    PoolingAssumptionBlank,
    /// The set holds no records.
    RecordsMissing,
    /// A record identity is blank.
    RecordIdBlank {
        /// Position of the record.
        index: usize,
    },
    /// Two records share an identity.
    RecordDuplicate {
        /// The repeated identity.
        record: String,
    },
    /// A record's decider is blank.
    DeciderBlank {
        /// The record.
        record: String,
    },
    /// A record lists fewer than two options.
    OptionsTooFew {
        /// The record.
        record: String,
        /// Number of options declared.
        options: usize,
    },
    /// An option identity is blank.
    OptionIdBlank {
        /// The record.
        record: String,
        /// Position of the option.
        index: usize,
    },
    /// Two options of a record share an identity.
    OptionDuplicate {
        /// The record.
        record: String,
        /// The repeated identity.
        option: String,
    },
    /// The chosen identity is not among the record's options.
    ChosenUnknown {
        /// The record.
        record: String,
        /// The chosen identity as declared.
        chosen: String,
    },
    /// An unavailable option does not name the constraint that excluded it.
    UnavailableConstraintBlank {
        /// The record.
        record: String,
        /// The option.
        option: String,
    },
    /// The chosen option is marked unavailable.
    ChosenUnavailable {
        /// The record.
        record: String,
        /// The chosen option.
        option: String,
    },
    /// A stated comparison does not name its elicitation exercise.
    StatedElicitationBlank {
        /// The record.
        record: String,
    },
    /// A stated comparison does not name the assumption that it reflects acted-on preferences.
    StatedAssumptionBlank {
        /// The record.
        record: String,
    },
    /// An option of a stated comparison is not given exactly by its attribute values.
    StatedBeliefNotExact {
        /// The record.
        record: String,
        /// The option.
        option: String,
    },
    /// A recorded forecast does not name its source.
    ForecastSourceBlank {
        /// The record.
        record: String,
    },
    /// An analysis belief does not name its result identity.
    AnalysisIdentityBlank {
        /// The record.
        record: String,
    },
    /// An analysis belief does not name its as-of date.
    AnalysisDateBlank {
        /// The record.
        record: String,
    },
    /// A realized-outcome belief does not name its assumption.
    RealizedAssumptionBlank {
        /// The record.
        record: String,
    },
    /// A belief of means does not hold one value per attribute.
    MeansWidthMismatch {
        /// The record.
        record: String,
        /// The option.
        option: String,
        /// Number of attributes.
        attributes: usize,
        /// Number of means given.
        means: usize,
    },
    /// A belief of draws holds no rows.
    DrawsEmpty {
        /// The record.
        record: String,
        /// The option.
        option: String,
    },
    /// A belief of draws is not a whole number of attribute rows.
    DrawsWidthMismatch {
        /// The record.
        record: String,
        /// The option.
        option: String,
        /// Number of attributes.
        attributes: usize,
        /// Number of values given.
        values: usize,
    },
    /// A belief value is not finite.
    BeliefNotFinite {
        /// The record.
        record: String,
        /// The option.
        option: String,
    },
    /// Draw weights are not one per row.
    WeightsLengthMismatch {
        /// The record.
        record: String,
        /// The option.
        option: String,
        /// Number of draw rows.
        rows: usize,
        /// Number of weights given.
        weights: usize,
    },
    /// A draw weight is not finite.
    WeightNotFinite {
        /// The record.
        record: String,
        /// The option.
        option: String,
        /// The draw row.
        row: usize,
    },
    /// A draw weight is negative.
    WeightNegative {
        /// The record.
        record: String,
        /// The option.
        option: String,
        /// The draw row.
        row: usize,
    },
    /// The draw weights do not have a positive, finite sum.
    WeightsSumInvalid {
        /// The record.
        record: String,
        /// The option.
        option: String,
    },
    /// A record names no observation rows.
    ObservationRowsMissing {
        /// The record.
        record: String,
    },
    /// An observation row identity is blank.
    ObservationRowBlank {
        /// The record.
        record: String,
        /// Position of the row.
        index: usize,
    },
    /// A record names the same observation row twice.
    ObservationRowRepeated {
        /// The record.
        record: String,
        /// The row identity, trimmed.
        row: String,
    },
    /// Two records rest on the same observation row, which would count it twice.
    ObservationRowDoubleCounted {
        /// The row identity, trimmed.
        row: String,
        /// The earlier record in declaration order.
        first_record: String,
        /// The later record.
        second_record: String,
    },
    /// The records' attributes differ from the value model's.
    AttributesDiffer {
        /// Position of the first difference.
        index: usize,
        /// The model's attribute there, if it has one.
        model: Option<Attribute>,
        /// The records' attribute there, if they have one.
        records: Option<Attribute>,
    },
    /// The value is not affine in parameters whose range is not a single value.
    ShapeParametersNotFixed {
        /// Identities of those parameters, in parameter order.
        parameters: Vec<String>,
    },
    /// A term that is not affine in the attributes has no expectation from attribute means.
    BeliefLawRequired {
        /// The record.
        record: String,
        /// The option.
        option: String,
        /// The term: `None` for the fixed part, `Some(j)` for parameter `j`'s term.
        term: Option<usize>,
    },
    /// A term's value or expectation is not finite.
    TermNotFinite {
        /// The record.
        record: String,
        /// The option.
        option: String,
    },
}

impl ChoiceRecordError {
    /// Stable detail identity of the refusal.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        use antecedent_core::reason_details;
        match self {
            Self::Model(_) => reason_details::choice_record::MODEL_INVALID,
            Self::AttributesMissing => reason_details::choice_record::ATTRIBUTES_MISSING,
            Self::AttributeIdBlank { .. } => reason_details::choice_record::ATTRIBUTE_ID_BLANK,
            Self::AttributeUnitsBlank { .. } => {
                reason_details::choice_record::ATTRIBUTE_UNITS_BLANK
            }
            Self::AttributeDuplicate { .. } => reason_details::choice_record::ATTRIBUTE_DUPLICATE,
            Self::PopulationBlank => reason_details::choice_record::POPULATION_BLANK,
            Self::PoolingAssumptionBlank => reason_details::choice_record::POOLING_ASSUMPTION_BLANK,
            Self::RecordsMissing => reason_details::choice_record::RECORDS_MISSING,
            Self::RecordIdBlank { .. } => reason_details::choice_record::RECORD_ID_BLANK,
            Self::RecordDuplicate { .. } => reason_details::choice_record::RECORD_DUPLICATE,
            Self::DeciderBlank { .. } => reason_details::choice_record::DECIDER_BLANK,
            Self::OptionsTooFew { .. } => reason_details::choice_record::OPTIONS_TOO_FEW,
            Self::OptionIdBlank { .. } => reason_details::choice_record::OPTION_ID_BLANK,
            Self::OptionDuplicate { .. } => reason_details::choice_record::OPTION_DUPLICATE,
            Self::ChosenUnknown { .. } => reason_details::choice_record::CHOSEN_UNKNOWN,
            Self::UnavailableConstraintBlank { .. } => {
                reason_details::choice_record::UNAVAILABLE_CONSTRAINT_BLANK
            }
            Self::ChosenUnavailable { .. } => reason_details::choice_record::CHOSEN_UNAVAILABLE,
            Self::StatedElicitationBlank { .. } => {
                reason_details::choice_record::STATED_ELICITATION_BLANK
            }
            Self::StatedAssumptionBlank { .. } => {
                reason_details::choice_record::STATED_ASSUMPTION_BLANK
            }
            Self::StatedBeliefNotExact { .. } => {
                reason_details::choice_record::STATED_BELIEF_NOT_EXACT
            }
            Self::ForecastSourceBlank { .. } => {
                reason_details::choice_record::FORECAST_SOURCE_BLANK
            }
            Self::AnalysisIdentityBlank { .. } => {
                reason_details::choice_record::ANALYSIS_IDENTITY_BLANK
            }
            Self::AnalysisDateBlank { .. } => reason_details::choice_record::ANALYSIS_DATE_BLANK,
            Self::RealizedAssumptionBlank { .. } => {
                reason_details::choice_record::REALIZED_ASSUMPTION_BLANK
            }
            Self::MeansWidthMismatch { .. } => reason_details::choice_record::MEANS_WIDTH_MISMATCH,
            Self::DrawsEmpty { .. } => reason_details::choice_record::DRAWS_EMPTY,
            Self::DrawsWidthMismatch { .. } => reason_details::choice_record::DRAWS_WIDTH_MISMATCH,
            Self::BeliefNotFinite { .. } => reason_details::choice_record::BELIEF_NOT_FINITE,
            Self::WeightsLengthMismatch { .. } => {
                reason_details::choice_record::WEIGHTS_LENGTH_MISMATCH
            }
            Self::WeightNotFinite { .. } => reason_details::choice_record::WEIGHT_NOT_FINITE,
            Self::WeightNegative { .. } => reason_details::choice_record::WEIGHT_NEGATIVE,
            Self::WeightsSumInvalid { .. } => reason_details::choice_record::WEIGHTS_SUM_INVALID,
            Self::ObservationRowsMissing { .. } => {
                reason_details::choice_record::OBSERVATION_ROWS_MISSING
            }
            Self::ObservationRowBlank { .. } => {
                reason_details::choice_record::OBSERVATION_ROW_BLANK
            }
            Self::ObservationRowRepeated { .. } => {
                reason_details::choice_record::OBSERVATION_ROW_REPEATED
            }
            Self::ObservationRowDoubleCounted { .. } => {
                reason_details::choice_record::OBSERVATION_ROW_DOUBLE_COUNTED
            }
            Self::AttributesDiffer { .. } => reason_details::choice_record::ATTRIBUTES_DIFFER,
            Self::ShapeParametersNotFixed { .. } => {
                reason_details::choice_record::SHAPE_PARAMETERS_NOT_FIXED
            }
            Self::BeliefLawRequired { .. } => reason_details::choice_record::BELIEF_LAW_REQUIRED,
            Self::TermNotFinite { .. } => reason_details::choice_record::TERM_NOT_FINITE,
        }
    }
}

impl fmt::Display for ChoiceRecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(f, "invalid value model ({})", error.detail()),
            Self::ObservationRowDoubleCounted { row, first_record, second_record } => write!(
                f,
                "observation row `{row}` is used by records `{first_record}` and `{second_record}`"
            ),
            Self::BeliefLawRequired { record, option, term } => match term {
                Some(j) => write!(
                    f,
                    "record `{record}` option `{option}`: term {j} is not affine in the attributes; draws are required"
                ),
                None => write!(
                    f,
                    "record `{record}` option `{option}`: the fixed part is not affine in the attributes; draws are required"
                ),
            },
            other => write!(f, "choice records refused ({}): {other:?}", other.detail()),
        }
    }
}

impl std::error::Error for ChoiceRecordError {}

impl ChoiceRecordSet {
    /// Check the set: attributes, population, pooling, every record, and that no observation row
    /// is used by two records.
    ///
    /// # Errors
    /// The first [`ChoiceRecordError`] found, in declaration order.
    pub fn validate(&self) -> Result<(), ChoiceRecordError> {
        self.validate_attributes()?;
        if is_blank(&self.population) {
            return Err(ChoiceRecordError::PopulationBlank);
        }
        if let Pooling::Pooled { assumption_id } = &self.pooling
            && is_blank(assumption_id)
        {
            return Err(ChoiceRecordError::PoolingAssumptionBlank);
        }
        if self.records.is_empty() {
            return Err(ChoiceRecordError::RecordsMissing);
        }
        let width = self.attributes.len();
        let mut ids = BTreeSet::new();
        let mut row_owner: BTreeMap<&str, &str> = BTreeMap::new();
        for (index, record) in self.records.iter().enumerate() {
            if is_blank(&record.id) {
                return Err(ChoiceRecordError::RecordIdBlank { index });
            }
            if !ids.insert(record.id.as_str()) {
                return Err(ChoiceRecordError::RecordDuplicate { record: record.id.clone() });
            }
            validate_record(record, width)?;
            let mut own = BTreeSet::new();
            for row in &record.observation_rows {
                let row = row.trim();
                if !own.insert(row) {
                    return Err(ChoiceRecordError::ObservationRowRepeated {
                        record: record.id.clone(),
                        row: row.to_owned(),
                    });
                }
                if let Some(first) = row_owner.insert(row, record.id.as_str()) {
                    return Err(ChoiceRecordError::ObservationRowDoubleCounted {
                        row: row.to_owned(),
                        first_record: first.to_owned(),
                        second_record: record.id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_attributes(&self) -> Result<(), ChoiceRecordError> {
        if self.attributes.is_empty() {
            return Err(ChoiceRecordError::AttributesMissing);
        }
        let mut seen = BTreeSet::new();
        for (index, attribute) in self.attributes.iter().enumerate() {
            if is_blank(&attribute.id) {
                return Err(ChoiceRecordError::AttributeIdBlank { index });
            }
            if is_blank(&attribute.units) {
                return Err(ChoiceRecordError::AttributeUnitsBlank {
                    attribute: attribute.id.clone(),
                });
            }
            if !seen.insert(attribute.id.as_str()) {
                return Err(ChoiceRecordError::AttributeDuplicate {
                    attribute: attribute.id.clone(),
                });
            }
        }
        Ok(())
    }

    /// Canonical digest of the set (domain `antecedent.choice_records.v1`): attributes in order,
    /// population, pooling, and the records sorted by identity. Within a record: decider, options
    /// in declared order with their beliefs (an unavailable option by its identity, a marker and
    /// its constraint identity), the chosen identity, the belief source, and the
    /// trimmed observation rows sorted. Independent of record order; changes with any value.
    ///
    /// # Errors
    /// The set is invalid ([`Self::validate`]).
    pub fn digest(&self) -> Result<String, ChoiceRecordError> {
        self.validate()?;
        let mut out = CanonicalWriter::new("antecedent.choice_records.v1");
        out.word(self.attributes.len() as u64);
        for attribute in &self.attributes {
            out.text(&attribute.id).text(&attribute.units);
        }
        out.text(&self.population);
        match &self.pooling {
            Pooling::PerDecider => out.word(1),
            Pooling::Pooled { assumption_id } => out.word(2).text(assumption_id),
        };
        let mut records: Vec<&ChoiceRecord> = self.records.iter().collect();
        records.sort_by(|a, b| a.id.cmp(&b.id));
        out.word(records.len() as u64);
        for record in records {
            out.bytes(&encode_record(record));
        }
        Ok(out.finish_hex())
    }

    /// Decision-makers named by the records, sorted and distinct.
    #[must_use]
    pub fn deciders(&self) -> Vec<String> {
        let set: BTreeSet<&str> = self.records.iter().map(|r| r.decider.as_str()).collect();
        set.into_iter().map(str::to_owned).collect()
    }

    /// Assumptions the set stands on, sorted and distinct: the pooling assumption, every
    /// assumption that realized outcomes stand for expectations, and every assumption that a
    /// stated comparison reflects the preferences the person would act on.
    #[must_use]
    pub fn assumptions(&self) -> Vec<String> {
        let mut set: BTreeSet<&str> = BTreeSet::new();
        if let Pooling::Pooled { assumption_id } = &self.pooling {
            set.insert(assumption_id.as_str());
        }
        for record in &self.records {
            match &record.belief_source {
                BeliefSource::RealizedOutcome { assumption_id }
                | BeliefSource::Stated { assumption_id, .. } => {
                    set.insert(assumption_id.as_str());
                }
                BeliefSource::RecordedForecast { .. } | BeliefSource::AnalysisAsOf { .. } => {}
            }
        }
        set.into_iter().map(str::to_owned).collect()
    }

    /// Identities of the stated comparisons ([`BeliefSource::Stated`]), sorted.
    #[must_use]
    pub fn stated_records(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .records
            .iter()
            .filter(|r| r.belief_source.is_stated())
            .map(|r| r.id.clone())
            .collect();
        ids.sort();
        ids
    }

    /// Constraints named by unavailable options, sorted and distinct.
    #[must_use]
    pub fn constraints(&self) -> Vec<String> {
        let set: BTreeSet<&str> = self
            .records
            .iter()
            .flat_map(|r| &r.options)
            .filter_map(|o| match &o.belief {
                OptionBelief::Unavailable { constraint_id } => Some(constraint_id.as_str()),
                OptionBelief::Means(_) | OptionBelief::Draws { .. } => None,
            })
            .collect();
        set.into_iter().map(str::to_owned).collect()
    }
}

fn validate_record(record: &ChoiceRecord, width: usize) -> Result<(), ChoiceRecordError> {
    let id = || record.id.clone();
    if is_blank(&record.decider) {
        return Err(ChoiceRecordError::DeciderBlank { record: id() });
    }
    if record.options.len() < 2 {
        return Err(ChoiceRecordError::OptionsTooFew {
            record: id(),
            options: record.options.len(),
        });
    }
    let mut options = BTreeSet::new();
    for (index, option) in record.options.iter().enumerate() {
        if is_blank(&option.id) {
            return Err(ChoiceRecordError::OptionIdBlank { record: id(), index });
        }
        if !options.insert(option.id.as_str()) {
            return Err(ChoiceRecordError::OptionDuplicate {
                record: id(),
                option: option.id.clone(),
            });
        }
        validate_belief(&record.id, option, width)?;
    }
    if !options.contains(record.chosen.as_str()) {
        return Err(ChoiceRecordError::ChosenUnknown {
            record: id(),
            chosen: record.chosen.clone(),
        });
    }
    if record.options.iter().any(|o| o.id == record.chosen && !o.belief.is_available()) {
        return Err(ChoiceRecordError::ChosenUnavailable {
            record: id(),
            option: record.chosen.clone(),
        });
    }
    match &record.belief_source {
        BeliefSource::RecordedForecast { source_id } if is_blank(source_id) => {
            return Err(ChoiceRecordError::ForecastSourceBlank { record: id() });
        }
        BeliefSource::AnalysisAsOf { result_identity, .. } if is_blank(result_identity) => {
            return Err(ChoiceRecordError::AnalysisIdentityBlank { record: id() });
        }
        BeliefSource::AnalysisAsOf { as_of, .. } if is_blank(as_of) => {
            return Err(ChoiceRecordError::AnalysisDateBlank { record: id() });
        }
        BeliefSource::RealizedOutcome { assumption_id } if is_blank(assumption_id) => {
            return Err(ChoiceRecordError::RealizedAssumptionBlank { record: id() });
        }
        BeliefSource::Stated { elicitation_id, .. } if is_blank(elicitation_id) => {
            return Err(ChoiceRecordError::StatedElicitationBlank { record: id() });
        }
        BeliefSource::Stated { assumption_id, .. } if is_blank(assumption_id) => {
            return Err(ChoiceRecordError::StatedAssumptionBlank { record: id() });
        }
        _ => {}
    }
    if record.belief_source.is_stated()
        && let Some(option) =
            record.options.iter().find(|o| !matches!(o.belief, OptionBelief::Means(_)))
    {
        return Err(ChoiceRecordError::StatedBeliefNotExact {
            record: id(),
            option: option.id.clone(),
        });
    }
    if record.observation_rows.is_empty() {
        return Err(ChoiceRecordError::ObservationRowsMissing { record: id() });
    }
    if let Some(index) = record.observation_rows.iter().position(|row| is_blank(row)) {
        return Err(ChoiceRecordError::ObservationRowBlank { record: id(), index });
    }
    Ok(())
}

fn validate_belief(
    record: &str,
    option: &ChoiceOption,
    width: usize,
) -> Result<(), ChoiceRecordError> {
    let names = || (record.to_owned(), option.id.clone());
    match &option.belief {
        OptionBelief::Means(means) => {
            if means.len() != width {
                let (record, option) = names();
                return Err(ChoiceRecordError::MeansWidthMismatch {
                    record,
                    option,
                    attributes: width,
                    means: means.len(),
                });
            }
            if !means.iter().all(|v| v.is_finite()) {
                let (record, option) = names();
                return Err(ChoiceRecordError::BeliefNotFinite { record, option });
            }
        }
        OptionBelief::Draws { rows, weights } => {
            if rows.is_empty() {
                let (record, option) = names();
                return Err(ChoiceRecordError::DrawsEmpty { record, option });
            }
            if rows.len() % width != 0 {
                let (record, option) = names();
                return Err(ChoiceRecordError::DrawsWidthMismatch {
                    record,
                    option,
                    attributes: width,
                    values: rows.len(),
                });
            }
            if !rows.iter().all(|v| v.is_finite()) {
                let (record, option) = names();
                return Err(ChoiceRecordError::BeliefNotFinite { record, option });
            }
            if let Some(weights) = weights {
                let n_rows = rows.len() / width;
                if weights.len() != n_rows {
                    let (record, option) = names();
                    return Err(ChoiceRecordError::WeightsLengthMismatch {
                        record,
                        option,
                        rows: n_rows,
                        weights: weights.len(),
                    });
                }
                for (row, weight) in weights.iter().enumerate() {
                    if !weight.is_finite() {
                        let (record, option) = names();
                        return Err(ChoiceRecordError::WeightNotFinite { record, option, row });
                    }
                    if *weight < 0.0 {
                        let (record, option) = names();
                        return Err(ChoiceRecordError::WeightNegative { record, option, row });
                    }
                }
                let total: f64 = weights.iter().sum();
                if !(total.is_finite() && total > 0.0) {
                    let (record, option) = names();
                    return Err(ChoiceRecordError::WeightsSumInvalid { record, option });
                }
            }
        }
        OptionBelief::Unavailable { constraint_id } => {
            if is_blank(constraint_id) {
                let (record, option) = names();
                return Err(ChoiceRecordError::UnavailableConstraintBlank { record, option });
            }
        }
    }
    Ok(())
}

fn encode_record(record: &ChoiceRecord) -> [u8; 32] {
    let mut out = CanonicalWriter::new("antecedent.choice_records.record.v1");
    out.text(&record.id).text(&record.decider).word(record.options.len() as u64);
    for option in &record.options {
        out.text(&option.id);
        match &option.belief {
            OptionBelief::Means(means) => {
                out.word(1).reals(means);
            }
            OptionBelief::Draws { rows, weights } => {
                out.word(2).reals(rows);
                match weights {
                    Some(weights) => out.word(1).reals(weights),
                    None => out.word(0),
                };
            }
            OptionBelief::Unavailable { constraint_id } => {
                out.word(3).text(constraint_id);
            }
        }
    }
    out.text(&record.chosen);
    match &record.belief_source {
        BeliefSource::RecordedForecast { source_id } => out.word(1).text(source_id),
        BeliefSource::AnalysisAsOf { result_identity, as_of } => {
            out.word(2).text(result_identity).text(as_of)
        }
        BeliefSource::RealizedOutcome { assumption_id } => out.word(3).text(assumption_id),
        BeliefSource::Stated { elicitation_id, assumption_id } => {
            out.word(4).text(elicitation_id).text(assumption_id)
        }
    };
    let mut rows: Vec<&str> =
        record.observation_rows.iter().map(String::as_str).map(str::trim).collect();
    rows.sort_unstable();
    out.texts(&rows);
    out.finish()
}

/// Expected value of every linear-form term for every available option of every record.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceFeatures {
    /// The value model's full parameter count (columns of [`ChoiceSetFeatures::terms`]).
    pub n_parameters: usize,
    /// One entry per record, in record order.
    pub sets: Vec<ChoiceSetFeatures>,
}

/// An option listed at a record that could not be chosen.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnavailableOption {
    /// Identity of the option.
    pub option_id: String,
    /// Identity of the constraint that excluded it.
    pub constraint_id: String,
}

/// Expected linear-form terms of the available options of one record. Unavailable options teach
/// nothing about preferences: they are left out of `option_ids`, `fixed` and `terms` and listed in
/// `unavailable`. A record with a single available option has one row and teaches nothing.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceSetFeatures {
    /// The record.
    pub record_id: String,
    /// Who chose.
    pub decider: String,
    /// Identities of the available options, in declared order.
    pub option_ids: Vec<String>,
    /// Position of the chosen option in `option_ids`.
    pub chosen: usize,
    /// Expected fixed part per available option, including the contribution of every fixed
    /// parameter.
    pub fixed: Vec<f64>,
    /// Row-major `available options x n_parameters`: the expected term of parameter `j` per
    /// available option. The column of a fixed parameter (`lower == upper`) is zero; its
    /// contribution is in `fixed`.
    pub terms: Vec<f64>,
    /// The options listed as unavailable, in declared order, with the constraint that excluded
    /// each.
    pub unavailable: Vec<UnavailableOption>,
}

/// Expected value of the fixed part and of every linear-form term for every available option of
/// every record, under each option's belief.
///
/// An unavailable option ([`OptionBelief::Unavailable`]) teaches nothing about preferences: it
/// has no features, is left out of the option rows (so `chosen` indexes the available options),
/// and is listed in [`ChoiceSetFeatures::unavailable`].
///
/// Every fixed parameter (`lower == upper`) is substituted by its value first, so its column is
/// zero and its contribution is folded into the fixed part. From attribute means a part's
/// expectation is its value at the means, allowed only for a part affine in the attributes; from
/// draws it is the weighted mean of its per-row values (weights normalised to sum one, uniform
/// when absent).
///
/// # Errors
/// - [`ChoiceRecordError::Model`] when the model is invalid, or any [`ChoiceRecordSet::validate`]
///   refusal;
/// - [`ChoiceRecordError::AttributesDiffer`] naming the first attribute that differs;
/// - [`ChoiceRecordError::ShapeParametersNotFixed`] when the value is not affine in parameters
///   whose range is not a single value;
/// - [`ChoiceRecordError::BeliefLawRequired`] for a means belief and a part not affine in the
///   attributes;
/// - [`ChoiceRecordError::TermNotFinite`] when a part's value or expectation is not finite.
pub fn choice_features(
    model: &ValueModel,
    records: &ChoiceRecordSet,
) -> Result<ChoiceFeatures, ChoiceRecordError> {
    model.validate().map_err(ChoiceRecordError::Model)?;
    records.validate()?;
    compare_attributes(&model.attributes, &records.attributes)?;
    let form = fixed_linear_form(model)?;
    let n_parameters = form.terms.len();
    let fixed_columns = fixed_values(model);
    let mut sets = Vec::with_capacity(records.records.len());
    for record in &records.records {
        let mut fixed = Vec::with_capacity(record.options.len());
        let mut terms = Vec::with_capacity(record.options.len() * n_parameters);
        let mut option_ids = Vec::with_capacity(record.options.len());
        let mut unavailable = Vec::new();
        for option in &record.options {
            if let OptionBelief::Unavailable { constraint_id } = &option.belief {
                unavailable.push(UnavailableOption {
                    option_id: option.id.clone(),
                    constraint_id: constraint_id.clone(),
                });
                continue;
            }
            option_ids.push(option.id.clone());
            fixed.push(expectation(&form.fixed, None, record, option, records.attributes.len())?);
            for (j, term) in form.terms.iter().enumerate() {
                let value = if fixed_columns.get(j).copied().flatten().is_some() {
                    0.0
                } else {
                    expectation(term, Some(j), record, option, records.attributes.len())?
                };
                terms.push(value);
            }
        }
        let chosen = option_ids.iter().position(|o| *o == record.chosen).unwrap_or(0);
        sets.push(ChoiceSetFeatures {
            record_id: record.id.clone(),
            decider: record.decider.clone(),
            option_ids,
            chosen,
            fixed,
            terms,
            unavailable,
        });
    }
    Ok(ChoiceFeatures { n_parameters, sets })
}

/// [`choice_features`] at given values of the shape parameters.
///
/// `parameters` holds one value per model parameter; only the values of the free shape
/// parameters (those the value is not affine in once the fixed parameters are substituted) are
/// read. Those values are substituted, every fixed parameter is substituted by its declared
/// value, and the features are taken as [`choice_features`] takes them: the columns of the shape
/// and fixed parameters are zero and their contribution is in the fixed part. For a value with no
/// shape parameter the result equals [`choice_features`] up to the rounding of the expectation.
///
/// # Errors
/// - [`ChoiceRecordError::Model`] for an invalid model, a point whose length is not the model's
///   dimension or a non-finite value at a shape parameter, or any
///   [`ChoiceRecordSet::validate`] refusal;
/// - [`ChoiceRecordError::AttributesDiffer`] naming the first attribute that differs;
/// - [`ChoiceRecordError::BeliefLawRequired`] for a means belief and a part not affine in the
///   attributes;
/// - [`ChoiceRecordError::TermNotFinite`] when a part's value or expectation is not finite.
pub fn choice_features_at(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    parameters: &[f64],
) -> Result<ChoiceFeatures, ChoiceRecordError> {
    let data = shape_data(model, records)?;
    let dimension = model.preferences.dimension();
    if parameters.len() != dimension {
        return Err(ChoiceRecordError::Model(PreferenceModelError::ParameterCount {
            expected: dimension,
            found: parameters.len(),
        }));
    }
    if let Some(&index) = data.split.shape.iter().find(|j| !parameters[**j].is_finite()) {
        return Err(ChoiceRecordError::Model(PreferenceModelError::NonFiniteParameterValue {
            index,
        }));
    }
    data.features_at(parameters)
}

/// One available option of a record as weighted attribute rows.
#[derive(Clone, Debug)]
pub(crate) struct OptionRows {
    /// Identity of the option.
    pub(crate) id: String,
    /// Attribute rows, one per draw (one row of means for a means belief).
    pub(crate) rows: Vec<Vec<f64>>,
    /// One weight per row, as declared (uniform when absent).
    pub(crate) weights: Vec<f64>,
    /// Sum of `weights`.
    pub(crate) total: f64,
}

impl OptionRows {
    /// Weights normalised to sum to one.
    pub(crate) fn normalised(&self) -> Vec<f64> {
        self.weights.iter().map(|w| w / self.total).collect()
    }

    /// The rows restricted to `attributes`, paired with the normalised weights and sorted: two
    /// options with equal signatures give every function of those attributes the same
    /// expectation.
    fn signature(&self, attributes: &[usize]) -> Vec<(f64, Vec<f64>)> {
        let mut signature: Vec<(f64, Vec<f64>)> = self
            .rows
            .iter()
            .zip(self.normalised())
            .map(|(row, weight)| {
                (weight, attributes.iter().map(|k| row.get(*k).copied().unwrap_or(0.0)).collect())
            })
            .collect();
        signature.sort_by(|a, b| {
            a.1.iter()
                .zip(&b.1)
                .map(|(x, y)| x.total_cmp(y))
                .find(|o| o.is_ne())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.total_cmp(&b.0))
        });
        signature
    }
}

/// The available options of one record.
#[derive(Clone, Debug)]
pub(crate) struct RecordRows {
    /// The record.
    pub(crate) record_id: String,
    /// Who chose.
    pub(crate) decider: String,
    /// The available options, in declared order.
    pub(crate) options: Vec<OptionRows>,
    /// Position of the chosen option in `options`.
    pub(crate) chosen: usize,
    /// The unavailable options, in declared order.
    pub(crate) unavailable: Vec<UnavailableOption>,
}

/// Records read under a value split into shape parameters and a part affine in the rest.
#[derive(Clone, Debug)]
pub(crate) struct ShapeData {
    /// The split of the value.
    pub(crate) split: ShapeSplit,
    /// The model's dimension.
    pub(crate) n_parameters: usize,
    /// Every record, in record order.
    pub(crate) records: Vec<RecordRows>,
}

/// The records as weighted rows under the model's shape split, with means beliefs checked
/// against parts that are not affine in the attributes.
pub(crate) fn shape_data(
    model: &ValueModel,
    records: &ChoiceRecordSet,
) -> Result<ShapeData, ChoiceRecordError> {
    model.validate().map_err(ChoiceRecordError::Model)?;
    records.validate()?;
    compare_attributes(&model.attributes, &records.attributes)?;
    let split = model.shape_split().ok_or_else(|| ChoiceRecordError::ShapeParametersNotFixed {
        parameters: model.preferences.parameters.iter().map(|p| p.id.clone()).collect(),
    })?;
    let width = records.attributes.len();
    let mut out = Vec::with_capacity(records.records.len());
    for record in &records.records {
        let mut options = Vec::with_capacity(record.options.len());
        let mut unavailable = Vec::new();
        for option in &record.options {
            let (rows, weights) = match &option.belief {
                OptionBelief::Unavailable { constraint_id } => {
                    unavailable.push(UnavailableOption {
                        option_id: option.id.clone(),
                        constraint_id: constraint_id.clone(),
                    });
                    continue;
                }
                OptionBelief::Means(means) => {
                    let parts = std::iter::once((None, &split.form.fixed))
                        .chain(split.linear.iter().map(|j| (Some(*j), &split.form.terms[*j])));
                    for (term, part) in parts {
                        if !part.is_affine_in_attributes() {
                            return Err(ChoiceRecordError::BeliefLawRequired {
                                record: record.id.clone(),
                                option: option.id.clone(),
                                term,
                            });
                        }
                    }
                    (vec![means.clone()], vec![1.0])
                }
                OptionBelief::Draws { rows, weights } => {
                    let rows: Vec<Vec<f64>> =
                        rows.chunks_exact(width).map(<[f64]>::to_vec).collect();
                    let weights = weights.clone().unwrap_or_else(|| vec![1.0; rows.len()]);
                    (rows, weights)
                }
            };
            let total: f64 = weights.iter().sum();
            options.push(OptionRows { id: option.id.clone(), rows, weights, total });
        }
        let chosen = options.iter().position(|o| o.id == record.chosen).unwrap_or(0);
        out.push(RecordRows {
            record_id: record.id.clone(),
            decider: record.decider.clone(),
            options,
            chosen,
            unavailable,
        });
    }
    Ok(ShapeData { split, n_parameters: model.preferences.dimension(), records: out })
}

impl ShapeData {
    /// The parts read by the features: `None` for the fixed part, `Some(j)` for the term of
    /// linear parameter `j`.
    fn parts(&self) -> impl Iterator<Item = (Option<usize>, &PreferenceExpr)> {
        std::iter::once((None, &self.split.form.fixed))
            .chain(self.split.linear.iter().map(|j| (Some(*j), &self.split.form.terms[*j])))
    }

    /// Expectation of one part under one option at a parameter point.
    pub(crate) fn expectation(
        &self,
        record: usize,
        option: usize,
        part: &PreferenceExpr,
        parameters: &[f64],
    ) -> Result<f64, ChoiceRecordError> {
        let set = &self.records[record];
        let rows = &set.options[option];
        let not_finite = || ChoiceRecordError::TermNotFinite {
            record: set.record_id.clone(),
            option: rows.id.clone(),
        };
        let mut weighted = 0.0;
        for (row, weight) in rows.rows.iter().zip(&rows.weights) {
            let value = part.evaluate(row, parameters).map_err(|_| not_finite())?;
            weighted += weight * value;
        }
        let value = weighted / rows.total;
        if value.is_finite() { Ok(value) } else { Err(not_finite()) }
    }

    /// Expectation of one part and of its gradient with respect to every parameter.
    pub(crate) fn expectation_with_gradient(
        &self,
        record: usize,
        option: usize,
        part: &PreferenceExpr,
        parameters: &[f64],
    ) -> Result<(f64, Vec<f64>), ChoiceRecordError> {
        let set = &self.records[record];
        let rows = &set.options[option];
        let not_finite = || ChoiceRecordError::TermNotFinite {
            record: set.record_id.clone(),
            option: rows.id.clone(),
        };
        let mut value = 0.0;
        let mut gradient = vec![0.0; parameters.len()];
        for (row, weight) in rows.rows.iter().zip(&rows.weights) {
            let (v, g) = part.evaluate_with_gradient(row, parameters).map_err(|_| not_finite())?;
            value += weight * v;
            for (slot, x) in gradient.iter_mut().zip(g) {
                *slot += weight * x;
            }
        }
        let value = value / rows.total;
        for slot in &mut gradient {
            *slot /= rows.total;
        }
        if value.is_finite() && gradient.iter().all(|g| g.is_finite()) {
            Ok((value, gradient))
        } else {
            Err(not_finite())
        }
    }

    /// The features of every record at a parameter point (see [`choice_features_at`]).
    pub(crate) fn features_at(
        &self,
        parameters: &[f64],
    ) -> Result<ChoiceFeatures, ChoiceRecordError> {
        let n_parameters = self.n_parameters;
        let mut sets = Vec::with_capacity(self.records.len());
        for (r, record) in self.records.iter().enumerate() {
            let mut fixed = Vec::with_capacity(record.options.len());
            let mut terms = vec![0.0; record.options.len() * n_parameters];
            for o in 0..record.options.len() {
                for (term, part) in self.parts() {
                    let value = self.expectation(r, o, part, parameters)?;
                    match term {
                        None => fixed.push(value),
                        Some(j) => terms[o * n_parameters + j] = value,
                    }
                }
            }
            sets.push(ChoiceSetFeatures {
                record_id: record.record_id.clone(),
                decider: record.decider.clone(),
                option_ids: record.options.iter().map(|o| o.id.clone()).collect(),
                chosen: record.chosen,
                fixed,
                terms,
                unavailable: record.unavailable.clone(),
            });
        }
        Ok(ChoiceFeatures { n_parameters, sets })
    }

    /// Whether, in some record of `members`, two available options differ in the rows of
    /// `attributes` (with their weights). When none does, every function of those attributes has
    /// the same expectation under every option of each of those records, whatever the
    /// parameters.
    pub(crate) fn varies(&self, attributes: &[usize], members: &[usize]) -> bool {
        members.iter().filter_map(|m| self.records.get(*m)).any(|record| {
            let mut signatures = record.options.iter().map(|o| o.signature(attributes));
            let Some(first) = signatures.next() else { return false };
            signatures.any(|other| {
                other.len() != first.len()
                    || other.iter().zip(&first).any(|(a, b)| {
                        a.0.total_cmp(&b.0).is_ne()
                            || a.1.iter().zip(&b.1).any(|(x, y)| x.total_cmp(y).is_ne())
                    })
            })
        })
    }

    /// Attributes read by the parts that read parameter `j`, summand by summand: a summand of
    /// the fixed part or a term that does not read `j` is left out.
    pub(crate) fn attributes_moving_with(&self, j: usize) -> Vec<usize> {
        let mut attributes = std::collections::BTreeSet::new();
        for (_, part) in self.parts() {
            for summand in summands(part) {
                if summand.parameters_used().contains(&j) {
                    attributes.extend(summand.attributes_used());
                }
            }
        }
        attributes.into_iter().collect()
    }

    /// Attributes read by the term of linear parameter `j`.
    pub(crate) fn term_attributes(&self, j: usize) -> Vec<usize> {
        self.split.form.terms.get(j).map(PreferenceExpr::attributes_used).unwrap_or_default()
    }

    /// Attributes read by the fixed part.
    pub(crate) fn fixed_attributes(&self) -> Vec<usize> {
        self.split.form.fixed.attributes_used()
    }
}

/// The summands of an expression through `+`, `-` and negation.
fn summands(expr: &PreferenceExpr) -> Vec<&PreferenceExpr> {
    match expr {
        PreferenceExpr::Add(a, b) | PreferenceExpr::Sub(a, b) => {
            let mut out = summands(a);
            out.extend(summands(b));
            out
        }
        PreferenceExpr::Neg(a) => summands(a),
        other => vec![other],
    }
}

fn compare_attributes(model: &[Attribute], records: &[Attribute]) -> Result<(), ChoiceRecordError> {
    let longest = model.len().max(records.len());
    for index in 0..longest {
        let (m, r) = (model.get(index), records.get(index));
        if m != r {
            return Err(ChoiceRecordError::AttributesDiffer {
                index,
                model: m.cloned(),
                records: r.cloned(),
            });
        }
    }
    Ok(())
}

/// The value of each parameter whose declared range is a single value, `None` for the others.
fn fixed_values(model: &ValueModel) -> Vec<Option<f64>> {
    model
        .preferences
        .parameters
        .iter()
        .map(|p| p.lower.total_cmp(&p.upper).is_eq().then_some(p.lower))
        .collect()
}

/// The linear form of the value after every fixed parameter is substituted. A parameter the
/// substituted value is still not affine in is a free shape parameter and is refused by name.
fn fixed_linear_form(model: &ValueModel) -> Result<LinearForm, ChoiceRecordError> {
    let substituted = ValueModel {
        attributes: model.attributes.clone(),
        preferences: model.preferences.clone(),
        value: substitute_parameters(&model.value, &fixed_values(model)),
        utility_units: model.utility_units.clone(),
    };
    let named = |indices: Vec<usize>| -> Vec<String> {
        indices
            .into_iter()
            .filter_map(|j| model.preferences.parameters.get(j).map(|p| p.id.clone()))
            .collect()
    };
    let free_shape = substituted.shape_parameters();
    if !free_shape.is_empty() {
        return Err(ChoiceRecordError::ShapeParametersNotFixed { parameters: named(free_shape) });
    }
    substituted
        .linear_form()
        .ok_or_else(|| ChoiceRecordError::ShapeParametersNotFixed { parameters: Vec::new() })
}

/// Replace `Parameter(j)` by `Const(values[j])` where a value is given; leave the rest.
fn substitute_parameters(expr: &PreferenceExpr, values: &[Option<f64>]) -> PreferenceExpr {
    use PreferenceExpr as E;
    let sub = |e: &PreferenceExpr| Box::new(substitute_parameters(e, values));
    match expr {
        E::Const(c) => E::Const(*c),
        E::Attribute(k) => E::Attribute(*k),
        E::Parameter(j) => match values.get(*j).copied().flatten() {
            Some(value) => E::Const(value),
            None => E::Parameter(*j),
        },
        E::Neg(a) => E::Neg(sub(a)),
        E::Add(a, b) => E::Add(sub(a), sub(b)),
        E::Sub(a, b) => E::Sub(sub(a), sub(b)),
        E::Mul(a, b) => E::Mul(sub(a), sub(b)),
        E::Min(a, b) => E::Min(sub(a), sub(b)),
        E::Max(a, b) => E::Max(sub(a), sub(b)),
        E::Exp(a) => E::Exp(sub(a)),
        E::Ln(a) => E::Ln(sub(a)),
    }
}

/// Expectation of a parameter-free part under one option's belief.
fn expectation(
    part: &PreferenceExpr,
    term: Option<usize>,
    record: &ChoiceRecord,
    option: &ChoiceOption,
    width: usize,
) -> Result<f64, ChoiceRecordError> {
    let not_finite = || ChoiceRecordError::TermNotFinite {
        record: record.id.clone(),
        option: option.id.clone(),
    };
    let at = |attributes: &[f64]| -> Result<f64, ChoiceRecordError> {
        let value = part.evaluate(attributes, &[]).map_err(|_| not_finite())?;
        if value.is_finite() { Ok(value) } else { Err(not_finite()) }
    };
    let value = match &option.belief {
        OptionBelief::Means(means) => {
            if !part.is_affine_in_attributes() {
                return Err(ChoiceRecordError::BeliefLawRequired {
                    record: record.id.clone(),
                    option: option.id.clone(),
                    term,
                });
            }
            at(means)?
        }
        OptionBelief::Draws { rows, weights } => {
            let mut weighted = 0.0;
            let mut total = 0.0;
            for (i, row) in rows.chunks_exact(width).enumerate() {
                let weight = weights.as_ref().and_then(|w| w.get(i)).copied().unwrap_or(1.0);
                weighted += weight * at(row)?;
                total += weight;
            }
            weighted / total
        }
        // Unavailable options are skipped before any expectation is taken; none is formed here.
        OptionBelief::Unavailable { .. } => return Err(not_finite()),
    };
    if value.is_finite() { Ok(value) } else { Err(not_finite()) }
}

impl ChoiceRecordError {
    /// Structured refusal for a choice-record declaration or feature-extraction failure.
    ///
    /// A value-model refusal keeps its own refusal.
    #[must_use]
    #[allow(clippy::too_many_lines, reason = "one arm per choice-record declaration failure")]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), DECLARE, self.detail())
                .remedy(remedy)
        };
        let unsatisfied = |remedy| {
            refusal(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                DECLARE,
                self.detail(),
            )
            .remedy(remedy)
        };
        let describe = |attribute: Option<&crate::preference_model::Attribute>| {
            attribute.map_or_else(|| "none".to_owned(), |a| format!("{} [{}]", a.id, a.units))
        };
        match self {
            Self::Model(error) => at_stage(error.to_refusal(), DECLARE),
            Self::AttributesMissing => {
                invalid("declare the attributes the recorded options are described by").build()
            }
            Self::AttributeIdBlank { index } => {
                invalid("give every attribute a non-blank id").offending(index.to_string()).build()
            }
            Self::AttributeUnitsBlank { attribute } => {
                invalid("state the units of every attribute").offending(attribute.clone()).build()
            }
            Self::AttributeDuplicate { attribute } => {
                invalid("give every attribute a distinct id").offending(attribute.clone()).build()
            }
            Self::PopulationBlank => {
                invalid("name the population whose choices were recorded").build()
            }
            Self::PoolingAssumptionBlank => {
                invalid("name the assumption that the deciders share one set of preferences")
                    .build()
            }
            Self::RecordsMissing => invalid("supply at least one recorded choice").build(),
            Self::RecordIdBlank { index } => {
                invalid("give every record a non-blank id").offending(index.to_string()).build()
            }
            Self::RecordDuplicate { record } => {
                invalid("give every record a distinct id").offending(record.clone()).build()
            }
            Self::DeciderBlank { record } => {
                invalid("name who made the choice").offending(record.clone()).build()
            }
            Self::OptionsTooFew { record, options } => {
                invalid("record every option that was available, at least two")
                    .offending(record.clone())
                    .expected("2")
                    .supplied(options.to_string())
                    .build()
            }
            Self::OptionIdBlank { record, index } => {
                invalid("give every option a non-blank id").offending(at(record, index)).build()
            }
            Self::OptionDuplicate { record, option } => {
                invalid("give every option a distinct id").offending(at(record, option)).build()
            }
            Self::ChosenUnknown { record, chosen } => {
                invalid("name one of the recorded options as chosen")
                    .offending(record.clone())
                    .supplied(chosen.clone())
                    .build()
            }
            Self::UnavailableConstraintBlank { record, option } => {
                invalid("name the constraint that made the option unavailable")
                    .offending(at(record, option))
                    .build()
            }
            Self::ChosenUnavailable { record, option } => {
                invalid("correct the chosen option or its availability")
                    .offending(at(record, option))
                    .build()
            }
            Self::ForecastSourceBlank { record } => {
                invalid("name where the forecast came from").offending(record.clone()).build()
            }
            Self::StatedElicitationBlank { record } => {
                invalid("name the elicitation the stated comparison came from")
                    .offending(record.clone())
                    .build()
            }
            Self::StatedAssumptionBlank { record } => invalid(
                "name the assumption that a stated comparison reflects the preferences acted on",
            )
            .offending(record.clone())
            .build(),
            Self::StatedBeliefNotExact { record, option } => {
                invalid("give every bundle of a stated comparison as exact attribute values")
                    .offending(at(record, option))
                    .build()
            }
            Self::AnalysisIdentityBlank { record } => {
                invalid("name the identity of the result the decider read")
                    .offending(record.clone())
                    .build()
            }
            Self::AnalysisDateBlank { record } => {
                invalid("name the date the decider read the result")
                    .offending(record.clone())
                    .build()
            }
            Self::RealizedAssumptionBlank { record } => {
                invalid("name the assumption that the decider expected the realized outcome")
                    .offending(record.clone())
                    .build()
            }
            Self::MeansWidthMismatch { record, option, attributes, means } => {
                invalid("supply one mean per attribute")
                    .offending(at(record, option))
                    .expected(attributes.to_string())
                    .supplied(means.to_string())
                    .build()
            }
            Self::DrawsEmpty { record, option } => {
                invalid("supply at least one draw row").offending(at(record, option)).build()
            }
            Self::DrawsWidthMismatch { record, option, attributes, values } => {
                invalid("supply one value per attribute in every draw row")
                    .offending(at(record, option))
                    .expected(format!("a multiple of {attributes}"))
                    .supplied(values.to_string())
                    .build()
            }
            Self::BeliefNotFinite { record, option } => {
                invalid("supply finite means or draws").offending(at(record, option)).build()
            }
            Self::WeightsLengthMismatch { record, option, rows, weights } => {
                invalid("supply one weight per draw row")
                    .offending(at(record, option))
                    .expected(rows.to_string())
                    .supplied(weights.to_string())
                    .build()
            }
            Self::WeightNotFinite { record, option, row } => invalid("supply finite draw weights")
                .offending(format!("{}[{row}]", at(record, option)))
                .build(),
            Self::WeightNegative { record, option, row } => {
                invalid("supply non-negative draw weights")
                    .offending(format!("{}[{row}]", at(record, option)))
                    .build()
            }
            Self::WeightsSumInvalid { record, option } => {
                invalid("supply draw weights with a positive, finite total")
                    .offending(at(record, option))
                    .build()
            }
            Self::ObservationRowsMissing { record } => {
                invalid("name the observation rows the record rests on")
                    .offending(record.clone())
                    .build()
            }
            Self::ObservationRowBlank { record, index } => {
                invalid("give every observation row a non-blank id")
                    .offending(at(record, index))
                    .build()
            }
            Self::ObservationRowRepeated { record, row } => {
                invalid("name each observation row once").offending(at(record, row)).build()
            }
            Self::ObservationRowDoubleCounted { row, first_record, second_record } => refusal(
                antecedent_core::reason_code!("evidence_double_counted"),
                DECLARE,
                self.detail(),
            )
            .offending(row.clone())
            .supplied(format!("{first_record},{second_record}"))
            .remedy("keep the observation row in one record only")
            .build(),
            Self::AttributesDiffer { index, model, records } => refusal(
                antecedent_core::reason_code!("quantity_semantics_mismatch"),
                DECLARE,
                self.detail(),
            )
            .offending(index.to_string())
            .expected(describe(model.as_ref()))
            .supplied(describe(records.as_ref()))
            .remedy("describe the recorded options in the value model's attributes and units")
            .build(),
            Self::ShapeParametersNotFixed { parameters } => refusal(
                antecedent_core::reason_code!("route_not_supported"),
                DECLARE,
                self.detail(),
            )
            .offending(parameters.join(","))
            .remedy("fix these parameters to a single value, or evaluate on the certified route")
            .build(),
            Self::BeliefLawRequired { record, option, .. } => {
                unsatisfied("record the belief as draws")
                    .offending(at(record, option))
                    .expected("draws")
                    .supplied("means")
                    .build()
            }
            Self::TermNotFinite { record, option } => {
                unsatisfied("check the recorded belief and the domain of the value")
                    .offending(at(record, option))
                    .build()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitution_replaces_only_given_parameters() {
        let expr = PreferenceExpr::Add(
            Box::new(PreferenceExpr::Mul(
                Box::new(PreferenceExpr::Parameter(0)),
                Box::new(PreferenceExpr::Attribute(0)),
            )),
            Box::new(PreferenceExpr::Exp(Box::new(PreferenceExpr::Parameter(1)))),
        );
        let got = substitute_parameters(&expr, &[None, Some(2.0)]);
        let want = PreferenceExpr::Add(
            Box::new(PreferenceExpr::Mul(
                Box::new(PreferenceExpr::Parameter(0)),
                Box::new(PreferenceExpr::Attribute(0)),
            )),
            Box::new(PreferenceExpr::Exp(Box::new(PreferenceExpr::Const(2.0)))),
        );
        assert_eq!(got, want);
    }

    #[test]
    fn attribute_comparison_names_the_first_difference() {
        let a = |id: &str| Attribute { id: id.to_owned(), units: "u".to_owned() };
        let err = compare_attributes(&[a("x"), a("y")], &[a("x")]).unwrap_err();
        assert_eq!(
            err,
            ChoiceRecordError::AttributesDiffer { index: 1, model: Some(a("y")), records: None }
        );
    }
}
