//! Descriptive association of an outcome with predictors, stated as a noncausal query.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::QueryError;
use crate::canonical_encode::{CanonicalEncode, encode_option, encode_variables};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;

/// Fewest folds a predictive model is scored over. The most is the number of
/// rows (leave-one-out), which only the data can decide.
pub const MIN_PREDICTIVE_FOLDS: u32 = 2;
/// Longest refusal reason a query retains.
pub const MAX_REFUSED_AFTER_BYTES: usize = 256;

/// How the measurement scale of the outcome is read.
///
/// The default reads the scale the schema declares for the outcome column. A column the
/// schema types as continuous that holds 0/1 indicators, or counts, is read as such only by
/// declaring it here: the selection rule never infers a scale from the values of a column the
/// schema types.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub enum AssociationOutcome {
    /// The scale the schema declares for the outcome column.
    #[default]
    Schema,
    /// A real-valued outcome.
    Continuous,
    /// A 0/1 indicator.
    Binary,
    /// A non-negative integer count.
    Count,
}

impl AssociationOutcome {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Schema => "schema",
            Self::Continuous => "continuous",
            Self::Binary => "binary",
            Self::Count => "count",
        }
    }

    /// Parse a stable name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Schema, Self::Continuous, Self::Binary, Self::Count]
            .into_iter()
            .find(|candidate| candidate.as_str() == name)
    }

    const fn tag(self) -> u64 {
        match self {
            Self::Schema => 0,
            Self::Continuous => 1,
            Self::Binary => 2,
            Self::Count => 3,
        }
    }
}

/// A symmetric measure of the association of two variables.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AssociationMeasure {
    /// Pearson product-moment correlation: a linear association, in `[-1, 1]`.
    Pearson,
    /// Distance correlation: zero exactly when the variables are independent, in `[0, 1]`.
    Distance,
}

impl AssociationMeasure {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pearson => "pearson_correlation",
            Self::Distance => "distance_correlation",
        }
    }

    /// Parse a stable name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Pearson, Self::Distance].into_iter().find(|candidate| candidate.as_str() == name)
    }

    const fn tag(self) -> u64 {
        match self {
            Self::Pearson => 0,
            Self::Distance => 1,
        }
    }
}

/// The descriptive quantity an association query reports.
///
/// None of these is the effect of an intervention. Each describes the observed joint law of the
/// columns: how the outcome varies with the predictors among the units that were observed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub enum AssociationFunctional {
    /// The declared selection rule picks the analysis from the outcome scale, the predictor
    /// scales and the data modality.
    #[default]
    Auto,
    /// The conditional mean (a real or count outcome) or conditional probability (a 0/1
    /// outcome) of the outcome in each stratum of the predictors and conditioning variables.
    Conditional,
    /// Regression coefficients of the predictors in a model of the outcome that also holds the
    /// conditioning variables: linear for a real outcome, logistic for a 0/1 outcome, Poisson
    /// for a count.
    Coefficients,
    /// A symmetric measure of association of the outcome with the one predictor.
    Measure(AssociationMeasure),
    /// A predictive model of the outcome on the predictors and conditioning variables, scored on
    /// held-out folds.
    Predictive {
        /// Held-out folds (blocks of consecutive time for a series).
        folds: u32,
    },
}

impl AssociationFunctional {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Conditional => "conditional",
            Self::Coefficients => "coefficients",
            Self::Measure(measure) => measure.as_str(),
            Self::Predictive { .. } => "predictive",
        }
    }
}

impl CanonicalEncode for AssociationFunctional {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Auto => {
                writer.word(0);
            }
            Self::Conditional => {
                writer.word(1);
            }
            Self::Coefficients => {
                writer.word(2);
            }
            Self::Measure(measure) => {
                writer.word(3).word(measure.tag());
            }
            Self::Predictive { folds } => {
                writer.word(4).word(u64::from(*folds));
            }
        }
    }
}

/// A noncausal association in the observed data: the outcome against predictors, optionally
/// holding conditioning variables fixed.
///
/// The columns are table columns (rows of a table, or the time points of a series). The query
/// states no intervention and no graph, and its answer is never an effect: it describes how the
/// outcome varies with the predictors among the units observed, with whatever confounding,
/// selection and reverse direction that carries. A study that asks it after a causal query was
/// refused records the refusal's reason code in `refused_after`, so the descriptive answer
/// stays attached to the reason the causal one could not be given.
#[derive(Clone, Debug, PartialEq)]
pub struct AssociationQuery {
    /// Outcome column.
    pub outcome: VariableId,
    /// Predictor columns, in the order their coefficients and strata are reported.
    pub predictors: Arc<[VariableId]>,
    /// Conditioning columns: held fixed in a regression, crossed into the strata of a
    /// conditional law.
    pub conditioning: Arc<[VariableId]>,
    /// What the answer describes.
    pub functional: AssociationFunctional,
    /// How the outcome scale is read.
    pub outcome_scale: AssociationOutcome,
    /// Sampling-weight column; `None` weighs every row `1`.
    pub weight: Option<VariableId>,
    /// Reason code of the refused causal query this one follows; `None` when it follows none.
    pub refused_after: Option<Arc<str>>,
}

/// Every field is written; predictors and conditioning keep their order.
impl CanonicalEncode for AssociationQuery {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.outcome.encode(writer);
        encode_variables(writer, &self.predictors);
        encode_variables(writer, &self.conditioning);
        self.functional.encode(writer);
        writer.word(self.outcome_scale.tag());
        encode_option(writer, self.weight.as_ref());
        writer.opt_text(self.refused_after.as_deref());
    }
}

impl AssociationQuery {
    /// An association of `outcome` with `predictors`, picked by the selection rule.
    #[must_use]
    pub fn new(outcome: VariableId, predictors: impl Into<Arc<[VariableId]>>) -> Self {
        Self {
            outcome,
            predictors: predictors.into(),
            conditioning: Arc::from([]),
            functional: AssociationFunctional::Auto,
            outcome_scale: AssociationOutcome::Schema,
            weight: None,
            refused_after: None,
        }
    }

    /// Hold `conditioning` fixed.
    #[must_use]
    pub fn with_conditioning(mut self, conditioning: impl Into<Arc<[VariableId]>>) -> Self {
        self.conditioning = conditioning.into();
        self
    }

    /// Report `functional` instead of letting the selection rule pick.
    #[must_use]
    pub fn with_functional(mut self, functional: AssociationFunctional) -> Self {
        self.functional = functional;
        self
    }

    /// Read the outcome on a declared scale instead of the schema's.
    #[must_use]
    pub fn with_outcome_scale(mut self, scale: AssociationOutcome) -> Self {
        self.outcome_scale = scale;
        self
    }

    /// Weigh the rows by a sampling-weight column.
    #[must_use]
    pub fn with_weight(mut self, weight: VariableId) -> Self {
        self.weight = Some(weight);
        self
    }

    /// Record the reason code of the refused causal query this one follows.
    #[must_use]
    pub fn following_refusal(mut self, reason: impl Into<Arc<str>>) -> Self {
        self.refused_after = Some(reason.into());
        self
    }

    /// Every column the query reads, outcome first.
    #[must_use]
    pub fn columns(&self) -> Vec<VariableId> {
        let mut columns = vec![self.outcome];
        columns.extend(self.predictors.iter().copied());
        columns.extend(self.conditioning.iter().copied());
        columns.extend(self.weight);
        columns
    }

    /// Validate the declared roles before compiling a study.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidAssociation`] when a column repeats across roles, the functional
    /// does not fit the roles (a measure takes one predictor and no conditioning, coefficients
    /// and a predictive model take a predictor, fewer than two folds), or the recorded
    /// refusal reason is empty or too long.
    pub fn validate(&self) -> Result<(), QueryError> {
        let invalid = |message: &str| Err(QueryError::InvalidAssociation(message.to_owned()));
        let columns = self.columns();
        let mut sorted = columns.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != columns.len() {
            return invalid(
                "declare distinct outcome, predictor, conditioning and weight columns; a column \
                 cannot take two roles",
            );
        }
        match self.functional {
            AssociationFunctional::Auto | AssociationFunctional::Conditional => {}
            AssociationFunctional::Coefficients => {
                if self.predictors.is_empty() {
                    return invalid("regression coefficients need at least one predictor");
                }
            }
            AssociationFunctional::Measure(_) => {
                if self.predictors.len() != 1 || !self.conditioning.is_empty() {
                    return invalid(
                        "an association measure relates the outcome to exactly one predictor and \
                         holds nothing fixed; use coefficients to adjust for conditioning columns",
                    );
                }
            }
            AssociationFunctional::Predictive { folds } => {
                if self.predictors.is_empty() {
                    return invalid("a predictive model needs at least one predictor");
                }
                if folds < MIN_PREDICTIVE_FOLDS {
                    return invalid("a predictive model is scored over at least 2 folds");
                }
            }
        }
        if let Some(reason) = &self.refused_after {
            if reason.is_empty() || reason.len() > MAX_REFUSED_AFTER_BYTES {
                return invalid("the recorded refusal reason must be 1 to 256 bytes");
            }
        }
        Ok(())
    }
}
