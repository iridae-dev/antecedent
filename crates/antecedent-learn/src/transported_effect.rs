//! A retained effect model applied to a cohort from a different population.
//!
//! A [`FittedEffect`] predicts a conditional treatment contrast `tau(x)` and retains the
//! box of feature values it was fitted on ([`EffectSupport`]). Applying it to a cohort
//! drawn from another population is a transport claim, and this module issues the weighted
//! cohort mean `sum_i w_i tau(x_i)` only when its two premises are met:
//!
//! 1. A [`SelectionShiftDeclaration`] states the selection-diagram premise: the selection
//!    node that separates the two populations points only into the declared effect
//!    modifiers, and those modifiers are exactly the model's features. Given them, the
//!    conditional contrast is the same function in both populations. The declaration is a
//!    premise and is not tested here.
//! 2. An [`OverlapAssessment`] of the target rows against the model's support contract
//!    finds no feature disjoint from the fitted box and no row outside it. A box is a
//!    necessary condition for positivity, not a density or overlap guarantee; the issued
//!    value says so and carries no weight-based claim.
//!
//! The value is point-only. It is the expectation of one cohort-mean contrast under the
//! declared premise, never target-population data, never two absolute potential-outcome
//! means and never a law. No standard error, interval or utility variance is inherited
//! from the model's own fit, and an interval or law request is refused with the evidence
//! that would license it ([`TransportedRequest`]).
//!
//! # Refresh with outcome-only data
//!
//! Nothing here refreshes in place: the value is a pure function of the retained model,
//! the target cohort and the declaration. New outcome-only data in the model's own
//! population means refitting the model, which is a new retained identity and a new value
//! (recompute). The declaration and cohort are kept only by being passed again. Outcome
//! data from the target population cannot move the issued value, because the cohort
//! carries covariates and weights, not outcomes; it is refused as a different route
//! (a natively fitted target model), never blended with the transported value.
//!
//! Every refusal names the unmet premise: a missing declaration
//! (`transport_missing_evidence`), a declaration that does not match the model, the
//! populations or the selection structure (`transport_not_certified`), no overlap or rows
//! outside the support contract (`transport_support_failure`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::{CanonicalWriter, ExecutionContext, VariableId};

use crate::fitted_effect::{EffectSupport, FittedEffect};
use crate::portable::PredictionMap;

/// Statement of the assumption a selection-shift declaration makes.
pub const SELECTION_SHIFT_ASSUMPTION: &str = "selection_node_enters_only_declared_effect_modifiers";
/// Largest allowed deviation of the cohort weight sum from one.
pub const TRANSPORT_WEIGHT_SUM_TOLERANCE: f64 = 1e-9;
/// Stored statement of what the issued value means.
pub const TRANSPORTED_POINT_SCOPE: &str = "weighted mean of the retained model's conditional treatment-contrast predictions over the declared target-population rows, under a declared selection-shift premise: an affine expectation of one cohort-mean contrast; point-only, no interval, no inherited standard error";

const IDENTITY_TAG: &str = "antecedent.transported_effect.v1";
const COHORT_TAG: &str = "antecedent.transported_effect.cohort.v1";

/// A typed refusal. `detail` is a stable `transported_effect.<snake_case>` literal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Stable refusal detail.
    pub detail: &'static str,
    /// The premise, row, feature or identity the refusal is about.
    pub subject: String,
    /// The declaration or evidence that would license the request.
    pub remedy: &'static str,
}

impl TransportRefusal {
    fn new(
        code: &'static str,
        detail: &'static str,
        subject: impl Into<String>,
        remedy: &'static str,
    ) -> Self {
        Self { code, detail, subject: subject.into(), remedy }
    }
}

impl std::fmt::Display for TransportRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}: {} ({})",
            antecedent_core::reason_code::PREFIX,
            self.code,
            self.detail,
            self.subject
        )
    }
}

impl std::error::Error for TransportRefusal {}

fn invalid(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> TransportRefusal {
    TransportRefusal::new(
        antecedent_core::reason_code!("invalid_argument"),
        detail,
        subject,
        remedy,
    )
}

fn schema_mismatch(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> TransportRefusal {
    TransportRefusal::new(antecedent_core::reason_code!("schema_mismatch"), detail, subject, remedy)
}

fn not_certified(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> TransportRefusal {
    TransportRefusal::new(
        antecedent_core::reason_code!("transport_not_certified"),
        detail,
        subject,
        remedy,
    )
}

fn support_failure(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> TransportRefusal {
    TransportRefusal::new(
        antecedent_core::reason_code!("transport_support_failure"),
        detail,
        subject,
        remedy,
    )
}

fn unlicensed(
    detail: &'static str,
    subject: impl Into<String>,
    remedy: &'static str,
) -> TransportRefusal {
    TransportRefusal::new(
        antecedent_core::reason_code!("cell_not_licensed"),
        detail,
        subject,
        remedy,
    )
}

/// A request the transported cohort mean cannot answer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransportedRequest {
    /// An interval, band or posterior for the transported cohort mean.
    Interval,
    /// A probability, quantile, nonlinear utility or joint law of the contrast.
    Law,
}

impl TransportedRequest {
    /// The typed refusal naming the evidence that would license the request.
    #[must_use]
    pub fn refusal(self) -> TransportRefusal {
        match self {
            Self::Interval => unlicensed(
                "transported_effect.interval_not_licensed",
                "interval",
                "none available: the transported value is point-only under a declared premise; \
                 an interval needs measured whole-method coverage that includes the transport \
                 step and the support check",
            ),
            Self::Law => TransportRefusal::new(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                "transported_effect.law_not_supplied",
                "law",
                "a transported mean of conditional contrasts is one expectation; supply aligned \
                 outcome draws under a declared transport model for a probability, quantile or \
                 nonlinear utility",
            ),
        }
    }
}

/// The selection-diagram premise of a covariate shift between two populations.
///
/// The selection node `S` marks where the populations differ. The declaration states which
/// variables `S` points into (`selection_children`) and which effect modifiers are
/// sufficient (`effect_modifiers`): given exactly those, the conditional contrast does not
/// depend on `S`. It is valid only if `S` points into nothing outside the modifiers, so it
/// reaches neither the outcome mechanism nor the treatment effect itself.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectionShiftDeclaration {
    /// Population the model was fitted for.
    pub source_population: String,
    /// Population the cohort is drawn from.
    pub target_population: String,
    /// Name of the selection node.
    pub selection_node: String,
    /// Variables the selection node points into, sorted and unique.
    pub selection_children: Vec<String>,
    /// Effect modifiers declared sufficient, sorted and unique.
    pub effect_modifiers: Vec<String>,
}

impl SelectionShiftDeclaration {
    /// Declare a selection shift. Name lists are sorted and deduplicated.
    #[must_use]
    pub fn new(
        source_population: &str,
        target_population: &str,
        selection_node: &str,
        selection_children: &[&str],
        effect_modifiers: &[&str],
    ) -> Self {
        let canonical = |names: &[&str]| -> Vec<String> {
            let set: BTreeSet<String> = names.iter().map(|n| (*n).to_owned()).collect();
            set.into_iter().collect()
        };
        Self {
            source_population: source_population.to_owned(),
            target_population: target_population.to_owned(),
            selection_node: selection_node.to_owned(),
            selection_children: canonical(selection_children),
            effect_modifiers: canonical(effect_modifiers),
        }
    }

    fn write_canonical(&self, canon: &mut CanonicalWriter) {
        canon
            .text(SELECTION_SHIFT_ASSUMPTION)
            .text(&self.source_population)
            .text(&self.target_population)
            .text(&self.selection_node)
            .texts(&self.selection_children)
            .texts(&self.effect_modifiers);
    }
}

/// Target-population rows: named finite numeric feature columns with row identities and
/// normalized weights.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetCohort {
    /// Population the rows are declared to come from.
    pub population_id: String,
    /// Row identities, unique and nonblank.
    pub row_ids: Vec<String>,
    /// Feature names, aligned with `columns`.
    pub features: Vec<String>,
    /// One finite column per feature.
    pub columns: Vec<Vec<f64>>,
    /// Weights, nonnegative and summing to one within [`TRANSPORT_WEIGHT_SUM_TOLERANCE`].
    pub weights: Vec<f64>,
}

impl TargetCohort {
    /// Neumaier-compensated total, nearly independent of row order.
    fn total(values: impl Iterator<Item = f64>) -> f64 {
        let mut sum = 0.0_f64;
        let mut compensation = 0.0_f64;
        for term in values {
            let next = sum + term;
            if sum.abs() >= term.abs() {
                compensation += (sum - next) + term;
            } else {
                compensation += (term - next) + sum;
            }
            sum = next;
        }
        sum + compensation
    }

    /// Check shape, finiteness, identities and weights.
    ///
    /// # Errors
    /// A blank population, an empty cohort, blank or duplicate row identities or feature
    /// names, ragged or nonfinite columns, and weights that are negative, nonfinite or do
    /// not sum to one.
    pub fn validate(&self) -> Result<(), TransportRefusal> {
        let remedy =
            "declare a target cohort with named finite feature rows and normalized weights";
        let rows = self.weights.len();
        if self.population_id.trim().is_empty() {
            return Err(invalid("transported_effect.cohort_invalid", "population", remedy));
        }
        if rows == 0 {
            return Err(invalid("transported_effect.cohort_invalid", "rows", remedy));
        }
        if self.features.is_empty()
            || self.features.iter().any(|f| f.trim().is_empty())
            || self.features.iter().collect::<BTreeSet<_>>().len() != self.features.len()
            || self.columns.len() != self.features.len()
        {
            return Err(invalid("transported_effect.cohort_invalid", "features", remedy));
        }
        if self.row_ids.len() != rows
            || self.row_ids.iter().any(|id| id.trim().is_empty())
            || self.row_ids.iter().collect::<BTreeSet<_>>().len() != rows
        {
            return Err(invalid("transported_effect.cohort_invalid", "row_ids", remedy));
        }
        for (name, column) in self.features.iter().zip(&self.columns) {
            if column.len() != rows || column.iter().any(|v| !v.is_finite()) {
                return Err(invalid("transported_effect.cohort_invalid", name.clone(), remedy));
            }
        }
        if self.weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(invalid("transported_effect.weights_invalid", "weights", remedy));
        }
        let total = Self::total(self.weights.iter().copied());
        if (total - 1.0).abs() > TRANSPORT_WEIGHT_SUM_TOLERANCE {
            return Err(invalid(
                "transported_effect.weights_not_normalized",
                format!("sum={total}"),
                "normalize the weights to sum to one",
            ));
        }
        Ok(())
    }

    fn digest(&self) -> String {
        let mut canon = CanonicalWriter::new(COHORT_TAG);
        canon.text(&self.population_id).texts(&self.row_ids).texts(&self.features);
        canon.word(self.columns.len() as u64);
        for column in &self.columns {
            canon.reals(column);
        }
        canon.reals(&self.weights);
        canon.finish_hex()
    }
}

/// Where one feature of the target rows falls against the fitted range.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureOverlap {
    /// Feature name.
    pub feature: String,
    /// Fitted range.
    pub source_lo: f64,
    /// Fitted range.
    pub source_hi: f64,
    /// Smallest target value.
    pub target_lo: f64,
    /// Largest target value.
    pub target_hi: f64,
    /// Target rows outside the fitted range of this feature.
    pub rows_outside: usize,
    /// Target weight outside the fitted range of this feature.
    pub mass_outside: f64,
    /// Whether the target range and the fitted range do not intersect.
    pub disjoint: bool,
}

/// The support-overlap check of target rows against a model's support contract.
#[derive(Clone, Debug, PartialEq)]
pub struct OverlapAssessment {
    /// Rows the model was fitted on.
    pub source_rows: u64,
    /// Target rows checked.
    pub rows_checked: usize,
    /// Rows with at least one feature outside its fitted range.
    pub rows_outside: usize,
    /// Target weight on those rows.
    pub mass_outside: f64,
    /// First such row, in input order.
    pub first_row: Option<usize>,
    /// Index of the first violated feature of that row.
    pub first_feature: Option<usize>,
    /// Per-feature detail, in model feature order.
    pub features: Vec<FeatureOverlap>,
}

impl OverlapAssessment {
    /// Whether every feature range intersects the fitted range.
    #[must_use]
    pub fn any_disjoint(&self) -> bool {
        self.features.iter().any(|f| f.disjoint)
    }

    /// Whether every target row lies inside the fitted box.
    #[must_use]
    pub fn fully_covered(&self) -> bool {
        self.rows_outside == 0
    }
}

/// Assess target columns (ordered as the model's features) against `support`.
///
/// # Errors
/// A column count or length that does not match the support contract.
pub fn assess_overlap(
    support: &EffectSupport,
    feature_names: &[String],
    columns: &[&[f64]],
    weights: &[f64],
) -> Result<OverlapAssessment, TransportRefusal> {
    let rows = weights.len();
    if columns.len() != support.ranges.len()
        || feature_names.len() != columns.len()
        || columns.iter().any(|c| c.len() != rows)
    {
        return Err(schema_mismatch(
            "transported_effect.feature_schema_mismatch",
            "support",
            "supply exactly the model's named features, one finite column each",
        ));
    }
    let mut features = Vec::with_capacity(columns.len());
    for ((name, column), range) in feature_names.iter().zip(columns).zip(&support.ranges) {
        let target_lo = column.iter().copied().fold(f64::INFINITY, f64::min);
        let target_hi = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let outside: Vec<usize> =
            (0..rows).filter(|&r| column[r] < range.lo || column[r] > range.hi).collect();
        features.push(FeatureOverlap {
            feature: name.clone(),
            source_lo: range.lo,
            source_hi: range.hi,
            target_lo,
            target_hi,
            rows_outside: outside.len(),
            mass_outside: TargetCohort::total(outside.iter().map(|&r| weights[r])),
            disjoint: target_lo > range.hi || target_hi < range.lo,
        });
    }
    let mut rows_outside = 0_usize;
    let mut first_row = None;
    let mut first_feature = None;
    let mut mass = Vec::new();
    for row in 0..rows {
        let violated = columns
            .iter()
            .zip(&support.ranges)
            .position(|(column, range)| column[row] < range.lo || column[row] > range.hi);
        if let Some(feature) = violated {
            rows_outside += 1;
            mass.push(weights[row]);
            if first_row.is_none() {
                first_row = Some(row);
                first_feature = Some(feature);
            }
        }
    }
    Ok(OverlapAssessment {
        source_rows: support.rows,
        rows_checked: rows,
        rows_outside,
        mass_outside: TargetCohort::total(mass.into_iter()),
        first_row,
        first_feature,
        features,
    })
}

/// What the issued value is: always point-only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransportedInference {
    /// A point and nothing else.
    PointOnly,
}

/// A transported weighted-mean contrast with the premise and overlap check that licensed it.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedCohortEffect {
    /// Weighted mean of the conditional contrast over the target rows.
    pub mean_contrast: f64,
    /// Conditional contrast at each target row, in row order.
    pub row_contrasts: Vec<f64>,
    /// The overlap check that passed.
    pub overlap: OverlapAssessment,
    /// The declaration that licensed the transport.
    pub declaration: SelectionShiftDeclaration,
    /// Digest of the target cohort.
    pub cohort_digest: String,
    /// Identity of the retained model supplied by the caller.
    pub model_identity: String,
    identity: String,
}

impl TransportedCohortEffect {
    /// Identity binding model, declaration, cohort, overlap and value.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Always [`TransportedInference::PointOnly`].
    #[must_use]
    pub fn inference(&self) -> TransportedInference {
        TransportedInference::PointOnly
    }

    /// The refusal for an interval, band or posterior request.
    #[must_use]
    pub fn interval_refusal(&self) -> TransportRefusal {
        TransportedRequest::Interval.refusal()
    }

    /// The refusal for a law-valued request.
    #[must_use]
    pub fn law_refusal(&self) -> TransportRefusal {
        TransportedRequest::Law.refusal()
    }

    /// Assumptions that travel with the value.
    #[must_use]
    pub fn assumptions(&self) -> Vec<String> {
        vec![
            SELECTION_SHIFT_ASSUMPTION.to_owned(),
            format!(
                "selection node {} enters only {}",
                self.declaration.selection_node,
                self.declaration.selection_children.join(",")
            ),
            "overlap is a fitted-box necessary condition, not a density or positivity guarantee"
                .to_owned(),
        ]
    }
}

fn check_declaration(
    model_population: &str,
    model_features: &[String],
    cohort_population: &str,
    shift: Option<&SelectionShiftDeclaration>,
) -> Result<SelectionShiftDeclaration, TransportRefusal> {
    if model_population == cohort_population {
        return Err(TransportRefusal::new(
            antecedent_core::reason_code!("quantity_semantics_mismatch"),
            "transported_effect.population_not_different",
            cohort_population.to_owned(),
            "the cohort is declared to be the model's population: decide it as a \
             within-population cohort, with no shift declaration",
        ));
    }
    let Some(declared) = shift else {
        return Err(TransportRefusal::new(
            antecedent_core::reason_code!("transport_missing_evidence"),
            "transported_effect.shift_declaration_required",
            format!("model={model_population} cohort={cohort_population}"),
            "declare the selection shift: source and target population, the selection node, \
             the variables it points into and the effect modifiers (the model's features) \
             given which the conditional contrast is invariant",
        ));
    };
    if declared.selection_node.trim().is_empty()
        || declared.source_population.trim().is_empty()
        || declared.target_population.trim().is_empty()
        || declared.selection_children.is_empty()
    {
        return Err(invalid(
            "transported_effect.declaration_invalid",
            "selection_node_or_children",
            "name the populations, the selection node and the variables it points into",
        ));
    }
    if declared.source_population != model_population
        || declared.target_population != cohort_population
    {
        return Err(not_certified(
            "transported_effect.shift_declaration_mismatch",
            format!("model={model_population} cohort={cohort_population}"),
            "the declaration must name the model's population as source and the cohort's \
             population as target",
        ));
    }
    let modifiers: BTreeSet<&String> = declared.effect_modifiers.iter().collect();
    if let Some(escaping) = declared.selection_children.iter().find(|c| !modifiers.contains(c)) {
        return Err(not_certified(
            "transported_effect.selection_enters_mechanism",
            escaping.clone(),
            "the selection node may point only into declared effect modifiers; a selection \
             arrow into the outcome or any other mechanism breaks the invariance premise",
        ));
    }
    let mut features = model_features.to_vec();
    features.sort();
    features.dedup();
    if declared.effect_modifiers != features {
        return Err(not_certified(
            "transported_effect.modifier_set_mismatch",
            format!(
                "declared={} model={}",
                declared.effect_modifiers.join(","),
                features.join(",")
            ),
            "the declared effect modifiers must be exactly the model's features",
        ));
    }
    Ok(declared.clone())
}

/// Apply `effect` to a cohort from a different population under a declared shift.
///
/// Order of refusal: cohort shape, the model's support contract and identity link, strict
/// feature schema, the shift declaration, overlap (disjoint feature, then rows outside),
/// prediction. `feature_names` are the model's feature names aligned with
/// [`FittedEffect::features`]; `model_identity` is the retained model's identity and is
/// bound into the result.
///
/// # Errors
/// A [`TransportRefusal`] naming the unmet premise; no value is issued.
pub fn transport_cohort_effect(
    effect: &FittedEffect,
    feature_names: &[String],
    model_identity: &str,
    model_population: &str,
    cohort: &TargetCohort,
    shift: Option<&SelectionShiftDeclaration>,
    ctx: &ExecutionContext,
) -> Result<TransportedCohortEffect, TransportRefusal> {
    cohort.validate()?;
    if model_identity.trim().is_empty() || model_population.trim().is_empty() {
        return Err(invalid(
            "transported_effect.model_invalid",
            "identity_or_population",
            "name the retained model and the population it was fitted for",
        ));
    }
    effect.validate().map_err(|_| {
        invalid(
            "transported_effect.model_invalid",
            "fitted_effect",
            "refit; the retained payload does not validate",
        )
    })?;
    if feature_names.len() != effect.features.len()
        || feature_names.iter().any(|f| f.trim().is_empty())
    {
        return Err(invalid(
            "transported_effect.model_invalid",
            "feature_names",
            "name every retained feature",
        ));
    }
    {
        let distinct: BTreeSet<&String> = feature_names.iter().collect();
        if distinct.len() != feature_names.len() {
            return Err(invalid(
                "transported_effect.model_invalid",
                "duplicate_feature_names",
                "name each retained feature once; a repeated name would feed one cohort column to two model features",
            ));
        }
    }
    let identity_link = match &effect.predictor.model {
        PredictionMap::Linear { logistic, .. } => !logistic,
        PredictionMap::Trees { logistic, probability, .. } => !logistic && !probability,
    };
    if !identity_link {
        return Err(TransportRefusal::new(
            antecedent_core::reason_code!("quantity_semantics_mismatch"),
            "transported_effect.link_not_identity",
            "predictor",
            "a probability-scale predictor is not a treatment contrast; the cohort mean needs \
             the identity-scale CATE map",
        ));
    }
    let Some(support) = &effect.support else {
        return Err(unlicensed(
            "transported_effect.support_contract_absent",
            "fitted_effect.support",
            "refit with a producer that retains the fitted-row support contract; overlap \
             cannot be checked without it",
        ));
    };
    let expected: BTreeSet<&String> = feature_names.iter().collect();
    let supplied: BTreeSet<&String> = cohort.features.iter().collect();
    if expected != supplied {
        let difference: Vec<String> =
            expected.symmetric_difference(&supplied).map(|s| (*s).clone()).collect();
        return Err(schema_mismatch(
            "transported_effect.feature_schema_mismatch",
            difference.join(","),
            "supply exactly the model's named features: none missing and none extra",
        ));
    }
    let declaration =
        check_declaration(model_population, feature_names, &cohort.population_id, shift)?;
    let ordered: Vec<&[f64]> = feature_names
        .iter()
        .map(|name| {
            let index = cohort.features.iter().position(|f| f == name).unwrap_or(0);
            cohort.columns[index].as_slice()
        })
        .collect();
    let overlap = assess_overlap(support, feature_names, &ordered, &cohort.weights)?;
    if let Some(feature) = overlap.features.iter().find(|f| f.disjoint) {
        return Err(support_failure(
            "transported_effect.no_overlap",
            format!(
                "feature={} target=[{},{}] fitted=[{},{}]",
                feature.feature,
                feature.target_lo,
                feature.target_hi,
                feature.source_lo,
                feature.source_hi
            ),
            "the target rows do not intersect the fitted support on this feature: no \
             transport is licensed; supply a model fitted on rows that cover the target",
        ));
    }
    if let (Some(row), Some(feature)) = (overlap.first_row, overlap.first_feature) {
        return Err(support_failure(
            "transported_effect.out_of_support",
            format!(
                "row={} feature={} rows_outside={}",
                cohort.row_ids[row], feature_names[feature], overlap.rows_outside
            ),
            "restrict the cohort to rows inside the fitted support (a restricted cohort is a \
             new, separately declared cohort), or supply a model fitted on rows that cover \
             the target population",
        ));
    }
    let ids: Vec<VariableId> = effect.features.iter().map(|v| VariableId::from_raw(*v)).collect();
    let row_contrasts =
        effect.predict(&ids, &ordered, cohort.weights.len(), ctx).map_err(|error| {
            unlicensed(
                "transported_effect.prediction_refused",
                error.to_string(),
                "the retained model could not predict at these rows; check the cohort features",
            )
        })?;
    let mean_contrast =
        TargetCohort::total(row_contrasts.iter().zip(&cohort.weights).map(|(v, w)| v * w));
    if !mean_contrast.is_finite() {
        return Err(unlicensed(
            "transported_effect.non_finite_result",
            "mean_contrast",
            "the cohort mean is not finite; check the model and the cohort",
        ));
    }
    let cohort_digest = cohort.digest();
    let mut canon = CanonicalWriter::new(IDENTITY_TAG);
    canon.text(model_identity).text(model_population).text(&cohort_digest);
    declaration.write_canonical(&mut canon);
    canon.word(overlap.source_rows).word(overlap.rows_checked as u64);
    canon.word(overlap.rows_outside as u64).real(mean_contrast).reals(&row_contrasts);
    let identity = canon.finish_hex();
    Ok(TransportedCohortEffect {
        mean_contrast,
        row_contrasts,
        overlap,
        declaration,
        cohort_digest,
        model_identity: model_identity.to_owned(),
        identity,
    })
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use crate::fitted_effect::FeatureRange;
    use crate::learner::LearnerProvenance;
    use crate::portable::PortablePredictor;

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(5)
    }

    /// tau(z, w) = 1 + 2 z - w, fitted on z in [0, 1], w in [0, 2].
    fn model(support: bool) -> FittedEffect {
        FittedEffect {
            version: 1,
            features: vec![3, 4],
            intercept: true,
            predictor: PortablePredictor {
                version: 1,
                columns: 3,
                provenance: LearnerProvenance {
                    spec: "linear".into(),
                    implementation: "test".into(),
                    version: "1".into(),
                },
                model: PredictionMap::Linear {
                    coefficients: vec![1.0, 2.0, -1.0],
                    logistic: false,
                },
            },
            support: support.then(|| EffectSupport {
                version: 1,
                rows: 400,
                ranges: vec![FeatureRange { lo: 0.0, hi: 1.0 }, FeatureRange { lo: 0.0, hi: 2.0 }],
            }),
        }
    }

    fn names() -> Vec<String> {
        vec!["z".into(), "w".into()]
    }

    fn cohort(z: Vec<f64>, w: Vec<f64>) -> TargetCohort {
        let n = z.len();
        TargetCohort {
            population_id: "clinic_b".into(),
            row_ids: (0..n).map(|i| format!("r{i}")).collect(),
            features: vec!["w".into(), "z".into()],
            columns: vec![w, z],
            weights: vec![1.0 / n as f64; n],
        }
    }

    fn declaration() -> SelectionShiftDeclaration {
        SelectionShiftDeclaration::new("clinic_a", "clinic_b", "S", &["z"], &["z", "w"])
    }

    fn run(
        effect: &FittedEffect,
        cohort: &TargetCohort,
        shift: Option<&SelectionShiftDeclaration>,
    ) -> Result<TransportedCohortEffect, TransportRefusal> {
        transport_cohort_effect(effect, &names(), "model-1", "clinic_a", cohort, shift, &ctx())
    }

    #[test]
    fn weighted_mean_matches_the_hand_calculation() {
        // Rows (z, w) = (0.5, 1), (1, 0), (0, 2): tau = 1, 3, -1; mean = 1.
        let mut c = cohort(vec![0.5, 1.0, 0.0], vec![1.0, 0.0, 2.0]);
        c.weights = vec![0.5, 0.25, 0.25];
        let value = run(&model(true), &c, Some(&declaration())).unwrap();
        assert_eq!(value.row_contrasts, vec![1.0, 3.0, -1.0]);
        assert!((value.mean_contrast - 1.0).abs() < 1e-12);
        assert_eq!(value.inference(), TransportedInference::PointOnly);
        assert!(value.overlap.fully_covered());
    }

    #[test]
    fn missing_declaration_names_the_premise() {
        let c = cohort(vec![0.5], vec![1.0]);
        let refusal = run(&model(true), &c, None).unwrap_err();
        assert_eq!(refusal.code, "transport_missing_evidence");
        assert_eq!(refusal.detail, "transported_effect.shift_declaration_required");
    }

    #[test]
    fn selection_into_the_mechanism_is_not_certified() {
        let c = cohort(vec![0.5], vec![1.0]);
        let bad =
            SelectionShiftDeclaration::new("clinic_a", "clinic_b", "S", &["z", "y"], &["z", "w"]);
        let refusal = run(&model(true), &c, Some(&bad)).unwrap_err();
        assert_eq!(refusal.code, "transport_not_certified");
        assert_eq!(refusal.detail, "transported_effect.selection_enters_mechanism");
        assert_eq!(refusal.subject, "y");
    }

    #[test]
    fn modifier_set_must_be_the_models_features() {
        let c = cohort(vec![0.5], vec![1.0]);
        let bad = SelectionShiftDeclaration::new("clinic_a", "clinic_b", "S", &["z"], &["z"]);
        let refusal = run(&model(true), &c, Some(&bad)).unwrap_err();
        assert_eq!(refusal.detail, "transported_effect.modifier_set_mismatch");
    }

    #[test]
    fn declaration_for_other_populations_is_a_mismatch() {
        let c = cohort(vec![0.5], vec![1.0]);
        let bad = SelectionShiftDeclaration::new("clinic_a", "clinic_c", "S", &["z"], &["z", "w"]);
        let refusal = run(&model(true), &c, Some(&bad)).unwrap_err();
        assert_eq!(refusal.detail, "transported_effect.shift_declaration_mismatch");
    }

    #[test]
    fn same_population_is_not_a_transport() {
        let mut c = cohort(vec![0.5], vec![1.0]);
        c.population_id = "clinic_a".into();
        let refusal = run(&model(true), &c, Some(&declaration())).unwrap_err();
        assert_eq!(refusal.detail, "transported_effect.population_not_different");
    }

    #[test]
    fn disjoint_feature_range_is_no_overlap() {
        let c = cohort(vec![2.0, 3.0], vec![1.0, 1.0]);
        let refusal = run(&model(true), &c, Some(&declaration())).unwrap_err();
        assert_eq!(refusal.code, "transport_support_failure");
        assert_eq!(refusal.detail, "transported_effect.no_overlap");
        assert!(refusal.subject.starts_with("feature=z"));
    }

    #[test]
    fn partial_overlap_names_the_first_row_outside() {
        let c = cohort(vec![0.5, 1.5], vec![1.0, 1.0]);
        let refusal = run(&model(true), &c, Some(&declaration())).unwrap_err();
        assert_eq!(refusal.detail, "transported_effect.out_of_support");
        assert_eq!(refusal.subject, "row=r1 feature=z rows_outside=1");
    }

    #[test]
    fn overlap_assessment_reports_mass_and_ranges() {
        let model = model(true);
        let support = model.support.as_ref().unwrap();
        let z = [0.5, 1.5, 0.2, 0.9];
        let w = [1.0, 1.0, 2.5, 1.0];
        let weights = [0.4, 0.3, 0.2, 0.1];
        let report = assess_overlap(support, &names(), &[&z, &w], &weights).unwrap();
        assert_eq!(report.rows_outside, 2);
        assert!((report.mass_outside - 0.5).abs() < 1e-12);
        assert_eq!(report.first_row, Some(1));
        assert!(!report.any_disjoint());
        assert_eq!(report.features[0].rows_outside, 1);
        assert_eq!(report.features[1].rows_outside, 1);
        assert_eq!((report.features[0].target_lo, report.features[0].target_hi), (0.2, 1.5));
    }

    #[test]
    fn duplicate_model_feature_names_are_refused_before_any_column_is_read() {
        // One cohort column would otherwise feed both model features.
        let c = TargetCohort {
            population_id: "clinic_b".into(),
            row_ids: vec!["r0".into()],
            features: vec!["z".into()],
            columns: vec![vec![0.5]],
            weights: vec![1.0],
        };
        let duplicated = vec!["z".to_owned(), "z".to_owned()];
        let refusal = transport_cohort_effect(
            &model(true),
            &duplicated,
            "model-1",
            "clinic_a",
            &c,
            Some(&declaration()),
            &ctx(),
        )
        .unwrap_err();
        assert_eq!(refusal.code, "invalid_argument");
        assert_eq!(refusal.detail, "transported_effect.model_invalid");
        assert_eq!(refusal.subject, "duplicate_feature_names");
    }

    #[test]
    fn absent_support_contract_refuses() {
        let c = cohort(vec![0.5], vec![1.0]);
        let refusal = run(&model(false), &c, Some(&declaration())).unwrap_err();
        assert_eq!(refusal.detail, "transported_effect.support_contract_absent");
    }

    #[test]
    fn requests_beyond_the_point_are_refused() {
        let c = cohort(vec![0.5], vec![1.0]);
        let value = run(&model(true), &c, Some(&declaration())).unwrap();
        assert_eq!(value.interval_refusal().detail, "transported_effect.interval_not_licensed");
        assert_eq!(value.law_refusal().code, "decision_contract_unsatisfied");
    }

    #[test]
    fn identity_changes_with_a_changed_row_at_unchanged_shape() {
        let a = run(&model(true), &cohort(vec![0.5, 0.6], vec![1.0, 1.0]), Some(&declaration()))
            .unwrap();
        let b = run(&model(true), &cohort(vec![0.5, 0.7], vec![1.0, 1.0]), Some(&declaration()))
            .unwrap();
        assert_ne!(a.identity(), b.identity());
        assert_ne!(a.cohort_digest, b.cohort_digest);
    }
}
