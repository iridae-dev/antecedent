//! Derived-treatment declarations for joint cells (2.2 E5).
//!
//! A *derived treatment* is a treatment built from source columns (a joint cell of binary
//! components, a product, a sum). Before any estimator sees it, the construction is declared:
//! the source columns, how they combine, when each is measured relative to the treatment, the
//! legal derived values, and what an intervention on the derived treatment means. Each source
//! is classified as **treatment construction** (a component of the treatment itself), an
//! **admissible pre-treatment covariate** (used to build the treatment but measured before
//! it, so it may stay in the adjustment set), or a **forbidden descendant** (measured after
//! the treatment, so it may never be adjusted for).
//!
//! The check refuses, never repairs. A constituent column or descendant in the adjustment set
//! is removed only by a *declared exclusion* naming the causal rule that justifies it; an
//! undeclared copy of a constituent, a covariate that tracks a constituent, and a numerically
//! rank-deficient adjustment design are refused with the columns named. The numerical rank
//! machinery of [`preflight_design`] detects these; it is never used to drop a column.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, VariableId, reason_code};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{
    EstimationError, FactorizedJointConfig, FactorizedJointFit, MAX_FACTORIZED_COMPONENTS,
    RefusalFields, fit_factorized_joint_cells,
};
use serde::{Deserialize, Serialize};

use super::preflight::{ArmSpec, PreflightInput, PreflightReport, preflight_design};
use crate::error::CausalError;

const STAGE: &str = "derived_treatment";

/// Role of one source column of a derived treatment.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstituentRole {
    /// A component of the treatment itself; never an adjustment column.
    TreatmentConstruction,
    /// Used to build the treatment but measured before it; may stay in the adjustment set.
    AdmissiblePreTreatmentCovariate,
    /// Measured after the treatment (a descendant); never an adjustment column.
    ForbiddenDescendant,
}

/// When a source column is measured relative to the treatment.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalPosition {
    /// Before the treatment is assigned.
    PreTreatment,
    /// At the moment of treatment (the treatment's own components).
    AtTreatment,
    /// After the treatment.
    PostTreatment,
}

/// One declared source column.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceColumn {
    /// Column name.
    pub name: String,
    /// Role in the construction.
    pub role: ConstituentRole,
    /// Measurement time relative to the treatment.
    pub when: TemporalPosition,
}

/// How the treatment-construction sources combine, in declared order.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Transformation {
    /// The joint cell of up to three binary components; bit `j` is the `j`-th construction
    /// source. Injective: setting the cell sets every component.
    JointCell,
    /// Product of the components. Many-to-one: it does not fix the components.
    Product,
    /// Sum of the components. Many-to-one: it does not fix the components.
    Sum,
}

/// What an intervention on the derived treatment means.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionMeaning {
    /// Every component is set jointly to the cell's levels.
    JointComponents,
    /// Only the derived value is set; the components are left free to vary.
    DerivedLevel,
}

/// The causal rule behind removing a column from the adjustment set.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExclusionRule {
    /// The column is a component of the treatment (valid only for a construction source).
    ConstituentOfTreatment,
    /// The column is a descendant of the treatment (valid only for a forbidden descendant).
    PostTreatmentDescendant,
}

/// A declared removal of one column from the adjustment set, with its causal rule.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredExclusion {
    /// The excluded column (a declared source that is in the supplied adjustment set).
    pub column: String,
    /// The causal rule that justifies the exclusion.
    pub rule: ExclusionRule,
    /// Free-text justification, required and recorded.
    pub justification: String,
}

/// An explicit declaration of a derived treatment.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedTreatmentDeclaration {
    /// Name of the derived treatment.
    pub name: String,
    /// Source columns with their roles and measurement times.
    pub sources: Vec<SourceColumn>,
    /// How the treatment-construction sources combine.
    pub transformation: Transformation,
    /// The derived values the declaration allows; an observed value outside is refused.
    pub legal_values: Vec<f64>,
    /// What an intervention on the derived treatment means.
    pub intervention: InterventionMeaning,
    /// Declared exclusions of sources from the adjustment set.
    #[serde(default)]
    pub exclusions: Vec<DeclaredExclusion>,
}

/// The accepted construction: what the estimator may use.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DerivedTreatmentPlan {
    /// Derived treatment name.
    pub name: String,
    /// Treatment-construction columns in declared order (the joint-cell components).
    pub treatment_columns: Vec<String>,
    /// The adjustment set after the declared exclusions (nothing else was removed).
    pub adjustment: Vec<String>,
    /// The exclusions applied, with their causal rules.
    pub exclusions: Vec<DeclaredExclusion>,
    /// Adjustment columns that are declared admissible pre-treatment sources.
    pub retained_covariates: Vec<String>,
    /// Distinct derived values observed on the complete-case rows, ascending.
    pub observed_levels: Vec<f64>,
    /// Complete-case rows.
    pub rows_complete: usize,
    /// The fit-free preflight report of the accepted design (review flags included).
    pub report: PreflightReport,
}

fn refuse(
    code: &'static str,
    detail: &str,
    message: &str,
    declaration: &str,
    columns: Vec<String>,
    remedy: &str,
) -> CausalError {
    let fields = RefusalFields {
        stage: Some(STAGE.to_string()),
        subject: Some(declaration.to_string()),
        reason: Some(message.to_string()),
        implicated_columns: columns,
        remedy: Some(remedy.to_string()),
        ..RefusalFields::default()
    };
    EstimationError::refused_with_fields(code, format!("{detail}: {message}"), fields).into()
}

fn invalid(declaration: &str, message: &str) -> CausalError {
    refuse(
        reason_code!("derived_treatment_invalid"),
        "joint_cells.derived_declaration_invalid",
        message,
        declaration,
        Vec::new(),
        "correct the declaration",
    )
}

impl DerivedTreatmentDeclaration {
    fn source(&self, name: &str) -> Option<&SourceColumn> {
        self.sources.iter().find(|s| s.name == name)
    }

    fn construction(&self) -> Vec<&SourceColumn> {
        self.sources.iter().filter(|s| s.role == ConstituentRole::TreatmentConstruction).collect()
    }

    /// Check the declaration on its own, before any data is read.
    ///
    /// # Errors
    ///
    /// `derived_treatment_invalid` for an empty or duplicated name, a role that contradicts its
    /// measurement time, no construction source (or more than three for a joint cell),
    /// invalid legal values, a many-to-one transformation declared as a joint-component
    /// intervention, or an exclusion without a justification or whose rule does not match the
    /// column's role.
    pub fn validate(&self) -> Result<(), CausalError> {
        let name = self.name.as_str();
        if name.trim().is_empty() {
            return Err(invalid(name, "the derived treatment needs a name"));
        }
        for (i, source) in self.sources.iter().enumerate() {
            if source.name.trim().is_empty()
                || self.sources[..i].iter().any(|s| s.name == source.name)
            {
                return Err(invalid(name, "source columns must have distinct, non-empty names"));
            }
            let consistent = match source.role {
                ConstituentRole::TreatmentConstruction => {
                    source.when != TemporalPosition::PostTreatment
                }
                ConstituentRole::AdmissiblePreTreatmentCovariate => {
                    source.when == TemporalPosition::PreTreatment
                }
                ConstituentRole::ForbiddenDescendant => {
                    source.when == TemporalPosition::PostTreatment
                }
            };
            if !consistent {
                return Err(invalid(
                    name,
                    &format!(
                        "source {:?} has role {:?} but is measured {:?}; a post-treatment column \
                         cannot be treatment construction or an admissible covariate, and a \
                         forbidden descendant is measured after the treatment",
                        source.name, source.role, source.when
                    ),
                ));
            }
        }
        let components = self.construction().len();
        let max_components = if self.transformation == Transformation::JointCell {
            MAX_FACTORIZED_COMPONENTS
        } else {
            usize::MAX
        };
        if components == 0 || components > max_components {
            return Err(invalid(
                name,
                "a derived treatment needs at least one treatment-construction source (at most \
                 three binary components for a joint cell)",
            ));
        }
        if self.transformation != Transformation::JointCell
            && self.intervention == InterventionMeaning::JointComponents
        {
            return Err(refuse(
                reason_code!("derived_treatment_invalid"),
                "joint_cells.derived_intervention_not_defined",
                "a product or sum does not fix its components, so an intervention on it cannot \
                 set them jointly; declare derived_level or use a joint cell",
                name,
                Vec::new(),
                "declare the intervention meaning derived_level, or construct a joint cell",
            ));
        }
        let mut sorted = self.legal_values.clone();
        sorted.sort_by(f64::total_cmp);
        sorted.dedup_by(|a, b| a.total_cmp(b).is_eq());
        if sorted.len() < 2
            || sorted.len() != self.legal_values.len()
            || self.legal_values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(name, "legal values must be at least two distinct finite numbers"));
        }
        if self.transformation == Transformation::JointCell {
            let cells = f64::from(1u32 << components);
            if self.legal_values.iter().any(|v| v.fract().abs() > 1e-12 || *v < 0.0 || *v >= cells)
            {
                return Err(invalid(
                    name,
                    "joint-cell legal values must be integer cell codes below 2^components",
                ));
            }
        }
        for (i, exclusion) in self.exclusions.iter().enumerate() {
            let column = exclusion.column.as_str();
            let mismatch = |message: &str| {
                refuse(
                    reason_code!("derived_treatment_invalid"),
                    "joint_cells.derived_exclusion_rule_mismatch",
                    message,
                    name,
                    vec![column.to_string()],
                    "an exclusion needs a causal rule that matches the column's declared role",
                )
            };
            let Some(source) = self.source(column) else {
                return Err(mismatch("an exclusion must name a declared source column"));
            };
            if self.exclusions[..i].iter().any(|e| e.column == exclusion.column) {
                return Err(mismatch("a column may be excluded only once"));
            }
            if exclusion.justification.trim().is_empty() {
                return Err(invalid(name, "an exclusion needs a written justification"));
            }
            let matches = matches!(
                (source.role, exclusion.rule),
                (ConstituentRole::TreatmentConstruction, ExclusionRule::ConstituentOfTreatment)
                    | (
                        ConstituentRole::ForbiddenDescendant,
                        ExclusionRule::PostTreatmentDescendant
                    )
            );
            if !matches {
                return Err(mismatch(
                    "the exclusion rule does not match the column's role; an admissible \
                     pre-treatment covariate has no causal rule that excludes it",
                ));
            }
        }
        Ok(())
    }
}

fn resolve(data: &TabularData, declaration: &str, name: &str) -> Result<VariableId, CausalError> {
    data.schema().id_of(name).map_err(|_| {
        refuse(
            reason_code!("invalid_argument"),
            "joint_cells.derived_unknown_column",
            &format!("column {name:?} is not in the data"),
            declaration,
            vec![name.to_string()],
            "name a column of the table",
        )
    })
}

/// Derived value of every complete-case row, refusing a value outside the legal set.
fn derived_values(
    data: &TabularData,
    declaration: &DerivedTreatmentDeclaration,
    components: &[VariableId],
    mask: &[bool],
) -> Result<Vec<f64>, CausalError> {
    let columns: Vec<Vec<f64>> = components
        .iter()
        .map(|&id| data.float64_masked(id, mask).map_err(CausalError::from))
        .collect::<Result<_, _>>()?;
    let rows = columns.first().map_or(0, Vec::len);
    let illegal = |value: f64, what: &str| {
        refuse(
            reason_code!("derived_treatment_invalid"),
            "joint_cells.derived_illegal_value",
            &format!("{what} {value} is outside the declared legal values"),
            &declaration.name,
            declaration.construction().iter().map(|s| s.name.clone()).collect(),
            "correct the data or widen the declared legal values",
        )
    };
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        let derived = match declaration.transformation {
            Transformation::JointCell => {
                let mut code = 0.0;
                for (j, column) in columns.iter().enumerate() {
                    let v = column[row];
                    if v.abs() > 1e-12 && (v - 1.0).abs() > 1e-12 {
                        return Err(illegal(v, "joint-cell component value"));
                    }
                    if v > 0.5 {
                        code += f64::from(1u32 << j);
                    }
                }
                code
            }
            Transformation::Product => columns.iter().map(|c| c[row]).product(),
            Transformation::Sum => columns.iter().map(|c| c[row]).sum(),
        };
        if !declaration.legal_values.iter().any(|l| (derived - l).abs() <= 1e-12) {
            return Err(illegal(derived, "derived value"));
        }
        values.push(derived);
    }
    Ok(values)
}

/// Check a derived-treatment declaration against the table and the proposed adjustment set.
///
/// The returned plan names the treatment components and the adjustment set after the declared
/// exclusions. Nothing is dropped except by a declared exclusion.
///
/// # Errors
///
/// `derived_treatment_invalid` for an invalid declaration, a constituent or forbidden
/// descendant (or the outcome) left in the adjustment set, an undeclared exact copy of a
/// constituent, duplicated constituents, a covariate that tracks a constituent, or an
/// illegal observed value; `design_rank_deficient` with the dependent columns named for a
/// rank-deficient adjustment design; `invalid_argument` for an unknown column; cancellation.
pub fn check_derived_treatment(
    data: &TabularData,
    declaration: &DerivedTreatmentDeclaration,
    outcome: &str,
    adjustment: &[String],
    ctx: &ExecutionContext,
) -> Result<DerivedTreatmentPlan, CausalError> {
    declaration.validate()?;
    let label = declaration.name.as_str();
    let named = |columns: Vec<String>, detail: &str, message: &str, remedy: &str| {
        refuse(reason_code!("derived_treatment_invalid"), detail, message, label, columns, remedy)
    };

    // Declared exclusions are the only way a column leaves the adjustment set.
    for exclusion in &declaration.exclusions {
        if !adjustment.contains(&exclusion.column) {
            return Err(named(
                vec![exclusion.column.clone()],
                "joint_cells.derived_exclusion_not_in_adjustment",
                "a declared exclusion names a column that is not in the adjustment set",
                "remove the stale exclusion",
            ));
        }
    }
    let kept: Vec<String> = adjustment
        .iter()
        .filter(|name| !declaration.exclusions.iter().any(|e| &e.column == *name))
        .cloned()
        .collect();
    if kept.iter().any(|name| name == outcome) {
        return Err(named(
            vec![outcome.to_string()],
            "joint_cells.derived_outcome_in_adjustment",
            "the outcome is in the adjustment set",
            "remove the outcome from the adjustment set",
        ));
    }
    let mut retained = Vec::new();
    for name in &kept {
        match declaration.source(name).map(|s| s.role) {
            Some(ConstituentRole::TreatmentConstruction) => {
                return Err(named(
                    vec![name.clone()],
                    "joint_cells.derived_constituent_in_adjustment",
                    "a treatment-construction column is in the adjustment set, which makes the \
                     derived treatment a function of its own covariates",
                    "remove it, or declare a constituent_of_treatment exclusion with a \
                     justification",
                ));
            }
            Some(ConstituentRole::ForbiddenDescendant) => {
                return Err(named(
                    vec![name.clone()],
                    "joint_cells.derived_descendant_in_adjustment",
                    "a column measured after the treatment is in the adjustment set",
                    "remove it, or declare a post_treatment_descendant exclusion with a \
                     justification",
                ));
            }
            Some(ConstituentRole::AdmissiblePreTreatmentCovariate) => retained.push(name.clone()),
            None => {}
        }
    }

    let components: Vec<String> =
        declaration.construction().iter().map(|s| s.name.clone()).collect();
    let component_ids: Vec<VariableId> =
        components.iter().map(|n| resolve(data, label, n)).collect::<Result<_, _>>()?;
    let outcome_id = resolve(data, label, outcome)?;
    let adjustment_ids: Vec<VariableId> =
        kept.iter().map(|n| resolve(data, label, n)).collect::<Result<_, _>>()?;

    let mut ids = component_ids.clone();
    ids.push(outcome_id);
    ids.extend_from_slice(&adjustment_ids);
    let mask = data.complete_case_mask(&ids)?;
    let values = derived_values(data, declaration, &component_ids, &mask)?;
    let mut observed_levels = values.clone();
    observed_levels.sort_by(f64::total_cmp);
    observed_levels.dedup_by(|a, b| a.total_cmp(b).is_eq());

    let input = PreflightInput {
        data,
        treatments: component_ids,
        outcome: outcome_id,
        adjustment: adjustment_ids,
        protected: Vec::new(),
        arms: vec![ArmSpec { role: "cell", levels: Vec::new() }],
    };
    let report = preflight_design(&input, ctx)?;

    for group in &report.duplicates {
        let in_group = |names: &[String]| -> Vec<String> {
            group.columns.iter().filter(|c| names.contains(c)).cloned().collect()
        };
        let duplicated_components = in_group(&components);
        if duplicated_components.len() >= 2 {
            return Err(named(
                duplicated_components,
                "joint_cells.derived_duplicate_constituents",
                "treatment-construction columns are exact copies of one another, so some joint \
                 cells cannot occur and the components are not separately identified",
                "construct the treatment from distinct components",
            ));
        }
        if group.columns.iter().any(|c| c == outcome) {
            return Err(named(
                group.columns.clone(),
                "joint_cells.derived_outcome_in_adjustment",
                "the outcome is duplicated by another column",
                "remove the copy of the outcome",
            ));
        }
        if !duplicated_components.is_empty() && !in_group(&kept).is_empty() {
            return Err(named(
                group.columns.clone(),
                "joint_cells.derived_constituent_copy_in_adjustment",
                "an adjustment column is an undeclared exact copy of a treatment-construction \
                 column",
                "remove the copy, or declare it as a source and exclude it under a causal rule",
            ));
        }
    }
    if let Some(finding) = report.findings.iter().find(|f| {
        matches!(
            f.code,
            "treatment_near_determined_by_adjustment" | "adjustment_column_tracks_treatment"
        ) && f.columns.iter().any(|c| components.contains(c))
    }) {
        return Err(named(
            finding.columns.clone(),
            "joint_cells.derived_treatment_tracked_by_covariates",
            &format!(
                "a treatment-construction column is near-deterministically predicted by the \
                 adjustment columns ({})",
                finding.detail
            ),
            "if the tracking column is a constituent or descendant, declare it as a source and \
             exclude it under a causal rule",
        ));
    }
    if let Some(rank) = report.rank.as_ref().filter(|r| !r.dependent.is_empty()) {
        let columns: Vec<String> = rank.dependent.iter().map(|d| d.column.clone()).collect();
        let message = format!(
            "the {} adjustment design columns have numerical rank {}; dependent: {}",
            rank.design_columns,
            rank.numerical_rank,
            columns.join(", ")
        );
        let fields = RefusalFields {
            stage: Some(STAGE.to_string()),
            subject: Some(declaration.name.clone()),
            reason: Some(message.clone()),
            numerical_rank: u64::try_from(rank.numerical_rank).ok(),
            design_columns: u64::try_from(rank.design_columns).ok(),
            implicated_columns: columns,
            remedy: Some(
                "remove a dependent column, or declare it as a source and exclude it under a \
                 causal rule; no column is dropped for numerical reasons"
                    .to_string(),
            ),
            ..RefusalFields::default()
        };
        return Err(EstimationError::refused_with_fields(
            reason_code!("design_rank_deficient"),
            format!("joint_cells.derived_rank_deficient: {message}"),
            fields,
        )
        .into());
    }

    Ok(DerivedTreatmentPlan {
        name: declaration.name.clone(),
        treatment_columns: components,
        adjustment: kept,
        exclusions: declaration.exclusions.clone(),
        retained_covariates: retained,
        observed_levels,
        rows_complete: report.rows_complete,
        report,
    })
}

/// Check the declaration, then estimate its joint cells with factorized propensities.
///
/// The declaration must be a [`Transformation::JointCell`] of at most three binary
/// components. The check runs first; its refusals are returned unchanged. The point estimates
/// and aligned scores of the supported cells come back with the plan; no interval is licensed.
///
/// # Errors
///
/// Every refusal of [`check_derived_treatment`], a transformation other than a joint cell
/// (`derived_treatment_invalid`), and every refusal of
/// [`fit_factorized_joint_cells`].
pub fn estimate_derived_joint_cells(
    data: &TabularData,
    declaration: &DerivedTreatmentDeclaration,
    outcome: &str,
    adjustment: &[String],
    orderings: &[Vec<usize>],
    config: &FactorizedJointConfig,
    ctx: &ExecutionContext,
) -> Result<(DerivedTreatmentPlan, FactorizedJointFit), CausalError> {
    if declaration.transformation != Transformation::JointCell {
        return Err(refuse(
            reason_code!("derived_treatment_invalid"),
            "joint_cells.derived_not_a_joint_cell",
            "factorized joint cells estimate a joint-cell transformation only",
            &declaration.name,
            Vec::new(),
            "declare the transformation joint_cell over binary components",
        ));
    }
    let plan = check_derived_treatment(data, declaration, outcome, adjustment, ctx)?;
    let schema = data.schema();
    let ids = |names: &[String]| -> Result<Vec<VariableId>, CausalError> {
        names.iter().map(|n| resolve(data, &declaration.name, n)).collect()
    };
    let treatments = ids(&plan.treatment_columns)?;
    let adjustment_ids = ids(&plan.adjustment)?;
    let outcome_id = schema.id_of(outcome).map_err(CausalError::from)?;
    let fit = fit_factorized_joint_cells(
        data,
        &treatments,
        outcome_id,
        &adjustment_ids,
        orderings,
        config,
        ctx,
    )?;
    Ok((plan, fit))
}
