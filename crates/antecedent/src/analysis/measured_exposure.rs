//! A measured exposure declared on the study's treatment: prepare, estimate, export.
//!
//! The exposure of an effect query is often not observed but recorded with error: a misclassified
//! 0/1 record, a per-unit probability, a covariate-dependent detection, several repeat codings of
//! a latent facet. This module routes one request through the layers that own each rule and adds
//! no scientific rule of its own:
//!
//! 1. **Prepare** ([`StudyBuilder::measured_exposure`]) imputes the latent exposure `m` times
//!    from the declared measurement, the outcome and the covariates
//!    (`antecedent_estimate::measurement::impute_exposure`): an unidentified declaration refuses
//!    before any draw.
//! 2. **Estimate** ([`PreparedMeasuredExposure::estimate`]) runs the caller's ordinary study on
//!    every completed table (the exposure column replaced by the draw), takes each run's effect
//!    and variance (any result that carries a point effect and a finite standard error: a
//!    structure-posterior mixture, a population-targeted effect, a plain adjustment), and pools
//!    them with Rubin's rules. The result carries the pooled effect, its Rubin `t` interval
//!    (calibration `not_measured`), the within, between and total variance, the estimate that
//!    takes the recorded column at face value and the stages the runs consumed.
//!    [`PreparedMeasuredExposure::estimate_over_structures`] adds a structure search: the
//!    posterior search runs on every completed table, the engine pools the posteriors into one
//!    accepted posterior with its provenance, and the caller's study (built on
//!    [`crate::StudyBuilder::accepted_graph_posterior`]) runs on every table over it.
//! 3. **Export** ([`PreparedMeasuredExposure::export_draws`]) writes the draws as the library's
//!    posterior artifact (one quantity per unit, one draw per imputation); [`consume_exposure_draws`]
//!    reads them back against a declaration identity retained out of band.
//!
//! The imputation outcome model must contain the relations the analysis relies on; that is the
//! declared assumption of the route and the draws are proper only when it holds.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{Assumption, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::measurement::{
    ExposureDraws, ExposureMeasurement, ImputationData, MeasuredEffect, MeasurementError,
    impute_exposure, pool_measured_effect,
};
use antecedent_prob::{PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};

use super::StudyBuilder;
use super::structure_route::{AcceptedPosterior, DiscoverySpec};
use crate::accepted::StructureProvenance;
use crate::io::{CausalPosteriorWire, IoError, PosteriorQuantityWire, encode_posterior_artifact};
use crate::schema::VERSION;
use crate::{CausalError, StudyResult};
use antecedent_io::decode_causal_posterior_bytes;

/// Backend id prefix of an exposure-draw posterior artifact.
const BACKEND_PREFIX: &str = "exposure_measurement";

/// A refusal of the measured-exposure route.
#[derive(Debug)]
#[non_exhaustive]
pub enum MeasuredExposureError {
    /// The declaration or the imputation refused.
    Measurement(MeasurementError),
    /// The caller's study refused or failed on a completed table.
    Analysis(CausalError),
    /// The columns, names or artifact were malformed.
    Input(String),
}

impl std::fmt::Display for MeasuredExposureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Measurement(error) => write!(f, "{error}"),
            Self::Analysis(error) => write!(f, "{error}"),
            Self::Input(message) => write!(f, "invalid measured-exposure input: {message}"),
        }
    }
}

impl std::error::Error for MeasuredExposureError {}

impl From<MeasurementError> for MeasuredExposureError {
    fn from(error: MeasurementError) -> Self {
        Self::Measurement(error)
    }
}

fn input(message: impl Into<String>) -> MeasuredExposureError {
    MeasuredExposureError::Input(message.into())
}

/// The named numeric columns of the analysis table.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredColumns {
    names: Vec<String>,
    values: Vec<Vec<f64>>,
}

impl MeasuredColumns {
    /// Columns of equal length with distinct names.
    ///
    /// # Errors
    /// No columns, columns of different lengths or a repeated name.
    pub fn new(names: Vec<String>, values: Vec<Vec<f64>>) -> Result<Self, MeasuredExposureError> {
        if names.is_empty() || names.len() != values.len() {
            return Err(input("columns need one name each"));
        }
        let rows = values[0].len();
        if values.iter().any(|v| v.len() != rows) {
            return Err(input("columns differ in length"));
        }
        let mut sorted: Vec<&String> = names.iter().collect();
        sorted.sort();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(input("column names must be distinct"));
        }
        Ok(Self { names, values })
    }

    fn get(&self, name: &str) -> Result<&[f64], MeasuredExposureError> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.values[i].as_slice())
            .ok_or_else(|| input(format!("column `{name}` is not in the table")))
    }

    /// Whether the table holds a column of this name.
    #[must_use]
    pub fn has_column(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    /// The table with `exposure` replaced in place by (or, when absent, appended as) `column`.
    fn table_with(
        &self,
        exposure: &str,
        column: &[f64],
    ) -> Result<TabularData, MeasuredExposureError> {
        let mut pairs: Vec<(String, &[f64])> = Vec::with_capacity(self.names.len() + 1);
        let mut replaced = false;
        for (name, values) in self.names.iter().zip(&self.values) {
            if name == exposure {
                pairs.push((name.clone(), column));
                replaced = true;
            } else {
                pairs.push((name.clone(), values.as_slice()));
            }
        }
        if !replaced {
            pairs.push((exposure.to_owned(), column));
        }
        TabularData::from_f64_columns(pairs).map_err(|error| input(error.to_string()))
    }

    /// The table as recorded.
    fn table(&self) -> Result<TabularData, MeasuredExposureError> {
        TabularData::from_f64_columns(
            self.names.iter().zip(&self.values).map(|(n, v)| (n.clone(), v.as_slice())),
        )
        .map_err(|error| input(error.to_string()))
    }
}

/// What a measured exposure is declared on and how many draws are imputed.
#[derive(Clone, Debug)]
pub struct MeasuredExposureRequest {
    /// Name of the exposure column the study's query treats.
    pub exposure: String,
    /// Name of the outcome column.
    pub outcome: String,
    /// Names of the columns the imputation conditions on besides the outcome (the covariates
    /// the analysis adjusts for).
    pub covariates: Vec<String>,
    /// Imputations `m`, at least two.
    pub draws: usize,
    /// Seed of the imputations.
    pub seed: u64,
}

/// A measured exposure whose draws have been imputed; the question is not yet answered.
#[derive(Clone, Debug)]
pub struct PreparedMeasuredExposure {
    request: MeasuredExposureRequest,
    columns: MeasuredColumns,
    draws: ExposureDraws,
}

impl StudyBuilder {
    /// Declare a measured exposure on the study's treatment and impute its draws.
    ///
    /// # Errors
    /// An unidentified or malformed declaration, a column that is not in `columns`, or
    /// cancellation.
    pub fn measured_exposure(
        declaration: &ExposureMeasurement,
        columns: MeasuredColumns,
        request: MeasuredExposureRequest,
        ctx: &ExecutionContext,
    ) -> Result<PreparedMeasuredExposure, MeasuredExposureError> {
        let outcome = columns.get(&request.outcome)?;
        let rows = outcome.len();
        let mut covariates = Vec::with_capacity(rows * request.covariates.len());
        for name in &request.covariates {
            if name == &request.exposure || name == &request.outcome {
                return Err(input("covariates must not include the exposure or the outcome"));
            }
            covariates.extend_from_slice(columns.get(name)?);
        }
        let data = ImputationData {
            outcome,
            covariates: &covariates,
            covariate_count: request.covariates.len(),
        };
        let draws = impute_exposure(declaration, &data, request.draws, request.seed, ctx)?;
        Ok(PreparedMeasuredExposure { request, columns, draws })
    }
}

/// The stages a pooled result consumed, named by what each one declared.
///
/// The measurement is the declaration the draws were imputed from; the structure stage is the
/// search the posterior came from, its structure prior and the pooling over the completed
/// tables; every other stage (a weighted target population, a composed or selected outcome
/// prior, a structure-posterior mixture) is the declaration the per-table runs recorded in the
/// assumptions of their results.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainProvenance {
    /// What the draws were imputed from: kind, declaration identity, data identity and the
    /// number of draws.
    pub measurement: String,
    /// The accepted posterior's provenance, when a structure search ran on the completed
    /// tables.
    pub structure: Option<Arc<StructureProvenance>>,
    /// `(id, statement)` of every stage declaration the per-table results recorded, in first
    /// appearance order and without repeats.
    pub declarations: Vec<(String, String)>,
}

impl ChainProvenance {
    /// Every stage the result consumed as one statement each: the measurement, the structure
    /// search and its pooling, and each recorded declaration.
    #[must_use]
    pub fn statements(&self) -> Vec<String> {
        let mut out = vec![self.measurement.clone()];
        if let Some(structure) = &self.structure {
            out.push(structure.statement());
        }
        out.extend(self.declarations.iter().map(|(_, statement)| statement.clone()));
        out
    }

    /// The statement of the recorded declaration with this id, when the runs recorded one.
    #[must_use]
    pub fn declaration(&self, id: &str) -> Option<&str> {
        self.declarations
            .iter()
            .find(|(declared, _)| declared == id)
            .map(|(_, statement)| statement.as_str())
    }
}

/// The pooled effect of a measured exposure.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredExposureEstimate {
    /// Rubin pooling of the per-draw effects, with the naive effect.
    pub effect: MeasuredEffect,
    /// Identity of the declaration the draws were imputed from.
    pub declaration_identity: String,
    /// Identity of the data the imputation read.
    pub data_identity: String,
    /// The stages the per-table runs consumed.
    pub provenance: ChainProvenance,
    /// The mean over the completed tables of the posterior mass a structure-posterior run left
    /// unanswered (unidentified, not evaluable for the declared target, or not evaluated); zero
    /// when the runs were not over a structure posterior.
    pub unanswered_mass: f64,
}

/// The point effect and its variance of one run, with the recorded stage declarations and the
/// mass its structure mixture left unanswered.
struct RunEffect {
    point: f64,
    variance: f64,
    declarations: Vec<(String, String)>,
    unanswered_mass: f64,
}

fn run_effect(result: &StudyResult) -> Result<RunEffect, MeasuredExposureError> {
    let effect = result.estimate().as_effect().ok_or_else(|| {
        input("a measured exposure is pooled over results that carry a point effect")
    })?;
    let analytic = effect.uncertainty.analytic.se;
    let se = if analytic.is_finite() {
        analytic
    } else {
        effect.uncertainty.bootstrap.se.filter(|se| se.is_finite()).unwrap_or(f64::NAN)
    };
    if !(effect.ate.is_finite() && se.is_finite()) {
        return Err(input("a per-draw effect or standard error is not finite"));
    }
    let declarations = effect
        .provenance
        .assumptions
        .entries
        .iter()
        .filter_map(|record| match &record.assumption {
            Assumption::ParametricRestriction(declared) => {
                Some((declared.id.to_string(), declared.description.to_string()))
            }
            Assumption::PriorRestriction(declared) => {
                Some((declared.id.to_string(), declared.description.to_string()))
            }
            Assumption::Custom { id, description } => {
                Some((id.to_string(), description.to_string()))
            }
            _ => None,
        })
        .collect();
    let unanswered_mass = result.structural_response.as_ref().map_or(0.0, |mixture| {
        mixture.unidentified_mass + mixture.unevaluable_mass + mixture.subsampled_out_mass
    });
    Ok(RunEffect { point: effect.ate, variance: se * se, declarations, unanswered_mass })
}

impl PreparedMeasuredExposure {
    /// The imputed draws.
    #[must_use]
    pub const fn draws(&self) -> &ExposureDraws {
        &self.draws
    }

    /// The request the draws were imputed for.
    #[must_use]
    pub const fn request(&self) -> &MeasuredExposureRequest {
        &self.request
    }

    /// The completed table of draw `k`: the columns with the exposure replaced by the draw.
    ///
    /// # Errors
    /// A draw out of range.
    pub fn completed_table(&self, k: usize) -> Result<TabularData, MeasuredExposureError> {
        if k >= self.draws.draw_count {
            return Err(input("draw index out of range"));
        }
        self.columns.table_with(&self.request.exposure, self.draws.column(k))
    }

    /// Run `analyse` (the caller's ordinary study) on every completed table and pool.
    ///
    /// When the exposure has a recorded column, `analyse` also runs on the table as recorded and
    /// the pooled result carries that estimate as the naive effect.
    ///
    /// # Errors
    /// The first refusal of `analyse`, a result without a point effect and a finite standard
    /// error, or pooling that cannot proceed.
    pub fn estimate(
        &self,
        analyse: impl FnMut(&TabularData) -> Result<StudyResult, CausalError>,
    ) -> Result<MeasuredExposureEstimate, MeasuredExposureError> {
        self.pool(analyse, true, None)
    }

    /// The posterior over structures of the completed tables: `spec` runs on every one, the
    /// engine pools the per-table posteriors with equal weight, and the pooled posterior comes
    /// back accepted, with the search, its structure prior, the pooling and this measurement as
    /// provenance. Hand it to a study with [`crate::StudyBuilder::accepted_graph_posterior`].
    ///
    /// `variables` are the searched columns, in the table's column order.
    ///
    /// # Errors
    /// A method that does not pool, or a search failure on any completed table.
    pub fn structure_posterior(
        &self,
        spec: &DiscoverySpec,
        variables: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<AcceptedPosterior, MeasuredExposureError> {
        let tables = (0..self.draws.draw_count)
            .map(|k| self.completed_table(k))
            .collect::<Result<Vec<_>, _>>()?;
        let source = self.measurement_statement();
        spec.pooled_posterior(&tables, variables, Some(&source), ctx)
            .map_err(MeasuredExposureError::Analysis)
    }

    /// The structure search on the completed tables and the effect over the pooled posterior,
    /// pooled by Rubin's rules.
    ///
    /// [`Self::structure_posterior`] runs once; `analyse` then runs on every completed table
    /// with the accepted posterior, and builds its study on it (a posterior-mixture study,
    /// optionally for a declared target population, whose uncertainty carries the
    /// between-structure spread). The naive effect is not computed: the recorded column has no
    /// structure search of its own.
    ///
    /// # Errors
    /// The refusals of [`Self::structure_posterior`] and [`Self::estimate`].
    pub fn estimate_over_structures(
        &self,
        spec: &DiscoverySpec,
        variables: &[VariableId],
        ctx: &ExecutionContext,
        mut analyse: impl FnMut(&TabularData, &AcceptedPosterior) -> Result<StudyResult, CausalError>,
    ) -> Result<MeasuredExposureEstimate, MeasuredExposureError> {
        let posterior = self.structure_posterior(spec, variables, ctx)?;
        self.pool(|table| analyse(table, &posterior), false, Some(&posterior))
    }

    fn measurement_statement(&self) -> String {
        format!(
            "exposure measurement: {} declaration {} on data {}, {} draws (seed {})",
            self.draws.kind,
            self.draws.declaration_identity,
            self.draws.data_identity,
            self.draws.draw_count,
            self.draws.seed
        )
    }

    fn pool(
        &self,
        mut analyse: impl FnMut(&TabularData) -> Result<StudyResult, CausalError>,
        with_naive: bool,
        posterior: Option<&AcceptedPosterior>,
    ) -> Result<MeasuredExposureEstimate, MeasuredExposureError> {
        let mut points = Vec::with_capacity(self.draws.draw_count);
        let mut variances = Vec::with_capacity(self.draws.draw_count);
        let mut declarations: Vec<(String, String)> = Vec::new();
        let mut unanswered = 0.0;
        for k in 0..self.draws.draw_count {
            let table = self.completed_table(k)?;
            let result = analyse(&table).map_err(MeasuredExposureError::Analysis)?;
            let run = run_effect(&result)?;
            points.push(run.point);
            variances.push(run.variance);
            unanswered += run.unanswered_mass;
            for declared in run.declarations {
                if !declarations.iter().any(|(id, _)| id == &declared.0) {
                    declarations.push(declared);
                }
            }
        }
        let naive = if with_naive && self.columns.has_column(&self.request.exposure) {
            let table = self.columns.table()?;
            let result = analyse(&table).map_err(MeasuredExposureError::Analysis)?;
            Some(run_effect(&result)?.point)
        } else {
            None
        };
        let effect = pool_measured_effect(&points, &variances, naive)?;
        Ok(MeasuredExposureEstimate {
            effect,
            declaration_identity: self.draws.declaration_identity.clone(),
            data_identity: self.draws.data_identity.clone(),
            provenance: ChainProvenance {
                measurement: self.measurement_statement(),
                structure: posterior.map(|accepted| Arc::new(accepted.provenance().clone())),
                declarations,
            },
            unanswered_mass: unanswered / self.draws.draw_count.max(1) as f64,
        })
    }

    /// The draws as a posterior artifact.
    ///
    /// # Errors
    /// Encoding failures.
    pub fn export_draws(&self) -> Result<Vec<u8>, MeasuredExposureError> {
        export_exposure_draws(&self.request.exposure, &self.draws)
    }
}

fn io_error(error: &IoError) -> MeasuredExposureError {
    input(error.to_string())
}

/// Write exposure draws as the library's posterior artifact: one `Effect` quantity per unit
/// named `"{exposure}[i]"` and one draw per imputation. The declaration and data identities,
/// the levels and the seed travel in the backend id.
///
/// # Errors
/// Encoding failures.
pub fn export_exposure_draws(
    exposure: &str,
    draws: &ExposureDraws,
) -> Result<Vec<u8>, MeasuredExposureError> {
    let (n, m) = (draws.rows, draws.draw_count);
    // Column-major: all draws of unit 0, then of unit 1, ...
    let mut column_major = Vec::with_capacity(n * m);
    for i in 0..n {
        for k in 0..m {
            column_major.push(draws.draws[k * n + i]);
        }
    }
    let schema = PosteriorSchema {
        quantities: (0..n).map(|_| PosteriorQuantityKind::Scalar { name: "q".into() }).collect(),
    };
    let summary = PosteriorDraws::from_column_major(schema, m, column_major.clone())
        .map_err(|error| input(error.to_string()))?
        .summarize();
    let meta = CausalPosteriorWire {
        quantities: (0..n)
            .map(|i| PosteriorQuantityWire::Effect { name: format!("{exposure}[{i}]") })
            .collect(),
        n_draws: u32::try_from(m).map_err(|_| input("too many draws"))?,
        mean: summary.mean.to_vec(),
        sd: summary.sd.to_vec(),
        q025: summary.q025.to_vec(),
        q975: summary.q975.to_vec(),
        identification: "identified_under_parametric_restrictions".into(),
        unidentified_mass: 0.0,
        subsampled_out_mass: 0.0,
        backend_id: format!(
            "{BACKEND_PREFIX}.{}.{}.{}.{}.{}",
            draws.kind, draws.declaration_identity, draws.data_identity, draws.levels, draws.seed
        ),
        converged: true,
        hessian_condition: f64::NAN,
        draws_encoding: "f64_le_colmajor".into(),
        treatment_contrast: None,
        preparation: None,
    };
    let artifact = encode_posterior_artifact(&meta, &column_major, "exposure-draws", VERSION)
        .map_err(|error| io_error(&error))?;
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes).map_err(|error| io_error(&error))?;
    Ok(bytes)
}

/// Read exposure draws back from the posterior artifact [`export_exposure_draws`] wrote.
///
/// Refuses an artifact that is not an exposure-draw record, that does not hold `exposure`, or
/// whose declaration identity differs from `expected_declaration_identity` (retained out of
/// band).
///
/// # Errors
/// A malformed or foreign artifact, or a mismatch.
pub fn consume_exposure_draws(
    bytes: &[u8],
    exposure: &str,
    expected_declaration_identity: Option<&str>,
) -> Result<ExposureDraws, MeasuredExposureError> {
    let (meta, column_major) =
        decode_causal_posterior_bytes(bytes).map_err(|error| io_error(&error))?;
    let parts: Vec<&str> = meta.backend_id.split('.').collect();
    if parts.len() != 6 || parts[0] != BACKEND_PREFIX {
        return Err(input("the artifact is not an exposure-measurement draw record"));
    }
    let prefix = format!("{exposure}[");
    let named = meta
        .quantities
        .iter()
        .all(|q| matches!(q, PosteriorQuantityWire::Effect { name } if name.starts_with(&prefix)));
    if meta.quantities.is_empty() || !named {
        return Err(input(format!("the artifact does not hold draws of `{exposure}`")));
    }
    if let Some(expected) = expected_declaration_identity
        && parts[2] != expected
    {
        return Err(input("the artifact's declaration identity differs from the expected one"));
    }
    let kind: &'static str = match parts[1] {
        "misclassified_exposure" => "misclassified_exposure",
        "soft_exposure" => "soft_exposure",
        "detected_exposure" => "detected_exposure",
        "repeat_coded_exposure" => "repeat_coded_exposure",
        other => return Err(input(format!("unknown exposure kind `{other}`"))),
    };
    let levels: usize = parts[4].parse().map_err(|_| input("malformed level count"))?;
    let seed: u64 = parts[5].parse().map_err(|_| input("malformed seed"))?;
    let (n, m) = (meta.quantities.len(), meta.n_draws as usize);
    let mut draws = vec![0.0; n * m];
    for i in 0..n {
        for k in 0..m {
            draws[k * n + i] = column_major[i * m + k];
        }
    }
    let denominator = (levels.max(2) - 1) as f64;
    let probability: Vec<f64> = (0..n)
        .map(|i| (0..m).map(|k| draws[k * n + i]).sum::<f64>() / m as f64 / denominator)
        .collect();
    Ok(ExposureDraws {
        kind,
        levels,
        rows: n,
        draw_count: m,
        probability,
        draws,
        declaration_identity: parts[2].to_owned(),
        data_identity: parts[3].to_owned(),
        seed,
    })
}
