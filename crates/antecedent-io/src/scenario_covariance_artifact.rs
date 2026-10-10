//! Independent artifacts for the shared-data covariance of scenario estimates (2.3A X2).
//!
//! Format version 1. The artifact stores the covariance result (scenario ids in
//! matrix order, the `K x K` covariance, the means, the method label, the
//! replicate-digest, the row-identity digest, the snapshot digest, the row count,
//! replicate and failure counts, the seed) together with everything a fresh
//! consumer needs to recompute it for the discrete plug-in family: the compact
//! integer row table (columns, unit ids, rows) and, per scenario, its declared
//! snapshot, declared dependence and plug-in functional.
//!
//! Two functionals are declared: `linear_score`, the plug-in
//! `sum_i m_i s(row_i) / n` of a per-row score built from coefficients on cell
//! patterns (`m_i` is the multiplicity of row `i` in the resampled multiset), and
//! `adjusted_contrast`, the stratified plug-in effect
//! `sum_z p(z) [P(y | x = treated, z) - P(y | x = control, z)]` over the adjustment
//! configurations present in the stored row table (an empty adjustment set gives
//! the crude contrast); a stratum or arm that the resample leaves empty fails that
//! replicate for every scenario jointly. Every sum runs in a fixed order, so the
//! matrix is bit-reproducible under the seed.
//!
//! The covariance is a point-only claim: it is never an interval. A consumer
//! recomputes the matrix from the stored table and declarations under the stored
//! method and seed and accepts only a bit-identical result. Two digests guard the
//! stored premises: the premises digest (scenario order, snapshots, dependence,
//! functionals, method) and a separate data digest (the row table). A changed row
//! snapshot, scenario order or unit id, even with both digests re-sealed, changes
//! the recomputed matrix and is refused; a scenario declared on a different
//! snapshot or unit list refuses with `scenario_covariance.unknown_dependence`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{ExecutionContext, IdentityDomain, reason_code};
use antecedent_estimate::EstimationError;
use antecedent_estimate::scenario_covariance::{
    COVARIANCE_INTERPRETATION, EXACT_ENUMERATION_HARD_CAP, ExactEnumerationOptions, MAX_REPLICATES,
    MAX_SCENARIOS, RowDependence, ScenarioCovariance, ScenarioRowEstimator,
    SharedRowBootstrapOptions, exact_enumeration_covariance, shared_row_bootstrap_covariance,
};
use serde::{Deserialize, Serialize};

use crate::IoError;

/// The artifact format this reader writes and accepts.
pub const SCENARIO_COVARIANCE_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const SCENARIO_COVARIANCE_ARTIFACT_FEATURE: &str = "scenario_shared_covariance_v1";
/// Most rows of one stored row table.
pub const MAX_TABLE_ROWS: usize = 10_000;
/// Most columns of one stored row table.
pub const MAX_TABLE_COLUMNS: usize = 64;

/// Why a covariance artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ScenarioCovarianceArtifactError {
    /// The feature marker or a stored field is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A recorded limit or stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The row table does not match the data digest.
    #[error("row snapshot data identity mismatch")]
    DataIdentityMismatch,
    /// The recomputed covariance differs from the stored one.
    #[error("covariance does not replay")]
    ReplayMismatch,
}

impl ScenarioCovarianceArtifactError {
    /// The registered reason code and `scenario_covariance.*` detail of the refusal.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, &'static str) {
        match self {
            Self::UnsupportedSemantics(_) => {
                (reason_code!("route_not_supported"), "scenario_covariance.unsupported_semantics")
            }
            Self::LimitsExceeded(_) => {
                (reason_code!("cell_not_licensed"), "scenario_covariance.consumer_limit_exceeded")
            }
            Self::PremisesMismatch => {
                (reason_code!("invalid_argument"), "scenario_covariance.premises_mismatch")
            }
            Self::DataIdentityMismatch => {
                (reason_code!("invalid_argument"), "scenario_covariance.data_identity_mismatch")
            }
            Self::ReplayMismatch => {
                (reason_code!("invalid_argument"), "scenario_covariance.replay_mismatch")
            }
        }
    }
}

impl From<ScenarioCovarianceArtifactError> for IoError {
    fn from(error: ScenarioCovarianceArtifactError) -> Self {
        let (code, detail) = error.refusal();
        Self::Refused { code, message: format!("{detail}: {error}") }
    }
}

fn invalid(detail: &str, message: &str) -> IoError {
    IoError::Refused {
        code: reason_code!("invalid_argument"),
        message: format!("{detail}: {message}"),
    }
}

fn unknown_dependence(message: &str) -> IoError {
    IoError::Refused {
        code: reason_code!("route_not_supported"),
        message: format!("scenario_covariance.unknown_dependence: {message}"),
    }
}

#[allow(clippy::cast_precision_loss)] // Row counts are far below 2^53.
const fn to_f64(n: usize) -> f64 {
    n as f64
}

/// Consumer bounds. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct ScenarioCovarianceConsumeLimits {
    /// Most rows the consumer will replay.
    pub max_rows: usize,
    /// Most columns the consumer will replay.
    pub max_columns: usize,
    /// Most bootstrap replicates the consumer will replay.
    pub max_replicates: usize,
    /// Most count vectors an exact enumeration the consumer will replay may visit.
    pub max_compositions: u64,
}

impl Default for ScenarioCovarianceConsumeLimits {
    fn default() -> Self {
        Self {
            max_rows: MAX_TABLE_ROWS,
            max_columns: MAX_TABLE_COLUMNS,
            max_replicates: MAX_REPLICATES,
            max_compositions: EXACT_ENUMERATION_HARD_CAP,
        }
    }
}

/// The compact discrete row table: integer cell codes with stable unit ids.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RowTableWire {
    /// Column names, distinct.
    pub columns: Vec<String>,
    /// One stable unit id per row, in row order.
    pub unit_ids: Vec<String>,
    /// `rows[i][c]` is the integer code of column `c` in row `i`.
    pub rows: Vec<Vec<i64>>,
}

impl RowTableWire {
    fn check(&self) -> Result<(), IoError> {
        if self.columns.is_empty() {
            return Err(invalid(
                "scenario_covariance.invalid_row_table",
                "the row table has no column",
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        if !self.columns.iter().all(|c| seen.insert(c.as_str())) {
            return Err(invalid("scenario_covariance.invalid_row_table", "column names repeat"));
        }
        if self.rows.len() != self.unit_ids.len() {
            return Err(invalid(
                "scenario_covariance.invalid_row_table",
                "the table needs exactly one unit id per row",
            ));
        }
        if self.rows.iter().any(|row| row.len() != self.columns.len()) {
            return Err(invalid(
                "scenario_covariance.invalid_row_table",
                "every row needs exactly one code per column",
            ));
        }
        Ok(())
    }

    fn column(&self, name: &str) -> Result<usize, IoError> {
        self.columns.iter().position(|c| c == name).ok_or_else(|| {
            invalid("scenario_covariance.invalid_functional", &format!("unknown column '{name}'"))
        })
    }
}

/// One term of a linear score: `coefficient` on rows matching every `(column, code)` pair.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScoreTermWire {
    /// Coefficient of the term.
    pub coefficient: f64,
    /// Cell pattern: the term applies to rows with each column equal to its code.
    pub pattern: Vec<(String, i64)>,
}

/// A scenario's plug-in functional of the resampled rows' cell proportions.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum FunctionalWire {
    /// `sum_i m_i s(row_i) / n` for the per-row score `s` the terms define.
    LinearScore {
        /// The score's terms.
        terms: Vec<ScoreTermWire>,
    },
    /// The stratified contrast of `P(outcome = outcome_value | treatment = .)`
    /// over the adjustment configurations of the stored table.
    AdjustedContrast {
        /// Treatment column.
        treatment: String,
        /// Outcome column.
        outcome: String,
        /// Adjustment columns (possibly none).
        adjustment: Vec<String>,
        /// Treated code.
        treated: i64,
        /// Control code.
        control: i64,
        /// Outcome code counted as the event.
        outcome_value: i64,
    },
}

type Estimator = Box<dyn Fn(&[u32]) -> Result<f64, EstimationError> + Send + Sync>;

fn matches_pattern(row: &[i64], pattern: &[(usize, i64)]) -> bool {
    pattern.iter().all(|(column, code)| row.get(*column) == Some(code))
}

fn linear_estimator(table: &RowTableWire, terms: &[ScoreTermWire]) -> Result<Estimator, IoError> {
    if terms.is_empty() {
        return Err(invalid(
            "scenario_covariance.invalid_functional",
            "a linear score needs a term",
        ));
    }
    let mut compiled = Vec::with_capacity(terms.len());
    for term in terms {
        if !term.coefficient.is_finite() {
            return Err(invalid(
                "scenario_covariance.invalid_functional",
                "a score coefficient must be finite",
            ));
        }
        let pattern = term
            .pattern
            .iter()
            .map(|(name, code)| table.column(name).map(|c| (c, *code)))
            .collect::<Result<Vec<_>, _>>()?;
        compiled.push((term.coefficient, pattern));
    }
    let scores: Vec<f64> = table
        .rows
        .iter()
        .map(|row| {
            compiled
                .iter()
                .map(
                    |(coefficient, pattern)| {
                        if matches_pattern(row, pattern) { *coefficient } else { 0.0 }
                    },
                )
                .sum::<f64>()
        })
        .collect();
    let n = to_f64(scores.len());
    Ok(Box::new(move |counts: &[u32]| {
        if counts.len() != scores.len() {
            return Err(EstimationError::data_msg("multiplicities and rows disagree in length"));
        }
        Ok(counts.iter().zip(&scores).map(|(c, s)| f64::from(*c) * s).sum::<f64>() / n)
    }))
}

/// Per-row facts of the stratified contrast.
struct ContrastRow {
    stratum: usize,
    arm: u8,
    event: bool,
}

#[allow(clippy::too_many_arguments)] // The six declared fields of the functional.
fn adjusted_estimator(
    table: &RowTableWire,
    treatment: &str,
    outcome: &str,
    adjustment: &[String],
    treated: i64,
    control: i64,
    outcome_value: i64,
) -> Result<Estimator, IoError> {
    if treated == control {
        return Err(invalid(
            "scenario_covariance.invalid_functional",
            "the treated and control codes must differ",
        ));
    }
    let (t, y) = (table.column(treatment)?, table.column(outcome)?);
    let adjust = adjustment.iter().map(|a| table.column(a)).collect::<Result<Vec<_>, _>>()?;
    let key = |row: &[i64]| -> Vec<i64> { adjust.iter().map(|c| row[*c]).collect() };
    let strata: BTreeMap<Vec<i64>, usize> = table
        .rows
        .iter()
        .map(|row| key(row))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .enumerate()
        .map(|(i, k)| (k, i))
        .collect();
    let rows: Vec<ContrastRow> = table
        .rows
        .iter()
        .map(|row| ContrastRow {
            stratum: strata.get(&key(row)).copied().unwrap_or(0),
            arm: if row[t] == treated {
                1
            } else if row[t] == control {
                0
            } else {
                2
            },
            event: row[y] == outcome_value,
        })
        .collect();
    let width = strata.len();
    Ok(Box::new(move |counts: &[u32]| {
        if counts.len() != rows.len() {
            return Err(EstimationError::data_msg("multiplicities and rows disagree in length"));
        }
        // Per stratum: total, treated, control, treated events, control events.
        let mut cells = vec![[0.0_f64; 5]; width];
        let mut total = 0.0_f64;
        for (row, count) in rows.iter().zip(counts) {
            let m = f64::from(*count);
            total += m;
            let cell = &mut cells[row.stratum];
            cell[0] += m;
            match (row.arm, row.event) {
                (1, true) => {
                    cell[1] += m;
                    cell[3] += m;
                }
                (1, false) => cell[1] += m,
                (0, true) => {
                    cell[2] += m;
                    cell[4] += m;
                }
                (0, false) => cell[2] += m,
                _ => {}
            }
        }
        let mut effect = 0.0_f64;
        for [stratum_total, n1, n0, y1, y0] in &cells {
            if *n1 <= 0.0 || *n0 <= 0.0 {
                return Err(EstimationError::unsupported("empty stratum arm in a resample"));
            }
            effect += stratum_total / total * (y1 / n1 - y0 / n0);
        }
        Ok(effect)
    }))
}

fn estimator_for(table: &RowTableWire, functional: &FunctionalWire) -> Result<Estimator, IoError> {
    match functional {
        FunctionalWire::LinearScore { terms } => linear_estimator(table, terms),
        FunctionalWire::AdjustedContrast {
            treatment,
            outcome,
            adjustment,
            treated,
            control,
            outcome_value,
        } => adjusted_estimator(
            table,
            treatment,
            outcome,
            adjustment,
            *treated,
            *control,
            *outcome_value,
        ),
    }
}

fn default_dependence() -> String {
    "shared_rows".into()
}

/// One scenario's declaration: its id, the snapshot and dependence it declares, and its functional.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CovarianceScenarioWire {
    /// Scenario id.
    pub id: String,
    /// The row snapshot this scenario's estimator declares.
    pub snapshot: String,
    /// `shared_rows`, `independent_sample` or `unknown`.
    #[serde(default = "default_dependence")]
    pub dependence: String,
    /// The unit list this scenario declares; the row table's when absent.
    #[serde(default)]
    pub unit_ids: Option<Vec<String>>,
    /// The plug-in functional.
    pub functional: FunctionalWire,
}

/// The covariance method and its options.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
pub enum CovarianceMethodWire {
    /// Seeded whole-row bootstrap with one replicate id across scenarios.
    SharedRowBootstrap {
        /// Replicates drawn.
        replicates: usize,
        /// Seed of the replicate-id stream.
        seed: u64,
        /// Largest tolerated fraction of replicates dropped for an estimator failure.
        max_failure_fraction: f64,
    },
    /// Exact enumeration of the multinomial resampling distribution.
    ExactEnumeration {
        /// Declared cap on the number of count vectors.
        max_compositions: u64,
        /// Largest tolerated probability mass dropped for an estimator failure.
        max_failure_mass: f64,
    },
}

/// Everything a covariance is computed from: the row table, the scenario
/// declarations in matrix order and the method.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CovarianceSpec {
    /// The shared discrete row table.
    pub table: RowTableWire,
    /// Scenario declarations, in matrix order.
    pub scenarios: Vec<CovarianceScenarioWire>,
    /// The method and its options.
    pub method: CovarianceMethodWire,
}

fn dependence_of(label: &str) -> Result<RowDependence, IoError> {
    match label {
        "shared_rows" => Ok(RowDependence::SharedRows),
        "independent_sample" => Ok(RowDependence::IndependentSample),
        "unknown" => Ok(RowDependence::Unknown),
        _ => Err(invalid("scenario_covariance.invalid_dependence", "unknown dependence label")),
    }
}

impl CovarianceSpec {
    /// Compile every scenario declaration against the row table.
    ///
    /// # Errors
    /// An invalid table or functional (`invalid_argument`), more rows or columns
    /// than the bounds allow (`cell_not_licensed`), or a scenario declaring a unit
    /// list of another length (`route_not_supported` /
    /// `scenario_covariance.unknown_dependence`).
    pub fn estimators(&self) -> Result<Vec<ScenarioRowEstimator>, IoError> {
        self.table.check()?;
        if self.table.rows.len() > MAX_TABLE_ROWS || self.table.columns.len() > MAX_TABLE_COLUMNS {
            return Err(IoError::Refused {
                code: reason_code!("cell_not_licensed"),
                message: format!(
                    "scenario_covariance.table_too_large: at most {MAX_TABLE_ROWS} rows and \
                     {MAX_TABLE_COLUMNS} columns"
                ),
            });
        }
        let table_units: Vec<Arc<str>> =
            self.table.unit_ids.iter().map(|u| Arc::from(u.as_str())).collect();
        let mut out = Vec::with_capacity(self.scenarios.len());
        for scenario in &self.scenarios {
            let units = match &scenario.unit_ids {
                None => table_units.clone(),
                Some(own) if own.len() == table_units.len() => {
                    own.iter().map(|u| Arc::from(u.as_str())).collect()
                }
                Some(_) => {
                    return Err(unknown_dependence(&format!(
                        "scenario '{}' declares a unit list of a different length than the row table",
                        scenario.id
                    )));
                }
            };
            let estimator = estimator_for(&self.table, &scenario.functional)?;
            out.push(
                ScenarioRowEstimator::new(
                    scenario.id.as_str(),
                    scenario.snapshot.as_str(),
                    units,
                    estimator,
                )
                .with_dependence(dependence_of(&scenario.dependence)?),
            );
        }
        Ok(out)
    }

    /// Compute the joint covariance of the declared scenario estimates.
    ///
    /// # Errors
    /// The refusals of [`Self::estimators`] and of the covariance estimators
    /// (`scenario_covariance.*`).
    pub fn compute(&self, ctx: &ExecutionContext) -> Result<ScenarioCovariance, IoError> {
        let estimators = self.estimators()?;
        match &self.method {
            CovarianceMethodWire::SharedRowBootstrap { replicates, seed, max_failure_fraction } => {
                shared_row_bootstrap_covariance(
                    &estimators,
                    &SharedRowBootstrapOptions {
                        replicates: *replicates,
                        seed: *seed,
                        max_failure_fraction: *max_failure_fraction,
                    },
                    ctx,
                )
            }
            CovarianceMethodWire::ExactEnumeration { max_compositions, max_failure_mass } => {
                exact_enumeration_covariance(
                    &estimators,
                    &ExactEnumerationOptions {
                        max_compositions: *max_compositions,
                        max_failure_mass: *max_failure_mass,
                    },
                    ctx,
                )
            }
        }
        .map_err(IoError::from)
    }
}

/// The stored covariance result.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CovarianceResultWire {
    /// Scenario ids in matrix order.
    pub scenario_ids: Vec<String>,
    /// Mean of each scenario's estimate over the retained resampling distribution.
    pub means: Vec<f64>,
    /// Row-major `K x K` covariance.
    pub covariance: Vec<f64>,
    /// `shared_row_bootstrap` or `exact_enumeration`.
    pub method: String,
    /// Digest of the method, snapshot, rows, scenario order and every replicate.
    pub replicate_digest: String,
    /// Digest of the ordered unit ids.
    pub row_identity_digest: String,
    /// The common row snapshot the scenarios declared.
    pub snapshot_digest: String,
    /// Rows resampled.
    pub n_rows: usize,
    /// Replicates drawn, or count vectors enumerated.
    pub replicates_total: u64,
    /// Replicates (count vectors) retained.
    pub replicates_used: u64,
    /// Replicates (count vectors) dropped because some estimator failed.
    pub failed_replicates: u64,
    /// Dropped probability mass (exact) or fraction (bootstrap).
    pub failed_mass: f64,
    /// Seed (bootstrap only).
    pub seed: Option<u64>,
    /// How the matrix may be read: point only.
    pub interpretation: String,
}

impl CovarianceResultWire {
    /// Encode a covariance.
    #[must_use]
    pub fn from_covariance(cov: &ScenarioCovariance) -> Self {
        Self {
            scenario_ids: cov.scenario_ids.iter().map(ToString::to_string).collect(),
            means: cov.means.clone(),
            covariance: cov.covariance.clone(),
            method: cov.method.label().into(),
            replicate_digest: cov.replicate_digest.clone(),
            row_identity_digest: cov.row_identity_digest.clone(),
            snapshot_digest: cov.snapshot_digest.to_string(),
            n_rows: cov.n_rows,
            replicates_total: cov.replicates_total,
            replicates_used: cov.replicates_used,
            failed_replicates: cov.failed_replicates,
            failed_mass: cov.failed_mass,
            seed: cov.seed,
            interpretation: cov.interpretation.into(),
        }
    }
}

/// Versioned shared-data covariance with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioCovarianceArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Scenario declarations in matrix order.
    pub scenarios: Vec<CovarianceScenarioWire>,
    /// The method and its options.
    pub method: CovarianceMethodWire,
    /// The shared discrete row table.
    pub table: RowTableWire,
    /// The stored covariance result.
    pub result: CovarianceResultWire,
    /// Digest of the scenario order, snapshots, dependence, functionals and method.
    pub premises_digest: String,
    /// Digest of the row table.
    pub data_digest: String,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

fn premises_identity(wire: &ScenarioCovarianceArtifactWire) -> Result<String, IoError> {
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &("scenario_shared_covariance_v1", &wire.scenarios, &wire.method),
    )?
    .to_hex())
}

fn data_identity(wire: &ScenarioCovarianceArtifactWire) -> Result<String, IoError> {
    Ok(crate::identity::digest_wire(
        IdentityDomain::TransportCertificate,
        &("scenario_shared_covariance_data_v1", &wire.table),
    )?
    .to_hex())
}

impl ScenarioCovarianceArtifactWire {
    /// Build an artifact from the spec and the covariance it produced.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn checked(
        spec: &CovarianceSpec,
        covariance: &ScenarioCovariance,
    ) -> Result<Self, IoError> {
        let mut wire = Self {
            version: SCENARIO_COVARIANCE_ARTIFACT_VERSION,
            required_features: vec![SCENARIO_COVARIANCE_ARTIFACT_FEATURE.into()],
            scenarios: spec.scenarios.clone(),
            method: spec.method.clone(),
            table: spec.table.clone(),
            result: CovarianceResultWire::from_covariance(covariance),
            premises_digest: String::new(),
            data_digest: String::new(),
        };
        wire.premises_digest = premises_identity(&wire)?;
        wire.data_digest = data_identity(&wire)?;
        Ok(wire)
    }

    /// The premises digest these stored premises would carry. A consumer never
    /// trusts it: re-sealing a mutated artifact with it still fails replay.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        premises_identity(self)
    }

    /// The data digest this stored row table would carry.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        data_identity(self)
    }

    /// The spec a consumer recomputes from.
    #[must_use]
    pub fn spec(&self) -> CovarianceSpec {
        CovarianceSpec {
            table: self.table.clone(),
            scenarios: self.scenarios.clone(),
            method: self.method.clone(),
        }
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version first.
    ///
    /// # Errors
    /// [`IoError::UnsupportedVersion`], a decoding failure, or a foreign feature.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != SCENARIO_COVARIANCE_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        if wire.required_features != [SCENARIO_COVARIANCE_ARTIFACT_FEATURE] {
            return Err(
                ScenarioCovarianceArtifactError::UnsupportedSemantics("required features").into()
            );
        }
        if wire.result.interpretation != COVARIANCE_INTERPRETATION {
            return Err(ScenarioCovarianceArtifactError::UnsupportedSemantics(
                "a covariance is point only",
            )
            .into());
        }
        Ok(wire)
    }

    fn check_limits(
        &self,
        limits: &ScenarioCovarianceConsumeLimits,
    ) -> Result<(), ScenarioCovarianceArtifactError> {
        let exceeded = ScenarioCovarianceArtifactError::LimitsExceeded;
        if self.table.rows.len() > limits.max_rows {
            return Err(exceeded("row count"));
        }
        if self.table.columns.len() > limits.max_columns {
            return Err(exceeded("column count"));
        }
        if self.scenarios.len() > MAX_SCENARIOS {
            return Err(exceeded("scenario count"));
        }
        match &self.method {
            CovarianceMethodWire::SharedRowBootstrap { replicates, .. } => {
                if *replicates > limits.max_replicates {
                    return Err(exceeded("replicate count"));
                }
            }
            CovarianceMethodWire::ExactEnumeration { max_compositions, .. } => {
                if *max_compositions > limits.max_compositions {
                    return Err(exceeded("exact enumeration cap"));
                }
            }
        }
        Ok(())
    }

    /// Decode and replay everything, accepting only a bit-identical covariance.
    ///
    /// # Errors
    /// A limit, digest, reconstruction, estimator or result mismatch; a scenario
    /// declared on a different snapshot or unit list refuses with
    /// `scenario_covariance.unknown_dependence`.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: ScenarioCovarianceConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, ScenarioCovariance), IoError> {
        let wire = Self::decode(bytes)?;
        wire.check_limits(&limits)?;
        if premises_identity(&wire)? != wire.premises_digest {
            return Err(ScenarioCovarianceArtifactError::PremisesMismatch.into());
        }
        if data_identity(&wire)? != wire.data_digest {
            return Err(ScenarioCovarianceArtifactError::DataIdentityMismatch.into());
        }
        let covariance = wire.spec().compute(ctx)?;
        let replayed = CovarianceResultWire::from_covariance(&covariance);
        if crate::to_cbor(&replayed)? != crate::to_cbor(&wire.result)? {
            return Err(ScenarioCovarianceArtifactError::ReplayMismatch.into());
        }
        Ok((wire, covariance))
    }
}
