//! Whole-method measured interval for an effect estimated on a sample split after discovery.
//!
//! The procedure is one construction: constraint-based discovery (PC with the partial
//! correlation test) on one part of the rows, the structure accepted only when it is a single
//! fully directed graph from a finished search, and the linear effect of the treatment on the
//! outcome estimated by adjusting for the treatment's parents in that graph on the other,
//! disjoint part of the rows. The interval is the ordinary least-squares coefficient interval.
//!
//! What a record attests is the coverage of the *true causal effect* by that whole procedure
//! over the declared family of data-generating processes, at the sample sizes it measured.
//! It is a statement about the method, never about one dataset: faithfulness and causal
//! sufficiency of a given table are untestable, so a given discovered graph is never
//! certified. A construction that reuses the discovery rows for the analysis is a different
//! construction ([`REUSE_STRUCTURE`]) and no record can license it here.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    MeasuredExpectation, MeasuredInference, ScalarExecution, authorize, compare_replay, decode,
};
use crate::IoError;
use antecedent_core::{CalibrationBasis, ExecutionContext, IdentityDomain, RngFactory, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_discovery::selective::reuse::{SplitRule, split_rows};
use antecedent_discovery::selective::{
    CiTestSpec, DiscoveredStructure, Mark, SelectiveDiscoveryError, StaticAlgorithm,
    StaticDiscoveryContract, StaticSelectiveDiscovery, StructureClass,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Route name the measured consumer registers.
pub const SPLIT_EFFECT_ROUTE: &str = "split_effect";
/// Calibration `structure` value of the sample-split design.
pub const SPLIT_STRUCTURE: &str = "discovered_sample_split";
/// Calibration `structure` value of discovery and analysis on the same rows. It is measured
/// only as a named boundary and is never authorized.
pub const REUSE_STRUCTURE: &str = "discovered_full_row_reuse";
/// Calibration `estimator` value of the whole procedure.
pub const ESTIMATOR: &str = "sample_split.pc_parcorr.linear_adjustment";
/// Calibration `dependence` value of the split design: the two parts share no row.
pub const SPLIT_DEPENDENCE: &str = "disjoint_rows";
/// Calibration `dependence` value of the reuse design.
pub const REUSE_DEPENDENCE: &str = "shared_rows";
/// Calibration `functional` value: the mean of the linear effect over the observed rows.
pub const FUNCTIONAL: &str = "all_observed.mean";
/// The only conditional-independence test of the measured construction.
pub const CI_TEST: &str = "parcorr";
/// The only significance threshold of the measured construction.
pub const ALPHA: f64 = 0.01;
/// The only share of rows given to discovery in the measured construction.
pub const DISCOVERY_FRACTION: f64 = 0.5;
/// The only nominal level of the measured construction.
pub const LEVEL: f64 = 0.95;
/// Fewest searched variables measured.
pub const MIN_VARIABLES: usize = 5;
/// Most searched variables measured.
pub const MAX_VARIABLES: usize = 7;
const MAX_ROWS: usize = 1 << 20;
const FORMAT_VERSION: u32 = 1;

fn refused(detail: &str, message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("cell_not_licensed"),
        message: format!("{detail}: {message}"),
    }
}

fn malformed(detail: &str, message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("{detail}: {message}"),
    }
}

pub(super) fn discovery_refusal(error: &SelectiveDiscoveryError) -> IoError {
    match error {
        SelectiveDiscoveryError::Refused { code, detail, remedy } => {
            IoError::Refused { code, message: format!("{detail}: {remedy}") }
        }
        other => malformed("split_effect.discovery_failed", &other.to_string()),
    }
}

/// The complete input of one split-effect execution: the table and the declared roles.
///
/// Everything else (algorithm, test, threshold, split share, level) is the measured
/// construction and is not a field, so a source cannot state another protocol.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitEffectSource {
    /// Format version.
    pub version: u32,
    /// Column names, in variable order; every column is searched.
    pub names: Vec<String>,
    /// Column values, aligned with `names`.
    pub columns: Vec<Vec<f64>>,
    /// Index of the treatment column.
    pub treatment: u32,
    /// Index of the outcome column.
    pub outcome: u32,
    /// Seed of the seeded row permutation that decides the split.
    pub split_seed: u64,
    /// Seed of the discovery contract.
    pub discovery_seed: u64,
}

impl SplitEffectSource {
    /// Build a source from float columns. Unequal column lengths or a non-finite value are
    /// refused at execution, not here.
    #[must_use]
    pub fn new(
        names: Vec<String>,
        columns: Vec<Vec<f64>>,
        treatment: u32,
        outcome: u32,
        split_seed: u64,
        discovery_seed: u64,
    ) -> Self {
        Self {
            version: FORMAT_VERSION,
            names,
            columns,
            treatment,
            outcome,
            split_seed,
            discovery_seed,
        }
    }

    /// Canonical bytes of this source.
    ///
    /// # Errors
    /// Canonical encoding failure.
    pub fn to_bytes(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode and re-encode, refusing bytes that are not the canonical encoding.
    ///
    /// # Errors
    /// Malformed or non-canonical bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, IoError> {
        let source: Self = crate::from_cbor(bytes)?;
        if crate::to_cbor(&source)? != bytes {
            return Err(malformed(
                "split_effect.source_not_canonical",
                "the source bytes are not the canonical encoding",
            ));
        }
        Ok(source)
    }

    fn validate(&self) -> Result<usize, IoError> {
        let k = self.columns.len();
        if self.version != FORMAT_VERSION || self.names.len() != k {
            return Err(malformed(
                "split_effect.source_invalid",
                "unknown format version or names and columns of different counts",
            ));
        }
        if !(MIN_VARIABLES..=MAX_VARIABLES).contains(&k) {
            return Err(refused(
                "split_effect.variable_count_not_measured",
                "only tables of five to seven searched variables were measured",
            ));
        }
        let rows = self.columns[0].len();
        if !(8..=MAX_ROWS).contains(&rows) || self.columns.iter().any(|c| c.len() != rows) {
            return Err(malformed(
                "split_effect.source_invalid",
                "columns must have one length between 8 and 2^20 rows",
            ));
        }
        if self.columns.iter().flatten().any(|v| !v.is_finite()) {
            return Err(malformed(
                "split_effect.source_invalid",
                "missing or non-finite values are not supported",
            ));
        }
        let (t, y) = (self.treatment as usize, self.outcome as usize);
        if t >= k || y >= k || t == y {
            return Err(malformed(
                "split_effect.source_invalid",
                "treatment and outcome must be two distinct columns",
            ));
        }
        Ok(rows)
    }

    fn frame(&self) -> Result<TabularData, IoError> {
        TabularData::from_f64_columns(
            self.names.iter().map(String::as_str).zip(self.columns.iter().map(Vec::as_slice)),
        )
        .map_err(|error| malformed("split_effect.source_invalid", &error.to_string()))
    }
}

/// A linear effect estimated by adjusting for the treatment's parents in an accepted graph.
#[derive(Clone, Debug, PartialEq)]
pub struct AdjustedEffect {
    /// Coefficient of the treatment.
    pub point: f64,
    /// Lower interval endpoint.
    pub lower: f64,
    /// Upper interval endpoint.
    pub upper: f64,
    /// Nominal level.
    pub level: f64,
    /// Standard error of the coefficient.
    pub standard_error: f64,
    /// Residual degrees of freedom of the Student quantile.
    pub degrees_of_freedom: f64,
    /// Rows the regression used.
    pub rows: u64,
    /// Variable indices adjusted for (the treatment's parents), ascending.
    pub adjustment_set: Vec<u32>,
}

impl AdjustedEffect {
    /// Calibration basis of this interval under a stated design.
    ///
    /// The runtime lookup keys on `structure` and `dependence`; the split design passes
    /// [`SPLIT_STRUCTURE`] and [`SPLIT_DEPENDENCE`].
    #[must_use]
    pub fn keyed_basis(&self, structure: &str, dependence: &str) -> CalibrationBasis {
        CalibrationBasis::new(
            [
                "AverageEffect",
                "Cpdag",
                structure,
                "tabular",
                "Frequentist",
                ESTIMATOR,
                "analytic_se",
                "homoskedastic",
                dependence,
                "",
                FUNCTIONAL,
            ]
            .map(Arc::<str>::from),
            self.level,
            Arc::from("point"),
            self.rows,
            None,
            None,
            0.0,
        )
    }
}

fn directed_parents(
    structure: &DiscoveredStructure,
    target: u32,
    variables: usize,
) -> Result<Vec<u32>, IoError> {
    if structure.class() != StructureClass::Cpdag {
        return Err(refused(
            "split_effect.structure_class",
            "only a completed partially directed graph over static variables is accepted",
        ));
    }
    let mut parents = Vec::new();
    for edge in structure.edges() {
        let known = |v: u32| (v as usize) < variables;
        if edge.a.kind != 0
            || edge.b.kind != 0
            || !known(edge.a.variable)
            || !known(edge.b.variable)
        {
            return Err(refused(
                "split_effect.structure_class",
                "a node is not a static searched variable",
            ));
        }
        let (source, sink) = match (edge.at_a, edge.at_b) {
            (Mark::Tail, Mark::Arrow) => (edge.a, edge.b),
            (Mark::Arrow, Mark::Tail) => (edge.b, edge.a),
            _ => {
                return Err(refused(
                    "split_effect.structure_unresolved",
                    "an edge is not definitely directed",
                ));
            }
        };
        if sink.variable == target {
            parents.push(source.variable);
        }
    }
    parents.sort_unstable();
    parents.dedup();
    Ok(parents)
}

/// Inverse of a symmetric `k x k` matrix by Gauss-Jordan elimination with partial pivoting.
fn invert(mut a: Vec<f64>, k: usize) -> Option<Vec<f64>> {
    let scale = (0..k).map(|i| a[i * k + i].abs()).fold(0.0_f64, f64::max);
    let mut inverse = vec![0.0; k * k];
    for i in 0..k {
        inverse[i * k + i] = 1.0;
    }
    for col in 0..k {
        let mut pivot = col;
        for row in (col + 1)..k {
            if a[row * k + col].abs() > a[pivot * k + col].abs() {
                pivot = row;
            }
        }
        if a[pivot * k + col].abs().partial_cmp(&(1e-12 * scale))
            != Some(std::cmp::Ordering::Greater)
        {
            return None;
        }
        if pivot != col {
            for c in 0..k {
                a.swap(pivot * k + c, col * k + c);
                inverse.swap(pivot * k + c, col * k + c);
            }
        }
        let diagonal = a[col * k + col];
        for c in 0..k {
            a[col * k + c] /= diagonal;
            inverse[col * k + c] /= diagonal;
        }
        for row in 0..k {
            if row == col {
                continue;
            }
            let factor = a[row * k + col];
            if factor != 0.0 {
                for c in 0..k {
                    a[row * k + c] -= factor * a[col * k + c];
                    inverse[row * k + c] -= factor * inverse[col * k + c];
                }
            }
        }
    }
    Some(inverse)
}

/// Linear effect of `treatment` on `outcome` adjusting for `adjustment` on `frame` (a table).
///
/// The interval is `coefficient +/- t * se` with the classical homoskedastic ordinary
/// least-squares standard error and a Student quantile on the residual degrees of freedom.
///
/// # Errors
/// A column that is not numeric, a singular design, or fewer rows than parameters plus two.
pub fn adjusted_effect(
    frame: &TabularData,
    treatment: u32,
    outcome: u32,
    adjustment: &[u32],
    level: f64,
) -> Result<AdjustedEffect, IoError> {
    let column = |index: u32| {
        frame
            .float64_values(VariableId::from_raw(index))
            .map_err(|error| malformed("split_effect.source_invalid", &error.to_string()))
    };
    let y = column(outcome)?;
    let x = column(treatment)?;
    let covariates: Vec<Vec<f64>> =
        adjustment.iter().map(|&index| column(index)).collect::<Result<_, _>>()?;
    let n = y.len();
    let k = 2 + covariates.len();
    if n < k + 2 || x.len() != n || covariates.iter().any(|c| c.len() != n) {
        return Err(refused(
            "split_effect.too_few_rows",
            "the analysis rows do not exceed the regression parameters",
        ));
    }
    let design = |row: usize, col: usize| -> f64 {
        match col {
            0 => 1.0,
            1 => x[row],
            c => covariates[c - 2][row],
        }
    };
    let mut gram = vec![0.0; k * k];
    let mut moment = vec![0.0; k];
    #[allow(
        clippy::needless_range_loop,
        reason = "the row indexes both the outcome and the design"
    )]
    for row in 0..n {
        for a in 0..k {
            let da = design(row, a);
            moment[a] += da * y[row];
            for b in a..k {
                gram[a * k + b] += da * design(row, b);
            }
        }
    }
    for a in 0..k {
        for b in 0..a {
            gram[a * k + b] = gram[b * k + a];
        }
    }
    let inverse = invert(gram, k).ok_or_else(|| {
        refused("split_effect.design_singular", "the adjusted regression design is singular")
    })?;
    let beta: Vec<f64> =
        (0..k).map(|a| (0..k).map(|b| inverse[a * k + b] * moment[b]).sum()).collect();
    let mut rss = 0.0;
    #[allow(
        clippy::needless_range_loop,
        reason = "the row indexes both the outcome and the design"
    )]
    for row in 0..n {
        let fitted: f64 = (0..k).map(|c| beta[c] * design(row, c)).sum();
        rss += (y[row] - fitted).powi(2);
    }
    #[allow(clippy::cast_precision_loss, reason = "bounded row and parameter counts")]
    let df = (n - k) as f64;
    let variance = rss / df * inverse[k + 1];
    if !(variance.is_finite() && variance > 0.0) {
        return Err(refused(
            "split_effect.degenerate_variance",
            "the coefficient has no positive sampling variance",
        ));
    }
    let se = variance.sqrt();
    let quantile = antecedent_stats::student_t_ppf(1.0 - (1.0 - level) / 2.0, df);
    let point = beta[1];
    Ok(AdjustedEffect {
        point,
        lower: point - quantile * se,
        upper: point + quantile * se,
        level,
        standard_error: se,
        degrees_of_freedom: df,
        rows: n as u64,
        adjustment_set: adjustment.to_vec(),
    })
}

/// The effect of a fully directed discovered structure on the `frame` table: the treatment's parents
/// in the structure are the adjustment set.
///
/// # Errors
/// A structure that is not one fully directed graph, an outcome that is a parent of the
/// treatment, or the refusals of [`adjusted_effect`].
pub fn effect_on_structure(
    structure: &DiscoveredStructure,
    frame: &TabularData,
    treatment: u32,
    outcome: u32,
    variables: usize,
    level: f64,
) -> Result<AdjustedEffect, IoError> {
    if !structure.is_fully_directed() {
        let unresolved = structure
            .edges()
            .iter()
            .filter(|edge| !edge.is_directed())
            .map(|edge| format!("{:?}-{:?} [{:?},{:?}]", edge.a, edge.b, edge.at_a, edge.at_b))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(refused(
            "split_effect.structure_unresolved",
            &format!(
                "the discovered structure has unresolved edges ({unresolved}); no completion is \
                 part of the measured construction"
            ),
        ));
    }
    let parents = directed_parents(structure, treatment, variables)?;
    if parents.contains(&outcome) {
        return Err(refused(
            "split_effect.outcome_precedes_treatment",
            "the discovered structure orients the outcome into the treatment",
        ));
    }
    adjusted_effect(frame, treatment, outcome, &parents, level)
}

/// One executed sample-split effect with the identities of the discovery it rests on.
#[derive(Clone, Debug, PartialEq)]
pub struct SplitEffectRun {
    /// The interval.
    pub effect: AdjustedEffect,
    /// Rows discovery used.
    pub discovery_rows: u64,
    /// Rows the effect was estimated on.
    pub analysis_rows: u64,
    /// Identity of the split receipt.
    pub receipt_identity: [u8; 32],
    /// Identity of the discovered structure.
    pub structure_identity: [u8; 32],
    /// Identity of the discovery result.
    pub result_identity: [u8; 32],
}

impl SplitEffectRun {
    /// Calibration basis of the sample-split construction.
    #[must_use]
    pub fn keyed_basis(&self) -> CalibrationBasis {
        self.effect.keyed_basis(SPLIT_STRUCTURE, SPLIT_DEPENDENCE)
    }
}

/// Execute the measured construction: split, discover on one part, accept only a resolved
/// finished structure, and estimate on the other part.
///
/// # Errors
/// An invalid source or one outside the measured variable count, a refused discovery, a
/// search that exhausted a bound or met conflicting orientation evidence, a structure with
/// unresolved edges, or the refusals of [`effect_on_structure`].
pub fn run(source: &SplitEffectSource, ctx: &ExecutionContext) -> Result<SplitEffectRun, IoError> {
    source.validate()?;
    let frame = source.frame()?;
    let k = source.columns.len();
    let variables: Vec<VariableId> =
        (0..k).map(|i| VariableId::from_raw(u32::try_from(i).unwrap_or(u32::MAX))).collect();
    let split = split_rows(
        &frame,
        &variables,
        SplitRule::SeededPermutation {
            seed: source.split_seed,
            discovery_fraction: DISCOVERY_FRACTION,
        },
    )
    .map_err(|error| discovery_refusal(&error))?;
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        variables,
        CiTestSpec::named(CI_TEST),
        source.discovery_seed,
    );
    contract.fdr = None;
    contract.constraints.alpha = ALPHA;
    let mut run_ctx = ctx.clone();
    run_ctx.rng = RngFactory::from_seed(source.discovery_seed);
    let mut adapter = StaticSelectiveDiscovery::new();
    adapter
        .run(&contract, &split.discovery, &run_ctx)
        .map_err(|error| discovery_refusal(&error))?;
    let outcome = adapter.outcome().ok_or_else(|| {
        malformed("split_effect.discovery_failed", "the discovery run retained no outcome")
    })?;
    if outcome.search_incomplete || outcome.orientation_conflict {
        return Err(refused(
            "split_effect.search_not_finished",
            "the discovery search exhausted a bound or met conflicting orientation evidence",
        ));
    }
    let effect = effect_on_structure(
        &outcome.structure,
        &split.analysis,
        source.treatment,
        source.outcome,
        k,
        LEVEL,
    )?;
    Ok(SplitEffectRun {
        effect,
        discovery_rows: split.receipt.n_discovery as u64,
        analysis_rows: split.receipt.n_analysis as u64,
        receipt_identity: split.receipt.identity,
        structure_identity: outcome.structure_identity,
        result_identity: outcome.result_identity,
    })
}

/// Replay `source_bytes` and bind the effect to a currently attesting coverage record.
///
/// # Errors
/// Any refusal of [`run`], a source outside the measured protocol, or a basis without a
/// current attesting non-boundary record covering the analysis row count.
pub fn measure(source_bytes: &[u8], ctx: &ExecutionContext) -> Result<MeasuredInference, IoError> {
    let source = SplitEffectSource::from_bytes(source_bytes)?;
    let executed = run(&source, ctx)?;
    let premises = crate::digest_hex(
        IdentityDomain::Claim,
        &(
            SPLIT_EFFECT_ROUTE,
            ESTIMATOR,
            CI_TEST,
            ALPHA,
            DISCOVERY_FRACTION,
            LEVEL,
            source.treatment,
            source.outcome,
            source.split_seed,
            source.discovery_seed,
        ),
    )?;
    let data = crate::digest_hex(IdentityDomain::Claim, &(&source.names, &source.columns))?;
    let effect = &executed.effect;
    let scope = serde_json::json!({
        "validation_design":"linear_gaussian_collider_oriented_dags_five_to_seven_variables",
        "evidence_interpretation":"coverage of the true linear causal effect by the whole procedure (split, PC discovery, resolved-structure acceptance, parent adjustment) over the measured family and sample sizes; never a statement about one dataset's graph",
        "declared_model_assumptions":["independent identically distributed rows","linear Gaussian structural equations","causal sufficiency","faithfulness","a fully orientable equivalence class"],
        "sampling_assumptions_authenticated_from_rows":false,
        "structure":SPLIT_STRUCTURE,
        "algorithm":"pc","ci_test":CI_TEST,"alpha":ALPHA,
        "discovery_fraction":DISCOVERY_FRACTION,
        "discovery_rows":executed.discovery_rows,"analysis_rows":executed.analysis_rows,
        "adjustment_set":effect.adjustment_set,
        "level":effect.level,"interval":"ols_student_t",
        "receipt_identity":crate::digest_hex(IdentityDomain::Claim, &executed.receipt_identity)?,
        "structure_identity":crate::digest_hex(IdentityDomain::Claim, &executed.structure_identity)?,
        "reported_scalar":"effect"
    });
    authorize(
        "split_effect",
        source_bytes.to_vec(),
        &premises,
        &data,
        scope,
        vec![ScalarExecution {
            name: "effect".into(),
            point: effect.point,
            interval: (effect.lower, effect.upper),
            basis: executed.keyed_basis(),
        }],
    )
}

/// Independently replay the original source and rederive the envelope's scalar evidence.
///
/// # Errors
/// Any source, expectation, protocol, evidence or scalar mismatch.
pub fn consume_measured(
    bytes: &[u8],
    expected: &MeasuredExpectation,
    ctx: &ExecutionContext,
) -> Result<MeasuredInference, IoError> {
    let wire = decode(bytes, expected)?;
    let measured = measure(&wire.source, ctx)?;
    compare_replay(&wire, &measured)?;
    Ok(measured)
}
