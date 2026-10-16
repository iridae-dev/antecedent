//! Whole-method measured interval for an effect estimated on a sample split after discovery,
//! when only the treatment's parent set has to be resolved.
//!
//! The procedure is one construction: constraint-based discovery (PC with the partial
//! correlation test, a conditioning-set size and a parent limit equal to the declared maximum
//! graph degree) on one part of the rows; acceptance of the discovered class when the search
//! finished and **every edge incident to the treatment is oriented**, whatever the state of
//! the unrelated edges; and the effect of the treatment on the outcome estimated by adjusting
//! for the treatment's parents on the other, disjoint part of the rows.
//!
//! In a completed partially directed graph an edge incident to the treatment that is directed
//! is directed in every member of the equivalence class, so the treatment's parents are the
//! same in every member and the parent adjustment is the total effect for the whole class.
//! Edges elsewhere may stay undirected: they are listed on the result, never completed. A
//! class in which some edge incident to the treatment is undirected contains graphs with
//! different parent sets; it is refused, with the edges named, and no member is chosen.
//!
//! A linear study reports the total effect: the treatment coefficient with the ordinary
//! least-squares Student interval. A log-linear study declares, before any data are read, that
//! every searched column is positive and enters the analysis as its natural logarithm; discovery
//! and the regression run on that scale, and the reported effect is the intervention mean ratio
//! `exp(shift * beta)` of multiplying the treatment by `exp(shift)`, with the interval the
//! monotone image of the log-scale Student interval (its delta-method standard error is
//! reported as well).
//!
//! What a record attests is the coverage of the *true* total effect (or intervention mean
//! ratio) by that whole procedure over the declared family of data-generating processes, at the
//! sample sizes and variable counts it measured. It is a statement about the method, never
//! about one dataset: faithfulness and causal sufficiency of a table are untestable, so a given
//! discovered graph is never certified.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::split_effect::{
    ALPHA, AdjustedEffect, CI_TEST, DISCOVERY_FRACTION, LEVEL, SPLIT_DEPENDENCE, adjusted_effect,
    discovery_refusal,
};
use super::{
    MeasuredExpectation, MeasuredInference, ScalarExecution, authorize, build_report,
    compare_replay, decode,
};
use crate::IoError;
use antecedent_core::{CalibrationBasis, ExecutionContext, IdentityDomain, RngFactory, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::selective::reuse::{SplitRule, split_rows};
use antecedent_discovery::selective::{
    CiTestSpec, DiscoveredStructure, Mark, StaticAlgorithm, StaticDiscoveryContract,
    StaticSelectiveDiscovery, StructureClass, StructureEdge,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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

/// Route name the measured consumer registers.
pub const PARENT_SET_EFFECT_ROUTE: &str = "split_parent_set_effect";
/// Calibration `structure` value of the treatment-parent-set sample-split design.
pub const PARENT_SET_STRUCTURE: &str = "discovered_sample_split.treatment_parent_set";
/// Calibration `estimator` value of a linear study.
pub const LINEAR_ESTIMATOR: &str = "parent_set_split.pc_parcorr.total_effect_adjustment";
/// Calibration `estimator` value of a log-linear study.
pub const LOG_LINEAR_ESTIMATOR: &str = "parent_set_split.pc_parcorr.log_linear_mean_ratio";
/// The topologies the construction is measured on, each its own record. A table does not
/// announce its topology, so a cell is licensed only when the record of every topology attests:
/// the conservative intersection.
pub const TOPOLOGIES: [&str; 3] = [
    "hub_confounded_multi_path",
    "layered_multi_mediator_nuisance",
    "confounded_direct_and_mediated",
];

/// The topologies whose records govern a cell of declared maximum degree `max_degree`. The
/// denser third core has degree four, so it governs degree four only.
#[must_use]
pub fn governing_topologies(max_degree: u32) -> &'static [&'static str] {
    if max_degree >= SUPPORTED_MAX_DEGREE { &TOPOLOGIES } else { &TOPOLOGIES[..2] }
}
/// Fewest searched variables measured.
pub const MIN_VARIABLES: usize = 10;
/// Most searched variables measured.
pub const MAX_VARIABLES: usize = 20;
/// Largest declared maximum graph degree the construction supports.
pub const SUPPORTED_MAX_DEGREE: u32 = 4;
/// The acceptance reason of every executed construction: all edges incident to the treatment
/// are oriented in the discovered class.
pub const ACCEPTANCE: &str = "treatment_parent_set_unambiguous";
/// Scalar name of a linear study's total effect.
pub const TOTAL_EFFECT: &str = "total_effect";
/// Scalar name of a log-linear study's intervention mean ratio.
pub const MEAN_RATIO: &str = "intervention_mean_ratio";
const MAX_ROWS: usize = 1 << 20;
const FORMAT_VERSION: u32 = 1;
const PARENT_LIMIT_DIAGNOSTIC: &str = "parent_limit_removed_adjacencies";

/// The scale on which the analysis runs, declared before any data are read.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transformation {
    /// The columns as given; the effect is the total effect per unit of the treatment.
    Linear,
    /// Every searched column enters as its natural logarithm; the effect is an intervention
    /// mean ratio.
    LogLinear,
}

impl Transformation {
    /// The declaration's name: `linear` or `log_linear`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::LogLinear => "log_linear",
        }
    }

    /// Parse a declaration.
    ///
    /// # Errors
    /// Anything other than `linear` or `log_linear`: the transformation is a declaration made
    /// before the data are touched, never inferred from them.
    pub fn parse(name: &str) -> Result<Self, IoError> {
        match name {
            "linear" => Ok(Self::Linear),
            "log_linear" => Ok(Self::LogLinear),
            other => Err(malformed(
                "parent_set_effect.transformation_undeclared",
                &format!(
                    "the transformation must be declared as `linear` or `log_linear` before any \
                     data are read; got `{other}`"
                ),
            )),
        }
    }

    const fn estimator(self) -> &'static str {
        match self {
            Self::Linear => LINEAR_ESTIMATOR,
            Self::LogLinear => LOG_LINEAR_ESTIMATOR,
        }
    }

    /// The reported scalar's name, also the stem of the calibration functional.
    const fn scalar(self) -> &'static str {
        match self {
            Self::Linear => TOTAL_EFFECT,
            Self::LogLinear => MEAN_RATIO,
        }
    }
}

/// Background knowledge the analyst asserts about the searched variables, by column index. It
/// is part of the source and of the calibration key: a construction that uses it is another
/// construction.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Knowledge {
    /// `(cause, effect)` pairs that are edges, with that direction.
    pub required: Vec<(u32, u32)>,
    /// `(cause, effect)` pairs that are not edges in that direction.
    pub forbidden: Vec<(u32, u32)>,
    /// Tiers of column indices: a later tier never causes an earlier one.
    pub tiers: Vec<Vec<u32>>,
}

impl Knowledge {
    /// Which statements are made, e.g. `required+tiers`; empty when none is.
    fn label(&self) -> String {
        let kinds = [
            ("required", !self.required.is_empty()),
            ("forbidden", !self.forbidden.is_empty()),
            ("tiers", !self.tiers.is_empty()),
        ];
        kinds.iter().filter(|(_, present)| *present).map(|(n, _)| *n).collect::<Vec<_>>().join("+")
    }
}

/// The complete input of one parent-set split-effect execution: the table, the declared roles
/// and the declared analysis family.
///
/// Everything else (algorithm, test, threshold, split share, level) is the measured
/// construction and is not a field, so a source cannot state another protocol.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParentSetEffectSource {
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
    /// Declared scale of the analysis.
    pub transformation: Transformation,
    /// Declared maximum graph degree: the PC conditioning-set size and parent limit.
    pub max_degree: u32,
    /// Log-scale shift of the prespecified intervention (the treatment is multiplied by
    /// `exp(shift)`); required and non-zero for a log-linear study, absent for a linear one.
    pub log_shift: Option<f64>,
    /// Background knowledge handed to discovery, when any is asserted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub knowledge: Option<Knowledge>,
}

impl ParentSetEffectSource {
    /// Build a source from float columns. Unequal column lengths, a non-finite value or a
    /// declaration outside the construction is refused at execution, not here.
    #[must_use]
    #[allow(clippy::too_many_arguments, reason = "the source is exactly these declared fields")]
    pub fn new(
        names: Vec<String>,
        columns: Vec<Vec<f64>>,
        treatment: u32,
        outcome: u32,
        split_seed: u64,
        discovery_seed: u64,
        transformation: Transformation,
        max_degree: u32,
        log_shift: Option<f64>,
    ) -> Self {
        Self {
            version: FORMAT_VERSION,
            names,
            columns,
            treatment,
            outcome,
            split_seed,
            discovery_seed,
            transformation,
            max_degree,
            log_shift,
            knowledge: None,
        }
    }

    /// This source with background knowledge asserted to discovery.
    #[must_use]
    pub fn with_knowledge(mut self, knowledge: Knowledge) -> Self {
        self.knowledge = Some(knowledge);
        self
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
                "parent_set_effect.source_not_canonical",
                "the source bytes are not the canonical encoding",
            ));
        }
        Ok(source)
    }

    fn validate(&self) -> Result<usize, IoError> {
        let k = self.columns.len();
        if self.version != FORMAT_VERSION || self.names.len() != k {
            return Err(malformed(
                "parent_set_effect.source_invalid",
                "unknown format version or names and columns of different counts",
            ));
        }
        if !(MIN_VARIABLES..=MAX_VARIABLES).contains(&k) {
            return Err(refused(
                "parent_set_effect.variable_count_not_measured",
                "only tables of ten to twenty searched variables were measured",
            ));
        }
        if self.max_degree == 0 {
            return Err(malformed(
                "parent_set_effect.source_invalid",
                "the declared maximum graph degree must be at least one",
            ));
        }
        if self.max_degree > SUPPORTED_MAX_DEGREE {
            return Err(refused(
                "parent_set_effect.degree_above_supported",
                &format!(
                    "the declared maximum graph degree is {} and the supported maximum is \
                     {SUPPORTED_MAX_DEGREE}",
                    self.max_degree
                ),
            ));
        }
        let rows = self.columns[0].len();
        if !(8..=MAX_ROWS).contains(&rows) || self.columns.iter().any(|c| c.len() != rows) {
            return Err(malformed(
                "parent_set_effect.source_invalid",
                "columns must have one length between 8 and 2^20 rows",
            ));
        }
        if self.columns.iter().flatten().any(|v| !v.is_finite()) {
            return Err(malformed(
                "parent_set_effect.source_invalid",
                "missing or non-finite values are not supported",
            ));
        }
        let (t, y) = (self.treatment as usize, self.outcome as usize);
        if t >= k || y >= k || t == y {
            return Err(malformed(
                "parent_set_effect.source_invalid",
                "treatment and outcome must be two distinct columns",
            ));
        }
        match (self.transformation, self.log_shift) {
            (Transformation::Linear, None) => {}
            (Transformation::LogLinear, Some(shift)) if shift.is_finite() && shift != 0.0 => {}
            _ => {
                return Err(malformed(
                    "parent_set_effect.source_invalid",
                    "a log-linear study declares a finite non-zero log shift of the treatment and \
                     a linear study declares none",
                ));
            }
        }
        if self.transformation == Transformation::LogLinear {
            for (name, column) in self.names.iter().zip(&self.columns) {
                let nonpositive = column.iter().filter(|v| **v <= 0.0).count();
                if nonpositive > 0 {
                    return Err(malformed(
                        "parent_set_effect.log_linear_input_not_positive",
                        &format!(
                            "column `{name}` has {nonpositive} non-positive value(s); a \
                             log-linear study requires every searched column to be positive"
                        ),
                    ));
                }
            }
        }
        if let Some(knowledge) = &self.knowledge {
            let known = |v: u32| (v as usize) < k;
            let links_ok = knowledge
                .required
                .iter()
                .chain(&knowledge.forbidden)
                .all(|&(a, b)| known(a) && known(b) && a != b);
            let tiers_ok = knowledge.tiers.iter().flatten().all(|&v| known(v));
            if !links_ok || !tiers_ok {
                return Err(malformed(
                    "parent_set_effect.knowledge_invalid",
                    "background knowledge names a column that is not searched, or a link from a \
                     column to itself",
                ));
            }
        }
        Ok(rows)
    }

    /// The table the analysis runs on: the declared transformation applied elementwise to
    /// every column, before the split.
    fn frame(&self) -> Result<TabularData, IoError> {
        let log = self.transformation == Transformation::LogLinear;
        let scaled: Vec<Vec<f64>> = self
            .columns
            .iter()
            .map(|column| column.iter().map(|&v| if log { v.ln() } else { v }).collect())
            .collect();
        TabularData::from_f64_columns(
            self.names.iter().map(String::as_str).zip(scaled.iter().map(Vec::as_slice)),
        )
        .map_err(|error| malformed("parent_set_effect.source_invalid", &error.to_string()))
    }
}

/// One edge of the discovered class that is not a definite directed edge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnresolvedPair {
    /// Column index of the smaller endpoint.
    pub a: u32,
    /// Column index of the larger endpoint.
    pub b: u32,
    /// Why the edge is unresolved (`snake_case`).
    pub reason: &'static str,
}

/// The treatment's neighbourhood in an accepted class, and the edges left open elsewhere.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TreatmentNeighbourhood {
    /// Column indices of the treatment's parents, ascending: the adjustment set.
    pub parents: Vec<u32>,
    /// Column indices of the treatment's directed children, ascending.
    pub children: Vec<u32>,
    /// Every directed edge of the class as `(cause, effect)`, ascending.
    pub directed: Vec<(u32, u32)>,
    /// Every edge not incident to the treatment that is not a definite directed edge.
    pub unresolved: Vec<UnresolvedPair>,
    /// Edges of the class in total.
    pub edges: usize,
}

/// The parent set of `treatment` in a discovered class, when it is the same in every member.
///
/// Accepted only when every edge incident to the treatment is directed. Edges not incident to
/// the treatment may be undirected; they are returned in [`TreatmentNeighbourhood::unresolved`]
/// and never completed.
///
/// # Errors
/// A structure that is not a completed partially directed graph over static searched
/// variables, an undirected edge incident to the treatment (named by column), or the outcome
/// oriented into the treatment.
pub fn treatment_parent_set(
    structure: &DiscoveredStructure,
    names: &[String],
    treatment: u32,
    outcome: u32,
) -> Result<TreatmentNeighbourhood, IoError> {
    if structure.class() != StructureClass::Cpdag {
        return Err(refused(
            "parent_set_effect.structure_class",
            "only a completed partially directed graph over static variables is accepted",
        ));
    }
    let variables = names.len();
    let mut parents = Vec::new();
    let mut children = Vec::new();
    let mut unresolved = Vec::new();
    let mut directed = Vec::new();
    let mut ambiguous = Vec::new();
    for edge in structure.edges() {
        let known = |v: u32| (v as usize) < variables;
        if edge.a.kind != 0
            || edge.b.kind != 0
            || !known(edge.a.variable)
            || !known(edge.b.variable)
        {
            return Err(refused(
                "parent_set_effect.structure_class",
                "a node is not a static searched variable",
            ));
        }
        let incident = edge.a.variable == treatment || edge.b.variable == treatment;
        if let Some(reason) = unresolved_reason(edge) {
            if incident {
                ambiguous.push((edge.a.variable, edge.b.variable, reason));
            } else {
                unresolved.push(UnresolvedPair { a: edge.a.variable, b: edge.b.variable, reason });
            }
            continue;
        }
        let (source, sink) = if edge.at_a == Mark::Tail {
            (edge.a.variable, edge.b.variable)
        } else {
            (edge.b.variable, edge.a.variable)
        };
        directed.push((source, sink));
        if sink == treatment {
            parents.push(source);
        } else if source == treatment {
            children.push(sink);
        }
    }
    if !ambiguous.is_empty() {
        let label = |v: u32| names.get(v as usize).map_or_else(|| format!("#{v}"), Clone::clone);
        let listed = ambiguous
            .iter()
            .map(|&(a, b, reason)| format!("{} - {} ({reason})", label(a), label(b)))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(refused(
            "parent_set_effect.treatment_parents_ambiguous",
            &format!(
                "the edges incident to the treatment `{}` are not all oriented: {listed}; the \
                 class holds graphs with different parent sets and no completion is chosen",
                label(treatment)
            ),
        ));
    }
    parents.sort_unstable();
    children.sort_unstable();
    directed.sort_unstable();
    if parents.contains(&outcome) {
        return Err(refused(
            "parent_set_effect.outcome_precedes_treatment",
            "the discovered structure orients the outcome into the treatment",
        ));
    }
    Ok(TreatmentNeighbourhood {
        parents,
        children,
        directed,
        unresolved,
        edges: structure.edges().len(),
    })
}

/// Why an edge is not a definite directed edge, or `None` when it is one.
fn unresolved_reason(edge: &StructureEdge) -> Option<&'static str> {
    if edge.is_directed() { None } else { Some(edge.unresolved_reason().unwrap_or("not_directed")) }
}

/// The effect on the declared scale of a regression coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct ReportedEffect {
    /// Scalar name: [`TOTAL_EFFECT`] or [`MEAN_RATIO`].
    pub name: &'static str,
    /// Point estimate.
    pub point: f64,
    /// Lower interval endpoint.
    pub lower: f64,
    /// Upper interval endpoint.
    pub upper: f64,
    /// Standard error: the coefficient's, or its delta-method image for a mean ratio.
    pub standard_error: f64,
}

/// The reported effect for a regression `coefficient` under the declared transformation.
fn reported_effect(
    transformation: Transformation,
    log_shift: Option<f64>,
    coefficient: &AdjustedEffect,
) -> ReportedEffect {
    match (transformation, log_shift) {
        (Transformation::LogLinear, Some(shift)) => {
            let image = |value: f64| (shift * value).exp();
            let (low, high) = (image(coefficient.lower), image(coefficient.upper));
            let point = image(coefficient.point);
            ReportedEffect {
                name: MEAN_RATIO,
                point,
                lower: low.min(high),
                upper: low.max(high),
                standard_error: point * shift.abs() * coefficient.standard_error,
            }
        }
        _ => ReportedEffect {
            name: TOTAL_EFFECT,
            point: coefficient.point,
            lower: coefficient.lower,
            upper: coefficient.upper,
            standard_error: coefficient.standard_error,
        },
    }
}

/// One executed parent-set sample-split effect with the identities of the discovery it rests on.
#[derive(Clone, Debug, PartialEq)]
pub struct ParentSetEffectRun {
    /// The regression on the declared scale: coefficient, interval and adjustment set.
    pub coefficient: AdjustedEffect,
    /// The reported effect on the estimand's scale.
    pub reported: ReportedEffect,
    /// The treatment's parents and children, and the edges left open elsewhere.
    pub neighbourhood: TreatmentNeighbourhood,
    /// Declared scale.
    pub transformation: Transformation,
    /// Declared maximum graph degree.
    pub max_degree: u32,
    /// Log-scale shift of the intervention, for a log-linear study.
    pub log_shift: Option<f64>,
    /// Which background-knowledge statements discovery was given (`required+tiers`), or empty.
    pub knowledge: String,
    /// Searched variables.
    pub variables: usize,
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

impl ParentSetEffectRun {
    /// Calibration basis of this construction at this variable count and declared degree, for
    /// the record of `topology` (one of [`TOPOLOGIES`]).
    #[must_use]
    pub fn keyed_basis_for(&self, topology: &str) -> CalibrationBasis {
        let functional = format!(
            "{}.variables_{}.max_degree_{}{}.topology_{topology}",
            self.transformation.scalar(),
            self.variables,
            self.max_degree,
            if self.knowledge.is_empty() {
                String::new()
            } else {
                format!(".knowledge_{}", self.knowledge)
            },
        );
        CalibrationBasis::new(
            [
                "AverageEffect",
                "Cpdag",
                PARENT_SET_STRUCTURE,
                "tabular",
                "Frequentist",
                self.transformation.estimator(),
                "analytic_se",
                "homoskedastic",
                SPLIT_DEPENDENCE,
                "",
                functional.as_str(),
            ]
            .map(Arc::<str>::from),
            self.coefficient.level,
            Arc::from("point"),
            self.coefficient.rows,
            None,
            None,
            0.0,
        )
    }

    /// The bases of every topology's record, in [`TOPOLOGIES`] order.
    #[must_use]
    pub fn keyed_bases(&self) -> Vec<CalibrationBasis> {
        governing_topologies(self.max_degree)
            .iter()
            .map(|topology| self.keyed_basis_for(topology))
            .collect()
    }

    /// The first topology's basis; attestation requires all of [`Self::keyed_bases`].
    #[must_use]
    pub fn keyed_basis(&self) -> CalibrationBasis {
        self.keyed_basis_for(governing_topologies(self.max_degree)[0])
    }
}

/// Execute the measured construction: transform as declared, split, discover on one part,
/// accept only a finished class whose treatment edges are all oriented, and estimate on the
/// other part.
///
/// # Errors
/// An invalid source or one outside the measured variable counts and degree, non-positive
/// log-linear inputs, a refused discovery, a search that exhausted a bound, met conflicting
/// orientation evidence or was cut by the parent limit, a treatment edge left undirected, or
/// the refusals of [`adjusted_effect`].
pub fn run(
    source: &ParentSetEffectSource,
    ctx: &ExecutionContext,
) -> Result<ParentSetEffectRun, IoError> {
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
    let degree = source.max_degree as usize;
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        variables,
        CiTestSpec::named(CI_TEST),
        source.discovery_seed,
    );
    contract.fdr = None;
    contract.constraints.alpha = ALPHA;
    contract.constraints.max_cond_size = degree;
    if let Some(knowledge) = &source.knowledge {
        let link = |&(cause, effect): &(u32, u32)| antecedent_discovery::LaggedLink {
            source: VariableId::from_raw(cause),
            source_lag: antecedent_core::Lag::CONTEMPORANEOUS,
            target: VariableId::from_raw(effect),
            target_lag: antecedent_core::Lag::CONTEMPORANEOUS,
        };
        let tiers: Vec<Arc<[VariableId]>> = knowledge
            .tiers
            .iter()
            .map(|tier| tier.iter().map(|&v| VariableId::from_raw(v)).collect())
            .collect();
        contract.constraints.required = knowledge.required.iter().map(link).collect();
        contract.constraints.forbidden = knowledge.forbidden.iter().map(link).collect();
        contract.constraints.tiers = tiers.into();
    }
    contract.constraints.max_parents = Some(degree);
    let mut run_ctx = ctx.clone();
    run_ctx.rng = RngFactory::from_seed(source.discovery_seed);
    let mut adapter = StaticSelectiveDiscovery::new();
    adapter
        .run(&contract, &split.discovery, &run_ctx)
        .map_err(|error| discovery_refusal(&error))?;
    let outcome = adapter.outcome().ok_or_else(|| {
        malformed("parent_set_effect.discovery_failed", "the discovery run retained no outcome")
    })?;
    if outcome.search_incomplete || outcome.orientation_conflict {
        return Err(refused(
            "parent_set_effect.search_not_finished",
            "the discovery search exhausted a bound or met conflicting orientation evidence",
        ));
    }
    if outcome.diagnostics.iter().any(|(code, _)| code.ends_with(PARENT_LIMIT_DIAGNOSTIC)) {
        return Err(refused(
            "parent_set_effect.parent_limit_bound",
            &format!(
                "the discovered skeleton exceeded the declared maximum graph degree {}; the \
                 limit removed adjacencies no test had separated, so the orientations rest on \
                 the limit",
                source.max_degree
            ),
        ));
    }
    let neighbourhood =
        treatment_parent_set(&outcome.structure, &source.names, source.treatment, source.outcome)?;
    let coefficient = adjusted_effect(
        &split.analysis,
        source.treatment,
        source.outcome,
        &neighbourhood.parents,
        LEVEL,
    )?;
    let reported = reported_effect(source.transformation, source.log_shift, &coefficient);
    Ok(ParentSetEffectRun {
        coefficient,
        reported,
        neighbourhood,
        transformation: source.transformation,
        max_degree: source.max_degree,
        log_shift: source.log_shift,
        knowledge: source.knowledge.as_ref().map(Knowledge::label).unwrap_or_default(),
        variables: k,
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
    let source = ParentSetEffectSource::from_bytes(source_bytes)?;
    let executed = run(&source, ctx)?;
    let premises = crate::digest_hex(
        IdentityDomain::Claim,
        &(
            PARENT_SET_EFFECT_ROUTE,
            executed.transformation.estimator(),
            CI_TEST,
            ALPHA,
            DISCOVERY_FRACTION,
            LEVEL,
            source.treatment,
            source.outcome,
            source.split_seed,
            source.discovery_seed,
            source.transformation.name(),
            source.max_degree,
            source.log_shift,
            &source.knowledge,
        ),
    )?;
    let data = crate::digest_hex(IdentityDomain::Claim, &(&source.names, &source.columns))?;
    let unresolved: Vec<(u32, u32, &str)> =
        executed.neighbourhood.unresolved.iter().map(|p| (p.a, p.b, p.reason)).collect();
    let scope = serde_json::json!({
        "validation_design":"sparse_connected_dags_ten_fifteen_twenty_variables_maximum_degree_four",
        "evidence_interpretation":"coverage of the true total effect (linear) or intervention mean ratio (log-linear) by the whole procedure (transformation, split, PC discovery, treatment-parent-set acceptance, parent adjustment) over the measured family, variable counts and sample sizes; never a statement about one dataset's graph",
        "declared_model_assumptions":["independent identically distributed rows","linear Gaussian structural equations on the declared scale (log-normal on the original scale for log-linear)","causal sufficiency","faithfulness with the published signal strength","maximum graph degree no larger than the declared degree","every edge incident to the treatment oriented by the equivalence class","positive treatment support under the prespecified intervention"],
        "sampling_assumptions_authenticated_from_rows":false,
        "structure":PARENT_SET_STRUCTURE,
        "algorithm":"pc","ci_test":CI_TEST,"alpha":ALPHA,
        "max_cond_size":executed.max_degree,"max_parents":executed.max_degree,
        "transformation":executed.transformation.name(),
        "log_shift":executed.log_shift,
        "discovery_fraction":DISCOVERY_FRACTION,
        "discovery_rows":executed.discovery_rows,"analysis_rows":executed.analysis_rows,
        "variables":executed.variables,
        "acceptance":ACCEPTANCE,
        "governing_topologies":governing_topologies(executed.max_degree),
        "background_knowledge":executed.knowledge,
        "topology_rule":"the cell is attested only when the record of every listed topology attests it (conservative intersection)",
        "adjustment_set":executed.coefficient.adjustment_set,
        "treatment_children":executed.neighbourhood.children,
        "unresolved_edges_not_incident_to_treatment":unresolved,
        "level":executed.coefficient.level,"interval":"ols_student_t",
        "receipt_identity":crate::digest_hex(IdentityDomain::Claim, &executed.receipt_identity)?,
        "structure_identity":crate::digest_hex(IdentityDomain::Claim, &executed.structure_identity)?,
        "reported_scalar":executed.reported.name
    });
    let scalar = |name: String, basis: CalibrationBasis| ScalarExecution {
        name,
        point: executed.reported.point,
        interval: (executed.reported.lower, executed.reported.upper),
        basis,
    };
    // A table does not announce its topology: the cell is licensed only when every topology's
    // record attests this construction, so each topology's record is resolved here and a
    // missing, stale or boundary record of any one refuses the whole.
    for topology in governing_topologies(executed.max_degree) {
        let name = format!("{}.topology_{topology}", executed.transformation.scalar());
        build_report(
            PARENT_SET_EFFECT_ROUTE,
            source_bytes,
            &premises,
            &data,
            scope.clone(),
            vec![scalar(name, executed.keyed_basis_for(topology))],
        )?;
    }
    authorize(
        PARENT_SET_EFFECT_ROUTE,
        source_bytes.to_vec(),
        &premises,
        &data,
        scope,
        vec![scalar(executed.transformation.scalar().into(), executed.keyed_basis())],
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
