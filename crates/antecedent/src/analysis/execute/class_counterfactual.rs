// SPDX-License-Identifier: MIT OR Apache-2.0
//! Counterfactual queries on a structure that latent confounding or an equivalence class leaves
//! open: an ADMG (or the contemporaneous projection of a temporal ADMG), a PAG and a CPDAG.
//!
//! Latent confounding breaks unit-level noise abduction: a latent common cause of the treatment
//! and the outcome is not recoverable from a unit's observed values, so the per-unit
//! counterfactual of a fitted structural model is not defined. Two readings remain defined.
//!
//! * A CPDAG assumes causal sufficiency, so each DAG completion of its equivalence class is an
//!   ordinary supplied-DAG study. The per-unit counterfactual route runs on every completion;
//!   completions that agree publish that value with its own uncertainty, completions that
//!   disagree leave the effect partially identified and the result publishes the
//!   enumeration-weighted mean with the identified set (the convention of the static CPDAG
//!   routes). Completion weights are enumeration weights, not posterior probabilities.
//! * An ADMG (and each ADMG a PAG's MAG completion stands for) states the pipeline query as the
//!   population-level cross-world quantity `E[Y_x | X = x'] - E[Y | X = x']`, where `x` is the
//!   query's active level and `x'` its control level: the mean effect, among the units whose
//!   observed treatment is the control level, of moving them to the active level. That
//!   quantity is the effect of treatment on the treated of the counterfactual-identification
//!   facade, so ID* decides it from the observational joint: the result is identified or the
//!   typed ID* refusal (with its hedge or conflicting-subscript witness) is returned. The
//!   identified functional is evaluated on the empirical cell table; its uncertainty is a
//!   row bootstrap (frequentist) or a Dirichlet-weighted Bayesian bootstrap (Bayesian) of that
//!   same functional, re-evaluated without re-identifying.
//!
//! The population-level reading is not a per-unit counterfactual and says so in the result.
//!
//! Several outcomes under one hard intervention are answered from the same graph class: each
//! outcome is identified (ID* per ADMG candidate, the per-unit route per DAG completion) and
//! evaluated on the same table, and the result publishes one contrast per outcome as a
//! counterfactual contrast set whose first outcome is the scalar estimate. A joint baseline of
//! simultaneous interventions is answered where each completion is a DAG (a CPDAG, whose
//! per-unit route replays every completion). The ID* decision procedure of the population-level
//! route observes one treatment per outcome event, so a joint baseline on an ADMG, a PAG or a
//! temporal ADMG is refused with that reason rather than reduced to one treatment of the set.

use super::{
    AcceptedGraph, Arc, CausalError, CausalExprArena, CausalPosterior, CausalQuery,
    ClassAtomOutcome, CounterfactualProcedure, DerivationMeta, DerivationTrace, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, DomainRef, EffectEstimate, EstimandMethod, EstimatorId,
    ExecutionContext, ExprNode, GcmSlot, GraphRef, IdentificationEnvelope,
    IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, Instant,
    OutcomeExprId, OverlapPolicy, PhysicalExecutionPlan, RefuteSuite, Strategy, StudyResult,
    TableView, TabularData, VariableId, elapsed_ns, provenance_ids,
};
use crate::counterfactual_id::{
    CounterfactualIdOptions, PreparedCounterfactualId, prepare_counterfactual_id,
};
use crate::gcm::{
    ContrastedOutcome, CounterfactualContrasts, ExogenousPosterior, NoiseInferenceKind,
};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CounterfactualEventQuery, CounterfactualQuery, RegimeId, Value,
};
use antecedent_estimate::{BootstrapAccounting, RowResampling};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, LawTolerance};
use antecedent_graph::{Admg, Cpdag, Dag, Pag};
use antecedent_identify::counterfactual_id::{
    COUNTERFACTUAL_ID_MAX_LEVELS, COUNTERFACTUAL_ID_MAX_VARIABLES, CounterfactualIdRefusal,
};
use antecedent_identify::{GraphIdentificationCase, ProbabilityMass};

/// Completions retained when the study names no cap (the generalized-adjustment default).
const COMPLETION_CAP: usize = 32;
/// Stream key of the frequentist replicates.
const FREQUENTIST_STREAM: u64 = 0x0CF1_C1A5;
/// Stream key of the Bayesian-bootstrap draws.
const BAYESIAN_STREAM: u64 = 0x0CF1_BA7E;
/// Relative tolerance below which two completions' values are one value.
const AGREEMENT_TOLERANCE: f64 = 1e-9;
/// Algorithm name written on the assumptions of the population-level route.
const ALGORITHM: &str = "counterfactual.id_star";
/// Backend name of the Bayesian-bootstrap posterior.
const POSTERIOR_BACKEND: &str = "counterfactual.id_star.bayesian_bootstrap";

/// A reason-coded refusal, spelled as the counterfactual-identification facade spells its own.
fn reasoned(code: &str, detail: &str, message: impl std::fmt::Display) -> CausalError {
    CausalError::Compile {
        message: format!("{}{code}: {detail}: {message}", crate::error::REASON_PREFIX),
    }
}

fn outside_contract(message: impl std::fmt::Display) -> CausalError {
    reasoned("route_not_supported", "counterfactual_id.query_outside_contract", message)
}

fn invalid_query(message: impl std::fmt::Display) -> CausalError {
    reasoned("invalid_argument", "counterfactual_id.invalid_query", message)
}

fn bounds(message: impl Into<String>) -> CausalError {
    let refusal = CounterfactualIdRefusal::bounds(message);
    CausalError::Compile { message: format!("{}{refusal}", crate::error::REASON_PREFIX) }
}

/// Say which outcome of a multi-outcome query a refusal belongs to (the text of a single outcome
/// is left as the facade wrote it).
fn name_outcome(error: CausalError, shape: &Shape, index: usize) -> CausalError {
    match error {
        CausalError::Compile { message } if shape.outcomes.len() > 1 => CausalError::Compile {
            message: format!("{message} (outcome {:?})", shape.outcomes[index]),
        },
        other => other,
    }
}

fn index_of(variable: VariableId) -> usize {
    variable.as_usize()
}

/// The treatment contrast and outcomes of a pipeline counterfactual query.
#[derive(Clone, Debug)]
struct Shape {
    treatment: VariableId,
    /// The first outcome: the one the result's scalar estimate describes.
    outcome: VariableId,
    /// Every outcome, in the query's order (the first is [`Self::outcome`]).
    outcomes: Vec<VariableId>,
    /// The query's active level `x`.
    active: f64,
    /// The query's control level `x'`: the treatment level the population is observed at.
    observed: f64,
}

/// The treatment and its two levels of a query that is not the one-outcome, one-`Set` form.
///
/// # Errors
///
/// A reason-coded refusal for a nested intervention, an intervention that is not a hard `Set`,
/// a control world on other variables than the active world, or a joint set of simultaneous
/// interventions, whose effect on the jointly treated the ID* decision does not state.
fn contrast_levels(query: &CounterfactualQuery) -> Result<(VariableId, f64, f64), CausalError> {
    if query.allow_nested {
        return Err(outside_contract(
            "a nested intervention replays an ordered sequence of worlds on each unit's abducted \
             noise, which a latent common cause makes undefined; the population-level route reads \
             one hard intervention against a hard control of the same variable",
        ));
    }
    let wrap = |error: CausalError| {
        outside_contract(format!(
            "{error}; the population-level route identifies hard interventions against a hard \
             control of the same variables"
        ))
    };
    let active = crate::gcm::world_levels(&query.interventions, false).map_err(wrap)?;
    let control = crate::gcm::world_levels(&query.control_world(), false).map_err(wrap)?;
    if active.iter().map(|(variable, _)| variable).ne(control.iter().map(|(variable, _)| variable))
    {
        return Err(outside_contract(
            "the control world must intervene on the same variables as the active world",
        ));
    }
    match (active.as_slice(), control.as_slice()) {
        (&[(treatment, active)], &[(_, control)]) => Ok((treatment, active, control)),
        ([], _) => Err(invalid_query("a counterfactual query names no intervention")),
        _ => Err(outside_contract(format!(
            "the query intervenes on {} variables at once ({:?}); the population-level route \
             decides P(Y_x = y, X = x') by ID*, whose licensed shape observes one treatment per \
             outcome event, so a joint baseline of simultaneous interventions on an Admg, a Pag \
             or a temporal Admg is refused rather than reduced to one treatment of the set. Next \
             attempt: state each treatment's contrast as its own query, or supply a Dag or a \
             Cpdag, which answer simultaneous interventions",
            active.len(),
            active.iter().map(|(variable, _)| *variable).collect::<Vec<_>>(),
        ))),
    }
}

/// Compile the pipeline query into the contrast the population-level route answers.
///
/// # Errors
///
/// A reason-coded refusal when the query is not one hard intervention against a hard control of
/// the same variable (on one or several distinct outcomes).
fn shape_of(query: &CounterfactualQuery) -> Result<Shape, CausalError> {
    query.validate()?;
    let (treatment, active, observed) = if query.outcomes.len() == 1
        && query.interventions.len() == 1
        && query.joint_control.is_empty()
        && !query.allow_nested
    {
        super::binary_cf_interventions(query).map_err(|error| {
            outside_contract(format!(
                "{error}; the population-level route identifies one hard intervention against a \
                 hard control of the same variable on one outcome"
            ))
        })?
    } else {
        contrast_levels(query)?
    };
    let outcomes: Vec<VariableId> = query.outcomes.to_vec();
    let outcome = *outcomes
        .first()
        .ok_or_else(|| invalid_query("a counterfactual query names no outcome"))?;
    for (index, candidate) in outcomes.iter().enumerate() {
        if *candidate == treatment {
            return Err(invalid_query("treatment and outcome must be distinct"));
        }
        if outcomes[..index].contains(candidate) {
            return Err(invalid_query("the outcomes of a counterfactual must be distinct"));
        }
    }
    if !active.is_finite()
        || !observed.is_finite()
        || (active + 0.0).to_bits() == (observed + 0.0).to_bits()
    {
        return Err(invalid_query(
            "the counterfactual and the control levels must be finite and distinct",
        ));
    }
    Ok(Shape { treatment, outcome, outcomes, active, observed })
}

/// The finite-discrete table of the graph's variables: levels, names and the cell of every row.
struct Table {
    names: Vec<String>,
    levels: Vec<Vec<f64>>,
    cells: Vec<usize>,
    cell_count: usize,
}

impl Table {
    /// Read the `order` variables of `data` (one per graph node, in node order) as complete
    /// finite-discrete columns.
    fn read(data: &TabularData, order: &[VariableId]) -> Result<Self, CausalError> {
        let variables = order.len();
        if variables == 0 || data.row_count() == 0 {
            return Err(invalid_query("a counterfactual table needs variables and rows"));
        }
        if variables > COUNTERFACTUAL_ID_MAX_VARIABLES {
            return Err(bounds(format!(
                "{variables} variables exceed the contract's {COUNTERFACTUAL_ID_MAX_VARIABLES}"
            )));
        }
        let mut names = Vec::with_capacity(variables);
        let mut levels: Vec<Vec<f64>> = Vec::with_capacity(variables);
        let mut columns: Vec<Vec<f64>> = Vec::with_capacity(variables);
        for (raw, &id) in order.iter().enumerate() {
            names.push(data.schema().get(id)?.name.to_string());
            let mut values = data.float64_values(id)?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err(invalid_query(format!(
                    "variable {raw} has a missing or non-finite value; the population-level \
                     route reads complete finite-discrete columns"
                )));
            }
            // One level, one spelling: -0.0 and 0.0 are the same level.
            for value in &mut values {
                *value += 0.0;
            }
            let mut distinct = values.clone();
            distinct.sort_by(f64::total_cmp);
            distinct.dedup();
            if distinct.len() > COUNTERFACTUAL_ID_MAX_LEVELS {
                return Err(bounds(format!(
                    "variable {raw} has {} levels; the contract covers at most \
                     {COUNTERFACTUAL_ID_MAX_LEVELS}",
                    distinct.len()
                )));
            }
            levels.push(distinct);
            columns.push(values);
        }
        let mut strides = vec![0_usize; variables];
        let mut cell_count = 1_usize;
        for index in (0..variables).rev() {
            strides[index] = cell_count;
            cell_count *= levels[index].len();
        }
        let mut cells = vec![0_usize; data.row_count()];
        for index in 0..variables {
            for (row, value) in columns[index].iter().enumerate() {
                let level = levels[index]
                    .binary_search_by(|probe| probe.total_cmp(value))
                    .map_err(|_| invalid_query("a value is not one of its column's levels"))?;
                cells[row] += level * strides[index];
            }
        }
        Ok(Self { names, levels, cells, cell_count })
    }

    fn axes(&self) -> Vec<DiscreteAxis> {
        self.levels
            .iter()
            .enumerate()
            .map(|(index, levels)| DiscreteAxis {
                variable: VariableId::from_raw(u32::try_from(index).unwrap_or(u32::MAX)),
                values: levels.iter().map(|level| Value::f64(*level)).collect::<Vec<_>>().into(),
            })
            .collect()
    }

    /// Cell counts of the rows weighted by integer `row_counts`.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "replicate row counts are non-negative whole numbers below the row count"
    )]
    fn counts(&self, row_counts: &[f64]) -> Vec<u64> {
        let mut counts = vec![0_u64; self.cell_count];
        for (&cell, &count) in self.cells.iter().zip(row_counts) {
            counts[cell] += count as u64;
        }
        counts
    }

    /// The empirical plug-in law of observed cell counts.
    #[allow(clippy::cast_precision_loss, reason = "cell counts are far below 2^53")]
    fn empirical_law(&self, counts: Vec<u64>) -> Result<ExactDiscreteLaw, CausalError> {
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return Err(invalid_query("the resampled table has no rows"));
        }
        let probabilities: Vec<f64> =
            counts.iter().map(|&count| count as f64 / total as f64).collect();
        ExactDiscreteLaw::try_empirical(
            "counterfactual_table",
            RegimeId::from_raw(0),
            [],
            self.axes(),
            probabilities,
            "empirical",
            LawTolerance::default(),
        )
        .and_then(|law| law.with_empirical_counts(counts))
        .map_err(invalid_query)
    }

    /// The law the Dirichlet-weighted rows put on the cells.
    fn weighted_law(&self, row_weights: &[f64]) -> Result<ExactDiscreteLaw, CausalError> {
        let mut mass = vec![0.0_f64; self.cell_count];
        for (&cell, &weight) in self.cells.iter().zip(row_weights) {
            mass[cell] += weight;
        }
        let total: f64 = mass.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return Err(invalid_query("the posterior draw put no mass on the table"));
        }
        for cell in &mut mass {
            *cell /= total;
        }
        ExactDiscreteLaw::try_new(
            "counterfactual_table",
            RegimeId::from_raw(0),
            [],
            self.axes(),
            mass,
            "bayesian_bootstrap",
            LawTolerance::default(),
        )
        .map_err(invalid_query)
    }
}

/// The event query of the facade for `outcome`: `P(Y_x = y | X = x')` at the outcome's first
/// level, whose published contrast `E[Y_x | X = x'] - E[Y | X = x']` is over every level.
fn event_query(
    shape: &Shape,
    outcome: VariableId,
    order: &[VariableId],
    table: &Table,
) -> Result<CounterfactualEventQuery, CausalError> {
    let node = |variable: VariableId| {
        order
            .iter()
            .position(|candidate| *candidate == variable)
            .and_then(|index| u32::try_from(index).ok())
            .map(VariableId::from_raw)
            .ok_or_else(|| invalid_query("a query variable is not a variable of the graph"))
    };
    let (treatment, outcome) = (node(shape.treatment)?, node(outcome)?);
    let level = table
        .levels
        .get(index_of(outcome))
        .and_then(|levels| levels.first().copied())
        .ok_or_else(|| invalid_query("the outcome is not a variable of the graph"))?;
    CounterfactualEventQuery::effect_on_treated(
        treatment,
        shape.active,
        shape.observed,
        outcome,
        level,
    )
    .map_err(invalid_query)
}

/// The variable of every node of a static structure, in node order.
fn node_variables(nodes: &[antecedent_core::NodeRef]) -> Result<Vec<VariableId>, CausalError> {
    nodes
        .iter()
        .map(|node| match node {
            antecedent_core::NodeRef::Static(variable) => Ok(*variable),
            _ => Err(outside_contract(
                "a counterfactual on a structure class reads static variables only",
            )),
        })
        .collect()
}

/// One identified candidate: a completion's candidate graph and its ID* derivation of each
/// outcome, in the query's order.
struct Case {
    prepared: Vec<PreparedCounterfactualId>,
    /// Share of its completion's unit weight (candidates with distinct functionals split it).
    share: f64,
}

/// The ADMG candidates of one completion, or why the completion could not be read.
enum Candidates {
    Open(Vec<Admg>),
    Refused(CausalError),
}

/// One completion of a PAG (or the single ADMG of an ADMG study) and the graphs it stands for.
struct Completion {
    graph: Option<Pag>,
    candidates: Candidates,
}

/// Bookkeeping row of the identification envelope, in envelope order.
struct Record {
    graph: Option<Pag>,
    case: Option<usize>,
    weight: f64,
    refusal: Option<String>,
}

/// The share-weighted mean over the cases of outcome `outcome`, from values laid out as
/// [`case_values`] lays them (case-major, `width` outcomes per case).
fn pool(values: &[f64], cases: &[Case], width: usize, outcome: usize) -> f64 {
    let total: f64 = cases.iter().map(|case| case.share).sum();
    cases
        .iter()
        .enumerate()
        .map(|(index, case)| values[index * width + outcome] * case.share)
        .sum::<f64>()
        / total
}

/// The values of `outcome` over the cases, from values laid out as [`case_values`] lays them.
fn outcome_column(values: &[f64], width: usize, outcome: usize) -> Vec<f64> {
    values.iter().skip(outcome).step_by(width).copied().collect()
}

/// Values of every case's identified functional on `law`: case-major, one value per outcome.
fn case_values(
    cases: &[Case],
    law: &ExactDiscreteLaw,
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, CausalError> {
    let mut values = Vec::with_capacity(cases.iter().map(|case| case.prepared.len()).sum());
    for case in cases {
        for prepared in &case.prepared {
            values.push(prepared.evaluate(law, ctx)?.effect.ok_or_else(|| {
                invalid_query("the outcome levels are not numeric; the contrast has no value")
            })?);
        }
    }
    Ok(values)
}

/// Whether every value is the first one, up to the agreement tolerance.
fn agree(values: &[f64]) -> bool {
    values.first().is_none_or(|first| {
        values
            .iter()
            .all(|value| (value - first).abs() <= AGREEMENT_TOLERANCE * (1.0 + first.abs()))
    })
}

fn assumptions() -> AssumptionSet {
    let mut set = AssumptionSet::default();
    let mut record = |assumption: Assumption, scope: AssumptionScope| {
        set.push(AssumptionRecord {
            assumption,
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from(ALGORITHM) },
            scope,
            status: AssumptionStatus::Declared,
        });
    };
    record(Assumption::CausalMarkov, AssumptionScope::Identification);
    record(Assumption::Consistency, AssumptionScope::Identification);
    record(Assumption::Positivity, AssumptionScope::Estimation);
    record(
        Assumption::Custom {
            id: Arc::from("counterfactual.population_level_cross_world"),
            description: Arc::from(
                "Under latent confounding the estimand is the population-level cross-world mean \
                 E[Y_x | X = x'] - E[Y | X = x']: the mean effect of moving the units observed at \
                 the control level x' to the active level x. It is identified by ID* from the \
                 observational joint and is not a per-unit counterfactual, because a latent \
                 common cause of the treatment and the outcome prevents abducting a unit's \
                 noise.",
            ),
        },
        AssumptionScope::Identification,
    );
    set
}

/// The identification result and estimand of one identified case: the first outcome's
/// functional is the estimand, and the derivation text names every outcome's ID* derivation.
fn identified_case(
    query: &CounterfactualQuery,
    shape: &Shape,
    prepared: &[PreparedCounterfactualId],
    required: &AssumptionSet,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut arena = CausalExprArena::new();
    let outcome = arena.intern_var_set([shape.outcome]);
    let do_treatment = arena.intern_intervention_set([shape.treatment]);
    let empty = arena.empty_var_set();
    let distribution =
        arena.intern_distribution(outcome, empty, do_treatment, DomainRef::Interventional);
    let functional = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(shape.outcome),
        distribution,
    });
    let derivation = prepared[0].derivation();
    let others = if prepared.len() > 1 {
        format!(
            "; the {} outcomes {:?} are each identified from the same graph by one ID* \
             derivation, and this functional names the first",
            prepared.len(),
            shape.outcomes,
        )
    } else {
        String::new()
    };
    arena.set_derivation(
        functional,
        DerivationMeta::rule(
            ALGORITHM,
            Some(Arc::from(format!(
                "ID* identification of P(Y_x = y, X = x') for treatment {:?} (active {}, \
                 control {}) and outcome {:?}: {} district term(s) over {} search operation(s); \
                 the functional carried here names the outcome and treatment, the derivation \
                 itself is the counterfactual-identification derivation{others}",
                shape.treatment,
                shape.active,
                shape.observed,
                shape.outcome,
                derivation.terms().len(),
                derivation.search.operations_consumed,
            ))),
        ),
    );
    let estimand = IdentifiedEstimand::new(
        EstimandMethod::GeneralId.as_str(),
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let identification =
        IdentificationResult::from_parts(antecedent_identify::IdentificationResultParts {
            status: IdentificationStatus::NonparametricallyIdentified,
            query: CausalQuery::Counterfactual(query.clone()),
            estimands: vec![estimand.clone()],
            arena,
            derivation: DerivationTrace::default(),
            required_assumptions: required.clone(),
            diagnostics: Vec::new(),
            performance: IdentificationPerformanceRecord::default(),
            hedge: None,
        });
    (identification, estimand)
}

/// The identification result of a case ID* refused.
fn refused_case(query: &CounterfactualQuery, message: &str) -> IdentificationResult {
    IdentificationResult::from_parts(antecedent_identify::IdentificationResultParts {
        status: IdentificationStatus::NotIdentified,
        query: CausalQuery::Counterfactual(query.clone()),
        estimands: Vec::new(),
        arena: CausalExprArena::new(),
        derivation: DerivationTrace::default(),
        required_assumptions: AssumptionSet::default(),
        diagnostics: vec![Diagnostic::new(
            "identify.counterfactual_id.refused",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            message,
        )],
        performance: IdentificationPerformanceRecord::default(),
        hedge: None,
    })
}

/// Posterior of columns of Bayesian-bootstrap draws, the first being the primary effect.
fn posterior_of(
    columns: Vec<(String, Vec<f64>)>,
    required: &AssumptionSet,
) -> Result<CausalPosterior, CausalError> {
    let mut posterior = super::attribution_path::attribution_posterior(
        columns,
        AssumptionSet::default(),
        IdentificationStatus::NonparametricallyIdentified,
        POSTERIOR_BACKEND,
        true,
    )?;
    posterior.assumptions = required.clone();
    Ok(posterior)
}

/// The posterior column of outcome `outcome`: the established name for one outcome, the
/// outcome's raw id beside it for several.
fn column_name(shape: &Shape, outcome: usize) -> String {
    if shape.outcomes.len() == 1 {
        "counterfactual_effect".to_owned()
    } else {
        format!("counterfactual_effect[{}]", shape.outcomes[outcome].raw())
    }
}

// Admission covers the nested matrix headers and values, plus the peak row
// weights and block-weight vector. The lease travels with the matrix.
fn admit_counterfactual_resamples(
    ctx: &ExecutionContext,
    draws: usize,
    width: usize,
    rows: usize,
) -> Result<antecedent_core::execution::AdmissionLease, CausalError> {
    let bytes = draws
        .checked_mul(width)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
        .and_then(|n| {
            draws
                .checked_mul(std::mem::size_of::<Vec<f64>>())
                .and_then(|headers| n.checked_add(headers))
        })
        .and_then(|n| {
            rows.checked_mul(2 * std::mem::size_of::<f64>())
                .and_then(|scratch| n.checked_add(scratch))
        })
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| CausalError::Resource {
            message: "counterfactual resample size overflow".into(),
        })?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: super::super::stage::STAGE_UNCERTAINTY }
        }
        other => {
            CausalError::Resource { message: format!("counterfactual resample admission: {other}") }
        }
    })
}

// Fields drop in declaration order: release the matrix before its admission.
struct AdmittedCounterfactualMatrix {
    matrix: Vec<Vec<f64>>,
    _lease: antecedent_core::execution::AdmissionLease,
}

/// What the resampling of the identified functional produced.
enum Resampled {
    /// Point only.
    None,
    /// Bootstrap replicates: one row of case values per successful replicate.
    Replicates { held: AdmittedCounterfactualMatrix, failed: u32, requested: u32 },
    /// Bayesian-bootstrap draws: one row of case values per draw.
    Draws { held: AdmittedCounterfactualMatrix },
}

/// Refuse a validation suite the population-level route has no refuter for.
fn ensure_validation_none(refute: RefuteSuite) -> Result<(), CausalError> {
    if refute != RefuteSuite::None {
        return Err(CausalError::Unsupported {
            message: "a counterfactual on an Admg, Pag, Cpdag or TemporalAdmg runs at validation \
                      none; cheap/full remain Dag-only",
        });
    }
    Ok(())
}

impl super::Study {
    /// Execute a counterfactual query on a structure other than a supplied DAG.
    pub(super) fn execute_counterfactual_class(
        &self,
        data: &TabularData,
        query: &CounterfactualQuery,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        ensure_validation_none(self.settings.refute)?;
        match self.graph.graph_ref() {
            GraphRef::Cpdag(cpdag) => {
                self.execute_counterfactual_cpdag(data, cpdag, query, physical, ctx)
            }
            GraphRef::Admg(_) | GraphRef::TemporalAdmg(_) | GraphRef::Pag(_) => {
                self.execute_counterfactual_latent(data, query, physical, ctx)
            }
            _ => Err(CausalError::Unsupported {
                message: "a counterfactual on a structure class runs on an Admg, TemporalAdmg, \
                          Pag or Cpdag",
            }),
        }
    }

    /// Completions retained: the study's `max_completions`, else the default bound.
    fn completion_cap(&self) -> usize {
        self.declared.max_completions.unwrap_or(COMPLETION_CAP)
    }

    /// The completions (or the one ADMG) a latent-confounded structure stands for.
    fn latent_completions(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<Completion>, Vec<VariableId>, bool), CausalError> {
        match self.graph.graph_ref() {
            GraphRef::Pag(pag) => {
                let limits = antecedent_core::SearchLimits {
                    operations: antecedent_graph::DEFAULT_COMPLETION_OPERATIONS,
                    depth: usize::MAX,
                };
                let mut sampler = antecedent_graph::CompletionSampler::with_budget(
                    pag,
                    self.completion_cap(),
                    limits,
                    ctx,
                )?;
                let mags: Vec<Pag> = sampler.by_ref().map(|completion| completion.graph).collect();
                crate::strategy_table::check_class_search_stop(sampler.search_stop())?;
                if sampler.validation_report().ambiguous_global_class {
                    return Err(outside_contract(
                        "the circle endpoints admit completions of more than one Markov \
                         equivalence class, so the structure is not the PAG of one class; \
                         rerun discovery or declare the graph",
                    ));
                }
                if mags.is_empty() {
                    return Err(outside_contract(
                        "the PAG has no maximal-ancestral-graph completion within the search \
                         bound; orient edges with background knowledge or declare the graph",
                    ));
                }
                let incomplete = sampler.hit_cap() || sampler.class_audit_incomplete();
                let identifier = crate::strategy_table::class_identifier_in_context(Some(ctx));
                let completions = mags
                    .into_iter()
                    .map(|mag| {
                        let candidates = match identifier.admg_candidates_of_mag(&mag) {
                            Ok(found) if found.bounded => Candidates::Refused(outside_contract(
                                "the ADMGs that share this MAG's m-separation model exceed the \
                                 enumeration bound, so the counterfactual was not decided on \
                                 every one of them",
                            )),
                            Ok(found) => Candidates::Open(found.members),
                            Err(error) if error.is_budget_or_cancel() => {
                                return Err(CausalError::from(error));
                            }
                            Err(error) => Candidates::Refused(outside_contract(format!(
                                "this MAG completion has no ADMG reading ({error})"
                            ))),
                        };
                        Ok(Completion { graph: Some(mag), candidates })
                    })
                    .collect::<Result<Vec<_>, CausalError>>()?;
                Ok((completions, node_variables(pag.nodes())?, incomplete))
            }
            other => {
                let admg = other.projected_admg()?.ok_or_else(|| CausalError::Compile {
                    message: "a counterfactual on a mixed graph requires an Admg".into(),
                })?;
                let order = node_variables(admg.nodes())?;
                Ok((
                    vec![Completion { graph: None, candidates: Candidates::Open(vec![admg]) }],
                    order,
                    false,
                ))
            }
        }
    }

    /// Resample the identified functionals: a row bootstrap or a Dirichlet-weighted bootstrap.
    fn resample_counterfactual(
        &self,
        table: &Table,
        cases: &[Case],
        ctx: &ExecutionContext,
    ) -> Result<Resampled, CausalError> {
        let rows = table.cells.len();
        let (resampling, _) = self.functional_row_resampling()?;
        let block = match resampling {
            RowResampling::Rows => None,
            RowResampling::SeriesBlocks | RowResampling::WindowBlocks { .. } => {
                Some(antecedent_data::circular_block_length(1, rows))
            }
        };
        let cancelled = || CausalError::Cancelled { stage: super::super::stage::STAGE_UNCERTAINTY };
        match CounterfactualProcedure::for_inference(&self.settings.inference)? {
            CounterfactualProcedure::DirichletRowWeights { draws } => {
                let width = cases
                    .iter()
                    .try_fold(0usize, |n, case| n.checked_add(case.prepared.len()))
                    .ok_or_else(|| CausalError::Resource {
                        message: "counterfactual resample width overflow".into(),
                    })?;
                let lease = admit_counterfactual_resamples(ctx, draws, width, rows)?;
                let mut rng =
                    ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, BAYESIAN_STREAM);
                let mut matrix = Vec::with_capacity(draws);
                for _ in 0..draws {
                    if ctx.cancellation.is_cancelled() {
                        return Err(cancelled());
                    }
                    let weights = match block {
                        None => {
                            antecedent_kernels::bayesian_bootstrap_weights(&mut rng, rows, false)
                        }
                        Some(length) => super::temporal_attribution::block_dirichlet_row_weights(
                            rows, length, &mut rng,
                        ),
                    };
                    matrix.push(case_values(cases, &table.weighted_law(&weights)?, ctx)?);
                }
                Ok(Resampled::Draws {
                    held: AdmittedCounterfactualMatrix { matrix, _lease: lease },
                })
            }
            _ if self.settings.bootstrap_replicates == 0 => Ok(Resampled::None),
            _ => {
                if block.is_none() {
                    super::gcm_bootstrap::require_subject_resampling(
                        self.declared.sampling_design,
                    )?;
                }
                let requested = self.settings.bootstrap_replicates;
                let width = cases
                    .iter()
                    .try_fold(0usize, |n, case| n.checked_add(case.prepared.len()))
                    .ok_or_else(|| CausalError::Resource {
                        message: "counterfactual resample width overflow".into(),
                    })?;
                let draws = usize::try_from(requested).map_err(|_| CausalError::Resource {
                    message: "counterfactual replicate count overflow".into(),
                })?;
                let lease = admit_counterfactual_resamples(ctx, draws, width, rows)?;
                let mut matrix = Vec::with_capacity(draws);
                let mut failed = 0_u32;
                for replicate in 0..requested {
                    if ctx.cancellation.is_cancelled() {
                        return Err(cancelled());
                    }
                    let mut rng =
                        super::gcm_bootstrap::replicate_stream(ctx, FREQUENTIST_STREAM, replicate);
                    let counts = match block {
                        None => super::gcm_bootstrap::bootstrap_counts(&mut rng, rows),
                        Some(length) => super::temporal_attribution::block_bootstrap_counts(
                            rows, length, &mut rng,
                        ),
                    };
                    let values = table
                        .empirical_law(table.counts(&counts))
                        .and_then(|law| case_values(cases, &law, ctx));
                    match values {
                        Ok(values) => matrix.push(values),
                        Err(_) if ctx.cancellation.is_cancelled() => return Err(cancelled()),
                        Err(_) => failed += 1,
                    }
                }
                Ok(Resampled::Replicates {
                    held: AdmittedCounterfactualMatrix { matrix, _lease: lease },
                    failed,
                    requested,
                })
            }
        }
    }

    /// The population-level route: ID* on every ADMG the structure stands for.
    #[allow(
        clippy::too_many_lines,
        reason = "one execution: identify each candidate, evaluate, resample, then finish the identified result"
    )]
    fn execute_counterfactual_latent(
        &self,
        data: &TabularData,
        query: &CounterfactualQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let shape = shape_of(query)?;
        let width = shape.outcomes.len();
        let (completions, order, incomplete) = self.latent_completions(ctx)?;
        let completions_retained = completions.len();
        let variables = order.len();
        let table = Table::read(data, &order)?;
        let events = shape
            .outcomes
            .iter()
            .map(|&outcome| event_query(&shape, outcome, &order, &table))
            .collect::<Result<Vec<_>, CausalError>>()?;
        let required = assumptions();

        // Identify every candidate of every completion. A completion any of whose candidates
        // ID* refuses is unidentified: no functional is correct for every graph it stands for.
        let mut cases: Vec<Case> = Vec::new();
        let mut records: Vec<Record> = Vec::new();
        let mut first_refusal: Option<CausalError> = None;
        for completion in completions {
            let found = match completion.candidates {
                Candidates::Refused(error) => Err(error),
                Candidates::Open(admgs) => {
                    let mut found: Vec<Vec<PreparedCounterfactualId>> = Vec::new();
                    let mut forms: Vec<String> = Vec::new();
                    let mut refusal = None;
                    for admg in &admgs {
                        if ctx.cancellation.is_cancelled() {
                            return Err(CausalError::Cancelled {
                                stage: super::super::stage::STAGE_ESTIMATE_POINT,
                            });
                        }
                        // One ID* derivation per outcome on this candidate: a candidate any of
                        // whose outcomes ID* refuses is refused as a whole.
                        let mut derived = Vec::with_capacity(width);
                        for (index, event) in events.iter().enumerate() {
                            match prepare_counterfactual_id(
                                admg.clone(),
                                table.names.clone(),
                                table.levels.clone(),
                                event,
                                CounterfactualIdOptions::default(),
                                ctx,
                            ) {
                                Ok(prepared) => derived.push(prepared),
                                Err(error) => {
                                    refusal = Some(name_outcome(error, &shape, index));
                                    break;
                                }
                            }
                        }
                        if refusal.is_some() {
                            break;
                        }
                        let form = derived
                            .iter()
                            .map(|prepared| prepared.derivation().canonical_text())
                            .collect::<Vec<_>>()
                            .join("\n");
                        if !forms.contains(&form) {
                            forms.push(form);
                            found.push(derived);
                        }
                    }
                    match refusal {
                        Some(error) => Err(error),
                        None if found.is_empty() => Err(outside_contract(
                            "the completion stands for no ADMG, so there is nothing to identify",
                        )),
                        None => Ok(found),
                    }
                }
            };
            match found {
                Ok(found) => {
                    #[allow(
                        clippy::cast_precision_loss,
                        reason = "a handful of candidate functionals"
                    )]
                    let share = 1.0 / found.len() as f64;
                    for prepared in found {
                        records.push(Record {
                            graph: completion.graph.clone(),
                            case: Some(cases.len()),
                            weight: share,
                            refusal: None,
                        });
                        cases.push(Case { prepared, share });
                    }
                }
                Err(error) => {
                    records.push(Record {
                        graph: completion.graph.clone(),
                        case: None,
                        weight: 1.0,
                        refusal: Some(error.to_string()),
                    });
                    first_refusal.get_or_insert(error);
                }
            }
        }
        if cases.is_empty() {
            return Err(first_refusal.unwrap_or_else(|| {
                outside_contract("no completion of the structure could be identified")
            }));
        }

        // The point: the identified functionals on the observed cell counts.
        let point_law = table.empirical_law(table.counts(&vec![1.0; table.cells.len()]))?;
        let values = case_values(&cases, &point_law, ctx)?;
        let columns: Vec<Vec<f64>> =
            (0..width).map(|outcome| outcome_column(&values, width, outcome)).collect();
        let pooled = pool(&values, &cases, width, 0);
        let unidentified = records.iter().any(|record| record.case.is_none());
        let agreeing = columns.iter().all(|column| agree(column));
        let resampled = self.resample_counterfactual(&table, &cases, ctx)?;

        // Identification: one case's own result for an ADMG, the completion envelope for a PAG.
        let case_results: Vec<(IdentificationResult, IdentifiedEstimand)> = cases
            .iter()
            .map(|case| identified_case(query, &shape, &case.prepared, &required))
            .collect();
        let envelope: Option<IdentificationEnvelope<Pag>> = records
            .iter()
            .map(|record| {
                let graph = record.graph.clone()?;
                let result = match record.case {
                    Some(case) => case_results[case].0.clone(),
                    None => refused_case(query, record.refusal.as_deref().unwrap_or("refused")),
                };
                Some(GraphIdentificationCase {
                    graph,
                    result,
                    weight: ProbabilityMass(record.weight),
                })
            })
            .collect::<Option<Vec<_>>>()
            .map(IdentificationEnvelope::from_cases);
        let identification = match &envelope {
            Some(envelope) => super::envelope_to_identification_result_for(
                envelope,
                CausalQuery::Counterfactual(query.clone()),
            ),
            None => case_results[0].0.clone(),
        };
        let estimand =
            identification.estimands.first().cloned().ok_or_else(|| CausalError::Compile {
                message: "counterfactual identification produced no estimand".into(),
            })?;

        // Structural uncertainty across completions and candidates.
        let outcomes: Vec<ClassAtomOutcome> = records
            .iter()
            .map(|record| match record.case {
                Some(case) => ClassAtomOutcome::Evaluated(columns[0][case]),
                None => ClassAtomOutcome::NotEvaluated,
            })
            .collect();
        let mut mixture =
            envelope.as_ref().filter(|_| !agreeing || unidentified).and_then(|envelope| {
                super::pag_path::static_class_structural_mixture(envelope, &outcomes)
            });
        let publishes_own_uncertainty = agreeing && !unidentified;

        // Sampling uncertainty of the identified functional.
        let mut estimate = EffectEstimate::new(
            pooled,
            f64::NAN,
            required.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        let mut diagnostics = vec![population_level_diagnostic(&shape, cases.len())];
        let mut posterior: Option<CausalPosterior> = None;
        let mut replicates_ok = None;
        let mut n_draws = None;
        // Each outcome's published mean and standard error: the pooled point and no standard
        // error until a resampling forms one.
        let mut means: Vec<f64> =
            (0..width).map(|outcome| pool(&values, &cases, width, outcome)).collect();
        let mut standard_errors: Vec<Option<f64>> = vec![None; width];
        match resampled {
            Resampled::None => diagnostics.push(Diagnostic::new(
                "estimate.counterfactual.uncertainty_unavailable",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "the population-level counterfactual is a point value of the identified \
                 functional on the empirical cell table; request bootstrap_replicates for a \
                 resampled standard error, or Bayesian inference for a Dirichlet-weighted \
                 bootstrap posterior",
            )),
            Resampled::Replicates { held, failed, requested } => {
                let matrix = &held.matrix;
                let ok = u32::try_from(matrix.len()).unwrap_or(u32::MAX);
                replicates_ok = Some(ok);
                if publishes_own_uncertainty {
                    let pooled_rows: Vec<f64> =
                        matrix.iter().map(|row| pool(row, &cases, width, 0)).collect();
                    estimate.uncertainty.bootstrap =
                        super::gcm_bootstrap::accounting(&pooled_rows, failed, requested);
                    standard_errors[0] = estimate.uncertainty.bootstrap.se;
                    for (outcome, standard_error) in standard_errors.iter_mut().enumerate().skip(1)
                    {
                        let rows: Vec<f64> =
                            matrix.iter().map(|row| pool(row, &cases, width, outcome)).collect();
                        *standard_error =
                            super::gcm_bootstrap::accounting(&rows, failed, requested).se;
                    }
                } else {
                    estimate.uncertainty.bootstrap = BootstrapAccounting {
                        se: None,
                        replicates_ok: Some(ok),
                        replicates_failed: Some(failed),
                        cancelled: false,
                        early_stopped: false,
                    };
                    diagnostics
                        .extend(super::envelope_se_omission_diagnostic(cases.len(), f64::NAN));
                    if let Some(mixture) = mixture.as_mut() {
                        let primary: Vec<Vec<f64>> =
                            matrix.iter().map(|row| outcome_column(row, width, 0)).collect();
                        mixture.identified_set_interval =
                            antecedent_estimate::imbens_manski_shared_replicates(
                                &columns[0],
                                &primary,
                                1.0,
                                table.cells.len(),
                                0.95,
                            );
                    }
                }
                diagnostics.push(
                    Diagnostic::new(
                        "estimate.counterfactual.bootstrap",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        format!(
                            "row bootstrap of the identified functional: each replicate \
                             resamples the table's rows and re-evaluates the ID* derivation on \
                             the resampled cell counts without re-identifying; \
                             replicates_ok={ok}, replicates_failed={failed} (a replicate whose \
                             conditioning event is empty is kept as failed mass). The standard \
                             error is the sample standard deviation of the successful \
                             replicates; no coverage record names this construction."
                        ),
                    )
                    .with_fields([
                        ("replicates_ok", ok.to_string()),
                        ("replicates_failed", failed.to_string()),
                        ("calibration", "unmeasured".to_owned()),
                    ]),
                );
            }
            Resampled::Draws { held } => {
                let matrix = &held.matrix;
                n_draws = Some(u32::try_from(matrix.len()).unwrap_or(u32::MAX));
                if publishes_own_uncertainty {
                    let named: Vec<(String, Vec<f64>)> = (0..width)
                        .map(|outcome| {
                            (
                                column_name(&shape, outcome),
                                matrix
                                    .iter()
                                    .map(|row| pool(row, &cases, width, outcome))
                                    .collect(),
                            )
                        })
                        .collect();
                    let draws = posterior_of(named, &required)?;
                    let index = draws.effect_column().ok_or_else(|| CausalError::Compile {
                        message: "counterfactual posterior missing effect column".into(),
                    })?;
                    estimate = EffectEstimate::new(
                        draws.summaries.mean[index],
                        draws.summaries.sd[index],
                        draws.assumptions.clone(),
                        OverlapPolicy::ExplicitOverride,
                    );
                    estimate.uncertainty.analytic.se = draws.summaries.sd[index];
                    for (outcome, (mean, standard_error)) in
                        means.iter_mut().zip(standard_errors.iter_mut()).enumerate()
                    {
                        *mean = draws.summaries.mean[outcome];
                        *standard_error = Some(draws.summaries.sd[outcome]);
                    }
                    posterior = Some(draws);
                } else {
                    diagnostics
                        .extend(super::envelope_se_omission_diagnostic(cases.len(), f64::NAN));
                    if let Some(mixture) = mixture.as_mut() {
                        for (atom, record) in mixture.atoms.iter_mut().zip(&records) {
                            if let Some(case) = record.case {
                                let column: Vec<f64> =
                                    matrix.iter().map(|row| row[case * width]).collect();
                                atom.posterior = Some(posterior_of(
                                    vec![(column_name(&shape, 0), column)],
                                    &required,
                                )?);
                            }
                        }
                    }
                }
                diagnostics.push(Diagnostic::new(
                    "estimate.counterfactual.bayesian_bootstrap",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "Dirichlet(1,...,1) Bayesian bootstrap over the empirical row law: each draw \
                     reweights the table's rows and evaluates the identified ID* functional on \
                     the weighted cell law without re-identifying. The posterior is \
                     conditional on the supplied structure and carries sampling uncertainty \
                     only; it does not address misspecification of the graph.",
                ));
            }
        }
        if let Some(envelope) = &envelope {
            diagnostics.push(super::pag_path::pag_envelope_diagnostic(envelope));
            if let Some(set) = mixture.as_ref().and_then(|mixture| {
                super::pag_path::static_class_identified_set_diagnostic(mixture, "pag")
            }) {
                diagnostics.push(set);
            }
        }
        if incomplete {
            diagnostics.push(Diagnostic::new(
                "estimate.counterfactual.completions_incomplete",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "the MAG completion search stopped at its cap or budget before it classified \
                     every endpoint assignment; the result covers the {} retained completions \
                     only (cap {}, {}), not the whole equivalence class; raise max_completions \
                     to retain more",
                    completions_retained,
                    self.completion_cap(),
                    if self.declared.max_completions.is_some() {
                        "set by the study"
                    } else {
                        "the default"
                    },
                ),
            ));
        }
        if let Some(derivation) = cases.first().map(|case| case.prepared[0].derivation()) {
            diagnostics.push(Diagnostic::new(
                "estimate.counterfactual.id_star",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "identified by ID* (Shpitser and Pearl) from the observational joint over \
                     {variables} finite-discrete variables; {} identified functional(s), the \
                     first with {} district term(s) after {} search operation(s). A refusal of \
                     ID* is not a proof of non-identifiability.",
                    cases.len(),
                    derivation.terms().len(),
                    derivation.search.operations_consumed,
                ),
            ));
        }

        if width > 1 {
            diagnostics.push(outcomes_diagnostic(
                &shape.outcomes,
                &columns,
                publishes_own_uncertainty,
            ));
        }

        // Several outcomes publish one contrast each. The contrast is the population-level mean
        // effect, so it carries no per-unit effects, support flags or noise abduction.
        let contrasts = (width > 1).then(|| {
            Box::new(CounterfactualContrasts {
                treatments: Arc::from([shape.treatment]),
                contrasts: shape
                    .outcomes
                    .iter()
                    .enumerate()
                    .map(|(index, &outcome)| ContrastedOutcome {
                        outcome,
                        unit_effects: Arc::from([]),
                        mean_ite: means[index],
                        standard_error: standard_errors[index],
                        unit_effect_intervals: None,
                        unit_extrapolative: None,
                        unit_effects_homogeneous: false,
                    })
                    .collect(),
                noise_inference: NoiseInferenceKind::AssumedNoise,
                exogenous: unabducted_exogenous(),
            })
        });

        let certificate = envelope.map(|envelope| crate::Identification::Envelope {
            envelope,
            strategy: IdentifierId::GeneralId,
            structure_version: self.graph.version(),
        });
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(IdentifierId::GeneralId, EstimatorId::FunctionalEffect),
                shape.treatment,
                shape.outcome,
            )
            .extra_diagnostics(diagnostics)
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(replicates_ok)
            .extras(IdentifiedExecuteExtras {
                certificate,
                n_draws,
                posterior,
                gcm: contrasts.map(GcmSlot::CounterfactualContrasts),
                structural_response: mixture,
                identify_provenance: Some(provenance_ids(
                    "identify.counterfactual_id",
                    "identify.counterfactual_id",
                )),
                estimate_provenance: Some(provenance_ids(
                    "counterfactual.id_star",
                    "counterfactual.id_star",
                )),
                ..Default::default()
            }),
        );
        // The functional carried by the identification is a carrier for the derivation, not a
        // functional with free variables the plug-in averages over.
        result.diagnostics.retain(|diagnostic| {
            diagnostic.code.as_ref() != "estimate.functional.free_variables_averaged"
        });
        Ok(result)
    }

    /// The per-unit counterfactual on every DAG completion of a CPDAG.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        reason = "one execution: run each completion, compare, then publish the point or the identified set; the completion count is far below 2^53"
    )]
    fn execute_counterfactual_cpdag(
        &self,
        data: &TabularData,
        cpdag: &Cpdag,
        query: &CounterfactualQuery,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let limits = antecedent_core::SearchLimits {
            operations: antecedent_graph::DEFAULT_COMPLETION_OPERATIONS,
            depth: usize::MAX,
        };
        let mut sampler = antecedent_graph::CpdagCompletionSampler::with_budget(
            cpdag.clone(),
            self.completion_cap(),
            limits,
            ctx,
        )?;
        let dags: Vec<Dag> = sampler.by_ref().map(|completion| completion.graph).collect();
        crate::strategy_table::check_class_search_stop(sampler.search_stop())?;
        let capped = sampler.hit_cap();
        if dags.is_empty() {
            return Err(CausalError::Compile {
                message: "the CPDAG has no DAG completion within the search bound".into(),
            });
        }
        super::report_identify_compute(ctx);
        let mut results = Vec::with_capacity(dags.len());
        for dag in &dags {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let mut completion = self.clone();
            completion.graph = Arc::new(AcceptedGraph::from(dag.clone()));
            completion.clear_identification_where(|cache| {
                matches!(
                    cache,
                    crate::analysis::IdentificationCache::Static(_)
                        | crate::analysis::IdentificationCache::Cpdag(_)
                )
            });
            results.push(completion.execute_counterfactual(data, dag, query, physical, ctx)?);
        }
        // One column of completion values per outcome: a contrast set carries each outcome's
        // mean effect, the single form its one scalar.
        let width = results[0].counterfactual_contrasts().map_or(1, |set| set.contrasts.len());
        let columns: Vec<Vec<f64>> = (0..width)
            .map(|outcome| {
                results
                    .iter()
                    .map(|result| match result.counterfactual_contrasts() {
                        Some(set) => set.contrasts.get(outcome).map_or(f64::NAN, |c| c.mean_ite),
                        None => result.estimate.as_effect().map_or(f64::NAN, |effect| effect.ate),
                    })
                    .collect()
            })
            .collect();
        let values: Vec<f64> = columns[0].clone();
        if columns.iter().flatten().any(|value| !value.is_finite()) {
            return Err(CausalError::Compile {
                message: "a DAG completion produced no finite counterfactual effect".into(),
            });
        }
        let cases: Vec<GraphIdentificationCase<Dag>> = dags
            .iter()
            .zip(&results)
            .map(|(dag, result)| GraphIdentificationCase {
                graph: dag.clone(),
                result: result.identification.clone(),
                weight: ProbabilityMass(1.0),
            })
            .collect();
        let posteriors: Vec<Option<CausalPosterior>> =
            results.iter().map(|result| result.posterior.clone()).collect();
        let envelope = IdentificationEnvelope::from_cases(cases);
        let agreeing = columns.iter().all(|column| agree(column));
        let outcomes: Vec<ClassAtomOutcome> =
            values.iter().map(|&value| ClassAtomOutcome::Evaluated(value)).collect();
        let mut mixture = if agreeing {
            None
        } else {
            super::pag_path::static_class_structural_mixture(&envelope, &outcomes)
        };

        let mut diagnostics = vec![super::pag_path::cpdag_envelope_diagnostic(&envelope)];
        let identification = super::envelope_to_identification_result_for(
            &envelope,
            CausalQuery::Counterfactual(query.clone()),
        );
        let mut result = results.remove(0);
        if agreeing {
            diagnostics.push(Diagnostic::new(
                "estimate.counterfactual.completions_agree",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "the per-unit counterfactual agrees across the {} DAG completion(s) of the \
                     equivalence class; the value and uncertainty are the first completion's own \
                     fit (completions are Markov equivalent, so no latent confounding is \
                     assumed and noise abduction is defined in each)",
                    dags.len()
                ),
            ));
        } else {
            let pooled = values.iter().sum::<f64>() / values.len() as f64;
            // Per-unit effects belong to no single completion: the contrast set keeps each
            // outcome's enumeration-weighted mean and nothing per unit.
            let pooled_set = result.counterfactual_contrasts().map(|set| CounterfactualContrasts {
                treatments: Arc::clone(&set.treatments),
                contrasts: set
                    .contrasts
                    .iter()
                    .zip(&columns)
                    .map(|(contrast, column)| ContrastedOutcome {
                        outcome: contrast.outcome,
                        unit_effects: Arc::from([]),
                        mean_ite: column.iter().sum::<f64>() / column.len() as f64,
                        standard_error: None,
                        unit_effect_intervals: None,
                        unit_extrapolative: None,
                        unit_effects_homogeneous: false,
                    })
                    .collect(),
                noise_inference: NoiseInferenceKind::AssumedNoise,
                exogenous: unabducted_exogenous(),
            });
            if let Some(effect) = result.estimate.as_effect_mut() {
                effect.ate = pooled;
                effect.uncertainty.analytic.se = f64::NAN;
                effect.uncertainty.bootstrap = BootstrapAccounting::default();
            }
            // Each completion keeps its own posterior on its atom; none is pooled.
            if let Some(mixture) = mixture.as_mut() {
                for (atom, posterior) in mixture.atoms.iter_mut().zip(posteriors) {
                    atom.posterior = posterior;
                }
            }
            result.posterior = None;
            result.payload = match pooled_set {
                Some(set) => crate::result::ResultPayload::CounterfactualContrasts(Box::new(set)),
                None => crate::result::ResultPayload::Scalar,
            };
            result.extras.interval = None;
            result.rebind_interval(self.settings.inference.is_bayesian());
            diagnostics.extend(super::envelope_se_omission_diagnostic(dags.len(), f64::NAN));
            if let Some(set) = mixture.as_ref().and_then(|mixture| {
                super::pag_path::static_class_identified_set_diagnostic(mixture, "cpdag")
            }) {
                diagnostics.push(set);
            }
            if width > 1 {
                diagnostics.push(outcomes_diagnostic(&query.outcomes, &columns, false));
            }
            diagnostics.push(Diagnostic::new(
                "estimate.counterfactual.completions_disagree",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "the per-unit counterfactual differs across the {} DAG completions of the \
                     equivalence class, so it is partially identified: the published value is \
                     the enumeration-weighted mean and the identified set is the range over \
                     completions. Completion weights are enumeration weights, not posterior \
                     probabilities, and per-unit effects belong to no single completion.",
                    dags.len()
                ),
            ));
        }
        if capped {
            diagnostics.push(Diagnostic::new(
                "estimate.counterfactual.completions_incomplete",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!(
                    "the completion search stopped at its cap or budget; the result covers the \
                     {} retained completions only (cap {}, {}), not the whole equivalence \
                     class; raise max_completions to retain more",
                    dags.len(),
                    self.completion_cap(),
                    if self.declared.max_completions.is_some() {
                        "set by the study"
                    } else {
                        "the default"
                    },
                ),
            ));
        }
        result.structural_response = mixture;
        result.identification = identification;
        result.diagnostics.extend(diagnostics);
        let certificate = crate::Identification::CpdagEnvelope {
            envelope,
            strategy: IdentifierId::GcmParametric,
            structure_version: self.graph.version(),
        };
        Ok(self.attach_certificate(result, certificate))
    }
}

/// The exogenous state of a contrast set no unit's noise was abducted for: empty, and labelled
/// as assumed because nothing was recovered from the factual rows.
fn unabducted_exogenous() -> ExogenousPosterior {
    ExogenousPosterior {
        noise: Arc::from([]),
        n_units: 0,
        n_nodes: 0,
        kind: NoiseInferenceKind::AssumedNoise,
        assumed_columns: Arc::from([]),
    }
}

/// What a result over several outcomes states: each outcome's contrast from the same structure,
/// and the range of each outcome's value over the identified functionals when they disagree.
fn outcomes_diagnostic(
    outcomes: &[VariableId],
    columns: &[Vec<f64>],
    own_uncertainty: bool,
) -> Diagnostic {
    let mut message = format!(
        "{} outcomes {:?} are contrasted from the same structure and table: the scalar estimate \
         is the first outcome's and the contrast set carries every outcome's mean effect",
        outcomes.len(),
        outcomes,
    );
    if !own_uncertainty {
        let ranges: Vec<String> = outcomes
            .iter()
            .zip(columns)
            .map(|(outcome, column)| {
                let low = column.iter().copied().fold(f64::INFINITY, f64::min);
                let high = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                format!(
                    "outcome {outcome:?} ranges over [{low}, {high}] across {} value(s)",
                    column.len()
                )
            })
            .collect();
        message = format!("{message}; {}", ranges.join("; "));
    }
    Diagnostic::new(
        "estimate.counterfactual.outcomes",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        message,
    )
}

/// The estimand the population-level route answers, stated in the result.
fn population_level_diagnostic(shape: &Shape, functionals: usize) -> Diagnostic {
    Diagnostic::new(
        "estimate.counterfactual.population_level",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "the estimand is the population-level cross-world mean E[Y_x | X = x'] - E[Y | X = x'] \
             with treatment {:?}, active level x = {} and control level x' = {}: the mean effect, \
             among the units whose observed treatment is the control level, of moving them to \
             the active level. Latent confounding prevents abducting a unit's noise, so no \
             per-unit counterfactual or unit effects are published; {functionals} identified \
             functional(s) were evaluated.",
            shape.treatment, shape.active, shape.observed,
        ),
    )
    .with_fields([
        ("estimand", "population_level_cross_world_mean".to_owned()),
        ("active_level", shape.active.to_string()),
        ("control_level", shape.observed.to_string()),
    ])
}

#[cfg(test)]
mod resample_admission_tests {
    use super::*;
    use antecedent_core::execution::AdmissionPool;

    #[test]
    fn resample_matrix_admission_checks_size_and_lifetime() {
        let pool = AdmissionPool::fixed(1, 256);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let lease = admit_counterfactual_resamples(&ctx, 2, 3, 4).unwrap();
        assert_eq!(
            pool.available().bytes(),
            256 - 2 * 3 * 8 - 2 * std::mem::size_of::<Vec<f64>>() as u64 - 4 * 16
        );
        assert!(admit_counterfactual_resamples(&ctx, usize::MAX, 2, 4).is_err());
        drop(lease);
        assert_eq!(pool.available().bytes(), 256);
        assert!(admit_counterfactual_resamples(&ctx, 10, 3, 4).is_err());
        assert_eq!(pool.available().bytes(), 256);
    }
}
