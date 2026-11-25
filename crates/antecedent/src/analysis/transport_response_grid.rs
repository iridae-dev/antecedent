//! Compiled augmented transported response grid from caller-supplied nuisances.
//!
//! The question is a [`TransportQuery`] whose inner response is a mean curve over a grid of
//! treatment values, on a selection diagram, with the trial columns of
//! [`crate::TransportTrialSpec`] and the caller's regressions and local response weights
//! ([`TransportResponseGridColumns`]). [`Study::compile_transport_response_grid`] decides the
//! question by sID once (the certificate the trial-to-target route runs, refusing a formula that
//! route cannot evaluate), reads the columns and binds them into a lowered program;
//! [`CompiledTransportResponseGrid::execute`] evaluates
//! `antecedent_estimate::transport_augmented_response_grid_with_context` on exactly those
//! inputs under the caller's execution policy.
//!
//! The kernel evaluates the trial-generalization augmented formula of Dahabreh et al. (2019,
//! Web Appendix 2) with a grid-local source correction:
//!
//! ```text
//! mean(g) = (1 / n_target) * ( sum_{target i} m_g(x_i)
//!                              + sum_{trial i} (1 - s_i) / s_i * w_g(i) * (y_i - m(x_i, a_i)) )
//! ```
//!
//! It fits no nuisance, chooses no bandwidth and forms no interval: the result is point-only
//! and does not claim that the composition inherits either component's double robustness or
//! inference theorem. The calibration of the route is `not_measured`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CanonicalEncode, CanonicalWriter, ExecutionContext, ResponseFunctional, SemanticDigest,
    TransportQuery, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{
    ResponseGridInputs, TransportResponseGridEstimate,
    transport_augmented_response_grid_with_context,
};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::TransportIdentification;

use super::compiled_point_program::PointProgram;
use super::program_handle::{CompiledProgramHandle, gate_program};
use super::supplied_identity::{Canonical, SetOf, encoded};
use super::transport_supplied_nuisance::TransportResponseGridColumns;
use crate::analysis::TransportTrialSpec;
use crate::analysis::execute::{Study, live_transport_identification};
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell};

const PLAN_ID: &str = "transport_augmented_response_grid.v1";
const EVALUATOR: &str = "antecedent_estimate::transport_augmented_response_grid";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record has earned its support,
/// so no valid use refuses before then; turning it on makes a release build refuse the route
/// while the unit is unsupported and a development build record what it would refuse.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// The declared question of an augmented transported response grid.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TransportResponseGridRequest {
    /// Selection diagram the transport is decided on.
    pub diagram: SelectionDiagram,
    /// Transport of a mean curve over the treatment grid the columns are laid out on.
    pub query: TransportQuery,
    /// Trial and target rows.
    pub data: TabularData,
    /// Trial membership and selection probability (the treatment probability is not read).
    pub trial: TransportTrialSpec,
    /// The caller's regressions and local response weights.
    pub columns: TransportResponseGridColumns,
}

impl TransportResponseGridRequest {
    /// A request over `data`.
    #[must_use]
    pub fn new(
        diagram: SelectionDiagram,
        query: TransportQuery,
        data: TabularData,
        trial: TransportTrialSpec,
        columns: TransportResponseGridColumns,
    ) -> Self {
        Self { diagram, query, data, trial, columns }
    }
}

/// The columns the kernel reads, in its layout.
#[derive(Clone, Debug)]
struct GridInputs {
    outcome: Vec<f64>,
    trial: Vec<bool>,
    selection: Vec<f64>,
    grid: Vec<f64>,
    observed_regression: Vec<f64>,
    /// `target_regression[g * n + i]`.
    target_regression: Vec<f64>,
    /// `response_weight[g * n + i]`.
    response_weight: Vec<f64>,
}

/// A decided, bound augmented response grid.
#[derive(Clone, Debug)]
pub struct CompiledTransportResponseGrid {
    identification: Arc<TransportIdentification>,
    inputs: Arc<GridInputs>,
    lowered: Arc<LoweredProgram>,
}

/// The grid the kernel evaluated, bound to the program that evaluated it.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct TransportResponseGridRun {
    /// Grid means, source effective sample sizes and selection overlap, as the kernel
    /// returned them.
    pub estimate: TransportResponseGridEstimate,
    /// Identity of the lowered program.
    pub lowered_program_identity: SemanticDigest,
    /// Always `point_only`: the route forms no interval.
    pub claim: &'static str,
    /// Always `not_measured`.
    pub calibration: &'static str,
}

fn read(data: &TabularData, column: VariableId) -> Result<Vec<f64>, CausalError> {
    data.float64_values(column).map_err(CausalError::rejected)
}

impl Study {
    /// Decide and bind an augmented transported response grid.
    ///
    /// # Errors
    ///
    /// A query that is not a mean curve over listed grid values, a column count that is not
    /// one per grid point, a missing column, a non-finite trial indicator, a transport sID does
    /// not certify or certifies by a formula the trial-to-target weighting cannot evaluate,
    /// cancellation, or a resource refusal.
    pub fn compile_transport_response_grid(
        request: TransportResponseGridRequest,
        ctx: &ExecutionContext,
    ) -> Result<CompiledTransportResponseGrid, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "transport response grid compile" });
        }
        let TransportResponseGridRequest { diagram, query, data, trial, columns } = request;
        query.validate()?;
        let ResponseFunctional::MeanCurve { outcome, treatment } = &query.response.functional
        else {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "an augmented transported response grid reads a mean curve over listed \
                 treatment values; transport a ResponseFunctional::MeanCurve"
            ));
        };
        let grid = treatment.grid.values()?;
        if grid.is_empty()
            || columns.grid_regressions.len() != grid.len()
            || columns.response_weights.len() != grid.len()
        {
            return Err(crate::unsupported_reason!(
                "invalid_argument",
                detail =
                    antecedent_core::reason_details::transport_response_grid::GRID_COLUMNS_MISMATCH,
                "the grid points of the query each need one regression and one response-weight \
                 column, in grid order. Next attempt: name one column of each per grid point"
            ));
        }
        let n = data.row_count();
        let bytes = (n as u64).checked_mul(8 * (4 + 2 * grid.len() as u64)).ok_or_else(|| {
            CausalError::Resource { message: "transport grid size overflow".into() }
        })?;
        let _lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "transport response grid compile" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let identification = live_transport_identification(&diagram, &query, ctx)?;
        let trial_values = read(&data, trial.trial)?;
        if let Some(row) = trial_values.iter().position(|value| !value.is_finite()) {
            return Err(CausalError::Compile {
                message: format!(
                    "transport response grid: the trial indicator is missing on row {row}; it \
                     would be read as a trial or a target row. Next attempt: drop or impute \
                     the row"
                ),
            });
        }
        let mut target_regression = Vec::with_capacity(n * grid.len());
        let mut response_weight = Vec::with_capacity(n * grid.len());
        for (&regression, &weight) in
            columns.grid_regressions.iter().zip(columns.response_weights.iter())
        {
            target_regression.extend(read(&data, regression)?);
            response_weight.extend(read(&data, weight)?);
        }
        let inputs = GridInputs {
            outcome: read(&data, *outcome)?,
            trial: trial_values.iter().map(|value| *value != 0.0).collect(),
            selection: read(&data, trial.selection_probability)?,
            grid,
            observed_regression: read(&data, columns.observed_regression)?,
            target_regression,
            response_weight,
        };
        let lowered = lower(&diagram, &query, &trial, &columns, &inputs, *outcome)?;
        Ok(CompiledTransportResponseGrid {
            identification: Arc::new(identification),
            inputs: Arc::new(inputs),
            lowered,
        })
    }
}

/// The declared question: the selection diagram (the graph's canonical identity, with sorted
/// edge lists, and the selection targets as a set), the query by its canonical encoding, the
/// trial columns, and the observed regression, grid regressions and response weights (the last
/// two aligned with the grid, so in declared order).
fn question_digest(
    diagram: &SelectionDiagram,
    query: &TransportQuery,
    trial: &TransportTrialSpec,
    columns: &TransportResponseGridColumns,
) -> [u8; 32] {
    let ids = |values: &[VariableId]| values.iter().map(|v| u64::from(v.raw())).collect::<Vec<_>>();
    let mut question = CanonicalWriter::new("antecedent.transport_response_grid.question.v2");
    Canonical::encode(diagram.causal_graph(), &mut question);
    // A set: identification reads the selection targets as a node set.
    Canonical::encode(&SetOf(diagram.selection_targets()), &mut question);
    CanonicalEncode::encode(query, &mut question);
    question
        .words(&[u64::from(trial.trial.raw()), u64::from(trial.selection_probability.raw())])
        .word(u64::from(columns.observed_regression.raw()))
        .words(&ids(&columns.grid_regressions[..]))
        .words(&ids(&columns.response_weights[..]));
    question.finish()
}

/// The numbers the kernel reads, every column in row order and the grid-aligned columns
/// grid-point-major: outcome, trial membership, selection probability, the grid, the observed
/// regression, the grid regressions and the response weights, each count-prefixed.
fn source_digest(inputs: &GridInputs) -> [u8; 32] {
    let mut source = CanonicalWriter::new("antecedent.transport_response_grid.inputs.v1");
    source
        .reals(&inputs.outcome)
        .words(&inputs.trial.iter().map(|&member| u64::from(member)).collect::<Vec<_>>())
        .reals(&inputs.selection)
        .reals(&inputs.grid)
        .reals(&inputs.observed_regression)
        .reals(&inputs.target_regression)
        .reals(&inputs.response_weight);
    source.finish()
}

fn lower(
    diagram: &SelectionDiagram,
    query: &TransportQuery,
    trial: &TransportTrialSpec,
    columns: &TransportResponseGridColumns,
    inputs: &GridInputs,
    outcome: VariableId,
) -> Result<Arc<LoweredProgram>, CausalError> {
    // A graph without a canonical identity refuses the compile (`encoded`).
    let question = encoded(PLAN_ID, || question_digest(diagram, query, trial, columns))?;
    let source = source_digest(inputs);
    let n = inputs.outcome.len() as u64;
    let points = inputs.grid.len() as u64;
    let mut query_variables = vec![outcome, trial.trial, trial.selection_probability];
    query_variables.push(columns.observed_regression);
    query_variables.extend(columns.grid_regressions.iter().copied());
    query_variables.extend(columns.response_weights.iter().copied());
    let mut supplied = vec![("supplied_observed_regression", Some(columns.observed_regression))];
    supplied
        .extend(columns.grid_regressions.iter().map(|&c| ("supplied_grid_regression", Some(c))));
    supplied
        .extend(columns.response_weights.iter().map(|&c| ("supplied_response_weight", Some(c))));
    PointProgram {
        cell: ProgramCell::TransportResponseGrid,
        plan_id: PLAN_ID,
        kind: FunctionalKind::InterventionalDistribution,
        outcome: Some(outcome),
        graph_class: "fixed_selection_admg",
        query_variables: Arc::from(query_variables),
        question,
        source,
        source_role: EvidenceRole::Trial,
        evaluator: EVALUATOR,
        supplied,
        rows: n,
        columns: 4 + 2 * points,
        input_bytes: n.saturating_mul(8 * (4 + 2 * points)),
        claim: antecedent_contract::ClaimKind::Response,
    }
    .lower()
}

impl CompiledProgramHandle for CompiledTransportResponseGrid {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledTransportResponseGrid {
    /// Inspect all six compiler layers.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the plan, including the support decision of its unit, without evaluating.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        CompiledProgramHandle::plan_view(self)
    }

    /// The certificate the grid is evaluated under.
    #[must_use]
    pub fn identification(&self) -> &TransportIdentification {
        &self.identification
    }

    /// Evaluate the augmented response equation on the bound inputs.
    ///
    /// # Errors
    ///
    /// The kernel's refusals (mismatched or non-finite inputs, selection probabilities outside
    /// `(0, 1)`, no trial or no target rows), cancellation, or the support gate once it is on.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<TransportResponseGridRun, CausalError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self)?;
        }
        if scoped.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "transport response grid" });
        }
        let inputs = &*self.inputs;
        let estimate = transport_augmented_response_grid_with_context(
            &self.identification,
            &inputs.outcome,
            &inputs.trial,
            &inputs.selection,
            &ResponseGridInputs {
                grid: &inputs.grid,
                observed_regression: &inputs.observed_regression,
                target_regression: &inputs.target_regression,
                response_weight: &inputs.response_weight,
            },
            &scoped,
        )?;
        Ok(TransportResponseGridRun {
            estimate,
            lowered_program_identity: self.lowered.identity(),
            claim: "point_only",
            calibration: "not_measured",
        })
    }
}

#[cfg(test)]
mod identity_tests {
    use antecedent_core::{ContinuousDomain, GridSpec, ResponseQuery};
    use antecedent_graph::{Admg, DenseNodeId};

    use super::*;
    use crate::identity_laws::{all_distinct, bit_exact, ordered_laws, set_laws};

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    /// A graph over three variables with `edges` inserted in the given order.
    fn diagram(edges: &[(u32, u32)], selections: &[u32]) -> SelectionDiagram {
        let mut graph = Admg::with_variables(3);
        for &(a, b) in edges {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let selections = selections.iter().copied().map(v).collect::<Vec<_>>();
        SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(selections)).unwrap()
    }

    fn digest(diagram: &SelectionDiagram, grid: &[f64], weights: &[u32]) -> [u8; 32] {
        let query = TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: v(2),
                treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from(grid))),
            }),
            "trial",
            "target",
            [v(0)],
        );
        let trial = TransportTrialSpec {
            trial: v(3),
            selection_probability: v(4),
            treatment_probability: v(5),
        };
        let columns = TransportResponseGridColumns::new(
            v(6),
            [v(7), v(8)],
            weights.iter().copied().map(v).collect::<Vec<_>>(),
        );
        question_digest(diagram, &query, &trial, &columns)
    }

    fn checked(selections: &[u32], weights: &[u32]) -> Option<[u8; 32]> {
        let mut graph = Admg::with_variables(3);
        for (a, b) in [(0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).ok()?;
        }
        let selections = selections.iter().copied().map(v).collect::<Vec<_>>();
        let selection_diagram =
            SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(selections)).ok()?;
        Some(digest(&selection_diagram, &[0.0, 1.0], weights))
    }

    #[test]
    fn site_transport_response_grid_question_sets_selections_and_orders_the_columns() {
        set_laws(&[1_u32, 2], |selections| checked(selections, &[9, 10]));
        ordered_laws(&[9_u32, 10, 11], |weights| checked(&[1, 2], weights));
        let graph = diagram(&[(0, 2), (1, 2)], &[1, 2]);
        ordered_laws(&[0.0, 1.0, 2.0], |grid| Some(digest(&graph, grid, &[9, 10])));
        bit_exact(|point| Some(digest(&graph, &[point, 1.0], &[9, 10])));
    }

    fn inputs(outcome: &[f64]) -> GridInputs {
        let n = outcome.len();
        GridInputs {
            outcome: outcome.to_vec(),
            trial: (0..n).map(|row| row % 2 == 0).collect(),
            selection: vec![0.5; n],
            grid: vec![0.0, 1.0],
            observed_regression: vec![1.0; n],
            target_regression: vec![2.0; 2 * n],
            response_weight: vec![1.0; 2 * n],
        }
    }

    #[test]
    fn site_transport_response_grid_source_orders_rows_and_frames_the_columns() {
        ordered_laws(&[1.0, 2.0, 3.0], |outcome| Some(source_digest(&inputs(outcome))));
        bit_exact(|value| Some(source_digest(&inputs(&[value, 2.0]))));
        let base = inputs(&[1.0, 2.0]);
        // A value moved from one column to the next is a different input.
        let mut moved = base.clone();
        moved.selection.push(moved.observed_regression.remove(0));
        let mut weights = base.clone();
        weights.response_weight[0] = 3.0;
        let mut regression = base.clone();
        regression.target_regression[0] = 3.0;
        all_distinct(&[
            source_digest(&base),
            source_digest(&moved),
            source_digest(&weights),
            source_digest(&regression),
        ]);
    }

    #[test]
    fn the_response_grid_question_identity_is_canonical() {
        let graph = diagram(&[(0, 2), (1, 2)], &[1, 2]);
        let base = digest(&graph, &[0.0, 1.0], &[9, 10]);
        assert_eq!(base, digest(&graph, &[0.0, 1.0], &[9, 10]), "repeatable");
        let reinserted = diagram(&[(1, 2), (0, 2)], &[2, 1]);
        assert_eq!(base, digest(&reinserted, &[0.0, 1.0], &[9, 10]), "edges and selections");
        assert_ne!(base, digest(&graph, &[0.0, 1.0], &[10, 9]), "weights are ordered");
        assert_ne!(base, digest(&graph, &[-0.0, 1.0], &[9, 10]), "reals by bits");
        assert_ne!(base, digest(&diagram(&[(0, 2)], &[1, 2]), &[0.0, 1.0], &[9, 10]));
    }
}
