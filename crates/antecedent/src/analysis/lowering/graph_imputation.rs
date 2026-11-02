//! Lowering for a bounded graph-selection / multiple-imputation composite.
//!
//! This is internal scaffolding. No interval is licensed by lowering and no raw caller-supplied
//! likelihood, posterior, child execution or coverage claim is accepted.
use std::sync::Arc;

use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, PhysicalEffect, PhysicalOperator, VariableId,
};
use antecedent_data::{ImputationModel, ImputationSpec, MissingnessAssumption};
use antecedent_discovery::adjacency_mask::{AdjacencyMask, dag_from_adjacency_mask};

use crate::analysis::{PreparedStudy, RetainedOperation};
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, GraphImputationLoweredParts, IdentificationLayer,
    LoweredProgram, PhysicalLayer, ProblemTarget, ProgramIdentityScope, RetainedIdentification,
    ScientificProblem, UncertaintyReading,
};
use crate::planner::statistical::{ModelRole, ModelSlot, Realization, StatisticalProgram};
use crate::{BayesianConfig, InferenceMode, RefuteSuite};

const TAG: &str = "antecedent.graph_imputation.declaration.v2";
const CELL: &str = "graph_imputation.exact_bic_average_effect";
const DESIGN: &str = "row imputation and native graph-weighted effects";

/// One exact graph and its native atom prepared before composite dispatch.
#[derive(Clone, Debug)]
pub(crate) struct RetainedAtomBinding {
    pub(crate) mask: u64,
    pub(crate) prepared: PreparedStudy,
}

/// Inputs must be constructed by the native compiler from the actual schema, rows and atom
/// preparation, not from an arbitrary application or decoded report.
pub(crate) struct LoweringInput {
    pub(crate) source_digest: [u8; 32],
    pub(crate) schema_digest: [u8; 32],
    pub(crate) rows: u64,
    pub(crate) variables: Arc<[VariableId]>,
    pub(crate) treatment: VariableId,
    pub(crate) outcome: VariableId,
    pub(crate) support: super::super::graph_imputation::GraphImputationSupport,
    pub(crate) imputation: ImputationSpec,
    /// Domain tags in the searched variable order (0 continuous, 1 binary).
    pub(crate) domains: Arc<[u8]>,
    pub(crate) posterior_draws: u32,
    pub(crate) prior_scale: f64,
    pub(crate) execution_seed: u64,
    pub(crate) atoms: Vec<RetainedAtomBinding>,
}

fn refuse(message: impl std::fmt::Display) -> CausalError {
    CausalError::Compile { message: format!("graph_imputation.lowering: {message}") }
}

fn operator(effect: PhysicalEffect, bytes: Option<u64>) -> Result<PhysicalOperator, CausalError> {
    let mut op = PhysicalOperator::declare(effect);
    if let Some(point) = effect.required_cancellation_point() {
        op = op.cancellation_point(point);
    }
    if let Some(bytes) = bytes {
        op = op.estimates(ByteEstimates::absent().with_output(bytes));
    }
    op.build().map_err(refuse)
}

/// Seal all semantic declarations, including every graph-mask-to-proof/program mapping, into
/// one six-layer program. No data fitting, imputation or graph scoring occurs here.
#[allow(
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    reason = "lowering seals the complete bounded declaration and owns its inputs"
)]
pub(crate) fn lower(
    input: LoweringInput,
    budget: &mut antecedent_core::search::SearchBudget<'_>,
) -> Result<LoweredProgram, super::super::graph_imputation::GraphImputationError> {
    let n = input.variables.len();
    if !(2..=3).contains(&n)
        || input.rows < 4
        || input.rows > 10_000
        || input.treatment == input.outcome
        || !input.variables.contains(&input.treatment)
        || !input.variables.contains(&input.outcome)
        || input.domains.len() != n
        || input.domains.iter().any(|&d| d > 1)
        || input.domains.get(input.treatment.raw() as usize) != Some(&1)
        || input.imputation.model != ImputationModel::ChainedEquations
        || input.imputation.assumption != MissingnessAssumption::MissingAtRandom
        || !(2..=64).contains(&input.imputation.imputations)
        || !(1..=64).contains(&input.imputation.iterations)
        || input.imputation.bootstrap_replicates != 0
        || !(100..=4096).contains(&input.posterior_draws)
        || !input.prior_scale.is_finite()
        || input.prior_scale <= 0.0
    {
        return Err(refuse("declaration is outside the bounded native cell").into());
    }
    for (index, variable) in input.variables.iter().enumerate() {
        if variable.raw() as usize != index {
            return Err(refuse("variables must use dense table order").into());
        }
    }
    let mut expected = Vec::new();
    let end = 1u64 << (n * (n - 1));
    for mask in 0..end {
        if let Err(stop) = budget.charge(1, 25 * 1024 * 1024) {
            return Err(super::super::graph_imputation::GraphImputationError::SearchStopped(
                budget.receipt(
                    stop,
                    vec![format!("verified:0..{mask}")],
                    vec![format!("remaining:{mask}..{end}")],
                ),
            ));
        }
        let adjacency = AdjacencyMask::from_word(mask);
        if antecedent_discovery::graph_imputation_enumeration::graph_imputation_supported_mask(
            &adjacency,
            n,
            input.treatment.raw() as usize,
            input.outcome.raw() as usize,
        ) && dag_from_adjacency_mask(adjacency, n).is_ok()
        {
            expected.push(mask);
        }
    }
    if input.atoms.len() != expected.len()
        || input.atoms.iter().zip(&expected).any(|(atom, mask)| atom.mask != *mask)
    {
        return Err(refuse(
            "every bounded DAG needs its exact ordered proof and checked atom binding",
        )
        .into());
    }
    let query = antecedent_core::AverageEffectQuery::binary_ate(input.treatment, input.outcome);
    let inference = InferenceMode::Bayesian(
        BayesianConfig::conjugate()
            .n_draws(input.posterior_draws as usize)
            .prior_scale(input.prior_scale),
    );
    let mut caches = Vec::with_capacity(input.atoms.len());
    let mut seals = Vec::with_capacity(input.atoms.len());
    for atom in &input.atoms {
        if atom.prepared.is_materialized() {
            return Err(refuse("atom bindings must be declaration-only").into());
        }
        let dag =
            dag_from_adjacency_mask(AdjacencyMask::from_word(atom.mask), n).map_err(refuse)?;
        if atom.prepared.study().graph.as_dag() != Some(&dag) {
            return Err(refuse("atom mask does not match its retained graph").into());
        }
        let RetainedOperation::BayesianGcomp { operation, validation } =
            atom.prepared.retained_operation()
        else {
            return Err(refuse("atom has no retained Bayesian g-computation operation").into());
        };
        if operation.query() != &query
            || operation.inference() != &inference
            || validation != RefuteSuite::None
        {
            return Err(refuse(
                "atom target, inference or validation differs from the composite declaration",
            )
            .into());
        }
        let cache = atom
            .prepared
            .study()
            .identification
            .as_ref()
            .ok_or_else(|| refuse("atom lacks retained identification"))?;
        let proof = crate::analysis::identification_seal::seal_cache(cache)?
            .ok_or_else(|| refuse("atom identification cannot be sealed"))?;
        let program = atom
            .prepared
            .plan()
            .lowered()
            .ok_or_else(|| refuse("atom has no lowered program"))?
            .identity();
        caches.push(cache.clone());
        seals.push((atom.mask, proof, program));
    }

    let mut declaration = CanonicalWriter::new(TAG);
    match input.support {
        super::super::graph_imputation::GraphImputationSupport::TreatmentRootDirectEffect => {
            declaration.text("treatment_root_direct_effect.v1");
        }
    }
    declaration
        .bytes(&input.source_digest)
        .bytes(&input.schema_digest)
        .word(input.rows)
        .word(n as u64)
        .word(u64::from(input.treatment.raw()))
        .word(u64::from(input.outcome.raw()))
        .word(u64::from(input.imputation.imputations))
        .word(u64::from(input.imputation.iterations))
        .word(input.imputation.seed)
        .word(input.execution_seed)
        .word(u64::from(input.posterior_draws))
        .word(input.prior_scale.to_bits());
    for domain in input.domains.iter() {
        declaration.word(u64::from(*domain));
    }
    declaration.word(input.atoms.len() as u64);
    for (mask, proof, program) in &seals {
        declaration.word(*mask).bytes(proof).bytes(program.as_bytes());
    }
    let query = declaration.finish();
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::GraphImputationEffect,
            outcome: Some(input.outcome),
            query,
        },
        plan_id: Arc::from(CELL),
        data: DataClassification::Tabular,
        graph_class: "dag",
        query_variables: Arc::clone(&input.variables),
        source_digest: Some(input.source_digest),
        support_coordinate: None,
        licensed: false,
        support_status: None,
    };
    let identification = IdentificationLayer::Retained(Arc::new(RetainedIdentification::new(
        None,
        caches,
        Some(query),
    )?));
    // This sequence does not claim to be one joint Bayesian posterior: the graph score
    // is BIC-based and the imputation/child-posterior composition has separate assumptions.
    let models = vec![
        ModelSlot {
            role: ModelRole::Nuisance,
            family: Arc::from("native.chained_equations"),
            inputs: Arc::clone(&input.variables),
            output: None,
        },
        ModelSlot {
            role: ModelRole::Outcome,
            family: Arc::from("native.conjugate_gaussian_gcomp"),
            inputs: Arc::clone(&input.variables),
            output: Some(input.outcome),
        },
    ];
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::GraphImputationV1,
        models: Arc::from(models),
        folds: None,
        resampling: None,
        posterior: None,
        design: DesignShape::not_design_based(DESIGN),
        operators: Arc::from(Vec::new()),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::Observations,
            digest: Some(input.source_digest),
        }]),
        composition: Composition::Chained { stages: 3 },
        decision: None,
        planning: None,
        chain: None,
    };
    let work_bytes = input
        .rows
        .checked_mul(n as u64)
        .and_then(|v| v.checked_mul(32))
        .and_then(|v| v.checked_add(u64::from(input.posterior_draws) * 8 * 25))
        .ok_or_else(|| refuse("admission size overflow"))?;
    let physical = PhysicalLayer {
        operators: Arc::from([
            operator(PhysicalEffect::Read, None)?,
            operator(PhysicalEffect::Fit, Some(work_bytes))?,
            operator(PhysicalEffect::Condition, None)?,
        ]),
        design: DesignShape::not_design_based(DESIGN),
    };
    let claim = ClaimDemand { uncertainty: UncertaintyReading::None, kind: Some(ClaimKind::Point) };
    Ok(GraphImputationLoweredParts {
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim,
    }
    .finish())
}
