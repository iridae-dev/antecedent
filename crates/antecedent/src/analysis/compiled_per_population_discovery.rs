//! Compiled per-population structure search over declared population row sets.
//!
//! [`Study::compile_per_population_discovery`] binds one static selective contract and the
//! rows of every declared population into a lowered program of the
//! `PerPopulationStructureSearch` family. Each population's search is the static selective
//! search the `DiscoverStatic` route runs for that contract on that population's rows (the
//! contract with its population set to the label and no further populations): the program's
//! identity is the canonical list of population ids (sorted by label), each population's
//! row-set identity (the selective data identity of its searched columns) and that
//! population's search specification (its contract identity), under the identity of the
//! declared contract. Compiling checks nothing the driver does not check itself, so a request
//! the driver refuses is refused when the program runs, with the driver's own typed error.
//!
//! [`CompiledPerPopulationDiscovery::execute`] runs
//! `antecedent_discovery::selective::populations::PerPopulationDiscovery::run`, the kernel the
//! direct entry (`antecedent::analysis::per_population_discovery::discover_per_population`)
//! runs, on a fresh driver with the declared contract and populations under the caller's
//! execution context, and returns the driver unchanged: its report, its per-population
//! adapters and their retained statistic stores. Nothing is pooled and nothing is tested; the
//! edge comparison stays descriptive, as the driver reports it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, PhysicalEffect, PhysicalOperator,
    SemanticDigest, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::selective::identity::static_data_identity;
use antecedent_discovery::selective::populations::{
    PerPopulationDiscovery, PerPopulationReport, PopulationRows,
};
use antecedent_discovery::selective::{SelectiveDiscoveryError, StaticDiscoveryContract};

use super::program_handle::{CompiledProgramHandle, gate_program};
use super::supplied_identity::{Canonical, Declaration};
use crate::analysis::execute::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram, EvidenceRole,
    EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram, PhysicalLayer,
    ProblemTarget, ProgramCell, ProgramIdentityScope, ScientificProblem,
    StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};

const PLAN_ID: &str = "per_population_structure_search.v1";
const QUESTION_TAG: &str = "antecedent.per_population_structure_search.question.v1";
const SOURCE_TAG: &str = "antecedent.per_population_structure_search.rows.v1";
const EVALUATOR: &str = "antecedent_discovery::selective::populations::PerPopulationDiscovery::run";
/// Whether `execute` passes the support gate of the unit the route is bound to
/// (`parity/compiled_units.toml`). It stays off until that unit's record lists this compiled
/// route and is earned again with it, so no valid use refuses before then.
const GATE_ON_UNIT_SUPPORT: bool = false;

/// One static selective contract and the rows of every declared population.
///
/// Each population is a label and its own rows; labels are distinct and the rows of two
/// populations are never the same table (the driver refuses both, by name, when the program
/// runs). The contract declares no populations of its own beyond its default and no pooled or
/// context-variable search.
#[derive(Clone, Debug)]
pub struct PerPopulationSearchRequest {
    contract: StaticDiscoveryContract,
    populations: Vec<(String, TabularData)>,
}

impl PerPopulationSearchRequest {
    /// Declare the contract and the populations, each as `(label, rows)`.
    #[must_use]
    pub fn new(contract: StaticDiscoveryContract, populations: Vec<(String, TabularData)>) -> Self {
        Self { contract, populations }
    }

    /// The declared contract.
    #[must_use]
    pub const fn contract(&self) -> &StaticDiscoveryContract {
        &self.contract
    }

    /// The declared populations, in declaration order.
    #[must_use]
    pub fn populations(&self) -> &[(String, TabularData)] {
        &self.populations
    }
}

/// A per-population structure search bound to its lowered program.
#[derive(Clone, Debug)]
pub struct CompiledPerPopulationDiscovery {
    request: PerPopulationSearchRequest,
    lowered: Arc<LoweredProgram>,
}

/// The driver a compiled per-population search ran, bound to the program that ran it.
#[derive(Debug)]
pub struct PerPopulationSearchRun {
    driver: PerPopulationDiscovery,
    lowered_program_identity: SemanticDigest,
}

impl PerPopulationSearchRun {
    /// The driver, as `PerPopulationDiscovery::run` left it: its committed report and one
    /// retained adapter per population.
    #[must_use]
    pub const fn driver(&self) -> &PerPopulationDiscovery {
        &self.driver
    }

    /// The driver by value, to serve a later request from its retained stores.
    #[must_use]
    pub fn into_driver(self) -> PerPopulationDiscovery {
        self.driver
    }

    /// The per-population report the run committed.
    #[must_use]
    pub const fn report(&self) -> Option<&PerPopulationReport> {
        self.driver.report()
    }

    /// Identity of the lowered program.
    #[must_use]
    pub const fn lowered_program_identity(&self) -> SemanticDigest {
        self.lowered_program_identity
    }
}

/// Why a compiled per-population search did not return a driver.
#[derive(Debug)]
#[non_exhaustive]
pub enum PerPopulationSearchError {
    /// The program cannot run here: a reentry into another program's dispatch, or the support
    /// gate of its unit.
    Program(CausalError),
    /// The driver refused or failed, with its own typed error (a pooled or malformed request,
    /// content-identical populations, a contract or table a population's search refuses, an
    /// algorithm failure or cancellation).
    Search(SelectiveDiscoveryError),
}

impl fmt::Display for PerPopulationSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Program(error) => write!(f, "{error}"),
            Self::Search(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for PerPopulationSearchError {}

/// A contract's identity, or the refusal that prevents one (the driver refuses that contract
/// when the program runs).
struct ContractIdentity<'a>(&'a StaticDiscoveryContract);

impl Canonical for ContractIdentity<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self.0.identity() {
            Ok(digest) => {
                out.text("contract").bytes(&digest);
            }
            Err(error) => {
                // The refusal's registered code and detail, never its message.
                out.text("unencodable");
                error.encode(out);
            }
        }
    }
}

/// The selective data identity of a table's searched columns (rows and digest), or the reason
/// it has none (the driver refuses that table when the program runs).
struct RowSet(Result<(u64, [u8; 32]), antecedent_discovery::DiscoveryError>);

impl RowSet {
    fn of(data: &TabularData, variables: &[VariableId]) -> Self {
        Self(static_data_identity(data, variables).map(|identity| (identity.rows, identity.digest)))
    }

    fn rows(&self) -> u64 {
        self.0.as_ref().map_or(0, |(rows, _)| *rows)
    }
}

impl Canonical for RowSet {
    fn encode(&self, out: &mut CanonicalWriter) {
        match &self.0 {
            Ok((rows, digest)) => {
                out.text("rows").word(*rows).bytes(digest);
            }
            Err(error) => {
                // The failure's stable kind and code: the kernel refuses this table when the
                // program runs, so the identity names that refusal, never its message.
                out.text("unreadable");
                error.encode(out);
            }
        }
    }
}

/// One population: its label, its row-set identity and the search specification its own
/// search runs under.
struct PopulationEntry<'a> {
    label: &'a str,
    rows: RowSet,
    search: StaticDiscoveryContract,
}

impl Canonical for PopulationEntry<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(self.label);
        self.rows.encode(out);
        ContractIdentity(&self.search).encode(out);
    }
}

/// The contract a population's own search runs under: the declared contract with its
/// population set to the label and no further populations, as the driver runs it.
fn population_contract(contract: &StaticDiscoveryContract, label: &str) -> StaticDiscoveryContract {
    let mut search = contract.clone();
    label.clone_into(&mut search.population);
    search.additional_populations.clear();
    search
}

/// Every population, in label order (the order the driver reports them in).
fn entries(request: &PerPopulationSearchRequest) -> Vec<PopulationEntry<'_>> {
    let mut entries: Vec<PopulationEntry<'_>> = request
        .populations
        .iter()
        .map(|(label, data)| PopulationEntry {
            label: label.as_str(),
            rows: RowSet::of(data, &request.contract.variables),
            search: population_contract(&request.contract, label),
        })
        .collect();
    entries.sort_by(|a, b| a.label.cmp(b.label));
    entries
}

/// The question: the declared contract's identity, then every population (label, row-set
/// identity, own search contract) in label order. The populations are a set keyed by label: a
/// repeated label is refused by the driver when the program runs.
fn question_digest(
    contract: &StaticDiscoveryContract,
    populations: &[PopulationEntry<'_>],
) -> [u8; 32] {
    let mut question = Declaration::new(QUESTION_TAG);
    question.field("search", &ContractIdentity(contract)).field("populations", populations);
    question.finish()
}

/// The source: every population's row-set identity, in the label order of the question.
fn source_digest(populations: &[PopulationEntry<'_>]) -> [u8; 32] {
    let mut source = Declaration::new(SOURCE_TAG);
    let rows: Vec<&RowSet> = populations.iter().map(|entry| &entry.rows).collect();
    source.field("rows", rows.as_slice());
    source.finish()
}

fn lower(request: &PerPopulationSearchRequest) -> Result<Arc<LoweredProgram>, CausalError> {
    let populations = entries(request);
    let question = question_digest(&request.contract, &populations);
    let source = source_digest(&populations);

    let total_rows: u64 = populations.iter().map(|entry| entry.rows.rows()).sum();
    let columns = request.contract.variables.len() as u64;
    let design = DesignShape::quoted(total_rows, columns, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::StructureSearch,
            outcome: None,
            query: question,
        },
        plan_id: Arc::from(PLAN_ID),
        data: DataClassification::Tabular,
        graph_class: "one_static_selective_structure_per_declared_population",
        query_variables: Arc::from(request.contract.variables.as_slice()),
        source_digest: Some(source),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint { evaluator: Arc::from(EVALUATOR) },
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: populations
            .iter()
            .map(|entry| {
                let mut writer = CanonicalWriter::new(SOURCE_TAG);
                writer.text(entry.label);
                entry.rows.encode(&mut writer);
                EvidenceSourceRef {
                    role: EvidenceRole::Observations,
                    digest: Some(writer.finish()),
                }
            })
            .collect(),
        // Every population is searched on its own rows; nothing is pooled or weighted.
        composition: Composition::ScenarioSet { scenarios: populations.len() },
        decision: None,
        planning: None,
        chain: None,
    };
    let operator = |effect: PhysicalEffect, bytes: Option<u64>| {
        let declared = PhysicalOperator::declare(effect);
        let declared = match bytes {
            Some(bytes) => declared.estimates(ByteEstimates::absent().with_output(bytes)),
            None => declared,
        };
        declared.build().map_err(|_| CausalError::Compile {
            message: format!("{PLAN_ID}: a physical operator cannot be declared"),
        })
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            operator(
                PhysicalEffect::Read,
                Some(total_rows.saturating_mul(columns).saturating_mul(8)),
            )?,
            operator(PhysicalEffect::Fit, None)?,
        ]),
        design,
    };
    let identification = IdentificationLayer::PerExecution {
        reason: "each population's structure is searched each time the program runs",
    };
    Ok(Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::PerPopulationStructureSearch,
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand { uncertainty: UncertaintyReading::None, kind: None },
        }
        .finish(),
    ))
}

impl Study {
    /// Bind a per-population structure search over the declared population row sets.
    ///
    /// # Errors
    ///
    /// `public_builder_reentry` inside another program's lowered dispatch, or a physical
    /// operator that cannot be declared. The request itself is checked by the driver when the
    /// program runs.
    pub fn compile_per_population_discovery(
        request: PerPopulationSearchRequest,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPerPopulationDiscovery, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        let lowered = lower(&request)?;
        Ok(CompiledPerPopulationDiscovery { request, lowered })
    }
}

impl CompiledProgramHandle for CompiledPerPopulationDiscovery {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

impl CompiledPerPopulationDiscovery {
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

    /// The bound request.
    #[must_use]
    pub const fn request(&self) -> &PerPopulationSearchRequest {
        &self.request
    }

    /// Search every population's rows on a fresh driver.
    ///
    /// The context's master seed must be the contract's, as for a single population.
    ///
    /// # Errors
    ///
    /// [`PerPopulationSearchError::Program`] inside another program's dispatch or, once it is
    /// on, from the support gate; [`PerPopulationSearchError::Search`] with the driver's own
    /// refusal or failure.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<PerPopulationSearchRun, PerPopulationSearchError> {
        crate::analysis::lowering::refuse_reentry(ctx)
            .map_err(PerPopulationSearchError::Program)?;
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if GATE_ON_UNIT_SUPPORT {
            gate_program(&scoped, self)
                .map_err(|refusal| PerPopulationSearchError::Program(CausalError::from(refusal)))?;
        }
        let rows: Vec<PopulationRows<'_>> = self
            .request
            .populations
            .iter()
            .map(|(population, data)| PopulationRows { population: population.as_str(), data })
            .collect();
        let mut driver = PerPopulationDiscovery::new();
        driver
            .run(&self.request.contract, &rows, &scoped)
            .map_err(PerPopulationSearchError::Search)?;
        Ok(PerPopulationSearchRun { driver, lowered_program_identity: self.lowered.identity() })
    }
}

#[cfg(test)]
mod site_tests {
    use antecedent_core::{SemanticDigest, VariableId};
    use antecedent_data::TabularData;
    use antecedent_discovery::selective::{CiTestSpec, StaticAlgorithm, StaticDiscoveryContract};

    use super::{PerPopulationSearchRequest, entries, lower, question_digest, source_digest};
    use crate::identity_laws::{all_distinct, set_laws};

    fn table(shift: f64) -> TabularData {
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = x.map(|value| value * shift);
        let z = [1.0, 0.0, 1.0, 0.0];
        TabularData::from_f64_columns([("x", &x[..]), ("y", &y[..]), ("z", &z[..])])
            .expect("a table")
    }

    fn contract(alpha: f64) -> StaticDiscoveryContract {
        let mut contract = StaticDiscoveryContract::new(
            StaticAlgorithm::Pc,
            (0..3).map(VariableId::from_raw).collect(),
            CiTestSpec::named("parcorr"),
            7,
        );
        contract.constraints.alpha = alpha;
        contract
    }

    fn request(populations: &[(&str, f64)], alpha: f64) -> PerPopulationSearchRequest {
        PerPopulationSearchRequest::new(
            contract(alpha),
            populations.iter().map(|&(label, shift)| (label.to_owned(), table(shift))).collect(),
        )
    }

    fn lowered(populations: &[(&str, f64)]) -> SemanticDigest {
        lower(&request(populations, 0.01)).expect("lowered").identity()
    }

    fn digests(populations: &[(&str, f64)], alpha: f64) -> ([u8; 32], [u8; 32]) {
        let declared = request(populations, alpha);
        let sorted = entries(&declared);
        (question_digest(&declared.contract, &sorted), source_digest(&sorted))
    }

    #[test]
    fn site_compiled_per_population_discovery_populations_are_a_set_keyed_by_label() {
        // The driver refuses a repeated label when the program runs.
        set_laws(&[("north", 1.0), ("south", 2.0), ("east", 3.0)], |populations| {
            let mut labels: Vec<&str> = populations.iter().map(|&(label, _)| label).collect();
            labels.sort_unstable();
            labels.dedup();
            (labels.len() == populations.len()).then(|| digests(populations, 0.01).0)
        });
        let pair = [("north", 1.0), ("south", 2.0)];
        all_distinct(&[
            digests(&pair, 0.01).0,
            digests(&pair, 0.001).0,
            digests(&[("north", 2.0), ("south", 1.0)], 0.01).0,
            digests(&[("north", 1.0), ("west", 2.0)], 0.01).0,
        ]);
        all_distinct(&[
            digests(&pair, 0.01).1,
            digests(&[("north", 1.0), ("south", 3.0)], 0.01).1,
            digests(&[("north", 1.0)], 0.01).1,
        ]);
        assert_eq!(lowered(&pair), lowered(&[("south", 2.0), ("north", 1.0)]));
        all_distinct(&[
            lowered(&pair),
            lowered(&[("north", 2.0), ("south", 1.0)]),
            lowered(&[("north", 1.0), ("west", 2.0)]),
        ]);
    }
}
