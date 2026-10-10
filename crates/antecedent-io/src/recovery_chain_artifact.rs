//! Independent artifact for the 2.3 B2 ordered-response recovery row (X10 second row).
//!
//! Format version 1. The artifact binds the m-graph (variables in dense order and
//! directed edges), the query roles, the decision and its evidence. A recovered
//! decision stores the checked plan (head axis, tail dependence, rule version, the
//! checked graphical premises and the recovery formula), the observed nine-cell
//! pattern law and the recovered law `P(X1, X2)`. A nonrecoverable decision stores
//! the self-censoring edge, the two witness models and the exact integer
//! verification (observed cells compared, the differing target cell and its two
//! masses over `60^4`).
//!
//! A consumer trusts nothing. It rebuilds the graph and query, re-decides under its
//! own budget and accepts only an identical plan or an identical witness, re-verifies
//! the witness by exact enumeration, re-validates the observed law, re-evaluates the
//! recovery formula and compares every recovered cell bit for bit. Two digests guard
//! the stored premises: the premises digest (graph, query, names, plan or witness)
//! and a separate data digest (the observed and recovered laws, bit for bit). A
//! changed edge, role, plan, witness model or cell is refused even when both digests
//! are re-sealed, because the replay then differs.
//!
//! What replay does not protect against: a producer that supplies a fabricated
//! observed pattern law consistently (the consumer checks the derivation and the
//! arithmetic, not that the table describes real data), and the m-graph's untestable
//! premises (for example that no further self-censoring edge exists): they are
//! declared, never verified from data. The row evaluates exact laws; any sampled
//! provider composed on it is calibrated separately and is unmeasured and closed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, IdentityDomain, VariableId};
use antecedent_estimate::{ChainPatternLaw, ChainRecoveredLaw, evaluate_chain_recovery};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{
    ChainPartial, ChainRecoveryDecision, ChainRecoveryDetail, ChainRecoveryError,
    ChainRecoveryPlan, ChainRecoveryQuery, ChainRecoveryWitness, ChainWitnessCheck,
    ChainWitnessMechanism, decide_chain_recovery, verify_chain_witness,
};
use serde::{Deserialize, Serialize};

use crate::IoError;

/// The artifact format this reader writes and accepts.
pub const RECOVERY_CHAIN_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const RECOVERY_CHAIN_ARTIFACT_FEATURE: &str = "ordered_response_recovery_v1";
/// Outcome tag of a recovered decision.
pub const RECOVERY_CHAIN_RECOVERED: &str = "recovered";
/// Outcome tag of a nonrecoverable decision.
pub const RECOVERY_CHAIN_NONRECOVERABLE: &str = "nonrecoverable";

/// Why a recovery-chain artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RecoveryChainArtifactError {
    /// The feature marker, outcome tag or result shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A route refusal: the re-decision, the witness verification, the observed law
    /// validation or the formula evaluation refused with its own typed reason.
    #[error("{0}")]
    Route(ChainRecoveryError),
    /// The re-decided plan, witness or outcome differs from the stored one.
    #[error("the stored decision does not replay: {0}")]
    DecisionMismatch(&'static str),
    /// The recomputed recovered law differs from the stored one.
    #[error("recovered law does not replay")]
    RecoveredMismatch,
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data digest does not match the stored laws.
    #[error("data identity digest mismatch")]
    DataMismatch,
    /// The caller's variable names are not the verified mapping.
    #[error("variable names do not match the verified name mapping")]
    NamesMismatch,
    /// The bytes do not decode.
    #[error("artifact does not decode: {0}")]
    Undecodable(String),
}

impl RecoveryChainArtifactError {
    /// `(reason code, detail)` of this refusal.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, &'static str) {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        match self {
            Self::UnsupportedSemantics(_) => (
                antecedent_core::reason_code!("route_not_supported"),
                "recovery_chain.unsupported_semantics",
            ),
            Self::Route(error) => (error.reason_code(), error.detail.detail()),
            Self::DecisionMismatch(_) => (invalid, "recovery_chain.decision_replay_mismatch"),
            Self::RecoveredMismatch => (invalid, "recovery_chain.recovered_replay_mismatch"),
            Self::PremisesMismatch => (invalid, "recovery_chain.premises_mismatch"),
            Self::DataMismatch => (invalid, "recovery_chain.data_identity_mismatch"),
            Self::NamesMismatch => (invalid, "recovery_chain.names_mismatch"),
            Self::Undecodable(_) => (invalid, "recovery_chain.undecodable"),
        }
    }
}

impl From<RecoveryChainArtifactError> for IoError {
    fn from(error: RecoveryChainArtifactError) -> Self {
        let (code, detail) = error.refusal();
        Self::Refused { code, message: format!("{detail}: recovery chain artifact: {error}") }
    }
}

impl From<ChainRecoveryError> for RecoveryChainArtifactError {
    fn from(error: ChainRecoveryError) -> Self {
        Self::Route(error)
    }
}

/// Portable m-graph: variable ids in dense node order and directed edges by id.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChainGraphWire {
    /// Variable id of every node, in dense order.
    pub nodes: Vec<u32>,
    /// Directed edges by variable id, sorted.
    pub directed: Vec<(u32, u32)>,
}

impl ChainGraphWire {
    /// Encode in canonical form.
    ///
    /// # Errors
    /// A non-static node or a bidirected edge (never part of this row).
    pub fn from_graph(graph: &Admg) -> Result<Self, IoError> {
        if graph.has_bidirected() {
            return Err(IoError::Convert("the m-graph of this row has no bidirected edge".into()));
        }
        let variable = |id: DenseNodeId| match graph.nodes().get(id.as_usize()) {
            Some(NodeRef::Static(v)) => Ok(v.raw()),
            _ => Err(IoError::Convert("the m-graph must have static nodes only".into())),
        };
        let mut nodes = Vec::new();
        let mut directed = Vec::new();
        for i in 0..graph.node_count() {
            let from = DenseNodeId::from_raw(u32::try_from(i).map_err(|_| IoError::TooLarge)?);
            nodes.push(variable(from)?);
            for child in graph.children(from) {
                directed.push((variable(from)?, variable(*child)?));
            }
        }
        directed.sort_unstable();
        Ok(Self { nodes, directed })
    }

    /// Decode.
    ///
    /// # Errors
    /// A duplicate node, an unknown edge endpoint or a cycle.
    pub fn to_graph(&self) -> Result<Admg, IoError> {
        let mut graph = Admg::empty();
        for raw in &self.nodes {
            graph
                .add_node(NodeRef::Static(VariableId::from_raw(*raw)))
                .map_err(crate::error::convert_err)?;
        }
        let dense = |raw: u32| {
            self.nodes
                .iter()
                .position(|n| *n == raw)
                .and_then(|i| u32::try_from(i).ok())
                .map(DenseNodeId::from_raw)
                .ok_or_else(|| IoError::Convert("an edge endpoint is not a node".into()))
        };
        for (from, to) in &self.directed {
            graph.insert_directed(dense(*from)?, dense(*to)?).map_err(crate::error::convert_err)?;
        }
        Ok(graph)
    }
}

/// Portable query: `(variable, response, proxy)` per declared axis.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChainQueryWire {
    /// First declared variable (axis 0).
    pub first: (u32, u32, u32),
    /// Second declared variable (axis 1).
    pub second: (u32, u32, u32),
}

impl ChainQueryWire {
    /// Encode.
    #[must_use]
    pub fn from_query(query: &ChainRecoveryQuery) -> Self {
        let part = |p: &ChainPartial| (p.variable.raw(), p.response.raw(), p.proxy.raw());
        Self { first: part(&query.first), second: part(&query.second) }
    }

    /// Decode. Validation happens when the query is decided.
    #[must_use]
    pub fn to_query(self) -> ChainRecoveryQuery {
        let part = |(x, r, p): (u32, u32, u32)| ChainPartial {
            variable: VariableId::from_raw(x),
            response: VariableId::from_raw(r),
            proxy: VariableId::from_raw(p),
        };
        ChainRecoveryQuery { first: part(self.first), second: part(self.second) }
    }
}

/// The checked plan on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChainPlanWire {
    /// Axis (0 or 1) of the head of the response chain.
    pub head: usize,
    /// Whether the tail response also depends on the head variable.
    pub tail_depends_on_head_variable: bool,
    /// Rule version.
    pub rule_version: String,
    /// Checked graphical premises, in order.
    pub premises: Vec<String>,
    /// The recovery formula in words.
    pub formula: String,
    /// Operations charged by the decision.
    pub operations_consumed: usize,
}

impl ChainPlanWire {
    /// Encode a checked plan.
    #[must_use]
    pub fn of_plan(plan: &ChainRecoveryPlan) -> Self {
        Self {
            head: plan.head,
            tail_depends_on_head_variable: plan.tail_depends_on_head_variable,
            rule_version: plan.rule_version.to_owned(),
            premises: plan.premises.clone(),
            formula: plan.formula.clone(),
            operations_consumed: plan.operations_consumed,
        }
    }
}

/// One witness mechanism on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChainMechanismWire {
    /// Node.
    pub node: u32,
    /// Graph parents, ascending.
    pub parents: Vec<u32>,
    /// Numerators over 60 per parent configuration.
    pub numerators: Vec<u32>,
}

/// The witness on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChainWitnessWire {
    /// The self-censoring edge `(X_i, R_i)`.
    pub edge: (u32, u32),
    /// Mechanisms of `X1, X2, R1, R2` of the first model.
    pub first: Vec<ChainMechanismWire>,
    /// Mechanisms of `X1, X2, R1, R2` of the second model.
    pub second: Vec<ChainMechanismWire>,
}

impl ChainWitnessWire {
    /// Encode a witness.
    #[must_use]
    pub fn of_witness(witness: &ChainRecoveryWitness) -> Self {
        let model = |m: &[ChainWitnessMechanism]| {
            m.iter()
                .map(|x| ChainMechanismWire {
                    node: x.node,
                    parents: x.parents.clone(),
                    numerators: x.numerators.clone(),
                })
                .collect()
        };
        Self { edge: witness.edge, first: model(&witness.first), second: model(&witness.second) }
    }

    fn to_witness(&self) -> ChainRecoveryWitness {
        let model = |m: &[ChainMechanismWire]| {
            m.iter()
                .map(|x| ChainWitnessMechanism {
                    node: x.node,
                    parents: x.parents.clone(),
                    numerators: x.numerators.clone(),
                })
                .collect()
        };
        ChainRecoveryWitness {
            edge: self.edge,
            first: model(&self.first),
            second: model(&self.second),
        }
    }
}

/// What the verified witness shows, on the wire (masses as decimal strings).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ChainWitnessCheckWire {
    /// Observed pattern cells compared (all equal).
    pub observed_cells: usize,
    /// A target cell `(x1, x2)` where the two models differ.
    pub differing_cell: (u8, u8),
    /// The two target masses at that cell.
    pub masses: (String, String),
    /// Common denominator `60^4`.
    pub denominator: String,
}

impl ChainWitnessCheckWire {
    /// Encode a verified witness check.
    #[must_use]
    pub fn of_check(check: &ChainWitnessCheck) -> Self {
        Self {
            observed_cells: check.observed_cells,
            differing_cell: check.differing_cell,
            masses: (check.masses.0.to_string(), check.masses.1.to_string()),
            denominator: check.denominator.to_string(),
        }
    }
}

/// An observed or recovered nine-cell law on the wire.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ChainLawWire {
    /// `P(R1=1, R2=1, X*1=x1, X*2=x2)`.
    pub both: [[f64; 2]; 2],
    /// `P(R1=1, R2=0, X*1=x1)`.
    pub only_first: [f64; 2],
    /// `P(R1=0, R2=1, X*2=x2)`.
    pub only_second: [f64; 2],
    /// `P(R1=0, R2=0)`.
    pub neither: f64,
}

impl ChainLawWire {
    /// Encode an observed pattern law.
    #[must_use]
    pub fn of_law(law: &ChainPatternLaw) -> Self {
        Self {
            both: *law.both(),
            only_first: *law.only_first(),
            only_second: *law.only_second(),
            neither: law.neither(),
        }
    }

    /// The validated law.
    ///
    /// # Errors
    /// `invalid_observed_law` when a cell is negative or non-finite or the cells do
    /// not sum to one.
    pub fn to_law(&self) -> Result<ChainPatternLaw, ChainRecoveryError> {
        ChainPatternLaw::new(self.both, self.only_first, self.only_second, self.neither)
    }

    fn bits(&self) -> Vec<u64> {
        self.both
            .iter()
            .flatten()
            .chain(&self.only_first)
            .chain(&self.only_second)
            .chain(std::iter::once(&self.neither))
            .map(|v| v.to_bits())
            .collect()
    }
}

/// The recovered law `P(X1, X2)` on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ChainRecoveredWire {
    /// `P(X1 = x1, X2 = x2)`.
    pub cells: [[f64; 2]; 2],
    /// The rule version that produced it.
    pub rule_version: String,
}

impl ChainRecoveredWire {
    /// Encode a recovered law.
    #[must_use]
    pub fn of_law(law: &ChainRecoveredLaw) -> Self {
        Self { cells: *law.cells(), rule_version: law.rule_version().to_owned() }
    }

    /// Bit-exact replay; non-finite values fail closed.
    fn replays(&self, fresh: &Self) -> bool {
        self.rule_version == fresh.rule_version
            && self
                .cells
                .iter()
                .flatten()
                .zip(fresh.cells.iter().flatten())
                .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits())
    }

    fn bits(&self) -> Vec<u64> {
        self.cells.iter().flatten().map(|v| v.to_bits()).collect()
    }
}

/// Versioned ordered-response recovery decision with every premise needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryChainArtifactWire {
    /// Independent format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// The m-graph.
    pub graph: ChainGraphWire,
    /// Query roles.
    pub query: ChainQueryWire,
    /// Variable name of every graph node (dense order), or empty.
    pub variable_names: Vec<String>,
    /// `recovered` or `nonrecoverable`.
    pub outcome: String,
    /// The checked plan (recovered only).
    pub plan: Option<ChainPlanWire>,
    /// The observed pattern law (recovered only).
    pub observed: Option<ChainLawWire>,
    /// The recovered law (recovered only).
    pub recovered: Option<ChainRecoveredWire>,
    /// The witness (nonrecoverable only).
    pub witness: Option<ChainWitnessWire>,
    /// The verified witness check (nonrecoverable only).
    pub witness_check: Option<ChainWitnessCheckWire>,
    /// Digest of the canonical premises.
    pub premises_digest: String,
    /// Digest of the observed and recovered laws.
    pub data_digest: String,
}

#[derive(Serialize)]
struct PremisesView<'a> {
    tag: &'static str,
    graph: &'a ChainGraphWire,
    query: &'a ChainQueryWire,
    variable_names: &'a [String],
    outcome: &'a str,
    plan: &'a Option<ChainPlanWire>,
    witness: &'a Option<ChainWitnessWire>,
    witness_check: &'a Option<ChainWitnessCheckWire>,
}

#[derive(Serialize)]
struct DataView {
    tag: &'static str,
    observed: Option<Vec<u64>>,
    recovered: Option<Vec<u64>>,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

/// Everything a replay reconstructed and recomputed.
#[derive(Clone, Debug)]
pub struct ConsumedRecoveryChain {
    /// The re-decided decision.
    pub decision: ChainRecoveryDecision,
    /// The recomputed recovered law (recovered decision only).
    pub recovered: Option<ChainRecoveredLaw>,
    /// The re-verified witness check (nonrecoverable decision only).
    pub check: Option<ChainWitnessCheck>,
    /// The decoded artifact.
    pub wire: RecoveryChainArtifactWire,
}

impl RecoveryChainArtifactWire {
    fn base(
        graph: &Admg,
        query: &ChainRecoveryQuery,
        variable_names: &[String],
        outcome: &str,
    ) -> Result<Self, IoError> {
        Ok(Self {
            version: RECOVERY_CHAIN_ARTIFACT_VERSION,
            required_features: vec![RECOVERY_CHAIN_ARTIFACT_FEATURE.to_owned()],
            graph: ChainGraphWire::from_graph(graph)?,
            query: ChainQueryWire::from_query(query),
            variable_names: variable_names.to_vec(),
            outcome: outcome.to_owned(),
            plan: None,
            observed: None,
            recovered: None,
            witness: None,
            witness_check: None,
            premises_digest: String::new(),
            data_digest: String::new(),
        })
    }

    fn finish(mut self) -> Result<Self, IoError> {
        self.premises_digest = self.expected_premises_digest()?;
        self.data_digest = self.expected_data_digest()?;
        self.validate_shape()?;
        Ok(self)
    }

    /// Build the artifact of a recovered decision from its checked plan, the observed
    /// pattern law and the recovered law.
    ///
    /// # Errors
    /// A graph that does not encode, or inconsistent names.
    pub fn from_recovered(
        graph: &Admg,
        query: &ChainRecoveryQuery,
        variable_names: &[String],
        plan: &ChainRecoveryPlan,
        observed: &ChainPatternLaw,
        recovered: &ChainRecoveredLaw,
    ) -> Result<Self, IoError> {
        if !plan.is_intact() || plan.query != *query {
            return Err(IoError::Convert("the recovery plan is not the checked query plan".into()));
        }
        let mut wire = Self::base(graph, query, variable_names, RECOVERY_CHAIN_RECOVERED)?;
        wire.plan = Some(ChainPlanWire::of_plan(plan));
        wire.observed = Some(ChainLawWire::of_law(observed));
        wire.recovered = Some(ChainRecoveredWire::of_law(recovered));
        wire.finish()
    }

    /// Build the artifact of a nonrecoverable decision from its verified witness.
    ///
    /// # Errors
    /// A graph that does not encode, or inconsistent names.
    pub fn from_nonrecoverable(
        graph: &Admg,
        query: &ChainRecoveryQuery,
        variable_names: &[String],
        witness: &ChainRecoveryWitness,
        check: &ChainWitnessCheck,
    ) -> Result<Self, IoError> {
        let mut wire = Self::base(graph, query, variable_names, RECOVERY_CHAIN_NONRECOVERABLE)?;
        wire.witness = Some(ChainWitnessWire::of_witness(witness));
        wire.witness_check = Some(ChainWitnessCheckWire::of_check(check));
        wire.finish()
    }

    /// The premises digest the stored premises should carry. Recomputing it grants
    /// nothing: a consumer still re-decides and recomputes every premise.
    ///
    /// # Errors
    /// The premises do not encode.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        let view = PremisesView {
            tag: "ordered_response_recovery_premises_v1",
            graph: &self.graph,
            query: &self.query,
            variable_names: &self.variable_names,
            outcome: &self.outcome,
            plan: &self.plan,
            witness: &self.witness,
            witness_check: &self.witness_check,
        };
        Ok(crate::identity::digest_wire(IdentityDomain::IdentificationProduct, &view)?.to_hex())
    }

    /// The data digest the stored laws should carry.
    ///
    /// # Errors
    /// The laws do not encode.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        let view = DataView {
            tag: "ordered_response_recovery_data_v1",
            observed: self.observed.as_ref().map(ChainLawWire::bits),
            recovered: self.recovered.as_ref().map(ChainRecoveredWire::bits),
        };
        Ok(crate::identity::digest_wire(IdentityDomain::DataSnapshot, &view)?.to_hex())
    }

    fn validate_shape(&self) -> Result<(), RecoveryChainArtifactError> {
        let unsupported = RecoveryChainArtifactError::UnsupportedSemantics;
        if self.required_features != [RECOVERY_CHAIN_ARTIFACT_FEATURE] {
            return Err(unsupported("required features"));
        }
        let recovered_shape = self.plan.is_some()
            && self.observed.is_some()
            && self.recovered.is_some()
            && self.witness.is_none()
            && self.witness_check.is_none();
        let witness_shape = self.plan.is_none()
            && self.observed.is_none()
            && self.recovered.is_none()
            && self.witness.is_some()
            && self.witness_check.is_some();
        match self.outcome.as_str() {
            RECOVERY_CHAIN_RECOVERED if recovered_shape => {}
            RECOVERY_CHAIN_NONRECOVERABLE if witness_shape => {}
            RECOVERY_CHAIN_RECOVERED | RECOVERY_CHAIN_NONRECOVERABLE => {
                return Err(unsupported("the stored fields do not match the outcome"));
            }
            _ => return Err(unsupported("outcome")),
        }
        if !self.variable_names.is_empty()
            && (self.variable_names.len() != self.graph.nodes.len()
                || self.variable_names.iter().any(|name| name.trim().is_empty())
                || self.variable_names.iter().collect::<std::collections::BTreeSet<_>>().len()
                    != self.variable_names.len())
        {
            return Err(unsupported("variable names"));
        }
        Ok(())
    }

    /// Check a caller's variable names against the verified name mapping.
    ///
    /// # Errors
    /// [`RecoveryChainArtifactError::NamesMismatch`] unless `names` equals the mapping.
    pub fn check_variable_names(&self, names: &[String]) -> Result<(), RecoveryChainArtifactError> {
        if self.variable_names.as_slice() == names {
            Ok(())
        } else {
            Err(RecoveryChainArtifactError::NamesMismatch)
        }
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version before the payload is interpreted.
    ///
    /// # Errors
    /// [`IoError::UnsupportedVersion`], or a decoding or shape failure.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != RECOVERY_CHAIN_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        wire.validate_shape()?;
        Ok(wire)
    }

    /// Decode and recheck everything, then recompute the recovered law or re-verify
    /// the witness. No external provider is accessed.
    ///
    /// # Errors
    /// Any reconstruction failure, with the reason code and `recovery_chain.*` detail.
    pub fn consume(bytes: &[u8], ctx: &ExecutionContext) -> Result<ConsumedRecoveryChain, IoError> {
        Self::consume_typed(bytes, ctx).map_err(IoError::from)
    }

    /// [`Self::consume`] with the typed refusal kind, for callers and tests that
    /// match it.
    ///
    /// # Errors
    /// As [`Self::consume`].
    pub fn consume_typed(
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<ConsumedRecoveryChain, RecoveryChainArtifactError> {
        let undecodable = |e: IoError| RecoveryChainArtifactError::Undecodable(e.to_string());
        let wire = Self::decode(bytes).map_err(|e| match e {
            IoError::Refused { .. } => RecoveryChainArtifactError::UnsupportedSemantics("shape"),
            other => undecodable(other),
        })?;
        if wire.expected_premises_digest().map_err(undecodable)? != wire.premises_digest {
            return Err(RecoveryChainArtifactError::PremisesMismatch);
        }
        if wire.expected_data_digest().map_err(undecodable)? != wire.data_digest {
            return Err(RecoveryChainArtifactError::DataMismatch);
        }
        let graph = wire.graph.to_graph().map_err(undecodable)?;
        let query = wire.query.to_query();
        let decision = decide_chain_recovery(&graph, &query, ctx)?;
        let outcome = wire.outcome.clone();
        match (&decision, outcome.as_str()) {
            (ChainRecoveryDecision::Recovered(plan), RECOVERY_CHAIN_RECOVERED) => {
                let recovered = wire.replay_recovered(plan)?;
                Ok(ConsumedRecoveryChain {
                    decision,
                    recovered: Some(recovered),
                    check: None,
                    wire,
                })
            }
            (ChainRecoveryDecision::NonRecoverable(witness), RECOVERY_CHAIN_NONRECOVERABLE) => {
                let check = wire.replay_witness(witness, &graph, &query, ctx)?;
                Ok(ConsumedRecoveryChain { decision, recovered: None, check: Some(check), wire })
            }
            _ => Err(RecoveryChainArtifactError::DecisionMismatch("outcome")),
        }
    }

    fn replay_recovered(
        &self,
        plan: &ChainRecoveryPlan,
    ) -> Result<ChainRecoveredLaw, RecoveryChainArtifactError> {
        let (Some(stored_plan), Some(observed), Some(stored)) =
            (&self.plan, &self.observed, &self.recovered)
        else {
            return Err(RecoveryChainArtifactError::UnsupportedSemantics("recovered fields"));
        };
        if *stored_plan != ChainPlanWire::of_plan(plan) {
            return Err(RecoveryChainArtifactError::DecisionMismatch("plan"));
        }
        let law = observed.to_law()?;
        let recovered = evaluate_chain_recovery(plan, &law)?;
        if !stored.replays(&ChainRecoveredWire::of_law(&recovered)) {
            return Err(RecoveryChainArtifactError::RecoveredMismatch);
        }
        Ok(recovered)
    }

    fn replay_witness(
        &self,
        witness: &ChainRecoveryWitness,
        graph: &Admg,
        query: &ChainRecoveryQuery,
        ctx: &ExecutionContext,
    ) -> Result<ChainWitnessCheck, RecoveryChainArtifactError> {
        let (Some(stored_witness), Some(stored_check)) = (&self.witness, &self.witness_check)
        else {
            return Err(RecoveryChainArtifactError::UnsupportedSemantics("witness fields"));
        };
        if *stored_witness != ChainWitnessWire::of_witness(witness) {
            return Err(RecoveryChainArtifactError::DecisionMismatch("witness"));
        }
        let check = verify_chain_witness(graph, query, &stored_witness.to_witness(), ctx)?;
        if *stored_check != ChainWitnessCheckWire::of_check(&check) {
            return Err(RecoveryChainArtifactError::DecisionMismatch("witness check"));
        }
        Ok(check)
    }
}

/// Registered detail of a nonrecoverable decision, for callers rendering it.
#[must_use]
pub const fn nonrecoverable_detail() -> (&'static str, &'static str) {
    (
        ChainRecoveryDetail::NonrecoverableWitness.reason_code(),
        ChainRecoveryDetail::NonrecoverableWitness.detail(),
    )
}
