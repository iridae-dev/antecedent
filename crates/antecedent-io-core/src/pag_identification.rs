//! Portable replay of the direct marginal and conditional PAG identification algorithms.

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::execution::{AdmissionError, AdmissionLease};
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest, VariableId};
use antecedent_expr::{
    Assignment, ExactDiscreteLaw, ExactDistribution, ExactEvaluationLimits, ExactEvaluationPlan,
    ExactEvaluationSpec, ExactTransportData, LawTolerance,
};
pub use antecedent_identify::pag_id::PagIdError;
use antecedent_identify::pag_id::{PagFunctional, PagObstruction, identify_pag_direct};
use serde::{Deserialize, Serialize};

use crate::convert::{from_cbor, pag_from_wire, to_cbor};
use crate::expr_wire::{ExprArenaWire, expr_arena_to_wire};
use crate::{IoError, PagWire};

const MAGIC: &[u8] = b"ANTE-PAG-ID-1\0";
const MAX_BYTES: usize = 16 * 1024 * 1024;

/// One oracle PAG query under the no-selection, strictly positive-law premises.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PagIdentificationProblem {
    /// Complete invariant graph marks of the declared Markov equivalence class.
    pub graph: PagWire,
    /// Unique stable names in graph variable order.
    pub names: Vec<String>,
    /// Hard intervention coordinates, sorted ascending.
    pub actions: Vec<u32>,
    /// Outcome coordinates, sorted ascending.
    pub outcomes: Vec<u32>,
    /// Observed conditioning coordinates, sorted ascending.
    pub given: Vec<u32>,
    /// Observational population of the named graph variables.
    pub population: String,
}

impl PagIdentificationProblem {
    /// Validate the compact declaration before symbolic work.
    ///
    /// # Errors
    /// Inconsistent graph names or disjoint query sets, or an absent population.
    pub fn validate(&self) -> Result<(), PagReplayError> {
        if self.names.len() != self.graph.node_count as usize
            || self.names.iter().any(|name| name.trim().is_empty())
            || self.names.iter().collect::<BTreeSet<_>>().len() != self.names.len()
            || self.population.trim().is_empty()
            || self.outcomes.is_empty()
        {
            return Err(PagReplayError::Invalid("invalid PAG names, population or outcomes"));
        }
        let mut variable_ids = BTreeSet::new();
        for ids in [&self.actions, &self.outcomes, &self.given] {
            if !ids.windows(2).all(|pair| pair[0] < pair[1])
                || !ids.iter().all(|id| *id < self.graph.node_count && variable_ids.insert(id))
            {
                return Err(PagReplayError::Invalid("invalid, unsorted or overlapping query sets"));
            }
        }
        Ok(())
    }

    /// Identity of the orientation and query premises, independently of the derived proof.
    ///
    /// # Errors
    /// Invalid declaration or encoding failure.
    pub fn identity(&self) -> Result<SemanticDigest, PagReplayError> {
        self.validate()?;
        Ok(SemanticDigest::from_bytes(CanonicalWriter::digest_bytes(
            "antecedent.pag_identification.problem.v1.oracle_no_selection_positive",
            &to_cbor(self)?,
        )))
    }
}

/// A deterministic search stop that a portable consumer can independently reproduce.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PagResourceStop {
    /// Work budget exhausted.
    Work,
    /// Graph-path or expression recursion depth exceeded.
    Depth,
}

/// Retained functional or graphical obstruction of one direct PAG query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum PagIdentificationProof {
    /// Observational functional returned by the complete algorithm.
    Identified {
        /// Interned expression arena.
        arena: ExprArenaWire,
        /// Root of the function of the observational law.
        root: u32,
    },
    /// A bounded execution ended before establishing either identification conclusion.
    ResourceStop {
        /// Which deterministic limit stopped the search.
        limit: PagResourceStop,
        /// Metered work before the stop.
        work: u64,
        /// Caller work budget at capture.
        work_limit: u64,
        /// Algorithm depth ceiling at capture when the depth limit stopped execution.
        #[serde(skip_serializing_if = "Option::is_none", default)]
        depth_limit: Option<usize>,
    },
    /// Marginal IDP's final unsolved subproblem.
    MarginalObstruction {
        /// Variables still available.
        available: Vec<u32>,
        /// Requested subkernel variables.
        required: Vec<u32>,
    },
    /// Conditional IDP's blocked rule 2 exchange.
    ConditionalObstruction {
        /// Bucket split by the intervention.
        bucket: Vec<u32>,
        /// Intervened part of that bucket.
        actions: Vec<u32>,
        /// Definite m-connecting path in the manipulated graph.
        path: Vec<u32>,
    },
}

/// What stopped a direct PAG operation.
#[derive(Debug, thiserror::Error)]
pub enum PagReplayError {
    /// Graph search, work or admission failure.
    #[error(transparent)]
    Identification(#[from] PagIdError),
    /// Invalid or mismatched retained inputs.
    #[error("direct PAG identification: {0}")]
    Invalid(&'static str),
    /// The original graph or derived expression could not be encoded.
    #[error(transparent)]
    Io(#[from] IoError),
    /// A supplied exact analysis table or its evaluation failed.
    #[error(transparent)]
    Evaluation(#[from] antecedent_expr::EvalError),
    /// Unsupported compiler route at the current support gate.
    #[error(transparent)]
    Unsupported(#[from] antecedent_core::NotSupported),
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    problem: PagIdentificationProblem,
    proof: PagIdentificationProof,
    work: u64,
    identity: String,
}

/// An independently replayable identification result and its original premises.
#[derive(Clone, Debug)]
pub struct PagIdentificationArtifact {
    problem: PagIdentificationProblem,
    proof: PagIdentificationProof,
    functional: Option<PagFunctional>,
    work: u64,
    identity: String,
}

fn raw(ids: Vec<VariableId>) -> Vec<u32> {
    ids.into_iter().map(VariableId::raw).collect()
}

fn admit_intake(bytes: u64, ctx: &ExecutionContext) -> Result<AdmissionLease, PagIdError> {
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        AdmissionError::Cancelled => PagIdError::Cancelled,
        other => PagIdError::Admission(other),
    })
}

impl PagIdentificationArtifact {
    /// Run the direct identification algorithm with the caller's resources.
    ///
    /// # Errors
    /// Invalid premises, cancellation or a resource stop.
    pub fn execute(
        problem: PagIdentificationProblem,
        ctx: &ExecutionContext,
    ) -> Result<Self, PagReplayError> {
        Self::execute_inner(problem, ctx, false)
    }

    /// Retain a reproducible operation or depth stop as a portable result.
    ///
    /// # Errors
    /// Invalid input, cancellation or an environment-dependent admission failure.
    pub fn execute_retaining_stop(
        problem: PagIdentificationProblem,
        ctx: &ExecutionContext,
    ) -> Result<Self, PagReplayError> {
        Self::execute_inner(problem, ctx, true)
    }

    fn execute_inner(
        problem: PagIdentificationProblem,
        ctx: &ExecutionContext,
        retain_stop: bool,
    ) -> Result<Self, PagReplayError> {
        // Hold graph and declaration intake apart from the symbolic kernel's own lease.
        let graph_bytes = u64::from(problem.graph.node_count)
            .saturating_mul(64)
            .saturating_add((problem.graph.edges.len() as u64).saturating_mul(32));
        let label_bytes =
            problem.names.iter().fold(problem.population.len() as u64, |sum, name| {
                sum.saturating_add(name.len() as u64)
            });
        let _graph_lease = admit_intake(graph_bytes.saturating_add(label_bytes), ctx)?;
        let problem_identity = problem.identity()?;
        let graph = pag_from_wire(&problem.graph)?;
        let to_ids =
            |ids: &[u32]| ids.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>();
        let observed = identify_pag_direct(
            &graph,
            &to_ids(&problem.actions),
            &to_ids(&problem.outcomes),
            &to_ids(&problem.given),
            &problem.population,
            ctx,
        );
        let identified = match observed {
            Ok(result) => result,
            Err(PagIdError::Budget(work)) if retain_stop => {
                return Self::seal(
                    problem,
                    problem_identity,
                    PagIdentificationProof::ResourceStop {
                        limit: PagResourceStop::Work,
                        work,
                        work_limit: ctx.id_work_limit(),
                        depth_limit: None,
                    },
                    None,
                    work,
                );
            }
            Err(PagIdError::Depth { work, limit }) if retain_stop => {
                return Self::seal(
                    problem,
                    problem_identity,
                    PagIdentificationProof::ResourceStop {
                        limit: PagResourceStop::Depth,
                        work,
                        work_limit: ctx.id_work_limit(),
                        depth_limit: Some(limit),
                    },
                    None,
                    work,
                );
            }
            Err(error) => return Err(error.into()),
        };
        let (proof, functional) = match identified.result {
            Ok(functional) => (
                PagIdentificationProof::Identified {
                    arena: expr_arena_to_wire(&functional.arena)?,
                    root: functional.root.raw(),
                },
                Some(functional),
            ),
            Err(PagObstruction::Marginal { available, required }) => (
                PagIdentificationProof::MarginalObstruction {
                    available: raw(available),
                    required: raw(required),
                },
                None,
            ),
            Err(PagObstruction::Conditional { bucket, interventions, path }) => (
                PagIdentificationProof::ConditionalObstruction {
                    bucket: raw(bucket),
                    actions: raw(interventions),
                    path: raw(path),
                },
                None,
            ),
        };
        Self::seal(problem, problem_identity, proof, functional, identified.work)
    }

    fn seal(
        problem: PagIdentificationProblem,
        problem_identity: SemanticDigest,
        proof: PagIdentificationProof,
        functional: Option<PagFunctional>,
        work: u64,
    ) -> Result<Self, PagReplayError> {
        let mut canon = CanonicalWriter::new("antecedent.pag_identification.result.v1");
        canon.bytes(problem_identity.as_bytes()).bytes(&to_cbor(&proof)?).word(work);
        let identity = canon.finish_hex();
        Ok(Self { problem, proof, functional, work, identity })
    }

    /// Retained result identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The graph and query that established the result.
    #[must_use]
    pub const fn problem(&self) -> &PagIdentificationProblem {
        &self.problem
    }

    /// Result's expression or obstruction.
    #[must_use]
    pub const fn proof(&self) -> &PagIdentificationProof {
        &self.proof
    }

    /// Metered work completed.
    #[must_use]
    pub const fn work(&self) -> u64 {
        self.work
    }

    /// Export the original premises and the derived proof.
    ///
    /// # Errors
    /// Encoding failure or oversized artifact.
    pub fn to_bytes(&self) -> Result<Vec<u8>, PagReplayError> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend(to_cbor(&Wire {
            problem: self.problem.clone(),
            proof: self.proof.clone(),
            work: self.work,
            identity: self.identity.clone(),
        })?);
        if bytes.len() > MAX_BYTES {
            return Err(IoError::TooLarge.into());
        }
        Ok(bytes)
    }

    /// Discover the version of the direct PAG artifact.
    #[must_use]
    pub fn recognizes(bytes: &[u8]) -> bool {
        bytes.starts_with(MAGIC)
    }

    /// Verify the recorded proof by independently rerunning identification.
    ///
    /// # Errors
    /// Unknown version, changed declaration or proof, wrong retained identity, or resource stop.
    pub fn consume(
        bytes: &[u8],
        expected: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<Self, PagReplayError> {
        if bytes.len() > MAX_BYTES {
            return Err(IoError::TooLarge.into());
        }
        let body = bytes
            .strip_prefix(MAGIC)
            .ok_or(PagReplayError::Invalid("unknown PAG artifact version"))?;
        // Keep the decoded proof alive until it has been compared with the fresh result.
        let _wire_lease = admit_intake((body.len() as u64).saturating_mul(8), ctx)?;
        let wire: Wire = from_cbor(body)?;
        if expected.is_some_and(|id| id != wire.identity) {
            return Err(PagReplayError::Invalid("unexpected retained PAG result"));
        }
        let live = if let PagIdentificationProof::ResourceStop { work_limit, .. } = &wire.proof {
            if *work_limit > ctx.id_work_limit() {
                return Err(PagIdError::Budget(ctx.id_work_limit()).into());
            }
            let replay_ctx = ctx.clone().with_id_work_limit(*work_limit);
            Self::execute_retaining_stop(wire.problem, &replay_ctx)?
        } else {
            Self::execute(wire.problem, ctx)?
        };
        if live.identity != wire.identity || live.proof != wire.proof || live.work != wire.work {
            return Err(PagReplayError::Invalid("stored PAG derivation does not replay"));
        }
        Ok(live)
    }

    /// Evaluate the retained functional on a declared, strictly positive finite joint law.
    ///
    /// The exact law must obey the independences of the declared oracle PAG and use
    /// the original population and observational regime 0. Graph compatibility is
    /// a scientific premise of the supplied law.
    /// Supply the original intervention and conditioning assignments, even when CIDP
    /// exchanged their roles in the formula. Nuisance variables absent from the query
    /// that still occur syntactically in the elimination formula are fixed to their
    /// first supported level. The IDP identities make the result independent of them.
    ///
    /// # Errors
    /// An unidentified query, wrong assignment coordinates, incomplete or nonpositive law,
    /// unsupported conditional event or exhausted evaluation resources.
    pub fn evaluate(
        &self,
        joint: ExactDiscreteLaw,
        bindings: &Assignment,
        limits: ExactEvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<ExactDistribution, PagReplayError> {
        let functional = self
            .functional
            .as_ref()
            .ok_or(PagReplayError::Invalid("PAG result is not identified"))?;
        let expected: BTreeSet<_> = self
            .problem
            .actions
            .iter()
            .chain(&self.problem.given)
            .copied()
            .map(VariableId::from_raw)
            .collect();
        let supplied: BTreeSet<_> = bindings.entries().iter().map(|(id, _)| *id).collect();
        let axes: BTreeSet<_> = joint.axes().iter().map(|axis| axis.variable.raw()).collect();
        if expected != supplied
            || axes != (0..self.problem.graph.node_count).collect()
            || joint.population() != self.problem.population
            || !joint.interventions().is_empty()
            || joint.regime().raw() != 0
            || joint.probabilities().iter().any(|p| *p <= 0.0)
        {
            return Err(PagReplayError::Invalid("the declared joint law or query binding differs"));
        }
        if bindings.entries().iter().any(|(variable, value)| {
            !joint
                .axes()
                .iter()
                .any(|axis| axis.variable == *variable && axis.values.contains(value))
        }) {
            return Err(PagReplayError::Invalid("a query assignment is outside the joint support"));
        }
        let outcomes: Arc<[VariableId]> = Arc::from(functional.outcomes.clone());
        let mut request = bindings.clone();
        for axis in joint.axes() {
            if !outcomes.contains(&axis.variable) && request.get(axis.variable).is_none() {
                request.set(axis.variable, axis.values[0].clone());
            }
        }
        let rows = joint.probabilities().len();
        let data = ExactTransportData::try_new([joint], rows)
            .map_err(|error| antecedent_expr::EvalError::ExactLaw(Box::new(error)))?;
        let plan = ExactEvaluationPlan::compile(
            ExactEvaluationSpec {
                arena: &functional.arena,
                root: functional.root,
                outcomes,
                request,
                limits,
                tolerance: LawTolerance::default(),
            },
            data,
            ctx,
        )?;
        Ok(plan.evaluate(ctx)?)
    }
}
