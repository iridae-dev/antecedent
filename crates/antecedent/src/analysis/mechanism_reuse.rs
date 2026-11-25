//! Reuse one fitted mechanism on a second attribution or simulation.
//!
//! The fit is the study's existing [`MechanismProgram`](super::structure_route::MechanismProgram)
//! fit. Nothing new is estimated. The retained object carries the observation digest
//! (`observation_identity_wire`), the score-reuse digest (`score_reuse_digest`, absent when
//! the study has no score table), and the data snapshot that digest binds. A second query
//! is prepared on its own, so its identification is its own. An observation mismatch
//! refuses before a snapshot mismatch. The admitted fit is installed with
//! [`MechanismProgram::with_prefit`](super::structure_route::MechanismProgram::with_prefit)
//! and the second study runs.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalQuery, ExecutionContext, SemanticDigest};
use antecedent_data::TabularData;

use super::builder::DataInput;
use super::structure_route::MechanismProgram;
use crate::error::CausalError;
use crate::gcm::FittedGcm;
use crate::result::StudyResult;
use crate::{PreparedStudy, Study};

/// A mechanism fit retained under the observation identity, score-reuse identity, and data
/// snapshot of the study that produced it.
#[derive(Debug)]
pub struct RetainedFittedMechanism {
    fitted: Arc<FittedGcm>,
    program: MechanismProgram,
    observation: SemanticDigest,
    score_reuse: Option<SemanticDigest>,
    data_snapshot: SemanticDigest,
    nodes: usize,
    edges: Vec<(u32, u32)>,
}

impl RetainedFittedMechanism {
    /// Observation digest the fit was retained under.
    #[must_use]
    pub const fn observation(&self) -> SemanticDigest {
        self.observation
    }

    /// Score-reuse digest of the source, when it has a score table.
    #[must_use]
    pub const fn score_reuse(&self) -> Option<SemanticDigest> {
        self.score_reuse
    }

    /// Data snapshot the fit was retained under.
    #[must_use]
    pub const fn data_snapshot(&self) -> SemanticDigest {
        self.data_snapshot
    }
}

/// Fit `source`'s mechanism program once and retain it under the identities that program
/// already publishes.
///
/// The source query has to be an attribution or a simulation: those are the routes that
/// execute `MechanismProgram::fit`.
///
/// # Errors
///
/// The source is not an attribution or simulation on one supplied DAG and table, or the
/// mechanism fit refuses.
pub fn retain_fitted_mechanism(
    source: &PreparedStudy,
) -> Result<RetainedFittedMechanism, CausalError> {
    let study = source.study();
    attribution_or_simulation(&study.query)?;
    let graph = supplied_dag(study)?;
    let data = table(study)?;
    let program = mechanism_program(study);
    let fitted = program.fit(graph, data)?;
    let identities = source.contract()?.identities;
    let score_reuse = source.score_reuse_identity()?;
    Ok(RetainedFittedMechanism {
        fitted,
        program,
        observation: identities.observation,
        score_reuse,
        data_snapshot: identities.data_snapshot,
        nodes: graph.node_count(),
        edges: graph.directed_edge_pairs_sorted(),
    })
}

/// Run `query` on `retained`.
///
/// Preparation of `query` writes its identification. The observation digest is compared
/// first, then the data snapshot, then the graph and the mechanism program. A match
/// installs the retained fit and runs `query`.
///
/// # Errors
///
/// `query` is not an attribution or simulation on one supplied DAG and table; its
/// observation, snapshot, graph, or mechanism program is not the one the fit was retained
/// under; or the run refuses.
pub fn evaluate_retained_mechanism(
    retained: &RetainedFittedMechanism,
    query: &Study,
    ctx: &ExecutionContext,
) -> Result<StudyResult, CausalError> {
    attribution_or_simulation(&query.query)?;
    let prepared = query.prepare(ctx)?;
    let identities = prepared.contract()?.identities;
    // The second query's score-reuse digest names its own identification. It is read and
    // not required to equal the retained one.
    let _score_reuse = prepared.score_reuse_identity()?;
    if identities.observation != retained.observation {
        return Err(CausalError::Conflict {
            what: "observation",
            detail: "fitted mechanism was retained under another observation identity",
        });
    }
    if identities.data_snapshot != retained.data_snapshot {
        return Err(CausalError::Conflict {
            what: "data_snapshot",
            detail: "fitted mechanism was retained under another data snapshot",
        });
    }
    let graph = supplied_dag(query)?;
    if graph.node_count() != retained.nodes || graph.directed_edge_pairs_sorted() != retained.edges
    {
        return Err(CausalError::Conflict {
            what: "graph",
            detail: "fitted mechanism was retained on another graph",
        });
    }
    if mechanism_program(query).label() != retained.program.label() {
        return Err(CausalError::Conflict {
            what: "mechanism_program",
            detail: "fitted mechanism was retained under another mechanism program",
        });
    }
    let study = install_prefit(query, &retained.program, &retained.fitted)?;
    study.run(ctx)
}

fn attribution_or_simulation(query: &CausalQuery) -> Result<(), CausalError> {
    if matches!(query, CausalQuery::Attribute(_) | CausalQuery::Simulate(_)) {
        return Ok(());
    }
    Err(CausalError::Unsupported {
        message: "a fitted mechanism is reused on an attribution or simulation query",
    })
}

fn supplied_dag(study: &Study) -> Result<&antecedent_graph::Dag, CausalError> {
    if study.graph_posterior.is_some() {
        return Err(CausalError::Unsupported {
            message: "a fitted mechanism is retained on one supplied DAG",
        });
    }
    study.graph.as_dag().ok_or(CausalError::Unsupported {
        message: "a fitted mechanism is retained on a supplied DAG",
    })
}

fn table(study: &Study) -> Result<&TabularData, CausalError> {
    match &study.data {
        DataInput::Tabular(data) => Ok(data),
        _ => Err(CausalError::Unsupported {
            message: "a fitted mechanism is retained from a table",
        }),
    }
}

fn mechanism_program(study: &Study) -> MechanismProgram {
    study.mechanisms.as_deref().cloned().unwrap_or_else(MechanismProgram::standard)
}

fn install_prefit(
    query: &Study,
    program: &MechanismProgram,
    fitted: &Arc<FittedGcm>,
) -> Result<Study, CausalError> {
    let data = table(query)?;
    let mut study = query.clone();
    let mut config = (*study.config).clone();
    config.mechanisms = Some(Arc::new(program.clone().with_prefit(Arc::clone(fitted), data)));
    study.config = Arc::new(config);
    Ok(study)
}

#[cfg(test)]
mod tests {
    use std::thread::Builder;

    use antecedent_core::{ExecutionContext, SimulateQuery, SimulationTarget, VariableId};
    use antecedent_data::TabularData;
    use antecedent_graph::{Dag, DenseNodeId};

    use super::{evaluate_retained_mechanism, retain_fitted_mechanism};
    use crate::result::{RuntimeRefutationAuthority, RuntimeRefutationOrigin};
    use crate::{CausalError, CausalQuery, RefuteSuite, Study};

    fn on_large_stack(body: impl FnOnce() + Send + 'static) {
        Builder::new()
            .name("mechanism-reuse".into())
            .stack_size(32 * 1024 * 1024)
            .spawn(body)
            .unwrap()
            .join()
            .unwrap();
    }

    fn chain(n: usize) -> (TabularData, Dag) {
        let x: Vec<f64> = (0..n).map(|i| f64::from(u32::try_from(i).unwrap()) * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|x| 1.0 + 2.0 * x).collect();
        let data =
            TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        (data, graph)
    }

    fn unit_effects(active: f64) -> CausalQuery {
        CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::UnitEffects {
            treatment: VariableId::from_raw(0),
            outcome: VariableId::from_raw(1),
            active,
            control: 0.0,
        }))
    }

    fn study(data: TabularData, graph: Dag, active: f64) -> Study {
        Study::tabular(data)
            .graph(graph)
            .query(unit_effects(active))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    }

    fn authority(result: &crate::StudyResult) -> (u8, Option<RuntimeRefutationOrigin>) {
        match &result.authority {
            RuntimeRefutationAuthority::Fresh => (0, None),
            RuntimeRefutationAuthority::Sealed { origin, .. } => (1, Some(*origin)),
            RuntimeRefutationAuthority::Invalidated => (2, None),
        }
    }

    #[test]
    fn second_query_reuses_a_fitted_mechanism_under_its_observation_identity() {
        on_large_stack(|| {
            let (data, graph) = chain(40);
            let ctx = ExecutionContext::for_tests(1);
            let source = study(data.clone(), graph.clone(), 1.0);
            let second = study(data.clone(), graph.clone(), 2.0);
            let prepared = source.prepare(&ctx).unwrap();
            let source_ids = prepared.contract().unwrap().identities;
            let retained = retain_fitted_mechanism(&prepared).unwrap();
            assert_eq!(retained.observation().as_bytes(), source_ids.observation.as_bytes());
            assert_eq!(retained.data_snapshot().as_bytes(), source_ids.data_snapshot.as_bytes());
            assert!(retained.score_reuse().is_none());

            let reused = evaluate_retained_mechanism(&retained, &second, &ctx).unwrap();
            let fresh = second.run(&ctx).unwrap();
            assert!(reused.effect().is_finite());
            assert_eq!(reused.effect().to_bits(), fresh.effect().to_bits());

            let reused_ids = reused.executed_contract().unwrap().identities;
            let fresh_ids = fresh.executed_contract().unwrap().identities;
            assert_eq!(reused_ids.identification.as_bytes(), fresh_ids.identification.as_bytes());
            assert_ne!(reused_ids.identification.as_bytes(), source_ids.identification.as_bytes());
            assert_eq!(reused_ids.observation.as_bytes(), fresh_ids.observation.as_bytes());
            assert_eq!(reused_ids.observation.as_bytes(), retained.observation().as_bytes());

            let contract = second.prepare(&ctx).unwrap().contract().unwrap();
            let reused_claim = reused.claim(&contract, &ctx).unwrap();
            let fresh_claim = fresh.claim(&contract, &ctx).unwrap();
            assert_eq!(reused_claim.kind, fresh_claim.kind);
            assert_eq!(reused_claim.domains, fresh_claim.domains);
            assert_eq!(&*reused_claim.calibration.status, &*fresh_claim.calibration.status);
            assert_eq!(
                reused_claim.identities.observation.as_bytes(),
                fresh_claim.identities.observation.as_bytes()
            );
            assert_eq!(authority(&reused), authority(&fresh));

            let delayed = Study::tabular(data.clone())
                .graph(graph.clone())
                .query(unit_effects(2.0))
                .observation_delayed_entry(VariableId::from_raw(0))
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            let refused = evaluate_retained_mechanism(&retained, &delayed, &ctx).unwrap_err();
            assert!(matches!(refused, CausalError::Conflict { what: "observation", .. }));

            let x: Vec<f64> = (0..40_u32).map(|i| f64::from(i) * 0.1).collect();
            let mut y: Vec<f64> = x.iter().map(|x| 1.0 + 2.0 * x).collect();
            y[0] += 1.0;
            let shifted =
                TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
            let changed = study(shifted, graph, 2.0);
            let refused = evaluate_retained_mechanism(&retained, &changed, &ctx).unwrap_err();
            assert!(matches!(refused, CausalError::Conflict { what: "data_snapshot", .. }));
        });
    }
}
