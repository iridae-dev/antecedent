//! Coupled abduction, action and prediction for a cross-world query (stage two).
//!
//! # Consistency is by construction; what is enforced is well-posed abduction
//!
//! The witness names the assumption `consistency`: a unit's observed values are
//! its values under the treatment it received. In this pipeline that holds *by
//! construction* and is not testable from the data: abduction defines each
//! exogenous term as the residual that makes the fitted mechanism regenerate the
//! observed value, so replaying it on the observed parents reproduces the data
//! for any data whatsoever. A round-trip "check" would be a tautology and is not
//! performed. What this stage enforces is that the abduction is well posed:
//! exact inversion ([`NoiseInferenceKind::Invertible`]; a posterior or
//! prior-drawn exogenous term does not pin the unit down) and a mechanism that is
//! invertible on the observed values. Otherwise it refuses with
//! [`CounterfactualError::AbductionNotExact`]. Model adequacy (that the fitted
//! mechanisms are the data-generating ones) is a named premise, not a check.
//!
//! One operation takes the factual table and the query and returns the observed
//! columns: it abducts the exogenous terms of every unit once, then evaluates
//! every world node by node under those same terms, each edge reading its
//! parent from the world its route names. There is no entry point that abducts
//! separately from predicting, so the shared-exogenous coupling cannot be
//! bypassed and a mechanism that is not additive in its noise (where the
//! exogenous term reaches the outcome through a parent) shows its per-unit
//! abduced value in the answer.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CrossWorldQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_model::{MechanismWorkspace, ParentBatch, evaluate_column};

use crate::engine::{AbductionMissingPolicy, CounterfactualEngine, NoiseInferenceKind};
use crate::error::CounterfactualError;

/// The two observed columns of a cross-world contrast, one entry per unit.
#[derive(Clone, Debug)]
pub struct CrossWorldEvaluation {
    /// The added observation per unit.
    pub plus: Arc<[f64]>,
    /// The subtracted observation per unit.
    pub minus: Arc<[f64]>,
    /// How the exogenous terms were obtained.
    pub noise: NoiseInferenceKind,
}

impl CrossWorldEvaluation {
    /// Units evaluated.
    #[must_use]
    pub fn n_units(&self) -> usize {
        self.plus.len()
    }

    /// Per-unit contrast `plus - minus`.
    #[must_use]
    pub fn unit_effects(&self) -> Vec<f64> {
        self.plus.iter().zip(self.minus.iter()).map(|(a, b)| a - b).collect()
    }

    /// The point: the mean of the per-unit contrasts.
    #[must_use]
    pub fn point(&self) -> f64 {
        let effects = self.unit_effects();
        effects.iter().sum::<f64>() / effects.len().max(1) as f64
    }
}

/// Abduce every unit's exogenous terms once, then act and predict every world of
/// `query` under them.
///
/// The cancellation token is polled before abduction and before every world of
/// every node, so a long evaluation stops with [`CounterfactualError::Cancelled`].
///
/// # Errors
///
/// A variable of the query missing from the model, a route naming an edge the
/// model does not have, abduction that cannot invert a mechanism, cancellation,
/// or a mechanism that is not fitted.
pub fn evaluate_cross_world(
    engine: &CounterfactualEngine,
    data: &TabularData,
    query: &CrossWorldQuery,
    ctx: &ExecutionContext,
) -> Result<CrossWorldEvaluation, CounterfactualError> {
    evaluate_probed(engine, data, query, ctx, &mut |_| {})
}

fn poll(ctx: &ExecutionContext) -> Result<(), CounterfactualError> {
    if ctx.cancellation.is_cancelled() { Err(CounterfactualError::Cancelled) } else { Ok(()) }
}

/// [`evaluate_cross_world`] with a hook called after every evaluated node, so a
/// test can cancel between nodes.
fn evaluate_probed(
    engine: &CounterfactualEngine,
    data: &TabularData,
    query: &CrossWorldQuery,
    ctx: &ExecutionContext,
    after_node: &mut dyn FnMut(usize),
) -> Result<CrossWorldEvaluation, CounterfactualError> {
    let model = &engine.model;
    poll(ctx)?;
    let exo = engine.abduct(data, AbductionMissingPolicy::Error, ctx).map_err(|e| match e {
        CounterfactualError::Model(inner @ antecedent_model::ModelError::Numerical { .. }) => {
            CounterfactualError::AbductionNotExact { message: inner.to_string() }
        }
        other => other,
    })?;
    if exo.kind != NoiseInferenceKind::Invertible {
        return Err(CounterfactualError::AbductionNotExact {
            message: format!(
                "abduction is not exact inversion ({:?}); the observed values do not determine \
                 the units' exogenous terms",
                exo.kind
            ),
        });
    }
    let n_units = exo.n_units;
    let n_nodes = model.n_nodes();
    let worlds = query.worlds();
    let dense = |variable: VariableId| {
        model.dense_of(variable).map(antecedent_graph::DenseNodeId::as_usize).ok_or_else(|| {
            CounterfactualError::model_msg(format!("variable {variable} is not in the model"))
        })
    };
    for world in worlds {
        for (variable, _) in world.interventions() {
            dense(variable)?;
        }
        for route in world.routes() {
            let (parent, child) = (dense(route.parent)?, dense(route.child)?);
            let is_edge = model.parent_gathers.iter().any(|g| {
                g.child.as_usize() == child && g.parents.iter().any(|p| p.as_usize() == parent)
            });
            if !is_edge {
                return Err(CounterfactualError::model_msg(format!(
                    "route {} -> {} is not an edge of the model",
                    route.parent, route.child
                )));
            }
        }
    }
    let mut values: Vec<Vec<f64>> = vec![vec![0.0; n_units * n_nodes]; worlds.len()];
    let mut parent_buf: Vec<f64> = Vec::new();
    let mut ws = MechanismWorkspace::default();
    // Parent gathers are in topological order, so every parent of a node is
    // final in every world before the node is evaluated.
    for (step, gather) in model.parent_gathers.iter().enumerate() {
        poll(ctx)?;
        let child = gather.child.as_usize();
        let child_variable = model.output_layout.variables[child];
        let need = gather.n_parents().max(1).saturating_mul(n_units);
        if parent_buf.len() < need {
            parent_buf.resize(need, 0.0);
        }
        for (index, world) in worlds.iter().enumerate() {
            if let Some(value) = world.intervention_of(child_variable) {
                values[index][child * n_units..(child + 1) * n_units].fill(value);
                continue;
            }
            for (slot, parent) in gather.parents.iter().enumerate() {
                let parent_variable = model.output_layout.variables[parent.as_usize()];
                let source = world
                    .routes()
                    .iter()
                    .find(|r| r.parent == parent_variable && r.child == child_variable)
                    .map_or(index, |r| r.source.index());
                let from = parent.as_usize() * n_units;
                parent_buf[slot * n_units..(slot + 1) * n_units]
                    .copy_from_slice(&values[source][from..from + n_units]);
            }
            let batch = ParentBatch {
                n_rows: n_units,
                n_parents: gather.n_parents(),
                values: &parent_buf[..gather.n_parents().saturating_mul(n_units)],
            };
            let noise = &exo.noise[child * n_units..(child + 1) * n_units];
            let out = &mut values[index][child * n_units..(child + 1) * n_units];
            evaluate_column(model.mechanisms.get(gather.child), batch, noise, out, &mut ws)?;
        }
        after_node(step);
    }
    let column = |observation: antecedent_core::WorldObservation| -> Result<Arc<[f64]>, CounterfactualError> {
        let node = dense(observation.variable)?;
        Ok(Arc::from(&values[observation.world.index()][node * n_units..(node + 1) * n_units]))
    };
    Ok(CrossWorldEvaluation {
        plus: column(query.plus())?,
        minus: column(query.minus())?,
        noise: exo.kind,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_data::TableView;
    use antecedent_graph::{Dag, DenseNodeId};
    use antecedent_model::{
        CompiledCausalModel, MechanismFamily, MechanismRegistry, SelectionPolicy,
    };

    fn fixture() -> (CounterfactualEngine, TabularData, CrossWorldQuery) {
        let n = 64usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
        let m: Vec<f64> =
            x.iter().enumerate().map(|(i, x)| 0.8 * x + (i as f64 * 0.91).cos()).collect();
        let y: Vec<f64> = x
            .iter()
            .zip(&m)
            .enumerate()
            .map(|(i, (x, m))| 1.7 * x + 4.0 * m + (i as f64 * 1.7).sin())
            .collect();
        let data = TabularData::from_f64_columns([
            ("x", x.as_slice()),
            ("m", m.as_slice()),
            ("y", y.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let compiled = CompiledCausalModel::compile(graph).unwrap();
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(
                &compiled,
                &data,
                SelectionPolicy::RequireFamily(MechanismFamily::LinearGaussian),
            )
            .unwrap();
        let v = VariableId::from_raw;
        let query = CrossWorldQuery::natural_direct(v(0), v(1), v(2), 0.0, 1.0).unwrap();
        (CounterfactualEngine::new(compiled.with_mechanisms(store)), data, query)
    }

    /// The token is polled before abduction: a pre-cancelled context does no work.
    #[test]
    fn a_pre_cancelled_context_stops_before_any_evaluation() {
        let (engine, data, query) = fixture();
        let ctx = ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        let mut visited = 0;
        let error = evaluate_probed(&engine, &data, &query, &ctx, &mut |_| visited += 1)
            .expect_err("cancelled");
        assert_eq!(error, CounterfactualError::Cancelled);
        assert_eq!(visited, 0);
        // The token is observed before abduction starts: a table on which abduction
        // itself would fail (a variable missing) is still reported as cancelled,
        // and as the abduction failure when the context is live.
        let missing = TabularData::from_f64_columns([
            ("x", data.float64_values(VariableId::from_raw(0)).unwrap().as_ref()),
            ("m", data.float64_values(VariableId::from_raw(1)).unwrap().as_ref()),
        ])
        .unwrap();
        let live = evaluate_cross_world(&engine, &missing, &query, &ExecutionContext::for_tests(1))
            .expect_err("abduction needs every variable");
        assert_ne!(live, CounterfactualError::Cancelled);
        assert_eq!(
            evaluate_cross_world(&engine, &missing, &query, &ctx).unwrap_err(),
            CounterfactualError::Cancelled
        );
    }

    /// The token is polled between nodes: cancelling after the first node stops
    /// the loop before the second is evaluated, and no partial answer escapes.
    #[test]
    fn a_context_cancelled_mid_run_stops_between_nodes() {
        let (engine, data, query) = fixture();
        let ctx = ExecutionContext::for_tests(2);
        let mut visited = Vec::new();
        let result = evaluate_probed(&engine, &data, &query, &ctx, &mut |step| {
            visited.push(step);
            if step == 0 {
                ctx.cancellation.cancel();
            }
        });
        assert_eq!(result.expect_err("cancelled"), CounterfactualError::Cancelled);
        assert_eq!(visited, vec![0], "no node after the cancelled one may be evaluated");
        // The same evaluation without cancellation visits every node and answers.
        let mut all = Vec::new();
        let ok =
            evaluate_probed(&engine, &data, &query, &ExecutionContext::for_tests(2), &mut |s| {
                all.push(s);
            })
            .unwrap();
        assert_eq!(all, vec![0, 1, 2]);
        assert_eq!(ok.n_units(), 64);
    }

    /// A route naming a pair that is not an edge of the model, and an
    /// intervention on a variable the model does not have, are refused before any
    /// world is built (they would otherwise be silently ignored).
    #[test]
    fn routes_and_interventions_outside_the_model_are_refused() {
        use antecedent_core::{EdgeRoute, ExogenousCoupling, WorldId, WorldObservation, WorldSpec};
        let (engine, data, _) = fixture();
        let ctx = ExecutionContext::for_tests(3);
        let v = VariableId::from_raw;
        let observe = |w: u8| WorldObservation { world: WorldId::new(w), variable: v(2) };
        let query = |worlds: Vec<WorldSpec>| {
            CrossWorldQuery::new(
                worlds,
                ExogenousCoupling::SharedAbducedExogenous,
                observe(1),
                observe(0),
            )
            .unwrap()
        };
        let base = WorldSpec::new([(v(0), 0.0)], []).unwrap();
        let backwards = EdgeRoute { parent: v(2), child: v(0), source: WorldId::new(0) };
        let bad_route =
            query(vec![base.clone(), WorldSpec::new([(v(0), 1.0)], [backwards]).unwrap()]);
        let text = evaluate_cross_world(&engine, &data, &bad_route, &ctx).unwrap_err().to_string();
        assert!(text.contains("is not an edge of the model"), "{text}");
        let unknown = query(vec![base, WorldSpec::new([(v(9), 1.0)], []).unwrap()]);
        let text = evaluate_cross_world(&engine, &data, &unknown, &ctx).unwrap_err().to_string();
        assert!(text.contains("is not in the model"), "{text}");
    }
}
