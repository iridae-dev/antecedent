//! Typed per-atom Bayesian fits for a static DAG posterior envelope.
//!
//! The graph wrapper resolves one prior against the first identified atom before
//! any interactive selection, then passes the same resolved prior to every fit.
//! Atom order and posterior draw columns remain explicit for the envelope mixer.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// One identified DAG atom ready for Bayesian g-computation.
#[derive(Clone, Debug)]
pub(crate) struct BayesianGraphPosteriorAtomInput {
    pub(crate) key: u64,
    pub(crate) weight: f64,
    pub(crate) status: IdentificationStatus,
    pub(crate) estimand: IdentifiedEstimand,
    pub(crate) prepared: PreparedBayesianProblem,
}

/// Prior resolved against the first identified atom in original posterior order.
#[derive(Clone, Debug)]
pub(crate) struct BayesianGraphPosteriorPriorAnchor {
    pub(crate) key: u64,
    pub(crate) prior: Option<PriorSet>,
    pub(crate) conflict: Option<antecedent_prob::ConflictSummary>,
}

impl BayesianGraphPosteriorPriorAnchor {
    /// Resolve the envelope-wide prior before subsampling or fitting later atoms.
    pub(crate) fn resolve(
        key: u64,
        config: &BayesianConfig,
        first: &PreparedBayesianProblem,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        let (prior, conflict) = resolve_envelope_prior_anchor(config, first, ctx)?;
        Ok(Self { key, prior, conflict })
    }
}

/// One atom fit retaining its own target, graph mass, prepared design, and draw matrix.
#[derive(Debug)]
pub(crate) struct BayesianGraphPosteriorAtomFit {
    pub(crate) key: u64,
    pub(crate) weight: f64,
    pub(crate) status: IdentificationStatus,
    pub(crate) estimand: IdentifiedEstimand,
    pub(crate) prepared: PreparedBayesianProblem,
    pub(crate) posterior: CausalPosterior,
    /// Resolved envelope prior, identical to the prior anchor for every atom.
    pub(crate) prior: Option<PriorSet>,
}

/// Ordered per-atom fit batch with reusable draw columns for envelope aggregation.
#[derive(Debug)]
pub(crate) struct BayesianGraphPosteriorAtomFits {
    pub(crate) prior_anchor: BayesianGraphPosteriorPriorAnchor,
    pub(crate) atoms: Vec<BayesianGraphPosteriorAtomFit>,
    pub(crate) draw_columns: Vec<GraphEffectDraws>,
}

impl BayesianGraphPosteriorAtomFits {
    /// Fit in stable atom order with one prior and indexed execution contexts.
    ///
    /// Each returned draw column has the same draw count and occupies the same
    /// draw index in the envelope. The graph wrapper can pass these columns to
    /// `aggregate_effect_envelope` without losing atom identity or mass.
    pub(crate) fn fit(
        mut estimator: BayesianGComputationAte,
        anchor: BayesianGraphPosteriorPriorAnchor,
        inputs: Vec<BayesianGraphPosteriorAtomInput>,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        if inputs.is_empty() {
            return Err(CausalError::Conflict {
                what: "Bayesian graph-posterior prior anchor",
                detail: "an anchored prior and at least one retained identified atom are required"
                    .into(),
            });
        }
        if inputs
            .iter()
            .position(|atom| atom.key == anchor.key)
            .is_some_and(|position| position != 0)
        {
            return Err(CausalError::Conflict {
                what: "Bayesian graph-posterior prior anchor",
                detail: "when retained, the original first identified atom must remain first"
                    .into(),
            });
        }
        let mut keys = std::collections::HashSet::with_capacity(inputs.len());
        for atom in &inputs {
            if !keys.insert(atom.key)
                || !atom.weight.is_finite()
                || atom.weight < 0.0
                || atom.prepared.design.ncols == 0
            {
                return Err(CausalError::Compile {
                    message: "Bayesian graph-posterior atom keys, weights, or designs are invalid"
                        .into(),
                });
            }
        }
        estimator.prior.clone_from(&anchor.prior);
        let fitted = ctx.map_indexed(inputs.len(), |index, inner| {
            let atom = &inputs[index];
            let mut per_atom = estimator.clone();
            // A single resolved prior is anchored before any latency selection;
            // no atom may independently reinterpret the source prior.
            per_atom.prior.clone_from(&anchor.prior);
            let mut workspace = BayesianGCompWorkspace::default();
            let posterior = per_atom
                .fit(&atom.prepared, atom.status, &mut workspace, inner)
                .map_err(CausalError::from)?;
            if posterior.draws.n_draws != estimator.n_draws {
                return Err(CausalError::Compile {
                    message: "Bayesian graph-posterior atoms produced different draw counts".into(),
                });
            }
            Ok::<_, CausalError>((index, posterior, per_atom.prior))
        })?;
        let mut atoms = Vec::with_capacity(inputs.len());
        let mut draw_columns = Vec::with_capacity(inputs.len());
        for (input, (index, posterior, prior)) in inputs.into_iter().zip(fitted) {
            if index != atoms.len() {
                return Err(CausalError::Compile {
                    message: "Bayesian graph-posterior atom order changed during fit".into(),
                });
            }
            draw_columns.push(graph_effect_draws(input.key, &posterior)?);
            atoms.push(BayesianGraphPosteriorAtomFit {
                key: input.key,
                weight: input.weight,
                status: input.status,
                estimand: input.estimand,
                prepared: input.prepared,
                posterior,
                prior,
            });
        }
        Ok(Self { prior_anchor: anchor, atoms, draw_columns })
    }
}

fn graph_effect_draws(
    key: u64,
    posterior: &CausalPosterior,
) -> Result<GraphEffectDraws, CausalError> {
    let column = posterior.effect_column().ok_or_else(|| CausalError::Compile {
        message: "Bayesian posterior missing its effect column".into(),
    })?;
    let draws = posterior
        .draws
        .column(column)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(GraphEffectDraws { graph_key: key, effect_draws: Arc::from(draws.to_vec()) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy_table::identify_static;
    use antecedent_core::ExecutionContext;
    use antecedent_data::{TableView, TabularData};
    use antecedent_estimate::BayesianGComputationAte;
    use antecedent_graph::Dag;

    fn data(n: usize) -> TabularData {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for row in 0..n {
            let treatment = f64::from(row % 2 == 0);
            let noise = ((row * 37 % 101) as f64 - 50.0) / 50.0;
            x.push(treatment);
            y.push(2.0 * treatment + noise);
        }
        TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
    }

    #[test]
    fn shared_prior_anchor_produces_ordered_graph_mass_and_aligned_draw_columns() {
        let data = data(600);
        let x = data.schema().id_of("x").unwrap();
        let y = data.schema().id_of("y").unwrap();
        let query = AverageEffectQuery::binary_ate(x, y);
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let identification =
            identify_static(IdentifierId::BackdoorAdjustment, &graph, &query).unwrap();
        let estimand = identification.estimands[0].clone();
        let mut estimator = BayesianGComputationAte::conjugate();
        estimator.n_draws = 128;
        estimator.seed = 81;
        let prepared = estimator.prepare(&data, &estimand, &query).unwrap();
        let anchor_prior = PriorSet::weakly_informative(prepared.design.ncols);
        let mut config = BayesianConfig::conjugate();
        config.n_draws = 128;
        let ctx = ExecutionContext::for_tests(81);
        let anchor = BayesianGraphPosteriorPriorAnchor {
            key: 11,
            prior: Some(anchor_prior.clone()),
            conflict: None,
        };
        let inputs = [11_u64, 17]
            .into_iter()
            .enumerate()
            .map(|(index, key)| BayesianGraphPosteriorAtomInput {
                key,
                weight: if index == 0 { 0.4 } else { 0.6 },
                status: identification.status,
                estimand: estimand.clone(),
                prepared: prepared.clone(),
            })
            .collect();
        let fits = BayesianGraphPosteriorAtomFits::fit(estimator, anchor, inputs, &ctx).unwrap();
        assert_eq!(fits.prior_anchor.prior.as_ref().unwrap().specs, anchor_prior.specs);
        assert_eq!(fits.atoms.iter().map(|atom| atom.key).collect::<Vec<_>>(), [11, 17]);
        assert_eq!(fits.atoms.iter().map(|atom| atom.weight).collect::<Vec<_>>(), [0.4, 0.6]);
        assert_eq!(fits.atoms.len(), 2);
        assert_eq!(fits.draw_columns.len(), 2);
        for atom in &fits.atoms {
            assert_eq!(atom.posterior.draws.n_draws, config.n_draws);
            assert_eq!(atom.prior.as_ref().unwrap().specs, anchor_prior.specs);
            assert_eq!(atom.prepared.design.ncols, prepared.design.ncols);
        }
        assert!(fits.draw_columns.iter().all(|column| column.effect_draws.len() == 128));
        assert!(
            fits.draw_columns
                .iter()
                .flat_map(|column| column.effect_draws.iter())
                .all(|draw| draw.is_finite())
        );
    }
}
