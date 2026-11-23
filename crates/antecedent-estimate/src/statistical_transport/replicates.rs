use super::{
    Arc, EstimationError, ExactDistribution, TransportIntervalReason, VariableId,
    percentile_interval, refuse_eval,
};

/// Per-atom and per-outcome-mean replicate columns of one request.
///
/// Every interval constructor on this path records its successful replicates
/// here and summarizes them through one rule, so the classical bootstrap, the
/// nominal z-transport bootstrap, and both posterior draw paths cannot drift
/// in how they check atom support or compute an equal-tail interval.
#[derive(Clone, Debug, Default)]
pub(super) struct ReplicateColumns {
    pub(super) outcomes: Vec<VariableId>,
    atoms: Vec<Vec<f64>>,
    pub(super) means: Vec<Vec<f64>>,
    pub(super) ok: u32,
}

impl ReplicateColumns {
    pub(super) fn new(outcomes: &[VariableId]) -> Self {
        Self {
            outcomes: outcomes.to_vec(),
            atoms: Vec::new(),
            means: vec![Vec::new(); outcomes.len()],
            ok: 0,
        }
    }

    /// Record one successful replicate distribution.
    pub(super) fn record(
        &mut self,
        distribution: &ExactDistribution,
    ) -> Result<(), EstimationError> {
        if self.ok > 0 && self.atoms.len() != distribution.probabilities.len() {
            return Err(EstimationError::refused(
                antecedent_core::reason_code!("transport_numerical_failure"),
                "transport atom support changed between replicates",
            ));
        }
        for (column, outcome) in self.means.iter_mut().zip(&self.outcomes) {
            column.push(distribution.mean(*outcome).map_err(|error| refuse_eval(&error))?);
        }
        for (atom, probability) in distribution.probabilities.iter().enumerate() {
            if self.atoms.len() == atom {
                self.atoms.push(Vec::new());
            }
            self.atoms[atom].push(*probability);
        }
        self.ok = self.ok.saturating_add(1);
        Ok(())
    }

    /// Replicate rows (replicate × atom), in replicate order.
    pub(super) fn atom_rows(&self) -> Vec<Arc<[f64]>> {
        (0..self.ok as usize)
            .map(|row| self.atoms.iter().map(|column| column[row]).collect())
            .collect()
    }

    /// Per-outcome mean replicates.
    pub(super) fn mean_columns(&self) -> Arc<[(VariableId, Arc<[f64]>)]> {
        self.outcomes
            .iter()
            .zip(&self.means)
            .map(|(outcome, column)| (*outcome, Arc::from(column.as_slice())))
            .collect()
    }

    /// Equal-tail intervals of every atom and outcome mean, or the reason none
    /// can be published.
    pub(super) fn intervals(
        &self,
        level: f64,
    ) -> Result<PointwiseIntervals, TransportIntervalReason> {
        let atoms = self
            .atoms
            .iter()
            .map(|column| percentile_interval(column, level))
            .collect::<Option<Arc<[_]>>>()
            .ok_or(TransportIntervalReason::NumericalFailure)?;
        let means = self
            .outcomes
            .iter()
            .zip(&self.means)
            .map(|(outcome, column)| {
                percentile_interval(column, level).map(|(lower, upper)| (*outcome, lower, upper))
            })
            .collect::<Option<Arc<[_]>>>()
            .ok_or(TransportIntervalReason::NumericalFailure)?;
        Ok(PointwiseIntervals { atoms, means })
    }
}

/// Pointwise atom and outcome-mean intervals.
pub(super) struct PointwiseIntervals {
    pub(super) atoms: Arc<[(f64, f64)]>,
    pub(super) means: Arc<[(VariableId, f64, f64)]>,
}
