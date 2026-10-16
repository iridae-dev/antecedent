// SPDX-License-Identifier: MIT OR Apache-2.0

use super::panel_path::PanelClusters;
use super::{
    CausalError, ExecutionContext, PanelData, PhysicalExecutionPlan, StudyResult, TemporalDag,
    TemporalEffectQuery,
};

impl super::Study {
    /// Multi-environment Pulse / single-step Sustained: the environments are the
    /// clusters of one pooled panel regression.
    ///
    /// Reading only `environment(0)` would answer the query on one environment and
    /// publish it as the multi-environment claim, and the answer would depend on the
    /// argument order.
    pub(super) fn execute_multi_env_temporal(
        &self,
        multi: &antecedent_data::MultiEnvironmentData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let units: Vec<antecedent_data::PanelUnit> = multi
            .environments()
            .iter()
            .enumerate()
            .map(|(index, series)| antecedent_data::PanelUnit {
                unit_id: u32::try_from(index).unwrap_or(u32::MAX),
                series: series.clone(),
            })
            .collect();
        let panel = PanelData::try_new(units)
            .map_err(|error| CausalError::Compile { message: format!("multi-env: {error}") })?;
        self.execute_pooled_panel(&panel, graph, query, physical, ctx, PanelClusters::Environments)
    }
}
