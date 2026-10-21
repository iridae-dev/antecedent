//! Quantity-bound downstream ports from the retained compiled program.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointPanelError, JointPanelFit, JointPanelResult, contracts, engine, source};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_design::decision_eval::MeanSource;
use antecedent_estimate::joint_panel::PanelResponse;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

impl JointPanelResult {
    pub(super) fn coordinates(&self) -> Result<Vec<ScientificQuantity>, JointPanelError> {
        let request = self.request();
        let variable = self
            .program
            .source_data()
            .schema()
            .get(request.columns.outcome)
            .map_err(JointPanelError::compile)?;
        let source = self.program.state.source_identity.to_hex();
        let mut coordinates = Vec::new();
        for world in &request.worlds {
            let context = contracts::context_identity(&world.program).to_hex();
            for period in 0..world.horizon() {
                coordinates.push(ScientificQuantity {
                    variable_id: format!("panel:{source}:{}", request.columns.outcome.raw()),
                    variable_name: variable.name.to_string(),
                    role: QuantityRole::Outcome,
                    units: variable.unit.as_deref().unwrap_or("unspecified").into(),
                    population_id: format!("panel-groups:{source}"),
                    regime_id: format!(
                        "{}:{}:{context}",
                        request.target.identity(),
                        world.world.raw()
                    ),
                    horizon: if contracts::target_worlds(&request.target)?
                        .iter()
                        .find(|target| target.id() == world.world)
                        .is_some_and(|target| target.time().is_static())
                    {
                        0
                    } else {
                        u32::try_from(period + 1).map_err(JointPanelError::compile)?
                    },
                    functional_id: "mean".into(),
                    conditioning: vec![],
                    transform_id: "identity".into(),
                });
            }
        }
        Ok(coordinates)
    }
    fn flatten(&self, rows: &[PanelResponse]) -> Result<Vec<f64>, JointPanelError> {
        if rows.len() != self.request().worlds.len() {
            return Err(JointPanelError::compile("response world axis differs"));
        }
        let mut values = Vec::new();
        for (row, world) in rows.iter().zip(&self.request().worlds) {
            if row.program != world.program.id
                || row.means.len() != world.horizon()
                || row.means.iter().any(|value| !value.is_finite())
            {
                return Err(JointPanelError::compile(
                    "response program, horizon or finite-value binding differs",
                ));
            }
            values.extend_from_slice(&row.means);
        }
        Ok(values)
    }
    /// Point mean port with exact world, horizon, source and unit bindings.
    /// This carries no sampling law, calibration or causal license.
    ///
    /// # Errors
    /// Misaligned or nonfinite response axes are refused.
    pub fn numerical_mean_source(&self) -> Result<MeanSource, JointPanelError> {
        Ok(MeanSource {
            coordinates: self.coordinates()?,
            means: self.flatten(&self.point)?,
            // The retained point is the likelihood fit, including for Bayesian requests;
            // posterior draws are exposed separately through the law ports.
            estimate: antecedent_design::decision_eval::MeanEstimate::Frequentist,
            provider_id: format!(
                "antecedent.compiled_joint_panel.{}",
                if self.identification().is_identified_claim() {
                    "identified_worlds_model_conditional_inference"
                } else {
                    "model_conditional"
                }
            ),
            snapshot_id: self.dependency.to_hex(),
            causal_contract_id: self.request().target.identity().to_hex(),
            rng_id: format!("causal_rng:{}", self.seed),
        })
    }
    /// Aligned law of the fitted mean functional; never an outcome-predictive law.
    /// Frequentist realizations retain sampling meaning; Bayesian realizations retain
    /// conditional posterior meaning. Failed slots and unconverged chains refuse a law.
    ///
    /// # Errors
    /// Missing draws, failed slots, invalid diagnostics or malformed coordinates refuse export.
    pub fn numerical_law_source(&self) -> Result<DistributionArtifact, JointPanelError> {
        let semantic = self.law_semantic()?;
        let cells = self
            .draws
            .len()
            .checked_mul(
                self.request().worlds.iter().map(super::JointPanelWorld::horizon).sum::<usize>(),
            )
            .ok_or_else(|| JointPanelError::compile("law projection size overflow"))?;
        let bytes = u64::try_from(
            cells
                .checked_mul(8)
                .ok_or_else(|| JointPanelError::compile("law projection size overflow"))?,
        )
        .map_err(JointPanelError::compile)?;
        source::check_memory(&self.context, bytes)?;
        let _lease = self
            .context
            .acquire_admission(0, bytes)
            .map_err(|error| engine::PanelError::Resource(error.to_string()))?;
        let coordinates = self.coordinates()?;
        let metadata = self.law_metadata(&coordinates, semantic)?;
        let mut draws = Vec::new();
        for row in &self.draws {
            let Some(row) = row else {
                return Err(JointPanelError::Unsupported("failed draw slot".into()));
            };
            draws.extend(self.flatten(row)?);
        }
        DistributionArtifact::new(metadata, draws).map_err(JointPanelError::compile)
    }
    fn law_semantic(&self) -> Result<DistributionMeaningWire, JointPanelError> {
        if self.draws.is_empty() || self.draws.iter().any(Option::is_none) {
            return Err(JointPanelError::Unsupported(
                "incomplete or absent uncertainty slots cannot be converted to a complete risk law"
                    .into(),
            ));
        }
        let bayes =
            matches!(self.request().options.inference, engine::PanelInference::Bayesian { .. });
        let converged = match &self.fit {
            JointPanelFit::Invariant(fit) => {
                if bayes {
                    fit.diagnostics.posterior_converged == Some(true)
                } else {
                    fit.diagnostics.optimizer_converged && fit.diagnostics.curvature_available
                }
            }
            JointPanelFit::Varying(fit) => !bayes || fit.posterior_converged == Some(true),
        };
        if !converged {
            return Err(JointPanelError::Unsupported(
                "fit diagnostics do not authorize a complete uncertainty law".into(),
            ));
        }
        Ok(if bayes {
            DistributionMeaningWire::CausalFunctionalPosterior
        } else {
            DistributionMeaningWire::EstimatorSampling
        })
    }
    fn law_metadata(
        &self,
        coordinates: &[ScientificQuantity],
        semantic: DistributionMeaningWire,
    ) -> Result<DistributionMetadata, JointPanelError> {
        use antecedent_io::distribution_artifact::{
            MAX_DISTRIBUTION_COORDINATES, MAX_DISTRIBUTION_DRAWS,
        };
        if self.draws.len() > MAX_DISTRIBUTION_DRAWS
            || coordinates.len() > MAX_DISTRIBUTION_COORDINATES
        {
            return Err(JointPanelError::compile(
                "projected law exceeds portable draw/coordinate bounds",
            ));
        }
        let identity = DistributionIdentity::new(
            semantic,
            coordinates,
            DrawAlignment::Joint,
            DistributionProvenance {
                source_id: self.program.state.source_identity.to_hex(),
                provider_id: format!(
                    "antecedent.compiled_joint_panel.{}",
                    if self.identification().is_identified_claim() {
                        "identified_worlds_model_conditional_inference"
                    } else {
                        "model_conditional"
                    }
                ),
                rng_id: format!("causal_rng:{}", self.seed),
                snapshot_id: self.dependency.to_hex(),
                causal_contract_id: self.request().target.identity().to_hex(),
            },
        )
        .map_err(JointPanelError::compile)?;
        Ok(DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [self.draws.len(), coordinates.len()],
            weights: None,
            supported: Some(vec![true; coordinates.len()]),
            calibration: DistributionCalibration::Unmeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        })
    }
    /// Compile finite program search with controls derived from actual retained schedules.
    /// The single-outcome search consumes each program's terminal mean. Utility changes
    /// reuse this result and its fit; no new mechanism is silently fitted.
    ///
    /// # Errors
    /// Missing causal design, incomplete control axes, heterogeneous horizons or invalid search contracts.
    pub(super) fn compile_numerical_search(
        &self,
        search_request: &super::search::JointPanelSearchRequest,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_design::program_search::PreparedFiniteProgramSearch, JointPanelError>
    {
        use antecedent_design::program_search::{
            PreparedFiniteProgramSearch, ProgramCandidate, ProgramSearchScope,
        };
        let functional = &search_request.functional;
        let contract = search_request.contract.clone();
        let domain = search_request.domain.clone();
        let utility_units = search_request.utility_units.clone();
        let max_candidates = search_request.max_candidates;
        let request = self.request();
        let h = request.worlds[0].horizon();
        let expected = (0..h)
            .flat_map(|offset| {
                request
                    .columns
                    .channels
                    .iter()
                    .copied()
                    .map(move |channel| (channel, i32::try_from(offset)))
            })
            .map(|(channel, offset)| {
                offset.map(|offset| (channel, offset)).map_err(JointPanelError::compile)
            })
            .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
        let supplied = domain
            .controls
            .iter()
            .map(|control| (control.variable, control.offset))
            .collect::<std::collections::BTreeSet<_>>();
        if domain.controls.len() != expected.len()
            || supplied != expected
            || request.worlds.iter().any(|w| w.horizon() != h)
        {
            return Err(JointPanelError::compile(
                "search must bind every actual channel at every explicit schedule offset and one common horizon",
            ));
        }
        let coordinates = self.projection(functional)?.0;
        let mut candidates = Vec::new();
        for (index, world) in request.worlds.iter().enumerate() {
            let terminal = coordinates[index].clone();
            let controls = domain
                .controls
                .iter()
                .map(|control| {
                    let id = control.variable.raw().to_string();
                    world
                        .program
                        .channels
                        .iter()
                        .find(|channel| channel.id == id)
                        .and_then(|channel| {
                            usize::try_from(control.offset)
                                .ok()
                                .and_then(|offset| channel.values.get(offset))
                        })
                        .copied()
                        .ok_or_else(|| {
                            JointPanelError::compile("compiled program control is missing")
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            candidates.push(ProgramCandidate {
                id: world.program.id.clone(),
                controls,
                inputs: vec![terminal],
            });
        }
        let first = &coordinates[0];
        PreparedFiniteProgramSearch::compile(
            contract,
            domain,
            candidates,
            ProgramSearchScope {
                population: first.population_id.clone(),
                horizon: first.horizon,
                utility_units,
                max_candidates,
            },
            ctx,
        )
        .map_err(JointPanelError::compile)
    }
}

/// Prespecified response quantity, shared unchanged across all programs and fit draws.
#[derive(Clone, Debug, PartialEq)]
pub enum JointPanelResponseFunctional {
    /// Mean outcome in the final program period.
    TerminalMean,
    /// Arithmetic mean over the inclusive zero-based period window.
    WindowAverage {
        /// First included period.
        from: usize,
        /// Last included period.
        until: usize,
    },
    /// Later-window mean minus an earlier-window mean; each world retains its own full horizon.
    WindowChange {
        /// Inclusive zero-based earlier window.
        before: (usize, usize),
        /// Inclusive zero-based later window, disjoint and after the earlier window.
        after: (usize, usize),
    },
    /// Explicit finite weighted sum over every program period.
    WeightedSum {
        /// One coefficient per period; zero is explicit.
        coefficients: Vec<f64>,
    },
}
impl JointPanelResult {
    pub(super) fn projection(
        &self,
        functional: &JointPanelResponseFunctional,
    ) -> Result<(Vec<ScientificQuantity>, Vec<Vec<f64>>), JointPanelError> {
        let coordinates = self.coordinates()?;
        let mut projected = Vec::new();
        let mut weights = Vec::new();
        let mut start = 0;
        for world in &self.request().worlds {
            let h = world.horizon();
            let values = match functional {
                JointPanelResponseFunctional::TerminalMean => {
                    let mut w = vec![0.0; h];
                    w[h - 1] = 1.0;
                    w
                }
                JointPanelResponseFunctional::WindowAverage { from, until } => {
                    if from > until || *until >= h {
                        return Err(JointPanelError::compile(
                            "response window is outside the actual program",
                        ));
                    }
                    let mut w = vec![0.0; h];
                    for value in &mut w[*from..=*until] {
                        *value = 1.0 / (until - from + 1) as f64;
                    }
                    w
                }
                JointPanelResponseFunctional::WindowChange { before, after } => {
                    if before.0 > before.1
                        || before.1 >= after.0
                        || after.0 > after.1
                        || after.1 >= h
                    {
                        return Err(JointPanelError::compile(
                            "change windows must be ordered, disjoint and inside every actual program",
                        ));
                    }
                    let mut weights = vec![0.0; h];
                    for value in &mut weights[before.0..=before.1] {
                        *value = -1.0 / (before.1 - before.0 + 1) as f64;
                    }
                    for value in &mut weights[after.0..=after.1] {
                        *value = 1.0 / (after.1 - after.0 + 1) as f64;
                    }
                    weights
                }
                JointPanelResponseFunctional::WeightedSum { coefficients } => {
                    if coefficients.len() != h || coefficients.iter().any(|v| !v.is_finite()) {
                        return Err(JointPanelError::compile(
                            "response weights must be finite and cover every actual program period",
                        ));
                    }
                    coefficients.clone()
                }
            };
            let mut writer =
                antecedent_core::CanonicalWriter::new("antecedent.panel.response_functional.v1");
            writer.bytes(self.request().target.identity().as_bytes()).word(values.len() as u64);
            for value in &values {
                writer.word(value.to_bits());
            }
            let mut quantity = coordinates[start + h - 1].clone();
            if quantity.horizon != 0 {
                quantity.horizon = u32::try_from(
                    values.iter().rposition(|value| *value != 0.0).map_or(h, |index| index + 1),
                )
                .map_err(JointPanelError::compile)?;
            }
            quantity.functional_id = format!(
                "weighted_mean:{}",
                antecedent_core::SemanticDigest::from_bytes(writer.finish())
            );
            projected.push(quantity);
            weights.push(values);
            start += h;
        }
        Ok((projected, weights))
    }
    fn project_rows(
        &self,
        rows: &[PanelResponse],
        weights: &[Vec<f64>],
    ) -> Result<Vec<f64>, JointPanelError> {
        self.flatten(rows)?;
        engine::project_panel_responses(rows, weights).map_err(JointPanelError::from)
    }
    /// One exact prespecified mean functional per world; no uncertainty or causal license is added.
    ///
    /// # Errors
    /// Invalid windows/weights or misaligned/nonfinite rows refuse projection.
    pub fn numerical_project_mean_source(
        &self,
        functional: &JointPanelResponseFunctional,
    ) -> Result<MeanSource, JointPanelError> {
        let (coordinates, weights) = self.projection(functional)?;
        let mut source = self.numerical_mean_source()?;
        source.coordinates = coordinates;
        source.means = self.project_rows(&self.point, &weights)?;
        Ok(source)
    }
    /// Apply the same declared response functional to every complete aligned draw row.
    ///
    /// # Errors
    /// Incomplete/convergence-refused laws or invalid projections remain typed refusals.
    pub fn numerical_project_law_source(
        &self,
        functional: &JointPanelResponseFunctional,
    ) -> Result<DistributionArtifact, JointPanelError> {
        let bytes = u64::try_from(
            self.draws
                .len()
                .checked_mul(self.request().worlds.len())
                .and_then(|n| n.checked_mul(8))
                .ok_or_else(|| JointPanelError::compile("projection size overflow"))?,
        )
        .map_err(JointPanelError::compile)?;
        source::check_memory(&self.context, bytes)?;
        let _lease = self
            .context
            .acquire_admission(0, bytes)
            .map_err(|error| engine::PanelError::Resource(error.to_string()))?;
        let semantic = self.law_semantic()?;
        let (coordinates, weights) = self.projection(functional)?;
        let metadata = self.law_metadata(&coordinates, semantic)?;
        let mut values = Vec::new();
        for row in self.draws.iter().flatten() {
            values.extend(self.project_rows(row, &weights)?);
        }
        DistributionArtifact::new(metadata, values).map_err(JointPanelError::compile)
    }
}

/// Required scientific permission, shared with the downstream decision owner.
pub use antecedent_design::scientific_source::ScientificConsumptionScope as JointPanelScientificScope;

impl JointPanelResult {
    fn scientific_source(
        &self,
        payload: antecedent_design::scientific_source::ScientificSourcePayload,
        weights: Option<Vec<Vec<f64>>>,
        terminal_only: bool,
    ) -> Result<antecedent_design::scientific_source::ScientificDecisionSource, JointPanelError>
    {
        if !terminal_only {
            return self.projected_scientific_source(payload, weights);
        }
        let bytes = super::projection::source_payload_quote(&payload)?;
        source::check_memory(&self.context, bytes)?;
        let retention = std::sync::Arc::new(
            self.context
                .acquire_admission(0, bytes)
                .map_err(|error| engine::PanelError::Resource(error.to_string()))?,
        );
        antecedent_design::scientific_source::ScientificDecisionSource::new(
            self.request().target.clone(),
            self.identification().clone(),
            self.dependency_identity(),
            payload,
        )
        .map(|source| source.retain_admission(retention))
        .map_err(JointPanelError::compile)
    }
    /// Point means retaining the exact compiler proof and scientific standing.
    ///
    /// # Errors
    /// Invalid response bindings or unbound compiler products refuse export.
    pub fn mean_source(
        &self,
    ) -> Result<antecedent_design::scientific_source::ScientificDecisionSource, JointPanelError>
    {
        self.scientific_source(
            antecedent_design::scientific_source::ScientificSourcePayload::Means(
                self.numerical_mean_source()?,
            ),
            None,
            self.request().worlds.iter().all(|w| w.horizon() == 1),
        )
    }
    /// Complete aligned fitted-mean law retaining its compiler proof and scientific standing.
    ///
    /// # Errors
    /// Failed slots, convergence refusal or invalid science remain typed errors.
    pub fn law_source(
        &self,
    ) -> Result<antecedent_design::scientific_source::ScientificDecisionSource, JointPanelError>
    {
        self.scientific_source(
            antecedent_design::scientific_source::ScientificSourcePayload::Law(Box::new(
                self.numerical_law_source()?,
            )),
            None,
            self.request().worlds.iter().all(|w| w.horizon() == 1),
        )
    }
    /// One prespecified response mean per world, inseparable from its checked scientific scope.
    ///
    /// # Errors
    /// Invalid projection or changed scientific binding refuses export.
    pub fn project_mean_source(
        &self,
        functional: &JointPanelResponseFunctional,
    ) -> Result<antecedent_design::scientific_source::ScientificDecisionSource, JointPanelError>
    {
        self.scientific_source(
            antecedent_design::scientific_source::ScientificSourcePayload::Means(
                self.numerical_project_mean_source(functional)?,
            ),
            Some(self.projection(functional)?.1),
            matches!(functional, JointPanelResponseFunctional::TerminalMean),
        )
    }
    /// One prespecified response functional per complete draw, with checked scientific scope.
    ///
    /// # Errors
    /// Invalid projection, failed slots, convergence or scientific scope refuses export.
    pub fn project_law_source(
        &self,
        functional: &JointPanelResponseFunctional,
    ) -> Result<antecedent_design::scientific_source::ScientificDecisionSource, JointPanelError>
    {
        self.scientific_source(
            antecedent_design::scientific_source::ScientificSourcePayload::Law(Box::new(
                self.numerical_project_law_source(functional)?,
            )),
            Some(self.projection(functional)?.1),
            matches!(functional, JointPanelResponseFunctional::TerminalMean),
        )
    }
}
