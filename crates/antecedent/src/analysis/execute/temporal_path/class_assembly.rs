//! Assembly of class-aware temporal response envelopes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalResponse, Diagnostic, EffectEstimate, EstimatorId, ExecutionContext,
    IdentificationEnvelope, IdentificationResult, IdentificationStatus, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, Instant, PhysicalExecutionPlan,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue,
    Strategy, StudyResult, VariableId, bare_class_atom, class_completion_key, elapsed_ns,
    shared_temporal_class_uncertainty, temporal_class_complete_values,
    temporal_class_response_masses,
};

/// Envelope assembly of a class-aware temporal response, shared by the series and
/// panel owners.
///
/// One completion at one horizon is an *atom*: the owner fits it (a series surface, or
/// the panel's equal-weight average of unit surfaces) and hands the surface here. The
/// assembly keeps the pointwise minimum and maximum over the completions of each
/// (cell, horizon), every atom's structural record and support, and publishes the
/// identified set as a `PartiallyIdentified` envelope over the full surfaces — or, when
/// mass is unidentified, unevaluable or capped, the completions' surfaces as
/// `GraphDependent` atoms. The grid comes from the query, so it is the same whatever
/// the horizon count.
pub(in crate::analysis::execute) struct ClassResponseAssembly {
    /// Cells of one horizon: the dose grid, or one intervention-response cell.
    pub(super) cells_per_horizon: usize,
    pub(super) n_horizons: usize,
    pub(super) lower: Vec<f64>,
    pub(super) upper: Vec<f64>,
    pub(in crate::analysis::execute) structural_atoms: Vec<crate::result::StructuralResponseAtom>,
    pub(super) supports: Vec<antecedent_core::SupportReport>,
    pub(super) horizon_supports: Vec<antecedent_core::SupportReport>,
    pub(super) horizon_fingerprints: Vec<Vec<u64>>,
    /// No horizon capped its completion enumeration.
    pub(in crate::analysis::execute) full_mass_scope: bool,
    pub(in crate::analysis::execute) truncated_atoms: usize,
    pub(super) horizon_values: Vec<Vec<f64>>,
    pub(super) support_start: usize,
}

/// What [`ClassResponseAssembly::finish`] publishes.
pub(in crate::analysis::execute) struct ClassResponseParts {
    pub response: CausalResponse,
    pub response_envelope: antecedent_core::ResponseEnvelope,
    pub identified_mass: f64,
    pub unidentified_mass: f64,
    pub unevaluable_mass: f64,
}

/// The identified-execute tail every class response executor (temporal Pulse/Sustained,
/// Sequence and panel) ends with: the strategy, certificate and extras are the same, only the
/// fitted pieces differ.
pub(in crate::analysis::execute) struct ClassResponseTail<'a> {
    pub physical: &'a PhysicalExecutionPlan,
    pub identification: IdentificationResult,
    pub estimand: IdentifiedEstimand,
    /// Scalar summary of the response, or a NaN effect when the response carries the result.
    pub estimate: EffectEstimate,
    pub identifier_id: IdentifierId,
    pub endpoints: (VariableId, VariableId),
    /// The completion envelope the certificate retains.
    pub envelope: Option<Arc<antecedent_identify::TemporalClassEnvelope>>,
    pub response: CausalResponse,
    pub structural: crate::result::StructuralResponseMixture,
    pub diagnostics: Vec<Diagnostic>,
    pub identify_cached: bool,
    /// The bootstrap replicate count the result records (`None`: the class band is withheld).
    pub bootstrap_replicates: Option<u32>,
    pub started: Instant,
    pub ctx: &'a ExecutionContext,
    /// Refutation reports. Class routes pass an empty list; the graph-posterior
    /// route passes the reports it already computed.
    pub refutations: Vec<antecedent_validate::RefutationReport>,
    /// Overrides the identifier provenance step. `None` keeps the strategy default.
    pub identify_provenance: Option<(Arc<str>, Arc<str>)>,
    /// Overrides the estimator provenance step. `None` keeps the strategy default.
    pub estimate_provenance: Option<(Arc<str>, Arc<str>)>,
}

impl super::super::Study {
    /// Publish a class response through the identified-execute finish.
    pub(in crate::analysis::execute) fn finish_class_response_tail(
        &self,
        tail: ClassResponseTail<'_>,
    ) -> StudyResult {
        let ClassResponseTail {
            physical,
            identification,
            estimand,
            estimate,
            identifier_id,
            endpoints: (treatment, outcome),
            envelope,
            response,
            structural,
            diagnostics,
            identify_cached,
            bootstrap_replicates,
            started,
            ctx,
            refutations,
            identify_provenance,
            estimate_provenance,
        } = tail;
        self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    identifier_id,
                    EstimatorId::temporal_response_for(&self.settings.inference),
                ),
                treatment,
                outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .cancelled(ctx.cancellation.is_cancelled())
            .extras(IdentifiedExecuteExtras {
                certificate: envelope.map(|envelope| crate::Identification::TemporalEnvelope {
                    envelope: Arc::unwrap_or_clone(envelope),
                    strategy: identifier_id,
                    structure_version: self.graph.version(),
                }),
                response: Some(response),
                structural_response: Some(structural),
                diagnostics: Some(diagnostics),
                bootstrap_replicates_requested: Some(bootstrap_replicates),
                identify_provenance,
                estimate_provenance,
                ..Default::default()
            }),
        )
    }
}

impl ClassResponseAssembly {
    pub(in crate::analysis::execute) const fn cells_per_horizon(&self) -> usize {
        self.cells_per_horizon
    }

    /// Open a horizon's completion envelope.
    pub(in crate::analysis::execute) fn begin_horizon(
        &mut self,
        envelope: &IdentificationEnvelope<antecedent_identify::TemporalCompletionGraph>,
    ) {
        self.horizon_fingerprints
            .push(envelope.cases.iter().map(|case| case.graph.fingerprint()).collect());
        self.full_mass_scope &= envelope.truncated_completions == 0;
        self.truncated_atoms += envelope.truncated_completions;
        self.horizon_values = Vec::new();
        self.support_start = self.supports.len();
    }

    /// Record a completion the owner did not evaluate at this horizon.
    pub(in crate::analysis::execute) fn push_unevaluated(
        &mut self,
        horizon_index: usize,
        case_index: usize,
        weight: f64,
        status: IdentificationStatus,
    ) {
        self.structural_atoms.push(bare_class_atom(
            class_completion_key(horizon_index, case_index),
            weight,
            status,
        ));
    }

    /// Record one fitted completion surface at this horizon; returns its atom index.
    ///
    /// # Errors
    ///
    /// An atom that is not a point surface, or whose cell count is not the horizon's.
    pub(in crate::analysis::execute) fn push_atom(
        &mut self,
        horizon_index: usize,
        case_index: usize,
        weight: f64,
        status: IdentificationStatus,
        horizon: u32,
        mut response: CausalResponse,
    ) -> Result<usize, CausalError> {
        if let ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) =
            response.estimate
        {
            response.estimate = ResponseIdentification::PointIdentified(ResponseValue::Surface {
                grid: Arc::from([f64::from(horizon)]),
                dimension: 1,
                mean: Arc::from([value]),
            });
        }
        let values = match &response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
                mean.to_vec()
            }
            _ => {
                return Err(CausalError::Compile {
                    message: "temporal class response atom did not produce a surface".into(),
                });
            }
        };
        if values.len() != self.cells_per_horizon {
            return Err(CausalError::Compile {
                message: "temporal class response atom shape mismatch".into(),
            });
        }
        self.horizon_values.push(values);
        self.structural_atoms.push(crate::result::StructuralResponseAtom {
            value: match &response.estimate {
                ResponseIdentification::PointIdentified(value)
                | ResponseIdentification::PartiallyIdentified(value) => Some(value.clone()),
                _ => None,
            },
            ..bare_class_atom(class_completion_key(horizon_index, case_index), weight, status)
        });
        let index = self.structural_atoms.len() - 1;
        self.supports.push(response.support.clone());
        self.structural_atoms[index].response = Some(response);
        Ok(index)
    }

    /// Close a horizon: mix its atoms' supports and take the pointwise envelope.
    ///
    /// # Errors
    ///
    /// No evaluable atom at this horizon.
    pub(in crate::analysis::execute) fn end_horizon(
        &mut self,
        horizon_index: usize,
        horizon: u32,
    ) -> Result<(), CausalError> {
        if self.horizon_values.is_empty() {
            return Err(CausalError::Compile {
                message: format!(
                    "temporal class response has no evaluable atom at horizon {horizon}"
                ),
            });
        }
        self.horizon_supports.push(super::super::response_path::mix_support_reports(
            &self.supports[self.support_start..].iter().collect::<Vec<_>>(),
        ));
        for cell in 0..self.cells_per_horizon {
            let destination = cell * self.n_horizons + horizon_index;
            self.lower[destination] =
                self.horizon_values.iter().map(|values| values[cell]).fold(f64::INFINITY, f64::min);
            self.upper[destination] = self
                .horizon_values
                .iter()
                .map(|values| values[cell])
                .fold(f64::NEG_INFINITY, f64::max);
        }
        Ok(())
    }

    /// Assemble the published response over every horizon's atoms.
    ///
    /// # Errors
    ///
    /// An unreadable dose grid.
    pub(in crate::analysis::execute) fn finish(
        &self,
        query: &ResponseQuery,
        temporal: &antecedent_core::TemporalResponseSpec,
        assumptions: antecedent_core::AssumptionSet,
        provenance_id: &'static str,
    ) -> Result<ClassResponseParts, CausalError> {
        let (grid, dimension) = match &query.functional {
            ResponseFunctional::MeanCurve { treatment, .. } => {
                let doses = treatment.grid.values()?;
                let grid: Vec<f64> = doses
                    .iter()
                    .flat_map(|dose| {
                        temporal.horizons.iter().flat_map(|horizon| [*dose, f64::from(*horizon)])
                    })
                    .collect();
                (grid, 2)
            }
            ResponseFunctional::InterventionResponse { .. } => {
                (temporal.horizons.iter().map(|horizon| f64::from(*horizon)).collect(), 1)
            }
            _ => {
                return Err(CausalError::Unsupported {
                    message: "class-aware temporal response supports curves and intervention responses",
                });
            }
        };
        let response_envelope = antecedent_core::ResponseEnvelope {
            grid: Arc::from(grid),
            dimension,
            lower: Arc::from(self.lower.clone()),
            upper: Arc::from(self.upper.clone()),
        };
        let graph_values = temporal_class_complete_values(
            &self.structural_atoms,
            &self.horizon_fingerprints,
            &response_envelope,
            self.n_horizons,
        );
        let (identified_mass, unidentified_mass, unevaluable_mass) =
            temporal_class_response_masses(&self.structural_atoms);
        let incomplete = unidentified_mass > 0.0 || unevaluable_mass > 0.0 || !self.full_mass_scope;
        let mut support = super::super::response_path::mix_support_reports(
            &self.supports.iter().collect::<Vec<_>>(),
        );
        support.point_status = (0..self.cells_per_horizon)
            .flat_map(|cell| (0..self.n_horizons).map(move |horizon| (cell, horizon)))
            .map(|(cell, horizon)| {
                self.horizon_supports[horizon].point_status.as_ref()?.get(cell).copied()
            })
            .collect::<Option<Vec<_>>>()
            .map(Arc::from);
        let status = if incomplete {
            IdentificationStatus::GraphDependent
        } else {
            IdentificationStatus::PartiallyIdentified
        };
        let response = CausalResponse {
            estimand: query.functional.clone(),
            identification_status: status,
            estimate: if incomplete {
                ResponseIdentification::GraphDependent(graph_values)
            } else {
                ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(
                    response_envelope.clone(),
                ))
            },
            uncertainty: if incomplete {
                ResponseUncertainty::None
            } else {
                shared_temporal_class_uncertainty(
                    &self.structural_atoms,
                    self.n_horizons,
                    self.cells_per_horizon,
                )
                .unwrap_or(ResponseUncertainty::None)
            },
            support,
            assumptions,
            provenance_id: Arc::from(provenance_id),
            horizon_identification: Some(
                temporal
                    .horizons
                    .iter()
                    .map(|&horizon| antecedent_core::HorizonIdentification {
                        horizon,
                        status,
                        method: Arc::from("temporal_class.completion_envelope"),
                        adjustment: Arc::from([]),
                    })
                    .collect::<Vec<_>>()
                    .into(),
            ),
            interaction_structurally_zero: false,
        };
        Ok(ClassResponseParts {
            response,
            response_envelope,
            identified_mass,
            unidentified_mass,
            unevaluable_mass,
        })
    }
}
