//! Bounded-tile streamed reductions of a fitted panel law.
//!
//! The retained result keeps every draw by every world by every period, and that explicit
//! full output stays available. A caller that needs one prespecified functional per world and
//! draw does not need those rows: this route evaluates the same shared compiled plan over the
//! same aligned draws one tile at a time and keeps only the reduced values. Draw order, failed
//! slots, the joint draw alignment across worlds and every reduced bit match the full-output
//! route followed by the same projection.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::ports::functional_weights;
use super::{
    JointPanelError, JointPanelFit, JointPanelResponseFunctional, JointPanelResult,
    JointPanelState, JointPanelWorld, engine, source,
};
use crate::PreparedStudy;
use antecedent_core::{ExecutionContext, ScientificTarget, SemanticDigest};

/// Default number of draws evaluated per tile when the caller names none.
pub const DEFAULT_TILE_DRAWS: usize = 256;

/// One prespecified functional per world for the point fit and every aligned draw slot.
#[derive(Debug)]
pub struct JointPanelProjectedLaw {
    projected: engine::PanelProjectedLaw,
    weights: Vec<Vec<f64>>,
    fit: JointPanelFit,
    dependency: SemanticDigest,
}
impl JointPanelProjectedLaw {
    /// Projected point value for each world, in request order.
    #[must_use]
    pub fn point(&self) -> &[f64] {
        self.projected.point()
    }
    /// Projected values per original draw slot; a failed slot stays `None` in place.
    #[must_use]
    pub fn draws(&self) -> &[Option<Vec<f64>>] {
        self.projected.draws()
    }
    /// Tiles used and the most draws ever held live at once.
    #[must_use]
    pub fn receipt(&self) -> &engine::PanelStreamReceipt {
        self.projected.receipt()
    }
    /// The exact period weights applied to each world.
    #[must_use]
    pub fn weights(&self) -> &[Vec<f64>] {
        &self.weights
    }
    /// The fitted mechanism these values were evaluated from.
    #[must_use]
    pub fn fit(&self) -> &JointPanelFit {
        &self.fit
    }
    /// Fit and execution dependency of the law, as for a full result.
    #[must_use]
    pub fn dependency_identity(&self) -> SemanticDigest {
        self.dependency
    }
    /// Draw-major, world-minor values of a complete law, the layout of a projected
    /// distribution artifact.
    ///
    /// # Errors
    /// Any failed slot: a partial law is not a complete one.
    pub fn complete_values(&self) -> Result<Vec<f64>, JointPanelError> {
        let mut values = Vec::new();
        for row in self.draws() {
            let Some(row) = row else {
                return Err(JointPanelError::Unsupported("failed draw slot".into()));
            };
            values.extend_from_slice(row);
        }
        Ok(values)
    }
}

fn worker_error(
    error: antecedent_core::execution::MapExecutionError<JointPanelError>,
) -> JointPanelError {
    match error {
        antecedent_core::execution::MapExecutionError::Work(error) => error,
        antecedent_core::execution::MapExecutionError::Admission(
            antecedent_core::execution::AdmissionError::Cancelled,
        ) => engine::PanelError::Cancelled.into(),
        antecedent_core::execution::MapExecutionError::Admission(error) => {
            engine::PanelError::Resource(error.to_string()).into()
        }
    }
}

fn project_fit(
    prepared: &PreparedStudy<JointPanelState>,
    fit: JointPanelFit,
    dependency: SemanticDigest,
    functional: &JointPanelResponseFunctional,
    tile_draws: usize,
    ctx: &ExecutionContext,
) -> Result<JointPanelProjectedLaw, JointPanelError> {
    source::check_cancelled(ctx)?;
    prepared
        .state
        .identification
        .validate_target(&prepared.state.request.target)
        .map_err(JointPanelError::compile)?;
    prepared.state.source_retention.observe_current();
    if let Some(lease) = &prepared.state.program_retention {
        lease.observe_current();
    }
    let worlds = &prepared.state.request.worlds;
    let programs = worlds.iter().map(|world| world.program.clone()).collect::<Vec<_>>();
    let weights = worlds
        .iter()
        .map(|world| functional_weights(world.horizon(), functional))
        .collect::<Result<Vec<_>, _>>()?;
    let projected = match &fit {
        JointPanelFit::Invariant(fit) => {
            let _ = fit.retained_bytes();
            engine::project_panel_law(
                &prepared.state.data,
                &fit.point,
                &fit.draws,
                &programs,
                &weights,
                tile_draws,
                ctx,
            )?
        }
        JointPanelFit::Varying(fit) => {
            let _ = fit.retained_bytes();
            engine::project_temporal_law(
                &prepared.state.data,
                &fit.point,
                &fit.draws,
                &programs,
                &weights,
                tile_draws,
                ctx,
            )?
        }
    };
    Ok(JointPanelProjectedLaw { projected, weights, fit, dependency })
}

impl PreparedStudy<JointPanelState> {
    /// Fit (or reuse the cached fit) and reduce the law without retaining full responses.
    ///
    /// The fit, its dependency key and the cache are exactly those of [`Self::estimate`]; only
    /// the response evaluation differs. Peak response storage is one tile of `tile_draws`
    /// draws, and the retained output is one value per world and draw slot.
    ///
    /// # Errors
    /// Numerical, dependency, cancellation and resource errors remain typed; a zero tile or
    /// a functional outside any world's horizon is refused.
    pub fn estimate_projected(
        &self,
        functional: &JointPanelResponseFunctional,
        tile_draws: usize,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelProjectedLaw, JointPanelError> {
        ctx.try_map_indexed(1, |_, worker| {
            source::check_cancelled(worker)?;
            self.state
                .identification
                .validate_target(&self.state.request.target)
                .map_err(JointPanelError::compile)?;
            let dependency = source::fit_dependency(&self.state, worker)?;
            let fit = self.fit(dependency, worker)?;
            project_fit(self, fit, dependency, functional, tile_draws, worker)
        })
        .map_err(worker_error)?
        .pop()
        .ok_or_else(|| JointPanelError::compile("admitted projected panel worker returned nothing"))
    }
}

impl JointPanelResult {
    /// Reduce this result's own frozen law again, tile by tile, without a second full copy.
    ///
    /// # Errors
    /// An invalid functional or tile, cancellation or resource refusal.
    pub fn project_streamed(
        &self,
        functional: &JointPanelResponseFunctional,
        tile_draws: usize,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelProjectedLaw, JointPanelError> {
        ctx.try_map_indexed(1, |_, worker| {
            project_fit(
                &self.program,
                self.fit.clone(),
                self.dependency,
                functional,
                tile_draws,
                worker,
            )
        })
        .map_err(worker_error)?
        .pop()
        .ok_or_else(|| JointPanelError::compile("admitted projected panel worker returned nothing"))
    }

    /// Evaluate other worlds on this frozen fit, reduced tile by tile.
    ///
    /// The streamed counterpart of [`Self::retarget_frozen`]: the fit and its dependencies stay
    /// frozen and the caller's context controls only resources and cancellation, but the
    /// worlds-by-draws-by-periods rows are never retained.
    ///
    /// # Errors
    /// Changed scientific/program bindings, an invalid functional or tile, cancellation or
    /// resource failures.
    pub fn project_frozen_streamed(
        &self,
        worlds: Vec<JointPanelWorld>,
        target: ScientificTarget,
        functional: &JointPanelResponseFunctional,
        tile_draws: usize,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelProjectedLaw, JointPanelError> {
        let prepared = self.program.retarget(worlds, target, ctx)?;
        ctx.try_map_indexed(1, |_, worker| {
            project_fit(
                &prepared,
                self.fit.clone(),
                self.dependency,
                functional,
                tile_draws,
                worker,
            )
        })
        .map_err(worker_error)?
        .pop()
        .ok_or_else(|| JointPanelError::compile("admitted projected panel worker returned nothing"))
    }
}
