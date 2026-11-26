//! Validation checks of a temporal structure search, through the compiler.
//!
//! A check runs on a study built with a temporal [`DiscoverySpec`]: the study is compiled, the
//! check is lowered as its own program composed over the search's lowered program
//! ([`DiscoveryValidation::lowered`] names the `DiscoveryValidation` route and its identity binds
//! the search program and the check), and the check runs the search's own engine, configured
//! from the study's own parameters, on the study's own data:
//!
//! * [`DiscoveryCheck::BlockBootstrap`], [`DiscoveryCheck::FalsePositive`],
//!   [`DiscoveryCheck::Alpha`], [`DiscoveryCheck::LagWindow`], [`DiscoveryCheck::CiTest`] and
//!   [`DiscoveryCheck::SyntheticNull`] check a PCMCI search over one series;
//! * [`DiscoveryCheck::Orientation`] checks a PCMCI+ search over one series;
//! * [`DiscoveryCheck::EnvironmentHoldout`] checks a J-PCMCI+ search over environments;
//! * [`DiscoveryCheck::RegimeStability`] checks a regime-PCMCI search over its declared regime
//!   labels, held fixed (no alternating relabelling), on one series.
//!
//! A check of another search, or on another modality, is refused before anything runs. Every
//! report is a diagnostic: the frequencies, rates and agreements carry no nominal level beyond
//! what each report states, and their calibration is not measured.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CanonicalWriter, CausalQuery, DiscoveryMethod, ExecutionContext, VariableId,
};
use antecedent_data::{EnvHoldoutSplit, TimeSeriesData};
use antecedent_discovery::DiscoveryWorkspace;
use antecedent_validate::{
    AlphaThresholdSensitivity, BlockBootstrapStability, CiTestSensitivity, EnvironmentHoldout,
    FalsePositiveCheck, LagWindowSensitivity, OrientationStability, RegimeStability,
    SyntheticNullCalibration,
};
/// The reports a check returns, and the surrogate transform of the false-positive check.
pub use antecedent_validate::{
    DiscoveryStabilityReport, EnvironmentHoldoutReport, FalsePositiveCheckReport,
    NullCalibrationReport, NullTransform, OrientationStabilityReport, RegimeStabilityReport,
};

use crate::analysis::Study;
use crate::analysis::program_handle::{CompiledProgramHandle, gate_program};
use crate::analysis::structure_route::{DiscoverData, DiscoverySpec};
use crate::discovery::DiscoverParams;
use crate::discovery::engines::{jpcmci_plus_for, pcmci_for, pcmci_plus_for, rpcmci_for};
use crate::error::CausalError;
use crate::planner::lowered::LoweredProgram;

/// Stage a cancelled check reports.
const VALIDATION_STAGE: &str = "discover.validation";

/// One validation check of the study's temporal structure search.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DiscoveryCheck {
    /// PCMCI link frequencies over fixed-length moving-block resamples of the series.
    BlockBootstrap {
        /// Resamples drawn.
        replicates: u32,
        /// Block length, in rows.
        block_size: usize,
    },
    /// PCMCI on surrogate series whose dependence the transform destroys: the empirical
    /// false-positive rate of the search.
    FalsePositive {
        /// Column permutation or phase randomization.
        transform: NullTransform,
        /// Surrogates drawn.
        replicates: u32,
    },
    /// PCMCI link selection over a grid of significance levels.
    Alpha {
        /// The levels, in the order reported.
        alphas: Arc<[f64]>,
    },
    /// PCMCI link selection over a grid of maximum lags.
    LagWindow {
        /// The maximum lags, in the order reported.
        max_lags: Arc<[u32]>,
    },
    /// PCMCI link selection over a grid of conditional-independence tests.
    CiTest {
        /// Registered test names, in the order reported.
        ci_names: Arc<[Arc<str>]>,
    },
    /// PCMCI+ orientation frequencies over moving-block resamples.
    Orientation {
        /// Resamples drawn.
        replicates: u32,
        /// Block length, in rows.
        block_size: usize,
    },
    /// The search's false-positive rate on simulated independent Gaussian series, judged
    /// against the study's significance level.
    SyntheticNull {
        /// Simulated series.
        n_sim: u32,
        /// Rows of each simulated series.
        n_obs: usize,
        /// Variables of each simulated series.
        n_vars: usize,
    },
    /// J-PCMCI+ on the first `n_discovery` environments against the rest.
    EnvironmentHoldout {
        /// Environments the discovery run reads; the others are held out.
        n_discovery: usize,
    },
    /// Per-regime PCMCI+ link frequencies over moving-block resamples, the study's regime
    /// labels held fixed.
    RegimeStability {
        /// Resamples drawn.
        replicates: u32,
        /// Block length, in rows.
        block_size: usize,
    },
}

impl DiscoveryCheck {
    /// Stable id, written into the lowered program's plan id.
    #[must_use]
    pub const fn id(&self) -> &'static str {
        match self {
            Self::BlockBootstrap { .. } => "block_bootstrap",
            Self::FalsePositive { .. } => "false_positive",
            Self::Alpha { .. } => "alpha",
            Self::LagWindow { .. } => "lag_window",
            Self::CiTest { .. } => "ci_test",
            Self::Orientation { .. } => "orientation",
            Self::SyntheticNull { .. } => "synthetic_null",
            Self::EnvironmentHoldout { .. } => "environment_holdout",
            Self::RegimeStability { .. } => "regime_stability",
        }
    }

    /// Canonical encoding of every option, so two checks that differ in any option are
    /// different programs.
    pub(crate) fn write(&self, writer: &mut CanonicalWriter) {
        writer.text(self.id());
        match self {
            Self::BlockBootstrap { replicates, block_size }
            | Self::Orientation { replicates, block_size }
            | Self::RegimeStability { replicates, block_size } => {
                writer.word(u64::from(*replicates)).word(*block_size as u64);
            }
            Self::FalsePositive { transform, replicates } => {
                writer.text(match transform {
                    NullTransform::ColumnPermute => "permute",
                    NullTransform::PhaseRandomize => "phase",
                });
                writer.word(u64::from(*replicates));
            }
            Self::Alpha { alphas } => {
                writer.reals(&alphas[..]);
            }
            Self::LagWindow { max_lags } => {
                writer.u32s(&max_lags[..]);
            }
            Self::CiTest { ci_names } => {
                writer.texts(&ci_names[..]);
            }
            Self::SyntheticNull { n_sim, n_obs, n_vars } => {
                writer.word(u64::from(*n_sim)).word(*n_obs as u64).word(*n_vars as u64);
            }
            Self::EnvironmentHoldout { n_discovery } => {
                writer.word(*n_discovery as u64);
            }
        }
    }

    /// The search this check validates, by its method id.
    const fn search_id(&self) -> &'static str {
        match self {
            Self::BlockBootstrap { .. }
            | Self::FalsePositive { .. }
            | Self::Alpha { .. }
            | Self::LagWindow { .. }
            | Self::CiTest { .. }
            | Self::SyntheticNull { .. } => "pcmci",
            Self::Orientation { .. } => "pcmci_plus",
            Self::EnvironmentHoldout { .. } => "jpcmci_plus",
            Self::RegimeStability { .. } => "rpcmci",
        }
    }

    /// Whether `method` is the search this check validates.
    const fn admits(&self, method: &DiscoveryMethod) -> bool {
        match self {
            Self::BlockBootstrap { .. }
            | Self::FalsePositive { .. }
            | Self::Alpha { .. }
            | Self::LagWindow { .. }
            | Self::CiTest { .. }
            | Self::SyntheticNull { .. } => matches!(method, DiscoveryMethod::Pcmci),
            Self::Orientation { .. } => matches!(method, DiscoveryMethod::PcmciPlus),
            Self::EnvironmentHoldout { .. } => matches!(method, DiscoveryMethod::JpcmciPlus),
            Self::RegimeStability { .. } => matches!(method, DiscoveryMethod::Rpcmci { .. }),
        }
    }
}

/// What a check reported.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum DiscoveryCheckReport {
    /// Link frequencies: block bootstrap and the alpha, lag-window and test grids.
    Stability(DiscoveryStabilityReport),
    /// The surrogate false-positive rate.
    FalsePositive(FalsePositiveCheckReport),
    /// PCMCI+ orientation frequencies.
    Orientation(OrientationStabilityReport),
    /// The synthetic-null false-positive rate.
    NullCalibration(NullCalibrationReport),
    /// Link agreement between discovery and held-out environments.
    EnvironmentHoldout(EnvironmentHoldoutReport),
    /// Per-regime link frequencies.
    RegimeStability(RegimeStabilityReport),
}

/// A validation check executed through its compiled program.
#[derive(Clone, Debug)]
pub struct DiscoveryValidation {
    lowered: Arc<LoweredProgram>,
    check: DiscoveryCheck,
    report: DiscoveryCheckReport,
}

impl DiscoveryValidation {
    /// The lowered program the check executed.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Identity of the lowered program the check executed.
    #[must_use]
    pub fn lowered_identity(&self) -> antecedent_core::SemanticDigest {
        self.lowered.identity()
    }

    /// The check.
    #[must_use]
    pub const fn check(&self) -> &DiscoveryCheck {
        &self.check
    }

    /// What the check reported.
    #[must_use]
    pub const fn report(&self) -> &DiscoveryCheckReport {
        &self.report
    }
}

/// A check on a modality its search does not read.
fn wrong_modality(message: impl std::fmt::Display) -> CausalError {
    crate::compile_reason!(
        "route_not_supported",
        "{}: {message}",
        antecedent_core::reason_details::discovery_validation::WRONG_MODALITY
    )
}

/// A check of a search it does not validate.
fn wrong_search(message: impl std::fmt::Display) -> CausalError {
    crate::compile_reason!(
        "route_not_supported",
        "{}: {message}",
        antecedent_core::reason_details::discovery_validation::WRONG_SEARCH
    )
}

fn one_series<'a>(
    check: &DiscoveryCheck,
    data: &DiscoverData<'a>,
) -> Result<&'a TimeSeriesData, CausalError> {
    match data {
        DiscoverData::Series(series) => Ok(*series),
        DiscoverData::Tabular(_) | DiscoverData::Panel(_) | DiscoverData::MultiEnv(_) => {
            Err(wrong_modality(format!(
                "the {} check runs on one series; build the study with Study::series",
                check.id()
            )))
        }
    }
}

/// Run `check` with the engine the study's search runs, on the study's data.
#[allow(clippy::too_many_lines, reason = "one arm per check, each calling its engine")]
fn run_check(
    check: &DiscoveryCheck,
    spec: &DiscoverySpec,
    params: &DiscoverParams,
    data: &DiscoverData<'_>,
    variables: &[VariableId],
    ctx: &ExecutionContext,
) -> Result<DiscoveryCheckReport, CausalError> {
    let mut workspace = DiscoveryWorkspace::default();
    Ok(match check {
        DiscoveryCheck::BlockBootstrap { replicates, block_size } => {
            let checker = BlockBootstrapStability {
                pcmci: pcmci_for(params),
                replicates: *replicates,
                block_size: *block_size,
            };
            DiscoveryCheckReport::Stability(checker.run(
                one_series(check, data)?,
                variables,
                &mut workspace,
                ctx,
            )?)
        }
        DiscoveryCheck::FalsePositive { transform, replicates } => {
            let checker = FalsePositiveCheck::new(pcmci_for(params), *transform, *replicates);
            DiscoveryCheckReport::FalsePositive(checker.run(
                one_series(check, data)?,
                variables,
                &mut workspace,
                ctx,
            )?)
        }
        DiscoveryCheck::Alpha { alphas } => {
            let checker = AlphaThresholdSensitivity::new(pcmci_for(params), Arc::clone(alphas));
            DiscoveryCheckReport::Stability(checker.run(
                one_series(check, data)?,
                variables,
                &mut workspace,
                ctx,
            )?)
        }
        DiscoveryCheck::LagWindow { max_lags } => {
            let checker = LagWindowSensitivity::new(pcmci_for(params), Arc::clone(max_lags));
            DiscoveryCheckReport::Stability(checker.run(
                one_series(check, data)?,
                variables,
                &mut workspace,
                ctx,
            )?)
        }
        DiscoveryCheck::CiTest { ci_names } => {
            let checker = CiTestSensitivity::new(pcmci_for(params), Arc::clone(ci_names));
            DiscoveryCheckReport::Stability(checker.run(
                one_series(check, data)?,
                variables,
                &mut workspace,
                ctx,
            )?)
        }
        DiscoveryCheck::Orientation { replicates, block_size } => {
            let checker = OrientationStability {
                pcmci_plus: pcmci_plus_for(params),
                replicates: *replicates,
                block_size: *block_size,
            };
            DiscoveryCheckReport::Orientation(checker.run(
                one_series(check, data)?,
                variables,
                &mut workspace,
                ctx,
            )?)
        }
        DiscoveryCheck::SyntheticNull { n_sim, n_obs, n_vars } => {
            let checker = SyntheticNullCalibration::new(
                pcmci_for(params),
                params.alpha,
                *n_sim,
                *n_obs,
                *n_vars,
            );
            DiscoveryCheckReport::NullCalibration(checker.run(&mut workspace, ctx)?)
        }
        DiscoveryCheck::EnvironmentHoldout { n_discovery } => {
            let DiscoverData::MultiEnv(multi) = data else {
                return Err(wrong_modality(
                    "the environment_holdout check runs on environments; build the study with \
                     Study::series_multi",
                ));
            };
            let split = EnvHoldoutSplit::try_prefix(multi.env_count(), *n_discovery)?;
            let checker = EnvironmentHoldout::new(jpcmci_plus_for(params), split);
            DiscoveryCheckReport::EnvironmentHoldout(checker.run(
                multi,
                variables,
                &mut workspace,
                ctx,
            )?)
        }
        DiscoveryCheck::RegimeStability { replicates, block_size } => {
            let assignment = spec.regime_assignment().cloned().ok_or_else(|| {
                crate::compile_reason!(
                    "required_option_missing",
                    "{}: the regime_stability check reads the regime labels the search declares",
                    antecedent_core::reason_details::discovery_validation::REGIMES_MISSING
                )
            })?;
            // The labels are held fixed: the check measures stability under the declared
            // labels, not a relabelling.
            let rpcmci = rpcmci_for(params, spec.min_regime_len()).with_alternating_iters(0);
            let mut checker = RegimeStability::new(rpcmci, assignment);
            checker.replicates = *replicates;
            checker.block_size = *block_size;
            DiscoveryCheckReport::RegimeStability(checker.run(
                one_series(check, data)?,
                variables,
                &mut workspace,
                ctx,
            )?)
        }
    })
}

impl Study {
    /// Run `check` on this study's temporal structure search, through its compiled program.
    ///
    /// The study is compiled first; the check is lowered over the search's lowered program and
    /// runs the search's engine, configured from the study's own parameters, on the study's
    /// own data.
    ///
    /// # Errors
    ///
    /// A study that searches nothing or searches with another method than the check validates,
    /// a modality the check does not read, the refusals of [`Study::compile`], an admission
    /// refusal, the check's own refusals, or cancellation.
    pub fn validate_discovery(
        &self,
        check: DiscoveryCheck,
        ctx: &ExecutionContext,
    ) -> Result<DiscoveryValidation, CausalError> {
        crate::analysis::lowering::refuse_reentry(ctx)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: VALIDATION_STAGE });
        }
        let CausalQuery::Discover(query) = &self.query else {
            return Err(crate::compile_reason!(
                "route_not_supported",
                "{}: a validation check runs on a study built with StudyBuilder::discovery",
                antecedent_core::reason_details::discovery_validation::NOT_A_SEARCH
            ));
        };
        let spec = self.discovery.as_deref().ok_or_else(|| {
            crate::compile_reason!(
                "route_not_supported",
                "{}: the study carries no executable search; build it with \
                 StudyBuilder::discovery",
                antecedent_core::reason_details::discovery_validation::NOT_A_SEARCH
            )
        })?;
        if !check.admits(spec.method()) {
            return Err(wrong_search(format!(
                "the {} check validates a {} search, and the study searches with {}",
                check.id(),
                check.search_id(),
                spec.method().id()
            )));
        }
        let params = spec.temporal_params().ok_or_else(|| {
            wrong_search(format!("the {} check validates a temporal search", check.id()))
        })?;
        let plan = self.compile(ctx)?;
        let search = plan.lowered().ok_or_else(|| {
            crate::compile_reason!(
                "invalid_argument",
                "{}: the compiled search carries no lowered program",
                antecedent_core::reason_details::discovery_validation::LOWERING
            )
        })?;
        let lowered =
            Arc::new(crate::analysis::lowering::discovery_validation::lower(search, &check)?);
        gate_program(ctx, &CheckProgram(Arc::clone(&lowered)))?;
        let data = DiscoverData::from(&self.data);
        let lease = data.admit_search(query, spec, ctx, "discovery validation")?;
        let report = run_check(&check, spec, params, &data, &query.variables, ctx)?;
        drop(lease);
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: VALIDATION_STAGE });
        }
        Ok(DiscoveryValidation { lowered, check, report })
    }
}

/// The lowered program of a check, before it runs: the handle its support gate reads.
struct CheckProgram(Arc<LoweredProgram>);

impl CompiledProgramHandle for CheckProgram {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.0
    }
}

impl CompiledProgramHandle for DiscoveryValidation {
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}

#[cfg(test)]
mod site_tests {
    use std::sync::Arc;

    use antecedent_core::CanonicalWriter;

    use super::{DiscoveryCheck, NullTransform};
    use crate::identity_laws::{all_distinct, bit_exact, ordered_laws};

    fn encoded(check: &DiscoveryCheck) -> [u8; 32] {
        let mut writer = CanonicalWriter::new("site");
        check.write(&mut writer);
        writer.finish()
    }

    #[test]
    fn site_discovery_validation_check_grids_are_ordered_and_variants_tagged() {
        // Each grid is reported point by point in the declared order.
        ordered_laws(&[0.01, 0.05, 0.1], |alphas| {
            Some(encoded(&DiscoveryCheck::Alpha { alphas: Arc::from(alphas) }))
        });
        ordered_laws(&[1_u32, 2, 3], |max_lags| {
            Some(encoded(&DiscoveryCheck::LagWindow { max_lags: Arc::from(max_lags) }))
        });
        ordered_laws(&["parcorr", "gpdc"], |names| {
            let ci_names: Vec<Arc<str>> = names.iter().map(|name| Arc::from(*name)).collect();
            Some(encoded(&DiscoveryCheck::CiTest { ci_names: Arc::from(ci_names) }))
        });
        bit_exact(|alpha| Some(encoded(&DiscoveryCheck::Alpha { alphas: Arc::from([alpha]) })));
        // Variants with the same option words are told apart by their id, and options by value.
        all_distinct(&[
            encoded(&DiscoveryCheck::BlockBootstrap { replicates: 4, block_size: 20 }),
            encoded(&DiscoveryCheck::Orientation { replicates: 4, block_size: 20 }),
            encoded(&DiscoveryCheck::RegimeStability { replicates: 4, block_size: 20 }),
            encoded(&DiscoveryCheck::BlockBootstrap { replicates: 20, block_size: 4 }),
            encoded(&DiscoveryCheck::FalsePositive {
                transform: NullTransform::ColumnPermute,
                replicates: 4,
            }),
            encoded(&DiscoveryCheck::FalsePositive {
                transform: NullTransform::PhaseRandomize,
                replicates: 4,
            }),
            encoded(&DiscoveryCheck::SyntheticNull { n_sim: 4, n_obs: 20, n_vars: 3 }),
            encoded(&DiscoveryCheck::EnvironmentHoldout { n_discovery: 4 }),
            encoded(&DiscoveryCheck::Alpha { alphas: Arc::from([]) }),
            encoded(&DiscoveryCheck::LagWindow { max_lags: Arc::from([]) }),
        ]);
    }
}
