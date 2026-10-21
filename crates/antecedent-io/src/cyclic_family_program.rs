// SPDX-License-Identifier: MIT OR Apache-2.0
//! Retained typed cyclic programs. Numerical work stays in model/estimate engines;
//! encoding and independent consumers use this same execution owner.
use super::{
    CyclicArtifactError, CyclicFamilyAnswer as A, CyclicFamilyReport, CyclicFamilyRequest as R,
    bounded_json, invalid,
};
use antecedent_core::{CanonicalWriter, ExecutionContext, KernelPolicy, execution::AdmissionLease};
use antecedent_model::compile::{
    CompiledSystemPlan, compile_constraints, compile_equilibrium, compile_ode, compile_ou,
};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub(super) enum FrozenFamilyFitValue {
    Gaussian(FrozenGaussianDensity),
    Structure {
        fit: antecedent_estimate::cyclic_likelihood::CyclicStructureFit,
        bootstrap: Vec<Option<antecedent_estimate::cyclic_likelihood::CyclicStructureFit>>,
    },
    Bootstrap(Vec<Option<Vec<f64>>>),
    Parameter(antecedent_estimate::cyclic_likelihood::CyclicPosterior),
    Joint(antecedent_estimate::cyclic_likelihood::CyclicJointAbduction),
    Partial(antecedent_estimate::cyclic_likelihood::CyclicAbductionPosterior),
    Ou(Option<antecedent_estimate::ou_mechanisms::OuMechanismFit>),
    Adam(antecedent_estimate::cyclic_adam::CyclicAdamFit),
    Gradient(antecedent_estimate::cyclic_gradient::CyclicGradientFit),
    Polynomial(antecedent_estimate::constraint_likelihood::PolynomialFit),
    Ode(antecedent_estimate::cyclic_dynamics::OdeFit),
    OuFit(antecedent_estimate::cyclic_dynamics::OuFit),
}
#[derive(Clone, Debug)]
pub(crate) struct FrozenGaussianDensity {
    pub(crate) mean: Vec<f64>,
    pub(crate) covariance: Vec<f64>,
    pub(crate) fitted_rows: Option<usize>,
}
#[derive(Debug)]
struct FrozenFit {
    kernel: KernelPolicy,
    value: FrozenFamilyFitValue,
    _retention: AdmissionLease,
}
impl FrozenFit {
    fn capture(
        answer: &A,
        kernel: KernelPolicy,
        ctx: &ExecutionContext,
    ) -> Result<Option<Self>, CyclicArtifactError> {
        fn retain<T: serde::Serialize + Clone>(
            value: &T,
            ctx: &ExecutionContext,
            wrap: impl FnOnce(T) -> FrozenFamilyFitValue,
        ) -> Result<(FrozenFamilyFitValue, AdmissionLease), CyclicArtifactError> {
            let bytes = crate::cyclic_system_artifact::cyclic_serialized_size(value, ctx)? as u64;
            let retention = ctx
                .acquire_admission(
                    0,
                    bytes.checked_mul(4).ok_or_else(|| invalid("fit retention size overflow"))?,
                )
                .map_err(invalid)?;
            Ok((wrap(value.clone()), retention))
        }
        let (value, retention) = match answer {
            A::GaussianFunctional(report) if report.fitted_rows.is_some() => retain(
                &(&report.mean, &report.covariance, report.fitted_rows),
                ctx,
                |(mean, covariance, fitted_rows)| {
                    FrozenFamilyFitValue::Gaussian(FrozenGaussianDensity {
                        mean: mean.clone(),
                        covariance: covariance.clone(),
                        fitted_rows,
                    })
                },
            )?,
            A::StructureFit { fit, bootstrap } => {
                retain(&(fit, bootstrap), ctx, |(fit, bootstrap)| {
                    FrozenFamilyFitValue::Structure {
                        fit: fit.clone(),
                        bootstrap: bootstrap.clone(),
                    }
                })?
            }
            A::FrequentistBootstrap { parameters, .. } => {
                retain(parameters, ctx, FrozenFamilyFitValue::Bootstrap)?
            }
            A::ParameterPosterior { posterior, .. } => {
                retain(posterior, ctx, FrozenFamilyFitValue::Parameter)?
            }
            A::JointCounterfactual { posterior, .. } => {
                retain(posterior, ctx, FrozenFamilyFitValue::Joint)?
            }
            A::PartialCounterfactual { abduction, .. } => {
                retain(abduction, ctx, FrozenFamilyFitValue::Partial)?
            }
            A::OuMechanisms { fit, .. } => retain(fit, ctx, FrozenFamilyFitValue::Ou)?,
            A::AdamFit(fit) => retain(fit, ctx, FrozenFamilyFitValue::Adam)?,
            A::GradientFit(fit) => retain(fit, ctx, FrozenFamilyFitValue::Gradient)?,
            A::PolynomialFit(fit) => retain(fit, ctx, FrozenFamilyFitValue::Polynomial)?,
            A::OdeFit(fit) => retain(fit, ctx, FrozenFamilyFitValue::Ode)?,
            A::OuFit(fit) => retain(fit, ctx, FrozenFamilyFitValue::OuFit)?,
            _ => return Ok(None),
        };
        Ok(Some(Self { kernel, value, _retention: retention }))
    }
}
/// One immutable scientific request, compiled family schedules and synchronized statistical products.
/// Specified models remain model conditional; compilation grants no scientific license.
#[derive(Clone, Debug)]
pub struct CompiledCyclicFamilyProgram {
    request: Arc<R>,
    identity: String,
    fit_identity: String,
    plans: Arc<[CompiledSystemPlan]>,
    fit: Arc<Mutex<Option<FrozenFit>>>,
    source: Option<Arc<crate::cyclic_functional::CheckedCyclicFunctionalSource>>,
    retention: Arc<AdmissionLease>,
}
impl CompiledCyclicFamilyProgram {
    /// Compile without fitting, sampling, solving a trajectory or encoding an artifact.
    pub fn compile(request: &R, ctx: &ExecutionContext) -> Result<Self, CyclicArtifactError> {
        Self::compile_bound(request, ctx, None)
    }
    fn compile_bound(
        request: &R,
        ctx: &ExecutionContext,
        previous: Option<&Self>,
    ) -> Result<Self, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(invalid("cyclic compilation cancelled"));
        }
        let plans = compile_plans(request)?;
        let retained_source = previous
            .filter(|previous| same_source(request, &previous.request))
            .and_then(|previous| previous.source.clone());
        let source = if retained_source.is_some() {
            retained_source
        } else {
            match request {
                R::Functional { source, source_identity, program, .. }
                | R::GaussianFunctional { source, source_identity, program, .. } => {
                    Some(Arc::new(crate::cyclic_functional::checked_functional_source(
                        source,
                        source_identity,
                        program,
                        ctx,
                    )?))
                }
                _ => None,
            }
        };
        let (bytes, retention) = bounded_json(request, ctx)?;
        let identity =
            CanonicalWriter::digest_bytes_hex("antecedent.cyclic_family.request.v1", &bytes);
        let mut dependencies: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(invalid)?;
        let excluded: &[&str] = match request {
            R::FrequentistBootstrap { .. } | R::ParameterPosterior { .. } => {
                &["worlds", "noise", "initial", "solve_policy"]
            }
            R::JointCounterfactual { .. } | R::PartialCounterfactual { .. } => {
                &["worlds", "solve_policy"]
            }
            R::OuMechanisms { .. } => &["worlds"],
            R::GaussianFunctional { .. } => &["assignments"],
            _ => &[],
        };
        if let Some(object) = dependencies.as_object_mut() {
            for field in excluded {
                object.remove(*field);
            }
        }
        let fit_bytes = serde_json::to_vec(&dependencies).map_err(invalid)?;
        let fit_identity =
            CanonicalWriter::digest_bytes_hex("antecedent.cyclic_family.fit.v1", &fit_bytes);
        Ok(Self {
            request: Arc::new(request.clone()),
            identity,
            fit_identity,
            plans: plans.into(),
            fit: Arc::default(),
            source,
            retention: Arc::new(retention),
        })
    }
    /// Original typed request, including actual statistical and initial-law dependencies.
    #[must_use]
    pub fn request(&self) -> &R {
        &self.request
    }
    /// Complete target/source/numerical-policy binding, inspectable before execution.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }
    /// Existing equilibrium/dynamic/equation compiler products; never coerced to DAG semantics.
    #[must_use]
    pub fn execution_plans(&self) -> &[CompiledSystemPlan] {
        &self.plans
    }
    /// Actual request reservation retained by this owner.
    #[must_use]
    pub fn retained_bytes(&self) -> u64 {
        self.retention.remainder().bytes()
    }
    /// Compile changed targets and share only statistical products whose actual inputs match.
    pub fn rebind(&self, request: &R, ctx: &ExecutionContext) -> Result<Self, CyclicArtifactError> {
        let mut next = Self::compile_bound(request, ctx, Some(self))?;
        if next.fit_identity == self.fit_identity {
            next.fit = Arc::clone(&self.fit);
        }
        Ok(next)
    }
    /// Execute retained typed operations; numerical algorithms remain in their original engines.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicFamilyReport, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(invalid("cyclic execution cancelled"));
        }
        antecedent_core::numerical_admission::with_context(ctx, || {
            let mut cache = self.fit.lock().map_err(|_| invalid("cyclic fit lock poisoned"))?;
            let retained = cache.as_ref().filter(|fit| fit.kernel == ctx.kernel_policy);
            if retained.is_some() {
                use antecedent_core::execution_attempt::{PhysicalEvent, record_physical};
                record_physical(PhysicalEvent::CacheHit, 1);
                record_physical(PhysicalEvent::ModelFit, 0);
            }
            let report = self.request.execute_bound(
                &self.identity,
                retained.map(|fit| &fit.value),
                self.source.as_deref(),
                ctx,
            )?;
            if retained.is_none() {
                *cache = FrozenFit::capture(&report.answer, ctx.kernel_policy, ctx)?;
            }
            Ok(report)
        })
    }
    /// Execute and then encode separately. No artifact is produced during compilation/inspection.
    pub fn export(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(CyclicFamilyReport, Vec<u8>), CyclicArtifactError> {
        let report = self.execute(ctx)?;
        let bytes = self.request.encode_report(&report, ctx)?;
        Ok((report, bytes))
    }
}
fn compile_plans(request: &R) -> Result<Vec<CompiledSystemPlan>, CyclicArtifactError> {
    let plans = match request {
        R::AdamFit { model, .. }
        | R::EquilibriumRoots { model, .. }
        | R::GradientFit { model, .. }
        | R::EquilibriumActions { model, .. }
        | R::FrequentistBootstrap { model, .. }
        | R::ParameterPosterior { model, .. }
        | R::JointCounterfactual { model, .. }
        | R::PartialCounterfactual { model, .. }
        | R::StructureFit { model, .. } => {
            model.validate()?;
            let worlds = match request {
                R::FrequentistBootstrap { worlds, .. }
                | R::ParameterPosterior { worlds, .. }
                | R::JointCounterfactual { worlds, .. }
                | R::PartialCounterfactual { worlds, .. } => Some(worlds),
                _ => None,
            };
            if let Some(worlds) = worlds {
                worlds
                    .iter()
                    .map(|world| compile_equilibrium(model, world))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                vec![compile_equilibrium(model, &[])?]
            }
        }
        R::Ode { model, .. } | R::OdeFit { model, .. } => vec![compile_ode(model)?],
        R::OuStationary { model } | R::OuTransient { model, .. } | R::OuFit { model, .. } => {
            vec![compile_ou(model)?]
        }
        R::Polynomial { model, .. } => vec![compile_constraints(model)?],
        R::PolynomialCausal { program } => program
            .candidates
            .iter()
            .map(|candidate| compile_constraints(&candidate.model))
            .collect::<Result<Vec<_>, _>>()?,
        R::PolynomialFit { candidates, .. } => candidates
            .iter()
            .map(|candidate| compile_constraints(&candidate.model))
            .collect::<Result<Vec<_>, _>>()?,
        R::OuMechanisms { model, .. } => {
            model.validate()?;
            vec![]
        }
        R::Functional { .. } | R::GaussianFunctional { .. } | R::Graph { .. } => vec![],
    };
    Ok(plans)
}

fn same_source(left: &R, right: &R) -> bool {
    match (left, right) {
        (
            R::Functional { source: a, source_identity: ai, program: ap, .. },
            R::Functional { source: b, source_identity: bi, program: bp, .. },
        )
        | (
            R::GaussianFunctional { source: a, source_identity: ai, program: ap, .. },
            R::GaussianFunctional { source: b, source_identity: bi, program: bp, .. },
        ) => a == b && ai == bi && ap == bp,
        _ => false,
    }
}
