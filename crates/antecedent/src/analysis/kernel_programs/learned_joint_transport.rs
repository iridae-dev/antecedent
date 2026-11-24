//! The learned and the Gaussian joint source-target transport fits as compiled program
//! families.
//!
//! A checked transport identification, the model declaration, the source trials, the target
//! covariate sample and the draw request are everything the joint transport producers read.
//! Compiling them binds their digest into a six-layer program under the `LearnedJointTransport`
//! route (the polynomial-basis model fitted through `antecedent-learn`) or the
//! `JointBayesianTransport` route (the dense conjugate Gaussian model, measured target effect).
//! Running passes the support gate of the route's unit and then calls the producer the direct
//! path calls with the same arguments: `LearnedJointArtifactWire::build` / `build_measured` or
//! `JointBayesianArtifactWire::build_measured`. The draws come from the declared draw seed, not
//! from the context, so the artifact, the fit and the measured scalar with its calibration slot
//! are the direct path's.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::ExecutionContext;
use antecedent_estimate::joint_bayesian_transport::{
    JointTransportFit, JointTransportModel, JointTransportOptions, SourceData, TargetData,
};
use antecedent_estimate::learned_joint_transport::{LearnedJointFit, LearnedJointModel};
use antecedent_identify::TransportIdentification;
use antecedent_io::IoError;
use antecedent_io::joint_bayesian_transport_artifact::{
    JointBayesianArtifactWire, JointBayesianInputWire,
};
use antecedent_io::learned_joint_transport_artifact::LearnedJointArtifactWire;
use antecedent_io::measured_inference::MeasuredInference;
use antecedent_io::{TransportIdentificationWire, transport_identification_to_wire};
use serde::Serialize;

use super::{KernelProgram, KernelProgramShape, KernelRealization, canonical_cbor, io_refusal};
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const LEARNED: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::LearnedJointTransport,
    plan_id: "transport.learned_joint",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "learned_polynomial_basis_joint_gaussian_posterior",
    realization: KernelRealization::Posterior,
    uncertainty: UncertaintyReading::Posterior,
    role: EvidenceRole::TransportSource,
    graph_class: "fixed_dag",
    data: DataClassification::MultiEnvironment,
    design: "randomized source trials and a target covariate sample",
};

const GAUSSIAN: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::JointBayesianTransport,
    plan_id: "transport.joint_bayesian",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "dense_conjugate_gaussian_joint_posterior",
    realization: KernelRealization::Posterior,
    uncertainty: UncertaintyReading::Posterior,
    role: EvidenceRole::TransportSource,
    graph_class: "fixed_dag",
    data: DataClassification::MultiEnvironment,
    design: "randomized source trials and a target covariate sample",
};

/// The checked identification and every declared input of a joint transport fit.
#[derive(Clone, Debug)]
struct JointInput<M> {
    identification: TransportIdentification,
    model: M,
    sources: Vec<SourceData>,
    target: TargetData,
    options: JointTransportOptions,
}

/// The canonical identity of a joint transport declaration.
#[derive(Serialize)]
struct Canonical {
    identification: TransportIdentificationWire,
    input: JointBayesianInputWire,
    basis_degree: Option<usize>,
}

/// The Gaussian part of a learned model: every field but the basis degree.
fn gaussian_part(model: &LearnedJointModel) -> JointTransportModel {
    JointTransportModel {
        graph: model.graph,
        features: model.features.clone(),
        varying: model.varying,
        sharing: model.sharing,
        dependence: model.dependence,
        priors: model.priors.clone(),
        max_unsupported_mass: model.max_unsupported_mass,
        conflict_z_threshold: model.conflict_z_threshold,
    }
}

/// A compiled learned joint transport fit: the declaration and the program that binds it.
#[derive(Clone, Debug)]
pub struct CompiledLearnedJointTransport {
    program: KernelProgram<JointInput<LearnedJointModel>>,
}

/// A compiled Gaussian joint transport fit: the declaration and the program that binds it.
#[derive(Clone, Debug)]
pub struct CompiledJointBayesianTransport {
    program: KernelProgram<JointInput<JointTransportModel>>,
}

impl Study {
    /// Compile the learned joint transport fit of `model` over `sources` and `target` under
    /// the checked `identification`, under the `LearnedJointTransport` route. Compiling
    /// executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, cancellation, or a declaration that does not
    /// encode.
    pub fn compile_learned_joint_transport(
        identification: TransportIdentification,
        model: LearnedJointModel,
        sources: Vec<SourceData>,
        target: TargetData,
        options: JointTransportOptions,
        ctx: &ExecutionContext,
    ) -> Result<CompiledLearnedJointTransport, CausalError> {
        let canonical = canonical_cbor(&Canonical {
            identification: transport_identification_to_wire(&identification),
            input: JointBayesianInputWire::from_declaration(
                &gaussian_part(&model),
                &sources,
                &target,
                options,
            ),
            basis_degree: Some(model.basis_degree),
        })?;
        let program = KernelProgram::compile(
            &LEARNED,
            JointInput { identification, model, sources, target, options },
            &canonical,
            ctx,
        )?;
        Ok(CompiledLearnedJointTransport { program })
    }

    /// Compile the Gaussian joint transport fit of `model` over `sources` and `target` under
    /// the checked `identification`, under the `JointBayesianTransport` route. Compiling
    /// executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, cancellation, or a declaration that does not
    /// encode.
    pub fn compile_joint_bayesian_transport(
        identification: TransportIdentification,
        model: JointTransportModel,
        sources: Vec<SourceData>,
        target: TargetData,
        options: JointTransportOptions,
        ctx: &ExecutionContext,
    ) -> Result<CompiledJointBayesianTransport, CausalError> {
        let canonical = canonical_cbor(&Canonical {
            identification: transport_identification_to_wire(&identification),
            input: JointBayesianInputWire::from_declaration(&model, &sources, &target, options),
            basis_degree: None,
        })?;
        let program = KernelProgram::compile(
            &GAUSSIAN,
            JointInput { identification, model, sources, target, options },
            &canonical,
            ctx,
        )?;
        Ok(CompiledJointBayesianTransport { program })
    }
}

impl CompiledLearnedJointTransport {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// Pass the support gate and fit the model into its candidate artifact, as
    /// `LearnedJointArtifactWire::build` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not supported,
    /// and the refusals of the producer.
    pub fn fit(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(LearnedJointArtifactWire, LearnedJointFit), IoError> {
        self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        let input = self.program.input();
        LearnedJointArtifactWire::build(
            &input.identification,
            &input.model,
            &input.sources,
            &input.target,
            input.options,
            ctx,
        )
    }

    /// Pass the support gate, fit the model and authorize the target-effect scalar at
    /// `level` from the coverage records, as `LearnedJointArtifactWire::build_measured` does.
    ///
    /// # Errors
    ///
    /// As [`Self::fit`], and inputs over the measured bounds. Inputs outside the frozen
    /// protocol are reported without a coverage claim.
    pub fn measure(
        &self,
        level: f64,
        ctx: &ExecutionContext,
    ) -> Result<(LearnedJointArtifactWire, LearnedJointFit, MeasuredInference), IoError> {
        self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        let input = self.program.input();
        LearnedJointArtifactWire::build_measured(
            &input.identification,
            &input.model,
            &input.sources,
            &input.target,
            input.options,
            level,
            ctx,
        )
    }
}

impl CompiledJointBayesianTransport {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// Pass the support gate, fit the model and authorize the target-effect scalar at
    /// `level` from the coverage records, as `JointBayesianArtifactWire::build_measured` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not supported,
    /// inputs over the measured bounds, and the refusals of the producer. Inputs outside the
    /// frozen protocol are reported without a coverage claim.
    pub fn measure(
        &self,
        level: f64,
        ctx: &ExecutionContext,
    ) -> Result<(JointBayesianArtifactWire, JointTransportFit, MeasuredInference), IoError> {
        self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        let input = self.program.input();
        JointBayesianArtifactWire::build_measured(
            &input.identification,
            &input.model,
            &input.sources,
            &input.target,
            input.options,
            level,
            ctx,
        )
    }
}
