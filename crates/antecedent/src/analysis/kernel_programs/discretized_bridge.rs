//! The discretized linear-Gaussian bridge as a compiled program family.
//!
//! `DiscretizedLinearGaussian::new`, `from_spec` and `mean_contrast` are the kernels.
//! Compiling binds the model (or the specification) and the plan, or a bridge and a contrast.
//! Executing passes the `model.discretized_linear_gaussian_scm` support gate and calls the same
//! kernel with those inputs. The value stays a point with its computed mean-error bound.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::{CanonicalWriter, ExecutionContext, NotSupported};
use antecedent_model::compose::{
    ComposedLinearGaussian, ComposedModelSpec, ContextCoordinate, CrossCovarianceBlock, Domain,
    InvarianceDeclaration, LatentDependence, MechanismContext, MechanismDescriptor, MechanismForm,
    MechanismOrigin, MechanismProvenance, MechanismUncertainty, ParameterId, ParameterTerm,
    ParentBinding, PosteriorDependence, SelectionDiagramDeclaration, SelectionNodeDeclaration,
    TabularRow, UncertaintyMeaning,
};
use antecedent_model::discretized_scm::{
    BridgeRefusal, DETAIL_CANCELLED, DETAIL_NUMERICAL, DiscreteArm, DiscreteContrast,
    DiscretizationMethod, DiscretizationPlan, DiscretizedContrast, DiscretizedLinearGaussian,
};

use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::analysis::Study;
use crate::analysis::discretized_composition::DiscretizationRefusal;
use crate::analysis::mechanism_composition::PreparedMechanismComposition;
use crate::analysis::supplied_identity::{
    Canonical, Declaration, SetOf, encoded, unencodable, variant,
};
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const TAG_BRIDGE: &str = "antecedent.kernel_program.discretized_linear_gaussian_bridge";
const TAG_CONTRAST: &str = "antecedent.kernel_program.discretized_mean_contrast";
const MULTISET_ITEM: &str = "antecedent.kernel_program.multiset_item.v1";

const BRIDGE: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::DiscretizedLinearGaussianBridge,
    plan_id: "model.discretized_linear_gaussian_scm",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "discretized_linear_gaussian_quantile_grid",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::Observations,
    graph_class: "linear_gaussian_composition",
    data: DataClassification::Tabular,
    design: "equal-probability quantile grid of a linear-Gaussian composition, point only",
};

const CONTRAST: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::DiscretizedLinearGaussianBridge,
    plan_id: "model.discretized_linear_gaussian_scm",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "discretized_linear_gaussian_mean_contrast",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::Observations,
    graph_class: "linear_gaussian_composition",
    data: DataClassification::Tabular,
    design: "mean contrast on a discretized linear-Gaussian bridge, with its computed error bound",
};

/// A list the kernel sorts and keeps duplicates of: order is irrelevant and a repeat is a
/// different declaration.
pub(super) struct Multiset<'a, T>(pub(super) &'a [T]);

impl<T: Canonical> Canonical for Multiset<'_, T> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let mut keyed: Vec<([u8; 32], usize)> = self
            .0
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let mut writer = CanonicalWriter::new(MULTISET_ITEM);
                item.encode(&mut writer);
                (writer.finish(), index)
            })
            .collect();
        keyed.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
        out.word(self.0.len() as u64);
        for (_, index) in keyed {
            self.0[index].encode(out);
        }
    }
}

/// One plan level: the variable and how many cells it is cut into.
struct Level<'a>(&'a str, usize);

impl Canonical for Level<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        self.0.encode(out);
        self.1.encode(out);
    }
}

/// What the bridge was compiled from. The two constructors share a cell and differ here.
#[derive(Clone, Debug)]
enum BridgeSource {
    Model(ComposedLinearGaussian),
    Spec(ComposedModelSpec),
}

#[derive(Clone, Debug)]
struct BridgeInput {
    source: BridgeSource,
    plan: DiscretizationPlan,
}

#[derive(Clone, Debug)]
struct ContrastInput {
    bridge: DiscretizedLinearGaussian,
    contrast: DiscreteContrast,
}

fn encode_domain(domain: &Domain, out: &mut CanonicalWriter) {
    match domain {
        Domain::Real => {
            variant(out, "real");
        }
        Domain::Interval { lo, hi } => {
            variant(out, "interval");
            lo.encode(out);
            hi.encode(out);
        }
        Domain::Levels(levels) => {
            variant(out, "levels");
            // Canonical sort already ran; a duplicate level stays, so this is a multiset
            // written in that order.
            levels.encode(out);
        }
    }
}

fn encode_term(term: &ParameterTerm, out: &mut CanonicalWriter) {
    match term {
        ParameterTerm::Intercept => {
            variant(out, "intercept");
        }
        ParameterTerm::Coefficient(parent) => {
            variant(out, "coefficient");
            parent.encode(out);
        }
    }
}

fn encode_parameter(parameter: &ParameterId, out: &mut CanonicalWriter) {
    parameter.mechanism.encode(out);
    encode_term(&parameter.term, out);
}

fn encode_parent(parent: &ParentBinding, out: &mut CanonicalWriter) {
    parent.variable.encode(out);
    parent.units.encode(out);
    parent.transform.encode(out);
    encode_domain(&parent.domain, out);
}

fn encode_latent(latent: &LatentDependence, out: &mut CanonicalWriter) {
    latent.latent.encode(out);
    latent.loading.encode(out);
    latent.shared_with.encode(out);
}

fn encode_row(row: &TabularRow, out: &mut CanonicalWriter) {
    row.parent_levels.encode(out);
    row.probabilities.encode(out);
}

fn encode_form(form: &MechanismForm, out: &mut CanonicalWriter) {
    match form {
        MechanismForm::LinearGaussian { intercept, sigma, coefficients } => {
            variant(out, "linear_gaussian");
            intercept.encode(out);
            sigma.encode(out);
            out.word(coefficients.len() as u64);
            for (parent, coefficient) in coefficients {
                parent.encode(out);
                coefficient.encode(out);
            }
        }
        MechanismForm::Tabular { levels, rows } => {
            variant(out, "tabular");
            levels.encode(out);
            out.word(rows.len() as u64);
            for row in rows {
                encode_row(row, out);
            }
        }
    }
}

fn encode_uncertainty(uncertainty: &MechanismUncertainty, out: &mut CanonicalWriter) {
    match uncertainty.meaning {
        UncertaintyMeaning::Posterior => variant(out, "posterior"),
        UncertaintyMeaning::SamplingDistribution => variant(out, "sampling_distribution"),
    };
    out.word(uncertainty.parameters.len() as u64);
    for term in &uncertainty.parameters {
        encode_term(term, out);
    }
    uncertainty.covariance.encode(out);
}

fn encode_origin(origin: MechanismOrigin, out: &mut CanonicalWriter) {
    match origin {
        MechanismOrigin::Fitted => variant(out, "fitted"),
        MechanismOrigin::ExternalBinding => variant(out, "external_binding"),
        MechanismOrigin::Declared => variant(out, "declared"),
    };
}

fn encode_provenance(provenance: &MechanismProvenance, out: &mut CanonicalWriter) {
    encode_origin(provenance.origin, out);
    provenance.artifact_id.encode(out);
    provenance.evidence_id.encode(out);
}

pub(super) fn encode_context(context: &MechanismContext, out: &mut CanonicalWriter) {
    context.population.encode(out);
    context.regime.encode(out);
}

pub(super) fn encode_descriptor(descriptor: &MechanismDescriptor, out: &mut CanonicalWriter) {
    descriptor.variable.encode(out);
    encode_form(&descriptor.form, out);
    out.word(descriptor.parents.len() as u64);
    for parent in &descriptor.parents {
        encode_parent(parent, out);
    }
    descriptor.conditioning_set.encode(out);
    descriptor.output_units.encode(out);
    descriptor.output_transform.encode(out);
    encode_domain(&descriptor.output_support, out);
    encode_context(&descriptor.context, out);
    encode_provenance(&descriptor.provenance, out);
    out.word(descriptor.latents.len() as u64);
    for latent in &descriptor.latents {
        encode_latent(latent, out);
    }
    match &descriptor.uncertainty {
        Some(uncertainty) => {
            out.presence(true);
            encode_uncertainty(uncertainty, out);
        }
        None => {
            out.presence(false);
        }
    }
}

fn encode_coordinate(coordinate: ContextCoordinate, out: &mut CanonicalWriter) {
    match coordinate {
        ContextCoordinate::Population => variant(out, "population"),
        ContextCoordinate::Regime => variant(out, "regime"),
    };
}

pub(super) fn encode_invariance(invariance: &InvarianceDeclaration, out: &mut CanonicalWriter) {
    invariance.variable.encode(out);
    encode_coordinate(invariance.coordinate, out);
    invariance.source.encode(out);
    invariance.target.encode(out);
    invariance.assumption_id.encode(out);
}

fn encode_node(node: &SelectionNodeDeclaration, out: &mut CanonicalWriter) {
    node.name.encode(out);
    node.points_into.encode(out);
}

pub(super) fn encode_diagram(diagram: &SelectionDiagramDeclaration, out: &mut CanonicalWriter) {
    diagram.source_population.encode(out);
    diagram.target_population.encode(out);
    out.word(diagram.selection_nodes.len() as u64);
    for node in &diagram.selection_nodes {
        encode_node(node, out);
    }
}

fn encode_block(block: &CrossCovarianceBlock, out: &mut CanonicalWriter) {
    block.first.encode(out);
    block.second.encode(out);
    out.word(block.first_parameters.len() as u64);
    for term in &block.first_parameters {
        encode_term(term, out);
    }
    out.word(block.second_parameters.len() as u64);
    for term in &block.second_parameters {
        encode_term(term, out);
    }
    block.values.encode(out);
}

pub(super) fn encode_dependence(dependence: &PosteriorDependence, out: &mut CanonicalWriter) {
    match dependence {
        PosteriorDependence::Unknown => {
            variant(out, "unknown");
        }
        PosteriorDependence::IndependentByDesign { justification } => {
            variant(out, "independent_by_design");
            justification.encode(out);
        }
        PosteriorDependence::Joint { meaning, parameters, covariance } => {
            variant(out, "joint");
            match meaning {
                UncertaintyMeaning::Posterior => variant(out, "posterior"),
                UncertaintyMeaning::SamplingDistribution => variant(out, "sampling_distribution"),
            };
            out.word(parameters.len() as u64);
            for parameter in parameters {
                encode_parameter(parameter, out);
            }
            covariance.encode(out);
        }
        PosteriorDependence::CrossCovariance { blocks } => {
            variant(out, "cross_covariance");
            out.word(blocks.len() as u64);
            for block in blocks {
                encode_block(block, out);
            }
        }
    }
}

/// The specification in the order [`ComposedModelSpec::canonical`] already established.
///
/// Mechanisms, edges, invariances, diagrams, parents, coefficients, levels and covariance
/// parameters are multisets: canonical sorting keeps a duplicate and drops the caller's order.
/// A joint covariance is then the ordered matrix in that parameter order.
pub(super) fn encode_spec(spec: &ComposedModelSpec, out: &mut CanonicalWriter) {
    let spec = spec.canonical();
    encode_context(&spec.target, out);
    out.word(spec.mechanisms.len() as u64);
    for mechanism in &spec.mechanisms {
        encode_descriptor(mechanism, out);
    }
    spec.edges.encode(out);
    out.word(spec.invariances.len() as u64);
    for invariance in &spec.invariances {
        encode_invariance(invariance, out);
    }
    out.word(spec.selection_diagrams.len() as u64);
    for diagram in &spec.selection_diagrams {
        encode_diagram(diagram, out);
    }
    encode_dependence(&spec.dependence, out);
    spec.admit_declared_mechanisms.encode(out);
}

fn encode_method(method: DiscretizationMethod, out: &mut CanonicalWriter) {
    match method {
        DiscretizationMethod::EqualProbabilityQuantile => {
            variant(out, "equal_probability_quantile");
        }
        _ => unencodable(out, "discretization method"),
    }
}

/// Level counts are a set: the plan stores them in a map keyed by variable.
fn encode_plan(plan: &DiscretizationPlan, out: &mut CanonicalWriter) {
    encode_method(plan.method, out);
    let levels: Vec<Level<'_>> =
        plan.levels.iter().map(|(name, count)| Level(name.as_str(), *count)).collect();
    SetOf(&levels).encode(out);
}

fn encode_arm(arm: &DiscreteArm, out: &mut CanonicalWriter) {
    arm.weight.encode(out);
    // `resolve` writes a position vector; naming a variable twice is a refusal, and the
    // order of distinct variables is not. A repeated pair stays, so this is a multiset.
    Multiset(&arm.interventions).encode(out);
}

impl Canonical for BridgeInput {
    fn encode(&self, out: &mut CanonicalWriter) {
        match &self.source {
            BridgeSource::Model(model) => {
                variant(out, "model");
                encode_spec(model.spec(), out);
            }
            BridgeSource::Spec(spec) => {
                variant(out, "spec");
                encode_spec(spec, out);
            }
        }
        encode_plan(&self.plan, out);
    }
}

impl Canonical for ContrastInput {
    fn encode(&self, out: &mut CanonicalWriter) {
        encode_spec(self.bridge.source_model().spec(), out);
        encode_method(self.bridge.method(), out);
        let variables = self.bridge.variables();
        out.word(variables.len() as u64);
        for variable in &variables {
            variable.encode(out);
            self.bridge.level_names(variable).encode(out);
            self.bridge.level_values(variable).encode(out);
        }
        self.contrast.outcome.encode(out);
        // Diagnostics are emitted in the order the arms were given.
        out.word(self.contrast.arms.len() as u64);
        for arm in &self.contrast.arms {
            encode_arm(arm, out);
        }
    }
}

fn bridge_bytes(input: &BridgeInput) -> Result<[u8; 32], CausalError> {
    encoded(BRIDGE.plan_id, || {
        let mut declaration = Declaration::new(TAG_BRIDGE);
        declaration.field("input", input);
        declaration.finish()
    })
}

fn contrast_bytes(input: &ContrastInput) -> Result<[u8; 32], CausalError> {
    encoded(CONTRAST.plan_id, || {
        let mut declaration = Declaration::new(TAG_CONTRAST);
        declaration.field("input", input);
        declaration.finish()
    })
}

fn cancelled_bridge() -> DiscretizationRefusal {
    DiscretizationRefusal::Bridge(BridgeRefusal {
        reason_code: antecedent_core::reason_code!("cancelled_no_claim"),
        detail: DETAIL_CANCELLED,
        subject: "discretization".to_owned(),
        remedy: "rerun the discretization to completion",
        findings: Vec::new(),
    })
}

fn bridge_compile_error(error: &CausalError) -> DiscretizationRefusal {
    if matches!(error, CausalError::Cancelled { .. }) {
        return cancelled_bridge();
    }
    DiscretizationRefusal::Bridge(BridgeRefusal {
        reason_code: antecedent_core::reason_code!("invalid_argument"),
        detail: DETAIL_NUMERICAL,
        subject: error.to_string(),
        remedy: "compile the discretization outside a lowered dispatch",
        findings: Vec::new(),
    })
}

fn bridge_support(refusal: &NotSupported) -> DiscretizationRefusal {
    DiscretizationRefusal::Bridge(BridgeRefusal {
        reason_code: refusal.code(),
        detail: refusal.detail,
        subject: refusal.missing.join(","),
        remedy: "support the discretized linear-Gaussian bridge before evaluating it",
        findings: Vec::new(),
    })
}

fn refuse_cancelled(ctx: &ExecutionContext) -> Result<(), DiscretizationRefusal> {
    if ctx.cancellation.is_cancelled() { Err(cancelled_bridge()) } else { Ok(()) }
}

/// A compiled discretization of a linear-Gaussian composition or of its specification.
#[derive(Clone, Debug)]
pub struct CompiledDiscretizedBridge {
    program: KernelProgram<BridgeInput>,
}

impl CompiledDiscretizedBridge {
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

    /// Pass the support gate, then call `DiscretizedLinearGaussian::new` or `from_spec`
    /// with the bound model or specification and plan.
    ///
    /// # Errors
    /// The support refusal, and every refusal of the constructor.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<DiscretizedLinearGaussian, DiscretizationRefusal> {
        let publication = self.program.gate(ctx).map_err(|refusal| bridge_support(&refusal))?;
        let input = self.program.input();
        publication.finish(match &input.source {
            BridgeSource::Model(model) => {
                DiscretizedLinearGaussian::new(model, &input.plan, ctx).map_err(Into::into)
            }
            BridgeSource::Spec(spec) => {
                DiscretizedLinearGaussian::from_spec(spec, &input.plan, ctx).map_err(Into::into)
            }
        })
    }
}

/// A compiled mean contrast on a discretized bridge.
#[derive(Clone, Debug)]
pub struct CompiledDiscretizedMeanContrast {
    program: KernelProgram<ContrastInput>,
}

impl CompiledDiscretizedMeanContrast {
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

    /// Pass the support gate, then call `DiscretizedLinearGaussian::mean_contrast` with the
    /// bound bridge and contrast.
    ///
    /// # Errors
    /// The support refusal, and every refusal of `mean_contrast`.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<DiscretizedContrast, DiscretizationRefusal> {
        let publication = self.program.gate(ctx).map_err(|refusal| bridge_support(&refusal))?;
        let input = self.program.input();
        publication.finish(input.bridge.mean_contrast(&input.contrast, ctx).map_err(Into::into))
    }
}

impl Study {
    /// Compile `DiscretizedLinearGaussian::new` for this model and plan. Compiling executes
    /// nothing.
    ///
    /// # Errors
    /// Cancellation, compiling from inside a lowered dispatch, or a plan method with no encoding.
    pub fn compile_discretized_linear_gaussian(
        model: &ComposedLinearGaussian,
        plan: &DiscretizationPlan,
        ctx: &ExecutionContext,
    ) -> Result<CompiledDiscretizedBridge, DiscretizationRefusal> {
        refuse_cancelled(ctx)?;
        let input = BridgeInput { source: BridgeSource::Model(model.clone()), plan: plan.clone() };
        let bytes = bridge_bytes(&input).map_err(|error| bridge_compile_error(&error))?;
        let program = KernelProgram::compile(&BRIDGE, input, &bytes, ctx)
            .map_err(|error| bridge_compile_error(&error))?;
        Ok(CompiledDiscretizedBridge { program })
    }

    /// Compile `DiscretizedLinearGaussian::from_spec` for this specification and plan.
    ///
    /// # Errors
    /// Cancellation, compiling from inside a lowered dispatch, or a plan method with no encoding.
    pub fn compile_discretized_linear_gaussian_from_spec(
        spec: &ComposedModelSpec,
        plan: &DiscretizationPlan,
        ctx: &ExecutionContext,
    ) -> Result<CompiledDiscretizedBridge, DiscretizationRefusal> {
        refuse_cancelled(ctx)?;
        let input = BridgeInput { source: BridgeSource::Spec(spec.clone()), plan: plan.clone() };
        let bytes = bridge_bytes(&input).map_err(|error| bridge_compile_error(&error))?;
        let program = KernelProgram::compile(&BRIDGE, input, &bytes, ctx)
            .map_err(|error| bridge_compile_error(&error))?;
        Ok(CompiledDiscretizedBridge { program })
    }

    /// Compile `DiscretizedLinearGaussian::new` for the Gaussian model of a prepared composition
    /// and a plan. The same call [`PreparedMechanismComposition::discretize`] makes.
    ///
    /// # Errors
    /// A composition that has no Gaussian executable, and every refusal of
    /// [`Self::compile_discretized_linear_gaussian`].
    pub fn compile_discretized_composition(
        composition: &PreparedMechanismComposition,
        plan: &DiscretizationPlan,
        ctx: &ExecutionContext,
    ) -> Result<CompiledDiscretizedBridge, DiscretizationRefusal> {
        let model = composition.gaussian().map_err(DiscretizationRefusal::Composition)?;
        Self::compile_discretized_linear_gaussian(model, plan, ctx)
    }

    /// Compile `DiscretizedLinearGaussian::mean_contrast` for this bridge and contrast.
    ///
    /// # Errors
    /// Cancellation, compiling from inside a lowered dispatch, or a plan method with no encoding.
    pub fn compile_discretized_mean_contrast(
        bridge: &DiscretizedLinearGaussian,
        contrast: &DiscreteContrast,
        ctx: &ExecutionContext,
    ) -> Result<CompiledDiscretizedMeanContrast, DiscretizationRefusal> {
        refuse_cancelled(ctx)?;
        let input = ContrastInput { bridge: bridge.clone(), contrast: contrast.clone() };
        let bytes = contrast_bytes(&input).map_err(|error| bridge_compile_error(&error))?;
        let program = KernelProgram::compile(&CONTRAST, input, &bytes, ctx)
            .map_err(|error| bridge_compile_error(&error))?;
        Ok(CompiledDiscretizedMeanContrast { program })
    }
}

#[cfg(test)]
mod site_tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::identity_laws::{ADVERSARIAL_TEXT, bit_exact, injective, ordered_laws, set_laws};
    use antecedent_model::compose::ParentBinding;

    fn levels_of(pairs: &[(String, usize)]) -> BTreeMap<String, usize> {
        pairs.iter().cloned().collect()
    }

    fn context(population: &str, regime: &str) -> MechanismContext {
        MechanismContext { population: population.to_owned(), regime: regime.to_owned() }
    }

    fn descriptor(variable: &str, intercept: f64) -> MechanismDescriptor {
        MechanismDescriptor {
            variable: variable.to_owned(),
            form: MechanismForm::LinearGaussian { intercept, sigma: 1.0, coefficients: Vec::new() },
            parents: Vec::new(),
            conditioning_set: Vec::new(),
            output_units: "u".to_owned(),
            output_transform: "identity".to_owned(),
            output_support: Domain::Real,
            context: context("pop", "observational"),
            provenance: MechanismProvenance {
                origin: MechanismOrigin::Fitted,
                artifact_id: format!("fit:{variable}"),
                evidence_id: format!("data:{variable}"),
            },
            latents: Vec::new(),
            uncertainty: None,
        }
    }

    fn spec(population: &str, regime: &str, intercept: f64) -> ComposedModelSpec {
        ComposedModelSpec {
            target: context(population, regime),
            mechanisms: vec![descriptor("X", intercept)],
            edges: Vec::new(),
            invariances: Vec::new(),
            selection_diagrams: Vec::new(),
            dependence: PosteriorDependence::Unknown,
            admit_declared_mechanisms: false,
        }
    }

    fn plan(levels: &[(String, usize)]) -> DiscretizationPlan {
        DiscretizationPlan {
            method: DiscretizationMethod::EqualProbabilityQuantile,
            levels: levels_of(levels),
        }
    }

    fn digest(
        population: &str,
        regime: &str,
        intercept: f64,
        levels: &[(String, usize)],
    ) -> [u8; 32] {
        bridge_bytes(&BridgeInput {
            source: BridgeSource::Spec(spec(population, regime, intercept)),
            plan: plan(levels),
        })
        .unwrap()
    }

    /// `bridge_bytes`: level counts are a set (the plan's map); a joint covariance, once its
    /// parameters are in canonical order, is the ordered matrix the checker reads; population
    /// and regime are framed text; an intercept enters by its bits.
    #[test]
    fn site_discretized_bridge_levels_are_a_set_and_covariance_is_ordered() {
        let base = [("X".to_owned(), 4), ("Y".to_owned(), 8)];
        set_laws(&base, |levels| Some(digest("pop", "observational", 0.0, levels)));
        injective(
            &ADVERSARIAL_TEXT,
            |parts| *parts,
            |&(population, regime)| Some(digest(population, regime, 0.0, &base)),
        );
        bit_exact(|intercept| Some(digest("pop", "observational", intercept, &base)));

        let covariance_of = |covariance: &[f64]| {
            let mut declared = spec("pop", "observational", 0.0);
            declared.dependence = PosteriorDependence::Joint {
                meaning: UncertaintyMeaning::Posterior,
                parameters: vec![
                    ParameterId { mechanism: "X".to_owned(), term: ParameterTerm::Intercept },
                    ParameterId {
                        mechanism: "X".to_owned(),
                        term: ParameterTerm::Coefficient("Z".to_owned()),
                    },
                ],
                covariance: covariance.to_vec(),
            };
            bridge_bytes(&BridgeInput { source: BridgeSource::Spec(declared), plan: plan(&base) })
                .ok()
        };
        ordered_laws(&[1.0, 0.25, 0.25, 2.0], covariance_of);
    }

    fn binding(variable: &str) -> ParentBinding {
        ParentBinding {
            variable: variable.to_owned(),
            units: "u".to_owned(),
            transform: "identity".to_owned(),
            domain: Domain::Real,
        }
    }

    fn linked(variable: &str, parent: &str) -> MechanismDescriptor {
        let mut descriptor = descriptor(variable, 0.0);
        descriptor.form = MechanismForm::LinearGaussian {
            intercept: 0.0,
            sigma: 1.0,
            coefficients: vec![(parent.to_owned(), 1.0)],
        };
        descriptor.parents = vec![binding(parent)];
        descriptor.conditioning_set = vec![parent.to_owned()];
        descriptor
    }

    fn tiny_bridge() -> DiscretizedLinearGaussian {
        let declared = ComposedModelSpec {
            target: context("pop", "observational"),
            mechanisms: vec![descriptor("X", 0.0), linked("Y", "X")],
            edges: vec![("X".to_owned(), "Y".to_owned())],
            invariances: Vec::new(),
            selection_diagrams: Vec::new(),
            dependence: PosteriorDependence::Unknown,
            admit_declared_mechanisms: false,
        };
        let model = ComposedLinearGaussian::compile(&declared).expect("a two-variable chain");
        let plan = DiscretizationPlan::uniform(
            DiscretizationMethod::EqualProbabilityQuantile,
            model.variables(),
            2,
        );
        DiscretizedLinearGaussian::new(&model, &plan, &ExecutionContext::for_tests(1))
            .expect("two levels of a standard normal")
    }

    /// `contrast_bytes`: arms are ordered (diagnostics follow the request); interventions in an
    /// arm are a multiset (a position vector, and a repeated variable is a different request).
    #[test]
    fn site_discretized_contrast_arms_are_ordered_and_interventions_are_a_multiset() {
        use crate::identity_laws::multiset_laws;

        let bridge = tiny_bridge();
        let digest = |arms: &[DiscreteArm]| {
            contrast_bytes(&ContrastInput {
                bridge: bridge.clone(),
                contrast: DiscreteContrast { outcome: "Y".to_owned(), arms: arms.to_vec() },
            })
            .ok()
        };
        ordered_laws(
            &[
                DiscreteArm { weight: 1.0, interventions: vec![("X".to_owned(), 1)] },
                DiscreteArm { weight: -1.0, interventions: vec![("X".to_owned(), 0)] },
            ],
            digest,
        );
        let interventions_of = |pairs: &[(String, usize)]| {
            digest(&[DiscreteArm { weight: 1.0, interventions: pairs.to_vec() }])
        };
        multiset_laws(&[("X".to_owned(), 0), ("Y".to_owned(), 1)], interventions_of);
    }
}
