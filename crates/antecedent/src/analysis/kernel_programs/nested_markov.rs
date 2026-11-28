//! The binary nested-Markov Fisher interval and posterior as compiled program families.
//!
//! A declared ADMG with its regime counts and fit options is everything the binary
//! nested-Markov producers read. Compiling one binds the digest of that declaration (with the
//! nominal level for the Fisher interval, or the prior and sampler for the posterior) into a
//! six-layer program under the `NestedMarkovFisherInterval` or `NestedMarkovPosterior` route.
//! Running passes the support gate of the route's unit and then calls the producer the direct
//! path calls with the same arguments: `NestedFisherArtifact::build` / `build_measured` or the
//! Bayesian `Artifact::build` / `build_measured`. The artifact, every number in it and the
//! measured scalars with their calibration slots are the direct path's.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_core::ExecutionContext;
use antecedent_estimate::nested_markov_binary::{
    AdmgDeclaration, FitOptions, NestedMarkovInput, Regime,
};
use antecedent_io::IoError;
use antecedent_io::measured_inference::MeasuredInference;
use antecedent_io::nested_markov_artifact::{
    NestedFisherArtifact, NestedGraphWire, NestedOptionsWire, NestedRegimeWire,
};
use antecedent_io::nested_markov_bayesian_artifact::Artifact as NestedPosteriorArtifact;
use antecedent_learn::nested_markov_bayesian::{Options, Prior};
use serde::Serialize;

use super::{KernelProgram, KernelProgramShape, KernelRealization, canonical_cbor, io_refusal};
use crate::analysis::Study;
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const FISHER: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::NestedMarkovFisherInterval,
    plan_id: "identify.binary_nested_markov_fisher",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "binary_verma_expected_fisher_delta",
    realization: KernelRealization::Score,
    uncertainty: UncertaintyReading::Sampling,
    role: EvidenceRole::Observations,
    graph_class: "admg",
    data: DataClassification::Tabular,
    design: "constrained nested-Markov likelihood over declared regime counts",
};

const POSTERIOR: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::NestedMarkovPosterior,
    plan_id: "identify.binary_nested_markov",
    functional: FunctionalKind::InterventionalDistribution,
    evaluator: "binary_verma_continuous_beta_slice_posterior",
    realization: KernelRealization::Posterior,
    uncertainty: UncertaintyReading::Posterior,
    role: EvidenceRole::Observations,
    graph_class: "admg",
    data: DataClassification::Tabular,
    design: "nested-Markov posterior over declared regime counts",
};

/// The declared graph as the kernel reads it, which is the form the artifacts store: the
/// variables in coordinate order, the directed edges sorted and each once, and the bidirected
/// edges as unordered pairs (lower endpoint first), sorted and each once. The kernel compares
/// and fits that canonical form, so two spellings of one graph are one declaration.
fn canonical_graph(graph: &AdmgDeclaration) -> NestedGraphWire {
    let mut directed = graph.directed.clone();
    directed.sort_unstable();
    directed.dedup();
    let mut bidirected: Vec<(usize, usize)> =
        graph.bidirected.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
    bidirected.sort_unstable();
    bidirected.dedup();
    NestedGraphWire { variables: graph.variables.clone(), directed, bidirected }
}

/// The declared graph, regime counts and fit options, as the artifacts store them.
#[derive(Serialize)]
struct Declaration {
    graph: NestedGraphWire,
    regimes: Vec<NestedRegimeWire>,
    options: NestedOptionsWire,
}

impl Declaration {
    fn of(input: &NestedMarkovInput, options: &FitOptions) -> Self {
        Self {
            graph: canonical_graph(&input.graph),
            regimes: input
                .regimes
                .iter()
                .map(|counts| {
                    let (regime, fixed) = match &counts.regime {
                        Regime::Observational => ("observational", Vec::new()),
                        Regime::Interventional(fixed) => ("interventional", fixed.clone()),
                    };
                    NestedRegimeWire {
                        regime: regime.into(),
                        fixed,
                        levels: counts.levels.clone(),
                        cells: counts.cells.clone(),
                    }
                })
                .collect(),
            options: NestedOptionsWire {
                max_iterations: options.max_iterations,
                tolerance: options.tolerance,
                refuse_constraint_residual_above: options.refuse_constraint_residual_above,
            },
        }
    }
}

/// The input of a compiled Fisher interval.
#[derive(Clone, Debug)]
struct FisherInput {
    input: NestedMarkovInput,
    options: FitOptions,
    nominal_level: f64,
}

/// The input of a compiled nested-Markov posterior.
#[derive(Clone, Debug)]
struct PosteriorInput {
    input: NestedMarkovInput,
    options: FitOptions,
    prior: Prior,
    sampler: Options,
}

/// A compiled binary nested-Markov Fisher interval: the declaration and the program that binds
/// it.
#[derive(Clone, Debug)]
pub struct CompiledNestedMarkovFisher {
    program: KernelProgram<FisherInput>,
}

/// A compiled binary nested-Markov posterior: the declaration, the prior, the sampler and the
/// program that binds them.
#[derive(Clone, Debug)]
pub struct CompiledNestedMarkovPosterior {
    program: KernelProgram<PosteriorInput>,
}

impl Study {
    /// Compile the binary nested-Markov Fisher interval of `input` under the
    /// `NestedMarkovFisherInterval` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, cancellation, or a declaration that does not
    /// encode.
    pub fn compile_nested_markov_fisher(
        input: NestedMarkovInput,
        options: FitOptions,
        nominal_level: f64,
        ctx: &ExecutionContext,
    ) -> Result<CompiledNestedMarkovFisher, CausalError> {
        #[derive(Serialize)]
        struct Canonical {
            declaration: Declaration,
            nominal_level: f64,
        }
        let canonical = canonical_cbor(&Canonical {
            declaration: Declaration::of(&input, &options),
            nominal_level,
        })?;
        let program = KernelProgram::compile(
            &FISHER,
            FisherInput { input, options, nominal_level },
            &canonical,
            ctx,
        )?;
        Ok(CompiledNestedMarkovFisher { program })
    }

    /// Compile the binary nested-Markov posterior of `input` under `prior` and `sampler` under
    /// the `NestedMarkovPosterior` route. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, cancellation, or a declaration that does not
    /// encode.
    pub fn compile_nested_markov_posterior(
        input: NestedMarkovInput,
        options: FitOptions,
        prior: Prior,
        sampler: Options,
        ctx: &ExecutionContext,
    ) -> Result<CompiledNestedMarkovPosterior, CausalError> {
        #[derive(Serialize)]
        struct Canonical<'a> {
            declaration: Declaration,
            prior: &'a Prior,
            sampler: &'a Options,
        }
        let canonical = canonical_cbor(&Canonical {
            declaration: Declaration::of(&input, &options),
            prior: &prior,
            sampler: &sampler,
        })?;
        let program = KernelProgram::compile(
            &POSTERIOR,
            PosteriorInput { input, options, prior, sampler },
            &canonical,
            ctx,
        )?;
        Ok(CompiledNestedMarkovPosterior { program })
    }
}

impl CompiledNestedMarkovFisher {
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

    /// Pass the support gate and build the Fisher candidate artifact, as
    /// `NestedFisherArtifact::build` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not supported,
    /// and the refusals of the producer.
    pub fn fit(&self, ctx: &ExecutionContext) -> Result<NestedFisherArtifact, IoError> {
        let publication = self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        let input = self.program.input();
        publication.finish(NestedFisherArtifact::build(
            &input.input,
            &input.options,
            input.nominal_level,
            ctx,
        ))
    }

    /// Pass the support gate, build the Fisher candidate and authorize every reported scalar
    /// from the coverage records, as `NestedFisherArtifact::build_measured` does.
    ///
    /// # Errors
    ///
    /// As [`Self::fit`], and labels over the measured byte bound. A fit protocol outside the
    /// measured one is reported with calibration `not_measured`.
    pub fn measure(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(NestedFisherArtifact, MeasuredInference), IoError> {
        let publication = self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        let input = self.program.input();
        publication.finish(NestedFisherArtifact::build_measured(
            &input.input,
            &input.options,
            input.nominal_level,
            ctx,
        ))
    }
}

impl CompiledNestedMarkovPosterior {
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

    /// Pass the support gate and fit the posterior candidate artifact, as the Bayesian
    /// `Artifact::build` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) in a release build when the unit is not supported,
    /// and the refusals of the producer.
    pub fn fit(&self, ctx: &ExecutionContext) -> Result<NestedPosteriorArtifact, IoError> {
        let publication = self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        let input = self.program.input();
        publication.finish(NestedPosteriorArtifact::build(
            &input.input,
            &input.options,
            input.prior.clone(),
            input.sampler.clone(),
            ctx,
        ))
    }

    /// Pass the support gate, fit the posterior candidate and authorize every reported scalar
    /// from the coverage records, as the Bayesian `Artifact::build_measured` does.
    ///
    /// # Errors
    ///
    /// As [`Self::fit`], and labels over the measured byte bound.
    pub fn measure(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(NestedPosteriorArtifact, MeasuredInference), IoError> {
        let publication = self.program.gate(ctx).map_err(|refusal| io_refusal(&refusal))?;
        let input = self.program.input();
        publication.finish(NestedPosteriorArtifact::build_measured(
            &input.input,
            &input.options,
            input.prior.clone(),
            input.sampler.clone(),
            ctx,
        ))
    }
}

#[cfg(test)]
mod site_tests {
    use antecedent_core::SemanticDigest;
    use antecedent_estimate::nested_markov_binary::RegimeCounts;

    use super::*;
    use crate::identity_laws::{all_distinct, ordered_laws, set_laws};

    fn input(graph: AdmgDeclaration, cells: Vec<f64>) -> NestedMarkovInput {
        NestedMarkovInput {
            graph,
            regimes: vec![RegimeCounts {
                regime: Regime::Observational,
                levels: vec![2, 2, 2, 2],
                cells,
            }],
        }
    }

    fn cells() -> Vec<f64> {
        (1..=16).map(f64::from).collect()
    }

    fn fisher(graph: AdmgDeclaration, cells: Vec<f64>, level: f64) -> SemanticDigest {
        let ctx = ExecutionContext::for_tests(0);
        Study::compile_nested_markov_fisher(input(graph, cells), FitOptions::default(), level, &ctx)
            .unwrap()
            .lowered()
            .identity()
    }

    fn posterior(graph: AdmgDeclaration, prior: Prior) -> SemanticDigest {
        let ctx = ExecutionContext::for_tests(0);
        Study::compile_nested_markov_posterior(
            input(graph, cells()),
            FitOptions::default(),
            prior,
            Options::default(),
            &ctx,
        )
        .unwrap()
        .lowered()
        .identity()
    }

    fn with_edges(directed: &[(usize, usize)], bidirected: &[(usize, usize)]) -> AdmgDeclaration {
        AdmgDeclaration {
            directed: directed.to_vec(),
            bidirected: bidirected.to_vec(),
            ..AdmgDeclaration::selected()
        }
    }

    /// `compile_nested_markov_fisher`: the graph's edges are sets (the kernel reads the sorted,
    /// deduplicated, unordered-bidirected form), the variables and the cells are ordered
    /// (coordinate order, `x1` most significant), the level by its bits.
    #[test]
    fn site_nested_markov_fisher_edges_are_sets() {
        let bidirected = [(1, 3)];
        set_laws(&[(0, 1), (1, 2), (2, 3)], |directed| {
            Some(fisher(with_edges(directed, &bidirected), cells(), 0.95))
        });
        let directed = [(0, 1), (1, 2), (2, 3)];
        set_laws(&[(1, 3), (0, 2)], |bidirected| {
            Some(fisher(with_edges(&directed, bidirected), cells(), 0.95))
        });
        assert_eq!(
            fisher(with_edges(&directed, &[(3, 1)]), cells(), 0.95),
            fisher(with_edges(&directed, &[(1, 3)]), cells(), 0.95),
            "a bidirected edge is an unordered pair"
        );
        ordered_laws(&["X1", "X2", "X3", "X4"], |names| {
            let graph = AdmgDeclaration {
                variables: names.iter().map(|name| (*name).to_owned()).collect(),
                ..AdmgDeclaration::selected()
            };
            Some(fisher(graph, cells(), 0.95))
        });
        let mut swapped = cells();
        swapped.swap(0, 15);
        assert_ne!(
            fisher(AdmgDeclaration::selected(), swapped, 0.95),
            fisher(AdmgDeclaration::selected(), cells(), 0.95),
            "the cells are ordered"
        );
        assert_ne!(
            fisher(AdmgDeclaration::selected(), cells(), 0.9),
            fisher(AdmgDeclaration::selected(), cells(), 0.95),
            "the level is read"
        );
    }

    /// `compile_nested_markov_posterior`: the same canonical graph, and the prior is read.
    #[test]
    fn site_nested_markov_posterior_edges_are_sets() {
        set_laws(&[(0, 1), (1, 2), (2, 3)], |directed| {
            Some(posterior(with_edges(directed, &[(1, 3)]), Prior::default()))
        });
        let mut other = Prior::default();
        other.alpha[0] = 2.0;
        all_distinct(&[
            posterior(AdmgDeclaration::selected(), Prior::default()),
            posterior(AdmgDeclaration::selected(), other),
        ]);
    }
}
