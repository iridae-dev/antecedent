use super::{
    CertificateBasis, CertificateGap, CyclicArtifactError, CyclicWorldOutcome, CyclicWorldResult,
    Deserialize, EquilibriumOutcome, OdeTrajectory, Serialize, WorldCertificate, WorldSolvedBy,
    invalid,
};

/// Action of one evaluated world.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicActionAnswer {
    /// Stable action kind (`hard_clamp`, `replace_mechanism`, `change_mechanism`,
    /// `additive_shock`, `initial_state_change`).
    pub kind: String,
    /// Target variable id.
    pub variable: u32,
    /// Model position.
    pub node: usize,
}

/// Conditioning diagnostic for one compiled equilibrium block.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicBlockDiagnosticAnswer {
    /// Model positions in the block.
    pub members: Vec<usize>,
    /// Structural residual infinity norm.
    pub residual_inf: f64,
    /// Infinity-norm condition number or numerical estimate.
    pub kappa_inf: f64,
    /// Whether the condition number is the closed form.
    pub exact: bool,
}

/// What the world evaluated to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum CyclicWorldOutcomeAnswer {
    /// Gaussian equilibrium law in variable order.
    GaussianLaw {
        /// Equilibrium mean.
        mean: Vec<f64>,
        /// Row-major equilibrium covariance.
        covariance: Vec<f64>,
        /// Finite structural residual of the compile, when one was reported.
        residual_inf: Option<f64>,
        /// Compiled block diagnostics; absent for direct numerical routes and older artifacts.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        block_diagnostics: Option<Vec<CyclicBlockDiagnosticAnswer>>,
    },
    /// One outcome per disturbance draw; failed draws stay in place.
    EquilibriumDraws {
        /// Draw-aligned outcomes.
        outcomes: Vec<EquilibriumOutcome>,
    },
    /// Trajectory at the declared observation times.
    Trajectory {
        /// Integrated trajectory.
        trajectory: OdeTrajectory,
    },
    /// No solution was returned.
    NoSolution {
        /// Why.
        cause: String,
        /// Next attempt name.
        next_attempt: String,
    },
    /// Not computed because a required certificate was not shown.
    Withheld,
}

/// Next attempt of a refusal or of an uncertified world.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NextAttemptAnswer {
    /// Stable name.
    pub name: String,
    /// What to declare instead.
    pub sentence: String,
}

/// Uniqueness certificate of one world. A certificate is not physical stability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldCertificateAnswer {
    /// Whether the world is certified for the declared model.
    pub certified: bool,
    /// What the certificate rests on, when certified.
    pub basis: Option<String>,
    /// Gap, when not certified.
    pub gap: Option<String>,
    /// Upper bound on `ρ(|K|)` of the intervened map, when the linear family was assessed.
    pub rho_abs_upper: Option<f64>,
    /// Draws covered by a nonlinear certificate.
    pub certified_draws: Option<usize>,
    /// Draws solved without a uniqueness certificate.
    pub uncertified_draws: Option<usize>,
    /// Draws that failed.
    pub failed_draws: Option<usize>,
    /// Draws withheld for lack of a certificate.
    pub withheld_draws: Option<usize>,
    /// What would earn the certificate, or what to try instead.
    pub next_attempt: Option<NextAttemptAnswer>,
}

/// How the world was solved.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldSolvedByAnswer {
    /// Stable route name.
    pub method: String,
    /// Topological order or free positions, when the route has one.
    pub order: Vec<usize>,
    /// Solve blocks, sources first, when the route has them.
    pub blocks: Vec<Vec<usize>>,
    /// The DAG algorithm's refusal that sent the world to block factoring.
    pub dag_refusal: Option<String>,
    /// Accepted integration steps.
    pub accepted_steps: Option<usize>,
    /// Rejected integration steps.
    pub rejected_steps: Option<usize>,
}

/// One evaluated typed world, keyed by its scientific world id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicWorldAnswer {
    /// Scientific world id.
    pub world: u32,
    /// Family the world was admitted on.
    pub family: String,
    /// Variable ids in outcome order.
    pub variables: Vec<u32>,
    /// Resolved actions, sorted by model position.
    pub actions: Vec<CyclicActionAnswer>,
    /// Outcome.
    pub outcome: CyclicWorldOutcomeAnswer,
    /// Uniqueness certificate.
    pub certificate: WorldCertificateAnswer,
    /// Solve route.
    pub solved_by: WorldSolvedByAnswer,
    /// Share of mass without a solution. Never renormalized away.
    pub failed_mass: f64,
    /// Share of mass withheld for lack of a certificate.
    pub withheld_mass: f64,
    /// Physical stability. Never claimed by these evaluators.
    pub physical_stability: String,
}

fn attempt(next: antecedent_core::refusal::NextAttempt) -> NextAttemptAnswer {
    NextAttemptAnswer { name: next.name().to_owned(), sentence: next.sentence().to_owned() }
}

fn certificate_answer(certificate: &WorldCertificate) -> WorldCertificateAnswer {
    let mut answer = WorldCertificateAnswer {
        certified: certificate.is_certified(),
        basis: None,
        gap: None,
        rho_abs_upper: None,
        certified_draws: None,
        uncertified_draws: None,
        failed_draws: None,
        withheld_draws: None,
        next_attempt: certificate.next_attempt().map(attempt),
    };
    match certificate {
        WorldCertificate::Certified { basis } => match basis {
            CertificateBasis::LinearFamilyContraction { rho_abs_upper } => {
                answer.basis = Some("linear_family_contraction".into());
                answer.rho_abs_upper = Some(*rho_abs_upper);
            }
            CertificateBasis::NonlinearBlockContraction { draws } => {
                answer.basis = Some("nonlinear_block_contraction".into());
                answer.certified_draws = Some(*draws);
            }
            _ => answer.basis = Some("declared_certificate".into()),
        },
        WorldCertificate::NotCertified { gap, .. } => match gap {
            CertificateGap::LinearFamilyBoundInsufficient { rho_abs_upper } => {
                answer.gap = Some("linear_family_bound_insufficient".into());
                answer.rho_abs_upper = Some(*rho_abs_upper).filter(|rho| rho.is_finite());
            }
            CertificateGap::NonlinearUniquenessUnproved { uncertified_draws } => {
                answer.gap = Some("nonlinear_uniqueness_unproved".into());
                answer.uncertified_draws = Some(*uncertified_draws);
            }
            CertificateGap::NonlinearMassLost { failed_draws, withheld_draws } => {
                answer.gap = Some("nonlinear_mass_lost".into());
                answer.failed_draws = Some(*failed_draws);
                answer.withheld_draws = Some(*withheld_draws);
            }
            CertificateGap::NumericalIntegrationOnly => {
                answer.gap = Some("numerical_integration_only".into());
            }
            CertificateGap::NoSolution => answer.gap = Some("no_solution".into()),
            _ => answer.gap = Some("not_certified".into()),
        },
    }
    answer
}

fn solved_by_answer(solved_by: &WorldSolvedBy) -> WorldSolvedByAnswer {
    let mut answer = WorldSolvedByAnswer {
        method: solved_by.as_str().to_owned(),
        order: Vec::new(),
        blocks: Vec::new(),
        dag_refusal: None,
        accepted_steps: None,
        rejected_steps: None,
    };
    match solved_by {
        WorldSolvedBy::AcyclicOrder { order } => answer.order.clone_from(order),
        WorldSolvedBy::StronglyConnectedBlocks { dag_refusal, blocks } => {
            answer.dag_refusal = Some(dag_refusal.clone());
            answer.blocks.clone_from(blocks);
        }
        WorldSolvedBy::DirectFreeBlockSolve { free } => answer.order.clone_from(free),
        WorldSolvedBy::NonlinearBlocks { blocks, .. } => answer.blocks.clone_from(blocks),
        WorldSolvedBy::AdaptiveIntegration { accepted_steps, rejected_steps, .. } => {
            answer.accepted_steps = Some(*accepted_steps);
            answer.rejected_steps = Some(*rejected_steps);
        }
        _ => {}
    }
    answer
}

pub(super) fn block_diagnostic_answer(
    block: &antecedent_model::scc_compile::BlockDiagnostic,
) -> Result<CyclicBlockDiagnosticAnswer, CyclicArtifactError> {
    if !block.residual_inf.is_finite() || !block.kappa_inf.is_finite() {
        return Err(invalid("non-finite cyclic block diagnostic"));
    }
    Ok(CyclicBlockDiagnosticAnswer {
        members: block.members.clone(),
        residual_inf: block.residual_inf,
        kappa_inf: block.kappa_inf,
        exact: block.exact,
    })
}

pub(super) fn world_answer(
    result: &CyclicWorldResult,
) -> Result<CyclicWorldAnswer, CyclicArtifactError> {
    let outcome = match result.outcome() {
        CyclicWorldOutcome::GaussianLaw { mean, covariance, residual_inf, block_diagnostics } => {
            CyclicWorldOutcomeAnswer::GaussianLaw {
                mean: mean.clone(),
                covariance: covariance.clone(),
                residual_inf: residual_inf.filter(|value| value.is_finite()),
                block_diagnostics: block_diagnostics
                    .as_ref()
                    .map(|blocks| blocks.iter().map(block_diagnostic_answer).collect())
                    .transpose()?,
            }
        }
        CyclicWorldOutcome::EquilibriumDraws { outcomes } => {
            CyclicWorldOutcomeAnswer::EquilibriumDraws { outcomes: outcomes.clone() }
        }
        CyclicWorldOutcome::Trajectory { trajectory } => {
            CyclicWorldOutcomeAnswer::Trajectory { trajectory: trajectory.clone() }
        }
        CyclicWorldOutcome::NoSolution { failure } => CyclicWorldOutcomeAnswer::NoSolution {
            cause: format!("{:?}", failure.cause),
            next_attempt: failure.next_attempt.name().to_owned(),
        },
        CyclicWorldOutcome::Withheld => CyclicWorldOutcomeAnswer::Withheld,
        _ => return Err(invalid("cyclic world outcome has no portable answer")),
    };
    Ok(CyclicWorldAnswer {
        world: result.world().raw(),
        family: result.family().as_str().to_owned(),
        variables: result.variables().iter().map(|id| id.raw()).collect(),
        actions: result
            .actions()
            .iter()
            .map(|action| CyclicActionAnswer {
                kind: action.kind().to_owned(),
                variable: action.variable().raw(),
                node: action.node(),
            })
            .collect(),
        outcome,
        certificate: certificate_answer(result.certificate()),
        solved_by: solved_by_answer(result.solved_by()),
        failed_mass: result.mass().failed(),
        withheld_mass: result.mass().withheld(),
        physical_stability: format!("{:?}", result.physical_stability()).to_lowercase(),
    })
}
