//! Repair a hedge with treatment or verified-surrogate experiments.

use super::{
    Admg, Arc, AverageEffectQuery, BTreeSet, Build, DenseNodeId, DistributionAvailability,
    DurableStudyCandidate, EvidenceObligation, EvidenceRegime, ExecutionContext, FailedContract,
    FamilyVerdict, HedgeCertificate, IdIdentifier, IdentificationStatus, IdentificationWorkspace,
    NodeRef, RepairCancelled, RepairDerivation, RepairError, RepairFamily, VariableId, id_list,
    invalid, is_blank, reaches_avoiding, regimes_of,
};

/// Experiments that answer a hedge.
///
/// A hedge is a proof that `P(y | do(x))` is not a functional of the observed
/// law. The family accepts only experiments: setting the treatment and measuring
/// the outcome observes the interventional law directly, and setting one
/// verified surrogate node (one that intercepts every directed path from
/// treatment to outcome while `P(w | do(x))` is identified from observation, so
/// `P(y | do(x)) = sum_w P(w | do(x)) P(y | do(w))`) identifies it. The
/// surrogate route also needs the observational law over every graph variable.
#[derive(Clone, Debug)]
pub struct HedgeRepairFamily {
    admg: Admg,
    treatment: VariableId,
    outcome: VariableId,
    population: Arc<str>,
    base_law: BTreeSet<VariableId>,
    certificate: Option<HedgeCertificate>,
    surrogates: Vec<VariableId>,
    obligations: Vec<EvidenceObligation>,
}

fn hedge_identifies(
    prepared: &antecedent_identify::PreparedAdmg,
    treatment: VariableId,
    outcome: VariableId,
) -> Result<antecedent_identify::IdentificationResult, RepairError> {
    IdIdentifier::new()
        .identify_ate(
            prepared,
            &AverageEffectQuery::binary_ate(treatment, outcome),
            &mut IdentificationWorkspace::default(),
        )
        .map_err(|e| RepairError::from_identification(&e))
}

impl HedgeRepairFamily {
    /// Freeze a failed contract in an ADMG: `base_law` are the variables
    /// observed jointly in `population`.
    ///
    /// # Errors
    /// `identification_repair.invalid_request` for an unknown or equal
    /// treatment/outcome, a non-static graph or an empty population;
    /// `identification_repair.not_a_failure` when the ID algorithm identifies the
    /// effect; `identification_repair.budget` when it stops.
    pub fn try_new(
        admg: Admg,
        treatment: VariableId,
        outcome: VariableId,
        population: impl Into<Arc<str>>,
        base_law: impl IntoIterator<Item = VariableId>,
    ) -> Result<Self, RepairError> {
        let population = population.into();
        let mut variables = Vec::new();
        for node in admg.nodes() {
            match node {
                NodeRef::Static(variable) => variables.push(*variable),
                _ => return Err(invalid("a failure graph holds only static variables")),
            }
        }
        let base_law: BTreeSet<VariableId> = base_law.into_iter().collect();
        if treatment == outcome
            || !variables.contains(&treatment)
            || !variables.contains(&outcome)
            || base_law.iter().any(|v| !variables.contains(v))
            || is_blank(&population)
        {
            return Err(invalid(
                "a hedge contract names a population and a treatment and outcome that differ and \
                 lie in the graph",
            ));
        }
        let prepared = IdIdentifier::new().prepare(&admg).map_err(|e| invalid(e.to_string()))?;
        let result = hedge_identifies(&prepared, treatment, outcome)?;
        if result.status == IdentificationStatus::NonparametricallyIdentified {
            return Err(RepairError::not_a_failure("the ID algorithm identifies the effect"));
        }
        let certificate =
            result.hedge.filter(|cert| cert.verify(&prepared, &[treatment], &[outcome]).is_ok());
        let mut family = Self {
            admg,
            treatment,
            outcome,
            population,
            base_law,
            certificate,
            surrogates: Vec::new(),
            obligations: Vec::new(),
        };
        family.surrogates = family.verified_surrogates(&prepared);
        family.obligations = family.derive_obligations()?;
        Ok(family)
    }

    /// Hedge nodes (other than treatment and outcome) whose experiment identifies
    /// the effect.
    fn verified_surrogates(&self, prepared: &antecedent_identify::PreparedAdmg) -> Vec<VariableId> {
        let Some(cert) = &self.certificate else { return Vec::new() };
        let mut nodes: BTreeSet<VariableId> =
            cert.f.iter().chain(cert.f_prime.iter()).copied().collect();
        nodes.remove(&self.treatment);
        nodes.remove(&self.outcome);
        nodes.into_iter().filter(|w| self.surrogate_resolves(prepared, *w)).collect()
    }

    fn surrogate_resolves(
        &self,
        prepared: &antecedent_identify::PreparedAdmg,
        w: VariableId,
    ) -> bool {
        let (Some(t), Some(y), Some(wd)) = (
            self.admg.dense_of_static(self.treatment),
            self.admg.dense_of_static(self.outcome),
            self.admg.dense_of_static(w),
        ) else {
            return false;
        };
        let children = |node: DenseNodeId| self.admg.children(node).to_vec();
        !reaches_avoiding(self.admg.node_count(), &children, t, y, &[wd])
            && hedge_identifies(prepared, self.treatment, w)
                .is_ok_and(|r| r.status == IdentificationStatus::NonparametricallyIdentified)
    }

    fn derive_obligations(&self) -> Result<Vec<EvidenceObligation>, RepairError> {
        let source = self.contract_id();
        let build = Build { family: "hedge", source: &source };
        let (why, step) = match &self.certificate {
            Some(cert) => (
                format!(
                    "observation cannot identify the effect: the graph holds a hedge (F=[{}], \
                     F'=[{}])",
                    id_list(&cert.f),
                    id_list(&cert.f_prime)
                ),
                format!("hedge:f=[{}];f_prime=[{}]", id_list(&cert.f), id_list(&cert.f_prime)),
            ),
            None => (
                "the ID algorithm does not identify the effect from observation".to_owned(),
                "id.not_identified".to_owned(),
            ),
        };
        let mut out = vec![build.experiment(
            self.treatment,
            self.outcome,
            &self.population,
            format!(
                "{why}; an experiment that sets treatment {} and measures outcome {} observes \
                 P(y | do(x)) directly",
                self.treatment.raw(),
                self.outcome.raw()
            ),
            format!("hedge:experiment:{}", self.treatment.raw()),
            step,
        )?];
        for w in &self.surrogates {
            out.push(build.experiment(
                *w,
                self.outcome,
                &self.population,
                format!(
                    "{why}; node {} lies on the hedge and intercepts every directed path from \
                     the treatment to the outcome, and P({} | do(treatment)) is identified from \
                     observation, so an experiment on {} with outcome {} identifies \
                     P(y | do(x)) = sum_w P(w | do(x)) P(y | do(w))",
                    w.raw(),
                    w.raw(),
                    w.raw(),
                    self.outcome.raw()
                ),
                format!("hedge:surrogate:{}", w.raw()),
                format!("hedge.surrogate:{}", w.raw()),
            )?);
        }
        Ok(out)
    }

    /// The failed contract's graph.
    #[must_use]
    pub fn admg(&self) -> &Admg {
        &self.admg
    }

    /// The hedge witness the ID algorithm produced and the verifier confirmed.
    #[must_use]
    pub fn certificate(&self) -> Option<&HedgeCertificate> {
        self.certificate.as_ref()
    }

    /// The hedge nodes whose experiment identifies the effect.
    #[must_use]
    pub fn surrogates(&self) -> &[VariableId] {
        &self.surrogates
    }

    /// The verdict over the single-target experiments on offer, as
    /// `(label, target)` pairs.
    fn decide(&self, experiments: &[(String, VariableId)]) -> FamilyVerdict {
        let mut reasons = Vec::new();
        for (label, target) in experiments {
            if *target == self.treatment {
                return FamilyVerdict::Sufficient(RepairDerivation {
                    checker: "hedge.experiment",
                    steps: vec![format!("experiment:{label}"), "direct:P(y | do(x))".to_owned()],
                    verified: true,
                });
            }
            if self.surrogates.contains(target) {
                let covered =
                    graph_variables_of(&self.admg).iter().all(|v| self.base_law.contains(v));
                if covered {
                    return FamilyVerdict::Sufficient(RepairDerivation {
                        checker: "hedge.experiment",
                        steps: vec![
                            format!("experiment:{label}"),
                            format!("surrogate:{}", target.raw()),
                            "P(y | do(x)) = sum_w P(w | do(x)) P(y | do(w))".to_owned(),
                        ],
                        verified: true,
                    });
                }
                reasons.push(format!(
                    "experiment {label} on surrogate {} needs the observational joint law over \
                     every graph variable to identify P(w | do(x))",
                    target.raw()
                ));
            } else {
                reasons.push(format!(
                    "experiment {label} on {} neither sets the treatment nor is a verified \
                     surrogate on the hedge",
                    target.raw()
                ));
            }
        }
        if experiments.is_empty() {
            reasons.push(
                "no experiment in the contract's population measures the outcome while setting \
                 one variable"
                    .to_owned(),
            );
        }
        FamilyVerdict::Insufficient { reasons }
    }

    /// Re-run the verdict on already-proposed `regimes`.
    ///
    /// # Errors
    /// Cancellation.
    pub fn check_regimes(
        &self,
        regimes: &[EvidenceRegime],
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        if ctx.cancellation.is_cancelled() {
            return Err(RepairCancelled);
        }
        let experiments: Vec<(String, VariableId)> = regimes
            .iter()
            .filter(|regime| {
                regime.population == self.population
                    && regime.interventions.len() == 1
                    && regime.conditioned_on.is_empty()
                    && regime.measured.contains(&self.outcome)
                    && outcome_available(&regime.distribution, self.outcome)
            })
            .map(|regime| (format!("regime:{}", regime.id.raw()), regime.interventions[0]))
            .collect();
        Ok(self.decide(&experiments))
    }
}

fn graph_variables_of(admg: &Admg) -> Vec<VariableId> {
    admg.nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Static(variable) => Some(*variable),
            _ => None,
        })
        .collect()
}

/// Whether the outcome's marginal is part of the law (a joint law, or a
/// separate marginal that names it).
fn outcome_available(distribution: &DistributionAvailability, outcome: VariableId) -> bool {
    match distribution {
        DistributionAvailability::Joint => true,
        DistributionAvailability::SeparateMarginals { variables } => variables.contains(&outcome),
    }
}

impl FailedContract for HedgeRepairFamily {
    fn treatment(&self) -> VariableId {
        self.treatment
    }

    fn outcome(&self) -> VariableId {
        self.outcome
    }

    fn population(&self) -> &str {
        &self.population
    }

    fn base_law(&self) -> Vec<VariableId> {
        self.base_law.iter().copied().collect()
    }
}

impl RepairFamily for HedgeRepairFamily {
    fn family_id(&self) -> &'static str {
        "hedge"
    }

    fn contract_id(&self) -> String {
        format!(
            "hedge:{}:treatment={}:outcome={}",
            self.population,
            self.treatment.raw(),
            self.outcome.raw()
        )
    }

    fn unresolved_obligations(&self) -> Vec<EvidenceObligation> {
        self.obligations.clone()
    }

    fn hypothetical_regimes(
        &self,
        studies: &[&DurableStudyCandidate],
    ) -> Result<Vec<EvidenceRegime>, Vec<String>> {
        regimes_of(studies)
    }

    fn check(
        &self,
        studies: &[&DurableStudyCandidate],
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        if ctx.cancellation.is_cancelled() {
            return Err(RepairCancelled);
        }
        let mut experiments = Vec::new();
        let mut malformed = Vec::new();
        for study in studies {
            if let Err(error) = study.validate() {
                malformed.push(error.to_string());
                continue;
            }
            for (k, evidence) in study.expected_evidence.iter().enumerate() {
                if evidence.population == self.population
                    && evidence.interventions.len() == 1
                    && evidence.conditioned_on.is_empty()
                    && evidence.measured.contains(&self.outcome)
                    && outcome_available(&evidence.distribution, self.outcome)
                {
                    experiments
                        .push((format!("{}#{k}", study.semantic_id()), evidence.interventions[0]));
                }
            }
        }
        if !malformed.is_empty() {
            return Ok(FamilyVerdict::Invalid { reasons: malformed });
        }
        Ok(self.decide(&experiments))
    }
}
