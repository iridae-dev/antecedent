//! Repair families for sensitivity-analysis findings.

use super::{
    Arc, BTreeSet, BackdoorRepairFamily, Dag, DurableStudyCandidate, EvidenceObligation,
    EvidenceObligationSpec, EvidenceRegime, ExecutionContext, FailedContract, Failure,
    FamilyVerdict, NodeRef, ObligationProvenance, RepairCancelled, RepairError, RepairFamily,
    VariableId, dense_of, graph_variables, invalid,
};

/// The sensitivity analysis a finding came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum SensitivityMethod {
    /// E-value: the risk-ratio strength of a confounder that explains the effect away.
    EValue,
    /// Omitted-variable bias: the partial R-squared of a confounder that does.
    OmittedVariableBias,
    /// A tipping search over the declared confounding parameter.
    Tipping,
}

impl SensitivityMethod {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EValue => "e_value",
            Self::OmittedVariableBias => "omitted_variable_bias",
            Self::Tipping => "tipping",
        }
    }

    /// The method named `name`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::EValue, Self::OmittedVariableBias, Self::Tipping]
            .into_iter()
            .find(|method| method.as_str() == name)
    }
}

/// What a sensitivity analysis found about unmeasured confounding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SensitivityFinding {
    /// The analysis.
    pub method: SensitivityMethod,
    /// The confounder strength at which the conclusion is overturned: the
    /// e-value, the robustness value or the tipping parameter.
    pub strength: f64,
    /// Whether the conclusion survives the confounding the analyst deems plausible.
    /// A robust conclusion owes no evidence.
    pub robust: bool,
}

/// A sensitivity finding as a repair family.
///
/// The conclusion is overturned by an unmeasured confounder `C` of the reported
/// strength. The family models `C` as one more common cause of the treatment and
/// the outcome and is repaired by a study that measures `C` jointly with the
/// treatment, the outcome and whatever else the back-door criterion then needs.
#[derive(Clone, Debug)]
pub struct SensitivityRepairFamily {
    base: Dag,
    finding: SensitivityFinding,
    confounder: VariableId,
    inner: BackdoorRepairFamily,
    obligations: Vec<EvidenceObligation>,
}

impl SensitivityRepairFamily {
    /// Freeze a non-robust sensitivity finding over `dag` (latent causes
    /// included); `base_law` are the variables observed jointly.
    ///
    /// # Errors
    /// `identification_repair.invalid_request` for an unknown or equal
    /// treatment/outcome, an empty population or a non-positive strength;
    /// `identification_repair.not_a_failure` when the finding is robust.
    pub fn try_new(
        dag: Dag,
        treatment: VariableId,
        outcome: VariableId,
        population: impl Into<Arc<str>>,
        base_law: impl IntoIterator<Item = VariableId>,
        finding: SensitivityFinding,
    ) -> Result<Self, RepairError> {
        if !(finding.strength.is_finite() && finding.strength > 0.0) {
            return Err(invalid("a sensitivity strength is finite and positive"));
        }
        if finding.robust {
            return Err(RepairError::not_a_failure(
                "the conclusion survives the plausible unmeasured confounding",
            ));
        }
        let population = population.into();
        let base_law: BTreeSet<VariableId> = base_law.into_iter().collect();
        // Validates the treatment, outcome, population and base law.
        Failure::try_new(
            dag.clone(),
            treatment,
            outcome,
            Arc::clone(&population),
            base_law.clone(),
        )?;
        let confounder = VariableId::from_raw(
            graph_variables(&dag).iter().map(|v| v.raw()).max().map_or(0, |m| m.saturating_add(1)),
        );
        let mut augmented = dag.clone();
        let node =
            augmented.add_node(NodeRef::Static(confounder)).map_err(|e| invalid(e.to_string()))?;
        for end in [treatment, outcome] {
            let end = dense_of(&augmented, end).ok_or_else(|| invalid("unknown variable"))?;
            augmented.insert_directed(node, end).map_err(|e| invalid(e.to_string()))?;
        }
        let inner = BackdoorRepairFamily::try_new(
            augmented,
            treatment,
            outcome,
            Arc::clone(&population),
            base_law,
            &[],
        )?;
        let mut family = Self { base: dag, finding, confounder, inner, obligations: Vec::new() };
        family.obligations = family.restate()?;
        Ok(family)
    }

    /// The back-door obligations of the confounded graph, restated with the
    /// sensitivity finding as their provenance.
    fn restate(&self) -> Result<Vec<EvidenceObligation>, RepairError> {
        let source = self.contract_id();
        let step = format!(
            "sensitivity.{}:strength={}",
            self.finding.method.as_str(),
            self.finding.strength
        );
        self.inner
            .unresolved_obligations()
            .into_iter()
            .map(|o| {
                EvidenceObligation::try_new(EvidenceObligationSpec {
                    quantities: o.quantities,
                    kind: o.kind,
                    scope: o.scope,
                    variables: o.variables,
                    population: o.population,
                    regime: o.regime,
                    reason: Arc::from(format!(
                        "an unmeasured confounder of strength {} ({}) would overturn the \
                         conclusion; {}",
                        self.finding.strength,
                        self.finding.method.as_str(),
                        o.reason
                    )),
                    required_slots: o.required_slots,
                    min_additional_samples: o.min_additional_samples,
                    provenance: ObligationProvenance {
                        family: Arc::from("sensitivity"),
                        source: Arc::from(source.as_str()),
                        proof_step: Some(Arc::from(step.as_str())),
                    },
                })
                .map_err(|e| invalid(e.to_string()))
            })
            .collect()
    }

    /// The graph the finding was reported on (without the modelled confounder).
    #[must_use]
    pub fn dag(&self) -> &Dag {
        &self.base
    }

    /// The finding.
    #[must_use]
    pub const fn finding(&self) -> SensitivityFinding {
        self.finding
    }

    /// The variable standing for the unmeasured confounder.
    #[must_use]
    pub const fn confounder(&self) -> VariableId {
        self.confounder
    }

    fn relabel(&self, verdict: FamilyVerdict) -> FamilyVerdict {
        match verdict {
            FamilyVerdict::Sufficient(mut derivation) => {
                derivation.checker = "sensitivity.backdoor_adjustment";
                derivation.steps.push(format!("confounder:{}", self.confounder.raw()));
                FamilyVerdict::Sufficient(derivation)
            }
            other => other,
        }
    }

    /// Re-run the back-door identifier on already-proposed `regimes`.
    ///
    /// # Errors
    /// Cancellation.
    pub fn check_regimes(
        &self,
        regimes: &[EvidenceRegime],
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        self.inner.check_regimes(regimes, ctx).map(|verdict| self.relabel(verdict))
    }
}

impl FailedContract for SensitivityRepairFamily {
    fn treatment(&self) -> VariableId {
        self.inner.treatment()
    }

    fn outcome(&self) -> VariableId {
        self.inner.outcome()
    }

    fn population(&self) -> &str {
        self.inner.population()
    }

    fn base_law(&self) -> Vec<VariableId> {
        self.inner.base_law()
    }
}

impl RepairFamily for SensitivityRepairFamily {
    fn family_id(&self) -> &'static str {
        "sensitivity"
    }

    fn contract_id(&self) -> String {
        format!(
            "sensitivity:{}:treatment={}:outcome={}:method={}",
            self.inner.population(),
            self.inner.treatment().raw(),
            self.inner.outcome().raw(),
            self.finding.method.as_str()
        )
    }

    fn unresolved_obligations(&self) -> Vec<EvidenceObligation> {
        self.obligations.clone()
    }

    fn hypothetical_regimes(
        &self,
        studies: &[&DurableStudyCandidate],
    ) -> Result<Vec<EvidenceRegime>, Vec<String>> {
        RepairFamily::hypothetical_regimes(&self.inner, studies)
    }

    fn check(
        &self,
        studies: &[&DurableStudyCandidate],
        ctx: &ExecutionContext,
    ) -> Result<FamilyVerdict, RepairCancelled> {
        self.inner.check(studies, ctx).map(|verdict| self.relabel(verdict))
    }
}
