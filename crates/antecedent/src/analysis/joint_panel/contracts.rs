//! Scientific target and actual design attempts for the panel realization.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointPanelColumns, JointPanelError, JointPanelRequest, JointPanelWorld, engine};
use antecedent_core::{
    AssumptionSet, CanonicalWriter, CausalSchema, Conditioning, GraphFamily, HistoryWindow,
    InferenceSelection, InitialLaw, Intervention, InterventionContrast, InterventionSequence,
    InterventionWorld, ModelAssumptions, ObservationUnit, OutcomeQuantity, QuantityUnits,
    RandomizationDesign, RandomizedEffectQuery, RandomizedEstimand, SequencedIntervention,
    TargetForm, TargetPopulation, TemporalPolicy, TimeFrame, Value, WorldAction, WorldBinding,
    WorldCollection,
};
use antecedent_core::{ScientificTarget, ScientificWorldId, SemanticDigest, VariableId};
use antecedent_estimate::joint_panel::PanelProgram;
use antecedent_identify::CompiledIdentificationProduct;
use antecedent_identify::compiled_product::IdentificationObligation;
use antecedent_identify::{
    CompiledProductParts, IdentificationClaimClass, IdentificationOutcome, UnresolvedObligation,
};

/// Requested assembly of the same jointly fitted worlds.
#[derive(Clone, Debug, PartialEq)]
pub enum JointPanelFunctional {
    /// Keep each world before selecting a downstream utility.
    Worlds,
    /// Prespecified finite weighted contrast, including higher-order interaction contrasts.
    Linear {
        /// Every participating world and its finite nonzero coefficient.
        terms: Vec<(ScientificWorldId, f64)>,
        /// Explicit reference world.
        reference: ScientificWorldId,
    },
    /// Signed world difference with an explicit baseline.
    Difference {
        /// Positive world.
        plus: ScientificWorldId,
        /// Negative world.
        minus: ScientificWorldId,
        /// Explicit baseline.
        reference: ScientificWorldId,
    },
    /// Joint response minus both marginal changes plus the baseline.
    Interaction {
        /// Joint action.
        joint: ScientificWorldId,
        /// Only A changes.
        a_at_reference: ScientificWorldId,
        /// Only B changes.
        b_at_reference: ScientificWorldId,
        /// Baseline.
        reference: ScientificWorldId,
    },
}
/// Actual outcome of a candidate causal design, separate from numerical feasibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointPanelDesignOutcome {
    /// Required design premises were not supplied.
    Unsupported,
    /// Supplied design contradicts the source/program bindings.
    Rejected,
    /// Known whole-program randomization identifies the requested means.
    Identified,
}
/// Exact joint-trajectory support; marginal dose ranges cannot establish this.
#[derive(Clone, Debug, PartialEq)]
pub struct JointPanelWorldSupport {
    /// Target world key.
    pub world: ScientificWorldId,
    /// Observed independent groups with this complete trajectory and matching context.
    pub matched_groups: Vec<String>,
    /// An accepted causal design, not mere matching, established this world.
    pub identified: bool,
    /// Support meaning and missing premise.
    pub reason: String,
}
/// Versioned proof representation shared by legacy and design-native compilers.
/// Historical randomized panel artifacts retain their existing v1 numerical decoder.
#[derive(Clone, Debug, PartialEq)]
pub enum JointPanelDesignProof {
    /// First typed design-proof representation.
    V1(JointPanelDesignTheorem),
}
/// Actual theorem and target retained by a checked method owner.
#[derive(Clone, Debug, PartialEq)]
pub enum JointPanelDesignTheorem {
    /// Existing complete-program randomization proof, preserved without semantic changes.
    RandomizedWholeProgram(RandomizedEffectQuery),
    /// Native joint-history, structural-feature, cohort or donor target proof, retaining
    /// the distinct target rather than silently converting it into a population effect.
    Native {
        /// Design-specific identifying functional, assumptions and original target keys.
        identification: super::designs::NativeJointIdentification,
        /// Independently retained source/design identity, not itself a causal assertion.
        source_dependency: SemanticDigest,
    },
}
impl JointPanelDesignProof {
    pub(super) fn randomized(query: RandomizedEffectQuery) -> Self {
        Self::V1(JointPanelDesignTheorem::RandomizedWholeProgram(query))
    }
}
/// Retained ordered design attempt with its exact source contract when accepted.
#[derive(Clone, Debug, PartialEq)]
pub struct JointPanelDesignAttempt {
    /// Stable design family.
    pub id: String,
    /// Actual result of checking that design.
    pub outcome: JointPanelDesignOutcome,
    /// Concrete incompatibility or accepted premises.
    pub detail: String,
    /// Next scientifically meaningful attempt after failure.
    pub next_attempt: Option<String>,
    /// Full scientific question this attempt concerns.
    pub target_identity: SemanticDigest,
    /// Exact per-world joint design support.
    pub worlds: Vec<JointPanelWorldSupport>,
    /// Validated existing randomization contract retained as the proof premise.
    pub proof: Option<JointPanelDesignProof>,
}
pub(super) fn history_identity(program: &PanelProgram) -> SemanticDigest {
    let mut writer = CanonicalWriter::new("antecedent.compiler.panel.initial_history.v1");
    writer.word(program.scenarios.len() as u64);
    for scenario in &program.scenarios {
        writer.text(&scenario.group).word(scenario.history.len() as u64);
        for history in &scenario.history {
            writer.text(&history.id);
            for value in history.values {
                writer.word(value.to_bits());
            }
        }
    }
    SemanticDigest::from_bytes(writer.finish())
}
pub(super) fn context_identity(program: &PanelProgram) -> SemanticDigest {
    let mut writer = CanonicalWriter::new("antecedent.compiler.panel.context.v2");
    writer.bytes(history_identity(program).as_bytes()).word(program.scenarios.len() as u64);
    for scenario in &program.scenarios {
        writer.text(&scenario.group).word(scenario.confounder.len() as u64);
        for value in &scenario.confounder {
            writer.word(value.to_bits());
        }
        writer.word(scenario.season.len() as u64);
        for &value in &scenario.season {
            writer.word(value as u64);
        }
    }
    SemanticDigest::from_bytes(writer.finish())
}
fn program_coordinates<'a>(
    columns: &JointPanelColumns,
    program: &'a PanelProgram,
) -> Result<Vec<(VariableId, &'a [f64])>, JointPanelError> {
    let mut coordinates = Vec::with_capacity(columns.channels.len());
    if program.channels.len() != columns.channels.len() {
        return Err(JointPanelError::compile(
            "program channels do not cover every declared source channel",
        ));
    }
    for &variable in &columns.channels {
        let id = variable.raw().to_string();
        let mut matches = program.channels.iter().filter(|channel| channel.id == id);
        let channel = matches.next().ok_or_else(|| {
            JointPanelError::compile("program omits an explicitly declared channel")
        })?;
        if matches.next().is_some() {
            return Err(JointPanelError::compile("duplicate program channel"));
        }
        coordinates.push((variable, channel.values.as_slice()));
    }
    Ok(coordinates)
}
fn world(
    schema: &CausalSchema,
    columns: &JointPanelColumns,
    lag: usize,
    world: &JointPanelWorld,
    temporal: bool,
) -> Result<InterventionWorld, JointPanelError> {
    let p = &world.program;
    let coordinates = program_coordinates(columns, p)?;
    let horizon = coordinates.first().map_or(0, |(_, values)| values.len());
    if horizon == 0 || coordinates.iter().any(|(_, values)| values.len() != horizon) {
        return Err(JointPanelError::compile("world has empty or unequal channel schedules"));
    }
    let binding = WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new("panel_group").map_err(JointPanelError::compile)?,
        OutcomeQuantity::mean(
            columns.outcome,
            QuantityUnits::new(
                schema
                    .get(columns.outcome)
                    .map_err(JointPanelError::compile)?
                    .unit
                    .as_deref()
                    .unwrap_or("unspecified"),
            )
            .map_err(JointPanelError::compile)?,
        )
        .map_err(JointPanelError::compile)?,
        Conditioning::unconditional(),
    )
    .map_err(JointPanelError::compile)?;
    let h = u32::try_from(horizon).map_err(JointPanelError::compile)?;
    let (action, time) = if lag == 0 && h == 1 && !temporal {
        (
            WorldAction::at_decision(
                coordinates
                    .iter()
                    .map(|(variable, values)| Intervention::set(*variable, Value::f64(values[0]))),
            )
            .map_err(JointPanelError::compile)?,
            TimeFrame::static_decision(),
        )
    } else {
        let mut steps = Vec::with_capacity(horizon * coordinates.len());
        for offset in 0..horizon {
            let at = i32::try_from(offset).map_err(JointPanelError::compile)?;
            for (variable, values) in &coordinates {
                steps.push(SequencedIntervention::new(
                    Intervention::set(*variable, Value::f64(values[offset])),
                    TemporalPolicy::pulse(at),
                ));
            }
        }
        (
            WorldAction::schedule(
                Intervention::Sequence(InterventionSequence::new(steps)),
                TemporalPolicy::sustained(
                    0,
                    i32::try_from(h - 1).map_err(JointPanelError::compile)?,
                ),
            )
            .map_err(JointPanelError::compile)?,
            TimeFrame::temporal(
                h,
                HistoryWindow::max_lag(u32::try_from(lag).map_err(JointPanelError::compile)?),
                InitialLaw::named(history_identity(p).to_hex())
                    .map_err(JointPanelError::compile)?,
            )
            .map_err(JointPanelError::compile)?,
        )
    };
    InterventionWorld::new(world.world, action, world.reference, binding, time)
        .map_err(JointPanelError::compile)
}
/// Build a complete target from the exact supplied schedules and future context law.
/// No fitting, causal identification or history extrapolation occurs here.
///
/// # Errors
/// Refuses inconsistent worlds, units, schedules or contrast coordinates.
pub fn joint_panel_target(
    schema: &CausalSchema,
    columns: &JointPanelColumns,
    lag_order: usize,
    worlds: &[JointPanelWorld],
    functional: JointPanelFunctional,
    inference: InferenceSelection,
) -> Result<ScientificTarget, JointPanelError> {
    let collection = WorldCollection::new(
        worlds
            .iter()
            .map(|w| world(schema, columns, lag_order, w, false))
            .collect::<Result<Vec<_>, _>>()?,
    )
    .map_err(JointPanelError::compile)?;
    let form = match functional {
        JointPanelFunctional::Worlds => TargetForm::Worlds(collection),
        JointPanelFunctional::Linear { terms, reference } => TargetForm::Contrast(
            InterventionContrast::linear(collection, terms, reference)
                .map_err(JointPanelError::compile)?,
        ),
        JointPanelFunctional::Difference { plus, minus, reference } => TargetForm::Contrast(
            InterventionContrast::difference(collection, plus, minus, reference)
                .map_err(JointPanelError::compile)?,
        ),
        JointPanelFunctional::Interaction { joint, a_at_reference, b_at_reference, reference } => {
            TargetForm::Contrast(
                InterventionContrast::interaction(
                    collection,
                    joint,
                    a_at_reference,
                    b_at_reference,
                    reference,
                )
                .map_err(JointPanelError::compile)?,
            )
        }
    };
    let mut assumptions = AssumptionSet::default();
    let mut ordered_worlds = worlds.iter().collect::<Vec<_>>();
    ordered_worlds.sort_by_key(|world| world.world);
    for w in ordered_worlds {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: format!("panel.context.world.{}", w.world.raw()).into(),
                description: context_identity(&w.program).to_hex().into(),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Global,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
    Ok(ScientificTarget::new(
        form,
        ModelAssumptions::new(GraphFamily::Unspecified, assumptions),
        inference,
    ))
}
pub(super) fn target_worlds(
    target: &ScientificTarget,
) -> Result<&[InterventionWorld], JointPanelError> {
    match target.form() {
        TargetForm::World(w) => Ok(std::slice::from_ref(w)),
        TargetForm::Worlds(ws) => Ok(ws.worlds()),
        TargetForm::Contrast(c) => Ok(c.worlds().worlds()),
        TargetForm::DoseTime(s) => Ok(s.worlds().worlds()),
        _ => Err(JointPanelError::compile(
            "panel programs do not establish mediation or another theorem-scoped target",
        )),
    }
}
pub(super) fn validate_target(
    schema: &CausalSchema,
    request: &JointPanelRequest,
) -> Result<(), JointPanelError> {
    let actual = target_worlds(&request.target)?;
    let expected = WorldCollection::new(
        request
            .worlds
            .iter()
            .map(|w| {
                let declared = actual.iter().find(|world| world.id() == w.world);
                let expected = world(
                    schema,
                    &request.columns,
                    request.lag_order,
                    w,
                    declared.is_some_and(|world| !world.time().is_static()),
                )?;
                if let TargetForm::DoseTime(surface) = request.target.form() {
                    dose_time_world(&expected, declared, w, &request.columns, surface.dose())
                } else {
                    Ok(expected)
                }
            })
            .collect::<Result<Vec<_>, _>>()?,
    )
    .map_err(JointPanelError::compile)?;
    for w in &request.worlds {
        let id = format!("panel.context.world.{}", w.world.raw());
        let context = context_identity(&w.program).to_hex();
        if !request.target.assumptions().records().entries.iter().any(|r|matches!(&r.assumption,antecedent_core::Assumption::Custom{id:key,description} if key.as_ref()==id && description.as_ref()==context) && r.status!=antecedent_core::AssumptionStatus::Contradicted) {
            return Err(JointPanelError::compile("target does not retain the exact declared world context law"));
        }
    }

    if expected.worlds() != actual {
        return Err(JointPanelError::compile(
            "target world action, history, initial/context law, population, units or outcome differs from the retained program",
        ));
    }
    let bayes = matches!(request.options.inference, engine::PanelInference::Bayesian { .. });
    let expected_inference =
        if bayes { InferenceSelection::Bayesian } else { InferenceSelection::Frequentist };
    if request.target.inference() != expected_inference {
        return Err(JointPanelError::compile(
            "target inference selection differs from the retained fit procedure",
        ));
    }
    Ok(())
}
// A core scalar surface declares one dose policy, rather than an intervention
// on every channel. Only translate it when that policy fully determines the
// explicit numerical path. Unassigned periods/channels need a different law.
fn dose_time_world(
    expected: &InterventionWorld,
    actual: Option<&InterventionWorld>,
    program: &JointPanelWorld,
    columns: &JointPanelColumns,
    dose: VariableId,
) -> Result<InterventionWorld, JointPanelError> {
    if columns.channels.as_slice() != [dose] || program.program.channels.len() != 1 {
        return Err(JointPanelError::Unsupported(
            "scalar dose-time cells cannot silently fix additional channel paths".into(),
        ));
    }
    let actual = actual.ok_or_else(|| JointPanelError::compile("missing dose-time cell"))?;
    let Some((Intervention::Set { variable, value: Value::Float64(level) }, policy)) =
        actual.action().schedule_parts()
    else {
        return Err(JointPanelError::compile("dose-time cell must declare its scalar Set policy"));
    };
    if *variable != dose
        || program.program.channels[0].values.iter().any(|value| value.to_bits() != level.to_bits())
    {
        return Err(JointPanelError::compile(
            "dose-time program does not follow its declared scalar dose",
        ));
    }
    let no_past_assignment = match policy {
        TemporalPolicy::Pulse { at } => *at >= 0,
        TemporalPolicy::Sustained { from, .. } => *from >= 0,
        TemporalPolicy::Dynamic { active_at, .. } => active_at.iter().all(|step| *step >= 0),
        _ => false,
    };
    let covered = (0..program.horizon()).all(|offset| {
        let Ok(offset) = i32::try_from(offset) else {
            return false;
        };
        match policy {
            TemporalPolicy::Pulse { at } => offset == *at,
            TemporalPolicy::Sustained { from, until } => (*from..=*until).contains(&offset),
            TemporalPolicy::Dynamic { active_at, .. } => active_at.contains(&offset),
            _ => false,
        }
    });
    if !no_past_assignment || !covered {
        return Err(JointPanelError::Unsupported("dose-time policy leaves program periods unassigned or changes the declared prehistory; no implicit exposure continuation law is available".into()));
    }
    InterventionWorld::new(
        expected.id(),
        actual.action().clone(),
        expected.is_reference(),
        expected.binding().clone(),
        expected.time().clone(),
    )
    .map_err(JointPanelError::compile)
}
fn matching_history(
    data: &engine::PanelData,
    group: &engine::PanelGroup,
    scenario: &engine::PanelScenario,
) -> bool {
    scenario.history.len() == data.channels.len()
        && data.channels.iter().zip(&group.history).all(|(channel, values)| {
            scenario
                .history
                .iter()
                .filter(|history| {
                    history.id == channel.id
                        && history.values.map(f64::to_bits) == values.map(f64::to_bits)
                })
                .count()
                == 1
        })
}
fn group_matches(
    data: &engine::PanelData,
    group: &engine::PanelGroup,
    program: &PanelProgram,
) -> bool {
    let Some(scenario) = program.scenarios.iter().find(|s| s.group == group.id) else {
        return false;
    };
    program.channels.len() == data.channels.len()
        && data.channels.iter().zip(&group.channels).all(|(channel, values)| {
            program
                .channels
                .iter()
                .filter(|program_channel| {
                    program_channel.id == channel.id && program_channel.values == *values
                })
                .count()
                == 1
        })
        && scenario.confounder == group.confounder
        && scenario.season == group.season
        && (data.lag_order == 0 || matching_history(data, group, scenario))
}
fn experiment(
    data: &engine::PanelData,
    request: &JointPanelRequest,
) -> Result<RandomizedEffectQuery, String> {
    let query = request
        .assignment
        .as_ref()
        .ok_or("no known whole-program randomization contract was supplied")?;
    let context_declared=request.target.assumptions().records().entries.iter().any(|record| {
        matches!(&record.assumption,antecedent_core::Assumption::Custom{id,..} if id.as_ref()=="panel.context_preassignment_exogenous")
        && record.scope==antecedent_core::AssumptionScope::Identification
        && record.status!=antecedent_core::AssumptionStatus::Contradicted
    });
    if !context_declared {
        return Err("whole-program conditional-context identification requires the explicit preassignment/exogenous context premise; observed equality cannot establish it".into());
    }

    query.validate().map_err(|error| error.to_string())?;
    if query.outcome != request.columns.outcome
        || query.estimand != RandomizedEstimand::IntentionToTreat
        || query.fixed_cuped.is_some()
        || !query.ancova_covariates.is_empty()
        || query.received_treatment.is_some()
    {
        return Err(
            "experimental target is not the exact unadjusted program-assignment mean".into()
        );
    }
    let RandomizationDesign::MultiArm { assignment, probabilities, arms } = &query.design else {
        return Err("this certificate requires known independent whole-program multi-arm assignment; per-period or switching assignment is a different design".into());
    };
    if arms.len() != request.worlds.len()
        || arms.iter().zip(&request.worlds).any(|(arm, world)| arm.as_ref() != world.program.id)
    {
        return Err("randomized arms are not the exact ordered compiled programs".into());
    }
    if assignment.len() != data.groups.len()
        || query.assignment_units.len() != data.groups.len()
        || query.outcome_units.len() != data.groups.len()
    {
        return Err("randomization units do not cover the exact independent source groups".into());
    }
    let mut counts = vec![0; arms.len()];
    for (index, group) in data.groups.iter().enumerate() {
        if query.assignment_units[index].as_ref() != group.id
            || query.outcome_units[index].as_ref() != group.id
        {
            return Err("assignment and outcome units differ from the source partition".into());
        }
        if probabilities[index].iter().any(|&p| p <= 0.0) {
            return Err(
                "every whole-program arm requires positive known assignment probability".into()
            );
        }
        let assigned = assignment[index];
        let Some(world) = request.worlds.get(assigned) else {
            return Err("unknown assigned program".into());
        };
        if !group_matches(data, group, &world.program) {
            return Err(format!(
                "group {} did not receive its assigned complete program/history/context",
                group.id
            ));
        }
        counts[assigned] += 1;
        // Future covariates must be pretreatment/exogenous to assignment for this exact reference-population law.
        for world in &request.worlds {
            let Some(s) = world.program.scenarios.iter().find(|s| s.group == group.id) else {
                return Err("world lacks a source group scenario".into());
            };
            if s.confounder != group.confounder
                || s.season != group.season
                || (request.lag_order == 2 && !matching_history(data, group, s))
            {
                return Err(
                    "experimental world changes the observed preassignment/context law".into()
                );
            }
        }
    }
    if counts.contains(&0) {
        return Err("an experimental program arm has no observed independent group".into());
    }
    Ok(query.clone())
}
pub(super) fn compile_support(
    data: &engine::PanelData,
    request: &JointPanelRequest,
) -> Result<(CompiledIdentificationProduct, Vec<JointPanelDesignAttempt>), JointPanelError> {
    let evidence = experiment(data, request);
    let identified = evidence.is_ok();
    let support=request.worlds.iter().map(|w|JointPanelWorldSupport{world:w.world,matched_groups:data.groups.iter().filter(|g|group_matches(data,g,&w.program)).map(|g|g.id.clone()).collect(),identified,reason:if identified {"known randomized whole-program support"} else {"exact trajectory matching is numerical design support, not an unconfoundedness proof"}.into()}).collect::<Vec<_>>();
    let admission = crate::analysis::prepared::joint_target_admission(&request.target);
    let attempts=vec![
        JointPanelDesignAttempt{id:"b04.gaussian_panel".into(),outcome:JointPanelDesignOutcome::Unsupported,detail:match &admission {Err(error)=>error.to_string(),Ok(())=>"static target admitted by the production joint Gaussian owner; source lacks its required retained exact joint-adjustment proof, and nonlinear mechanism/AR inference is a different numerical realization".into()},next_attempt:Some("test exact joint intervention design".into()),target_identity:request.target.identity(),worlds:support.clone(),proof:None},
        JointPanelDesignAttempt{id:"joint_adjustment.observational".into(),outcome:JointPanelDesignOutcome::Unsupported,detail:"no matching temporal joint-adjustment or independent mechanism certificate is supplied; a static ATE or marginal dose range cannot identify this complete panel target".into(),next_attempt:Some("known randomized assignment of complete multi-channel programs over independent groups".into()),target_identity:request.target.identity(),worlds:support.clone(),proof:None},
        JointPanelDesignAttempt{id:"randomized.multi_arm_itt.whole_program".into(),outcome:if identified {JointPanelDesignOutcome::Identified} else if request.assignment.is_some(){JointPanelDesignOutcome::Rejected}else{JointPanelDesignOutcome::Unsupported},detail:match &evidence {Ok(_)=>"existing multi-arm ITT theorem applied to exact complete programs, independent group units and fixed observed context; nonlinear fitting and uncertainty remain model-conditional and unmeasured".into(),Err(e)=>e.clone()},next_attempt:if identified {None}else{Some("collect known positive-probability whole-program randomized arm assignments with exact prehistory and pretreatment context".into())},target_identity:request.target.identity(),worlds:support,proof:evidence.as_ref().ok().cloned().map(JointPanelDesignProof::randomized)},
    ];
    let outcome = if identified {
        IdentificationOutcome::Identified
    } else {
        IdentificationOutcome::Unsupported
    };
    let class = if identified {
        IdentificationClaimClass::IdentifiedClaim
    } else {
        IdentificationClaimClass::ModelConditional
    };
    let unresolved = if identified {
        vec![]
    } else {
        vec![UnresolvedObligation::new(
            "joint_program_design",
            evidence.as_ref().err().cloned().unwrap_or_default(),
        )]
    };
    let mut parts = CompiledProductParts::new(outcome, class);
    parts.world_obligations = request
        .worlds
        .iter()
        .map(|w| {
            IdentificationObligation::for_world(w.world, outcome, class, unresolved.clone(), None)
        })
        .collect();
    parts.unresolved_obligations.clone_from(&unresolved);
    if matches!(request.target.form(), TargetForm::Contrast(_) | TargetForm::DoseTime(_)) {
        parts.assembled =
            Some(IdentificationObligation::assembled(outcome, class, unresolved, None));
    }
    let product = match evidence {
        Ok(query) => {
            let (mut result, _) = crate::analysis::execute::randomized_identification(&query);
            result
                .required_assumptions
                .extend_unique(&request.target.assumptions().records().entries);
            CompiledIdentificationProduct::retain_for_target(&result, parts, &request.target)
        }
        Err(_) => CompiledIdentificationProduct::new(
            vec![],
            antecedent_expr::CausalExprArena::new(),
            antecedent_identify::DerivationTrace::default(),
            request.target.assumptions().records().clone(),
            vec![],
            parts,
        )
        .and_then(|p| p.bind_target(&request.target)),
    }
    .map_err(JointPanelError::compile)?;
    Ok((product, attempts))
}

/// Explicitly declare the premise required to hold each group's future context fixed
/// across randomized whole-program worlds. This is a user assumption, not an empirical test:
/// the confounder/season trajectories are measured before assignment or generated exogenously,
/// are unaffected by any arm, and the same observed population remains the target.
#[must_use]
pub fn declare_joint_panel_exogenous_context(target: &ScientificTarget) -> ScientificTarget {
    let mut records = target.assumptions().records().clone();
    records.push(antecedent_core::AssumptionRecord{
        assumption:antecedent_core::Assumption::Custom{id:"panel.context_preassignment_exogenous".into(),description:"Group-specific future confounder and season trajectories are preassignment/exogenous, unaffected by program assignment, and define the same observed reference population".into()},
        source:antecedent_core::AssumptionSource::UserDeclared,scope:antecedent_core::AssumptionScope::Identification,status:antecedent_core::AssumptionStatus::Declared,
    });
    ScientificTarget::new(
        target.form().clone(),
        ModelAssumptions::new(target.assumptions().graph(), records),
        target.inference(),
    )
}

pub(super) fn conditional_projection_product(
    request: &JointPanelRequest,
) -> Result<CompiledIdentificationProduct, JointPanelError> {
    conditional_target_product(&request.target)
}
pub(super) fn conditional_target_product(
    target: &ScientificTarget,
) -> Result<CompiledIdentificationProduct, JointPanelError> {
    let outcome = IdentificationOutcome::Unsupported;
    let class = IdentificationClaimClass::ModelConditional;
    let unresolved = vec![UnresolvedObligation::new(
        "projection_target_scope",
        "the retained final-horizon design does not independently authorize intermediate or window response coordinates",
    )];
    let mut parts = CompiledProductParts::new(outcome, class);
    parts.world_obligations = target_worlds(target)?
        .iter()
        .map(|world| {
            IdentificationObligation::for_world(
                world.id(),
                outcome,
                class,
                unresolved.clone(),
                None,
            )
        })
        .collect();
    parts.unresolved_obligations.clone_from(&unresolved);
    if matches!(target.form(), TargetForm::Contrast(_) | TargetForm::DoseTime(_)) {
        parts.assembled =
            Some(IdentificationObligation::assembled(outcome, class, unresolved, None));
    }
    CompiledIdentificationProduct::new(
        vec![],
        antecedent_expr::CausalExprArena::new(),
        antecedent_identify::DerivationTrace::default(),
        target.assumptions().records().clone(),
        vec![],
        parts,
    )
    .and_then(|product| product.bind_target(target))
    .map_err(JointPanelError::compile)
}
