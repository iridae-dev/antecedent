use super::{
    CoherenceFinding, CoherenceKind, CompositionError, ContextCoordinate, DETAIL_CYCLE,
    DETAIL_DEPENDENCE_INVALID, DETAIL_INVARIANCE_CONTRADICTED, DETAIL_INVARIANCE_LEVEL,
    DETAIL_PARENT_WITHOUT_MECHANISM, DETAIL_PARENTS_CHANGED, DETAIL_REGIME_NOT_DECLARED,
    DETAIL_SELECTION_TARGET_UNKNOWN, DETAIL_SHIFTED_WITHOUT_TARGET_EVIDENCE, FittedMechanism,
    InvarianceDeclaration, MeanSpec, MechanismOrigin, MechanismRole, ParameterDependence,
    TAG_IDENTITY, TYPED_TRANSPORT_VERSION, TransportedTypedModel, TypedDisposition,
    TypedTransportRequest, invariance_check, selection_targets, valid_covariance,
};
use antecedent_core::identity::CanonicalWriter;
use std::collections::{BTreeMap, BTreeSet};

fn finding(
    kind: CoherenceKind,
    detail: &'static str,
    subject: &str,
    remedy: &'static str,
) -> CoherenceFinding {
    CoherenceFinding::new(kind, detail, subject.to_owned(), remedy)
}

/// Carry invariant typed mechanisms from the source and use target refits for
/// shifted ones, or refuse with every finding.
///
/// # Errors
/// [`CompositionError::Incoherent`] with every defect: a diagram for other
/// populations, a shifted mechanism with no target fit, a declared invariance
/// the target fit contradicts, fits that do not share a specification or design,
/// a carried mechanism that is not fitted or not from the diagram's source
/// population, a regime change without a declaration, a parent without a
/// mechanism, a cycle, and a malformed dependence declaration.
#[allow(
    clippy::too_many_lines,
    reason = "the scientific calculation and its checks form one bounded procedure"
)]
pub fn transport_typed(
    request: &TypedTransportRequest,
) -> Result<TransportedTypedModel, CompositionError> {
    let shifted: BTreeSet<String> = selection_targets(&request.diagram).into_iter().collect();
    let mut findings: Vec<CoherenceFinding> = Vec::new();

    if !(request.invariance_level > 0.0 && request.invariance_level < 0.5) {
        findings.push(finding(
            CoherenceKind::Shape,
            DETAIL_INVARIANCE_LEVEL,
            "invariance_level",
            "test declared invariances at a level strictly between 0 and 0.5",
        ));
    }
    if request.diagram.target_population != request.target.population {
        findings.push(
            finding(
                CoherenceKind::Population,
                DETAIL_DIAGRAM_TARGET_MISMATCH_LOCAL,
                "selection_diagram",
                "declare the selection diagram between the source population and this target population",
            )
            .comparing(
                request.target.population.clone(),
                request.diagram.target_population.clone(),
            ),
        );
    }

    let mut native: BTreeMap<&str, &FittedMechanism> = BTreeMap::new();
    for m in &request.target_mechanisms {
        if m.context.population != request.target.population
            || m.context.regime != request.target.regime
        {
            findings.push(
                finding(
                    CoherenceKind::Population,
                    DETAIL_TARGET_NOT_NATIVE_LOCAL,
                    m.variable(),
                    "supply as a target mechanism only a mechanism fitted in the target population and regime",
                )
                .comparing(
                    format!("{}/{}", request.target.population, request.target.regime),
                    format!("{}/{}", m.context.population, m.context.regime),
                ),
            );
        }
        if m.provenance.origin != MechanismOrigin::Fitted {
            findings.push(finding(
                CoherenceKind::Provenance,
                DETAIL_ORIGIN_NOT_FITTED_LOCAL,
                m.variable(),
                "only a mechanism estimated from retained evidence is used",
            ));
        }
        if native.insert(m.variable(), m).is_some() {
            findings.push(finding(
                CoherenceKind::VariableIdentity,
                DETAIL_DUPLICATE_LOCAL,
                m.variable(),
                "give each variable exactly one target mechanism",
            ));
        }
    }
    let mut carried: BTreeMap<&str, &FittedMechanism> = BTreeMap::new();
    for m in &request.source_mechanisms {
        if carried.insert(m.variable(), m).is_some() {
            findings.push(finding(
                CoherenceKind::VariableIdentity,
                DETAIL_DUPLICATE_LOCAL,
                m.variable(),
                "give each variable exactly one source mechanism",
            ));
        }
    }

    let mut chosen: BTreeMap<String, FittedMechanism> = BTreeMap::new();
    let mut dispositions: BTreeMap<String, TypedDisposition> = BTreeMap::new();
    let variables: BTreeSet<&str> = native
        .keys()
        .chain(carried.keys())
        .copied()
        .chain(shifted.iter().map(String::as_str))
        .collect();
    for variable in variables {
        let is_shifted = shifted.contains(variable);
        let source = carried.get(variable).copied();
        let target = native.get(variable).copied();
        let (mechanism, role, consistency) = match (is_shifted, source, target) {
            (_, None, None) => {
                findings.push(finding(
                    CoherenceKind::VariableIdentity,
                    DETAIL_SELECTION_TARGET_UNKNOWN,
                    variable,
                    "a selection node points into a variable no supplied mechanism produces",
                ));
                continue;
            }
            (true, _, None) => {
                findings.push(finding(
                    CoherenceKind::Provenance,
                    DETAIL_SHIFTED_WITHOUT_TARGET_EVIDENCE,
                    variable,
                    "a selection node points into this variable, so its mechanism differs in the target; fit it on target data, it is never carried",
                ));
                continue;
            }
            (true, source, Some(target)) => {
                if let Some(source) = source {
                    if source.spec.parents != target.spec.parents {
                        findings.push(
                            finding(
                                CoherenceKind::CausalParents,
                                DETAIL_PARENTS_CHANGED,
                                variable,
                                "a shifted mechanism's target fit keeps the declared parent set",
                            )
                            .comparing(
                                source.spec.parents.join(","),
                                target.spec.parents.join(","),
                            ),
                        );
                    }
                }
                (target, MechanismRole::NativeInTarget, None)
            }
            (false, None, Some(target)) => (target, MechanismRole::NativeInTarget, None),
            (false, Some(source), target) => {
                check_carried_source(request, source, &mut findings);
                let mut check = None;
                if let Some(target) = target {
                    match invariance_check(source, target, request.invariance_level) {
                        Ok(c) => {
                            if c.rejected {
                                findings.push(
                                    finding(
                                        CoherenceKind::Population,
                                        DETAIL_INVARIANCE_CONTRADICTED,
                                        variable,
                                        "target data reject the declared invariance of this mechanism; declare a selection node into it and refit it in the target",
                                    )
                                    .comparing(
                                        format!("p >= {}", request.invariance_level),
                                        format!("p = {:.3e}", c.p_value),
                                    ),
                                );
                            }
                            check = Some(c);
                        }
                        Err(CompositionError::Incoherent { findings: f }) => findings.extend(f),
                        Err(_) => findings.push(finding(
                            CoherenceKind::UncertaintyAlignment,
                            DETAIL_DEPENDENCE_INVALID,
                            variable,
                            "the summed fit covariance of the consistency test is singular",
                        )),
                    }
                }
                (source, MechanismRole::Transported, check)
            }
        };
        chosen.insert(variable.to_owned(), mechanism.clone());
        dispositions.insert(
            variable.to_owned(),
            TypedDisposition {
                variable: variable.to_owned(),
                role,
                family: mechanism.spec.family,
                fitted_population: mechanism.context.population.clone(),
                artifact_id: mechanism.provenance.artifact_id.clone(),
                evidence_id: mechanism.provenance.evidence_id.clone(),
                covariance_source: match role {
                    MechanismRole::Transported => "source_fit",
                    MechanismRole::NativeInTarget => "target_fit",
                },
                consistency,
            },
        );
    }

    // Structure: every parent has a mechanism; acyclic; topological order.
    let mut order: Vec<String> = Vec::new();
    for (variable, m) in &chosen {
        for parent in &m.spec.parents {
            if !chosen.contains_key(parent) {
                findings.push(finding(
                    CoherenceKind::VariableIdentity,
                    DETAIL_PARENT_WITHOUT_MECHANISM,
                    &format!("{parent}->{variable}"),
                    "supply a source or target mechanism for every parent",
                ));
            }
        }
    }
    if findings.is_empty() {
        let mut placed: BTreeSet<&str> = BTreeSet::new();
        while placed.len() < chosen.len() {
            let ready: Vec<&String> = chosen
                .iter()
                .filter(|(v, m)| {
                    !placed.contains(v.as_str())
                        && m.spec.parents.iter().all(|p| placed.contains(p.as_str()))
                })
                .map(|(v, _)| v)
                .collect();
            if ready.is_empty() {
                findings.push(finding(
                    CoherenceKind::Cycle,
                    DETAIL_CYCLE,
                    "structure",
                    "a typed transported composition is acyclic",
                ));
                break;
            }
            for v in ready {
                placed.insert(v.as_str());
                order.push(v.clone());
            }
        }
    }
    validate_dependence(&request.dependence, &chosen, &mut findings);

    if !findings.is_empty() {
        findings
            .sort_by(|a, b| (a.kind, a.detail, &a.subject).cmp(&(b.kind, b.detail, &b.subject)));
        findings.dedup();
        return Err(CompositionError::Incoherent { findings });
    }

    let mechanisms: Vec<FittedMechanism> = order.iter().map(|v| chosen[v].clone()).collect();
    let dispositions: Vec<TypedDisposition> = dispositions.into_values().collect();
    let identity = identity_of(request, &mechanisms, &dispositions);
    Ok(TransportedTypedModel {
        mechanisms,
        dispositions,
        target: request.target.clone(),
        diagram: request.diagram.clone(),
        dependence: request.dependence.clone(),
        identity,
    })
}

const DETAIL_DIAGRAM_TARGET_MISMATCH_LOCAL: &str =
    crate::mechanism_transport::DETAIL_DIAGRAM_TARGET_MISMATCH;
const DETAIL_TARGET_NOT_NATIVE_LOCAL: &str = crate::mechanism_transport::DETAIL_TARGET_NOT_NATIVE;
const DETAIL_ORIGIN_NOT_FITTED_LOCAL: &str = crate::mechanism_transport::DETAIL_ORIGIN_NOT_FITTED;
const DETAIL_DUPLICATE_LOCAL: &str = crate::mechanism_transport::DETAIL_DUPLICATE_MECHANISM;
const DETAIL_SOURCE_MISMATCH_LOCAL: &str =
    crate::mechanism_transport::DETAIL_DIAGRAM_SOURCE_MISMATCH;

fn check_carried_source(
    request: &TypedTransportRequest,
    source: &FittedMechanism,
    out: &mut Vec<CoherenceFinding>,
) {
    if source.provenance.origin != MechanismOrigin::Fitted {
        out.push(finding(
            CoherenceKind::Provenance,
            DETAIL_ORIGIN_NOT_FITTED_LOCAL,
            source.variable(),
            "only a mechanism estimated from retained evidence is carried",
        ));
    }
    if source.context.population != request.diagram.source_population {
        out.push(
            finding(
                CoherenceKind::Population,
                DETAIL_SOURCE_MISMATCH_LOCAL,
                source.variable(),
                "declare the selection diagram between the population this mechanism was fitted in and the target",
            )
            .comparing(
                request.diagram.source_population.clone(),
                source.context.population.clone(),
            ),
        );
    }
    if source.context.regime != request.target.regime {
        let declared = request.invariances.iter().any(|d| {
            d.variable == source.variable()
                && d.coordinate == ContextCoordinate::Regime
                && d.source == source.context.regime
                && d.target == request.target.regime
        });
        if !declared {
            out.push(
                finding(
                    CoherenceKind::Regime,
                    DETAIL_REGIME_NOT_DECLARED,
                    source.variable(),
                    "declare the invariance of this mechanism across the two regimes explicitly",
                )
                .comparing(request.target.regime.clone(), source.context.regime.clone()),
            );
        }
    }
}

fn validate_dependence(
    dependence: &ParameterDependence,
    chosen: &BTreeMap<String, FittedMechanism>,
    out: &mut Vec<CoherenceFinding>,
) {
    let bad = |subject: &str, remedy: &'static str| {
        finding(CoherenceKind::PosteriorDependence, DETAIL_DEPENDENCE_INVALID, subject, remedy)
    };
    match dependence {
        ParameterDependence::Unknown => {}
        ParameterDependence::IndependentByDesign { justification } => {
            if justification.trim().is_empty() {
                out.push(bad("independence", "state why the fits are independent"));
            }
        }
        ParameterDependence::Joint { variables, covariance } => {
            let mut dim = 0;
            let mut seen = BTreeSet::new();
            for v in variables {
                match chosen.get(v) {
                    Some(m) if seen.insert(v.as_str()) => dim += m.theta.len(),
                    _ => out
                        .push(bad(v, "a joint covariance lists each model mechanism at most once")),
                }
            }
            if covariance.len() != dim * dim || !valid_covariance(covariance, dim) {
                out.push(bad("joint", "a joint covariance is symmetric positive semidefinite and aligned to the listed parameters"));
            }
        }
        ParameterDependence::CrossCovariance { blocks } => {
            let mut seen = BTreeSet::new();
            for b in blocks {
                let pair =
                    if b.first <= b.second { (&b.first, &b.second) } else { (&b.second, &b.first) };
                if !seen.insert(pair) {
                    out.push(bad(&b.first, "declare each unordered cross-covariance pair once"));
                }
                match (chosen.get(&b.first), chosen.get(&b.second)) {
                    (Some(a), Some(c)) if b.first != b.second => {
                        if b.values.len() != a.theta.len() * c.theta.len()
                            || b.values.iter().any(|v| !v.is_finite())
                        {
                            out.push(bad(
                                &b.first,
                                "a cross-covariance block is dim(first) x dim(second) and finite",
                            ));
                        }
                    }
                    _ => out.push(bad(
                        &b.first,
                        "a cross-covariance block names two distinct model mechanisms",
                    )),
                }
            }
        }
    }
}

fn write_mechanism(w: &mut CanonicalWriter, m: &FittedMechanism) {
    w.text(m.variable()).word(m.spec.family.code());
    w.texts(&m.spec.parents);
    match &m.spec.mean {
        MeanSpec::Polynomial { degree, interactions } => {
            w.word(1).word(u64::from(*degree)).word(u64::from(*interactions));
        }
        MeanSpec::Expression { expression, start } => {
            w.word(2).text(&format!("{expression:?}")).reals(start);
        }
    }
    w.word(m.scaling.len() as u64);
    for (c, s) in &m.scaling {
        w.real(*c).real(*s);
    }
    w.word(m.n_obs as u64).reals(&m.theta).reals(&m.covariance);
    w.text(&m.context.population).text(&m.context.regime);
    w.text(&m.provenance.artifact_id).text(&m.provenance.evidence_id);
}

fn identity_of(
    request: &TypedTransportRequest,
    mechanisms: &[FittedMechanism],
    dispositions: &[TypedDisposition],
) -> [u8; 32] {
    let mut w = CanonicalWriter::new(TAG_IDENTITY);
    w.word(u64::from(TYPED_TRANSPORT_VERSION));
    w.text(&request.target.population).text(&request.target.regime);
    w.real(request.invariance_level);
    let mut sorted: Vec<&FittedMechanism> = mechanisms.iter().collect();
    sorted.sort_by(|a, b| a.variable().cmp(b.variable()));
    w.word(sorted.len() as u64);
    for m in sorted {
        let role =
            dispositions.iter().find(|d| d.variable == m.variable()).map_or(0, |d| match d.role {
                MechanismRole::Transported => 1,
                MechanismRole::NativeInTarget => 2,
            });
        w.word(role);
        write_mechanism(&mut w, m);
    }
    w.word(request.invariances.len() as u64);
    let mut invariances: Vec<&InvarianceDeclaration> = request.invariances.iter().collect();
    invariances.sort_by(|a, b| {
        (&a.variable, a.coordinate, &a.source, &a.target).cmp(&(
            &b.variable,
            b.coordinate,
            &b.source,
            &b.target,
        ))
    });
    for i in invariances {
        w.text(&i.variable).word(i.coordinate as u64).text(&i.source).text(&i.target);
        w.text(&i.assumption_id);
    }
    w.text(&request.diagram.source_population).text(&request.diagram.target_population);
    let mut nodes: Vec<(&String, Vec<&String>)> = request
        .diagram
        .selection_nodes
        .iter()
        .map(|n| {
            let mut into: Vec<&String> = n.points_into.iter().collect();
            into.sort();
            (&n.name, into)
        })
        .collect();
    nodes.sort();
    w.word(nodes.len() as u64);
    for (name, into) in nodes {
        w.text(name).word(into.len() as u64);
        for v in into {
            w.text(v);
        }
    }
    match &request.dependence {
        ParameterDependence::Unknown => {
            w.word(0);
        }
        ParameterDependence::IndependentByDesign { justification } => {
            w.word(1).text(justification);
        }
        ParameterDependence::Joint { variables, covariance } => {
            w.word(2).texts(variables).reals(covariance);
        }
        ParameterDependence::CrossCovariance { blocks } => {
            w.word(3).word(blocks.len() as u64);
            for b in blocks {
                w.text(&b.first).text(&b.second).reals(&b.values);
            }
        }
    }
    w.finish()
}

// --------------------------------------------------------------- execution
