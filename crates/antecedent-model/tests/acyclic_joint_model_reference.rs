//! Composed acyclic mechanisms against exact rational references.
//!
//! `oracle_data/acyclic_joint_model_reference.py` (frozen in the JSON beside it) evaluates, with
//! exact rational arithmetic and an algorithm independent of the product's forward substitution,
//! the joint mean and covariance of random linear-Gaussian compositions (six and eight nodes,
//! shared latents, a near-deterministic child, unsorted variable names) under no, one and several
//! hard interventions; the Schur-complement conditional law and evidence log density; and the
//! first-order delta-method gradient and variance of a weighted contrast of interventional means
//! under a joint, a cross-covariance and an independent-by-design parameter covariance (the
//! covariance rank-deficient). It also enumerates every joint state of finite compositions with
//! exact zero cells, unsorted level names and extreme odds, with and without interventions. The
//! product must reproduce all of them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::ExecutionContext;
use antecedent_model::compose::{
    ComposedFinite, ComposedLinearGaussian, ComposedModelSpec, ContrastArm, CrossCovarianceBlock,
    Domain, LatentDependence, MeanContrast, MechanismContext, MechanismDescriptor, MechanismForm,
    MechanismOrigin, MechanismProvenance, MechanismUncertainty, ParameterId, ParameterTerm,
    ParentBinding, PosteriorDependence, TabularRow, UncertaintyMeaning,
};
use serde_json::Value;

fn doc() -> Value {
    serde_json::from_str(include_str!("oracle_data/acyclic_joint_model_reference.json")).unwrap()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let scale = want.abs().max(1.0);
    assert!((got - want).abs() <= rel * scale, "{what}: {got} vs {want}");
}

fn provenance(var: &str) -> MechanismProvenance {
    MechanismProvenance {
        origin: MechanismOrigin::Fitted,
        artifact_id: format!("fit:{var}"),
        evidence_id: format!("data:{var}"),
    }
}

fn binding(var: &str, domain: Domain) -> ParentBinding {
    ParentBinding { variable: var.into(), units: "u".into(), transform: "identity".into(), domain }
}

fn descriptor(
    var: &str,
    form: MechanismForm,
    parents: Vec<ParentBinding>,
    support: Domain,
) -> MechanismDescriptor {
    MechanismDescriptor {
        variable: var.into(),
        form,
        conditioning_set: parents.iter().map(|p| p.variable.clone()).collect(),
        parents,
        output_units: "u".into(),
        output_transform: "identity".into(),
        output_support: support,
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: provenance(var),
        latents: Vec::new(),
        uncertainty: None,
    }
}

fn spec_of(
    mechanisms: Vec<MechanismDescriptor>,
    dependence: PosteriorDependence,
) -> ComposedModelSpec {
    let mut edges = Vec::new();
    for m in &mechanisms {
        for p in &m.parents {
            edges.push((p.variable.clone(), m.variable.clone()));
        }
    }
    ComposedModelSpec {
        target: MechanismContext { population: "pop".into(), regime: "observational".into() },
        mechanisms,
        edges,
        invariances: Vec::new(),
        selection_diagrams: Vec::new(),
        dependence,
        admit_declared_mechanisms: false,
    }
}

fn linear_mechanisms(case: &Value) -> Vec<MechanismDescriptor> {
    let nodes = case["nodes"].as_array().unwrap();
    // The members of each latent, so that every sharing is declared reciprocally.
    let mut members: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for nd in nodes {
        for u in nd["loadings"].as_object().unwrap().keys() {
            members.entry(u.clone()).or_default().push(nd["name"].as_str().unwrap().to_owned());
        }
    }
    nodes
        .iter()
        .map(|nd| {
            let name = nd["name"].as_str().unwrap();
            let coefficients: Vec<(String, f64)> = nd["parents"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(p, b)| (p.clone(), b.as_f64().unwrap()))
                .collect();
            let parents = coefficients.iter().map(|(p, _)| binding(p, Domain::Real)).collect();
            let mut d = descriptor(
                name,
                MechanismForm::LinearGaussian {
                    intercept: nd["intercept"].as_f64().unwrap(),
                    sigma: nd["sigma"].as_f64().unwrap(),
                    coefficients,
                },
                parents,
                Domain::Real,
            );
            d.latents = nd["loadings"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(u, g)| LatentDependence {
                    latent: u.clone(),
                    loading: Some(g.as_f64().unwrap()),
                    shared_with: members[u]
                        .iter()
                        .filter(|m| m.as_str() != name)
                        .cloned()
                        .collect(),
                })
                .collect();
            d
        })
        .collect()
}

fn pairs<'a>(object: &'a Value) -> Vec<(&'a str, f64)> {
    object.as_object().unwrap().iter().map(|(k, v)| (k.as_str(), v.as_f64().unwrap())).collect()
}

fn term(value: &str) -> ParameterTerm {
    if value == "intercept" {
        ParameterTerm::Intercept
    } else {
        ParameterTerm::Coefficient(value.into())
    }
}

#[test]
fn linear_gaussian_laws_conditioning_and_delta_method_equal_the_rational_reference() {
    let doc = doc();
    for case in doc["linear"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let model = ComposedLinearGaussian::compile(&spec_of(
            linear_mechanisms(case),
            PosteriorDependence::Unknown,
        ))
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));

        for law in case["laws"].as_array().unwrap() {
            let fixed = pairs(&law["do"]);
            let got = model.law(&fixed).unwrap();
            let names: Vec<&str> =
                law["names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
            for (i, a) in names.iter().enumerate() {
                close(
                    got.mean_of(a).unwrap(),
                    law["mean"][i].as_f64().unwrap(),
                    1e-11,
                    &format!("{name} do{fixed:?} mean {a}"),
                );
                for (j, b) in names.iter().enumerate() {
                    close(
                        got.covariance_of(a, b).unwrap(),
                        law["cov"][i][j].as_f64().unwrap(),
                        1e-11,
                        &format!("{name} do{fixed:?} cov {a},{b}"),
                    );
                }
            }
        }

        for cond in case["conditionings"].as_array().unwrap() {
            let fixed = pairs(&cond["do"]);
            let observed = pairs(&cond["observed"]);
            let got = model.law(&fixed).unwrap().condition(&observed).unwrap();
            let names: Vec<&str> =
                cond["names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
            assert_eq!(got.variables().len(), names.len(), "{name}");
            for (i, a) in names.iter().enumerate() {
                close(
                    got.mean_of(a).unwrap(),
                    cond["mean"][i].as_f64().unwrap(),
                    1e-9,
                    &format!("{name} cond mean {a}"),
                );
                close(
                    got.variance_of(a).unwrap(),
                    cond["cov"][i][i].as_f64().unwrap(),
                    1e-9,
                    &format!("{name} cond var {a}"),
                );
            }
            close(
                got.evidence_log_density(),
                cond["log_density"].as_f64().unwrap(),
                1e-9,
                &format!("{name} evidence"),
            );
        }

        let contrast = MeanContrast {
            outcome: case["contrast"]["outcome"].as_str().unwrap().into(),
            arms: case["contrast"]["arms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| ContrastArm {
                    weight: a["weight"].as_f64().unwrap(),
                    interventions: pairs(&a["do"])
                        .into_iter()
                        .map(|(k, v)| (k.to_owned(), v))
                        .collect(),
                })
                .collect(),
        };
        close(
            model.mean_contrast(&contrast).unwrap(),
            case["contrast"]["point"].as_f64().unwrap(),
            1e-12,
            &format!("{name} contrast"),
        );

        // The delta method under three licensed dependence declarations.
        let parameters: Vec<ParameterId> = case["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| ParameterId {
                mechanism: p["mechanism"].as_str().unwrap().into(),
                term: term(p["term"].as_str().unwrap()),
            })
            .collect();
        let k = parameters.len();
        let sigma: Vec<f64> = case["sigma"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let gradient: Vec<f64> =
            case["gradient"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let mechanisms = linear_mechanisms(case);
        let with_marginals = |mechanisms: &[MechanismDescriptor]| -> Vec<MechanismDescriptor> {
            mechanisms
                .iter()
                .cloned()
                .map(|mut d| {
                    let own: Vec<usize> =
                        (0..k).filter(|&i| parameters[i].mechanism == d.variable).collect();
                    d.uncertainty = Some(MechanismUncertainty {
                        meaning: UncertaintyMeaning::Posterior,
                        parameters: own.iter().map(|&i| parameters[i].term.clone()).collect(),
                        covariance: own
                            .iter()
                            .flat_map(|&a| own.iter().map(move |&b| (a, b)))
                            .map(|(a, b)| sigma[a * k + b])
                            .collect(),
                    });
                    d
                })
                .collect()
        };
        let mut names_sorted: Vec<String> = mechanisms.iter().map(|d| d.variable.clone()).collect();
        names_sorted.sort();
        let mut blocks = Vec::new();
        for (x, first) in names_sorted.iter().enumerate() {
            for second in &names_sorted[x + 1..] {
                let fa: Vec<usize> =
                    (0..k).filter(|&i| &parameters[i].mechanism == first).collect();
                let sb: Vec<usize> =
                    (0..k).filter(|&i| &parameters[i].mechanism == second).collect();
                blocks.push(CrossCovarianceBlock {
                    first: first.clone(),
                    second: second.clone(),
                    first_parameters: fa.iter().map(|&i| parameters[i].term.clone()).collect(),
                    second_parameters: sb.iter().map(|&i| parameters[i].term.clone()).collect(),
                    values: fa
                        .iter()
                        .flat_map(|&a| sb.iter().map(move |&b| (a, b)))
                        .map(|(a, b)| sigma[a * k + b])
                        .collect(),
                });
            }
        }
        let declarations = [
            (
                "joint",
                spec_of(
                    mechanisms.clone(),
                    PosteriorDependence::Joint {
                        meaning: UncertaintyMeaning::Posterior,
                        parameters: parameters.clone(),
                        covariance: sigma.clone(),
                    },
                ),
                "variance_joint",
            ),
            (
                "cross",
                spec_of(
                    with_marginals(&mechanisms),
                    PosteriorDependence::CrossCovariance { blocks },
                ),
                "variance_joint",
            ),
            (
                "independent",
                spec_of(
                    with_marginals(&mechanisms),
                    PosteriorDependence::IndependentByDesign {
                        justification: "disjoint evidence".into(),
                    },
                ),
                "variance_independent",
            ),
        ];
        for (label, spec, key) in declarations {
            let model = ComposedLinearGaussian::compile(&spec)
                .unwrap_or_else(|e| panic!("{name} {label}: {e:?}"));
            let result = model
                .mean_contrast_uncertainty(&contrast)
                .unwrap_or_else(|e| panic!("{name} {label}: {e:?}"));
            close(
                result.point,
                case["contrast"]["point"].as_f64().unwrap(),
                1e-12,
                &format!("{name} {label} point"),
            );
            close(
                result.variance,
                case[key].as_f64().unwrap(),
                1e-10,
                &format!("{name} {label} variance"),
            );
            close(
                result.standard_error,
                case[key].as_f64().unwrap().sqrt(),
                1e-10,
                &format!("{name} {label} se"),
            );
            let got: BTreeMap<ParameterId, f64> = result.gradient.iter().cloned().collect();
            assert_eq!(got.len(), k, "{name} {label}: gradient length");
            for (id, want) in parameters.iter().zip(&gradient) {
                close(got[id], *want, 1e-11, &format!("{name} {label} gradient {id:?}"));
            }
        }
    }
}

fn finite_mechanisms(case: &Value) -> Vec<MechanismDescriptor> {
    let levels = &case["levels"];
    let level_names = |v: &str| -> Vec<String> {
        levels[v].as_array().unwrap().iter().map(|l| l.as_str().unwrap().to_owned()).collect()
    };
    case["variables"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            let var = v.as_str().unwrap();
            let own = level_names(var);
            let parent_names: Vec<&str> = case["parents"][var]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.as_str().unwrap())
                .collect();
            let rows = case["tables"][var]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| TabularRow {
                    parent_levels: parent_names
                        .iter()
                        .zip(row["parents"].as_array().unwrap())
                        .map(|(p, l)| ((*p).to_owned(), l.as_str().unwrap().to_owned()))
                        .collect(),
                    probabilities: own
                        .iter()
                        .zip(row["probs"].as_array().unwrap())
                        .map(|(l, p)| (l.clone(), p.as_f64().unwrap()))
                        .collect(),
                })
                .collect();
            let parents =
                parent_names.iter().map(|p| binding(p, Domain::Levels(level_names(p)))).collect();
            descriptor(
                var,
                MechanismForm::Tabular { levels: own.clone(), rows },
                parents,
                Domain::Levels(own),
            )
        })
        .collect()
}

#[test]
fn finite_laws_equal_exact_enumeration_of_the_truncated_factorization() {
    let doc = doc();
    let ctx = ExecutionContext::for_tests(1);
    for case in doc["finite"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let model = ComposedFinite::compile(&spec_of(
            finite_mechanisms(case),
            PosteriorDependence::Unknown,
        ))
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        for scenario in case["scenarios"].as_array().unwrap() {
            let fixed: Vec<(&str, &str)> = scenario["do"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str().unwrap()))
                .collect();
            let law = model.law(&fixed, &ctx).unwrap();
            let mut total = 0.0;
            for state in scenario["states"].as_array().unwrap() {
                let event: Vec<(&str, &str)> = state["state"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str().unwrap()))
                    .collect();
                let want = state["p"].as_f64().unwrap();
                let got = law.probability(&event).unwrap();
                assert!(
                    (got - want).abs() <= 1e-14 + 1e-12 * want,
                    "{name} do{fixed:?} {event:?}: {got} vs {want}"
                );
                total += got;
            }
            assert!((total - 1.0).abs() < 1e-12, "{name}: the law sums to {total}");
        }
    }
}
