//! Independent references for the numerics the umbrella cyclic-systems record owns on the model
//! side: the Dormand-Prince 5(4) integrator, the SplitMix64 / Box-Muller counter noise, the
//! Euler-Maruyama and diagonal Milstein ensembles, and Krawczyk polynomial-system isolation.
//!
//! `oracle_data/cyclic_general_systems_reference.py` (frozen in the JSON beside it) states each
//! method from its definition with 60-digit arithmetic and exact Fractions, verifies its own
//! tableau against the rooted-tree order conditions, and recomputes every library noise stream
//! in Python big integers.  Nothing in it reads product code and no paper was retrieved.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_model::continuous_cyclic::{OdePolicy, OdeSystem, TimedClamp};
use antecedent_model::equation_system::{IsolationPolicy, PolynomialSystem};
use antecedent_model::nonlinear_sde::{
    SdeContrast, SdeSimulationPolicy, SdeSystem, SdeWorld, counter_normal, sample_sde_paths,
    simulate_sde, splitmix_mix,
};
use serde_json::Value;

fn document() -> Value {
    serde_json::from_str(include_str!("oracle_data/cyclic_general_systems_reference.json")).unwrap()
}

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn grid(value: &Value) -> Vec<Vec<f64>> {
    value.as_array().unwrap().iter().map(numbers).collect()
}

fn close(actual: f64, expected: f64, relative: f64, label: &str) {
    let tolerance = relative * (1.0 + expected.abs());
    assert!((actual - expected).abs() <= tolerance, "{label}: {actual} vs {expected}");
}

fn close_grid(actual: &[Vec<f64>], expected: &[Vec<f64>], relative: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} rows");
    for (t, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.len(), e.len(), "{label} columns");
        for (i, (x, y)) in a.iter().zip(e).enumerate() {
            close(*x, *y, relative, &format!("{label}[{t}][{i}]"));
        }
    }
}

fn unsigned(value: &Value) -> u64 {
    value.as_str().unwrap().parse().unwrap()
}

// ---------------------------------------------------------------- Dormand-Prince

#[test]
fn cyclic_general_systems_dormand_prince_matches_exact_rk_and_step_control() {
    let document = document();
    let ctx = ExecutionContext::for_tests(0);
    let mut single_step_seen = false;
    for case in document["ode"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let system: OdeSystem = serde_json::from_value(case["system"].clone()).unwrap();
        let policy: OdePolicy = serde_json::from_value(case["policy"].clone()).unwrap();
        let schedule: Vec<TimedClamp> = serde_json::from_value(case["schedule"].clone()).unwrap();
        let trajectory = system
            .trajectory(
                &numbers(&case["initial"]),
                &numbers(&case["times"]),
                &schedule,
                policy,
                &ctx,
            )
            .unwrap();
        // The stage arithmetic, the error norm and the controller reproduce the 60-digit replay:
        // the accepted / rejected counts are exact and the states agree to rounding.
        assert_eq!(
            trajectory.accepted_steps as u64,
            case["accepted"].as_u64().unwrap(),
            "{name} accepted"
        );
        assert_eq!(
            trajectory.rejected_steps as u64,
            case["rejected"].as_u64().unwrap(),
            "{name} rejected"
        );
        close_grid(&trajectory.states, &grid(&case["states"]), 5e-11, name);
        // And the integration is what the equation says: it matches the closed form to the
        // policy's accuracy (the replay's own distance to the exact solution bounds it).
        if let Some(exact) = case.get("exact") {
            let reference_error = case["exact_error_of_reference"].as_f64().unwrap();
            let exact = grid(exact);
            for (row, e) in trajectory.states.iter().zip(&exact) {
                for (x, y) in row.iter().zip(e) {
                    assert!(
                        (x - y).abs() <= 2.0 * reference_error + 1e-10,
                        "{name}: {x} vs exact {y} (reference error {reference_error})"
                    );
                }
            }
        }
        if name == "single_accepted_step" {
            single_step_seen = true;
            assert_eq!((trajectory.accepted_steps, trajectory.rejected_steps), (1, 0));
        }
        if name == "zero_error_grows_by_five" {
            // 0.01, 0.05, 0.25, 1.25, 6.25, then the remaining 2.44 of the horizon
            assert_eq!(trajectory.accepted_steps, 6);
        }
    }
    assert!(single_step_seen);
}

// ---------------------------------------------------------------- counter noise

#[test]
fn cyclic_general_systems_splitmix_and_box_muller_match_big_integer_definition() {
    let document = document();
    for pair in document["rng"]["mixes"].as_array().unwrap() {
        assert_eq!(splitmix_mix(unsigned(&pair["x"])), unsigned(&pair["mix"]), "{pair}");
    }
    // The published seed-zero output sequence of the SplitMix64 generator: the k-th output is the
    // finalizer of k times the golden gamma.
    let golden: u64 = 0x9E37_79B9_7F4A_7C15;
    let recalled: Vec<u64> = document["rng"]["recalled_seed_zero_outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(unsigned)
        .collect();
    for (k, expected) in recalled.iter().enumerate() {
        assert_eq!(splitmix_mix((k as u64).wrapping_mul(golden)), *expected, "output {k}");
    }
    for case in document["rng"]["cases"].as_array().unwrap() {
        let value = counter_normal(
            unsigned(&case["seed"]),
            unsigned(&case["path"]),
            unsigned(&case["step"]),
            unsigned(&case["component"]),
        );
        let expected = case["value"].as_f64().unwrap();
        assert!((value - expected).abs() <= 2e-14 * (1.0 + expected.abs()), "{case}: {value}");
        // The documented definition rounds the uniform to a double.  Away from the top of the
        // range that is the exact Box-Muller value; at the very top the double rounding of
        // (k + 1/2) / 2^53 can reach one and the deviate is then exactly zero, a deviation from
        // the real-number transform of about 1e-8.
        let ideal = case["ideal_value"].as_f64().unwrap();
        assert!((value - ideal).abs() <= 2e-8, "{case}");
        if case.get("label").and_then(Value::as_str) == Some("u1_max") {
            assert_eq!(value, 0.0);
        }
    }
}

#[test]
fn cyclic_general_systems_counter_normals_are_standard_normal_and_distinct_by_address() {
    let n = 1u64 << 18;
    let draws: Vec<f64> = (0..n).map(|k| counter_normal(7, k >> 6, k & 63, 1)).collect();
    let nf = n as f64;
    let mean = draws.iter().sum::<f64>() / nf;
    let var = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    let kurtosis = draws.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / nf / (var * var);
    assert!(mean.abs() < 5.0 / nf.sqrt(), "mean {mean}");
    assert!((var - 1.0).abs() < 5.0 * (2.0 / nf).sqrt(), "variance {var}");
    assert!((kurtosis - 3.0).abs() < 5.0 * (24.0 / nf).sqrt(), "kurtosis {kurtosis}");
    for (threshold, cdf) in
        [(-1.0, 0.158_655_253_931_457_07), (0.0, 0.5), (1.0, 0.841_344_746_068_542_9)]
    {
        let frequency = draws.iter().filter(|&&x| x <= threshold).count() as f64 / nf;
        let sd = (cdf * (1.0 - cdf) / nf).sqrt();
        assert!((frequency - cdf).abs() < 5.0 * sd, "P(Z <= {threshold}) = {frequency}");
    }
    // each address coordinate changes the deviate
    let base = counter_normal(1, 2, 3, 4);
    for other in [
        counter_normal(2, 2, 3, 4),
        counter_normal(1, 3, 3, 4),
        counter_normal(1, 2, 4, 4),
        counter_normal(1, 2, 3, 5),
    ] {
        assert_ne!(base.to_bits(), other.to_bits());
    }
    // adjacent addresses are uncorrelated at the 5-sigma level
    let lag: f64 = draws.windows(2).map(|w| w[0] * w[1]).sum::<f64>() / nf;
    assert!(lag.abs() < 5.0 / nf.sqrt(), "lag-one product {lag}");
}

// ---------------------------------------------------------------- Euler-Maruyama / Milstein

#[test]
fn cyclic_general_systems_euler_maruyama_and_milstein_replay_the_frozen_noise_exactly() {
    let document = document();
    let ctx = ExecutionContext::for_tests(0);
    for case in document["sde"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let system: SdeSystem = serde_json::from_value(case["system"].clone()).unwrap();
        let compiled = system.compile().unwrap();
        let policy: SdeSimulationPolicy = serde_json::from_value(case["policy"].clone()).unwrap();
        let worlds: Vec<SdeWorld> = serde_json::from_value(case["worlds"].clone()).unwrap();
        let theta = numbers(&case["theta"]);
        let initial = numbers(&case["initial"]);
        let contrasts: Vec<SdeContrast> = if worlds.is_empty() {
            vec![]
        } else {
            vec![SdeContrast {
                id: "contrast".into(),
                treated: worlds[0].id.clone(),
                control: "baseline".into(),
            }]
        };
        let report =
            simulate_sde(&compiled, Some(&theta), &initial, &policy, &worlds, &contrasts, &ctx)
                .unwrap();
        let d = compiled.dimension();
        let references = case["reports"].as_array().unwrap();
        assert_eq!(report.worlds.len(), references.len(), "{name} worlds");
        for (world, reference) in report.worlds.iter().zip(references) {
            let label = format!("{name}/{}", world.id);
            assert_eq!(world.id, reference["id"].as_str().unwrap());
            // raw declared-step paths
            let plan_world = if world.id == "baseline" {
                SdeWorld { id: "baseline".into(), ..SdeWorld::default() }
            } else {
                worlds.iter().find(|w| w.id == world.id).unwrap().clone()
            };
            let paths =
                sample_sde_paths(&compiled, Some(&theta), &initial, &policy, &plan_world, &ctx)
                    .unwrap();
            let expected_paths = reference["coarse"].as_array().unwrap();
            let times = paths.times.len();
            for (p, path) in expected_paths.iter().enumerate() {
                for (t, row) in grid(path).iter().enumerate() {
                    for (i, expected) in row.iter().enumerate() {
                        close(
                            paths.states[(p * times + t) * d + i],
                            *expected,
                            1e-10,
                            &format!("{label} path {p} time {t} coord {i}"),
                        );
                    }
                }
            }
            close_grid(&world.mean, &grid(&reference["mean"]), 1e-10, &format!("{label} mean"));
            close_grid(
                &world.variance,
                &grid(&reference["variance"]),
                1e-9,
                &format!("{label} variance"),
            );
            close_grid(
                &world.covariance,
                &grid(&reference["covariance"]),
                1e-9,
                &format!("{label} covariance"),
            );
            let numerical = world.numerical.as_ref().unwrap();
            assert_eq!(numerical.refined_step, policy.step / 2.0);
            close_grid(
                &numerical.weak_difference,
                &grid(&reference["weak_difference"]),
                1e-9,
                &format!("{label} weak"),
            );
            close_grid(
                &numerical.weak_difference_standard_error,
                &grid(&reference["weak_difference_standard_error"]),
                1e-9,
                &format!("{label} weak se"),
            );
            close_grid(
                &numerical.strong_rms_difference,
                &grid(&reference["strong_rms_difference"]),
                1e-9,
                &format!("{label} strong"),
            );
            close_grid(
                &numerical.richardson_mean,
                &grid(&reference["richardson_mean"]),
                1e-9,
                &format!("{label} richardson"),
            );
        }
        for (contrast, reference) in
            report.contrasts.iter().zip(case["contrasts"].as_array().unwrap())
        {
            let label = format!("{name}/contrast");
            close_grid(
                &contrast.mean_difference,
                &grid(&reference["mean_difference"]),
                1e-9,
                &label,
            );
            let numerical = contrast.numerical.as_ref().unwrap();
            close_grid(
                &numerical.weak_difference,
                &grid(&reference["weak_difference"]),
                1e-9,
                &format!("{label} weak"),
            );
            close_grid(
                &numerical.weak_difference_standard_error,
                &grid(&reference["weak_difference_standard_error"]),
                1e-9,
                &format!("{label} weak se"),
            );
        }
    }
}

// ---------------------------------------------------------------- Krawczyk isolation

#[test]
fn cyclic_general_systems_krawczyk_isolation_agrees_with_exact_sturm_roots() {
    let document = document();
    let ctx = ExecutionContext::for_tests(0);
    for case in document["krawczyk"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let system: PolynomialSystem = serde_json::from_value(case["system"].clone()).unwrap();
        let policy: IsolationPolicy = serde_json::from_value(case["policy"].clone()).unwrap();
        let solutions = system.isolate(policy, &ctx).unwrap();
        let roots = case["roots"].as_array().unwrap();
        let brackets: Vec<Vec<[f64; 2]>> = roots
            .iter()
            .map(|r| {
                r["brackets"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|b| {
                        let b = numbers(b);
                        [b[0], b[1]]
                    })
                    .collect()
            })
            .collect();
        let hits = |region: &[antecedent_model::equation_system::Interval], root: &[[f64; 2]]| {
            region.iter().zip(root).all(|(b, r)| b.lower <= r[1] && b.upper >= r[0])
        };
        // Soundness of every certificate: a certified box meets exactly one exact root and is
        // narrow; a root is never both certified and of multiplicity above one.
        for region in &solutions.roots {
            let meeting: Vec<usize> =
                (0..brackets.len()).filter(|&k| hits(region, &brackets[k])).collect();
            assert_eq!(
                meeting.len(),
                1,
                "{name}: certified box {region:?} meets roots {meeting:?}"
            );
            assert_eq!(
                roots[meeting[0]]["multiplicity"].as_u64().unwrap(),
                1,
                "{name}: certified a multiple root"
            );
            for b in region {
                assert!(b.upper - b.lower <= 1e-6, "{name}: certified box is wide");
            }
        }
        // No exact root is lost: each lies in a certified box or in an unresolved one; an
        // interior simple root is certified.
        for (k, root) in brackets.iter().enumerate() {
            let multiplicity = roots[k]["multiplicity"].as_u64().unwrap();
            let certified = solutions.roots.iter().filter(|r| hits(r, root)).count();
            let unresolved = solutions.unresolved.iter().filter(|r| hits(r, root)).count();
            assert!(certified + unresolved >= 1, "{name}: root {k} lies in no box {solutions:?}");
            if multiplicity == 1 && name != "roots_on_the_domain_boundary" {
                assert_eq!(certified, 1, "{name}: simple root {k} was not isolated");
            }
            if multiplicity > 1 {
                assert_eq!(certified, 0, "{name}: multiple root {k}");
                assert!(unresolved >= 1, "{name}: multiple root {k} must stay unresolved");
            }
        }
        let simple = roots.iter().filter(|r| r["multiplicity"].as_u64().unwrap() == 1).count();
        if name != "roots_on_the_domain_boundary" {
            assert_eq!(solutions.roots.len(), simple, "{name}: number of certified roots");
        }
        // Without multiple roots or boundary roots the search is complete.
        let has_multiple = roots.iter().any(|r| r["multiplicity"].as_u64().unwrap() > 1);
        if !has_multiple && name != "roots_on_the_domain_boundary" {
            assert!(
                solutions.unresolved.is_empty(),
                "{name}: unresolved {:?}",
                solutions.unresolved
            );
        }
        if case["distinct_real_roots"].as_u64().unwrap() == 0 {
            assert!(solutions.roots.is_empty() && solutions.unresolved.is_empty(), "{name}");
        }
    }
}
