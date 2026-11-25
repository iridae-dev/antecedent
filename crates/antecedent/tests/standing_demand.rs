//! Every compiled route's claim demand states the kind of claim the route publishes.
//!
//! A lowering states the claim kind its program demands (`ClaimDemand::kind`, or the
//! `claim` of the terminal, parent, interval and point program specs that build one).
//! The demand is what the plan view shows before anything runs, so it must never present
//! a stronger or other claim than the route publishes. These tests read the crate's own
//! source and find every construction that states a demand, so a route added without a
//! declared publication, or a demand changed without it, fails:
//!
//! * each construction is declared in [`ROUTES`] with what its route publishes, and the
//!   kind it states is that publication's kind;
//! * a terminal program's demand agrees with the functional it publishes (an identified
//!   set is bounds and only bounds; a search or plan publishes no claim; an association
//!   is never a causal point);
//! * a program cell of the associational route family never demands a causal point,
//!   response, mixture or policy value;
//! * the routes a `Study` seals into a claim are checked against the sealed claim itself
//!   by `family_contract_conformance::a_saved_result_is_re_derived_by_an_independent_consumer`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The construction forms that state a claim demand, with the field that holds it.
const FORMS: [(&str, &str); 5] = [
    ("ClaimDemand", "kind"),
    ("TerminalSpec", "claim"),
    ("ParentSpec", "claim"),
    ("MeasuredIntervalProgram", "claim"),
    ("PointProgram", "claim"),
];

/// What a route publishes, by construction: file under `src/`, enclosing function, form,
/// and publication. Publications:
///
/// * `decision`: the value of a decision (`PolicyValue`);
/// * `point`, `bounds`, `response`, `association`, `mixture`: a claim of that kind;
/// * `none`: no claim (a search, an assessment, a plan, a recomputation bundle, a check);
/// * `forwarded`: the kind a caller states, each caller listed under its own form;
/// * `sealed`: decided per study at lowering from its identification; the sealed claim of
///   an executed result is checked against it by the conformance suite;
/// * `derived:K,...`: chosen among exactly the kinds listed, by the reading or target.
const ROUTES: &[(&str, &str, &str, &str)] = &[
    ("analysis/attached_decision.rs", "lower_parent", "TerminalSpec", "forwarded"),
    (
        "analysis/bridge_transport_decision.rs",
        "compile_assessed_bridge_posterior_decision",
        "ClaimDemand",
        "decision",
    ),
    (
        "analysis/bridge_transport_decision.rs",
        "compile_assessed_bridge_robust_decision",
        "ClaimDemand",
        "decision",
    ),
    (
        "analysis/bridge_transport_decision.rs",
        "compile_assessed_bridge_transport_decision",
        "ClaimDemand",
        "decision",
    ),
    (
        "analysis/bridge_transport_decision.rs",
        "compile_bridge_transport_decision",
        "ClaimDemand",
        "decision",
    ),
    (
        "analysis/bridge_transport_premise_assessment.rs",
        "compile_bridge_transport_premise_assessment",
        "ClaimDemand",
        "none",
    ),
    (
        "analysis/claim_decision_programs.rs",
        "compile_contract_decision",
        "TerminalSpec",
        "decision",
    ),
    ("analysis/claim_decision_programs.rs", "compile_decision_functional", "TerminalSpec", "point"),
    (
        "analysis/claim_decision_programs.rs",
        "compile_identified_set_decision",
        "TerminalSpec",
        "bounds",
    ),
    (
        "analysis/claim_decision_programs.rs",
        "compile_supplied_claim_decision",
        "TerminalSpec",
        "decision",
    ),
    ("analysis/compiled_adaptive_interval.rs", "lower", "MeasuredIntervalProgram", "point"),
    ("analysis/compiled_choice_interval.rs", "lower", "MeasuredIntervalProgram", "point"),
    ("analysis/compiled_cohort_interval.rs", "lower", "MeasuredIntervalProgram", "point"),
    ("analysis/compiled_count_fitted_interval.rs", "lower", "MeasuredIntervalProgram", "point"),
    ("analysis/compiled_discovered_transport.rs", "lower", "ClaimDemand", "none"),
    ("analysis/compiled_dose_grid_bias_aware.rs", "lower", "ClaimDemand", "response"),
    ("analysis/compiled_external_feedback.rs", "lower", "PointProgram", "point"),
    ("analysis/compiled_joint_cells_learners.rs", "lower", "ClaimDemand", "point"),
    ("analysis/compiled_measured_interval.rs", "lower", "ClaimDemand", "forwarded"),
    (
        "analysis/compiled_measurement_transport.rs",
        "compile_measurement_transport",
        "PointProgram",
        "point",
    ),
    (
        "analysis/compiled_measurement_transport.rs",
        "compile_pooled_differential_transport",
        "PointProgram",
        "point",
    ),
    ("analysis/compiled_per_population_discovery.rs", "lower", "ClaimDemand", "none"),
    ("analysis/compiled_point_program.rs", "lower", "ClaimDemand", "forwarded"),
    ("analysis/compiled_smoothed_dose_learners.rs", "lower", "PointProgram", "response"),
    ("analysis/compiled_temporal_interval.rs", "lower", "MeasuredIntervalProgram", "response"),
    ("analysis/compiled_transport_scenarios.rs", "lower", "PointProgram", "mixture"),
    (
        "analysis/compiled_transported_cohort.rs",
        "compile_transported_cohort",
        "PointProgram",
        "point",
    ),
    ("analysis/compiled_trial_aipw_learners.rs", "lower", "PointProgram", "point"),
    ("analysis/composition.rs", "compile_bridge_bundle_recomputation", "ClaimDemand", "none"),
    ("analysis/composition_decision_terminal.rs", "compile_decision_bundle", "ClaimDemand", "none"),
    (
        "analysis/composition_sequential_bundle.rs",
        "compile_sequential_reliability_node_bundle",
        "ClaimDemand",
        "none",
    ),
    (
        "analysis/cyclic_action_world_decision.rs",
        "compile_cyclic_action_world_decision_encoded",
        "ClaimDemand",
        "decision",
    ),
    (
        "analysis/cyclic_posterior_action_decision.rs",
        "compile_cyclic_posterior_action_decision_checked",
        "ClaimDemand",
        "decision",
    ),
    (
        "analysis/cyclic_target_feedback_decision.rs",
        "compile_cyclic_target_feedback_decision",
        "ClaimDemand",
        "decision",
    ),
    ("analysis/descriptive_programs.rs", "lower", "ClaimDemand", "association"),
    (
        "analysis/design_planning_programs.rs",
        "compile_choice_study_ranking",
        "TerminalSpec",
        "point",
    ),
    (
        "analysis/decision_reliability_programs.rs",
        "compile_decision_reliability_assessment",
        "TerminalSpec",
        "point",
    ),
    (
        "analysis/decision_reliability_programs.rs",
        "compile_decision_reliability_derivation",
        "TerminalSpec",
        "point",
    ),
    (
        "analysis/decision_reliability_programs.rs",
        "compile_producer_carrier_assessment",
        "TerminalSpec",
        "point",
    ),
    (
        "analysis/decision_reliability_programs.rs",
        "compile_producer_carrier_derivation",
        "TerminalSpec",
        "point",
    ),
    ("analysis/design_planning_programs.rs", "compile_design_ranking", "TerminalSpec", "none"),
    (
        "analysis/design_planning_programs.rs",
        "compile_identification_repair",
        "TerminalSpec",
        "none",
    ),
    ("analysis/design_planning_programs.rs", "compile_study_addition_plan", "TerminalSpec", "none"),
    (
        "analysis/design_planning_programs.rs",
        "compile_z_transport_evidence_plan",
        "TerminalSpec",
        "none",
    ),
    (
        "analysis/design_planning_programs.rs",
        "compile_z_transport_failure_snapshot",
        "TerminalSpec",
        "none",
    ),
    ("analysis/design_planning_programs.rs", "structure_spec", "TerminalSpec", "none"),
    ("analysis/execute/compile_decision_program.rs", "parent_over", "ParentSpec", "none"),
    ("analysis/execute/compile_planning.rs", "lower_planning", "ClaimDemand", "none"),
    ("analysis/execute/compile_preference.rs", "parent_of", "ParentSpec", "decision"),
    ("analysis/execute/compile_robust.rs", "compile", "ParentSpec", "none"),
    (
        "analysis/fitted_dependent_measurement.rs",
        "compile_fitted_dependent_measurement",
        "ClaimDemand",
        "point",
    ),
    ("analysis/graph_imputation_effect_decision.rs", "lower_outer", "ClaimDemand", "decision"),
    (
        "analysis/graph_imputation_effect_decision.rs",
        "lower_paired_outer",
        "ClaimDemand",
        "decision",
    ),
    (
        "analysis/graph_imputation_kernel_transport_decision.rs",
        "lower_outer",
        "ClaimDemand",
        "decision",
    ),
    (
        "analysis/graph_imputation_mar_transport_decision.rs",
        "compile_graph_imputation_mar_transport_with_cluster",
        "ClaimDemand",
        "decision",
    ),
    ("analysis/graph_imputation_transport_decision.rs", "lower_outer", "ClaimDemand", "decision"),
    (
        "analysis/input_decision_programs.rs",
        "compile_adaptive_policy_search",
        "TerminalSpec",
        "decision",
    ),
    (
        "analysis/input_decision_programs.rs",
        "compile_decision_draws_ranking",
        "TerminalSpec",
        "decision",
    ),
    (
        "analysis/input_decision_programs.rs",
        "compile_producer_carrier_ranking",
        "TerminalSpec",
        "decision",
    ),
    ("analysis/input_decision_programs.rs", "compile_evidence_obligations", "TerminalSpec", "none"),
    (
        "analysis/input_decision_programs.rs",
        "compile_finite_action_decision",
        "TerminalSpec",
        "decision",
    ),
    (
        "analysis/input_decision_programs.rs",
        "compile_finite_enumeration_baseline",
        "TerminalSpec",
        "decision",
    ),
    ("analysis/input_decision_programs.rs", "compile_input_functional", "TerminalSpec", "point"),
    ("analysis/input_decision_programs.rs", "compile_inverse_query", "TerminalSpec", "decision"),
    (
        "analysis/input_decision_programs.rs",
        "compile_prior_signal_adaptation",
        "TerminalSpec",
        "none",
    ),
    (
        "analysis/input_decision_programs.rs",
        "compile_sensitivity_decision",
        "TerminalSpec",
        "decision",
    ),
    (
        "analysis/input_decision_programs.rs",
        "compile_supported_input_decision",
        "TerminalSpec",
        "decision",
    ),
    (
        "analysis/joint_transport_effect_decision.rs",
        "compile_joint_transport_effect_decision_as",
        "ClaimDemand",
        "decision",
    ),
    (
        "analysis/joint_transport_predictive_decision.rs",
        "compile_joint_transport_predictive_decision_as",
        "ClaimDemand",
        "decision",
    ),
    ("analysis/kernel_programs/mod.rs", "lower", "ClaimDemand", "none"),
    ("analysis/lowering/associational.rs", "lower_selected", "ClaimDemand", "association"),
    ("analysis/lowering/categorical_posterior_candidate.rs", "lower", "ClaimDemand", "none"),
    (
        "analysis/lowering/chain.rs",
        "lower_chain_with_decision",
        "ClaimDemand",
        "derived:Associational,Bounds,Point",
    ),
    (
        "analysis/lowering/cyclic.rs",
        "lower_cyclic_family_payloads",
        "ClaimDemand",
        "derived:Bounds,Response",
    ),
    ("analysis/lowering/design_based.rs", "lower", "ClaimDemand", "sealed"),
    ("analysis/lowering/discovery.rs", "lower_search", "ClaimDemand", "point"),
    ("analysis/lowering/discovery_validation.rs", "lower", "ClaimDemand", "none"),
    ("analysis/lowering/events.rs", "lower", "ClaimDemand", "derived:Associational,Response"),
    ("analysis/lowering/graph_falsification.rs", "lower", "ClaimDemand", "none"),
    ("analysis/lowering/graph_imputation.rs", "lower", "ClaimDemand", "point"),
    ("analysis/lowering/graph_posterior.rs", "lower", "ClaimDemand", "derived:Mixture,Response"),
    ("analysis/lowering/joint_panel/design.rs", "lower", "ClaimDemand", "response"),
    ("analysis/lowering/joint_panel/experiment.rs", "claim", "ClaimDemand", "decision"),
    ("analysis/lowering/joint_panel/experiment.rs", "claim", "ClaimDemand", "decision"),
    ("analysis/lowering/joint_panel/experiment.rs", "claim", "ClaimDemand", "response"),
    ("analysis/lowering/joint_panel/imputation.rs", "lower", "ClaimDemand", "response"),
    ("analysis/lowering/joint_panel/response.rs", "lower", "ClaimDemand", "response"),
    ("analysis/lowering/joint_panel/search.rs", "lower", "ClaimDemand", "decision"),
    ("analysis/lowering/latent_soft_intervention.rs", "lower", "ClaimDemand", "response"),
    ("analysis/lowering/local_cyclic_callback.rs", "lower", "ClaimDemand", "none"),
    ("analysis/lowering/mnar_fixed_shift.rs", "lower", "ClaimDemand", "none"),
    ("analysis/lowering/model_operation.rs", "lower", "ClaimDemand", "none"),
    ("analysis/lowering/static_effect.rs", "lower", "ClaimDemand", "sealed"),
    ("analysis/lowering/static_response.rs", "lower", "ClaimDemand", "response"),
    ("analysis/lowering/structural.rs", "lower_mediation", "ClaimDemand", "point"),
    ("analysis/lowering/structural.rs", "lower_structural", "ClaimDemand", "point"),
    ("analysis/lowering/temporal.rs", "assemble", "ClaimDemand", "sealed"),
    ("analysis/lowering/transported_path_specific.rs", "lower", "ClaimDemand", "none"),
    (
        "analysis/observation_query.rs",
        "compile_dependent_measured_observation",
        "ClaimDemand",
        "point",
    ),
    ("analysis/observation_query.rs", "compile_joint_measured_observation", "ClaimDemand", "point"),
    (
        "analysis/observational_policy.rs",
        "compile_observational_policy_as",
        "ClaimDemand",
        "decision",
    ),
    (
        "analysis/observational_policy.rs",
        "compile_policy_subsampling_as",
        "ClaimDemand",
        "decision",
    ),
    ("analysis/preference_learning_programs.rs", "compile_model", "TerminalSpec", "association"),
    ("analysis/preference_learning_programs.rs", "compile_set", "TerminalSpec", "bounds"),
    (
        "analysis/sampling_assumption_union.rs",
        "compile_cluster_sampling_assumption_union",
        "ClaimDemand",
        "bounds",
    ),
    (
        "analysis/sampling_assumption_union.rs",
        "compile_iid_sampling_assumption_union",
        "ClaimDemand",
        "bounds",
    ),
    (
        "analysis/sampling_assumption_union.rs",
        "compile_moving_block_sampling_assumption_union",
        "ClaimDemand",
        "bounds",
    ),
    (
        "analysis/sampling_assumption_union.rs",
        "compile_panel_block_sampling_assumption_union",
        "ClaimDemand",
        "bounds",
    ),
    (
        "analysis/sampling_assumption_union.rs",
        "compile_sampling_assumption_union",
        "ClaimDemand",
        "bounds",
    ),
    (
        "analysis/sampling_assumption_union.rs",
        "compile_stratified_cluster_sampling_assumption_union",
        "ClaimDemand",
        "bounds",
    ),
    (
        "analysis/sampling_assumption_union.rs",
        "compile_stratified_moving_block_sampling_assumption_union",
        "ClaimDemand",
        "bounds",
    ),
    (
        "analysis/sequential_policy_reliability.rs",
        "compile_sequential_policy_reliability_as",
        "ClaimDemand",
        "decision",
    ),
    (
        "analysis/supplied_sensitivity_programs.rs",
        "compile_matched_design_sensitivity",
        "TerminalSpec",
        "bounds",
    ),
    (
        "analysis/supplied_sensitivity_programs.rs",
        "compile_supplied_estimate_evalue",
        "TerminalSpec",
        "bounds",
    ),
    ("analysis/supplied_terminal.rs", "lower_over", "ClaimDemand", "forwarded"),
    ("analysis/temporal_linear_prediction.rs", "lower", "ClaimDemand", "association"),
    (
        "analysis/temporal_path_transport.rs",
        "compile_continuous_chain_transport",
        "ClaimDemand",
        "point",
    ),
    ("analysis/temporal_path_transport.rs", "lower", "ClaimDemand", "point"),
    ("analysis/temporal_path_transport.rs", "lower_sampled", "ClaimDemand", "point"),
    ("analysis/tradeoff_decision_programs.rs", "compile_law", "TerminalSpec", "decision"),
    (
        "analysis/tradeoff_decision_programs.rs",
        "compile_tradeoff_structural_decision",
        "TerminalSpec",
        "decision",
    ),
    (
        "analysis/transport_assumption_program.rs",
        "compile_transport_assumption_range",
        "TerminalSpec",
        "bounds",
    ),
    ("analysis/transport_identification.rs", "lower", "ClaimDemand", "none"),
    ("analysis/transport_response_grid.rs", "lower", "PointProgram", "response"),
    (
        "analysis/trial_ipw_predictive_decision.rs",
        "compile_trial_ipw_predictive_decision_as",
        "ClaimDemand",
        "decision",
    ),
    (
        "analysis/trial_ipw_transport_decision.rs",
        "compile_trial_ipw_transport_decision_as",
        "ClaimDemand",
        "decision",
    ),
    (
        "analysis/z_transport_sensitivity_program.rs",
        "compile_z_transport_mechanism_sensitivity",
        "TerminalSpec",
        "bounds",
    ),
];

/// One construction that states a claim demand.
#[derive(Clone, Debug)]
struct Construction {
    /// File under `src/`, with `/` separators.
    file: String,
    /// Name of the enclosing function.
    function: String,
    /// The construction form (`ClaimDemand`, `TerminalSpec`, ...).
    form: &'static str,
    /// The demand's expression, whitespace and comments removed.
    kind: String,
    /// A terminal spec's `functional` expression, the same way.
    functional: String,
    /// Program cells named in the enclosing function up to the end of the construction.
    cells: Vec<String>,
}

const fn is_ident(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Skip a comment or string literal at `i`, returning where scanning resumes.
fn skip_literal(text: &[u8], i: usize) -> Option<usize> {
    if text[i..].starts_with(b"//") {
        return Some(text[i..].iter().position(|&b| b == b'\n').map_or(text.len(), |p| i + p));
    }
    if text[i..].starts_with(b"/*") {
        let end =
            text[i + 2..].windows(2).position(|w| w == b"*/").map_or(text.len(), |p| i + 2 + p + 2);
        return Some(end);
    }
    if text[i] == b'"' {
        let mut j = i + 1;
        while j < text.len() && text[j] != b'"' {
            j += if text[j] == b'\\' { 2 } else { 1 };
        }
        return Some(j + 1);
    }
    None
}

/// The expression starting at `i`, up to the next `,` or closing bracket at its own depth,
/// with whitespace and comments removed.
fn expression(text: &[u8], mut i: usize) -> String {
    let mut depth = 0usize;
    let mut out = Vec::new();
    while i < text.len() {
        let byte = text[i];
        if byte != b'"'
            && let Some(next) = skip_literal(text, i)
        {
            i = next;
            continue;
        }
        if byte == b'"' {
            let next = skip_literal(text, i).unwrap_or(text.len());
            out.extend_from_slice(&text[i..next.min(text.len())]);
            i = next;
            continue;
        }
        match byte {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' | b',' if depth == 0 => break,
            b')' | b']' | b'}' => depth -= 1,
            _ => {}
        }
        if !byte.is_ascii_whitespace() {
            out.push(byte);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The expression of `field` in the struct literal whose body starts at `start`, or
/// `<absent>` when the literal does not name it. A shorthand field is its own name.
fn literal_field(text: &[u8], start: usize, field: &str) -> String {
    let mut i = start;
    let mut depth = 0usize;
    let mut at_field_start = true;
    while i < text.len() {
        let byte = text[i];
        if let Some(next) = skip_literal(text, i) {
            if byte == b'"' {
                at_field_start = false;
            }
            i = next;
            continue;
        }
        if byte.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if depth == 0 && at_field_start && is_ident(byte) {
            let mut j = i;
            while j < text.len() && is_ident(text[j]) {
                j += 1;
            }
            let name = &text[i..j];
            let mut k = j;
            while k < text.len() && text[k].is_ascii_whitespace() {
                k += 1;
            }
            if name == field.as_bytes() && k < text.len() {
                if matches!(text[k], b',' | b'}') {
                    return field.to_owned();
                }
                if text[k] == b':' && text.get(k + 1) != Some(&b':') {
                    return expression(text, k + 1);
                }
            }
            at_field_start = false;
            i = j;
            continue;
        }
        match byte {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' if depth == 0 => return "<absent>".to_owned(),
            b')' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                at_field_start = true;
                i += 1;
                continue;
            }
            _ => {}
        }
        at_field_start = false;
        i += 1;
    }
    "<absent>".to_owned()
}

/// The last `fn name` before `end`, with its start.
fn enclosing_function(text: &[u8], end: usize) -> (String, usize) {
    let mut found = ("<none>".to_owned(), 0);
    let mut i = 0;
    while i + 2 < end {
        if &text[i..i + 2] == b"fn"
            && (i == 0 || !is_ident(text[i - 1]))
            && text[i + 2].is_ascii_whitespace()
        {
            let mut j = i + 2;
            while j < end && text[j].is_ascii_whitespace() {
                j += 1;
            }
            let k = (j..end).find(|&k| !is_ident(text[k])).unwrap_or(end);
            if k > j {
                found = (String::from_utf8_lossy(&text[j..k]).into_owned(), i);
            }
        }
        i += 1;
    }
    found
}

/// Program cells named as `ProgramCell::Name` in `text`.
fn program_cells(text: &[u8]) -> Vec<String> {
    const PREFIX: &[u8] = b"ProgramCell::";
    let mut cells = Vec::new();
    let mut i = 0;
    while let Some(at) = text[i..].windows(PREFIX.len()).position(|w| w == PREFIX) {
        let start = i + at + PREFIX.len();
        let end = (start..text.len()).find(|&k| !is_ident(text[k])).unwrap_or(text.len());
        cells.push(String::from_utf8_lossy(&text[start..end]).into_owned());
        i = end;
    }
    cells.sort();
    cells.dedup();
    cells
}

/// Every construction stating a claim demand in `text` (the file `file`).
fn constructions(file: &str, text: &str) -> Vec<Construction> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    for (form, field) in FORMS {
        let pattern = format!("{form} {{");
        let mut from = 0;
        while let Some(at) = text[from..].find(&pattern) {
            let index = from + at;
            from = index + 1;
            if index > 0 && is_ident(bytes[index - 1]) {
                continue;
            }
            let before = text[..index].trim_end().trim_end_matches('&').trim_end();
            if ["struct", "impl", "->", "for", "enum"].iter().any(|word| before.ends_with(word)) {
                continue;
            }
            let body = index + pattern.len();
            let (function, function_start) = enclosing_function(bytes, index);
            let end = (body + 600).min(bytes.len());
            out.push(Construction {
                file: file.to_owned(),
                function,
                form,
                kind: literal_field(bytes, body, field),
                functional: if form == "TerminalSpec" {
                    literal_field(bytes, body, "functional")
                } else {
                    String::new()
                },
                cells: program_cells(&bytes[function_start..end]),
            });
        }
    }
    out
}

fn sources(dir: &Path, root: &Path, out: &mut Vec<(String, String)>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
        .map(|entry| entry.expect("a directory entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            sources(&path, root, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let relative = path.strip_prefix(root).expect("under src");
            let name = relative
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            let text = std::fs::read_to_string(&path).expect("a UTF-8 source file");
            out.push((name, text));
        }
    }
}

fn src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every construction stating a claim demand in the crate.
fn scan() -> Vec<Construction> {
    let root = src();
    let mut files = Vec::new();
    sources(&root, &root, &mut files);
    // A route's demand is what its production code states: test modules and test files build
    // demands as fixtures and are not routes.
    files
        .iter()
        .filter(|(name, _)| !name.ends_with("tests.rs") && !name.contains("/tests/"))
        .flat_map(|(name, text)| {
            let production =
                text.find("\n#[cfg(test)]\nmod ").map_or(text.as_str(), |at| &text[..at]);
            constructions(name, production)
        })
        .collect()
}

/// The publication a demand expression states, in the vocabulary of [`ROUTES`].
fn publication(kind: &str) -> String {
    let kind = kind.replace("antecedent_contract::", "");
    let fixed = kind
        .strip_prefix("Some(")
        .and_then(|rest| rest.strip_suffix(')'))
        .filter(|inner| inner.starts_with("ClaimKind::") && inner[11..].bytes().all(is_ident))
        .unwrap_or(&kind);
    let named = match fixed {
        "ClaimKind::PolicyValue" => Some("decision"),
        "ClaimKind::Point" => Some("point"),
        "ClaimKind::Bounds" => Some("bounds"),
        "ClaimKind::Response" => Some("response"),
        "ClaimKind::Associational" => Some("association"),
        "ClaimKind::Mixture" => Some("mixture"),
        "None" => Some("none"),
        "spec.claim" | "Some(self.claim)" => Some("forwarded"),
        "status.map(claim_kind)" | "Some(cell.claim)" | "Some(kind)" => Some("sealed"),
        _ => None,
    };
    if let Some(named) = named {
        return named.to_owned();
    }
    let mut kinds: Vec<&str> = kind
        .match_indices("ClaimKind::")
        .map(|(at, _)| {
            let rest = &kind[at + 11..];
            &rest[..rest.bytes().position(|b| !is_ident(b)).unwrap_or(rest.len())]
        })
        .collect();
    kinds.sort_unstable();
    kinds.dedup();
    format!("derived:{}", kinds.join(","))
}

/// Route families by program cell, read from the generator table.
fn families() -> BTreeMap<String, String> {
    let text = std::fs::read_to_string(src().join("analysis/lowering/program_families.rs"))
        .expect("the program family table");
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        let Some((cell, rest)) = line.split_once(" => ") else { continue };
        let Some((family, _)) = rest.split_once(", unit") else { continue };
        if cell.bytes().all(is_ident) && family.bytes().all(is_ident) {
            out.insert(cell.to_owned(), family.to_owned());
        }
    }
    out
}

type Row = (String, String, String, String);

#[test]
fn every_claim_demand_states_the_kind_its_route_publishes() {
    let mut found: Vec<Row> = scan()
        .into_iter()
        .map(|c| (c.file, c.function, c.form.to_owned(), publication(&c.kind)))
        .collect();
    let mut declared: Vec<Row> = ROUTES
        .iter()
        .map(|&(file, function, form, publishes)| {
            (file.to_owned(), function.to_owned(), form.to_owned(), publishes.to_owned())
        })
        .collect();
    found.sort();
    declared.sort();
    let undeclared: Vec<&Row> = found.iter().filter(|row| !declared.contains(row)).collect();
    let stale: Vec<&Row> = declared.iter().filter(|row| !found.contains(row)).collect();
    assert!(
        undeclared.is_empty() && stale.is_empty() && found.len() == declared.len(),
        "claim demands that state another kind than their route publishes, or routes without a \
         declared publication:\n  in source, not declared: {undeclared:#?}\n  declared, not in \
         source: {stale:#?}"
    );
    assert!(found.len() >= 100, "the scan found only {} demands", found.len());
}

#[test]
fn a_terminal_program_demands_the_kind_its_functional_publishes() {
    let mut wrong = Vec::new();
    for c in scan().into_iter().filter(|c| c.form == "TerminalSpec") {
        let publication = publication(&c.kind);
        let functional = c.functional.rsplit("::").next().unwrap_or("").to_owned();
        let admitted: &[&str] = match functional.as_str() {
            // An identified set is bounds, never a point or a decision value.
            "IdentifiedSet" => &["bounds"],
            // A description of the observed law is never a causal point.
            "Association" => &["association", "bounds"],
            // A search, plan, chain adaptation or obligation list publishes no claim.
            "StructureSearch" | "PropagatedChain" | "InterventionalDistribution" => &["none"],
            "PolicyValue" | "RegimeValue" => &["decision", "point", "forwarded"],
            _ => &[],
        };
        if !admitted.contains(&publication.as_str())
            || (publication == "bounds"
                && functional != "IdentifiedSet"
                && functional != "Association")
        {
            wrong.push(format!(
                "{}::{}: {} demands {}",
                c.file, c.function, c.functional, publication
            ));
        }
    }
    assert!(wrong.is_empty(), "terminal demands that disagree with their functional: {wrong:#?}");
}

#[test]
fn an_associational_program_never_demands_a_causal_claim() {
    let families = families();
    assert!(families.len() >= 100, "the program family table was not read");
    let mut wrong = Vec::new();
    for c in scan() {
        let associational: Vec<&String> = c
            .cells
            .iter()
            .filter(|cell| families.get(*cell).is_some_and(|f| f == "Associational"))
            .collect();
        if associational.is_empty() {
            continue;
        }
        let publication = publication(&c.kind);
        if !["association", "bounds", "none"].contains(&publication.as_str()) {
            wrong.push(format!(
                "{}::{} ({associational:?}) demands {publication}",
                c.file, c.function
            ));
        }
    }
    assert!(wrong.is_empty(), "associational programs demanding a causal claim: {wrong:#?}");
}

#[test]
fn the_demand_scan_reads_every_construction_form() {
    let text = r#"
pub struct ClaimDemand { kind: Option<ClaimKind> }
impl ClaimDemand { fn kind(&self) {} }
fn signature() -> ClaimDemand { todo!() }
fn accessor(&self) -> &ClaimDemand { todo!() }
fn plain() {
    let a = ClaimDemand { uncertainty: UncertaintyReading::None, kind: Some(ClaimKind::Bounds) };
}
fn shorthand(kind: Option<ClaimKind>) {
    let b = ClaimDemand {
        uncertainty, // a kind: Some(ClaimKind::Point), in a comment
        kind,
    };
}
fn nested() {
    let c = ClaimDemand {
        uncertainty: UncertaintyReading::None,
        kind: Some(match target { Target { kind: Inner::Set, .. } => ClaimKind::Bounds, _ => ClaimKind::Response }),
    };
    let d = lower(TerminalSpec { cell: ProgramCell::Example, plan_id: "a{b,c}", functional: FunctionalKind::Association, claim: None });
}
"#;
    let found = constructions("example.rs", text);
    let summary: Vec<(String, &str, String)> =
        found.iter().map(|c| (c.function.clone(), c.form, publication(&c.kind))).collect();
    assert_eq!(
        summary,
        vec![
            ("plain".to_owned(), "ClaimDemand", "bounds".to_owned()),
            ("shorthand".to_owned(), "ClaimDemand", "derived:".to_owned()),
            ("nested".to_owned(), "ClaimDemand", "derived:Bounds,Response".to_owned()),
            ("nested".to_owned(), "TerminalSpec", "none".to_owned()),
        ]
    );
    assert_eq!(found[3].functional, "FunctionalKind::Association");
    assert_eq!(found[3].cells, vec!["Example".to_owned()]);
    // A demand edited to another kind is another publication.
    assert_eq!(publication("Some(ClaimKind::Point)"), "point");
    assert_eq!(publication("Some(antecedent_contract::ClaimKind::PolicyValue)"), "decision");
    assert_eq!(publication("ClaimKind::Mixture"), "mixture");
}
