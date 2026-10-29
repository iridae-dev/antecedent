// SPDX-License-Identifier: MIT OR Apache-2.0
//! A time-varying mechanism declared on a static (no-carryover) joint panel is refused with a
//! typed reason under every inference, never estimated as a static contrast.
//!
//! The static specialisation of the joint panel (`lag_order = 0`) executes as a static contrast
//! of the fitted panel (the control below). A path declaration (`temporal`) asks for a
//! mechanism that changes over the observed periods, which a static specialisation has no
//! lags or histories to express, so the specification is refused at preparation rather than
//! ignored.
#[path = "../../../examples/rust/joint_panel_fixture.rs"]
mod fixture;
use antecedent::{
    Study,
    analysis::joint_panel::{
        JointPanelError, JointPanelFunctional, JointPanelRequest, joint_panel_target,
    },
};
use antecedent_core::{ExecutionContext, InferenceSelection, ScientificWorldId};
use antecedent_estimate::joint_panel::{
    PanelError, PanelInference, PanelPrior,
    time_varying::{PanelMeanParameter, TemporalPanelSpec},
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(20_261_201)
}

fn spec() -> TemporalPanelSpec {
    TemporalPanelSpec {
        varying: vec![PanelMeanParameter::Interaction { id: "ab".into() }],
        change_points: vec![8, 12, 16],
        minimum_segment: 4,
        weak_change_gap: 4.0,
        smooth_knots: vec![0, 8, 16, 23],
        smoothing: vec![1.0],
        path_prior_sd: 1.0,
    }
}

/// The static request on the fixture panel: no carryover, horizon zero.
fn static_request(
    data: &antecedent_data::PanelData,
    inference: PanelInference,
    temporal: Option<TemporalPanelSpec>,
) -> JointPanelRequest {
    let selection = if matches!(&inference, PanelInference::Bayesian { .. }) {
        InferenceSelection::Bayesian
    } else {
        InferenceSelection::Frequentist
    };
    let mut request = fixture::request(data, false, inference);
    request.lag_order = 0;
    request.target = joint_panel_target(
        data.schema(),
        &request.columns,
        0,
        &request.worlds,
        JointPanelFunctional::Interaction {
            joint: ScientificWorldId::from_raw(3),
            a_at_reference: ScientificWorldId::from_raw(1),
            b_at_reference: ScientificWorldId::from_raw(2),
            reference: ScientificWorldId::from_raw(0),
        },
        selection,
    )
    .unwrap();
    request.temporal = temporal;
    request
}

fn inferences() -> Vec<(&'static str, PanelInference)> {
    vec![
        ("frequentist", PanelInference::Frequentist { draws: 2 }),
        ("bootstrap", PanelInference::FrequentistBootstrap { draws: 2 }),
        (
            "bayesian",
            PanelInference::Bayesian {
                prior: PanelPrior::default(),
                sampler: fixture::posterior_sampling(),
            },
        ),
    ]
}

#[test]
fn unavailable_b_static_panel_without_a_path_prepares_and_with_a_path_is_refused() {
    let data = fixture::source_static();
    for (label, inference) in inferences() {
        // Control: the same static panel with no time-varying declaration prepares.
        Study::prepare_joint_panel(
            data.clone(),
            static_request(&data, inference.clone(), None),
            &ctx(),
        )
        .unwrap_or_else(|error| panic!("{label}: the static control must prepare: {error}"));
        let refused = Study::prepare_joint_panel(
            data.clone(),
            static_request(&data, inference, Some(spec())),
            &ctx(),
        );
        let error = match refused {
            Ok(_) => panic!("{label}: a time-varying path on a static panel was accepted"),
            Err(error) => error,
        };
        assert!(
            matches!(&error, JointPanelError::Engine(PanelError::Invalid(message))
                if message.contains("temporal candidate windows")),
            "{label}: {error}"
        );
    }
}
