//! Memo of the response fits of one envelope execution.
//!
//! The completions of a class envelope very often share an adjustment set: the
//! envelope lists graphs, not adjustment sets, and many completions are
//! adjusted by the same covariates. A response fit reads the table, the query,
//! the estimator options, the inference mode and the context seed, all of which
//! are fixed for the execution that owns a memo, and, per completion, its
//! adjustment set, its identification status and its required assumptions.
//! Two completions that agree on those three therefore run the same
//! deterministic fit, and the second can be served the first's value instead of
//! refitting.
//!
//! The key is the whole of what varies. Nothing else a fit reads changes
//! within one execution, so a memo must not outlive one: it is created by the
//! executor that owns the fixed inputs and dropped with it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AssumptionSet, IdentificationStatus, VariableId};
use std::sync::Arc;

/// What distinguishes the response fits of one execution.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct CompletionFitKey {
    adjustment_set: Arc<[VariableId]>,
    status: IdentificationStatus,
    assumptions: AssumptionSet,
}

impl CompletionFitKey {
    pub(super) fn new(
        adjustment_set: &Arc<[VariableId]>,
        status: IdentificationStatus,
        assumptions: &AssumptionSet,
    ) -> Self {
        Self {
            adjustment_set: Arc::clone(adjustment_set),
            status,
            assumptions: assumptions.clone(),
        }
    }
}

/// What distinguishes the temporal response fits of one horizon of one
/// execution: the fit's whole per-completion input.
///
/// The temporal response estimators are handed the identified estimand and
/// its indexer as the only per-completion identification state (never the
/// proof arena or the graph), plus the status and the assumptions they stamp
/// on the response. All four are compared in full, so two completions share a
/// fit exactly when the fit was given the same inputs; the table, the
/// horizon's query, the inference mode and the seed are fixed per horizon.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ResponseFitKey {
    estimand: antecedent_expr::IdentifiedEstimand,
    indexer: antecedent_core::TemporalIndexer,
    status: IdentificationStatus,
    assumptions: AssumptionSet,
}

impl ResponseFitKey {
    pub(super) fn new(
        estimand: &antecedent_expr::IdentifiedEstimand,
        indexer: &antecedent_core::TemporalIndexer,
        status: IdentificationStatus,
        assumptions: &AssumptionSet,
    ) -> Self {
        Self {
            estimand: estimand.clone(),
            indexer: indexer.clone(),
            status,
            assumptions: assumptions.clone(),
        }
    }
}

/// What distinguishes the mediation fits of one horizon of one execution.
///
/// The temporal mediation fits read the estimand only through its method
/// (which licenses the route and is recorded on the Bayesian problem) and its
/// mediators, and the adjustment only through the lag-aligned columns derived
/// from its adjustment set and the completion's indexer. The data, the
/// horizon query, the inference mode and the seed are fixed per horizon. The
/// status is the one thing a completion adds to the fit itself.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct MediationFitKey {
    method: Arc<str>,
    mediators: Arc<[VariableId]>,
    adjustment: Arc<[antecedent_data::LaggedColumn]>,
    status: IdentificationStatus,
}

impl MediationFitKey {
    pub(super) fn new(
        estimand: &antecedent_expr::IdentifiedEstimand,
        adjustment: &Arc<[antecedent_data::LaggedColumn]>,
        status: IdentificationStatus,
    ) -> Self {
        Self {
            method: Arc::clone(&estimand.method),
            mediators: Arc::clone(&estimand.mediators),
            adjustment: Arc::clone(adjustment),
            status,
        }
    }
}

/// Fits already run by one execution, keyed by [`CompletionFitKey`] (or the
/// key type `K` of another fit family).
///
/// Linear search: an envelope holds tens of completions, and a hit replaces a
/// regression.
#[derive(Debug)]
pub(super) struct CompletionFitMemo<V, K = CompletionFitKey> {
    fits: Vec<(K, V)>,
}

impl<V, K> Default for CompletionFitMemo<V, K> {
    fn default() -> Self {
        Self { fits: Vec::new() }
    }
}

impl<V, K: PartialEq> CompletionFitMemo<V, K> {
    /// The value an earlier completion fitted for `key`, if any.
    pub(super) fn get(&self, key: &K) -> Option<&V> {
        self.fits.iter().find(|(known, _)| known == key).map(|(_, value)| value)
    }

    /// Record the value fitted for `key`.
    ///
    /// A failed fit is never recorded: the caller propagates the error. Taking
    /// the fit as two steps (look up, then record) keeps each estimator call
    /// in its caller's frame, where the debug-build stack already runs close
    /// to the default thread's limit.
    pub(super) fn insert(&mut self, key: K, value: V) {
        #[cfg(test)]
        FITS_RUN.with(|count| count.set(count.get() + 1));
        self.fits.push((key, value));
    }
}

#[cfg(test)]
thread_local! {
    static FITS_RUN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Response fits run through a memo on this thread since it started.
#[cfg(test)]
pub(crate) fn fits_run() -> usize {
    FITS_RUN.with(std::cell::Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::Lag;

    fn key(set: &[u32], status: IdentificationStatus) -> CompletionFitKey {
        let set: Arc<[VariableId]> = set.iter().copied().map(VariableId::from_raw).collect();
        CompletionFitKey::new(&set, status, &AssumptionSet::default())
    }

    #[test]
    fn an_identical_completion_is_served_and_a_different_one_is_fitted() {
        let mut memo = CompletionFitMemo::<u32>::default();
        let mut runs = 0;
        let mut fit = |memo: &mut CompletionFitMemo<u32>, k: CompletionFitKey, value: u32| {
            if let Some(known) = memo.get(&k) {
                return *known;
            }
            runs += 1;
            memo.insert(k, value);
            value
        };
        let ok = IdentificationStatus::NonparametricallyIdentified;
        assert_eq!(fit(&mut memo, key(&[1, 2], ok), 10), 10);
        assert_eq!(fit(&mut memo, key(&[1, 2], ok), 99), 10, "same inputs reuse the first fit");
        assert_eq!(fit(&mut memo, key(&[1], ok), 20), 20, "another adjustment set refits");
        assert_eq!(
            fit(&mut memo, key(&[1, 2], IdentificationStatus::PartiallyIdentified), 30),
            30,
            "another status refits"
        );
        assert_eq!(runs, 3);
    }

    /// `z — t -> y`: both completions of the class adjust nothing, so the
    /// envelope's two completions are one fit.
    #[test]
    fn completions_sharing_an_adjustment_set_run_one_class_response_fit() {
        use crate::Study;
        use antecedent_core::{
            CausalQuery, ExecutionContext, Intervention, ResponseFunctional, ResponseQuery, Value,
        };
        use antecedent_data::TabularData;
        use antecedent_graph::{Cpdag, DenseNodeId};

        let n = 400;
        let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let zi = (f64::from(i) * 0.37).sin();
            let ti = 0.6 * zi + (f64::from(i) * 1.13).cos();
            z.push(zi);
            t.push(ti);
            y.push(1.5 * ti + 0.2 * (f64::from(i) * 0.71).sin());
        }
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap();
        let node = DenseNodeId::from_raw;
        let mut graph = Cpdag::with_variables(3);
        graph.insert_undirected(node(2), node(0)).unwrap();
        graph.insert_directed(node(0), node(1)).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
        });
        let study = Study::tabular(data)
            .graph(graph)
            .query(CausalQuery::Response(query))
            .refute(crate::RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let ctx = ExecutionContext::for_tests(11);
        // The debug frame of an envelope execution exceeds the default test
        // thread's stack.
        let (completions, fits) = std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(move || {
                let before = fits_run();
                let result = study.run(&ctx).unwrap();
                let atoms = result.structural_response.as_ref().unwrap().atoms.clone();
                (atoms, fits_run() - before)
            })
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(completions.len(), 2, "the class has two completions");
        assert_eq!(fits, 1, "completions with one adjustment set fit once");
        let values: Vec<_> = completions.iter().map(|atom| atom.value.clone()).collect();
        assert!(values[0].is_some() && values[0] == values[1]);
    }

    /// A `TemporalCpdag` whose completions share every fit input: `t@1 -> y`,
    /// `z@1 -> y`, and an undirected contemporaneous `z — w` between two
    /// variables that touch neither the treatment nor the outcome, so both
    /// orientations adjust and fit identically.
    fn two_completion_temporal_cpdag() -> antecedent_graph::TemporalCpdag {
        let mut graph = antecedent_graph::TemporalCpdag::empty();
        let t0 = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let z0 = graph.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        let z1 = graph.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
        let w0 = graph.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
        let _ = t0;
        graph.insert_directed(t1, y0).unwrap();
        graph.insert_directed(z1, y0).unwrap();
        graph.insert_undirected(z0, w0).unwrap();
        graph
    }

    fn txyz_series(shift: f64, n: usize) -> antecedent_data::TimeSeriesData {
        let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let w: Vec<f64> = (0..n).map(|i| (i as f64 * 0.91).cos()).collect();
        for i in 0..n {
            let phase = i as f64 * 0.37;
            z[i] = phase.sin();
            t[i] = f64::from(u8::from(phase.cos() > 0.0));
            if i > 0 {
                y[i] = shift + 1.4 * t[i - 1] + 0.6 * z[i - 1] + 0.1 * (phase * 1.7).sin();
            }
        }
        antecedent_data::TimeSeriesData::from_f64_columns(
            [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice()), ("w", w.as_slice())],
            1,
        )
        .unwrap()
    }

    /// Run `body` on a thread with the stack an envelope execution needs in a
    /// debug build, returning how many fits the memos ran.
    fn fits_during<R: Send + 'static>(body: impl FnOnce() -> R + Send + 'static) -> (R, usize) {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(move || {
                let before = fits_run();
                let out = body();
                (out, fits_run() - before)
            })
            .unwrap()
            .join()
            .unwrap()
    }

    #[test]
    fn temporal_class_response_completions_with_one_input_run_one_fit() {
        use crate::Study;
        use antecedent_core::{
            CausalQuery, ExecutionContext, Intervention, ResponseFunctional, ResponseQuery,
            TemporalPolicy, TemporalResponseSpec, Value,
        };
        let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: VariableId::from_raw(1),
            interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
        })
        .with_temporal(TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap());
        let study = Study::series(txyz_series(0.0, 60))
            .graph(two_completion_temporal_cpdag())
            .query(CausalQuery::Response(query))
            .refute(crate::RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let (atoms, fits) = fits_during(move || {
            let result = study.run(&ExecutionContext::for_tests(3)).unwrap();
            result.structural_response.as_ref().unwrap().atoms.clone()
        });
        assert_eq!(atoms.len(), 2, "the class has two completions");
        assert_eq!(fits, 1, "completions given one estimand and indexer fit once");
        assert!(atoms[0].value.is_some() && atoms[0].value == atoms[1].value);
    }

    #[test]
    fn temporal_class_mediation_completions_with_one_design_run_one_fit() {
        use crate::Study;
        use antecedent_core::{CausalQuery, ExecutionContext, MediationContrast, MediationQuery};
        let n = 120;
        let (mut t, mut m, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let w: Vec<f64> = (0..n).map(|i| (i as f64 * 0.91).cos()).collect();
        for i in 0..n {
            z[i] = if i % 4 < 2 { 1.0 } else { -1.0 };
            t[i] = z[i] + if i % 4 == 0 || i % 4 == 2 { 1.0 } else { -1.0 };
        }
        for i in 1..n {
            let noise = (i % 5) as f64 - 2.0;
            m[i] = 0.8 * t[i - 1] + 0.6 * z[i - 1] + 0.15 * noise;
            y[i] = 1.0 + 0.25 * t[i - 1] + 0.55 * m[i] + 5.0 * z[i - 1];
        }
        let data = antecedent_data::TimeSeriesData::from_f64_columns(
            [
                ("t", t.as_slice()),
                ("m", m.as_slice()),
                ("y", y.as_slice()),
                ("z", z.as_slice()),
                ("w", w.as_slice()),
            ],
            1,
        )
        .unwrap();
        let mut graph = antecedent_graph::TemporalCpdag::empty();
        graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let m0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let y0 = graph.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        let z0 = graph.add_lagged(VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
        let z1 = graph.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
        for (from, to) in [(z1, y0), (z1, m0), (t1, y0), (t1, m0), (m0, y0)] {
            graph.insert_directed(from, to).unwrap();
        }
        let w0 = graph.add_lagged(VariableId::from_raw(4), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_undirected(z0, w0).unwrap();
        let query = MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        )
        .with_horizons(vec![1])
        .unwrap();
        for inference in [
            crate::InferenceMode::Frequentist,
            crate::InferenceMode::Bayesian(crate::BayesianConfig::conjugate().n_draws(64)),
        ] {
            let study = Study::series(data.clone())
                .graph(graph.clone())
                .query(CausalQuery::Mediation(query.clone()))
                .inference(inference.clone())
                .refute(crate::RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            let (atoms, fits) = fits_during(move || {
                let result = study.run(&ExecutionContext::for_tests(4)).unwrap();
                result.structural_response.as_ref().map(|s| s.atoms.len())
            });
            assert_eq!(atoms, Some(2), "the class has two completions");
            assert_eq!(fits, 1, "{inference:?}: both completions fit one mediation design once");
        }
    }

    #[test]
    fn panel_class_response_completions_with_one_input_fit_the_units_once() {
        use crate::Study;
        use antecedent_core::{
            CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional,
            ResponseQuery, TemporalPolicy, TemporalResponseSpec,
        };
        use antecedent_data::{PanelData, PanelUnit};
        let panel = PanelData::try_new(Arc::from([
            PanelUnit { unit_id: 0, series: txyz_series(0.1, 60) },
            PanelUnit { unit_id: 1, series: txyz_series(0.4, 60) },
            PanelUnit { unit_id: 2, series: txyz_series(0.7, 60) },
        ]))
        .unwrap();
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0, 1.0])),
            ),
        })
        .with_temporal(TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap());
        let study = Study::panel(panel)
            .graph(two_completion_temporal_cpdag())
            .query(CausalQuery::Response(query))
            .refute(crate::RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let (atoms, fits) = fits_during(move || {
            let result = study.run(&ExecutionContext::for_tests(5)).unwrap();
            result.structural_response.as_ref().unwrap().atoms.len()
        });
        assert_eq!(atoms, 2, "the class has two completions");
        assert_eq!(fits, 1, "completions given one estimand and indexer fit the units once");
    }
}
