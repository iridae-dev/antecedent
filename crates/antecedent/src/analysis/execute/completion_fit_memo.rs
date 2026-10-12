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

/// Fits already run by one execution, keyed by [`CompletionFitKey`].
///
/// Linear search: an envelope holds tens of completions, and a hit replaces a
/// regression.
#[derive(Debug)]
pub(super) struct CompletionFitMemo<V> {
    fits: Vec<(CompletionFitKey, V)>,
}

impl<V> Default for CompletionFitMemo<V> {
    fn default() -> Self {
        Self { fits: Vec::new() }
    }
}

impl<V> CompletionFitMemo<V> {
    /// The value an earlier completion fitted for `key`, if any.
    pub(super) fn get(&self, key: &CompletionFitKey) -> Option<&V> {
        self.fits.iter().find(|(known, _)| known == key).map(|(_, value)| value)
    }

    /// Record the value fitted for `key`.
    ///
    /// A failed fit is never recorded: the caller propagates the error. Taking
    /// the fit as two steps (look up, then record) keeps each estimator call
    /// in its caller's frame, where the debug-build stack already runs close
    /// to the default thread's limit.
    pub(super) fn insert(&mut self, key: CompletionFitKey, value: V) {
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
}
