//! Hedge certificates for non-identifiability.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::NodeRef;
use antecedent_core::VariableId;
use antecedent_graph::{Admg, BitSet, DenseNodeId, GraphWorkspace};

use crate::error::IdentificationError;
use crate::prepared::PreparedAdmg;

/// The problem a hedge was found in: the ADMG (variable per dense node, directed and
/// bidirected edges by dense id) and the treatments and outcomes, so the certificate can be
/// re-verified by whoever holds it, with no other input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HedgeProblem {
    /// Variable of each dense node, in dense order.
    pub variables: Arc<[VariableId]>,
    /// Directed edges `(parent, child)` by dense id, sorted.
    pub directed: Arc<[(u32, u32)]>,
    /// Bidirected edges `(low, high)` by dense id, sorted.
    pub bidirected: Arc<[(u32, u32)]>,
    /// Treatments `X`.
    pub treatments: Arc<[VariableId]>,
    /// Outcomes `Y`.
    pub outcomes: Arc<[VariableId]>,
}

impl HedgeProblem {
    /// Capture the prepared graph and the treatment / outcome sets.
    ///
    /// # Errors
    ///
    /// A node without a variable identity.
    pub fn capture(
        prepared: &PreparedAdmg,
        x: &BitSet,
        y: &BitSet,
    ) -> Result<Self, IdentificationError> {
        let admg = prepared.admg();
        let n = admg.node_count();
        let mut variables = Vec::with_capacity(n);
        let mut directed = Vec::new();
        let mut bidirected = Vec::new();
        for i in 0..n {
            let id = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
            variables.push(prepared.dense_to_var(id)?);
            directed.extend(admg.children(id).iter().map(|c| (id.raw(), c.raw())));
            bidirected.extend(
                admg.bidirected_neighbors(id)
                    .iter()
                    .filter(|b| b.raw() > id.raw())
                    .map(|b| (id.raw(), b.raw())),
            );
        }
        directed.sort_unstable();
        bidirected.sort_unstable();
        let vars_of = |set: &BitSet| -> Result<Arc<[VariableId]>, IdentificationError> {
            set.to_dense_ids().into_iter().map(|d| prepared.dense_to_var(d)).collect()
        };
        Ok(Self {
            variables: variables.into(),
            directed: directed.into(),
            bidirected: bidirected.into(),
            treatments: vars_of(x)?,
            outcomes: vars_of(y)?,
        })
    }

    /// Rebuild the prepared ADMG.
    ///
    /// # Errors
    ///
    /// Duplicate variables, out-of-range edges, or a directed cycle.
    pub fn prepare(&self) -> Result<PreparedAdmg, IdentificationError> {
        let mut admg = Admg::empty();
        for &v in self.variables.iter() {
            admg.add_node(NodeRef::Static(v))?;
        }
        for &(a, b) in self.directed.iter() {
            admg.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))?;
        }
        for &(a, b) in self.bidirected.iter() {
            admg.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))?;
        }
        PreparedAdmg::new(admg)
    }
}

/// Witness that `P(Y | do(X))` is not identifiable (Shpitser & Pearl 2006 hedge).
///
/// Recovered from ID line 5: the pair `(F, F')` of R-rooted C-forests where
/// `F` is the current subgraph `G` and `F'` is the C-component `S` of `G[V\X]`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HedgeCertificate {
    /// Nodes of the larger C-forest `F` (current active subgraph).
    pub f: Arc<[VariableId]>,
    /// Nodes of the smaller C-forest `F' ⊆ F` (C-component of `G[V\X]`).
    pub f_prime: Arc<[VariableId]>,
    /// Dense ids of `F` (stable for diagnostics).
    pub f_dense: Arc<[DenseNodeId]>,
    /// Dense ids of `F'`.
    pub f_prime_dense: Arc<[DenseNodeId]>,
    /// The problem the hedge was found in, when the producer had it; makes the certificate
    /// re-verifiable by [`Self::verify_carried`] wherever it travels.
    pub problem: Option<HedgeProblem>,
}

impl HedgeCertificate {
    /// Build a certificate from dense node sets and a variable map.
    #[must_use]
    pub fn from_sets(
        f: &BitSet,
        f_prime: &BitSet,
        dense_to_var: impl Fn(DenseNodeId) -> VariableId,
    ) -> Self {
        let f_dense = f.to_dense_ids();
        let f_prime_dense = f_prime.to_dense_ids();
        let f_vars: Vec<VariableId> = f_dense.iter().copied().map(&dense_to_var).collect();
        let fp_vars: Vec<VariableId> = f_prime_dense.iter().copied().map(&dense_to_var).collect();
        Self {
            f: Arc::from(f_vars),
            f_prime: Arc::from(fp_vars),
            f_dense: Arc::from(f_dense),
            f_prime_dense: Arc::from(f_prime_dense),
            problem: None,
        }
    }

    /// Attach the problem (graph, treatments, outcomes) this hedge witnesses.
    #[must_use]
    pub fn with_problem(mut self, problem: HedgeProblem) -> Self {
        self.problem = Some(problem);
        self
    }

    /// [`Self::verify`] against the carried problem.
    ///
    /// # Errors
    ///
    /// No problem is carried, the carried graph is invalid, or a hedge condition fails.
    pub fn verify_carried(&self) -> Result<(), IdentificationError> {
        let problem = self.problem.as_ref().ok_or_else(|| {
            IdentificationError::msg("hedge certificate carries no graph to verify against")
        })?;
        self.verify(&problem.prepare()?, &problem.treatments, &problem.outcomes)
    }

    /// Check that `(F, F')` is a hedge for `P(outcomes | do(treatments))` in
    /// the prepared graph, straight from the definition (Shpitser & Pearl
    /// 2006, Def. 6) and independently of how the certificate was produced.
    ///
    /// A hedge is a pair of `R`-rooted C-forests `F' ⊆ F` with
    /// `F ∩ X ≠ ∅`, `F' ∩ X = ∅` and `R ⊆ An(Y)` in `G` with edges into `X`
    /// removed. The certificate names node sets, so the check is existential
    /// over edge subsets: an induced subgraph on `N` contains an `R`-rooted
    /// C-forest spanning `N` exactly when `N` is bidirected-connected and
    /// every node of `N` reaches `R` by directed edges inside `N` (keep a
    /// bidirected spanning tree and, for each non-root, one edge of a shortest
    /// directed path to `R`). The largest admissible root set,
    /// `R = F' ∩ An(Y)_{G_{\bar X}}`, is used: ancestral closure is monotone in
    /// `R`, so if any admissible root set works this one does.
    ///
    /// # Errors
    ///
    /// Unknown variables, or a message naming the violated hedge condition.
    pub fn verify(
        &self,
        prepared: &PreparedAdmg,
        treatments: &[VariableId],
        outcomes: &[VariableId],
    ) -> Result<(), IdentificationError> {
        let fail = |what: &str| Err(IdentificationError::msg(format!("not a hedge: {what}")));
        let n = prepared.admg().node_count();
        let to_set = |dense: &[DenseNodeId], vars: &[VariableId]| {
            if dense.len() != vars.len() {
                return None;
            }
            let mut set = BitSet::with_len(n);
            for (&node, &var) in dense.iter().zip(vars) {
                if node.as_usize() >= n || prepared.dense_to_var(node).ok()? != var {
                    return None;
                }
                set.insert(node);
            }
            Some(set)
        };
        let (Some(f), Some(f_prime)) =
            (to_set(&self.f_dense, &self.f), to_set(&self.f_prime_dense, &self.f_prime))
        else {
            return fail("certificate nodes do not belong to this graph");
        };
        let mut x = BitSet::with_len(n);
        for &t in treatments {
            x.insert(prepared.var_to_dense(t)?);
        }
        let mut y = BitSet::with_len(n);
        for &o in outcomes {
            y.insert(prepared.var_to_dense(o)?);
        }
        if !f_prime.any() || !f_prime.is_subset_of(&f) {
            return fail("F' must be a non-empty subset of F");
        }
        if !f.to_dense_ids().iter().any(|node| x.contains(*node)) {
            return fail("F does not meet the treatments");
        }
        if f_prime.to_dense_ids().iter().any(|node| x.contains(*node)) {
            return fail("F' meets the treatments");
        }
        if !prepared.is_single_c_component(&f) || !prepared.is_single_c_component(&f_prime) {
            return fail("F and F' must each be bidirected-connected");
        }
        let mut ws = GraphWorkspace::default();
        let mut all = BitSet::with_len(n);
        for node in 0..n {
            all.insert(DenseNodeId::from_raw(u32::try_from(node).expect("fit")));
        }
        let mut roots = prepared.ancestors_bar_x(&y, &all, &x, &mut ws);
        roots.intersect_with(&f_prime);
        if !roots.any() {
            return fail("no node of F' is an ancestor of the outcomes once edges into X are cut");
        }
        let none = BitSet::with_len(n);
        if !prepared.ancestors_bar_x(&roots, &f, &none, &mut ws).equal_set(&f)
            || !prepared.ancestors_bar_x(&roots, &f_prime, &none, &mut ws).equal_set(&f_prime)
        {
            return fail("F and F' are not both forests rooted in the same set");
        }
        Ok(())
    }
}
