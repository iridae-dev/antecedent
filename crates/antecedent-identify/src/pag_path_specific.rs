//! Path-restricted natural effects over the ADMGs a PAG leaves open.
//!
//! # Why the MAG is not enough
//!
//! A PAG describes a Markov equivalence class of maximal ancestral graphs. A MAG is a
//! summary of one latent projection only up to m-separation, and a path-specific effect is
//! not a function of the m-separation model: it is a functional of the structural equations,
//! which see the bidirected edges and the directed edges of the projection itself. Two
//! ADMGs with the same MAG can disagree on whether the effect is identified at all.
//! `T -> M -> Y, T -> Y` and the same graph with `M <-> Y` added both have the complete
//! tournament as their MAG (an ancestral graph cannot carry a confounder between a node and
//! its descendant, so the bidirected edge is absorbed into `M -> Y`), yet the effect along
//! `T -> M -> Y` is identified without the confounder and has a recanting district with it.
//! Running the district formula on one MAG completion therefore answers for one of the
//! ADMGs and silently for none of the others.
//!
//! # The ADMGs a MAG stands for
//!
//! Let `G` be the latent projection of any structural model the MAG `M` represents, so that
//! the MAG of `G` is `M` (Richardson & Spirtes 2002: `a` and `b` are adjacent iff no set
//! inside `An(a) ∪ An(b)` m-separates them, and the mark at `b` is a tail iff `a ∈ An(b)`).
//! Then
//!
//! * a directed edge `a -> b` of `G` is a directed edge of `M` (it makes `a` an ancestor of
//!   `b` and the pair adjacent);
//! * a bidirected edge `a <-> b` of `G` joins an `M`-adjacent pair, and if that pair is a
//!   directed edge of `M` the edge is invisible (Zhang 2008, Lemma 9: a visible edge excludes
//!   a latent common cause);
//! * a pair that is not adjacent in `M` carries no edge in `G`.
//!
//! So `G` is an edge-subgraph of the dominating ADMG that keeps every edge of `M` and adds
//! `a <-> b` beside each invisible `a -> b`. Adjacency and ancestry only grow when edges are
//! added, which makes the set of subgraphs whose MAG is exactly `M` searchable by
//! backtracking over the pool: an undecided edge can still add adjacency and ancestry but
//! never remove it, so a branch is cut as soon as the graph with every undecided edge
//! included cannot reach a required adjacency or ancestry of `M`, or the graph with none of
//! them already has an adjacency or ancestry `M` forbids. The members are the ADMGs `M`
//! cannot tell apart, and the latent projection of the true model is one of them.
//!
//! # What the route reports
//!
//! The path-specific identifier decides every member on its own graph, because which edges
//! start a selected path, which districts recant and which kernels hedge are all properties
//! of that graph. The completion is read as follows.
//!
//! * One member, or members that all yield the same functional (same canonical expression),
//!   identify the effect by that functional: the PAG determines the effect even though it
//!   may not determine the graph. A PAG whose MAG has only visible edges, no redundant
//!   edge and no circle mark has exactly one member, the MAG itself, and the district
//!   formula on it is the answer.
//! * Members that identify different functionals leave the effect an identified set: one
//!   case per distinct functional, estimated separately and published as the set.
//! * A member the identifier refuses (a recanting district, a hedge, a witness) makes the
//!   completion unidentified. No functional is correct for every model the MAG represents,
//!   so the completion carries unidentified mass and the route refuses rather than report a
//!   point.
//! * A search that stops at its bound has not decided the members. The completion is
//!   refused with a search-bounded diagnostic, never read as a point.
//!
//! Circle marks add the other layer of ambiguity: every MAG completion of the PAG is read
//! this way, and completions that disagree are an identified set across completions in the
//! same envelope.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity};
use antecedent_graph::{Admg, BitSet, DSeparationWorkspace, DenseNodeId, Endpoint, Pag};

use crate::envelope::{carries_identified_mass, GraphFeature, IdentificationEnvelope};
use crate::error::IdentificationError;
use crate::generalized::{invisible_directed_edges, GeneralizedAdjustmentIdentifier};
use crate::identifier::IdentificationWorkspace;
use crate::path_specific::PathSpecificIdentifier;
use crate::result::IdentificationResult;
use std::sync::Arc;

/// Diagnostic code of a completion whose ADMG candidates could not all be enumerated within
/// the search bound. The refusal it accompanies is a bound, not a statement about the graph.
pub const PAG_ADMG_SEARCH_BOUNDED_DIAGNOSTIC_CODE: &str =
    "identify.path_specific_pag.search_bounded";

/// Backtracking states the candidate search may visit for one MAG.
const MAX_CANDIDATE_STATES: usize = 1 << 15;
/// ADMGs the candidate search may collect for one MAG.
const MAX_CANDIDATE_MEMBERS: usize = 1 << 9;

/// The ADMGs a circle-free MAG cannot distinguish from the latent projection of the model.
#[derive(Clone, Debug)]
pub struct AdmgCandidates {
    /// Every ADMG whose MAG is the given MAG (node labels preserved).
    pub members: Vec<Admg>,
    /// The search stopped at its bound; `members` is then incomplete and must not be read as
    /// the full set.
    pub bounded: bool,
}

/// Relation of an adjacent pair in the MAG, with the lower dense index first.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MagPair {
    Absent,
    /// Lower index is the tail: `lo -> hi`.
    Forward,
    /// Higher index is the tail: `hi -> lo`.
    Backward,
    Bidirected,
}

/// Adjacency and ancestry of one pair in a candidate ADMG.
#[derive(Clone, Copy, Debug, Default)]
struct PairInfo {
    adjacent: bool,
    lo_anc_hi: bool,
    hi_anc_lo: bool,
}

impl MagPair {
    /// Whether a graph that can still gain edges can reach this pair: `info` is the graph
    /// with every undecided edge included.
    fn reachable_from(self, info: PairInfo) -> bool {
        match self {
            Self::Absent => true,
            Self::Forward => info.adjacent && info.lo_anc_hi,
            Self::Backward => info.adjacent && info.hi_anc_lo,
            Self::Bidirected => info.adjacent,
        }
    }

    /// Whether the pair is still consistent in a graph that can only gain edges: `info` is
    /// the graph with no undecided edge included.
    fn admits_from_below(self, info: PairInfo) -> bool {
        match self {
            Self::Absent => !info.adjacent,
            Self::Forward => !info.hi_anc_lo,
            Self::Backward => !info.lo_anc_hi,
            Self::Bidirected => !info.lo_anc_hi && !info.hi_anc_lo,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum PoolEdge {
    Directed(DenseNodeId, DenseNodeId),
    Bidirected(DenseNodeId, DenseNodeId),
}

/// Adjacency and ancestry of every pair `i < j` (index `i * n + j`) of `admg`.
///
/// Adjacent means an inducing path exists, which Richardson & Spirtes (Cor. 5.3) decide by
/// one m-separation test given `(An(a) ∪ An(b)) \ {a, b}`.
fn pair_infos(
    admg: &Admg,
    ws: &mut DSeparationWorkspace,
) -> Result<Vec<PairInfo>, IdentificationError> {
    let n = admg.node_count();
    let ancestors = admg.strict_ancestor_sets();
    let mut out = vec![PairInfo::default(); n * n];
    let mut given = BitSet::with_len(n);
    for i in 0..n {
        let x = DenseNodeId::from_index(i);
        for j in (i + 1)..n {
            let y = DenseNodeId::from_index(j);
            let direct = admg.children(x).contains(&y)
                || admg.children(y).contains(&x)
                || admg.bidirected_neighbors(x).contains(&y);
            let adjacent = direct || {
                given.clone_from(&ancestors[i]);
                given.union_with(&ancestors[j]);
                given.remove(x);
                given.remove(y);
                !admg
                    .is_m_separated(x, y, &given.to_dense_ids(), ws)
                    .map_err(IdentificationError::from)?
            };
            out[i * n + j] = PairInfo {
                adjacent,
                lo_anc_hi: ancestors[j].contains(x),
                hi_anc_lo: ancestors[i].contains(y),
            };
        }
    }
    Ok(out)
}

/// Backtracking enumeration of the sub-ADMGs of the dominating pool whose MAG is the target.
struct CandidateSearch<'a> {
    nodes: usize,
    pool: &'a [PoolEdge],
    target: &'a [MagPair],
    blank: &'a Admg,
    states: usize,
    members: Vec<Vec<bool>>,
    ws: DSeparationWorkspace,
}

impl CandidateSearch<'_> {
    fn graph(
        &self,
        include: &[bool],
        decided: usize,
        rest: bool,
    ) -> Result<Admg, IdentificationError> {
        let mut admg = self.blank.clone();
        for (index, edge) in self.pool.iter().enumerate() {
            let keep = if index < decided { include[index] } else { rest };
            if !keep {
                continue;
            }
            let inserted = match *edge {
                PoolEdge::Directed(a, b) => admg.insert_directed(a, b),
                PoolEdge::Bidirected(a, b) => admg.insert_bidirected(a, b),
            };
            inserted.map_err(IdentificationError::from)?;
        }
        Ok(admg)
    }

    /// Visit the branch with the first `decided` pool edges fixed. `false` once a bound is hit.
    fn visit(&mut self, decided: usize, include: &mut [bool]) -> Result<bool, IdentificationError> {
        self.states += 1;
        if self.states > MAX_CANDIDATE_STATES {
            return Ok(false);
        }
        let upper = self.graph(include, decided, true)?;
        let upper_info = pair_infos(&upper, &mut self.ws)?;
        let lower = self.graph(include, decided, false)?;
        let lower_info = pair_infos(&lower, &mut self.ws)?;
        let n = self.nodes;
        for i in 0..n {
            for j in (i + 1)..n {
                let target = self.target[i * n + j];
                if !target.reachable_from(upper_info[i * n + j])
                    || !target.admits_from_below(lower_info[i * n + j])
                {
                    return Ok(true);
                }
            }
        }
        if decided == self.pool.len() {
            // Nothing is undecided: both bounds are the graph itself, so it has the MAG.
            self.members.push(include.to_vec());
            return Ok(self.members.len() <= MAX_CANDIDATE_MEMBERS);
        }
        include[decided] = true;
        if !self.visit(decided + 1, include)? {
            return Ok(false);
        }
        include[decided] = false;
        self.visit(decided + 1, include)
    }
}

/// An ADMG over the nodes of `pag` (labels preserved) with no edges.
fn blank_admg(pag: &Pag) -> Result<Admg, IdentificationError> {
    let mut admg = Admg::empty();
    for &node in pag.nodes() {
        admg.add_node(node).map_err(IdentificationError::from)?;
    }
    Ok(admg)
}

/// The dominating pool and the MAG's pair relations.
fn pool_of(mag: &Pag) -> Result<(Vec<PoolEdge>, Vec<MagPair>), IdentificationError> {
    let n = mag.node_count();
    let mut pool = Vec::new();
    let mut bidirected = Vec::new();
    let mut target = vec![MagPair::Absent; n * n];
    for i in 0..n {
        let a = DenseNodeId::from_index(i);
        for (b, at_a, at_b) in mag.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            let slot = i * n + b.as_usize();
            match (at_a, at_b) {
                (Endpoint::Tail, Endpoint::Arrow) => {
                    pool.push(PoolEdge::Directed(a, b));
                    target[slot] = MagPair::Forward;
                }
                (Endpoint::Arrow, Endpoint::Tail) => {
                    pool.push(PoolEdge::Directed(b, a));
                    target[slot] = MagPair::Backward;
                }
                (Endpoint::Arrow, Endpoint::Arrow) => {
                    bidirected.push(PoolEdge::Bidirected(a, b));
                    target[slot] = MagPair::Bidirected;
                }
                _ => {
                    return Err(IdentificationError::unsupported(
                        "ADMG candidates of a MAG require directed and bidirected marks only",
                    ));
                }
            }
        }
    }
    for (tail, head) in invisible_directed_edges(mag) {
        bidirected.push(PoolEdge::Bidirected(tail, head));
    }
    pool.extend(bidirected);
    Ok((pool, target))
}

impl GeneralizedAdjustmentIdentifier {
    /// Every ADMG whose MAG is `mag`: the graphs that share its m-separation model and so
    /// cannot be told apart from the PAG, one of which is the latent projection of the true
    /// model.
    ///
    /// # Errors
    ///
    /// A graph that is not a circle-free directed/bidirected MAG, or a graph error.
    pub fn admg_candidates_of_mag(&self, mag: &Pag) -> Result<AdmgCandidates, IdentificationError> {
        let (pool, target) = pool_of(mag)?;
        let blank = blank_admg(mag)?;
        let mut search = CandidateSearch {
            nodes: mag.node_count(),
            pool: &pool,
            target: &target,
            blank: &blank,
            states: 0,
            members: Vec::new(),
            ws: DSeparationWorkspace::default(),
        };
        let mut include = vec![false; pool.len()];
        let complete = search.visit(0, &mut include)?;
        let members = search
            .members
            .iter()
            .map(|mask| search.graph(mask, pool.len(), false))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AdmgCandidates { members, bounded: !complete })
    }

    /// Identify a path-restricted natural effect, or a natural direct effect, over a PAG.
    ///
    /// Each MAG completion is read through the ADMGs it stands for (see the module
    /// documentation): the path-specific identifier decides every one, completions whose
    /// candidates all agree on one functional identify the effect by it, candidates that
    /// differ yield one case per distinct functional, and a refused candidate leaves the
    /// completion unidentified. The envelope's status is nonparametrically identified only
    /// when every candidate of every completion yields the same functional.
    ///
    /// # Errors
    ///
    /// A query the path-specific identifier does not read, or a completion that fails to
    /// prepare.
    pub fn identify_pag_path_specific_envelope(
        &self,
        pag: &Pag,
        query: &CausalQuery,
    ) -> Result<IdentificationEnvelope<Pag>, IdentificationError> {
        let identifier = PathSpecificIdentifier::new();
        let mut workspace = IdentificationWorkspace::default();
        let mut completions = 0usize;
        let mut candidates = 0usize;
        let mut distinct = 0usize;
        let mut envelope = self.pag_envelope_with_candidates(pag, |mag| {
            completions += 1;
            let found = self.admg_candidates_of_mag(mag)?;
            candidates += found.members.len();
            if found.bounded {
                let mut result = crate::generalized::not_identified(
                    query.clone(),
                    "the ADMG candidates of this MAG exceed the enumeration bound",
                );
                result.diagnostics.push(Diagnostic::new(
                    PAG_ADMG_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
                    DiagnosticKind::Execution,
                    DiagnosticSeverity::Warning,
                    "the ADMGs that share this MAG's m-separation model could not all be \
                     enumerated within the search bound, so the path-specific effect was not \
                     decided on every one of them; the completion is refused, not read as a point",
                ));
                return Ok(vec![result]);
            }
            let mut forms: Vec<Option<String>> = Vec::new();
            let mut results = Vec::new();
            for admg in &found.members {
                let prepared = identifier.prepare(admg)?;
                let mut result = identifier.identify(&prepared, query, &mut workspace)?;
                if !carries_identified_mass(result.status) {
                    result.diagnostics.push(Diagnostic::new(
                        "identify.path_specific_pag.unidentified_candidate",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        format!(
                            "one of the {} ADMG(s) with this MAG leaves the effect unidentified; \
                             no functional is correct for every model the MAG represents",
                            found.members.len()
                        ),
                    ));
                    return Ok(vec![result]);
                }
                let form = functional_form(&result);
                if forms.contains(&form) {
                    continue;
                }
                forms.push(form);
                results.push(result);
            }
            distinct += results.len();
            Ok(results)
        })?;
        envelope.push_features([GraphFeature {
            kind: Arc::from("pag_admg_candidates"),
            detail: Arc::from(format!(
                "{completions} MAG completion(s) read through {candidates} ADMG candidate(s) \
                 sharing their m-separation models; {distinct} distinct functional(s)"
            )),
        }]);
        Ok(envelope)
    }
}

/// Canonical text of a result's first functional, the form candidates are compared by.
fn functional_form(result: &IdentificationResult) -> Option<String> {
    let estimand = result.estimands.first()?;
    ((estimand.functional.raw() as usize) < result.arena.len())
        .then(|| result.arena.pretty(estimand.functional))
}
