//! Declared overlap of the observations behind count tables.
//!
//! Two count tables may rest on disjoint units (independent), on the same units
//! (shared), or on units whose relation is not declared (unknown). Unknown is
//! never read as independent, and a declaration that contradicts the counts
//! (independent tables that hold identical observations) is refused as double
//! counting. Shared units are modelled only where the shared observations are
//! recoverable from the counts, i.e. the two tables are the same dataset
//! registered under two labels: such tables are one multinomial sample, not two.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CanonicalWriter;

use super::counts::{SampledRegimeCounts, SampledTableKey};
use super::{
    DETAIL_OVERLAP_DOUBLE, DETAIL_OVERLAP_INVALID, DETAIL_OVERLAP_UNKNOWN, DETAIL_OVERLAP_UNLINKED,
    refused,
};
use crate::error::EstimationError;

/// Declared relation of two count tables' units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DatasetOverlap {
    /// The units are disjoint: independent samples.
    Independent,
    /// The tables hold the same units (one dataset under two labels).
    SharedUnits,
    /// The relation is not declared; the request is refused.
    Unknown,
}

impl DatasetOverlap {
    /// Stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::SharedUnits => "shared_units",
            Self::Unknown => "unknown",
        }
    }

    /// Parse a stable name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "independent" => Some(Self::Independent),
            "shared_units" => Some(Self::SharedUnits),
            "unknown" => Some(Self::Unknown),
            _ => None,
        }
    }

    const fn word(self) -> u64 {
        match self {
            Self::Independent => 0,
            Self::SharedUnits => 1,
            Self::Unknown => 2,
        }
    }
}

/// The declared relation of one pair of tables.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OverlapPair {
    /// First table (the smaller key).
    pub a: SampledTableKey,
    /// Second table.
    pub b: SampledTableKey,
    /// Their relation.
    pub relation: DatasetOverlap,
}

/// The overlap declaration of a request: a relation for every pair of tables,
/// from pair-specific entries and one default for the rest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OverlapDeclaration {
    default: DatasetOverlap,
    pairs: Vec<OverlapPair>,
}

impl OverlapDeclaration {
    /// A declaration whose every undeclared pair has relation `default`.
    #[must_use]
    pub const fn with_default(default: DatasetOverlap) -> Self {
        Self { default, pairs: Vec::new() }
    }

    /// Every pair declared independent.
    #[must_use]
    pub const fn all_independent() -> Self {
        Self::with_default(DatasetOverlap::Independent)
    }

    /// Nothing declared: every pair unknown (a request with two tables is refused).
    #[must_use]
    pub const fn undeclared() -> Self {
        Self::with_default(DatasetOverlap::Unknown)
    }

    /// Declare one pair.
    ///
    /// # Errors
    ///
    /// `invalid_argument` for a pair of one table with itself or a pair already
    /// declared with another relation.
    pub fn with_pair(
        mut self,
        a: SampledTableKey,
        b: SampledTableKey,
        relation: DatasetOverlap,
    ) -> Result<Self, EstimationError> {
        if a == b {
            return Err(refused(
                antecedent_core::reason_code!("invalid_argument"),
                DETAIL_OVERLAP_INVALID,
                "an overlap pair names one table twice",
            ));
        }
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        if let Some(existing) = self.pairs.iter().find(|p| p.a == a && p.b == b) {
            if existing.relation != relation {
                return Err(refused(
                    antecedent_core::reason_code!("invalid_argument"),
                    DETAIL_OVERLAP_INVALID,
                    "a pair of tables is declared with two relations",
                ));
            }
            return Ok(self);
        }
        self.pairs.push(OverlapPair { a, b, relation });
        self.pairs.sort_by(|x, y| (&x.a, &x.b).cmp(&(&y.a, &y.b)));
        Ok(self)
    }

    /// Relation applied to undeclared pairs.
    #[must_use]
    pub const fn default_relation(&self) -> DatasetOverlap {
        self.default
    }

    /// Declared pairs, ascending.
    #[must_use]
    pub fn pairs(&self) -> &[OverlapPair] {
        &self.pairs
    }

    /// The relation of two tables.
    #[must_use]
    pub fn relation(&self, a: &SampledTableKey, b: &SampledTableKey) -> DatasetOverlap {
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        self.pairs.iter().find(|p| &p.a == a && &p.b == b).map_or(self.default, |p| p.relation)
    }

    pub(crate) fn write_identity(&self, w: &mut CanonicalWriter) {
        w.word(self.default.word()).word(self.pairs.len() as u64);
        for pair in &self.pairs {
            for key in [&pair.a, &pair.b] {
                w.text(&key.population).word(u64::from(key.regime.raw()));
                w.word(key.interventions.len() as u64);
                for (v, bits) in &key.interventions {
                    w.word(u64::from(v.raw())).word(*bits);
                }
            }
            w.word(pair.relation.word());
        }
    }
}

/// Tables grouped into datasets: tables of one group hold the same units and are
/// one multinomial sample; distinct groups are independent samples.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DatasetGroups {
    /// Table indices of each dataset, ascending; groups ordered by first member.
    pub(crate) groups: Vec<Vec<usize>>,
}

/// Union-find root with path halving.
fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Resolve a declaration against the tables (sorted canonically by the caller).
///
/// # Errors
///
/// `sampling_dependence_unknown` for an unknown relation; `evidence_double_counted`
/// for independent tables that hold identical observations; `route_not_supported`
/// for shared units whose observations are not identical tables (the covariance
/// is not recoverable from counts); `invalid_argument` for a pair that names a
/// table the request does not hold.
pub(crate) fn resolve_overlap(
    tables: &[SampledRegimeCounts],
    declaration: &OverlapDeclaration,
) -> Result<DatasetGroups, EstimationError> {
    let keys: Vec<SampledTableKey> = tables.iter().map(SampledRegimeCounts::key).collect();
    for pair in declaration.pairs() {
        if !keys.contains(&pair.a) || !keys.contains(&pair.b) {
            return Err(refused(
                antecedent_core::reason_code!("invalid_argument"),
                DETAIL_OVERLAP_INVALID,
                "an overlap declaration names a table the request does not supply",
            ));
        }
    }
    let n = tables.len();
    let mut parent: Vec<usize> = (0..n).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            let relation = declaration.relation(&keys[i], &keys[j]);
            let identical = tables[i].same_observations(&tables[j]);
            match relation {
                DatasetOverlap::Unknown => {
                    return Err(refused(
                        antecedent_core::reason_code!("sampling_dependence_unknown"),
                        DETAIL_OVERLAP_UNKNOWN,
                        format!(
                            "the overlap of the units behind population {} regime {} and \
                             population {} regime {} is not declared; independence is never \
                             assumed. Declare the pair independent (disjoint units) or shared \
                             units, or derive the declaration from an evidence ledger",
                            tables[i].population(),
                            tables[i].regime().raw(),
                            tables[j].population(),
                            tables[j].regime().raw()
                        ),
                    ));
                }
                DatasetOverlap::Independent => {
                    if identical && tables[i].observed_cells() >= 2 {
                        return Err(refused(
                            antecedent_core::reason_code!("evidence_double_counted"),
                            DETAIL_OVERLAP_DOUBLE,
                            format!(
                                "population {} regime {} and population {} regime {} are \
                                 declared independent but hold identical observations; the same \
                                 units would be counted twice. Declare them shared units (one \
                                 dataset) or supply the disjoint samples",
                                tables[i].population(),
                                tables[i].regime().raw(),
                                tables[j].population(),
                                tables[j].regime().raw()
                            ),
                        ));
                    }
                }
                DatasetOverlap::SharedUnits => {
                    if !identical {
                        return Err(refused(
                            antecedent_core::reason_code!("route_not_supported"),
                            DETAIL_OVERLAP_UNLINKED,
                            "two tables are declared to share units but are not the same \
                             observations; the covariance of overlapping but different tables is \
                             not recoverable from counts, so only an identical dataset is \
                             modelled as shared",
                        ));
                    }
                    let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                    if ri != rj {
                        parent[rj.max(ri)] = rj.min(ri);
                    }
                }
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut root_group: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        let r = find(&mut parent, i);
        let known = root_group[r];
        if let Some(g) = known {
            groups[g].push(i);
        } else {
            root_group[r] = Some(groups.len());
            groups.push(vec![i]);
        }
    }
    Ok(DatasetGroups { groups })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{RegimeId, VariableId};

    use super::*;
    use crate::counterfactual_transport_sampled::counts::{
        SampledAxis, SampledRegimeCountsParts, SamplingDesign,
    };

    fn table(population: &str, regime: u32, counts: Vec<u64>) -> SampledRegimeCounts {
        SampledRegimeCounts::try_new(SampledRegimeCountsParts {
            population: Arc::from(population),
            regime: RegimeId::from_raw(regime),
            interventions: Vec::new(),
            axes: vec![SampledAxis {
                variable: VariableId::from_raw(0),
                levels: vec![0.0, 1.0, 2.0],
            }],
            counts,
            snapshot_identity: Arc::from(format!("{population}:{regime}")),
            design: SamplingDesign::IndependentUnits,
            structural_zeros: Vec::new(),
        })
        .unwrap()
    }

    fn code(error: &EstimationError) -> &'static str {
        match error {
            EstimationError::Refused { code, .. } => code,
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn unknown_overlap_is_never_independence() {
        let tables = vec![table("a", 0, vec![5, 6, 7]), table("b", 1, vec![9, 3, 4])];
        let error = resolve_overlap(&tables, &OverlapDeclaration::undeclared()).unwrap_err();
        assert_eq!(code(&error), "sampling_dependence_unknown");
        let groups = resolve_overlap(&tables, &OverlapDeclaration::all_independent()).unwrap();
        assert_eq!(groups.groups, vec![vec![0], vec![1]]);
    }

    #[test]
    fn identical_observations_declared_independent_are_double_counting() {
        let tables = vec![table("a", 0, vec![5, 6, 7]), table("b", 1, vec![5, 6, 7])];
        let error = resolve_overlap(&tables, &OverlapDeclaration::all_independent()).unwrap_err();
        assert_eq!(code(&error), "evidence_double_counted");
    }

    #[test]
    fn shared_units_make_identical_tables_one_dataset_and_refuse_different_ones() {
        let same = vec![table("a", 0, vec![5, 6, 7]), table("b", 1, vec![5, 6, 7])];
        let shared = OverlapDeclaration::all_independent()
            .with_pair(same[0].key(), same[1].key(), DatasetOverlap::SharedUnits)
            .unwrap();
        let groups = resolve_overlap(&same, &shared).unwrap();
        assert_eq!(groups.groups, vec![vec![0, 1]]);
        // The relation is symmetric in the order the pair is named.
        let reversed = OverlapDeclaration::all_independent()
            .with_pair(same[1].key(), same[0].key(), DatasetOverlap::SharedUnits)
            .unwrap();
        assert_eq!(reversed, shared);

        let different = vec![table("a", 0, vec![5, 6, 7]), table("b", 1, vec![5, 7, 6])];
        let declared = OverlapDeclaration::all_independent()
            .with_pair(different[0].key(), different[1].key(), DatasetOverlap::SharedUnits)
            .unwrap();
        let error = resolve_overlap(&different, &declared).unwrap_err();
        assert_eq!(code(&error), "route_not_supported");
    }

    #[test]
    fn a_pair_cannot_be_declared_twice_with_different_relations_or_name_absent_tables() {
        let tables = vec![table("a", 0, vec![5, 6, 7]), table("b", 1, vec![9, 3, 4])];
        let (a, b) = (tables[0].key(), tables[1].key());
        let once = OverlapDeclaration::all_independent()
            .with_pair(a.clone(), b.clone(), DatasetOverlap::Independent)
            .unwrap();
        assert!(once.clone().with_pair(a.clone(), b.clone(), DatasetOverlap::SharedUnits).is_err());
        assert!(once.clone().with_pair(a.clone(), a.clone(), DatasetOverlap::Independent).is_err());
        let absent = table("c", 2, vec![1, 1, 1]).key();
        let names_absent = OverlapDeclaration::all_independent()
            .with_pair(a, absent, DatasetOverlap::Unknown)
            .unwrap();
        assert_eq!(code(&resolve_overlap(&tables, &names_absent).unwrap_err()), "invalid_argument");
    }
}
