//! Method provenance: the published sources and documented caveats behind an executed route.
//!
//! The only implementation of the lookup. The runtime result
//! (`antecedent::StudyResult::method_provenance`), the artifact encoder
//! ([`METHOD_PROVENANCE_SECTION`]) and the Python surface all call
//! [`method_provenance`] on the identifier and estimator ids the route executed, the same ids
//! the contract section seals. Records come from the generated
//! [`data/method_provenance.json`](../../data/method_provenance.json) table, written by
//! `scripts/generate_method_provenance.py` from `provenance/*.toml`, the strategy-table
//! provenance pointers, `parity/method_specs/specs.toml` and `parity/capability_records.toml`,
//! and kept current by `scripts/gate_provenance_schema.sh`.
//!
//! Each cited record carries its id and digest (the digest `method_review.record_sha256`
//! binds), the published sources with the DOI registry's authors and year, the implementation
//! deviations the record documents, and what its bounded method review leaves out. Whole
//! records are never embedded. An id whose provenance pointer names no record is reported in
//! [`MethodProvenanceWire::unlinked`], never filled with a guess.
//!
//! Provenance is descriptive: it changes no number, gate or support status.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::{EncodedArtifact, IoError, from_cbor};

/// Container section that carries [`MethodProvenanceWire`] on an `analysis_result` artifact.
///
/// Additive: old readers ignore it. It is derived from the contract section's sealed
/// `identifier` and `estimator`; [`decode_method_provenance_section`] checks that it names
/// exactly those ids.
pub const METHOD_PROVENANCE_SECTION: &str = "analysis_result.method_provenance";

/// Format of [`MethodProvenanceWire`].
pub const METHOD_PROVENANCE_FORMAT: u16 = 1;

const SNAPSHOT: &str = include_str!("../data/method_provenance.json");

/// One published source a provenance record cites.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceWire {
    /// Title as recorded (checked against the DOI registry when a DOI is present).
    pub title: String,
    /// Authors in registry order, from the DOI registry; empty without a DOI.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub authors: Vec<String>,
    /// Publication year from the DOI registry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub year: Option<u16>,
    /// DOI, when the source has a verified one. Absent rather than guessed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doi: Option<String>,
    /// Whether the DOI resolved to the recorded title in the retained registry metadata.
    #[serde(default)]
    pub doi_verified: bool,
    /// Link for a source without a DOI.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// The algorithms, theorems or sections the implementation follows.
    #[serde(default)]
    pub sections: Vec<String>,
}

/// What a record is cited for on this route.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceRole {
    /// The identification strategy.
    Identifier,
    /// The estimator or posterior construction.
    Estimator,
    /// A capability the route belongs to.
    Capability,
    /// A record whose numerics a cited wrapper record delegates to.
    CoveredBy,
}

impl ProvenanceRole {
    /// Wire text.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Identifier => "identifier",
            Self::Estimator => "estimator",
            Self::Capability => "capability",
            Self::CoveredBy => "covered_by",
        }
    }
}

/// A route component a record is cited for: its role and the id it was looked up by.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(deny_unknown_fields)]
pub struct CitedForWire {
    /// Role of the component.
    pub role: ProvenanceRole,
    /// Identifier, estimator or capability id; for [`ProvenanceRole::CoveredBy`], the wrapper
    /// record's id.
    pub key: String,
}

/// One provenance record behind the route, reduced to what a citation needs.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceRecordWire {
    /// Record id (`provenance/<id>.toml`).
    pub id: String,
    /// SHA-256 of the record without its review tables, as `method_review.record_sha256` binds.
    pub digest: String,
    /// `method_review` or `non_algorithm_review` when that review still attests this digest,
    /// else `none`.
    pub review: String,
    /// What the record is cited for on this route.
    #[serde(default)]
    pub cited_for: Vec<CitedForWire>,
    /// Published sources; empty for a record that justifies having none.
    #[serde(default)]
    pub sources: Vec<SourceWire>,
    /// Documented departures from the published method.
    #[serde(default)]
    pub deviations: Vec<String>,
    /// What the bounded method review does not cover.
    #[serde(default)]
    pub scope_limits: Vec<String>,
    /// Records a wrapper record delegates its numerics to.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub covered_by: Vec<String>,
}

/// The provenance of one result: every record behind its route.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MethodProvenanceWire {
    /// [`METHOD_PROVENANCE_FORMAT`].
    pub format: u16,
    /// Cited records in role order (identifier, estimator, capability, covered-by), then id.
    pub records: Vec<ProvenanceRecordWire>,
    /// Route components that name no provenance record.
    #[serde(default)]
    pub unlinked: Vec<CitedForWire>,
}

/// Which departure a [`CaveatWire`] states.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum CaveatKind {
    /// A documented implementation deviation from the published method.
    Deviation,
    /// A part of the implementation the bounded method review does not cover.
    ScopeLimit,
}

/// A caveat to state beside a citation: what was not implemented as published.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct CaveatWire {
    /// Record that documents it.
    pub record: String,
    /// Deviation or scope limit.
    pub kind: CaveatKind,
    /// User-facing text, as the record states it.
    pub text: String,
}

/// One entry of a bibliography: a source, the records that cite it and their caveats.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CitationWire {
    /// The source.
    pub source: SourceWire,
    /// Records citing it, with their digests, sorted by id.
    pub records: Vec<(String, String)>,
    /// Caveats of those records, sorted by record then kind.
    pub caveats: Vec<CaveatWire>,
}

impl CitationWire {
    /// One-line reference: `Authors (year). Title. https://doi.org/<doi>`.
    #[must_use]
    pub fn reference(&self) -> String {
        let source = &self.source;
        let mut text = String::new();
        if !source.authors.is_empty() {
            text.push_str(&author_list(&source.authors));
            text.push(' ');
        }
        if let Some(year) = source.year {
            text.push_str(&format!("({year}). "));
        } else if !source.authors.is_empty() {
            text.push_str("(n.d.). ");
        }
        text.push_str(source.title.trim_end_matches('.'));
        text.push('.');
        if let Some(doi) = &source.doi {
            text.push_str(&format!(" https://doi.org/{doi}"));
        } else if let Some(url) = &source.url {
            text.push_str(&format!(" {url}"));
        }
        text
    }
}

fn author_list(authors: &[String]) -> String {
    match authors {
        [] => String::new(),
        [one] => one.clone(),
        [first, second] => format!("{first} and {second}"),
        [first, ..] if authors.len() > 6 => format!("{first} et al."),
        [rest @ .., last] => format!("{}, and {last}", rest.join(", ")),
    }
}

#[derive(Deserialize)]
struct Snapshot {
    format: u16,
    records: Vec<SnapshotRecord>,
    identifier: BTreeMap<String, Vec<String>>,
    estimator: BTreeMap<String, Vec<String>>,
    capability: BTreeMap<String, Vec<String>>,
    unlinked: BTreeMap<String, Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SnapshotRecord {
    id: String,
    digest: String,
    review: String,
    sources: Vec<SourceWire>,
    deviations: Vec<String>,
    scope_limits: Vec<String>,
    #[serde(default)]
    covered_by: Vec<String>,
}

struct Table {
    records: HashMap<String, SnapshotRecord>,
    identifier: BTreeMap<String, Vec<String>>,
    estimator: BTreeMap<String, Vec<String>>,
    capability: BTreeMap<String, Vec<String>>,
}

fn table() -> &'static Table {
    static TABLE: OnceLock<Table> = OnceLock::new();
    TABLE.get_or_init(|| {
        let snapshot: Snapshot = serde_json::from_str(SNAPSHOT)
            .unwrap_or_else(|error| panic!("method provenance snapshot is malformed: {error}"));
        assert_eq!(snapshot.format, METHOD_PROVENANCE_FORMAT, "method provenance format");
        // Unlinked ids are kept in the snapshot for reviewers; a lookup of one reports it the
        // same way as any other id without a record.
        let _ = snapshot.unlinked;
        Table {
            records: snapshot.records.into_iter().map(|r| (r.id.clone(), r)).collect(),
            identifier: snapshot.identifier,
            estimator: snapshot.estimator,
            capability: snapshot.capability,
        }
    })
}

/// Record ids the table links to an id of `role`; `None` when it links none.
#[must_use]
pub fn linked_records(role: ProvenanceRole, key: &str) -> Option<&'static [String]> {
    let table = table();
    let keyed = match role {
        ProvenanceRole::Identifier => &table.identifier,
        ProvenanceRole::Estimator => &table.estimator,
        ProvenanceRole::Capability => &table.capability,
        ProvenanceRole::CoveredBy => return None,
    };
    keyed.get(key).map(Vec::as_slice)
}

/// Every capability id the table links to provenance records, sorted.
#[must_use]
pub fn capability_ids() -> Vec<&'static str> {
    table().capability.keys().map(String::as_str).collect()
}

/// Provenance of a route from the identifier and estimator ids it executed.
///
/// `None` leaves that component out. Ids are the strategy-table names the contract section
/// records (`IdentifierId::as_str`, `EstimatorId::as_str`).
#[must_use]
pub fn method_provenance(
    identifier: Option<&str>,
    estimator: Option<&str>,
) -> MethodProvenanceWire {
    let mut components = Vec::new();
    if let Some(identifier) = identifier {
        components.push((ProvenanceRole::Identifier, identifier));
    }
    if let Some(estimator) = estimator {
        components.push((ProvenanceRole::Estimator, estimator));
    }
    route_provenance(&components)
}

/// Provenance of a capability-specific route.
#[must_use]
pub fn capability_provenance(capability: &str) -> MethodProvenanceWire {
    route_provenance(&[(ProvenanceRole::Capability, capability)])
}

/// Provenance of a route from its components, each looked up in the generated table.
#[must_use]
pub fn route_provenance(components: &[(ProvenanceRole, &str)]) -> MethodProvenanceWire {
    let table = table();
    let mut cited: BTreeMap<String, Vec<CitedForWire>> = BTreeMap::new();
    let mut unlinked = Vec::new();
    for &(role, key) in components {
        let for_this = CitedForWire { role, key: key.to_owned() };
        match linked_records(role, key) {
            Some(ids) => {
                for id in ids {
                    cite(&mut cited, id, &for_this);
                }
            }
            None => unlinked.push(for_this),
        }
    }
    // A wrapper's numerics live in the records it is covered by; cite those too.
    let mut pending: Vec<String> = cited.keys().cloned().collect();
    while let Some(id) = pending.pop() {
        let Some(record) = table.records.get(&id) else { continue };
        for covered in &record.covered_by {
            let first = !cited.contains_key(covered);
            cite(
                &mut cited,
                covered,
                &CitedForWire { role: ProvenanceRole::CoveredBy, key: id.clone() },
            );
            if first {
                pending.push(covered.clone());
            }
        }
    }
    let mut records: Vec<ProvenanceRecordWire> = cited
        .into_iter()
        .filter_map(|(id, mut cited_for)| {
            let record = table.records.get(&id)?;
            cited_for.sort();
            cited_for.dedup();
            Some(ProvenanceRecordWire {
                id,
                digest: record.digest.clone(),
                review: record.review.clone(),
                cited_for,
                sources: record.sources.clone(),
                deviations: record.deviations.clone(),
                scope_limits: record.scope_limits.clone(),
                covered_by: record.covered_by.clone(),
            })
        })
        .collect();
    records.sort_by(|a, b| a.cited_for.first().cmp(&b.cited_for.first()).then(a.id.cmp(&b.id)));
    MethodProvenanceWire { format: METHOD_PROVENANCE_FORMAT, records, unlinked }
}

fn cite(cited: &mut BTreeMap<String, Vec<CitedForWire>>, id: &str, for_this: &CitedForWire) {
    cited.entry(id.to_owned()).or_default().push(for_this.clone());
}

impl MethodProvenanceWire {
    /// Whether no record is cited.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The cited record with `id`.
    #[must_use]
    pub fn record(&self, id: &str) -> Option<&ProvenanceRecordWire> {
        self.records.iter().find(|record| record.id == id)
    }

    /// The sources of this result, de-duplicated, each with the caveats of the records citing it.
    #[must_use]
    pub fn citations(&self) -> Vec<CitationWire> {
        bibliography([self])
    }
}

/// De-duplicated citations across results, in first-cited order.
///
/// Sources are the same when their DOIs agree (case-insensitively), or, without a DOI, when
/// their titles agree after case and punctuation are dropped. A merged source keeps every
/// citing record and every caveat, and the union of the cited sections.
#[must_use]
pub fn bibliography<'a, I>(provenances: I) -> Vec<CitationWire>
where
    I: IntoIterator<Item = &'a MethodProvenanceWire>,
{
    let mut order: Vec<String> = Vec::new();
    let mut entries: HashMap<String, CitationWire> = HashMap::new();
    for provenance in provenances {
        for record in &provenance.records {
            for source in &record.sources {
                let key = source_key(source);
                let entry = entries.entry(key.clone()).or_insert_with(|| {
                    order.push(key);
                    CitationWire {
                        source: SourceWire { sections: Vec::new(), ..source.clone() },
                        records: Vec::new(),
                        caveats: Vec::new(),
                    }
                });
                for section in &source.sections {
                    if !entry.source.sections.contains(section) {
                        entry.source.sections.push(section.clone());
                    }
                }
                let cited = (record.id.clone(), record.digest.clone());
                if !entry.records.contains(&cited) {
                    entry.records.push(cited);
                }
                let caveats =
                    record.deviations.iter().map(|text| (CaveatKind::Deviation, text)).chain(
                        record.scope_limits.iter().map(|text| (CaveatKind::ScopeLimit, text)),
                    );
                for (kind, text) in caveats {
                    let caveat = CaveatWire { record: record.id.clone(), kind, text: text.clone() };
                    if !entry.caveats.contains(&caveat) {
                        entry.caveats.push(caveat);
                    }
                }
            }
        }
    }
    order
        .into_iter()
        .filter_map(|key| entries.remove(&key))
        .map(|mut entry| {
            entry.records.sort();
            entry.caveats.sort();
            entry
        })
        .collect()
}

fn source_key(source: &SourceWire) -> String {
    match &source.doi {
        Some(doi) => format!("doi:{}", doi.to_lowercase()),
        None => format!(
            "title:{}",
            source
                .title
                .chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        ),
    }
}

/// Decode the optional [`METHOD_PROVENANCE_SECTION`]. Missing is `Ok(None)`.
///
/// `identifier` and `estimator` are the ids the contract section seals (declared, so `None`
/// when the study left the choice to the planner). The section must name each declared id for
/// its role and at most one id per role; its records are the producer's table at export and
/// are descriptive, so a newer table citing other records is not a mismatch.
///
/// # Errors
///
/// A duplicate section, invalid CBOR, an unsupported format, or components that disagree with
/// the declared identifier and estimator.
pub fn decode_method_provenance_section(
    artifact: &EncodedArtifact,
    identifier: Option<&str>,
    estimator: Option<&str>,
) -> Result<Option<MethodProvenanceWire>, IoError> {
    let mut found = artifact.sections.iter().filter(|s| s.id == METHOD_PROVENANCE_SECTION);
    let Some(section) = found.next() else {
        return Ok(None);
    };
    if found.next().is_some() {
        return Err(IoError::Convert(format!("duplicate section `{METHOD_PROVENANCE_SECTION}`")));
    }
    let provenance: MethodProvenanceWire = from_cbor(&section.data)?;
    check_bound(&provenance, identifier, estimator)?;
    Ok(Some(provenance))
}

/// Whether `provenance` is the provenance of a route with these declared ids: the supported
/// format, at most one id per role, and each declared id named for its role.
///
/// # Errors
///
/// An unsupported format or components that disagree with the declared ids.
pub fn check_bound(
    provenance: &MethodProvenanceWire,
    identifier: Option<&str>,
    estimator: Option<&str>,
) -> Result<(), IoError> {
    if provenance.format != METHOD_PROVENANCE_FORMAT {
        return Err(IoError::Convert(format!(
            "unsupported method provenance format {}",
            provenance.format
        )));
    }
    let named = |role: ProvenanceRole| {
        let mut keys: Vec<&str> = provenance
            .records
            .iter()
            .flat_map(|record| record.cited_for.iter())
            .chain(&provenance.unlinked)
            .filter(|cited| cited.role == role)
            .map(|cited| cited.key.as_str())
            .collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    };
    for (role, declared) in
        [(ProvenanceRole::Identifier, identifier), (ProvenanceRole::Estimator, estimator)]
    {
        let keys = named(role);
        let agrees = match declared {
            Some(declared) => keys == [declared],
            None => keys.len() <= 1,
        };
        if !agrees {
            return Err(IoError::Convert(format!(
                "method provenance names {} {keys:?}, the contract declares {declared:?}",
                role.as_str()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backdoor_aipw_cites_its_records_and_deviations() {
        let provenance = method_provenance(Some("backdoor.adjustment"), Some("aipw"));
        assert!(provenance.unlinked.is_empty());
        let ids: Vec<&str> = provenance.records.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["identify.backdoor", "estimate.aipw"]);
        let aipw = provenance.record("estimate.aipw").expect("aipw record");
        assert_eq!(
            aipw.cited_for,
            [CitedForWire { role: ProvenanceRole::Estimator, key: "aipw".into() }]
        );
        assert_eq!(aipw.digest.len(), 64);
        assert!(aipw.deviations.iter().any(|d| d.starts_with("no cross-fitting")));
        assert!(aipw.sources.iter().any(|s| s.doi.as_deref() == Some("10.1111/ectj.12097")));
    }

    #[test]
    fn an_unknown_id_is_unlinked_not_guessed() {
        let provenance = method_provenance(None, Some("no.such.estimator"));
        assert!(provenance.is_empty());
        assert_eq!(
            provenance.unlinked,
            [CitedForWire { role: ProvenanceRole::Estimator, key: "no.such.estimator".into() }]
        );
    }

    #[test]
    fn bibliography_merges_a_source_cited_by_two_results() {
        let one = method_provenance(Some("backdoor.adjustment"), Some("aipw"));
        let two = method_provenance(Some("backdoor.adjustment"), Some("linear.adjustment.ate"));
        let both = bibliography([&one, &two]);
        let mut keys: Vec<String> = both.iter().map(|c| source_key(&c.source)).collect();
        let before = keys.len();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), before, "every source appears once");
        let separate = one.citations().len() + two.citations().len();
        assert!(both.len() < separate, "the shared identifier source is merged");
    }

    #[test]
    fn reference_formats_authors_year_and_doi() {
        let citation = CitationWire {
            source: SourceWire {
                title: "Doubly Robust Estimation of Causal Effects".into(),
                authors: vec!["A One".into(), "B Two".into(), "C Three".into()],
                year: Some(2011),
                doi: Some("10.1093/aje/kwq439".into()),
                doi_verified: true,
                url: None,
                sections: vec![],
            },
            records: vec![],
            caveats: vec![],
        };
        assert_eq!(
            citation.reference(),
            "A One, B Two, and C Three (2011). Doubly Robust Estimation of Causal Effects. \
             https://doi.org/10.1093/aje/kwq439"
        );
    }
}
