use super::{
    Arc, CatalogDistribution, EstimationError, EvidenceCatalog, EvidenceProjection,
    ExactDiscreteLaw, VariableId,
};

/// How one cited law is drawn: from the table of its dataset as it is, or as
/// a recorded margin of that dataset's finest table.
struct LawSource {
    /// Index of the resampled table the law reads.
    dataset: usize,
    /// For a projected margin, the law cell of every table cell.
    projection: Option<Vec<usize>>,
}

/// The cited count tables of a z-transport functional, one per independent
/// dataset: laws whose catalog binding names the same forwarded dataset share
/// one table and one resampling stream.
pub(super) struct CitedTables<'a> {
    laws: &'a [ExactDiscreteLaw],
    /// How each law is rebuilt from the resampled tables.
    sources: Vec<LawSource>,
    /// The counts of each resampled table.
    pub(super) counts: Vec<&'a [u64]>,
}

/// How laws naming the same forwarded dataset are resampled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AliasPolicy {
    /// Every alias carries the identical table; anything else is a data error.
    Identical,
    /// Aliases are identical or a recorded margin of the finest shared table;
    /// any other pair withholds the interval.
    JointProjection,
}

/// Forwarded dataset identity of the catalog binding of `law`'s regime.
fn forwarded_dataset(catalog: &EvidenceCatalog, law: &ExactDiscreteLaw) -> Option<Arc<str>> {
    catalog
        .bindings
        .iter()
        .find(|binding| binding.regime == law.regime())
        .and_then(|binding| binding.dataset_identity.clone())
}

/// Whether two catalog descriptors describe the same law family: everything
/// but the source entry, its snapshot and the projection trail agrees.
fn same_law_family(a: &CatalogDistribution, b: &CatalogDistribution) -> bool {
    let set = |ids: &[VariableId]| ids.iter().copied().collect::<std::collections::BTreeSet<_>>();
    let values = |d: &CatalogDistribution| {
        let mut out =
            d.intervention_values.iter().map(|a| (a.variable, a.value.clone())).collect::<Vec<_>>();
        out.sort_by_key(|(variable, _)| *variable);
        out
    };
    a.study == b.study
        && a.population == b.population
        && a.kind == b.kind
        && a.evidence_kind == b.evidence_kind
        && a.origin == b.origin
        && a.selection == b.selection
        && set(&a.interventions) == set(&b.interventions)
        && values(a) == values(b)
        && set(&a.measured) == set(&b.measured)
        && a.availability == b.availability
        && set(&a.conditioned_on) == set(&b.conditioned_on)
        && a.dataset == b.dataset
}

/// How a law relates to the finest table of the dataset it shares.
enum Relation {
    /// The identical table.
    Identical,
    /// A recorded margin: the law cell of every root cell.
    Margin(Vec<usize>),
}

/// How `law` relates to the finest table `root` of the dataset they share:
/// [`Relation::Identical`] for the identical table, [`Relation::Margin`] for a recorded margin
/// (the catalog descriptor of `law` is `root`'s marginalized over the dropped
/// coordinates, and its counts are exactly that margin of `root`'s counts), and
/// `None` when neither holds.
fn relate_to_root(
    catalog: &EvidenceCatalog,
    root: &ExactDiscreteLaw,
    law: &ExactDiscreteLaw,
) -> Option<Relation> {
    let (Some(root_counts), Some(counts)) = (root.empirical_counts(), law.empirical_counts())
    else {
        return None;
    };
    let (Some(root_d), Some(law_d)) =
        (catalog.distribution(root.regime()), catalog.distribution(law.regime()))
    else {
        return None;
    };
    let same_world = root.population() == law.population()
        && root.interventions().len() == law.interventions().len()
        && law.interventions().iter().all(|a| root.interventions().contains(a));
    if !same_world {
        return None;
    }
    let same_axes = |a: &antecedent_expr::DiscreteAxis, b: &antecedent_expr::DiscreteAxis| {
        a.variable == b.variable && a.values == b.values
    };
    if root.axes().len() == law.axes().len()
        && root.axes().iter().zip(law.axes()).all(|(a, b)| same_axes(a, b))
    {
        return (same_law_family(&root_d, &law_d) && root_counts == counts)
            .then_some(Relation::Identical);
    }
    let drop = root_d
        .measured
        .iter()
        .filter(|variable| !law_d.measured.contains(variable))
        .copied()
        .collect::<Vec<_>>();
    let projected = root_d.project(EvidenceProjection::Marginalize { drop: drop.into() }).ok()?;
    if !same_law_family(&projected, &law_d) {
        return None;
    }
    // Each law axis is a root axis with the same levels; the law cell of a root
    // cell reads those axes' levels (last axis fastest in both layouts).
    let positions = law
        .axes()
        .iter()
        .map(|axis| root.axes().iter().position(|root_axis| same_axes(root_axis, axis)))
        .collect::<Option<Vec<_>>>()?;
    let mut map = Vec::with_capacity(root_counts.len());
    let mut level = vec![0usize; root.axes().len()];
    for _ in 0..root_counts.len() {
        map.push(
            positions
                .iter()
                .zip(law.axes())
                .fold(0usize, |cell, (position, axis)| cell * axis.values.len() + level[*position]),
        );
        for (axis, value) in root.axes().iter().zip(level.iter_mut()).rev() {
            *value += 1;
            if *value < axis.values.len() {
                break;
            }
            *value = 0;
        }
    }
    let mut margin = vec![0u64; counts.len()];
    for (cell, count) in map.iter().zip(root_counts) {
        margin[*cell] = margin[*cell].checked_add(*count)?;
    }
    (margin == counts).then_some(Relation::Margin(map))
}

impl<'a> CitedTables<'a> {
    pub(super) fn new(
        functional: &dyn crate::transport::BoundZFormula,
        data: &'a antecedent_expr::ExactTransportData,
        policy: AliasPolicy,
    ) -> Result<Result<Self, &'static str>, EstimationError> {
        let laws = data.laws();
        let catalog = functional.catalog();
        let mut tables = Vec::with_capacity(laws.len());
        for law in laws {
            tables.push(law.empirical_counts().ok_or_else(|| {
                EstimationError::data_msg("z-transport interval requires counted laws")
            })?);
        }
        let identities = laws.iter().map(|law| forwarded_dataset(catalog, law)).collect::<Vec<_>>();
        // The finest law of each forwarded dataset, by cell count then order.
        let root_of = |index: usize| -> usize {
            let Some(identity) = &identities[index] else {
                return index;
            };
            match policy {
                AliasPolicy::Identical => (0..laws.len())
                    .find(|j| identities[*j].as_ref() == Some(identity))
                    .unwrap_or(index),
                AliasPolicy::JointProjection => (0..laws.len())
                    .filter(|j| identities[*j].as_ref() == Some(identity))
                    .max_by_key(|j| (tables[*j].len(), std::cmp::Reverse(*j)))
                    .unwrap_or(index),
            }
        };
        let roots = (0..laws.len()).map(root_of).collect::<Vec<_>>();
        // Resampled tables in first-appearance order; the joint policy orders
        // them by (population, regime) so its streams do not depend on the
        // order laws were supplied in.
        let mut order = Vec::new();
        for root in &roots {
            if !order.contains(root) {
                order.push(*root);
            }
        }
        if policy == AliasPolicy::JointProjection {
            order.sort_by_key(|root| (laws[*root].population(), laws[*root].regime().raw()));
        }
        let counts = order.iter().map(|root| tables[*root]).collect::<Vec<_>>();
        let mut sources = Vec::with_capacity(laws.len());
        for (index, law) in laws.iter().enumerate() {
            let root = roots[index];
            let dataset = order.iter().position(|r| *r == root).unwrap_or_default();
            let projection = if root == index {
                None
            } else {
                match policy {
                    AliasPolicy::Identical => {
                        if tables[root] != tables[index] {
                            return Err(EstimationError::data_msg(
                                "conflicting forwarded dataset aliases",
                            ));
                        }
                        None
                    }
                    AliasPolicy::JointProjection => match relate_to_root(catalog, &laws[root], law)
                    {
                        Some(Relation::Identical) => None,
                        Some(Relation::Margin(map)) => Some(map),
                        None => return Ok(Err("transport.unsupported_dependence")),
                    },
                }
            };
            sources.push(LawSource { dataset, projection });
        }
        Ok(Ok(Self { laws, sources, counts }))
    }

    /// Rebuild every cited law from one resampled table per dataset; a projected
    /// law is the margin of its dataset's table in the same replicate.
    pub(super) fn rebuild(
        &self,
        tables: &[Vec<f64>],
        counts: Option<&[Vec<u64>]>,
    ) -> Result<Vec<ExactDiscreteLaw>, EstimationError> {
        self.laws
            .iter()
            .zip(&self.sources)
            .map(|(law, source)| {
                let dataset = source.dataset;
                let (probabilities, law_counts) = match &source.projection {
                    None => (tables[dataset].clone(), counts.map(|c| c[dataset].clone())),
                    Some(map) => {
                        let cells = law.probabilities().len();
                        if let Some(counts) = counts {
                            let mut margin = vec![0u64; cells];
                            for (cell, count) in map.iter().zip(&counts[dataset]) {
                                margin[*cell] += count;
                            }
                            let total = margin.iter().sum::<u64>() as f64;
                            let p = margin.iter().map(|c| *c as f64 / total).collect();
                            (p, Some(margin))
                        } else {
                            let mut p = vec![0.0; cells];
                            for (cell, mass) in map.iter().zip(&tables[dataset]) {
                                p[*cell] += mass;
                            }
                            (p, None)
                        }
                    }
                };
                let rebuilt = match law_counts {
                    Some(law_counts) => antecedent_expr::ExactDiscreteLaw::try_empirical(
                        law.population(),
                        law.regime(),
                        law.interventions().to_vec(),
                        law.axes().to_vec(),
                        probabilities,
                        law.snapshot_identity(),
                        law.tolerance(),
                    )
                    .and_then(|rebuilt| rebuilt.with_empirical_counts(law_counts)),
                    None => antecedent_expr::ExactDiscreteLaw::try_bayesian_posterior(
                        law.population(),
                        law.regime(),
                        law.interventions().to_vec(),
                        law.axes().to_vec(),
                        probabilities,
                        law.snapshot_identity(),
                        law.tolerance(),
                    ),
                };
                rebuilt.map_err(EstimationError::from)
            })
            .collect()
    }
}
