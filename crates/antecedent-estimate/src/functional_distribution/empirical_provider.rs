use super::{
    Arc, Assignment, CausalRng, DataError, DiscreteColumn, DomainRef, EmpiricalTableProvider,
    EstimationError, ExecutionContext, FactorKey, FactorSignature, FactorSpec, HashMap, HashSet,
    InterventionAssignment, MAX_DISCRETE_LEVELS, Mutex, RetainedFactorFit, RowResampling,
    TableView, TabularData, Value, VariableId, eval_err,
};

/// Discrete columns pre-encoded to `u32` level codes.
///
/// Encoding once (per fit) turns every bootstrap replicate into an index gather and every
/// CPT count into integer arithmetic: no per-row `Value` clones, hashing or key allocation.
/// The level table is shared (`Arc`) across replicates; a replicate's own domain is the set
/// of codes that occur in it, in order of first appearance.
#[derive(Clone, Debug)]
pub(super) struct EncodedColumns {
    n: usize,
    pub(super) row_ids: Vec<usize>,
    columns: HashMap<VariableId, EncodedColumn>,
    /// Mixed-radix row keys by variable list, computed on first use. They depend on the
    /// codes alone, so every weighted draw over these rows shares them.
    row_keys: RowKeys,
}

/// A row with a missing level under [`RowKeys`].
const NO_KEY: u32 = u32::MAX;

#[derive(Debug, Default)]
struct RowKeys(Mutex<HashMap<Vec<VariableId>, Arc<[u32]>>>);

impl Clone for RowKeys {
    fn clone(&self) -> Self {
        Self::default()
    }
}

#[derive(Clone, Debug)]
pub(super) struct EncodedColumn {
    /// Level index per row, or [`DiscreteColumn::MISSING`].
    pub(super) codes: Vec<u32>,
    /// Level value by code.
    pub(super) levels: Arc<[Value]>,
    /// [`Self::observed_codes`], computed on first use.
    observed: std::sync::OnceLock<Vec<u32>>,
}

impl EncodedColumns {
    /// Read `ids` through the shared discrete reader (missing = invalid or outside the
    /// analysis mask), failing closed on empty or oversized domains.
    pub(super) fn read(
        data: &TabularData,
        ids: &HashSet<VariableId>,
    ) -> Result<Self, EstimationError> {
        let n = data.row_count();
        let mut columns = HashMap::with_capacity(ids.len());
        for &id in ids {
            let col = data.discrete_column(id).map_err(|e| match e {
                DataError::TypeMismatch { .. } => EstimationError::unsupported_coded(
                    antecedent_core::reason_code!("route_not_supported"),
                    "functional.distribution supports float64 / int64 / categorical columns only",
                ),
                other => EstimationError::from(other),
            })?;
            if col.codes.len() != n {
                return Err(EstimationError::data_msg("column length mismatch"));
            }
            if col.levels.is_empty() {
                return Err(EstimationError::data_msg("empty discrete domain"));
            }
            if col.levels.len() > MAX_DISCRETE_LEVELS.max(n / 4) {
                return Err(EstimationError::unsupported(
                    "variable exceeds discrete level cap for functional.distribution",
                ));
            }
            columns.insert(id, EncodedColumn::new(col.codes, Arc::from(col.levels)));
        }
        Ok(Self { n, row_ids: (0..n).collect(), columns, row_keys: RowKeys::default() })
    }

    /// Row count.
    pub(super) fn n(&self) -> usize {
        self.n
    }

    /// Rows where every column has a level.
    pub(super) fn complete_rows(&self) -> Vec<usize> {
        (0..self.n)
            .filter(|&row| self.columns.values().all(|c| c.codes[row] != DiscreteColumn::MISSING))
            .collect()
    }

    /// The rows `idx` (repeats allowed) as a new sample sharing the level tables.
    pub(super) fn gather(&self, idx: &[usize]) -> Self {
        let columns = self
            .columns
            .iter()
            .map(|(&id, col)| {
                let codes = idx
                    .iter()
                    .map(|&i| col.codes.get(i).copied().unwrap_or(DiscreteColumn::MISSING))
                    .collect();
                (id, EncodedColumn::new(codes, Arc::clone(&col.levels)))
            })
            .collect();
        Self {
            n: idx.len(),
            row_ids: idx.iter().map(|&i| self.row_ids[i]).collect(),
            columns,
            row_keys: RowKeys::default(),
        }
    }

    pub(super) fn column(&self, id: VariableId) -> Result<&EncodedColumn, EstimationError> {
        self.columns.get(&id).ok_or_else(|| EstimationError::data_msg("missing column"))
    }

    /// The mixed-radix key of every row over `ids` (first variable most significant), or
    /// [`NO_KEY`] where a level is missing; rows past the shortest column have no key.
    /// `cells` is the product of the columns' level counts, bounded by the caller.
    fn keys_over(
        &self,
        ids: &[VariableId],
        cols: &[&EncodedColumn],
        cells: usize,
    ) -> Result<Arc<[u32]>, EstimationError> {
        let mut cache = self.row_keys.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(keys) = cache.get(ids) {
            return Ok(Arc::clone(keys));
        }
        if u32::try_from(cells).map_or(true, |cells| cells >= NO_KEY) {
            return Err(EstimationError::data_msg("too many joint cells to key a CPT"));
        }
        let rows = cols.iter().map(|column| column.codes.len()).min().unwrap_or(self.n);
        let mut keys = vec![0_u32; rows];
        for column in cols {
            let radix = column.levels.len() as u32;
            for (key, &code) in keys.iter_mut().zip(&column.codes) {
                *key = if *key == NO_KEY || code == DiscreteColumn::MISSING {
                    NO_KEY
                } else {
                    *key * radix + code
                };
            }
        }
        let keys: Arc<[u32]> = Arc::from(keys);
        cache.insert(ids.to_vec(), Arc::clone(&keys));
        Ok(keys)
    }

    /// Test constructor from explicit cells (`None` = missing); levels in first-seen order.
    #[cfg(test)]
    pub(super) fn from_values(values: HashMap<VariableId, Vec<Option<Value>>>) -> Self {
        let n = values.values().next().map_or(0, Vec::len);
        let columns = values
            .into_iter()
            .map(|(id, cells)| {
                let mut levels: Vec<Value> = Vec::new();
                let codes = cells
                    .into_iter()
                    .map(|cell| match cell {
                        None => DiscreteColumn::MISSING,
                        Some(v) => {
                            let at = levels.iter().position(|l| *l == v).unwrap_or_else(|| {
                                levels.push(v);
                                levels.len() - 1
                            });
                            at as u32
                        }
                    })
                    .collect();
                (id, EncodedColumn::new(codes, Arc::from(levels)))
            })
            .collect();
        Self { n, row_ids: (0..n).collect(), columns, row_keys: RowKeys::default() }
    }

    /// Test view of one column as cells.
    #[cfg(test)]
    pub(super) fn values_of(&self, id: VariableId) -> Vec<Option<Value>> {
        let col = &self.columns[&id];
        col.codes
            .iter()
            .map(|&c| (c != DiscreteColumn::MISSING).then(|| col.levels[c as usize].clone()))
            .collect()
    }
}

impl EncodedColumn {
    fn new(codes: Vec<u32>, levels: Arc<[Value]>) -> Self {
        Self { codes, levels, observed: std::sync::OnceLock::new() }
    }

    /// Codes that occur in the sample, in order of first appearance.
    fn observed_codes(&self) -> Vec<u32> {
        self.observed
            .get_or_init(|| {
                let mut seen = vec![false; self.levels.len()];
                let mut order = Vec::new();
                for &code in &self.codes {
                    if code != DiscreteColumn::MISSING
                        && !std::mem::replace(&mut seen[code as usize], true)
                    {
                        order.push(code);
                    }
                }
                order
            })
            .clone()
    }
}

pub(super) fn provider_from_columns(
    columns: &EncodedColumns,
    n: usize,
    factors: &[FactorKey],
    signatures: &[FactorSignature],
    weights: Option<&[f64]>,
) -> Result<EmpiricalTableProvider, EstimationError> {
    let mut provider = EmpiricalTableProvider::new();
    for (&id, col) in &columns.columns {
        provider.set_domain(
            id,
            col.observed_codes().into_iter().map(|c| col.levels[c as usize].clone()),
        );
    }

    // Vacuous empty factor used by some ID edge cases.
    let empty_spec = FactorSpec {
        variables: &[],
        conditioned_on: &[],
        intervention: &[],
        domain: DomainRef::Observational,
        population: "",
        regime: None,
    };
    provider.insert_probability(&empty_spec, &Assignment::from_pairs([]), 1.0).map_err(eval_err)?;

    for (vars, cond) in factors {
        if vars.is_empty() && cond.is_empty() {
            continue;
        }
        // One row scan per (vars, cond); every alias below reads the same entries.
        let entries = cpt_entries(columns, n, vars, cond, weights)?;
        // Observational CPT.
        insert_cpt(&mut provider, vars, cond, &[], DomainRef::Observational, &entries)?;
        // Duplicate under every interventional signature with the same (vars, cond).
        for (s_vars, s_cond, interv, domain) in signatures {
            if s_vars.as_ref() != vars.as_ref() || s_cond.as_ref() != cond.as_ref() {
                continue;
            }
            if *domain == DomainRef::Observational && interv.is_empty() {
                continue;
            }
            // The intervention slot of an identified functional binds values; it does not
            // change the law the factor is read from. A factor that generates an intervened
            // variable (ID line 7 keeps `P(x | w, z)` on the napkin graph) is therefore the
            // observational conditional evaluated at the fixed value, not a point mass.
            insert_cpt(&mut provider, vars, cond, interv.as_ref(), *domain, &entries)?;
        }
    }

    Ok(provider)
}

// One random observational law per draw. All marginal/conditional factors and
// interventional aliases are derived from the same row masses, preserving
// probability identities even when the compiled expression reuses a kernel.
//
// With a block length the rows are one dependent series: one `Exp(1)` weight per
// contiguous block of rows, shared by every row of the block.
pub(super) fn provider_from_bayesian_bootstrap(
    columns: &EncodedColumns,
    n: usize,
    factors: &[FactorKey],
    signatures: &[FactorSignature],
    (block, rng): (Option<usize>, &mut CausalRng),
) -> Result<EmpiricalTableProvider, EstimationError> {
    let weights = match block {
        None => antecedent_kernels::bayesian_bootstrap_weights(rng, n, false),
        Some(block) => {
            let block = block.clamp(1, n.max(1));
            let draws =
                antecedent_kernels::bayesian_bootstrap_weights(rng, n.div_ceil(block), false);
            (0..n).map(|row| draws[row / block]).collect()
        }
    };
    provider_from_columns(columns, n, factors, signatures, Some(&weights))
}

/// The frequentist bootstrap of a functional plug-in under its row resampling and
/// declared design: single rows, or circular blocks of one series' consecutive rows.
pub(super) fn resample_rows(
    (replicates, resampling, declared): (u32, RowResampling, crate::util::SamplingDeclaration),
    ctx: &ExecutionContext,
    stream_base: u64,
    n: usize,
    estimate: impl Fn(&[usize]) -> Result<Option<f64>, EstimationError> + Sync,
) -> Result<crate::util::BootstrapSeResult, EstimationError> {
    crate::util::bootstrap_by_design(
        crate::util::BootstrapDesign {
            clusters: None,
            series_block: resampling.block_length(n),
            declared,
        },
        replicates,
        ctx,
        stream_base,
        n,
        || (),
        |(), idx| estimate(idx),
    )
}

/// Every combination of one code per variable (first variable varies slowest).
fn cartesian_codes(domains: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut rows: Vec<Vec<u32>> = vec![Vec::new()];
    for domain in domains {
        let mut next = Vec::with_capacity(rows.len() * domain.len());
        for prefix in &rows {
            for &code in domain {
                let mut row = prefix.clone();
                row.push(code);
                next.push(row);
            }
        }
        rows = next;
    }
    rows
}

/// Mixed-radix key of one code per column, `None` on `u64` overflow.
fn pack_codes(radices: &[u64], codes: impl Iterator<Item = u32>) -> Option<u64> {
    let mut key = 0u64;
    for (radix, code) in radices.iter().zip(codes) {
        key = key.checked_mul(*radix)?.checked_add(u64::from(code))?;
    }
    Some(key)
}

/// Observed level codes of each column, numerically ordered (first-seen order breaks ties).
fn ordered_domains(cols: &[&EncodedColumn]) -> Result<Vec<Vec<u32>>, EstimationError> {
    cols.iter()
        .map(|c| {
            let mut domain = c.observed_codes();
            if domain.is_empty() {
                return Err(EstimationError::data_msg("empty domain in CPT insert"));
            }
            domain.sort_by(|&a, &b| {
                match (c.levels[a as usize].as_f64(), c.levels[b as usize].as_f64()) {
                    (Some(x), Some(y)) => x.total_cmp(&y),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => a.cmp(&b),
                }
            });
            Ok(domain)
        })
        .collect()
}

/// Conditional-probability table `P(vars | cond)` as `(assignment, probability)` entries.
///
/// Depends only on the rows and weights, not on the intervention slot or domain a
/// factor is later registered under, so it is computed once per `(vars, cond)`. Counts
/// are keyed by the mixed-radix integer of the rows' level codes.
/// Bounded retained empirical factor fits, shared only under an explicit execution scope.
#[derive(Default, Debug)]
pub struct EmpiricalFactorCache {
    state: Mutex<RetainedFactorFits>,
}

/// The retained fits in insertion order with their running cost, so eviction never re-sums.
#[derive(Default, Debug)]
struct RetainedFactorFits {
    entries: Vec<RetainedFactorFit>,
    cost: usize,
}
thread_local! {static ACTIVE_FACTOR_CACHE:std::cell::RefCell<Option<std::sync::Arc<EmpiricalFactorCache>>>=const{std::cell::RefCell::new(None)};}
impl EmpiricalFactorCache {
    /// Empty bounded retained factor cache.
    #[must_use]
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::default())
    }
    /// Execute existing factor construction with exact physical identities and scoped reuse.
    pub fn scope<R>(self: &std::sync::Arc<Self>, work: impl FnOnce() -> R) -> R {
        struct Restore(Option<std::sync::Arc<EmpiricalFactorCache>>);
        impl Drop for Restore {
            fn drop(&mut self) {
                ACTIVE_FACTOR_CACHE.with(|active| *active.borrow_mut() = self.0.take());
            }
        }
        let _restore = Restore(
            ACTIVE_FACTOR_CACHE.with(|active| active.replace(Some(std::sync::Arc::clone(self)))),
        );
        work()
    }
    /// Retained numeric factor identities; these are not identification supports.
    pub fn identities(&self) -> Vec<antecedent_core::recalc::StageIdentity> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entries
            .iter()
            .map(|entry| entry.0)
            .collect()
    }
}
/// The shared, immutable entries of one fitted factor.
type FactorEntries = Arc<Vec<(Assignment, f64)>>;

/// Most retained fits.
pub(super) const MAX_RETAINED_FITS: usize = 64;
/// Retained-cost budget, in stored values.
const MAX_RETAINED_VALUES: usize = 1 << 24;

pub(super) fn cpt_entries(
    columns: &EncodedColumns,
    n: usize,
    vars: &[VariableId],
    cond: &[VariableId],
    weights: Option<&[f64]>,
) -> Result<FactorEntries, EstimationError> {
    let active = ACTIVE_FACTOR_CACHE.with(|active| active.borrow().clone());
    let Some(active) = active else {
        return cpt_entries_uncached(columns, n, vars, cond, weights).map(Arc::new);
    };
    let identity = factor_identity(columns, vars, cond, weights)?;
    let lock =
        || active.state.lock().map_err(|_| EstimationError::data_msg("factor cache lock poisoned"));
    if let Some(entry) = lock()?.entries.iter().find(|entry| entry.0 == identity) {
        return Ok(Arc::clone(&entry.1));
    }
    // The fit runs unlocked: concurrent callers on other keys are not serialized behind it.
    // Two callers racing on one key both fit; the first to insert wins and both return the
    // same values.
    let result = Arc::new(cpt_entries_uncached(columns, n, vars, cond, weights)?);
    let cost = result
        .len()
        .saturating_mul(vars.len().saturating_add(cond.len()).saturating_mul(2).saturating_add(1));
    let mut state = lock()?;
    if let Some(entry) = state.entries.iter().find(|entry| entry.0 == identity) {
        return Ok(Arc::clone(&entry.1));
    }
    if cost <= MAX_RETAINED_VALUES {
        while !state.entries.is_empty()
            && (state.entries.len() >= MAX_RETAINED_FITS
                || state.cost.saturating_add(cost) > MAX_RETAINED_VALUES)
        {
            let (_, _, evicted) = state.entries.remove(0);
            state.cost -= evicted;
        }
        state.entries.push((identity, Arc::clone(&result), cost));
        state.cost = state.cost.saturating_add(cost);
    }
    Ok(result)
}

/// Content identity of one factor fit: its variables, the row ids and level codes it reads,
/// and its weights.
///
/// The digested bytes are the `Debug` text of each part (`[0, 1, 2]` for a code vector), so
/// retained identities are stable; the code lists are written directly instead of through
/// `format!`, which dominated a cache hit.
pub(super) fn factor_identity(
    columns: &EncodedColumns,
    vars: &[VariableId],
    cond: &[VariableId],
    weights: Option<&[f64]>,
) -> Result<antecedent_core::recalc::StageIdentity, EstimationError> {
    let mut pieces: Vec<Vec<u8>> = Vec::with_capacity(vars.len() + cond.len() + 2);
    let mut head = format!("{vars:?}:{cond:?}:").into_bytes();
    push_debug_list(&mut head, columns.row_ids.iter().map(|&row| row as u64));
    pieces.push(head);
    for &id in vars.iter().chain(cond) {
        let column = columns.column(id)?;
        let mut piece = format!("{:?}:", column.levels).into_bytes();
        push_debug_list(&mut piece, column.codes.iter().map(|&code| u64::from(code)));
        pieces.push(piece);
    }
    if let Some(weights) = weights {
        pieces.push(weights.iter().flat_map(|value| value.to_bits().to_le_bytes()).collect());
    }
    Ok(antecedent_core::recalc::StageIdentity::of(
        "empirical.cpt.fit.v1",
        &pieces.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    ))
}

/// Append `values` as `{:?}` prints an integer vector: `[1, 20, 3]`.
fn push_debug_list(out: &mut Vec<u8>, values: impl ExactSizeIterator<Item = u64>) {
    out.reserve(2 + values.len() * 4);
    out.push(b'[');
    for (index, mut value) in values.enumerate() {
        if index > 0 {
            out.extend_from_slice(b", ");
        }
        let mut digits = [0u8; 20];
        let mut start = digits.len();
        loop {
            start -= 1;
            digits[start] = b'0' + (value % 10) as u8;
            value /= 10;
            if value == 0 {
                break;
            }
        }
        out.extend_from_slice(&digits[start..]);
    }
    out.push(b']');
}

pub(super) fn cpt_entries_uncached(
    columns: &EncodedColumns,
    n: usize,
    vars: &[VariableId],
    cond: &[VariableId],
    weights: Option<&[f64]>,
) -> Result<Vec<(Assignment, f64)>, EstimationError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::FactorConstruction,
        || construct_cpt_entries(columns, n, vars, cond, weights),
    )
}

fn construct_cpt_entries(
    columns: &EncodedColumns,
    n: usize,
    vars: &[VariableId],
    cond: &[VariableId],
    weights: Option<&[f64]>,
) -> Result<Vec<(Assignment, f64)>, EstimationError> {
    let var_cols: Vec<&EncodedColumn> =
        vars.iter().map(|&v| columns.column(v)).collect::<Result<_, _>>()?;
    let cond_cols: Vec<&EncodedColumn> =
        cond.iter().map(|&v| columns.column(v)).collect::<Result<_, _>>()?;
    // Bound the complete table before enumerating either Cartesian domain.
    // The retained-cache budget cannot protect an uncached allocation.
    let cells = var_cols.iter().chain(&cond_cols).try_fold(1usize, |cells, column| {
        cells.checked_mul(column.levels.len()).filter(|&product| product <= 1_000_000)
    });
    if cells.is_none() {
        return Err(EstimationError::refused_detail(
            antecedent_core::reason_code!("invalid_argument"),
            "functional.factor_cell_budget",
            "empirical factor exceeds one million Cartesian cells",
        ));
    }
    let radices = |cols: &[&EncodedColumn]| -> Vec<u64> {
        cols.iter().map(|c| c.levels.len() as u64).collect()
    };
    let (var_radices, cond_radices) = (radices(&var_cols), radices(&cond_cols));
    let joint_radices: Vec<u64> = var_radices.iter().chain(cond_radices.iter()).copied().collect();
    let too_many = || EstimationError::data_msg("too many joint cells to key a CPT");

    // Count (vars, cond) joint and cond marginal among complete cases. The table is bounded
    // above, so counts are held densely under the rows' mixed-radix key: every cell
    // accumulates its rows in row order.
    let cond_cells: usize = cond_radices.iter().map(|&radix| radix as usize).product();
    let var_cells: usize = var_radices.iter().map(|&radix| radix as usize).product();
    let mut joint = vec![0.0_f64; var_cells * cond_cells];
    let mut marg = vec![0.0_f64; cond_cells];
    let mut total = 0.0;
    let cond_keys = columns.keys_over(cond, &cond_cols, cond_cells)?;
    let var_keys = columns.keys_over(vars, &var_cols, var_cells)?;
    for (row, (&cond_key, &var_key)) in cond_keys.iter().zip(var_keys.iter()).enumerate().take(n) {
        let weight = weights.map_or(1.0, |w| w.get(row).copied().unwrap_or(0.0));
        if weight <= 0.0 || cond_key == NO_KEY || var_key == NO_KEY {
            continue;
        }
        let (cond_key, var_key) = (cond_key as usize, var_key as usize);
        *marg.get_mut(cond_key).ok_or_else(too_many)? += weight;
        *joint.get_mut(var_key * cond_cells + cond_key).ok_or_else(too_many)? += weight;
        total += weight;
    }
    let count_at = |counts: &[f64], key: u64| -> f64 {
        usize::try_from(key).ok().and_then(|key| counts.get(key)).copied().unwrap_or(0.0)
    };

    let level_of = |cols: &[&EncodedColumn], k: usize, code: u32| -> Value {
        cols[k].levels[code as usize].clone()
    };

    let mut entries = Vec::new();
    let var_rows = cartesian_codes(&ordered_domains(&var_cols)?);
    if cond.is_empty() {
        if total <= 0.0 {
            return Err(EstimationError::data_msg("no complete cases for CPT"));
        }
        for var_codes in &var_rows {
            let key = pack_codes(&var_radices, var_codes.iter().copied()).ok_or_else(too_many)?;
            let count = count_at(&joint, key);
            let assign = Assignment::from_pairs(
                vars.iter()
                    .copied()
                    .zip(var_codes.iter().enumerate().map(|(k, &c)| level_of(&var_cols, k, c))),
            );
            entries.push((assign, count / total));
        }
    } else {
        let cond_rows = cartesian_codes(&ordered_domains(&cond_cols)?);
        for cond_codes in &cond_rows {
            let cond_key =
                pack_codes(&cond_radices, cond_codes.iter().copied()).ok_or_else(too_many)?;
            let cond_count = count_at(&marg, cond_key);
            // An unobserved conditioning cell is not P=0. Leave it absent so
            // `probability()` returns `MissingTableEntry` — the evaluator's
            // existing "can I evaluate this cell?" contract.
            if cond_count <= 0.0 {
                continue;
            }
            for var_codes in &var_rows {
                let joint_key =
                    pack_codes(&joint_radices, var_codes.iter().chain(cond_codes).copied())
                        .ok_or_else(too_many)?;
                let count = count_at(&joint, joint_key);
                let p = count / cond_count;
                let assign = Assignment::from_pairs(
                    vars.iter()
                        .copied()
                        .zip(var_codes.iter().enumerate().map(|(k, &c)| level_of(&var_cols, k, c)))
                        .chain(cond.iter().copied().zip(
                            cond_codes.iter().enumerate().map(|(k, &c)| level_of(&cond_cols, k, c)),
                        )),
                );
                entries.push((assign, p));
            }
        }
    }
    antecedent_expr::execution_counts::note_static_work(
        antecedent_expr::execution_counts::StaticWork::FactorBuild,
    );
    Ok(entries)
}

/// Register precomputed [`cpt_entries`] under one intervention slot and domain.
fn insert_cpt(
    provider: &mut EmpiricalTableProvider,
    vars: &[VariableId],
    cond: &[VariableId],
    intervention: &[InterventionAssignment],
    domain: DomainRef,
    entries: &[(Assignment, f64)],
) -> Result<(), EstimationError> {
    let spec = FactorSpec {
        variables: vars,
        conditioned_on: cond,
        intervention,
        domain,
        population: "",
        regime: None,
    };
    for (assign, probability) in entries {
        provider.insert_probability(&spec, assign, *probability).map_err(eval_err)?;
    }
    Ok(())
}
