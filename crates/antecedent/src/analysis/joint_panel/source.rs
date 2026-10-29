//! Exact panel source adaptation and execution dependencies.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::data_policy::{
    PanelCellRole, PanelDataAccounting, PanelMissingPolicy, PanelUnitAccounting, PanelUnitDrop,
};
use super::{JointPanelError, JointPanelRequest, JointPanelState, JointPanelWorld, engine};
use antecedent_core::execution_attempt::{PhysicalEvent, record_physical};
use antecedent_core::physical_choice::PhysicalChoice;
use antecedent_core::physical_choice::{
    AvailableCapabilities, CostQuotes, DeclaredCost, OperatorCapability, ResourceCapability,
};
use antecedent_core::{CanonicalWriter, Determinism};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_data::PanelData;
use antecedent_data::{ColumnView, SamplingRegularity, TableView};

pub(super) fn check_cancelled(ctx: &ExecutionContext) -> Result<(), JointPanelError> {
    if ctx.cancellation.is_cancelled() { Err(engine::PanelError::Cancelled.into()) } else { Ok(()) }
}
pub(super) fn check_memory(ctx: &ExecutionContext, bytes: u64) -> Result<(), JointPanelError> {
    if ctx.memory_limit_bytes().is_some_and(|limit| bytes > limit) {
        return Err(engine::PanelError::Resource(format!(
            "retained panel allocation needs {bytes} bytes above hard limit"
        ))
        .into());
    }
    Ok(())
}
fn overflow() -> JointPanelError {
    engine::PanelError::Resource("panel size overflow".into()).into()
}
fn add(total: &mut usize, value: usize) -> Result<(), JointPanelError> {
    *total = total.checked_add(value).ok_or_else(overflow)?;
    Ok(())
}
pub(super) fn source_quote(
    source: &PanelData,
    request: &JointPanelRequest,
) -> Result<u64, JointPanelError> {
    let mut bytes = std::mem::size_of::<JointPanelState>();
    add(&mut bytes, std::mem::size_of::<JointPanelRequest>())?;
    for variable in source.schema().variables() {
        add(&mut bytes, std::mem::size_of_val(variable))?;
        add(&mut bytes, variable.name.len())?;
        if let Some(unit) = &variable.unit {
            add(&mut bytes, unit.len())?;
        }
        if let Some(description) = &variable.measurement.description {
            add(&mut bytes, description.len())?;
        }
    }
    for channel in &request.columns.channels {
        add(
            &mut bytes,
            std::mem::size_of::<engine::PanelChannel>() + channel.raw().to_string().len(),
        )?;
    }
    for interaction in &request.interactions {
        let nodes = wire_nodes(&interaction.expression, 1)?;
        let wire_bytes = nodes
            .checked_mul(std::mem::size_of::<antecedent_io::query_wire::ValueExprWire>())
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(overflow)?;
        add(&mut bytes, wire_bytes)?;
        add(&mut bytes, interaction.id.len().checked_mul(2).ok_or_else(overflow)?)?;
        let plan = antecedent_model::program::NodeProgram::column_compile_bytes(
            request.columns.channels.len(),
            nodes,
        )
        .map_err(JointPanelError::compile)?;
        add(&mut bytes, usize::try_from(plan).map_err(|_| overflow())?)?;
    }
    if let Some(spatial) = &request.family.spatial {
        // The request and the engine's source each retain the distance matrix.
        add(
            &mut bytes,
            spatial
                .distances
                .len()
                .checked_mul(16)
                .and_then(|n| n.checked_add(spatial.groups.iter().map(String::len).sum::<usize>()))
                .ok_or_else(overflow)?,
        )?;
    }
    for unit in source.units() {
        add(&mut bytes, unit_quote(unit, request)?)?;
    }
    add(&mut bytes, program_quote(&request.worlds)?)?;
    u64::try_from(bytes).map_err(|_| overflow())
}

fn program_quote(worlds: &[JointPanelWorld]) -> Result<usize, JointPanelError> {
    let mut bytes = 0_usize;
    for world in worlds {
        add(&mut bytes, std::mem::size_of::<JointPanelWorld>())?;
        add(&mut bytes, world.program.id.len())?;
        for channel in &world.program.channels {
            add(&mut bytes, std::mem::size_of::<engine::PanelChannelProgram>())?;
            add(&mut bytes, channel.id.len())?;
            add(&mut bytes, channel.values.len().checked_mul(8).ok_or_else(overflow)?)?;
        }
        for scenario in &world.program.scenarios {
            add(&mut bytes, std::mem::size_of::<engine::PanelScenario>())?;
            add(&mut bytes, scenario.group.len())?;
            for history in &scenario.history {
                add(&mut bytes, std::mem::size_of::<engine::PanelChannelHistory>())?;
                add(&mut bytes, history.id.len())?;
            }
            add(
                &mut bytes,
                scenario
                    .confounder
                    .len()
                    .checked_add(scenario.season.len())
                    .and_then(|n| n.checked_mul(8))
                    .ok_or_else(overflow)?,
            )?;
        }
    }
    Ok(bytes)
}
/// Retained temporal-parameter declarations are distinct from world/output storage.
fn add_temporal_spec_bytes(
    bytes: &mut usize,
    spec: &engine::time_varying::TemporalPanelSpec,
) -> Result<(), JointPanelError> {
    use engine::time_varying::PanelMeanParameter;
    for parameter in &spec.varying {
        add(bytes, std::mem::size_of_val(parameter))?;
        match parameter {
            PanelMeanParameter::Channel { id } | PanelMeanParameter::Interaction { id } => {
                add(bytes, id.len())?
            }
            PanelMeanParameter::Lag { channel, .. }
            | PanelMeanParameter::GeometricDecay { channel }
            | PanelMeanParameter::HillShape { channel } => add(bytes, channel.len())?,
            PanelMeanParameter::Intercept
            | PanelMeanParameter::Confounder
            | PanelMeanParameter::Season { .. }
            | PanelMeanParameter::ClassLogHalfSaturation { .. }
            | PanelMeanParameter::LogHalfSaturation => {}
        }
    }
    add(
        bytes,
        spec.change_points
            .len()
            .checked_add(spec.smooth_knots.len())
            .and_then(|n| n.checked_mul(std::mem::size_of::<usize>()))
            .ok_or_else(overflow)?,
    )?;
    add(bytes, spec.smoothing.len().checked_mul(8).ok_or_else(overflow)?)?;
    Ok(())
}

pub(super) fn retarget_quote(
    request: &JointPanelRequest,
    worlds: &[JointPanelWorld],
    target: &antecedent_core::ScientificTarget,
) -> Result<u64, JointPanelError> {
    let mut bytes = std::mem::size_of::<JointPanelRequest>();
    // One retained program plus lower's simultaneously validated program copies.
    add(&mut bytes, program_quote(worlds)?.checked_mul(2).ok_or_else(overflow)?)?;
    add(
        &mut bytes,
        request
            .columns
            .channels
            .len()
            .checked_mul(std::mem::size_of::<antecedent_core::VariableId>())
            .ok_or_else(overflow)?,
    )?;
    for feature in &request.interactions {
        add(&mut bytes, feature.id.len() + std::mem::size_of_val(feature))?;
        add(
            &mut bytes,
            wire_nodes(&feature.expression, 1)?
                .checked_mul(std::mem::size_of::<antecedent_io::query_wire::ValueExprWire>())
                .ok_or_else(overflow)?,
        )?;
    }
    if let Some(spatial) = &request.family.spatial {
        add(
            &mut bytes,
            spatial
                .distances
                .len()
                .checked_mul(8)
                .and_then(|n| n.checked_add(spatial.groups.iter().map(String::len).sum::<usize>()))
                .ok_or_else(overflow)?,
        )?;
    }
    if let engine::PanelInference::Bayesian { prior, .. } = &request.options.inference {
        add(
            &mut bytes,
            prior.lag.len().checked_mul(std::mem::size_of::<[f64; 3]>()).ok_or_else(overflow)?,
        )?;
    }
    if let Some(initial) = &request.options.initial {
        add(&mut bytes, std::mem::size_of_val(initial))?;
        for channel in &initial.channels {
            add(&mut bytes, std::mem::size_of_val(channel) + channel.id.len())?;
        }
        for feature in &initial.interactions {
            add(&mut bytes, std::mem::size_of_val(feature) + feature.id.len())?;
        }
        add(&mut bytes, initial.season_effects.len().checked_mul(8).ok_or_else(overflow)?)?;
    }
    if let Some(spec) = &request.temporal {
        add_temporal_spec_bytes(&mut bytes, spec)?;
    }
    add(
        &mut bytes,
        target
            .assumptions()
            .records()
            .len()
            .checked_mul(std::mem::size_of::<antecedent_core::AssumptionRecord>())
            .ok_or_else(overflow)?,
    )?;
    for world in worlds {
        let terms =
            world.horizon().checked_mul(request.columns.channels.len()).ok_or_else(overflow)?;
        let term_width = std::mem::size_of::<antecedent_core::Intervention>()
            + std::mem::size_of::<antecedent_core::SequencedIntervention>()
            + std::mem::size_of::<antecedent_core::Value>();
        add(&mut bytes, terms.checked_mul(term_width).ok_or_else(overflow)?)?;
        add(
            &mut bytes,
            std::mem::size_of::<antecedent_core::InterventionWorld>()
                + std::mem::size_of::<super::JointPanelWorldSupport>(),
        )?;
        for scenario in &world.program.scenarios {
            add(&mut bytes, scenario.group.len() + std::mem::size_of::<String>())?;
        }
    }
    u64::try_from(bytes).map_err(|_| overflow())
}

fn unit_quote(
    unit: &antecedent_data::PanelUnit,
    request: &JointPanelRequest,
) -> Result<usize, JointPanelError> {
    let mut bytes = 0_usize;
    add(&mut bytes, std::mem::size_of::<engine::PanelGroup>())?;
    add(&mut bytes, std::mem::size_of_val(unit))?;
    add(
        &mut bytes,
        request
            .columns
            .channels
            .len()
            .checked_mul(std::mem::size_of::<Vec<f64>>() + std::mem::size_of::<[f64; 2]>())
            .ok_or_else(overflow)?,
    )?;
    add(&mut bytes, unit.unit_id.to_string().len())?;
    let storage = unit.series.storage();
    for column in storage.columns() {
        let view = column.as_view();
        let (n, width) = match view {
            ColumnView::Float64(c) => (c.values.len(), 8),
            ColumnView::Int64(c) => (c.values.len(), 8),
            ColumnView::Boolean(c) => (c.values.len(), 1),
            ColumnView::Timestamp(c) => (c.values_ns.len(), 8),
            ColumnView::Categorical(c) => {
                for level in c.domain.levels.iter() {
                    add(&mut bytes, level.label.len())?;
                }
                (c.codes.len(), 4)
            }
            ColumnView::FixedVector(c) => (c.values.len(), 8),
        };
        add(&mut bytes, n.checked_mul(width).ok_or_else(overflow)?)?;
        add(&mut bytes, view.len().div_ceil(8))?;
    }
    if let Some(weights) = storage.weights() {
        add(&mut bytes, weights.len().checked_mul(8).ok_or_else(overflow)?)?;
    }
    if storage.analysis_mask().is_some() {
        add(&mut bytes, unit.series.row_count().div_ceil(8))?;
    }
    // Actual retained numeric columns and simultaneous coercion buffers.
    let rows = unit.series.row_count();
    add(
        &mut bytes,
        rows.saturating_sub(request.lag_order)
            .checked_mul(request.columns.channels.len().checked_add(3).ok_or_else(overflow)?)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(overflow)?,
    )?;
    for role in request.columns.channels.iter().copied().chain([
        request.columns.outcome,
        request.columns.confounder,
        request.columns.season,
    ]) {
        if !matches!(
            unit.series.column(role).map_err(JointPanelError::compile)?,
            ColumnView::Float64(_)
        ) {
            add(&mut bytes, rows.checked_mul(8).ok_or_else(overflow)?)?;
        }
    }
    Ok(bytes)
}
pub(super) fn adapt(
    source: &PanelData,
    request: &JointPanelRequest,
    policy: PanelMissingPolicy,
    ctx: &ExecutionContext,
) -> Result<(engine::PanelData, SemanticDigest, PanelDataAccounting), JointPanelError> {
    let c = &request.columns;
    let mut roles = c.channels.clone();
    roles.extend([c.outcome, c.confounder, c.season]);
    if roles.iter().copied().collect::<std::collections::BTreeSet<_>>().len() != roles.len()
        || c.channels.is_empty()
    {
        return Err(JointPanelError::compile("panel roles must be distinct"));
    }
    let mut writer = CanonicalWriter::new("antecedent.compiler.joint_panel.source.v1");
    // Full schema semantics are bound, independently of display-only program labels.
    writer.bytes(
        &antecedent_io::to_cbor(&antecedent_io::schema_to_wire(source.schema()))
            .map_err(JointPanelError::compile)?,
    );
    for role in roles {
        source.schema().get(role).map_err(JointPanelError::compile)?;
        writer.word(u64::from(role.raw()));
    }
    let prefix = request.lag_order;
    if !matches!(prefix, 0 | 2) {
        return Err(JointPanelError::compile("only lag orders zero and two are supported"));
    }
    let weighted_units =
        source.units().iter().filter(|unit| unit.series.storage().weights().is_some()).count();
    if weighted_units != 0 && weighted_units != source.units().len() {
        return Err(JointPanelError::compile(
            "sampling weights must be carried by every unit or by none: a unit without one has no declared inclusion probability",
        ));
    }
    let mut groups = Vec::with_capacity(source.units().len());
    let mut declared = Vec::with_capacity(source.units().len());
    let mut units = Vec::with_capacity(source.units().len());
    let mut cadence = None;
    for unit in source.units() {
        let (group, accounting) =
            adapt_group(unit, request, policy, ctx, &mut cadence, &mut writer)?;
        if let Some(group) = group {
            declared.push(accounting.declared_weight);
            groups.push(group);
        }
        units.push(accounting);
    }
    let weighted = normalize_weights(&mut groups, &declared)?;
    let accounting = PanelDataAccounting { policy, weighted, units };
    if policy != PanelMissingPolicy::Refuse || weighted || !accounting.is_complete() {
        writer.word(0x4441_5441_504f_4c).word(policy.tag()).word(u64::from(weighted));
    }
    let data = engine::PanelData {
        groups,
        channels: c
            .channels
            .iter()
            .map(|id| engine::PanelChannel { id: id.raw().to_string() })
            .collect(),
        interactions: request.interactions.clone(),
        seasons: request.seasons,
        lag_order: prefix,
        family: request.family.clone(),
    };
    writer.bytes(data.identity()?.as_bytes());
    Ok((data, SemanticDigest::from_bytes(writer.finish()), accounting))
}
/// Relative unit sampling weights scaled to mean one. Equal weights are the unweighted
/// panel: they change neither the estimate nor the sandwich, so they are stored as one.
fn normalize_weights(
    groups: &mut [engine::PanelGroup],
    declared: &[f64],
) -> Result<bool, JointPanelError> {
    let Some(&first) = declared.first() else { return Ok(false) };
    if declared.iter().all(|weight| weight.to_bits() == first.to_bits()) {
        for group in groups {
            group.weight = 1.0;
        }
        return Ok(false);
    }
    let mean = declared.iter().sum::<f64>() / declared.len() as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return Err(JointPanelError::compile("sampling weights overflow when normalized"));
    }
    for (group, weight) in groups.iter_mut().zip(declared) {
        group.weight = weight / mean;
    }
    Ok(true)
}
/// The follow-up periods whose own cells and whole exposure window are available.
struct RowAvailability {
    /// Every assignment channel of the row is observed and the row is not masked.
    channels: Vec<bool>,
    /// Outcome, confounder and season of the row are observed and the row is not masked.
    own: Vec<bool>,
}
impl RowAvailability {
    fn usable(&self, row: usize, prefix: usize) -> bool {
        self.own[row]
            && self.channels[row]
            && (prefix == 0 || (self.channels[row - 1] && self.channels[row - 2]))
    }
}
#[expect(
    clippy::too_many_lines,
    reason = "one unit's masks, missing cells, weights and rows are read and accounted together"
)]
fn adapt_group(
    unit: &antecedent_data::PanelUnit,
    request: &JointPanelRequest,
    policy: PanelMissingPolicy,
    ctx: &ExecutionContext,
    cadence: &mut Option<u64>,
    writer: &mut CanonicalWriter,
) -> Result<(Option<engine::PanelGroup>, PanelUnitAccounting), JointPanelError> {
    let c = &request.columns;
    let prefix = request.lag_order;
    check_cancelled(ctx)?;
    let series = &unit.series;
    let interval = match series.time_index().regularity {
        SamplingRegularity::Regular { interval_ns } if interval_ns > 0 => interval_ns,
        _ => {
            return Err(JointPanelError::compile(
                "panel requires explicit positive regular cadence",
            ));
        }
    };
    if cadence.is_some_and(|previous| previous != interval)
        || series.time_index().length != series.row_count()
    {
        return Err(JointPanelError::compile("panel cadence or time-index length differs"));
    }
    *cadence = Some(interval);
    let rows = series.row_count();
    if rows < prefix + 3 {
        return Err(JointPanelError::compile(
            "panel needs observed prehistory and at least three follow-up rows",
        ));
    }
    writer.word(u64::from(unit.unit_id)).word(interval).bytes(&series.storage().content_digest());
    let channel_values = c
        .channels
        .iter()
        .map(|&id| series.float64_cow(id).map_err(JointPanelError::compile))
        .collect::<Result<Vec<_>, _>>()?;
    let y = series.float64_cow(c.outcome).map_err(JointPanelError::compile)?;
    let z = series.float64_cow(c.confounder).map_err(JointPanelError::compile)?;
    let season = series.float64_cow(c.season).map_err(JointPanelError::compile)?;

    // Rows the analysis mask hides, and the cells the analysis needs but the source lacks.
    // History rows contribute their assignments only; follow-up rows contribute all five.
    let mask = series.storage().analysis_mask();
    let masked_rows: Vec<usize> =
        (0..rows).filter(|&row| mask.is_some_and(|mask| !mask.is_valid(row))).collect();
    let mut missing_cells = Vec::new();
    for row in 0..rows {
        if masked_rows.binary_search(&row).is_ok() {
            // A hidden row is not read, so what it lacks is not missing from the analysis.
            continue;
        }
        for (index, values) in channel_values.iter().enumerate() {
            if values[row].is_nan() {
                missing_cells.push((row, PanelCellRole::Channel(index)));
            }
        }
        if row >= prefix {
            for (values, role) in [
                (&y, PanelCellRole::Outcome),
                (&z, PanelCellRole::Confounder),
                (&season, PanelCellRole::Season),
            ] {
                if values[row].is_nan() {
                    missing_cells.push((row, role));
                }
            }
        }
    }
    if !masked_rows.is_empty() || !missing_cells.is_empty() || policy != PanelMissingPolicy::Refuse
    {
        writer.word(0x4d41_534b).word(masked_rows.len() as u64);
        for &row in &masked_rows {
            writer.word(row as u64);
        }
        writer.word(missing_cells.len() as u64);
    }

    // Sampling weight: one value for the whole unit. A weight that varies over a serially
    // dependent unit has no pseudo-likelihood: the unit's law is joint over its periods, so
    // there is no per-period term to weight.
    let declared_weight = match series.storage().weights() {
        None => 1.0,
        Some(weights) => {
            let mut kept = (0..rows).filter(|row| masked_rows.binary_search(row).is_err());
            let first = kept.next().map_or(f64::NAN, |row| weights[row]);
            if !first.is_finite()
                || first <= 0.0
                || kept.any(|row| weights[row].to_bits() != first.to_bits())
            {
                return Err(JointPanelError::compile(format!(
                    "unit {} sampling weights must be one strictly positive finite value over the unit: the unit's periods are jointly dependent, so a per-period weight has no pseudo-likelihood term",
                    unit.unit_id
                )));
            }
            writer.word(0x5747_5453).word(first.to_bits());
            first
        }
    };
    let periods = rows - prefix;
    let accounting = |analysed: usize, excluded: Vec<usize>, dropped| PanelUnitAccounting {
        unit_id: unit.unit_id,
        rows,
        masked_rows: masked_rows.clone(),
        missing_cells: missing_cells.clone(),
        periods,
        analysed_periods: analysed,
        excluded_periods: excluded,
        declared_weight,
        dropped,
    };
    match policy {
        PanelMissingPolicy::Refuse if !missing_cells.is_empty() => {
            if missing_cells.iter().any(|(_, role)| *role == PanelCellRole::Season) {
                return Err(JointPanelError::compile(
                    "season must be a finite declared integer level",
                ));
            }
            let (row, role) = missing_cells[0];
            return Err(engine::PanelError::Invalid(format!(
                "group IDs, finite values, cadence or vector dimensions: unit {} has a missing cell ({role:?}) at row {row}; declare a missing-data policy to analyse it",
                unit.unit_id
            ))
            .into());
        }
        PanelMissingPolicy::DropUnits if !missing_cells.is_empty() => {
            return Ok((None, accounting(0, (0..periods).collect(), Some(PanelUnitDrop::MissingCell))));
        }
        _ => {}
    }

    let availability = RowAvailability {
        channels: (0..rows)
            .map(|row| {
                masked_rows.binary_search(&row).is_err()
                    && channel_values.iter().all(|values| !values[row].is_nan())
            })
            .collect(),
        own: (0..rows)
            .map(|row| {
                row < prefix
                    || (masked_rows.binary_search(&row).is_err()
                        && !y[row].is_nan()
                        && !z[row].is_nan()
                        && !season[row].is_nan())
            })
            .collect(),
    };
    let mut levels = Vec::with_capacity(periods);
    let mut outcome = Vec::with_capacity(periods);
    let mut confounder = Vec::with_capacity(periods);
    let mut excluded = Vec::new();
    for row in prefix..rows {
        let usable = availability.usable(row, prefix);
        if !usable {
            excluded.push(row - prefix);
        }
        outcome.push(if usable { y[row] } else { f64::NAN });
        confounder.push(if availability.own[row] { z[row] } else { 0.0 });
        let value = if availability.own[row] { season[row] } else { 0.0 };
        if !value.is_finite()
            || value < 0.0
            || value.fract() != 0.0
            || value >= request.seasons as f64
        {
            return Err(JointPanelError::compile("season must be a finite declared integer level"));
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "finite nonnegative integer strictly below declared usize season count checked above"
        )]
        let level = value as usize;
        levels.push(level);
    }
    let analysed = periods - excluded.len();
    if analysed < 3 {
        return Ok((
            None,
            accounting(analysed, excluded, Some(PanelUnitDrop::TooFewAnalysedPeriods { analysed })),
        ));
    }
    let fill = |values: &[f64], row: usize| {
        if availability.channels[row] || values[row].is_infinite() { values[row] } else { 0.0 }
    };
    let history = channel_values
        .iter()
        .map(|values| if prefix == 2 { [fill(values, 0), fill(values, 1)] } else { [0.0; 2] })
        .collect();
    let channels = channel_values
        .iter()
        .map(|values| (prefix..rows).map(|row| fill(values, row)).collect())
        .collect();
    let group = engine::PanelGroup {
        id: unit.unit_id.to_string(),
        history,
        channels,
        outcome,
        confounder,
        season: levels,
        weight: 1.0,
    };
    record_physical(PhysicalEvent::Scan, (c.channels.len() + 3) as u64);
    record_physical(PhysicalEvent::Copy, (c.channels.len() + 3) as u64);
    record_physical(
        PhysicalEvent::CopiedBytes,
        ((series.row_count() - prefix) * (c.channels.len() + 3) * 8) as u64,
    );
    Ok((Some(group), accounting(analysed, excluded, None)))
}
pub(super) fn physical_choice(data: &engine::PanelData) -> Result<PhysicalChoice, JointPanelError> {
    let rows = data.groups.iter().map(|group| group.outcome.len() as u64).sum::<u64>();
    let cost = DeclaredCost::from_raw(rows.saturating_mul(
        (data.seasons + 8 + data.channels.len() * 4 + data.interactions.len()) as u64,
    ));
    PhysicalChoice::choose(
        CostQuotes::new(cost, cost, cost, cost),
        AvailableCapabilities::empty()
            .with_operator(OperatorCapability::PreparedNumericView)
            .with_resource(ResourceCapability::PreparedColumns),
    )
    .map_err(JointPanelError::compile)
}
pub(super) fn fit_dependency(
    state: &JointPanelState,
    ctx: &ExecutionContext,
) -> Result<SemanticDigest, JointPanelError> {
    let mut writer = CanonicalWriter::new("antecedent.compiler.joint_panel.fit.v1");
    writer
        .bytes(state.source_identity.as_bytes())
        .bytes(state.options_identity.as_bytes())
        .word(ctx.rng.master_seed())
        .word(u64::from(matches!(ctx.determinism, Determinism::Strict)))
        .word(u64::from(ctx.kernel_policy.allow_portable_optimized))
        .word(u64::from(ctx.kernel_policy.allow_arch_simd))
        .word(u64::from(ctx.kernel_policy.force_scalar));
    if let Some(spec) = &state.request.temporal {
        writer.word(1).bytes(spec.identity(&state.data)?.as_bytes());
    } else {
        writer.word(0);
    }
    Ok(SemanticDigest::from_bytes(writer.finish()))
}
pub(super) fn observe_reuse() {
    record_physical(PhysicalEvent::CacheHit, 1);
}

fn wire_nodes(
    expression: &antecedent_io::query_wire::ValueExprWire,
    depth: usize,
) -> Result<usize, JointPanelError> {
    use antecedent_io::query_wire::ValueExprWire as Expr;
    if depth > antecedent_core::value_expr::MAX_EXPR_DEPTH {
        return Err(JointPanelError::compile(
            "interaction expression exceeds the common depth bound",
        ));
    }
    let next = depth + 1;
    let children = match expression {
        Expr::Const { .. } | Expr::Generated | Expr::Var { .. } | Expr::Lag { .. } => 0,
        Expr::Neg { operand } | Expr::Not { operand } | Expr::Indicator { operand } => {
            wire_nodes(operand, next)?
        }
        Expr::Arith { lhs, rhs, .. }
        | Expr::Compare { lhs, rhs, .. }
        | Expr::Logic { lhs, rhs, .. } => {
            wire_nodes(lhs, next)?.checked_add(wire_nodes(rhs, next)?).ok_or_else(overflow)?
        }
        Expr::Clip { value, lower, upper } => {
            let a = wire_nodes(value, next)?;
            let b = wire_nodes(lower, next)?;
            let c = wire_nodes(upper, next)?;
            a.checked_add(b).and_then(|n| n.checked_add(c)).ok_or_else(overflow)?
        }
        Expr::If { condition, then_branch, else_branch } => {
            let a = wire_nodes(condition, next)?;
            let b = wire_nodes(then_branch, next)?;
            let c = wire_nodes(else_branch, next)?;
            a.checked_add(b).and_then(|n| n.checked_add(c)).ok_or_else(overflow)?
        }
        Expr::Callback { .. } => {
            return Err(JointPanelError::compile(
                "panel features require callback-free common expressions",
            ));
        }
    };
    let nodes = children.checked_add(1).ok_or_else(overflow)?;
    if nodes > antecedent_core::value_expr::MAX_EXPR_NODES {
        return Err(JointPanelError::compile(
            "interaction expression exceeds the common node bound",
        ));
    }
    Ok(nodes)
}
