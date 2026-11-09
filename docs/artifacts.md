# Artifact format

Library package version is tracked independently from the durable artifact format.
The durable format is **`FormatVersion { major: 0, minor: 5 }`**
(`antecedent_io::STABLE_FORMAT`). Format 0.5 includes the identified-set interval
on structural-mixture analysis results (below); format 0.4 added temporal
response-query fields and dose × horizon response surfaces.

A wire change advances the format and provides migration from prior stable
versions; a package-version change alone does not change artifact bytes.

## Python persistence

`outcome.export(...)` returns an immutable `ant.artifacts.ExportedArtifact[T]`: the container
bytes (`bytes(artifact)` or `.data`), a `.kind`, a typed `.expectation` when available, and
family-specific `.metadata`. Write them with `Path.write_bytes` and read them with
`Path.read_bytes`; persistence does not take file paths implicitly. Ledger carried/not-carried
receipts remain in this metadata. Operations that select a query or record retain their explicit
selection arguments (`quantity=`).

```python
import antecedent as ant

artifact = result.export()
path.write_bytes(bytes(artifact))
# Store this expectation in an independent trusted record, separate from the bytes.
expected = artifact.expectation
metadata = ant.artifacts.inspect(path.read_bytes())
loaded = ant.load(path.read_bytes(), expected=expected)
```

`ant.load` dispatches registered scientific families to their existing native consumers
and returns the family-specific consumed result in an `Outcome`. Required expectations
remain required: container metadata never supplies an independent identity binding.
For bundle consumption,
`ant.artifacts.LoadContext(supplied_sources=..., actual_claims=...)` passes existing source
declarations and actual native claims to the consumer. Loading never invokes a callback or
reconstructs a live study.

Acceptance reports its status (`loaded`, `replayed` or `unavailable`), identity binding
and explicit resumability separately. `ant.artifacts.inspect(...)` is metadata
inspection, not acceptance of a scientific claim. Unknown recognized families can be
inspected and forwarded with `export()` while scientific consumption remains
unavailable; malformed containers are
errors. Consumed read models are recursively immutable, and `to_dict()` produces an
independent JSON-safe editable copy.

See [the notebook-to-system journey](guides/notebook-to-system.md) and
[the migration guide](migration.md).

<a id="package-15-payloads"></a>

## Score tables and distribution payloads

Estimates can carry score tables, per-arm CDF values, raw-score
inference and covariance, and candidate-selection provenance. These fields have
separate meanings:

- `score_table` retains cross-fitted scores, column metadata, original row
  indices, held-out propensities, and the certified adjustment set.
- `exceedance_cdf` is the projected per-arm CDF. AIPW score-table inference
  reports raw exceedance coordinates; the conditional-grid path reports raw CDF
  coordinates. To complement exceedance bands, use `[1-upper, 1-lower]`.
  Projection does not turn raw-score bands into uncertainty for the projected curve.
- Unsupported coordinates carry false support flags. Bounds can be NaN or
  finite placeholders; neither supports an interval when the flag is false.
  CBOR preserves NaN bounds. Do not convert them into zero-width intervals or
  assume a generic JSON round trip preserves non-finite floats.
- Multi-threshold estimates have no scalar ATE. Candidate-selection records
  describe screening and sample splits; their presence does not establish
  selection-adjusted inference.

Search-budget diagnostics use the `IdentificationStatus` encoding; a cap has no
status variant of its own. That compatibility statement does not establish
forward compatibility for new outcome functionals or all optional payloads. Do
not assume an older reader can interpret or preserve them. Use a
reader that supports the requested functional and retain support/uncertainty
fields when handing estimates to downstream applications.

## Container

See ADR 0002 / 0017:

```text
magic (CAUSAL\0\0) | container_version (u32 LE = 1)
| manifest_len (u32 LE) | canonical CBOR manifest
| repeated: section_len (u32 LE) | section bytes
```

Manifest fields: `format_version`, `minimum_reader_version`, `artifact_kind`,
`library_version`, `artifact_id`, `sections[]`, `provenance`.

### External-execution artifacts

The three artifacts that record a foreign execution use the shared container
and load through `antecedent.artifacts.loads`, which returns the *declared*
header (a producer's `trust` is a claim there, never evidence):

| Kind | Sections | `payload_kind` |
|---|---|---|
| `provider_result_v1` | `provider_result.header` (CBOR), `provider_result.artifact` (the provider's opaque bytes) | `provider_result` |
| `external_callback_v1` | `external_callback.meta` (CBOR: request and provider digests, policy, claim identity), `external_callback.claim` (the original claim artifact) | `external_callback` |
| `external_column_claim_v1` | `external_column_claim.meta` (CBOR: claim identity, row count, element type, support counts) | loaded with `ExternalSpec.load` under a retained identity |

The earlier `ANTEPROV2` and `ANTCALLBACK1` frames are read, validated and
resealed into these kinds, but never written. A provider result is opened as
verified only against a registry's evidence digests (`open_verified_provider_envelope`).

### Section compression

Sections may use Zstandard (`compression = "zstd"`). The length prefix and
BLAKE3 checksum cover **on-wire** (possibly compressed) bytes.
`uncompressed_size` is the logical payload length. Auto compression applies when
the logical size is ≥ 4 KiB and compression improves size (≈95% ratio).

### Selective read / mmap / shared writes

| API | Behavior |
|-----|----------|
| `EncodedArtifact::read_from` | Full materialization (compat) |
| `EncodedArtifact::read_selective` | Stream-hash + discard unselected sections; no retained payload |
| `ArtifactReader::open_seek` | Index section offsets; seek-skip without allocating payloads |
| `MappedArtifactReader::open_path` | owned snapshot of the file; `load_section_mapped` for uncompressed views |
| `MappedArtifactReader::open_path_mapped` | `unsafe` file-backed `memmap2` map; caller guarantees immutability |
| `SectionBytes.data: Arc<[u8]>` | Shared logical buffers; Never-compress write avoids clone |
| `decode_posterior_meta_from_seek` / `_from_path` | Metadata without loading draws |

Compressed sections cannot be mapped as logical views (`IoError::MappedCompressed`);
use `load_section` to decompress into owned bytes.

## Supported kinds

| Kind | Sections |
|------|----------|
| `analysis_result` | `analysis_result.header`, `analysis_result.body`, `analysis_result.contract` (the sealed execution contract), `analysis_result.method_provenance` (below); optional `analysis_result.source_witness` |
| `schema_graph` | `schema` (CBOR SchemaWire v2), `dag` (CBOR `DagWire`) |
| `analysis_trace` | `analysis.trace` (CBOR) |
| `causal_posterior` | `posterior.meta` (CBOR), `posterior.draws` (f64 LE col-major) |
| `model_bundle` | required: `bundle.header`, `schema`, `dag`, `mechanisms`; optional: `contrast`, `query`, `analysis.trace`, `identification`, `estimate`, `refutations`, plans, `provenance`, posterior, discovery |
| response payload | CBOR `CausalResponseWire`, including estimand, identification payload, uncertainty, empirical support, assumptions, and provenance id |
| `other("causal_payload")` | `causal_payload.header` (payload kind plus variable names) and `causal_payload.body` (the existing canonical query/result wire type) |

### Format 0.4 static mediation and counterfactual results

The `static_result` causal payload stores the original identification query and
derivation, requested contrast, optional sampling standard error, estimation
assumptions, empirical support diagnostics and native refutation reports.
Mediation stores total/direct/indirect values; counterfactuals store unit ITEs,
their mean and both intervention levels. Counterfactual sampling errors remain
absent. Family/payload mismatches and invalid numerical values refuse on load.
`outcome.export(quantity="record")` produces this payload after estimation;
`artifacts.loads` / `artifacts.dumps` round-trip it. Derivatives and observation
curves continue using `response_result` with their existing four-axis wire.

### Format 0.4 temporal response

`ResponseQueryWire.temporal` is optional. Its absence means the existing
static response query. When present, `TemporalResponseSpecWire` contains:

- `horizons: Vec<u32>` — non-empty, strictly increasing outcome horizons;
- `policy: TemporalPolicyWire` — pulse or sustained for temporal *response*
  queries (`Dynamic` remains a `TemporalEffect` spelling and is refused on
  `TemporalResponseSpec`);
- `max_history_lag: Option<u32>` — optional finite-unfolding history cap.
- `feedback: Option<FeedbackDeclarationWire>` — optional declared delayed
  feedback, omitted when absent so queries without it keep their bytes and
  digests. Its fields are `delay_ticks` (at least 1), `level_bits`,
  `autoregression_bits`, `input_gain_bits`, `innovation_variance_bits`, and
  `initial_state_bits` (IEEE-754 bit patterns), and `initial_law` (a law name;
  omitted for the contemporaneous observed law). The dynamics are declared, not
  fitted; a declaration changes the target digest and publishes the
  finite-horizon rollout instead of the fitted surface.

A temporal `MeanCurve` result uses `ResponseValueWire::Surface` with
`dimension = 2`. Values are dose-major:
`mean[dose_index * n_horizons + horizon_index]`. The flattened `grid` stores
`[dose, horizon]` coordinate pairs in that same order. A temporal
`InterventionResponse` is a horizon path with `dimension = 1`; its grid is the
horizon vector. Uncertainty and support arrays follow the corresponding mean
layout. `SupportReportWire.point_status` is optional: present on temporal
surfaces (dose-major, same index formula as `mean`) and omitted on static
curves. `status` on a temporal surface is the three-way cell-grid summary,
not the union of per-horizon treatment ranges.
`CausalResponseWire.horizon_identification` is optional: present on temporal
surfaces as one entry per requested horizon (`I(h)`), omitted on static
curves. A union of those adjustment sets is not a valid shared `Z`.

Python exposes this container through `antecedent.artifacts.dumps` / `loads`.
The returned mapping is the canonical Rust wire representation; the Python layer
does not define or maintain a parallel JSON schema. Response, transport, and
interference query/result artifacts therefore migrate and validate through the
same format-0.5 reader as Rust artifacts.

### Format 0.5 identified-set interval

`StructuralResponseMixtureWire.identified_set_interval` is optional. It carries
the identified-set interval published on class-aware temporal Pulse / Sustained
effects (`StructuralResponseMixture::identified_set_interval`). With `method`
`imbens_manski_shared_block` (Frequentist; per-completion endpoints from shared
circular-block replicates) it covers the true effect with asymptotic
probability at least `level` whenever that effect is one retained identified
completion's effect. With `product_posterior_envelope_quantile` (Bayesian;
alias `imbens_manski_posterior_draws` on read) its endpoints are quantiles of
the per-draw min / max of completion posteriors drawn from distinct
`StreamDomain` RNG streams, and every
retained completion's posterior puts at most `1 − Φ(critical_value)` of its
mass outside each endpoint. Fields: `level`, `lower`, `upper` (the interval),
`bound_lower`, `bound_upper` (the estimated `min` / `max` over completions),
`lower_se`, `upper_se` (endpoint SDs: `lower = bound_lower − critical_value ·
lower_se`, `upper = bound_upper + critical_value · upper_se`),
`critical_value`, `width_retained`, `completions` (every fitted identified
completion, in both methods), `replicates`, `method`, and `truncated` (present
and `true` only when the completion enumeration or its equivalence audit was
capped, so the set spans retained completions only; absent reads as `false`).
Encoders refuse a construction with no wire tag rather than relabel it. The
field is omitted when no interval was computed. Readers refuse non-finite
values, a level outside `(0, 1)`, inverted endpoints or bounds, negative SDs or
critical value, zero completions, and fewer than two replicates. A format-0.4
analysis result has no such field and migrates with it absent; a 0.4 reader
refuses a 0.5 artifact rather than silently dropping the interval.

## Exporting prepared results

From Python, export the execution result directly:

```python
import antecedent as ant

result = ant.analyze(data, graph=graph, query=query)
encoded = result.export()
loaded = ant.load(encoded)
assert loaded.acceptance.recognized
assert loaded.answer == result.answer
report = loaded.inspect().to_dict()
assert bytes(loaded.export()) == bytes(encoded)

# Rebinding the study does not change this execution or its archive.
# The second click is estimate(rebind(study, data=...)); a second analyze() re-prepares.
updated = ant.estimate(ant.rebind(result.study, data=new_data))
assert result.export() == encoded
```

The export retains the compiled execution contract and applicable numerical
payloads, including posterior draws, response functions and structural mixtures.
Loading checks that the artifact is intact; it does not verify causal assumptions or
reconstruct a live study. Every reported interval states its calibration on the
claim. See the [workflow guide](start/first-analysis.md) for retention boundaries.

Loading is cheap. It checks that the artifact is an intact Antecedent result, with its
contract section present, understood, and agreeing with its digests, and keeps the recorded
answer. It recomputes nothing: a result you just produced needs no replay.

For a result that came from outside your own authoring, ask for a replay. It recomputes the
published numbers from the evidence the artifact carries and never changes the recorded
answer. Because a replay is always asked for, it runs under larger work limits than the
implicit paths:

```python
loaded = ant.load(encoded, expected=encoded.expectation, replay=True)
# or, on an existing load:
replayed = loaded.replay()
replayed.acceptance.status  # "replayed" when every published part agrees
```

`ant.load` returns an `Outcome` whose `value` is the `LoadedResult`.
`loaded.acceptance.status` is one of three outcomes, and `repr(loaded.value)` shows it:

- `loaded`: the artifact is intact and the recorded answer is available.
  Nothing was recomputed.
- `replayed`: a replay recomputed every published part from the bytes and each
  agreed. `acceptance.replayed` is `True`.
- `unavailable`: the contract is missing, unrecognized, or fails its digests
  (`acceptance.integrity` names why). `loaded.answer` is explicitly unavailable
  and the bytes can still be forwarded unchanged.

A replay that does not reach `replayed` says why, in two separate lists. `acceptance.not_replayable`
names parts whose evidence the artifact does not carry (for example
`dependencies.checked_conditional_effect_operation`); that is a limit of the artifact, not a
finding against it, and it does not restrict using the loaded answer.
`acceptance.mismatches` names parts that were recomputed and differ.

In Rust, `antecedent::io::load_analysis_result(&bytes)` returns an `AnalysisResultLoad` that
checks integrity (`is_intact()`, `integrity`) and replays nothing.
`AnalysisResultLoad::replay(&ReplayOptions::default())` returns a `ReplayReport` with the same
`integrity`, `not_replayable` and `mismatches` lists and `replayed()`;
`antecedent::io::replay_loaded_result` does the same and also replays the randomized-arm IPW
procedure. `antecedent::io::replay_analysis_result` replays straight from bytes and returns an
`AnalysisResultConsumption` with its acceptance report.

### Method provenance section

An `analysis_result` exported with its contract also carries an
`analysis_result.method_provenance` section (`MethodProvenanceWire`, format 1). For the
identifier and estimator the result executed, it lists the provenance records the producer's
table cited at export, each with its id and digest, published sources, documented deviations
from the published method and review scope limits; a route component with no record is listed
as unlinked. The section is descriptive and additive: it takes no part in acceptance or replay,
and readers that do not know it ignore it.

Rust consumers decode it with
`antecedent_io::method_provenance::decode_method_provenance_section`, which returns
an error when the section names a different identifier or estimator than the contract declares.
`replay_analysis_result` returns it as `AnalysisResultConsumption::method_provenance`, `None`
when it is absent or not bound to the contract. To cite a result, read its provenance where it
was produced (`outcome.method_provenance`, `StudyResult::method_provenance()`); see
[cite the methods behind a study](guides/citing-a-study.md).

### Posterior and query payloads for stage consumers

`Bayesian(prior_from=...)` and the prior catalog consume a posterior payload,
not a full `analysis_result` archive. Select it from the completed outcome with
`quantity="posterior"`; no refit happens, and the source outcome is unchanged:

```python
source = ant.analyze(data, graph=graph, query=query, inference=ant.Bayesian())
posterior = source.export(quantity="posterior")
next_result = ant.analyze(
    independent_data, graph=graph, query=query,
    inference=ant.Bayesian(prior_from=posterior),
)
query_artifact = source.export(quantity="query")
```

`export(quantity="posterior")` selects the fitted scalar Bayesian posterior, response
result, or static result (mediation contrast or unit ITE) without refitting.
`export(quantity="query")` selects the frozen original query, including its
query kind and original schema variable IDs rather than an identification-time
rewrite. Conditional queries and temporal pulse/sustained policies therefore
retain their original meaning across the language boundary.

A scalar posterior artifact retains draws, quantity names, identification
status, mixture mass and backend summaries. It does not retain the complete
analysis assumption or validation ledger; keep the analysis result alongside
it. Response artifacts retain response-specific assumptions, identification,
uncertainty and support. These exports use the stable format 0.5.

## Migration

`antecedent_io::migrate_artifact` / `read_and_migrate` accept
source formats `0.1` through `0.5`. Formats `0.1`–`0.4` migrate to 0.5;
format `0.1` skinny `SchemaWireV01 { variable_names }` is rewritten to full
`SchemaWire` with Continuous defaults. Format `0.2` sections pass through unchanged.
Format `0.3` response queries have no temporal field and therefore retain static
response semantics. Format `0.4` sections pass through unchanged; their
structural-mixture results carry no identified-set interval.
An analysis result whose mediation grid withholds a point is written at the scoped
format `0.6`, which no other kind uses; it reads without migration, and any other
`0.6` artifact is refused.
Unknown format
versions fail with `IoError::UnsupportedFormat`. Breaking changes require
migration from at least the previous two stable versions.

## Graph interchange (non-artifact)

DOT, JSON, GML, and NetworkX-compatible (`node_link_data` / `adjacency_data`)
codecs live alongside wire types for string-graph import/export.

Supported graph classes: **DAG**, **CPDAG**, **PAG**, and **ADMG**
(`DagWire` / `CpdagWire` / `PagWire` / `AdmgWire`, plus matching `*_from_*` /
`*_to_*` APIs). CPDAG undirected edges use DOT `--` / GML `undirected 1` /
NetworkX `undirected=true`. ADMG bidirected edges use DOT `dir=both` / GML
`bidirected 1` / NetworkX `bidirected=true`. PAG circle/conflict marks use
`mark_a` / `mark_b` attributes.

### Interactive / artifact-first (discovery off the estimate click)

Discovery is evidence and is **not** on the interactive estimate path. Preferred
spreadsheet / dashboard spine:

```text
ant.discovery.discover(data, method=..., accept_discovered=True)  once  →  AcceptedGraph
  →  ant.prepare(data, graph=accepted, query=..., inference=ant.Frequentist(profile="interactive"))
  →  outcome = ant.estimate(study)
  →  ant.estimate(ant.rebind(study, data=new_data))
  →  rediscover only by calling ant.discovery.discover again
```

Static graphs: Dag / Cpdag / Pag / Admg. Temporal completions: `TemporalDag`
(and reviewed `TemporalCpdag` / `TemporalPag`) from PCMCI / PCMCI+ / LPCMCI —
estimate-only pulse clicks never re-run discovery.

Discovery is always an explicit, executed step. `ant.prepare` and `ant.analyze` refuse an
unexecuted discovery configuration and name `ant.discovery.discover` as the remedy; with
`accept_discovered=False` pending review raises `ReviewRequired`. See
`examples/python/discover_then_estimate.py` and `parity/discovery.toml` capability
`discovery.ux.artifact_first`.

## Source-bound workflow artifacts

Decision contracts, joint laws, external claims, study rankings and composition
bundles have method-specific semantic consumers. Exported bytes retain their
scientific coordinates, provenance and limits; consumers check retained expected
identities rather than granting authority from a digest alone. See
[composition and bundles](composition.md).

The distinct `inference.MeasuredInference` envelope binds each named scalar to
its original method and current attesting coverage record. Its independent
consumer replays the original source calculation under caller-retained identity
and bounds. An original source artifact can retain not_measured diagnostics or
parameter summaries inside a measured envelope: only the named endpoints in
the [measured scope](population-time-uncertainty.md) receive that standing.
A general `antecedent.load` call is not a substitute for this method-specific
consumer.

### Summary-only period refresh

A period-refresh fit that stopped retaining its rows has its own artifact, a different kind
(`period_refresh_summary_v1`) from the full ledger (`period_refresh_ledger_v1`), so full-retention
bytes and their reader are unchanged. The sealed content is the retained least-squares summary
(Gram, cross-product, squared responses, observation count, coefficients, residual variance),
the source digest, the row counts and the stated retention (`summary-only`). It holds no row and
no block.

An independent consumer (`SummaryOnlyFit::consume` in Rust,
`PeriodRefreshSummary.consume` in Python) checks the seal and the retention, then checks that the
stored coefficients and residual variance are what the stored summary gives. It serves the
observation count, coefficients, residual variance, source digest and row counts. A refit from
rows, the eviction of a row, a mask or weight change and the replay of rows are refused with the
retention mode and the operation named (`summary-only cannot serve ...`); there are no rows to
rebuild from. The two forms are never read as each other: the full-ledger reader refuses a
summary artifact (`period_refresh.summary_only`), and the summary reader refuses a full ledger.
Export it from a fit already reduced with `into_summary_only` (`SummaryOnlyFit::from_fit`), or
from a live period refresh with `refresh.export(retention="summary-only")`, which leaves that fit's
rows in place. `refresh.export()` retains the full ledger. Both return typed artifacts;
`bytes(artifact)` exposes the unchanged encoding for file storage. Load either form with
`antecedent.load(artifact, expected=artifact.expectation)` to invoke its independent native
consumer. The full ledger is replayable; the summary checks its retained least-squares
statistics and cannot replay rows. Its `row_counts` property returns an immutable
`PeriodRowCounts(rows, live_rows, effective_n)` read model from the native summary ledger.
A singular design retains `beta=None`; a fit without residual degrees of freedom retains
`residual_variance=None`. Neither load restores a live session or dataset. The summary form
carries no posterior and no period ledger; those need the rows.
