# Static/class execution-context boundary audit and fixes

This source/caller audit traces the actual execution context through identification work. Integration owns builds, formatting, tests and commits; [validation](validation.md) records executed outcomes. The source corrections below do not by themselves establish a strict whole-workspace pass.

## Concrete caller matrix

| Actual caller | Original lost boundary | Current owner/path |
|---|---|---|
| prepare and fresh Cpdag/Pag ATE/conditional/response graph posterior | build_class_graph_posterior_identification_cache used map_posterior_graphs(ctx) but ignored inner and called default identify_cpdag/pag; `.ok` hid errors | worker passes Some(inner) to context dispatch; scientific refusal alone becomes absent atom; Resource/Cancelled and other actual errors propagate |
| DAG ATE posterior | identify_unique_adjacency_masks(ctx) ignored inner; identify_static default exact search and blanket error→None | actual inner passed to identify_static_query_in_context; same typed refusal filter |
| ADMG posterior ATE/response | context-free complete General ID (no exact subset search), but blanket errors dropped | General ID remains intentionally context-free; actual errors now propagate except scientific refusal; outer indexed owner observes cancellation |
| supplied class path/natural/nested/distribution | execute_class_path_specific/class_distribution(ctx) called default GAI dispatch | private *_in_context dispatch variants pass actual ctx; obsolete crate-private default wrappers are removed, while public context-free APIs remain unchanged |
| class response prepare and execute | prepare_pag/cpdag_identification had Optionctx but response branch discarded it; response_path fallback called default response dispatcher | existing ctx passed through new internal response dispatch to configured GAI methods for general single-treatment, derivative pairs and joint responses |
| graph-posterior retained class path/distribution compilation | atom_answer(ctx) → class_atom_identification omitted ctx | ctx now forwarded into configured class dispatch; original typed refusal-only demotion preserved |
| mediation class refutation | refute_mediation_class(ctx) rebuilt each contrast proof with defaults | all three original contrasts use actual ctx, preserving slot/error order |
| static supplied and posterior attribution | class_attribution completion helpers used new() despite entry ctx | full private chain now carries ctx; Cpdag/Pag/TemporalCpdag/TemporalPag samplers use same original default operations/depth but real admission/token; Memory/Cancelled propagate before members/caps are published |
| temporal attribution over posterior classes | series_posterior_members rebuilt class members without ctx | ctx reaches the same original class sampler chain; source/weight/first-occurrence order unchanged |
| class counterfactual | sampler already received ctx but treated interruption as empty/bounded class; GAI MAG candidates ignored ctx | check stop before class claims; candidate identifier uses actual ctx; resource/cancel candidate errors propagate rather than being stored as candidate refusal |
| class transport supplied and posterior | default GAI and `.ok` around sID | actual ctx reaches GAI completion owner and member TransportIdentifier::identify_with_context; graph conversion nonmembership remains None, identification errors propagate |
| direct trial transport fresh/prepare and posterior | context-free live_transport_identification and transport_atom | helper now receives ctx; posterior passes actual worker inner; errors remain typed through CausalError::from |
| common-cause target validation | random_common_cause(ctx) → posterior_with_common_cause dropped ctx and enumerated default CPDAG members | actual ctx sampler plus stop check; failed(&CausalError) preserves Cancelled/Resource using the typed ValidationError::Resource; rebuilding/refitting resources cannot turn into unsupported scientific report |
| direct static execution fallbacks | static_path, bayesian_path, basis, specialist, response branches used default exact dispatch despite ctx | calls now use existing identify_static_query_in_context with Some(ctx); their prior cache/proof checks and entry guards stay in place |

Related checked compiler and batch preparation corrections are recorded by the [facade owner](facade.md). Discovered transport completion, the TransportIdentifier standardizer search and learned/smoothed preparation now have context-aware source paths recorded in [context ownership](context_ownership.md). This matrix covers the methods reviewer boundaries; verification of the integrated chain remains a separate recorded obligation.

## Shared lower ownership and interruption semantics

`generalized::check_completion_stop` is shared by static and temporal completion owners. Only Memory and Cancelled become typed IdentificationError; Operations/Depth retain the existing bounded-envelope diagnostic and partial-identification behavior. This distinction prevents physical inability to allocate or cancellation from becoming scientific unidentified mass while preserving user-declared enumeration semantics.

PAG MAG→ADMG candidate search previously ignored the configured context entirely. It now reserves a conservative shape-derived bound for both live recursion graph/pair scratch and the fixed-cap candidate masks before pool allocation. After search, it leases result graphs for the actual number of successful masks, rather than reserving every possible output graph for a small successful search. Context cancellation is observed before allocation and each recursive state. Existing 32K-state/512-member bounds, DFS branch order, graph equality, functional comparison, and scientific bounded result stay unchanged. Leases cover this search computation and drop on every return/error; this does not introduce a new claim that returned public graph values have a caller-owned retained-program lease.

The old exported context-free identify_* functions still intentionally choose their original default context/configuration. Cross-crate callers needed four additive methods on the already configurable GeneralizedAdjustmentIdentifier: identify_{cpdag,pag}_response_general and identify_{cpdag,pag}_derivative_pairs. These methods permit reuse of the same actual configured completion/adjustment owner instead of resetting it to default inside a free function. Old free function signatures/behavior remain wrappers. Facade's new dispatch helpers are crate-private. TemporalPagCompletionSampler gains with_budget because the prior default-window constructor exposed no context-aware equivalent; its original window synthesis remains identical.

Exact adjustment search may legitimately fall back to a mathematically certified ancestral set when a smaller-set search cannot be funded; that returns a still-valid identified set with the original bounded/minimality semantics. It differs from a completion enumeration that could not establish a class-wide claim at all. Deterministic complete General ID/finite closure checks without an exact search retain their intentional context-free core APIs and are cancellation-checked by actual surrounding owners.

## Regression scope and verification status

- Actual class-posterior builder warms unconstrained computation, then executes with parallel worker contexts and a starved live pool under disabled, bounded and enabled-unbounded cache policy. All must return typed Resource, preserve the held pool remainder and return all leases; a previous warm result cannot bypass the caller.
- Actual class-posterior builder deterministically cancels inside completion work with a shared check-count token; must return Cancelled rather than publish unidentified atoms and restore CPU admission.
- Shared MAG candidate search rejects tiny admission, observes deterministic cancellation, matches the original context-free graph/member order and bounded flag when funded, and returns every lease.
- Existing direct CPDAG/PAG context tests now require typed Resource on memory exhaustion instead of accepting an empty scientific envelope.

No RNG seeds, stream calls, arithmetic, query/estimand canonical bytes, declaration bounds or scientific refusal classification changed for completed work. The intentional error correction propagates physical Memory/Cancelled and malformed/internal identification errors instead of demoting them to missing atoms or unsupported checks. The tests listed above are implemented regression scope; this record claims no execution result for them without an entry in [validation](validation.md). The final strict workspace-library check and affected execution results are recorded in [validation](validation.md); release-wide Q13 remains separate.

## Executing-test follow-through: delegated response and Auto search

The broad integration run executed the new unsealed cell-AIPW proof regression
and exposed two additional actual context resets: static `ResponseBackdoor`
dispatch constructed a default `ResponseIdentifier`, and joint response delegated
to a default generalized identifier. Both now carry the actual context while
retaining the original scientific limits. The existing checked-compile Resource
and Cancelled assertions remain intact, with diagnostic error messages.

The adjacent caller audit found Auto dispatch configured only its backdoor owner,
leaving efficient adjustment outside caller admission/cancellation. Dispatch now
sets both. `AutoRun` now propagates only physical Memory and Cancelled; declared
work bounds, unsupported methods, scientific unavailability, warnings and ordered
strategy precedence retain their previous handling. The same private interruption
classification covers its explicit distribution/path dispatch.

All eight algorithmic enumeration-funding boundaries were reviewed (the two
generalized-adjustment branches now share one private funding helper). Backdoor and efficient
adjustment legitimately return independently certified ancestral/O-set fallbacks;
transport returns typed Memory. Joint response has no certified fallback and now
returns Memory on in-ceiling admission denial. Conditional and MAG adjustment
also now distinguish physical denial from declared `max_candidates` caps; the
old explicit-cap diagnostics and envelope truncation semantics remain. Completed
formulas, candidate order and examination bounds are unchanged.

New lower regressions exercise actual direct/delegated joint response, conditional
and MAG search, and first/later Auto strategies. They assert funded scientific
result/claim equivalence, independently known adjustment sets where applicable,
typed cancellation/memory, preserved explicit caps and returned leases. Root executed all of them in the final 1,194-test facade/identify library run,
followed by 380 affected integration cases; both runs pass. See
[validation](validation.md) for the exact scope and remaining measured-evidence
assertions, which are not licensed by these software checks.
