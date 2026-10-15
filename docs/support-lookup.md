# Find a supported analysis cell

Choose the exact query, graph class, structure source, inference and validation settings. This view filters **licensed graph-based cells** from the same generated registry used by the native support matrix; it does not decide whether your graph is true or your observations have adequate support. An interval's calibration must be checked separately at execution.

<div id="support-lookup">
<p><label>Search query or estimator <input id="support-search" type="search" placeholder="AverageEffect, aipw, temporal…"></label></p>
<div id="support-filters" class="support-filters"></div>
<p id="support-count" role="status" aria-live="polite">Loading licensed cells…</p>
<div class="support-results" tabindex="0" aria-label="Supported cells">
<table><thead><tr><th>Query</th><th>Graph</th><th>Structure</th><th>Inference</th><th>Validation</th><th>Estimator routes</th></tr></thead><tbody id="support-rows"></tbody></table>
</div>
</div>

No matching row means this view supplies no license for that configuration. Check [the full support matrix and refusal reasons](support-matrix.md), or prepare the actual query and inspect its structured outcome. The [graphless design-family matrix](graphless-support-matrix.md) covers randomized experiments, policy, quasi-experimental, survival and longitudinal designs on their separate axes.

If JavaScript is unavailable, use the complete tables linked above. See [the glossary](glossary.md) for identified, licensed, calibrated and identity-bound, and [supported analyses](supported-analyses.md) to choose a workflow before configuring a cell.
