# Bring your own table

Start with a small reviewed table and an explicit scientific question. The recipe below creates a temporary CSV so it can run without shipping a dataset; replace that path with your own file.

## Declare the meaning before estimating

| Input | Declare and check |
| --- | --- |
| Treatment | What contrast is being requested and how treatment is encoded. This recipe uses numeric 0/1. |
| Outcome | Units and observation mechanism; censoring and missingness require their own assumptions and supported routes. |
| Adjustment columns | Pre-treatment variables justified by the graph, not a convenient list of predictors. |
| Unit IDs | Stable observational or assignment units; a row index is not automatically a subject identity. |
| Target population | The default all-observed population or an explicit supported population declaration. |
| Design | For a randomized study, declare assignment probabilities and units through the experiment design API. |

<!-- example: examples/python/your_data.py -->

The estimate should be close to 2 in this simulated table. The recipe rejects missing required values rather than silently filling or dropping them. For your data, choose and document an observation policy before preparing the analysis; see [the observation contract](../observation-contract.md).

Use numeric columns for this recipe and inspect categorical encodings before generalizing it. See [Python workflow options](../python-options.md) for populations and data settings, and [preflight diagnostics](preflight-diagnostics.md) for fit-free checks.

## Refresh or reprepare?

`refresh` keeps the prepared scientific question and retained design. For named experimental assignments and IDs, supply outcome-only updates when the design is unchanged. Changed assignments, IDs, row order, graph or query require the corresponding explicit new preparation. [Notebook to system](notebook-to-system.md) demonstrates both operations and a refused design change.
