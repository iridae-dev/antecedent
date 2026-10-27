# Example datasets

Examples that use real data describe it here. Each dataset has a manifest, `<dataset>.toml`, recording where the data come from, how to cite them, the licence, the version, the SHA-256 of every file, the retrieval date, the preprocessing and the columns. `fetch.py` downloads and verifies the files.

Real observational data are never presented as known causal truth. An example built on a dataset also carries a synthetic stand-in with the same columns and a stated truth, so it runs offline and its assertions can check something known.

## Datasets

| Dataset | Used by | Licence |
| --- | --- | --- |
| [`nsw_psid`](nsw_psid.toml): NSW job-training experiment with a PSID comparison group | `examples/python/observational_study.py` | not stated by the publisher; downloaded on request, not redistributed |

## Fetch, verify, use offline

```bash
python examples/data/fetch.py nsw_psid             # download, then verify every checksum
python examples/data/fetch.py nsw_psid --offline   # verify the cache without the network
```

Files are cached in `$ANTECEDENT_DATA_DIR`, or `~/.cache/antecedent/data` when that is unset. A file is used only if its SHA-256 matches the manifest; a manifest whose checksum is empty is never treated as verified. Nothing is fetched when a module is imported.

An example reads the cache by default and falls back to its synthetic stand-in, saying so in its output. Set `ANTECEDENT_DATA=download` to let it download and verify the real files first.

## Adding a dataset

Write the manifest first. If the publisher states no redistribution licence, ship no data: record a stable URL and checksum, and give the example a labelled stand-in. A maintainer fills each `sha256` from the exact bytes with `python examples/data/fetch.py <dataset> --print-sha256` after checking the file against the publisher's description.
