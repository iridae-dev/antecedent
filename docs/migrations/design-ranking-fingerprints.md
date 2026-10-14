# Design ranking request and signal fingerprints

## What changed

The request fingerprint, the law digest and the signal identity stored for each candidate of a design ranking are derived differently, and the artifact digest and ranking identity follow from them. The layout and the version constant of `design_ranking_v1` are unchanged.

## Class

Inspectable only. The earlier artifact decodes (`DesignRankingArtifactWire::from_bytes`) and its candidates, utilities and ranks can be read, but a consumer recomputes every fingerprint and signal identity from the stored request and refuses the artifact.

## Who is affected

Anyone holding a design ranking exported by the previous release.

## How to regenerate

Rank the same candidates against the same decision contract and sources with the current producer and export the result. The ranking values are not changed by this; only the identities are.
