# Processing workflow

`drifterlab` is organized around this scientific data flow:

```text
RAW
  ↓
QC
  ↓
RECONSTRUCTION
  ↓
TRAJECTORY ZARR
  ↓
DEPLOYMENT / CLUSTER METADATA
  ↓
GROUPED / PAIR ZARR
  ↓
SCIENCE
```

The stages are intentionally explicit. Campaign-specific schemas and policies
belong under `drifterlab.experiments`; reusable calculations and product models
belong in the generic modules.

| Stage | Purpose | Input | Output | Owner | Status |
| --- | --- | --- | --- | --- | --- |
| Raw | Read source files without silently changing their meaning. | Campaign source files. | Normalized, provenance-preserving signals or trajectory records. | Generic MATLAB handling and raw reader dispatch: `drifterlab.io`; supplied-QC ARCTERX reader: `drifterlab.experiments.arcterx.microsvp`. | Generic MicroSVP MAT drogue-signal/native-position extraction and supplied-QC trajectory ingestion are implemented. |
| QC | Calculate explicit validity masks and advisory flags; apply reviewed decisions separately from source data. | Normalized records/signals and review tables. | Transparent automatic results and separate reviewed decisions. | Generic drogue and native-position calculations/workflows/review: `drifterlab.qc`, `drifterlab.workflows`, and `drifterlab.review`; unrelated ARCTERX policy remains under `drifterlab.experiments.arcterx`. | Standalone drogue-loss and native-position QC/review plus existing supplied-QC audits are implemented. |
| Reconstruction | Produce regular 30-minute and 60-minute trajectories from reviewed native positions. | Reviewed native trajectories. | Reconstructed position series with method provenance. | Future `drifterlab.reconstruction` package. | Planned. Current ARCTERX ingestion only preserves the reconstructions supplied by the campaign. |
| Trajectory Zarr | Inventory trajectories and publish individual-trajectory arrays on explicit observation axes. | QC records and any available reconstructed series. | Inventory Parquet and trajectory Zarr. | `drifterlab.trajectories`, orchestrated for ARCTERX by `drifterlab.experiments.arcterx.pipeline`. | Implemented for supplied ARCTERX QC MicroSVP data. |
| Deployment / cluster metadata | Record reviewed deployment membership without inferring it during I/O. | Trajectory product plus campaign deployment evidence. | A frozen deployment/group metadata table. | Future `drifterlab.grouping` package, with ARCTERX conventions under `drifterlab.experiments.arcterx`. | Planned. |
| Grouped / pair Zarr | Synchronize group members and construct explicit physical pairs. | Trajectory Zarr and frozen group metadata. | Group and pair Zarr products. | Future `drifterlab.grouping` package. | Planned. |
| Science | Compute diagnostics such as relative dispersion, FSLE, and structure functions. | Versioned grouped/pair products. | Reproducible scientific results. | Downstream analysis code; not preprocessing. | Planned and outside the current package scope. |

## Implemented now

- generic MATLAB time conversion, normalized trajectory records, inventories,
  and bounded-memory Zarr writing;
- standalone raw-signal drogue-loss detection and manual review, without using
  supplied/reference drogue-loss dates;
- generic raw native-position temporal/local QC, final resolution, and review,
  without reconstructed-track evidence;
- ARCTERX QC MicroSVP schema mapping and ingestion;
- non-destructive time, position, speed, and drogue-validity calculations;
- explicit ARCTERX drogue-review table handling;
- preservation of the supplied 30-minute and 60-minute reconstructed tracks;
- legacy ARCTERX position-review utilities and investigative diagnostics,
  retained outside the normal generic QC command.

The legacy ARCTERX iteration artifacts are preserved without migration. The normal
workflow is now `drifterlab-position-qc`; it uses one `<platform_code>.parquet`
per native trajectory, `position_review.csv`, and an optional tiny resume file.

## Planned next

1. separate downstream validation of finalized drogue estimates against supplied
   ARCTERX dates;
2. 30-minute and 60-minute reconstruction from resolved native positions;
3. individual trajectory Zarr generated from those reviewed products;
4. deployment-group metadata;
5. grouped and pair Zarr products.

No reconstruction, grouping, or science algorithm is implemented yet.
