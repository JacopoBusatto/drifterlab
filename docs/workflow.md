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
| Raw | Read source files without silently changing their meaning. | Campaign source files. | Normalized, provenance-preserving signals or trajectory records. | Generic MATLAB handling: `drifterlab.io.matlab`; ARCTERX readers: `drifterlab.experiments.arcterx.microsvp` and `raw_drogue`. | Raw ARCTERX TTFF/strain/hull-temperature extraction and supplied-QC trajectory ingestion are implemented. |
| QC | Calculate explicit validity masks and advisory flags; apply reviewed decisions separately from source data. | Normalized records/signals and review tables. | Transparent automatic results and separate reviewed decisions. | Generic calculations: `drifterlab.qc`; ARCTERX policy/wiring: `drifterlab.experiments.arcterx`. | Standalone drogue-loss detection/review and existing supplied-QC audits are implemented. Native position QC remains planned. |
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
- ARCTERX QC MicroSVP schema mapping and ingestion;
- non-destructive time, position, speed, and drogue-validity calculations;
- explicit ARCTERX drogue-review table handling;
- preservation of the supplied 30-minute and 60-minute reconstructed tracks;
- optional ARCTERX position-review utilities and investigative diagnostics,
  separate from the normal preprocessing CLI.

The position reviewer is preserved because it already contains useful work. It
does not make the planned native position-QC stage complete, and its suggestions
never change data without an explicit review decision.

## Planned next

1. separate downstream validation of finalized drogue estimates against supplied
   ARCTERX dates;
2. native position QC;
3. manual review integrated with that position-QC stage;
4. 30-minute and 60-minute reconstruction;
5. individual trajectory Zarr generated from those reviewed products;
6. deployment-group metadata;
7. grouped and pair Zarr products.

No reconstruction, grouping, or science algorithm is implemented yet.
