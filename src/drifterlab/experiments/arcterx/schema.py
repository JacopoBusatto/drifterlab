"""Mappings established by inspecting all 150 supplied QC MicroSVP files."""

NATIVE_FIELDS = {
    "longitude": "lon_qc", "latitude": "lat_qc", "SST": "sst_qc",
    "SLP": "slp_qc", "battery": "battery_qc", "drogue": "drogue_strain_qc",
    "speed": "speed_qc_source",
}
INTERP_FIELDS = {
    "longitude_30min": "lon_interp_30m", "latitude_30min": "lat_interp_30m",
    "longitude_60min": "lon_interp_60m", "latitude_60min": "lat_interp_60m",
    "SST": "sst_interp", "speed_30min": "speed_interp_30m_source",
    "speed_60min": "speed_interp_60m_source",
}
NATIVE_METADATA_FIELDS = {"time", "PlatformId", "ID", "type", "drogue_off"}

REPRESENTATION_PROVENANCE = {
    "qc": "Supplied native quality-controlled fixes; ingestion applies no additional destructive QC.",
    "interp_30m": "Supplied 30-minute subsampling, spline fitting and averaging across start phases, evaluated at nominal 5-minute resolution.",
    "interp_60m": "Supplied 60-minute subsampling, spline fitting and averaging across start phases, evaluated at nominal 5-minute resolution.",
}


def field_attributes(structure: str, source: str, target: str) -> dict:
    attrs = {"source_field": f"{structure}.{source}"}
    if target.startswith("lon_"):
        attrs.update(units="degrees_east", standard_name="longitude")
    elif target.startswith("lat_"):
        attrs.update(units="degrees_north", standard_name="latitude")
    elif target.startswith("speed_"):
        attrs.update(units="m s-1", long_name="Supplied speed; source differentiation method is not established")
    elif target == "drogue_strain_qc":
        attrs.update(units="1", long_name="Non-dimensional strain; not drogue status")
    elif target == "sst_interp":
        attrs["long_name"] = "Supplied SST on the reconstructed time axis; reconstruction method unspecified"
    for representation, description in REPRESENTATION_PROVENANCE.items():
        if target in {f"lon_{representation}", f"lat_{representation}"}:
            attrs["provenance"] = description
    return attrs
