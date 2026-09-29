"""Small Cartopy projection adapter, adapted locally from kinematicParcels."""

from __future__ import annotations

import cartopy.crs as ccrs


_SUPPORTED = {
    "PlateCarree": ccrs.PlateCarree,
    "Mercator": ccrs.Mercator,
    "SouthPolarStereo": ccrs.SouthPolarStereo,
    "NorthPolarStereo": ccrs.NorthPolarStereo,
    "Robinson": ccrs.Robinson,
}


def get_projection(name: str, *, central_longitude: float = 0) -> ccrs.CRS:
    """Return a configured Cartopy CRS without importing another project."""
    if name not in _SUPPORTED:
        raise ValueError(f"Unsupported projection {name!r}; choose from {sorted(_SUPPORTED)}")
    return _SUPPORTED[name](central_longitude=central_longitude)

