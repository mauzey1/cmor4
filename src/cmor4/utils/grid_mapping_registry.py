"""CF-1.12 grid mapping names and allowed attributes."""

from __future__ import annotations

from typing import FrozenSet


COMMON_GRID_MAPPING_ATTRIBUTES: FrozenSet[str] = frozenset({
    "crs_wkt",
    "earth_radius",
    "GeoTransform",
    "geographic_coordinate_system_name",
    "geographic_crs_name",
    "geoid_name",
    "geopotential_datum_name",
    "grid_mapping_name",
    "horizontal_datum_name",
    "inverse_flattening",
    "long_name",
    "longitude_of_prime_meridian",
    "prime_meridian_name",
    "projected_coordinate_system_name",
    "projected_crs_name",
    "reference_ellipsoid_name",
    "semi_major_axis",
    "semi_minor_axis",
    "spatial_ref",
    "towgs84",
})


TEXT_GRID_MAPPING_ATTRIBUTES: FrozenSet[str] = frozenset({
    "crs_wkt",
    "fixed_angle_axis",
    "GeoTransform",
    "geographic_coordinate_system_name",
    "geographic_crs_name",
    "geoid_name",
    "geopotential_datum_name",
    "grid_mapping_name",
    "horizontal_datum_name",
    "long_name",
    "prime_meridian_name",
    "projected_coordinate_system_name",
    "projected_crs_name",
    "reference_ellipsoid_name",
    "spatial_ref",
    "sweep_angle_axis",
})


GRID_MAPPING_ATTRIBUTES: dict[str, FrozenSet[str]] = {
    "albers_conical_equal_area": frozenset({
        "standard_parallel",
        "longitude_of_central_meridian",
        "latitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "azimuthal_equidistant": frozenset({
        "longitude_of_projection_origin",
        "latitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "geostationary": frozenset({
        "latitude_of_projection_origin",
        "longitude_of_projection_origin",
        "perspective_point_height",
        "false_easting",
        "false_northing",
        "sweep_angle_axis",
        "fixed_angle_axis",
    }),
    "lambert_azimuthal_equal_area": frozenset({
        "longitude_of_projection_origin",
        "latitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "lambert_conformal_conic": frozenset({
        "standard_parallel",
        "longitude_of_central_meridian",
        "latitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "lambert_cylindrical_equal_area": frozenset({
        "longitude_of_central_meridian",
        "standard_parallel",
        "scale_factor_at_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "latitude_longitude": frozenset(),
    "mercator": frozenset({
        "longitude_of_projection_origin",
        "standard_parallel",
        "scale_factor_at_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "oblique_mercator": frozenset({
        "azimuth_of_central_line",
        "latitude_of_projection_origin",
        "longitude_of_projection_origin",
        "scale_factor_at_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "orthographic": frozenset({
        "longitude_of_projection_origin",
        "latitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "polar_stereographic": frozenset({
        "longitude_of_projection_origin",
        "straight_vertical_longitude_from_pole",
        "latitude_of_projection_origin",
        "standard_parallel",
        "scale_factor_at_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "rotated_latitude_longitude": frozenset({
        "grid_north_pole_latitude",
        "grid_north_pole_longitude",
        "north_pole_grid_longitude",
    }),
    "sinusoidal": frozenset({
        "longitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "stereographic": frozenset({
        "longitude_of_projection_origin",
        "latitude_of_projection_origin",
        "scale_factor_at_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "transverse_mercator": frozenset({
        "scale_factor_at_central_meridian",
        "longitude_of_central_meridian",
        "latitude_of_projection_origin",
        "false_easting",
        "false_northing",
    }),
    "vertical_perspective": frozenset({
        "latitude_of_projection_origin",
        "longitude_of_projection_origin",
        "perspective_point_height",
        "false_easting",
        "false_northing",
    }),
}


def allowed_grid_mapping_attributes(mapping_name: str) -> frozenset[str] | None:
    """Return CF-1.12 attributes allowed for *mapping_name*, or ``None``."""
    specific = GRID_MAPPING_ATTRIBUTES.get(mapping_name)
    if specific is None:
        return None
    allowed = set(COMMON_GRID_MAPPING_ATTRIBUTES)
    allowed.update(specific)
    if "standard_parallel" in allowed:
        allowed.update({"standard_parallel1", "standard_parallel2"})
    return frozenset(allowed)
