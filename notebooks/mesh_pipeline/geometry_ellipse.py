"""Ellipse geometry builder — a simple gmsh domain for template_ellipse.py.

Builds the ellipse at semi_major/semi_minor * internal_scale_factor so gmsh
stays numerically stable for very small physical sizes, then reports
internal_scale_factor back so the caller can pass it to export_coupled_mesh()
to recover physical units.

The ellipse is built with its major axis along +x, centered at `center`, then
rotated in-plane by `angle_of_attack_deg` (degrees, counter-clockwise) about
its own center. angle_of_attack_deg=0 leaves the major axis aligned with x.
"""

import math

import gmsh


def build_ellipse(
    title,
    semi_major,
    semi_minor,
    center=(0.0, 0.0),
    angle_of_attack_deg=0.0,
    lc_fraction=1e-2,
    internal_scale_factor=None,
    element_type="tri",
):
    """Build and mesh an elliptical plane surface. Returns (surface, internal_scale_factor).

    semi_major, semi_minor: semi-axis lengths. semi_major must be >= semi_minor
        (gmsh's addEllipseArc requires the major-axis point to lie on the
        longer axis).
    angle_of_attack_deg: rotation of the major axis from +x, in degrees,
        counter-clockwise, applied about `center`.
    element_type: "tri" for 3-node triangles (default) or "quad" for 4-node
        quadrilaterals (gmsh recombines the triangulation into quads).
    """
    if element_type not in ("tri", "quad"):
        raise ValueError(f"unknown element_type: {element_type!r}")
    if semi_major < semi_minor:
        raise ValueError(
            f"semi_major ({semi_major}) must be >= semi_minor ({semi_minor})"
        )

    gmsh.initialize()
    gmsh.model.add(title)

    if internal_scale_factor is None:
        internal_scale_factor = 1.0 / semi_major

    lc = semi_minor * lc_fraction
    scaled_a = semi_major * internal_scale_factor
    scaled_b = semi_minor * internal_scale_factor
    scaled_lc = lc * internal_scale_factor
    scaled_center = (center[0] * internal_scale_factor, center[1] * internal_scale_factor)

    center_point = gmsh.model.geo.addPoint(scaled_center[0], scaled_center[1], 0, scaled_lc)
    point_major_pos = gmsh.model.geo.addPoint(scaled_center[0] + scaled_a, scaled_center[1], 0, scaled_lc)
    point_minor_pos = gmsh.model.geo.addPoint(scaled_center[0], scaled_center[1] + scaled_b, 0, scaled_lc)
    point_major_neg = gmsh.model.geo.addPoint(scaled_center[0] - scaled_a, scaled_center[1], 0, scaled_lc)
    point_minor_neg = gmsh.model.geo.addPoint(scaled_center[0], scaled_center[1] - scaled_b, 0, scaled_lc)

    ellipse_arcs = [
        gmsh.model.geo.addEllipseArc(point_major_pos, center_point, point_major_pos, point_minor_pos),
        gmsh.model.geo.addEllipseArc(point_minor_pos, center_point, point_major_pos, point_major_neg),
        gmsh.model.geo.addEllipseArc(point_major_neg, center_point, point_major_pos, point_minor_neg),
        gmsh.model.geo.addEllipseArc(point_minor_neg, center_point, point_major_pos, point_major_pos),
    ]

    curve_loop = gmsh.model.geo.addCurveLoop(ellipse_arcs)
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    if angle_of_attack_deg != 0.0:
        angle_rad = math.radians(angle_of_attack_deg)
        gmsh.model.geo.rotate(
            [(2, surface)],
            scaled_center[0], scaled_center[1], 0,
            0, 0, 1,
            angle_rad,
        )

    gmsh.model.geo.synchronize()
    if element_type == "quad":
        gmsh.model.mesh.setRecombine(2, surface)
    gmsh.model.mesh.generate(2)

    return surface, internal_scale_factor
