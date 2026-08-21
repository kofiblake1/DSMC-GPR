"""Circle geometry builder — a simple gmsh domain for template_circle.py.

Builds the circle at radius * internal_scale_factor so gmsh stays numerically
stable for very small physical radii, then reports internal_scale_factor back
so the caller can pass it to export_coupled_mesh() to recover physical units.
"""

import gmsh


def build_circle(title, radius, center=(0.0, 0.0), lc_fraction=1e-2, internal_scale_factor=None, element_type="tri"):
    """Build and mesh a circular plane surface. Returns (surface, internal_scale_factor).

    element_type: "tri" for 3-node triangles (default) or "quad" for 4-node
        quadrilaterals (gmsh recombines the triangulation into quads).
    """
    if element_type not in ("tri", "quad"):
        raise ValueError(f"unknown element_type: {element_type!r}")

    gmsh.initialize()
    gmsh.model.add(title)

    if internal_scale_factor is None:
        internal_scale_factor = 1.0 / radius

    lc = radius * lc_fraction
    scaled_radius = radius * internal_scale_factor
    scaled_lc = lc * internal_scale_factor
    scaled_center = (center[0] * internal_scale_factor, center[1] * internal_scale_factor)

    center_point = gmsh.model.geo.addPoint(scaled_center[0], scaled_center[1], 0, scaled_lc)
    point_east = gmsh.model.geo.addPoint(scaled_center[0] + scaled_radius, scaled_center[1], 0, scaled_lc)
    point_north = gmsh.model.geo.addPoint(scaled_center[0], scaled_center[1] + scaled_radius, 0, scaled_lc)
    point_west = gmsh.model.geo.addPoint(scaled_center[0] - scaled_radius, scaled_center[1], 0, scaled_lc)
    point_south = gmsh.model.geo.addPoint(scaled_center[0], scaled_center[1] - scaled_radius, 0, scaled_lc)

    circle_arcs = [
        gmsh.model.geo.addCircleArc(point_east, center_point, point_north),
        gmsh.model.geo.addCircleArc(point_north, center_point, point_west),
        gmsh.model.geo.addCircleArc(point_west, center_point, point_south),
        gmsh.model.geo.addCircleArc(point_south, center_point, point_east),
    ]

    curve_loop = gmsh.model.geo.addCurveLoop(circle_arcs)
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    gmsh.model.geo.synchronize()
    if element_type == "quad":
        gmsh.model.mesh.setRecombine(2, surface)
    gmsh.model.mesh.generate(2)

    return surface, internal_scale_factor
