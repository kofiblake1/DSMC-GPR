"""Rectangle geometry builder — a simple gmsh domain for template_rectangle.py."""

import gmsh


def build_rectangle(title, width, height, origin=(0.0, 0.0), lc=1e-3):
    """Build and mesh a rectangular plane surface. Returns the surface tag."""
    gmsh.initialize()
    gmsh.model.add(title)

    x0, y0 = origin
    # Counter-clockwise starting at bottom-left.
    base_points = [
        (x0, y0),
        (x0, y0 + height),
        (x0 + width, y0 + height),
        (x0 + width, y0),
    ]
    points = [gmsh.model.geo.addPoint(x, y, 0, lc) for x, y in base_points]

    lines = [
        gmsh.model.geo.addLine(points[0], points[1]),
        gmsh.model.geo.addLine(points[1], points[2]),
        gmsh.model.geo.addLine(points[2], points[3]),
        gmsh.model.geo.addLine(points[3], points[0]),
    ]

    curve_loop = gmsh.model.geo.addCurveLoop(lines)
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    return surface
