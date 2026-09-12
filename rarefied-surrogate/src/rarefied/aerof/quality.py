"""Mesh gate: hard checks that must pass, plus reported diagnostics.

Two separate ideas here, deliberately not mixed:

  * `problems` are disqualifying. A mesh with any of these is unusable and, in
    the dataset loop, the shape is rejected and logged.
  * `notes` are advisory. High aspect ratio near the wall is expected from a
    one-cell extrusion and is reported rather than gated on.

The group-area checks do double duty: they are the only thing that verifies
geo.py's assumption about the order in which gmsh's Extrude returns lateral
surfaces. If that ordering ever changes, the wall and farfield areas swap and
these checks fail loudly instead of silently producing a mesh with the
boundary conditions on the wrong surfaces.
"""

import numpy as np

from rarefied.geometry import curves as _curves
from rarefied.aerof import geo as _geo


def _group_area(mesh, name):
    tris = mesh.tris_of(name)
    if len(tris) == 0:
        return 0.0
    p = mesh.nodes[tris]
    return float(0.5 * np.linalg.norm(
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum())


def check(mesh, curve, resolved, area_tol=0.02, wall_dev_tol=1e-9):
    """Validate a meshed extruded shape. Returns (ok, problems, notes, report).

    `resolved` is the dict returned by geo.MeshParams.resolve / write_geo.
    Areas are compared against the *polygon* the mesh actually represents, not
    the smooth shape it approximates, except for the farfield where gmsh
    meshes a true arc and the discretisation deficit is absorbed by area_tol.
    """
    problems, notes, rep = [], [], {}

    t = resolved["thickness"]
    R = resolved["farfield_radius"]
    n_surf = resolved["n_surf"]
    size_min = resolved["size_min"]
    size_max = resolved["size_max"]

    per = _curves.perimeter(curve)
    a_shape = abs(_curves.signed_area(curve))
    a_plane = np.pi * R * R - a_shape

    rep["counts"] = {
        "nodes": int(len(mesh.nodes)),
        "tets": int(len(mesh.tets)),
        "boundary_tris": int(len(mesh.tris)),
    }

    # -- groups exist -------------------------------------------------------
    have = {nm for _d, nm in mesh.phys_names.values()}
    want = {nm for nm, _tag in _geo.SURFACE_GROUPS} | {_geo.GROUP_VOLUME[0]}
    missing = want - have
    if missing:
        problems.append("mesh is missing physical groups: %s" % sorted(missing))
        return False, problems, notes, rep
    if have - want:
        notes.append("mesh has extra physical groups: %s" % sorted(have - want))

    # -- group areas, which also verify the Extrude ordering ----------------
    expect = {
        _geo.GROUP_WALL[0]: per * t,
        _geo.GROUP_FARFIELD[0]: 2.0 * np.pi * R * t,
        _geo.GROUP_SIDE_LO[0]: a_plane,
        _geo.GROUP_SIDE_HI[0]: a_plane,
    }
    rep["group_areas"] = {}
    for name, want_a in expect.items():
        got = _group_area(mesh, name)
        rel = (got - want_a) / want_a if want_a else np.inf
        rep["group_areas"][name] = {
            "n_tris": int(len(mesh.tris_of(name))),
            "area": got, "expected": want_a, "rel_err": float(rel),
        }
        if abs(rel) > area_tol:
            problems.append(
                "group %s has area %.6g, expected %.6g (%.1f%% off). This "
                "usually means gmsh's Extrude returned lateral surfaces in a "
                "different order than geo.py assumes, so the boundary "
                "conditions are on the wrong surfaces."
                % (name, got, want_a, 100.0 * rel))

    # -- volume -------------------------------------------------------------
    vol = mesh.tet_volumes()
    total, want_v = float(vol.sum()), a_plane * t
    rep["volume"] = {"total": total, "expected": want_v,
                     "rel_err": float((total - want_v) / want_v)}
    if abs(total - want_v) / want_v > area_tol:
        problems.append("mesh volume %.6g, expected %.6g" % (total, want_v))

    n_neg = int((vol <= 0).sum())
    rep["tet_volume"] = {"min": float(vol.min()), "max": float(vol.max()),
                         "n_non_positive": n_neg}
    if n_neg:
        problems.append("%d tets have non-positive volume" % n_neg)

    # -- element shape ------------------------------------------------------
    ar = mesh.tet_aspect_ratio()
    rep["aspect_ratio"] = {
        "min": float(ar.min()), "median": float(np.median(ar)),
        "p99": float(np.percentile(ar, 99)), "max": float(ar.max()),
    }
    if ar.max() > 1e4:
        notes.append("worst tet aspect ratio is %.3g; the one-cell extrusion "
                     "makes some anisotropy unavoidable, but this is extreme "
                     "and may make wall-gradient features noisy" % ar.max())

    # -- the wall is what we asked for -------------------------------------
    wall_nodes = mesh.nodes_of_group(_geo.GROUP_WALL[0])
    rep["wall"] = {"n_nodes": int(len(wall_nodes)), "expected_nodes": 2 * n_surf}
    if len(wall_nodes) != 2 * n_surf:
        problems.append(
            "wall has %d nodes, expected 2 x n_surf = %d. Transfinite Curve "
            "should pin exactly the input curve points on each of the two "
            "extruded planes." % (len(wall_nodes), 2 * n_surf))

    # Wall nodes must lie on the input curve, in whichever two axes the shape
    # occupies (see MeshParams.extrude_axis).
    ax = list(resolved.get("plane_axes", (0, 2)))
    ext_ax = resolved.get("extrude_axis_index", 1)
    wp = mesh.nodes[wall_nodes][:, ax]
    dev = _curves.distance_to_curve(wp, curve)
    rep["wall"]["max_deviation_from_curve"] = float(dev.max())
    if dev.max() > wall_dev_tol * max(1.0, resolved["characteristic_diameter"]):
        problems.append("wall nodes deviate up to %.3g from the input curve"
                        % dev.max())

    span = np.unique(np.round(mesh.nodes[wall_nodes][:, ext_ax], 12))
    rep["wall"]["extrude_planes"] = [float(v) for v in span]
    rep["wall"]["extrude_axis"] = resolved.get("extrude_axis", "y")
    if len(span) != 2:
        problems.append("wall nodes sit on %d %s-planes, expected exactly 2 "
                        "(one extrusion layer)"
                        % (len(span), resolved.get("extrude_axis", "y")))

    # -- realised size grading ---------------------------------------------
    # Measured on the lower side plane, which is the 2D mesh itself.
    tris = mesh.tris_of(_geo.GROUP_SIDE_LO[0])
    p = mesh.nodes[tris][:, :, ax]
    edge = np.stack([np.linalg.norm(p[:, (i + 1) % 3] - p[:, i], axis=1)
                     for i in range(3)], axis=1)
    size = edge.mean(axis=1)
    cent = p.mean(axis=1)
    dist = _curves.distance_to_curve(cent, curve)

    near = dist < 2.0 * size_min
    rep["grading"] = {
        "size_min_requested": size_min,
        "size_max_requested": size_max,
        "size_at_wall_measured": float(np.median(size[near])) if near.any() else None,
        "size_overall_min": float(size.min()),
        "size_overall_max": float(size.max()),
        "growth_rate_requested": float(resolved.get("implied_growth_rate", np.nan)),
    }
    # Fit the realised slope dh/dd over the graded band; for a linear size law
    # this slope is the per-cell growth ratio.
    band = dist < resolved["dist_max"]
    if band.sum() > 50:
        slope = float(np.polyfit(dist[band], size[band], 1)[0])
        rep["grading"]["growth_rate_measured"] = slope
        if slope > 0.35:
            notes.append("measured size growth is %.2f per cell; above ~0.2 "
                         "the near-wall transition is coarse" % slope)

    return (len(problems) == 0), problems, notes, rep


def format_report(rep, problems, notes):
    """Human-readable summary for the run log."""
    L = []
    c = rep.get("counts", {})
    L.append("nodes %d   tets %d   boundary tris %d"
             % (c.get("nodes", 0), c.get("tets", 0), c.get("boundary_tris", 0)))
    if "group_areas" in rep:
        L.append("")
        L.append("%-18s %8s %14s %14s %9s" % ("group", "tris", "area", "expected", "rel err"))
        for name, g in rep["group_areas"].items():
            L.append("%-18s %8d %14.6g %14.6g %8.3f%%"
                     % (name, g["n_tris"], g["area"], g["expected"], 100 * g["rel_err"]))
    if "volume" in rep:
        v = rep["volume"]
        L.append("")
        L.append("volume %.6g   expected %.6g   rel err %.4f%%"
                 % (v["total"], v["expected"], 100 * v["rel_err"]))
    if "tet_volume" in rep:
        t = rep["tet_volume"]
        L.append("tet volume min %.3g  max %.3g  non-positive %d"
                 % (t["min"], t["max"], t["n_non_positive"]))
    if "aspect_ratio" in rep:
        a = rep["aspect_ratio"]
        L.append("aspect ratio  min %.3g  median %.3g  p99 %.3g  max %.3g"
                 % (a["min"], a["median"], a["p99"], a["max"]))
    if "wall" in rep:
        w = rep["wall"]
        L.append("wall nodes %d (expected %d)  max deviation from curve %.3g  extrude planes %s"
                 % (w["n_nodes"], w["expected_nodes"],
                    w.get("max_deviation_from_curve", float("nan")),
                    w.get("extrude_planes")))
    if "grading" in rep:
        g = rep["grading"]
        L.append("size at wall %.4g (requested %.4g)   growth measured %.3f (requested %.3f)"
                 % (g.get("size_at_wall_measured") or float("nan"),
                    g["size_min_requested"], g.get("growth_rate_measured", float("nan")),
                    g["growth_rate_requested"]))
    L.append("")
    L.append("PROBLEMS: %s" % ("none" if not problems else ""))
    for s in problems:
        L.append("  - %s" % s)
    if notes:
        L.append("NOTES:")
        for s in notes:
            L.append("  - %s" % s)
    return "\n".join(L)
