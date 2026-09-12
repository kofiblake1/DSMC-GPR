"""Characterization reference for the stage [4a] mesh pipeline.

Freezes the current output of the reference-circle mesh case -- group areas,
volume, element-quality summary, wall-node geometry, grading, and the station
set -- so later refactors of src/rarefied/{geometry/curves,aerof/*} can be
proven non-destructive by diffing against it.

Unlike kn_eq_1p0_reference.npz, this one is only *partly* a pure snapshot.
Several of the frozen quantities have exact analytic expectations, because the
reference geometry is an analytic circle:

    wall area   = polygon perimeter x thickness
    volume      = (pi R^2 - A_shape) x thickness
    wall nodes  = 2 x n_surf, all at r = D/2

Those are checked against their analytic values by aerof/quality.py on every
run, independently of this file. So this reference guards the things that have
no closed form -- element counts, aspect-ratio distribution, realised size
grading, partition balance -- rather than pretending to validate the ones that
do.

CLUSTER ONLY: needs gmsh, gmsh2top, mpmetis and sower (see PIPELINE.md's
capability-tiers note), and must run inside a Slurm allocation.

Run:
    source env.sh
    srun -p dev -c 4 --mem=16G -t 00:30:00 \
        python3 tests/characterization/build_mesh_reference.py
"""

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

CASE_PATH = REPO_ROOT / "config/cases/circle_1m.yaml"
OUTPUT_PATH = Path(__file__).resolve().parent / "mesh_reference.npz"

from rarefied.geometry import curves            # noqa: E402
from rarefied.aerof import geo, mshio, quality, topfile, partition  # noqa: E402


def build():
    import yaml
    cfg = yaml.safe_load(CASE_PATH.read_text())
    g, m, p = cfg["geometry"], cfg["mesh"], cfg["partition"]

    run_root = os.environ.get("MESH_RUN_ROOT")
    if not run_root:
        sys.exit("MESH_RUN_ROOT is unset -- run `source env.sh` first")
    outdir = Path(run_root) / "characterization_mesh"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = str(outdir / "circle_1m")

    curve, cmetrics = curves.prepare(
        curves.circle(g["diameter"], g["n_surf"]), g["n_surf"])

    params = geo.MeshParams(
        farfield_factor=m["farfield_factor"],
        thickness_factor=m["thickness_factor"],
        size_max_factor=m["size_max_factor"],
        size_min_factor=m["size_min_factor"],
        growth_rate=m["growth_rate"],
        reference_diameter=m["reference_diameter"],
        extrude_axis=m["extrude_axis"],
        n_station=g["n_station"],
        algorithm=m["algorithm"],
        optimize=m["optimize"],
    )

    resolved = geo.write_geo(stem + ".geo", curve, params)
    geo.run_gmsh(stem + ".geo", stem + ".msh", stem + ".gmsh.log")
    mesh = mshio.read_msh22(stem + ".msh")
    ok, problems, notes, rep = quality.check(mesh, curve, resolved)
    if not ok:
        sys.exit("refusing to freeze a mesh that fails its own gate:\n  "
                 + "\n  ".join(problems))

    summary = topfile.run_gmsh2top(stem + ".msh", stem + ".gmsh2top.log")
    dec, parts = partition.run_mpmetis(summary["top_file"], p["nparts"],
                                       log_path=stem + ".mpmetis.log",
                                       n_elements=len(mesh.tets))
    pq = partition.partition_quality(parts, mesh.tets, mesh.nodes, p["nparts"])
    st = geo.stations(curve, g["n_station"])

    ar = mesh.tet_aspect_ratio()
    vol = mesh.tet_volumes()
    ref = {
        # counts
        "n_nodes": len(mesh.nodes),
        "n_tets": len(mesh.tets),
        "n_tris": len(mesh.tris),
        # curve / stations
        "curve": curve,
        "station_indices": st,
        "signed_area": cmetrics["signed_area"],
        "perimeter": cmetrics["perimeter"],
        # group areas
        "group_names": np.array([nm for nm, _ in geo.SURFACE_GROUPS]),
        "group_areas": np.array([rep["group_areas"][nm]["area"]
                                 for nm, _ in geo.SURFACE_GROUPS]),
        "group_ntris": np.array([rep["group_areas"][nm]["n_tris"]
                                 for nm, _ in geo.SURFACE_GROUPS]),
        # volume + quality
        "volume_total": rep["volume"]["total"],
        "tet_volume_min": float(vol.min()),
        "n_non_positive_tets": int((vol <= 0).sum()),
        "aspect_ratio_quantiles": np.percentile(ar, [0, 50, 99, 100]),
        # grading
        "size_min": resolved["size_min"],
        "size_max": resolved["size_max"],
        "growth_measured": rep["grading"].get("growth_rate_measured", np.nan),
        # wall geometry
        "wall_n_nodes": rep["wall"]["n_nodes"],
        "wall_max_deviation": rep["wall"]["max_deviation_from_curve"],
        "extrude_planes": np.array(rep["wall"]["extrude_planes"]),
        # partition
        "partition_counts": np.array(pq["elements_per_part"]),
        "interface_node_fraction": pq["interface_node_fraction"],
    }

    np.savez(OUTPUT_PATH, **ref)
    print("wrote %s" % OUTPUT_PATH)
    for k in ("n_nodes", "n_tets", "volume_total", "n_non_positive_tets",
              "wall_n_nodes", "wall_max_deviation", "interface_node_fraction"):
        print("  %-24s %s" % (k, ref[k]))
    print("  %-24s %s" % ("aspect_ratio (min/50/99/max)",
                          np.round(ref["aspect_ratio_quantiles"], 3)))
    print("  %-24s %d stations, first: %s"
          % ("stations", len(st), st[:5]))
    if notes:
        print("notes (advisory, not gated):")
        for n in notes:
            print("  - %s" % n)


if __name__ == "__main__":
    build()
