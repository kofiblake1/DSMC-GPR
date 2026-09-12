"""One curve -> mesh -> .top -> partition -> sower, with gating and a report.

This is the single per-shape unit of work. `scripts/gen_mesh_batch.py` calls
it in a loop; `scripts/gen_mesh.py` calls it once. Everything is written under
a per-case directory so a failed case leaves its artefacts behind for
inspection.
"""

import json
import os
import time
import traceback

import numpy as np

from rarefied.geometry import curves as _curves
from rarefied.aerof import geo as _geo
from rarefied.aerof import mshio as _mshio
from rarefied.aerof import partition as _partition
from rarefied.aerof import quality as _quality
from rarefied.aerof import quicklook as _quicklook
from rarefied.aerof import topfile as _topfile


class Stage:
    CURVE = "curve"
    MESH = "mesh"
    QUALITY = "quality"
    TOP = "top"
    PARTITION = "partition"
    SOWER = "sower"
    ARTEFACTS = "artefacts"


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def run_case(name, curve, outdir, n_surf, params=None, nparts=4,
             make_png=True, make_vtu=True, metis_gtype=None, provenance=None):
    """Carry one curve through the whole mesh pipeline.

    Returns a record dict with `ok`, `failed_stage`, timings and every metric
    gathered along the way. Raises nothing: failures are recorded, because in
    a dataset sweep one bad shape must not stop the run.
    """
    os.makedirs(outdir, exist_ok=True)
    rec = {"name": name, "outdir": outdir, "ok": False, "failed_stage": None,
           "problems": [], "notes": [], "timings": {}, "files": {},
           "provenance": provenance or {}}
    stem = os.path.join(outdir, name)
    t_start = time.time()

    def _time(stage, t0):
        rec["timings"][stage] = round(time.time() - t0, 3)

    try:
        # -- curve ----------------------------------------------------------
        t0 = time.time()
        stage = Stage.CURVE
        curve, cmetrics = _curves.prepare(curve, n_surf)
        rec["curve"] = cmetrics
        # Column labels follow the configured orientation rather than being
        # hard-coded, so the artifact is self-describing.
        _lab = "".join("xyz"[i] for i in
                       ((0, 1) if (params or _geo.MeshParams()).extrude_axis == "z"
                        else (0, 2)))
        np.savetxt(stem + ".curve", curve, fmt="%.17g",
                   header="%s  (clockwise, station 0 on the +x ray; M13)" % " ".join(_lab))
        rec["files"]["curve"] = stem + ".curve"
        _time(stage, t0)

        # -- geometry + mesh ------------------------------------------------
        t0 = time.time()
        stage = Stage.MESH
        resolved = _geo.write_geo(stem + ".geo", curve, params or _geo.MeshParams())
        rec["geometry"] = {k: v for k, v in resolved.items() if k != "params"}
        # Shared surface stations (M6): indices into the wall curve, hence
        # exact mesh nodes. The on-disk manifest format is still open (see
        # Contract A/D), so only the indices are recorded for now.
        st = _geo.stations(curve, resolved.get("n_station", 0) if
                           resolved.get("n_station", 0) != len(curve) else 0)
        rec["stations"] = {"n": int(len(st)), "indices": st.tolist(),
                           "origin": "+x ray from area centroid, clockwise (M13)"}
        rec["mesh_params"] = resolved["params"]
        gm = _geo.run_gmsh(stem + ".geo", stem + ".msh", stem + ".gmsh.log")
        rec["gmsh_warnings"] = gm["warnings"]
        rec["files"]["geo"] = stem + ".geo"
        rec["files"]["msh"] = stem + ".msh"
        _time(stage, t0)

        # -- quality gate ---------------------------------------------------
        t0 = time.time()
        stage = Stage.QUALITY
        mesh = _mshio.read_msh22(stem + ".msh")
        ok, problems, notes, qrep = _quality.check(mesh, curve, resolved)
        rec["quality"] = qrep
        rec["problems"] += problems
        rec["notes"] += notes
        with open(stem + ".quality.txt", "w") as f:
            f.write(_quality.format_report(qrep, problems, notes) + "\n")
        rec["files"]["quality"] = stem + ".quality.txt"
        _time(stage, t0)
        if not ok:
            rec["failed_stage"] = Stage.QUALITY
            return _finish(rec, stem, t_start)

        # -- .top -----------------------------------------------------------
        t0 = time.time()
        stage = Stage.TOP
        summary = _topfile.run_gmsh2top(stem + ".msh", stem + ".gmsh2top.log")
        expected = {_geo.GROUP_VOLUME[0]: len(mesh.tets)}
        for nm, _tag in _geo.SURFACE_GROUPS:
            expected[nm] = int((mesh.tri_phys == mesh.phys_id(nm)).sum())
        tprobs, on_disk = _topfile.verify(mesh, summary, expected)
        rec["top"] = {"groups": on_disk["groups"], "n_nodes": on_disk["n_nodes"],
                      "n_volume_elements": summary["n_volume_elements"],
                      "n_boundary_elements": summary["n_boundary_elements"]}
        rec["files"]["top"] = summary["top_file"]
        if tprobs:
            rec["problems"] += tprobs
            rec["failed_stage"] = Stage.TOP
            return _finish(rec, stem, t_start)
        _time(stage, t0)

        # -- partition ------------------------------------------------------
        t0 = time.time()
        stage = Stage.PARTITION
        dec, parts = _partition.run_mpmetis(
            summary["top_file"], nparts, gtype=metis_gtype,
            log_path=stem + ".mpmetis.log", n_elements=len(mesh.tets))
        pq = _partition.partition_quality(parts, mesh.tets, mesh.nodes, nparts)
        rec["partition"] = pq
        rec["files"]["dec"] = dec
        # A scrambled decomposition is still a *legal* partition, so sower
        # would accept it silently. Coherence is the only signal.
        if pq["interface_node_fraction"] > 0.5:
            rec["problems"].append(
                "decomposition is not spatially coherent (interface node "
                "fraction %.3f); element ids in the .dec probably do not "
                "line up with .top element order"
                % pq["interface_node_fraction"])
            rec["failed_stage"] = Stage.PARTITION
            return _finish(rec, stem, t_start)
        _time(stage, t0)

        # -- sower ----------------------------------------------------------
        t0 = time.time()
        stage = Stage.SOWER
        sw = _partition.run_sower(summary["top_file"], dec, nparts, stem,
                                  stem + ".sower.log")
        rec["sower"] = {"bc_codes": sw["bc_codes"], "n_subdomains": sw["n_subdomains"],
                        "files": sw["files"], "warnings": sw["warnings"]}
        if sw["n_subdomains"] != nparts:
            rec["problems"].append("sower made %s subdomains, expected %d"
                                   % (sw["n_subdomains"], nparts))
            rec["failed_stage"] = Stage.SOWER
            return _finish(rec, stem, t_start)
        # Record the BC codes actually assigned, so the mapping is evidence
        # rather than assumption.
        expect_codes = {_geo.GROUP_WALL[0]: -3, _geo.GROUP_FARFIELD[0]: 4,
                        _geo.GROUP_SIDE_LO[0]: 2, _geo.GROUP_SIDE_HI[0]: 2}
        for nm, code in expect_codes.items():
            got = sw["bc_codes"].get(nm)
            if got != code:
                rec["problems"].append(
                    "sower gave %s bc code %s, expected %d" % (nm, got, code))
        if rec["problems"]:
            rec["failed_stage"] = Stage.SOWER
            return _finish(rec, stem, t_start)
        _time(stage, t0)

        # -- artefacts ------------------------------------------------------
        t0 = time.time()
        stage = Stage.ARTEFACTS
        if make_png:
            rec["files"]["png"] = _quicklook.render_png(
                stem + ".png", mesh, curve, resolved, title=name)
        if make_vtu:
            rec["files"]["volume_vtu"] = _quicklook.write_volume_vtu(
                stem + "_volume.vtu", mesh, parts=parts)
            rec["files"]["surface_vtu"] = _quicklook.write_surface_vtu(
                stem + "_surface.vtu", mesh)
            rec["vtu_group_legend"] = _quicklook.group_legend()
        _time(stage, t0)

        rec["ok"] = True
        return _finish(rec, stem, t_start)

    except Exception as exc:                      # noqa: BLE001 - recorded, not raised
        rec["failed_stage"] = stage
        rec["problems"].append("%s: %s" % (type(exc).__name__, exc))
        rec["traceback"] = traceback.format_exc()
        return _finish(rec, stem, t_start)


def _finish(rec, stem, t_start):
    rec["timings"]["total"] = round(time.time() - t_start, 3)
    with open(stem + ".record.json", "w") as f:
        json.dump(rec, f, indent=2, default=_jsonable)
    rec["files"]["record"] = stem + ".record.json"
    return rec
