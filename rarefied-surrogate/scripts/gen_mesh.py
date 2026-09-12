#!/usr/bin/env python3
"""Stage [4a]: mesh a single shape end to end (curve -> gmsh -> .top -> sower).

Default case is the 1 m diameter circle, which is exact by construction and
therefore the validation target: its wall nodes must sit at r = 0.5 to
round-off, its wall area must equal the polygon perimeter times the extrusion
thickness, and its volume must equal the plane area times the thickness.

Cluster-only (needs gmsh, gmsh2top, mpmetis, sower) -- see PIPELINE.md's
capability-tiers note.

Usage (inside a Slurm allocation -- do not run on the login node):
    source env.sh
    python3 scripts/gen_mesh.py --case config/cases/circle_1m.yaml
    python3 scripts/gen_mesh.py --case config/cases/circle_1m.yaml --nparts 8
    python3 scripts/gen_mesh.py --shape bezier --seed 3     # no config needed

Explicit command-line flags always override the config file.
"""

import argparse
import json
import os
import sys

import numpy as np

from rarefied.geometry import curves
from rarefied.aerof import geo
from rarefied.aerof import pipeline
from rarefied.aerof import shapes_bridge


def load_case(path):
    """Flatten a config/cases/*.yaml into argparse-style defaults.

    Only keys present in the file are returned, so anything omitted falls back
    to the parser's own default rather than to a silent zero.
    """
    import yaml                                    # noqa: PLC0415
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    flat = {}
    if "name" in cfg:
        flat["name"] = cfg["name"]
    g = cfg.get("geometry", {}) or {}
    m = cfg.get("mesh", {}) or {}
    p = cfg.get("partition", {}) or {}
    for src, key, dest in (
        (g, "source", "shape"), (g, "diameter", "diameter"),
        (g, "n_surf", "n_surf"), (g, "n_station", "n_station"),
        (m, "reference_diameter", "reference_diameter"),
        (m, "size_min_factor", "size_min_factor"),
        (m, "size_max_factor", "size_max_factor"),
        (m, "growth_rate", "growth_rate"),
        (m, "farfield_factor", "farfield_factor"),
        (m, "thickness_factor", "thickness_factor"),
        (m, "extrude_axis", "extrude_axis"),
        (m, "algorithm", "algorithm"), (m, "optimize", "optimize"),
        (p, "nparts", "nparts"),
    ):
        if key in src:
            flat[dest] = src[key]
    return flat, cfg


def _bezier_curve(seed, n_control, n_sampling):
    """A random Bezier shape from the upstream `shapes` package.

    Loaded by path via shapes_bridge so $SHAPES_DIR stays a pristine clone and
    its commit is recorded for provenance.
    """
    Shape, info = shapes_bridge.load_shape_class()
    rng = np.random.default_rng(seed)
    radius = rng.uniform(0.0, 1.0, size=n_control)
    edgy = rng.uniform(0.0, 1.0, size=n_control)
    sh = Shape("shape_%d" % seed, None, n_control, n_sampling, radius, edgy)
    sh.generate(magnify=1.0)
    return curves.from_shape(sh), info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default=None,
                    help="config/cases/*.yaml; explicit flags override it")
    ap.add_argument("--shape", choices=("circle", "bezier"), default="circle")
    ap.add_argument("--name", default=None, help="case name (default: shape type)")
    ap.add_argument("--diameter", type=float, default=1.0,
                    help="circle diameter in metres (default 1.0)")
    ap.add_argument("--n-surf", type=int, default=300,
                    help="wall points = mesh wall nodes; fixed across a dataset")
    ap.add_argument("--n-station", type=int, default=0,
                    help="shared surface stations (M6); 0 => every wall node")
    ap.add_argument("--size-min-factor", type=float, default=0.0,
                    help="near-wall cell size as a fraction of D. Should come "
                         "from M4's wall-gradient convergence study; 0 falls "
                         "back to perimeter/n_surf")
    ap.add_argument("--reference-diameter", type=float, default=None,
                    help="pin D instead of deriving it from enclosed area")
    ap.add_argument("--algorithm", type=int, default=6)
    ap.add_argument("--optimize", action="store_true", default=True)
    ap.add_argument("--nparts", type=int, default=4,
                    help="subdomains for mpmetis/sower (default 4)")
    ap.add_argument("--seed", type=int, default=0, help="bezier shape seed")
    ap.add_argument("--n-control", type=int, default=4, help="bezier control points")
    ap.add_argument("--outdir", default=None,
                    help="default $MESH_RUN_ROOT/<name>")
    ap.add_argument("--farfield-factor", type=float, default=20.0)
    ap.add_argument("--thickness-factor", type=float, default=1.0)
    ap.add_argument("--size-max-factor", type=float, default=1.0)
    ap.add_argument("--growth-rate", type=float, default=0.15)
    ap.add_argument("--extrude-axis", choices=("z", "y"), default="z",
                    help="z: shape in x-y (AERO-F freestream convention, "
                         "matches the FRG short-course mesh); y: shape in x-z "
                         "(matches pavery's NACA0012.geo)")
    ap.add_argument("--metis-gtype", default=None, choices=(None, "dual", "nodal"))
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--no-vtu", action="store_true")

    # Two-pass parse so an explicit flag beats the config file: parse once to
    # find --case, fold its values in as defaults, then re-parse.
    pre, _ = ap.parse_known_args(argv)
    case_cfg = {}
    if pre.case:
        flat, case_cfg = load_case(pre.case)
        ap.set_defaults(**flat)
    args = ap.parse_args(argv)

    name = args.name or args.shape
    root = os.environ.get("MESH_RUN_ROOT")
    if not root:
        sys.exit("MESH_RUN_ROOT is unset -- run `source env.sh` first")
    outdir = args.outdir or os.path.join(root, name)

    provenance = {}
    if args.shape == "circle":
        curve = curves.circle(args.diameter, args.n_surf)
        # The polygon under-measures the circle it approximates, so pin D to
        # the exact requested diameter and keep the derived lengths clean.
        ref_d = args.diameter
    else:
        curve, provenance = _bezier_curve(args.seed, args.n_control, 50)
        ref_d = 0.0                                # derive from enclosed area
    if args.reference_diameter is not None:
        ref_d = args.reference_diameter

    params = geo.MeshParams(
        farfield_factor=args.farfield_factor,
        thickness_factor=args.thickness_factor,
        size_max_factor=args.size_max_factor,
        size_min_factor=args.size_min_factor,
        growth_rate=args.growth_rate,
        reference_diameter=ref_d,
        extrude_axis=args.extrude_axis,
        n_station=args.n_station,
        algorithm=args.algorithm,
        optimize=args.optimize,
    )
    if pre.case:
        provenance = dict(provenance, case_file=os.path.abspath(pre.case),
                          case_config=case_cfg)

    rec = pipeline.run_case(name, curve, outdir, args.n_surf, params=params,
                            nparts=args.nparts, make_png=not args.no_png,
                            make_vtu=not args.no_vtu, metis_gtype=args.metis_gtype,
                            provenance=dict(provenance, shape=args.shape,
                                            seed=args.seed))

    print("=" * 72)
    print("case %s  ->  %s" % (rec["name"], rec["outdir"]))
    print("=" * 72)
    if os.path.exists(os.path.join(outdir, name + ".quality.txt")):
        print(open(os.path.join(outdir, name + ".quality.txt")).read())
    if rec.get("partition"):
        p = rec["partition"]
        print("partition        %s  balance %.4f  interface nodes %d (%.2f%%)"
              % (p["elements_per_part"], p["balance_max_over_mean"],
                 p["interface_nodes"], 100 * p["interface_node_fraction"]))
    if rec.get("sower"):
        print("sower bc codes   %s" % json.dumps(rec["sower"]["bc_codes"]))
        print("sower files      %s"
              % [os.path.basename(f) for f in rec["sower"]["files"]])
    print("timings          %s" % json.dumps(rec["timings"]))
    print()
    if rec["ok"]:
        print("RESULT: ok")
    else:
        print("RESULT: FAILED at stage %r" % rec["failed_stage"])
        for s in rec["problems"]:
            print("  - %s" % s)
        if "traceback" in rec:
            print(rec["traceback"])
    print("record           %s" % rec["files"].get("record"))
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
