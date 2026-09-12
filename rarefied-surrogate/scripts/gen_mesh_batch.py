#!/usr/bin/env python3
"""Mesh a batch of random Bezier shapes, with a reject/retry loop.

Mirrors the structure of the upstream dataset.py (which retries until a shape
meshes) but gates on the mesh quality report instead of a triangle count, and
records why each rejection happened so the failure modes are visible rather
than merely worked around.

Every case gets its own directory under the run root, so a failed case leaves
its .geo, gmsh log and quality report behind for inspection. One manifest line
per attempt is appended to manifest.jsonl.

Usage (inside a Slurm allocation):
    source env.sh
    python3 scripts/gen_mesh_batch.py --n 24 --run-name pilot
    python3 scripts/gen_mesh_batch.py --n 200 --run-name set1 --shard 0 --nshards 8
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


def _make_curve(Shape, seed, n_control_range, n_sampling, rng):
    """Draw one random shape; returns (curve, params) or (None, params)."""
    n_control = int(rng.integers(n_control_range[0], n_control_range[1] + 1))
    radius = rng.uniform(0.0, 1.0, size=n_control)
    edgy = rng.uniform(0.0, 1.0, size=n_control)
    sh = Shape("shape_%d" % seed, None, n_control, n_sampling, radius, edgy)
    params = {"seed": int(seed), "n_control": n_control,
              "radius": radius.tolist(), "edgy": edgy.tolist()}
    try:
        sh.generate(magnify=1.0)
        return curves.from_shape(sh), params
    except Exception as exc:                       # noqa: BLE001
        params["generate_error"] = "%s: %s" % (type(exc).__name__, exc)
        return None, params


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24, help="successful meshes wanted")
    ap.add_argument("--run-name", default="batch")
    ap.add_argument("--n-surf", type=int, default=300,
                    help="wall points, fixed across the dataset (default 300)")
    ap.add_argument("--nparts", type=int, default=4)
    ap.add_argument("--n-control-min", type=int, default=3)
    ap.add_argument("--n-control-max", type=int, default=7)
    ap.add_argument("--n-sampling", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--max-attempts-factor", type=float, default=4.0,
                    help="give up after n * this many attempts")
    ap.add_argument("--farfield-factor", type=float, default=20.0)
    ap.add_argument("--thickness-factor", type=float, default=1.0)
    ap.add_argument("--size-max-factor", type=float, default=1.0)
    ap.add_argument("--growth-rate", type=float, default=0.15)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--no-vtu", action="store_true",
                    help="skip VTUs; recommended for large batches")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="assemble a grid PNG of the successful cases")
    args = ap.parse_args(argv)

    root = os.environ.get("MESH_RUN_ROOT")
    if not root:
        sys.exit("MESH_RUN_ROOT is unset -- run `source env.sh` first")
    run_root = os.path.join(root, args.run_name)
    if args.nshards > 1:
        run_root = os.path.join(run_root, "shard%02d" % args.shard)
    os.makedirs(run_root, exist_ok=True)

    Shape, provenance = shapes_bridge.load_shape_class()
    params = geo.MeshParams(
        farfield_factor=args.farfield_factor,
        thickness_factor=args.thickness_factor,
        size_max_factor=args.size_max_factor,
        growth_rate=args.growth_rate,
    )

    manifest = os.path.join(run_root, "manifest.jsonl")
    config = {"args": vars(args), "mesh_params": params.__dict__,
              "provenance": provenance, "run_root": run_root}
    with open(os.path.join(run_root, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=pipeline._jsonable)

    # Distinct seed stream per shard so shards never draw the same shape.
    base = args.seed + args.shard * 1_000_000
    rng = np.random.default_rng(base)

    n_ok = 0
    attempts = 0
    max_attempts = int(args.n * args.max_attempts_factor) + 10
    rejects = {}
    ok_records = []

    print("run root      %s" % run_root)
    print("target        %d meshes  (max %d attempts)" % (args.n, max_attempts))
    print("shapes commit %s   stubs %s"
          % (provenance.get("shapes_commit"), provenance.get("stubbed_modules")))
    print("-" * 72)

    while n_ok < args.n and attempts < max_attempts:
        seed = base + attempts
        attempts += 1
        name = "shape_%06d" % seed
        outdir = os.path.join(run_root, name)

        curve, shape_params = _make_curve(
            Shape, seed, (args.n_control_min, args.n_control_max),
            args.n_sampling, rng)

        if curve is None:
            rec = {"name": name, "ok": False, "failed_stage": "generate",
                   "problems": [shape_params.get("generate_error", "generate failed")],
                   "shape_params": shape_params}
        else:
            rec = pipeline.run_case(
                name, curve, outdir, args.n_surf, params=params,
                nparts=args.nparts, make_png=not args.no_png,
                make_vtu=not args.no_vtu,
                provenance=dict(provenance, shape="bezier", **shape_params))
            rec["shape_params"] = shape_params

        with open(manifest, "a") as f:
            f.write(json.dumps(rec, default=pipeline._jsonable) + "\n")

        if rec["ok"]:
            n_ok += 1
            ok_records.append(rec)
            print("[%4d/%4d] %s  ok   %d tets  %.1fs"
                  % (n_ok, args.n, name, rec["quality"]["counts"]["tets"],
                     rec["timings"]["total"]))
        else:
            stage = rec["failed_stage"]
            rejects[stage] = rejects.get(stage, 0) + 1
            first = rec["problems"][0] if rec["problems"] else "?"
            print("[    reject] %s  at %-8s %s" % (name, stage, first[:90]))

    print("-" * 72)
    print("meshed %d of %d in %d attempts (%.0f%% yield)"
          % (n_ok, args.n, attempts, 100.0 * n_ok / max(attempts, 1)))
    if rejects:
        print("rejections by stage: %s" % json.dumps(rejects))
    print("manifest %s" % manifest)

    if args.contact_sheet and ok_records:
        try:
            from rarefied.geometry import contact_sheet
            p = contact_sheet.build(run_root, ok_records,
                                    os.path.join(run_root, "contact_sheet.png"))
            print("contact sheet %s" % p)
        except Exception as exc:                   # noqa: BLE001
            print("contact sheet failed: %s: %s" % (type(exc).__name__, exc))

    return 0 if n_ok == args.n else 1


if __name__ == "__main__":
    raise SystemExit(main())
