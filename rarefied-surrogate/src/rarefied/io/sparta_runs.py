"""Parsers for a SPARTA run directory: region metadata, ITEM tables, and the
per-region cache (histogram + surface/grid tables) that the moments pipeline
consumes.

Relocated verbatim (cut/paste, no edits) from
src/rarefied/moments/proess_3D_distributions.ipynb, cells 1 and 3, as part of the
docs/CLEANUP_TODO.md Phase 4 extraction.

NOTE (docs/CLEANUP_TODO.md CONFIRM #1): src/rarefied/geometry/process_shape_data.ipynb
defines a DIFFERENT function also named `read_region_metadata` (circle-union
fields, not this module's tangent-angle-oriented fields). That notebook has not
been extracted yet (Phase 5). When it is, this name will collide -- resolve the
naming there deliberately rather than as a side effect of extraction; do not
rename either implementation as part of a routine relocation.

NOTE (docs/CLEANUP_TODO.md CONFIRM #13): placed under io/ rather than sparta/
because these parse SPARTA *output* (a loader concern, matching the README's
description of io/), whereas sparta/ is for generating SPARTA *inputs*
(sampling-region + run setup).
"""

from pathlib import Path
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from rarefied.io.histograms import read_all_timesteps_histogram_2D, read_all_timesteps_histogram_3D

def read_region_metadata(filepath):
    """Parse region metadata with a text header and a CSV-like table section."""
    with open(filepath, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("Region_Index"):
            header_idx = idx
            break

    if header_idx is None:
        raise ValueError("Could not find region table header line.")

    columns = [col.strip() for col in lines[header_idx].split(",")]
    records = []
    for line in lines[header_idx + 1:]:
        if line.startswith("#"):
            continue
        parts = [val.strip() for val in line.split(",")]
        if len(parts) != len(columns):
            continue
        record = {}
        for col, val in zip(columns, parts):
            if col in {"Region_Index", "Num_Points"}:
                record[col] = int(val)
            else:
                record[col] = float(val)
        records.append(record)

    output = pd.DataFrame.from_records(records, columns=columns)
    
    # Ensure tangent columns exist (new format includes them, old format derives them)
    if "Tangent_X" not in output.columns or "Tangent_Y" not in output.columns:
        output["Tangent_X"] = output["Normal_Y"]
        output["Tangent_Y"] = -output["Normal_X"]

    # Standardize AoA radians column name if present
    if "Local_AoA_Rad" in output.columns and "AoA_Rad" not in output.columns:
        output["AoA_Rad"] = output["Local_AoA_Rad"]

    # Angle between +x and tangent in [-pi, pi]
    output["Tangent_Angle_Rad"] = np.arctan2(output["Tangent_Y"], output["Tangent_X"])

    display_cols = ["Region_Index", "Normal_X", "Normal_Y", "Tangent_X", "Tangent_Y", "Tangent_Angle_Rad"]
    for extra_col in ["AoA_Rad", "BBox_Area", "Object_Intersection_Area", "Collection_Area"]:
        if extra_col in output.columns:
            display_cols.append(extra_col)

    return output

def load_all_histograms_from_run_2D(directory, metadata_path):
    histogram_files = glob.glob(str(Path(directory) / "*_2D.histo"))
    histogram_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))

    region_df = read_region_metadata(metadata_path)
    data = []

    if len(region_df) != len(histogram_files):
        print(f"Warning: Number of regions in metadata ({len(region_df)}) does not match number of histogram files ({len(histogram_files)}).")

    for i in range(len(histogram_files)):
        data_2D = read_all_timesteps_histogram_2D(histogram_files[i])
        data.append(data_2D)

        print(f"  Loaded histogram for region {i} with {len(data_2D[max(data_2D.keys())]['bin_coords'])} bins")
        
    return (region_df, data)

def parse_sparta_item_table(filepath, item_name):
    """Parse a SPARTA text output table following an 'ITEM: <item_name>' header."""
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith(f"ITEM: {item_name}"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find ITEM: {item_name} in {filepath}")

    header_parts = lines[header_idx].split()
    col_names = header_parts[2:]  # skip 'ITEM:' and item_name

    records = []
    for line in lines[header_idx + 1:]:
        if line.startswith("ITEM:"):
            break
        parts = line.split()
        if len(parts) != len(col_names):
            continue

        rec = {}
        for c, v in zip(col_names, parts):
            if c == "id":
                rec[c] = int(float(v))
            else:
                rec[c] = float(v)
        records.append(rec)

    return pd.DataFrame(records)

def relabel_surface_columns(df):
    """Relabel surface columns by expected order after id."""
    ordered = [
        "n", "nflux", "nflux_incident", "mflux", "mflux_incident",
        "fx", "fy", "press", "px", "py", "shx", "shy", "ke", "etot"
    ]
    non_id = [c for c in df.columns if c != "id"]
    rename = {old: new for old, new in zip(non_id, ordered)}
    return df.rename(columns=rename)

def relabel_grid_columns(df):
    """Relabel grid columns by expected order after id."""
    ordered = [
        "n", "nrho", "mass", "massrho", "u", "v", "ke", "temp",
        "pxrho", "pyrho", "kerho", "temp_thermal", "pressure_thermal"
    ]
    non_id = [c for c in df.columns if c != "id"]
    rename = {old: new for old, new in zip(non_id, ordered)}
    return df.rename(columns=rename)

def load_run_data_3D(run_dir, metadata_path):
    """Load all region data for a given run directory."""
    run_dir = Path(run_dir)
    hist_files = sorted(
        run_dir.glob("*_3D.histo"),
        key=lambda p: int(p.stem.split("_")[0])
    )
    print(f"Found {len(hist_files)} histogram files in {run_dir}")
    # region_ids = [int(p.stem) for p in hist_files if p.stem.isdigit()]
    region_ids = [int(p.stem.split("_")[0]) for p in hist_files]

    data_cache = {}
    for region in tqdm(region_ids, desc="Loading regions"):
        hist_file = run_dir / f"{region}_3D.histo"
        surf_file = run_dir / f"out_region{region}surf.50000"
        grid_file = run_dir / f"out_region{region}grid.50000"

        if not (hist_file.exists() and surf_file.exists() and grid_file.exists()):
            continue

        data = read_all_timesteps_histogram_3D(hist_file)
        common_ts = sorted(set(data.keys()))
        if not common_ts:
            continue
        ts = common_ts[-1]

        grid_df = relabel_grid_columns(parse_sparta_item_table(grid_file, "CELLS"))
        surf_df = relabel_surface_columns(parse_sparta_item_table(surf_file, "SURFS"))

        data_cache[region] = {
            "ts": ts,
            "data": data[ts],
            "grid_df": grid_df,
            "surf_df": surf_df,
        }

    region_df = read_region_metadata(metadata_path)

    return data_cache, region_df
