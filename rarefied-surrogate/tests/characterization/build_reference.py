"""Characterization reference for the moments/reconstruction notebook.

Freezes the CURRENT output of src/rarefied/moments/proess_3D_distributions.ipynb's
moment-fitting and reconstruction code -- unmodified, executed in place, exactly as
it exists in the notebook today -- on the kn_eq_1p0_data pickle, resolved through
data/registry.yaml (never a hard-coded path).

This is a snapshot of present behavior, NOT an independently-derived or
physically-validated ground truth. Its only job is to give Phase 3 (relocating the
same functions into src/rarefied/moments/ and src/rarefied/reconstruct/) something
to diff against, so each relocation can be proven non-destructive before the next
one starts. See docs/CLEANUP_TODO.md, Phase 2/3.

Run: python3.10 tests/characterization/build_reference.py
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "src/rarefied/moments/proess_3D_distributions.ipynb"
REGISTRY_PATH = REPO_ROOT / "data/registry.yaml"
OUTPUT_PATH = Path(__file__).resolve().parent / "kn_eq_1p0_reference.npz"

CASE_NAME = "kn_eq_1p0_data"
N_TERMS_RANGE = range(1, 7)  # matches the notebook's own live driver cell (cell 6)

# Cells 0-3 define the IO/histogram/region-parsing helpers that
# calculate_and_compare_moments (cell 4) depends on. Executed verbatim, in
# notebook order -- nothing here edits their source. Cells 5 (plotting), 6 (the
# driver script itself), and 7 (dead commented-out duplicate code) are not needed.
NOTEBOOK_CELLS_TO_RUN = [0, 1, 2, 3, 4]


def load_notebook_namespace():
    nb = json.loads(NOTEBOOK_PATH.read_text())
    namespace = {}
    for i in NOTEBOOK_CELLS_TO_RUN:
        source = "".join(nb["cells"][i]["source"])
        exec(compile(source, f"{NOTEBOOK_PATH.name}#cell{i}", "exec"), namespace)
    return namespace


def resolve_local_path(name):
    registry = yaml.safe_load(REGISTRY_PATH.read_text())
    entry = registry[name]
    if entry["status"] != "verified":
        raise ValueError(
            f"registry entry {name!r} has status={entry['status']!r}, expected 'verified' "
            "-- see data/registry.yaml before trusting this pickle's structure"
        )
    return os.path.expanduser(entry["local_path"])


def capture_coefficients(namespace, region_cache, n_terms):
    """Replays the exact fit call calculate_and_compare_moments makes internally
    (cell 4: get_mean_var_3D -> *1.25 -> fit_hermite_distribution_3d with
    use_quadrature=False, PRECISION=5, type=1), so the raw Hermite coefficient
    vectors -- discarded by calculate_and_compare_moments after it derives the
    flux values -- are captured too. Calls only existing, unmodified functions.
    """
    get_mean_var_3D = namespace["get_mean_var_3D"]
    fit_hermite_distribution_3d = namespace["fit_hermite_distribution_3d"]

    terms_list, error_l1, error_l2, error_linf = [], [], [], []
    mean_T_list, mean_N_list, mean_Z_list = [], [], []
    var_T_list, var_N_list, var_Z_list = [], [], []

    for region in region_cache.values():
        data = region["data"]
        mean_T1, mean_N1, mean_Z1, var_T1, var_N1, var_Z1 = get_mean_var_3D(data)
        mean1 = mean_T1 * 1.25, mean_N1 * 1.25, mean_Z1 * 1.25
        var1 = var_T1 * 1.25, var_N1 * 1.25, var_Z1 * 1.25
        fit, error = fit_hermite_distribution_3d(
            data, use_quadrature=False, mean=mean1, var=var1, PRECISION=5, type=1, n_terms=n_terms
        )
        terms_list.append(fit["terms"])
        error_l1.append(error["mean_L1"])
        error_l2.append(error["mean_L2"])
        error_linf.append(error["L_inf"])
        mean_T_list.append(fit["mean_T"])
        mean_N_list.append(fit["mean_N"])
        mean_Z_list.append(fit["mean_Z"])
        var_T_list.append(fit["var_T"])
        var_N_list.append(fit["var_N"])
        var_Z_list.append(fit["var_Z"])

    return {
        "terms": np.stack(terms_list),
        "error_mean_L1": np.array(error_l1),
        "error_mean_L2": np.array(error_l2),
        "error_L_inf": np.array(error_linf),
        "mean_T": np.array(mean_T_list),
        "mean_N": np.array(mean_N_list),
        "mean_Z": np.array(mean_Z_list),
        "var_T": np.array(var_T_list),
        "var_N": np.array(var_N_list),
        "var_Z": np.array(var_Z_list),
    }


def main():
    namespace = load_notebook_namespace()
    calculate_and_compare_moments = namespace["calculate_and_compare_moments"]

    pkl_path = resolve_local_path(CASE_NAME)
    print(f"Loading {CASE_NAME} from registry -> {pkl_path}")
    with open(pkl_path, "rb") as f:
        region_cache, region_df = pickle.load(f)
    print(f"Loaded: {len(region_cache)} regions, region_df shape {region_df.shape}")

    reference = {"case": CASE_NAME, "n_terms_range": np.array(list(N_TERMS_RANGE))}
    for n in N_TERMS_RANGE:
        moments = calculate_and_compare_moments(region_cache, region_df, n_terms=n, verbose=False)
        for key, arr in moments.items():
            reference[f"n{n}_{key}"] = arr

        coeffs = capture_coefficients(namespace, region_cache, n)
        for key, arr in coeffs.items():
            reference[f"n{n}_{key}"] = arr

        print(f"n_terms={n}: captured {len(moments)} moment arrays + {len(coeffs)} coefficient arrays")

    np.savez(OUTPUT_PATH, **reference)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
