"""Per-region moment-fit vs. DSMC-ground-truth comparison glue.

Relocated verbatim (cut/paste, no edits) from
src/rarefied/moments/proess_3D_distributions.ipynb, cell 4, as part of the
docs/CLEANUP_TODO.md Phase 3 extraction.
"""

import numpy as np

from rarefied.moments.hermite import fit_hermite_distribution_3d, get_mean_var_3D
from rarefied.reconstruct.flux import (
    fast_heat_flux_from_3D_hermite_fit,
    fast_momentum_flux_from_3D_hermite_fit,
    heat_flux_from_3D_distribution,
    momentum_flux_from_3d_distribution,
)

def calculate_and_compare_moments(region_cache, region_df, n_terms, verbose=False):

    # Region-level Hermite traction comparison in global coordinates (uses preloaded region_cache + region_df)
    results = []
    t_model_x_list = []
    t_model_y_list = []
    t_data_x_list = []
    t_data_y_list = []
    energy_model_list = []
    energy_data_list = []
    energy_dist_list = []

    t_model_t_list = []
    t_model_n_list = []
    t_model_t_dist_list = []
    t_model_n_dist_list = []
    t_data_t_list = []
    t_data_n_list = []


    for ind, region in region_cache.items():
        ts = region["ts"]
        data = region["data"]
        grid_df = region["grid_df"]
        surf_df = region["surf_df"]

        surf_nz = surf_df.loc[surf_df["n"] > 0].copy()
        if surf_nz.empty:
            t_model_n_list.append(np.nan)
            t_model_t_list.append(np.nan)
            t_model_n_dist_list.append(np.nan)
            t_model_t_dist_list.append(np.nan)  
            t_data_n_list.append(np.nan)
            t_data_t_list.append(np.nan)
            energy_model_list.append(np.nan)
            energy_data_list.append(np.nan)
            energy_dist_list.append(np.nan)

            print(f"Warning: No nonzero nrho values for region {ind}, skipping.")
            continue

        # Region orientation and unit normal from metadata
        region_row = region_df.loc[region_df["Region_Index"] == ind]
        if region_row.empty:
            t_model_n_list.append(0)
            t_model_t_list.append(0)
            t_model_n_dist_list.append(0)
            t_model_t_dist_list.append(0)  
            t_data_n_list.append(0)
            t_data_t_list.append(0)
            energy_model_list.append(0)
            energy_data_list.append(0)
            energy_dist_list.append(0)
            print(f"Warning: No metadata found for region {ind}, skipping.")
            continue

        theta = float(region_row["Tangent_Angle_Rad"].iloc[0])
        nx = float(region_row["Normal_X"].iloc[0])
        ny = float(region_row["Normal_Y"].iloc[0])
        n_norm = np.hypot(nx, ny)
        if n_norm == 0.0:
            t_model_n_list.append(0)
            t_model_t_list.append(0)
            t_model_n_dist_list.append(0)
            t_model_t_dist_list.append(0)  
            t_data_n_list.append(0)
            t_data_t_list.append(0)
            energy_model_list.append(0)
            energy_data_list.append(0)
            energy_dist_list.append(0)
            print(f"Warning: Zero normal vector for region {ind}, skipping.")
            continue
        n_hat = np.array([nx, ny], dtype=float) / n_norm

        mean_T1, mean_N1, mean_Z1, var_T1, var_N1, var_Z1 = get_mean_var_3D(data)
        mean1 = mean_T1*1.25, mean_N1*1.25, mean_Z1*1.25
        var1 = var_T1*1.25, var_N1*1.25, var_Z1*1.25
        fit,error = fit_hermite_distribution_3d(data, use_quadrature=False, mean=mean1, var=var1, PRECISION=5, type=1, n_terms=n_terms)

        nrho_nonzero = grid_df.loc[grid_df["nrho"] > 0, "nrho"]
        if len(nrho_nonzero) == 0:
            t_model_n_list.append(0)
            t_model_t_list.append(0)
            t_model_n_dist_list.append(0)
            t_model_t_dist_list.append(0)  
            t_data_n_list.append(0)
            t_data_t_list.append(0)
            energy_model_list.append(0)
            energy_data_list.append(0)
            energy_dist_list.append(0)
            print(f"Warning: No nonzero nrho values for region {ind}, skipping.")
            continue
        n_avg = float(nrho_nonzero.mean())

        # Local (t,n) momentum-flux components
        # flux_tt = momentum_flux_from_3D_hermite_fit(0, 0, fit, n_avg)
        # flux_tn = momentum_flux_from_3D_hermite_fit(0, 1, fit, n_avg)
        # flux_nn = momentum_flux_from_3D_hermite_fit(1, 1, fit, n_avg)

        flux_tt = fast_momentum_flux_from_3D_hermite_fit(0, 0, fit, n_avg)
        flux_tn = fast_momentum_flux_from_3D_hermite_fit(0, 1, fit, n_avg)
        flux_nn = fast_momentum_flux_from_3D_hermite_fit(1, 1, fit, n_avg)

        # Local stress tensor
        sigma_local = -np.array([[flux_tt, flux_tn],
                                [flux_tn, flux_nn]], dtype=float)
        if verbose:
            print(f"Region {ind}, ts {ts}:")
            print(f"Local stress tensor (t,n):\n{sigma_local}")
            print(f"theta (deg): {np.degrees(theta):.2f}, n_hat: {n_hat}")

        n_hat_normal = np.array([0.0, -1.0])
        n_hat_normal = n_hat_normal / np.linalg.norm(n_hat_normal)

        n_hat_tangential = np.array([1.0, 0.0])
        n_hat_tangential = n_hat_tangential / np.linalg.norm(n_hat_tangential)

        # Traction vector
        t_model2 = sigma_local @ n_hat_normal

        # Normal component
        t_normal_component = np.dot(t_model2, n_hat_normal) * n_hat_normal

        # Tangential component (projection subtraction)
        t_tangential_component = t_model2 - t_normal_component

        # Store scalar components if desired
        t_model_n_list.append((t_model2[1]))
        t_model_t_list.append((t_tangential_component[0]))

        # Same thing but calculated directly from distribution function
        flux_tt_dist = momentum_flux_from_3d_distribution(0, 0, data, n_avg)
        flux_tn_dist = momentum_flux_from_3d_distribution(0, 1, data, n_avg)
        flux_nn_dist = momentum_flux_from_3d_distribution(1, 1, data, n_avg)

        sigma_local = -np.array([[flux_tt_dist, flux_tn_dist],
                                [flux_tn_dist, flux_nn_dist]], dtype=float)
        if verbose:
            print(f"Region {ind}, ts {ts}:")
            print(f"Local stress tensor (t,n):\n{sigma_local}")
            print(f"theta (deg): {np.degrees(theta):.2f}, n_hat: {n_hat}")

        n_hat_normal = np.array([0.0, -1.0])
        n_hat_normal = n_hat_normal / np.linalg.norm(n_hat_normal)

        n_hat_tangential = np.array([1.0, 0.0])
        n_hat_tangential = n_hat_tangential / np.linalg.norm(n_hat_tangential)

        # Traction vector
        t_model2 = sigma_local @ n_hat_normal

        # Normal component
        t_normal_component = np.dot(t_model2, n_hat_normal) * n_hat_normal

        # Tangential component (projection subtraction)
        t_tangential_component = t_model2 - t_normal_component

        # Store scalar components if desired
        t_model_n_dist_list.append((t_model2[1]))
        t_model_t_dist_list.append((t_tangential_component[0]))

        R = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]], dtype=float)
        
        sigma_global = R @ sigma_local @ R.T
        # print(f"Global stress tensor (x,y):\n{sigma_global}")
        # Model traction in global coordinates: t_model = sigma_global n_hat
        t_model = n_hat.T @ sigma_global
        t_model_x = float(t_model[0])
        t_model_y = float(t_model[1])
        

        # Data traction components in global coordinates
        t_data_t = float((surf_nz["shx"] + surf_nz["px"]).mean())
        t_data_n = float((surf_nz["shy"] + surf_nz["py"]).mean())
        t_data_tn_vec = np.array([t_data_t, t_data_n])
        R2 = np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta),  np.cos(theta)]], dtype=float)
        t_datatn = R2 @ t_data_tn_vec
        t_data_t_list.append(-t_datatn[0])
        t_data_n_list.append(-t_datatn[1])
        t_data_x = float((surf_nz["px"] + surf_nz["shx"]).mean())
        t_data_y = float((surf_nz["py"] + surf_nz["shy"]).mean())
        # print(f"t_data_x: {t_data_x:.4e}, t_data_Y: {t_data_y:.4e}")
        # print(f"t_model_x: {t_model_x:.4e}, t_model_y: {t_model_y:.4e}")
        # print(f"Error x: {(t_model_x - t_data_x) / t_data_x:.2%}, Error y: {(t_model_y - t_data_y) / t_data_y:.2%}")

        t_model_x_list.append(t_model_x)
        t_model_y_list.append(t_model_y)
        t_data_x_list.append(t_data_x)
        t_data_y_list.append(t_data_y)

        # energy_model = -heat_flux_from_3D_hermite_fit(fit, n_avg)[1]
        energy_model = -fast_heat_flux_from_3D_hermite_fit(fit, n_avg)
        # # energy_model = kinetic_energy_flux_from_2D(fit, n_avg)
        energy_data = float((surf_nz["ke"].mean()))
        energy_model_list.append(energy_model)
        energy_data_list.append(energy_data)

        # energy_model_list.append(0)
        # energy_data_list.append(0)

        energy_dist = -heat_flux_from_3D_distribution(data, n_avg)[1]
        energy_dist_list.append(energy_dist)
        # energy_dist_list.append(0)
        # print(f"Heat flux model: {energy_model:.4e}, Heat flux data: {energy_data:.4e}, Error: {(energy_model - energy_data) / energy_data:.2%}")

    moments = {
        'pn_model': np.array(t_model_n_list),
        'pt_model': np.array(t_model_t_list),
        'pn_dist': np.array(t_model_n_dist_list),
        'pt_dist': np.array(t_model_t_dist_list),
        'pn_dsmc': np.array(t_data_n_list),
        'pt_dsmc': np.array(t_data_t_list),
        'e_model': np.array(energy_model_list),
        'e_dsmc': np.array(energy_data_list),
        'e_dist': np.array(energy_dist_list),
    }

    # for key, value in moments.items():
    #     print(f"{key}: len={len(value)}")
    return moments
