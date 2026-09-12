"""Hand-derived Hermite-expansion fitting for the near-wall VDF.

Relocated verbatim (cut/paste, no edits) from
src/rarefied/moments/proess_3D_distributions.ipynb, cell 2, as part of the
docs/CLEANUP_TODO.md Phase 3 extraction. Canonical per PIPELINE.md stage [5] --
do not "clean up" the math.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import factorial
from scipy.signal import convolve2d

def hermite_polynomial(n, x, alpha=1):
    """Compute the nth Hermite polynomial H_n(x) with scaling factor alpha."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x * hermite_polynomial(n-1, x, alpha)
    else:
        return x * hermite_polynomial(n-1,x,alpha) - (n-1) * alpha * hermite_polynomial(n-2,x,alpha)

def pdf_hermite_3d(x,y,z,terms,alpha):
    """Evaluate the 3D PDF from the Hermite expansion at points (x,y,z)."""
    pdf = np.zeros_like(x)
    n_terms = terms.shape[0]
    for l in range(n_terms):
        Hl = hermite_polynomial(l, x, alpha=alpha)
        for m in range(n_terms):
            Hm = hermite_polynomial(m, y, alpha=alpha)
            for n in range(n_terms):
                Hn = hermite_polynomial(n, z, alpha=alpha)
                pdf += terms[l, m, n] * Hl * Hm * Hn * np.exp(-(x**2 + y**2 + z**2)/(2*alpha)) * 1 / ((2 * np.pi * alpha)**(3/2))
    return pdf

def pdf_hermite_2d(x,y,terms,alpha):
    """Evaluate the 2D PDF from the Hermite expansion at points (x,y)."""
    pdf = np.zeros_like(x)
    n_terms = terms.shape[0]
    for m in range(n_terms):
        Hm = hermite_polynomial(m, x, alpha=alpha)
        for n in range(n_terms):
            Hn = hermite_polynomial(n, y, alpha=alpha)
            pdf += terms[m, n] * Hm * Hn * np.exp(-(x**2 + y**2)/(2*alpha)) * 1 / (2 * np.pi * alpha)
    return pdf

def get_mean_var_3D(data):
    """Calculate mean and variance from 3D histogram data."""
    Tcenters = data['Tdata']
    Ncenters = data['Ndata']
    Zcenters = data['Zdata']
    pmf = data['joint_pmf']
    pmf /= np.sum(np.abs(pmf)) # Ensure it's normalized

    mean_T = np.sum(Tcenters * pmf)
    mean_N = np.sum(Ncenters * pmf)
    mean_Z = np.sum(Zcenters * pmf)
    var_T = np.sum((Tcenters - mean_T)**2 * pmf)
    var_N = np.sum((Ncenters - mean_N)**2 * pmf)
    var_Z = np.sum((Zcenters - mean_Z)**2 * pmf)
    # print(f"Mean T: {mean_T:.2f}, Mean N: {mean_N:.2f}, Mean Z: {mean_Z:.2f}")
    # print(f"Var T: {var_T:.2e}, Var N: {var_N:.2e}, Var Z: {var_Z:.2e}")
    return mean_T, mean_N, mean_Z, var_T, var_N, var_Z

def get_mean_var_2D(data):
    """Calculate mean and variance from 2D histogram data."""
    Tcenters = data['Tdata']
    Ncenters = data['Ndata']
    pmf = data['joint_pmf']
    pmf /= np.sum(np.abs(pmf)) # Ensure it's normalized

    mean_T = np.sum(Tcenters * pmf)
    mean_N = np.sum(Ncenters * pmf)
    var_T = np.sum((Tcenters - mean_T)**2 * pmf)
    var_N = np.sum((Ncenters - mean_N)**2 * pmf)
    print(f"Mean T: {mean_T:.2f}, Mean N: {mean_N:.2f}")
    print(f"Var T: {var_T:.2e}, Var N: {var_N:.2e}")
    return mean_T, mean_N, var_T, var_N

def fit_hermite_distribution_3d(data, type=1, use_quadrature=False, PRECISION = 20, mean=None, var=None, keep_positive = False,m_spec=6.36e-26, n_terms=3, plot=False, window=None):
    """Fit a PDF using a Hermite polynomial expansion on normalized data.
    
    Parameters:
    -----------
    data : dict
        Dictionary with 'bin_coords' and 'bin_normalized' keys
    type : int
        Type of data rescaling/fit to perform
            1 = Standardize using calculated means/variances
            2 = Standardize using given means and variances
    bin_width : float
        Width of histogram bins
    m_spec : float
        Mass of particle (default: argon)
    n_terms : int
        Number of Hermite terms to compute
    use_quadrature : bool
        If True, use Gaussian quadrature for integration
    plot : bool
        If True, plot both the normalized and unnormalized distribution fits
    window : tuple of (min, max) or None
        If provided, only data within [min, max] range is used for fitting

    Returns:
    --------
    mean, alpha, T, terms : tuple
        Mean and variance of original distribution, temperature, and Hermite coefficients
    """
    
    # Extract data
    Tcenters = data['Tdata']
    Ncenters = data['Ndata']
    Zcenters = data['Zdata']
    pmf = data['joint_pmf']
    pmf /= np.sum(np.abs(pmf)) # Ensure it's normalized
    pdf = pmf / ((Tcenters[2, 1, 1] - Tcenters[1, 1, 1]) * 
                 (Ncenters[1, 2, 1] - Ncenters[1, 1 ,1]) * 
                 (Zcenters[1, 1, 2] - Zcenters[1, 1, 1])) # Convert PMF to PDF by dividing by bin volume

    
    terms = np.zeros((n_terms, n_terms, n_terms))
    if type == 1:
        mean_T = np.sum(Tcenters * pmf)
        mean_N = np.sum(Ncenters * pmf)
        mean_Z = np.sum(Zcenters * pmf)
        var_T = np.sum((Tcenters - mean_T)**2 * pmf)
        var_N = np.sum((Ncenters - mean_N)**2 * pmf)
        var_Z = np.sum((Zcenters - mean_Z)**2 * pmf)
        std_T = np.sqrt(var_T)
        std_N = np.sqrt(var_N)
        std_Z = np.sqrt(var_Z)

        # print(f"Calculated Mean T: {mean_T:.2f}, Mean N: {mean_N:.2f}, Mean Z: {mean_Z:.2f}")
        # print(f" CalculatedVar T: {var_T:.2e}, Var N: {var_N:.2e}, Var Z: {var_Z:.2e}")
        # print(f"Calculated Std T: {std_T:.2f}, Std N: {std_N:.2f}, Std Z: {std_Z:.2f}")
    
        normalized_T = (Tcenters - mean_T) / std_T
        normalized_N = (Ncenters - mean_N) / std_N
        normalized_Z = (Zcenters - mean_Z) / std_Z

        dT_norm = normalized_T[2, 1, 1] - normalized_T[1, 1, 1]
        dN_norm = normalized_N[1, 2, 1] - normalized_N[1, 1, 1]
        dZ_norm = normalized_Z[1, 1, 2] - normalized_Z[1, 1, 1]
        # dN_norm = normalized_N[2, 1] - normalized_N[1, 1]
        # print("BBBBBBB dt_Norm:", dT_norm, "dN_norm:", dN_norm)
        normalized_pdf = pdf * std_T*std_N*std_Z # Adjust for change of variables
        
        if use_quadrature:
            # Get quadrature points and weights
            xq, wq = hermegauss(PRECISION)
            xq_T, xq_N, xq_Z = np.meshgrid(xq, xq, xq, indexing="ij")
            W = wq[:, None, None] * wq[None, :, None] * wq[None, None, :]

            # Extract 1D coordinate axes (assumes structured grid)
            T_axis = normalized_T[:, 0, 0]
            N_axis = normalized_N[0, :, 0]
            Z_axis = normalized_Z[0, 0, :]

            # Find nearest indices for each quadrature point
            T_idx = np.abs(T_axis[:, None] - xq[None, :]).argmin(axis=0)
            N_idx = np.abs(N_axis[:, None] - xq[None, :]).argmin(axis=0)
            Z_idx = np.abs(Z_axis[:, None] - xq[None, :]).argmin(axis=0)

            # Build 2D index grids
            T_idx_3d, N_idx_3d, Z_idx_3d = np.meshgrid(
                T_idx, N_idx, Z_idx, indexing="ij"
            )

            # Evaluate PDF at quadrature points via nearest bin
            fq = normalized_pdf[T_idx_3d, N_idx_3d, Z_idx_3d]

            gq = fq / np.exp(-0.5 * (xq_T**2 + xq_N**2 + xq_Z**2))
            
            for m in range(n_terms):
                Hm = hermite_polynomial(m, xq_T, alpha=1)
                for n in range(n_terms):
                    Hn = hermite_polynomial(n, xq_N, alpha=1)
                    for p in range(n_terms):
                        Hp = hermite_polynomial(p, xq_Z, alpha=1)

                        xi = np.sum(W * gq * Hm * Hn * Hp)
                        terms[m, n, p] = xi / (
                            factorial(m) * factorial(n) * factorial(p)
                        )
        else:
            for m in range(n_terms):
                Hm = hermite_polynomial(m, normalized_T, alpha=1)
                for n in range(n_terms):
                    Hn = hermite_polynomial(n, normalized_N, alpha=1)
                    for p in range(n_terms):
                        Hp = hermite_polynomial(p, normalized_Z, alpha=1)
                        terms[m, n, p] = np.sum(normalized_pdf * Hm * Hn * Hp * dT_norm * dN_norm * dZ_norm) / (factorial(m) * factorial(n) * factorial(p))

    elif type == 2:
        mean_T, mean_N, mean_Z = mean
        var_T, var_N, var_Z = var
        std_T = np.sqrt(var_T)
        std_N = np.sqrt(var_N)
        std_Z = np.sqrt(var_Z)
        normalized_T = (Tcenters - mean_T) / std_T
        normalized_N = (Ncenters - mean_N) / std_N
        normalized_Z = (Zcenters - mean_Z) / std_Z

        dT_norm = normalized_T[2, 1, 1] - normalized_T[1, 1, 1]
        dN_norm = normalized_N[1, 2, 1] - normalized_N[1, 1, 1]
        dZ_norm = normalized_Z[1, 1, 2] - normalized_Z[1, 1, 1]
        normalized_pdf = pdf * std_T*std_N*std_Z # Adjust for change of variables

        if use_quadrature:
            # Get quadrature points and weights
            xq, wq = hermegauss(PRECISION)
            xq_T, xq_N, xq_Z = np.meshgrid(xq, xq, xq, indexing="ij")
            W = wq[:, None, None] * wq[None, :, None] * wq[None, None, :]

            # Extract 1D coordinate axes (assumes structured grid)
            T_axis = normalized_T[:, 0, 0]
            N_axis = normalized_N[0, :, 0]
            Z_axis = normalized_Z[0, 0, :]

            # Find nearest indices for each quadrature point
            T_idx = np.abs(T_axis[:, None] - xq[None, :]).argmin(axis=0)
            N_idx = np.abs(N_axis[:, None] - xq[None, :]).argmin(axis=0)
            Z_idx = np.abs(Z_axis[:, None] - xq[None, :]).argmin(axis=0)

            # Build 2D index grids
            T_idx_3d, N_idx_3d, Z_idx_3d = np.meshgrid(
                T_idx, N_idx, Z_idx, indexing="ij"
            )

            # Evaluate PDF at quadrature points via nearest bin
            fq = normalized_pdf[T_idx_3d, N_idx_3d, Z_idx_3d]

            gq = fq / np.exp(-0.5 * (xq_T**2 + xq_N**2 + xq_Z**2))
            
            for m in range(n_terms):
                Hm = hermite_polynomial(m, xq_T, alpha=1)
                for n in range(n_terms):
                    Hn = hermite_polynomial(n, xq_N, alpha=1)
                    for p in range(n_terms):
                        Hp = hermite_polynomial(p, xq_Z, alpha=1)

                        xi = np.sum(W * gq * Hm * Hn * Hp)
                        terms[m, n, p] = xi / (
                            factorial(m) * factorial(n) * factorial(p)
                        )
        else: 
            for m in range(n_terms):
                Hm = hermite_polynomial(m, normalized_T, alpha=1)
                for n in range(n_terms):
                    Hn = hermite_polynomial(n, normalized_N, alpha=1)
                    for p in range(n_terms):
                        Hp = hermite_polynomial(p, normalized_Z, alpha=1)
                        terms[m, n, p] = np.sum(normalized_pdf * Hm * Hn * Hp * dT_norm * dN_norm * dZ_norm) / (factorial(m) * factorial(n) * factorial(p))

    # print("Hermite coefficients:", terms)
    fit = {
        'terms' : terms,
        'mean_T': mean_T,
        'mean_N': mean_N,
        'mean_Z': mean_Z,
        'var_T': var_T,
        'var_N': var_N,
        'var_Z': var_Z
    }
    # Calculate Error
    approx_pdf = pdf_hermite_3d(normalized_T, normalized_N, normalized_Z, terms, alpha=1)

    if keep_positive:
        approx_pdf = np.maximum(approx_pdf, 0)
        approx_pdf /= np.sum(approx_pdf * dT_norm * dN_norm * dZ_norm)

    # Sanity Checking
    # print("Normalized Integral: ", np.sum(normalized_pdf * dT_norm * dN_norm * dZ_norm))
    # print("dt_Norm: ", dT_norm)
    # print("dN_Norm: ", dN_norm)
    # print("dZ_Norm: ", dZ_norm)
    # print("sum approx_pdf:", np.sum(approx_pdf))
    # print("Approx Integral: ", np.sum(approx_pdf * dT_norm * dN_norm * dZ_norm))

    abs_error = np.abs(normalized_pdf - approx_pdf)
    mean_L1_error = np.sum(abs_error) / np.size(abs_error)
    mean_L2_error = np.sqrt(np.sum(abs_error**2) / np.size(abs_error))
    L_inf_error = np.max(abs_error)
    error = {
        'mean_L1': mean_L1_error,
        'mean_L2': mean_L2_error,
        'L_inf': L_inf_error
    }

    if plot==True:
        elev = 0
        azim = 90
        # Normalized Distribution (Centered about zero with unit variance)
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(221, projection='3d')
        srf = ax.plot_surface(normalized_T, normalized_N, normalized_pdf, cmap='viridis')
        ax.set_title('Normalized Distribution')
        ax.set_xlabel('T [-]')
        ax.set_ylabel('N [-]')
        ax.view_init(elev=elev, azim=azim)

        # Unnormalized Distribution (Original T and N coordinates)
        ax = fig.add_subplot(222, projection='3d')
        srf = ax.plot_surface(Tcenters, Ncenters, pdf, cmap='viridis')
        ax.set_title('Original Distribution')
        ax.set_xlabel('T [m/s]')
        ax.set_ylabel('N [m/s]')
        ax.set_zlabel('Probability Density')
        ax.view_init(elev=elev, azim=azim)

        #Hermite Approximation (Back in Normalized Coordinates)
        approx_pdf = np.zeros_like(pmf)
        for m in range(n_terms):
            Hm = hermite_polynomial(m, normalized_T, alpha=1)
            for n in range(n_terms):
                Hn = hermite_polynomial(n, normalized_N, alpha=1)
                for p in range(n_terms):
                    Hp = hermite_polynomial(p, normalized_Z, alpha=1)
                    approx_pdf += terms[m, n, p] * Hm * Hn * Hp * np.exp(-0.5 * (normalized_T**2 + normalized_N**2 + normalized_Z**2)) / (2 * np.pi)**(3/2)

        if keep_positive:
            approx_pdf = np.maximum(approx_pdf, 0)
            approx_pdf /= np.sum(approx_pdf * dT_norm * dN_norm * dZ_norm)
        ax = fig.add_subplot(223, projection='3d')
        srf = ax.plot_surface(normalized_T, normalized_N, approx_pdf, cmap='viridis')
        ax.set_title('Hermite Approximation')
        ax.set_xlabel('T [-]')
        ax.set_ylabel('N [-]')
        ax.set_zlabel('Probability Density')
        ax.view_init(elev=elev, azim=azim)

        # Hermite Approximation in Original Coordinates (Back-transforming the approximation)
        ax = fig.add_subplot(224, projection='3d')
        approx_pdf_unnormalized = approx_pdf / (std_T * std_N * std_Z) # Adjust for change of variables
        srf = ax.plot_surface(Tcenters, Ncenters, approx_pdf_unnormalized, cmap='viridis')
        ax.set_title('Hermite Approximation (Original Coordinates)')
        ax.set_xlabel('T [m/s]')
        ax.set_ylabel('N [m/s]')
        ax.set_zlabel('Probability Density')
        ax.view_init(elev=elev, azim=azim)
        plt.show()

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(121, projection='3d')

        # import numpy as np

        # 1. Define a smoothing kernel (3x3 average)
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

        # 2. Apply the smoothing
        # 'same' keeps the output matrix the same size as the input
        smoothed_pdf = convolve2d(normalized_pdf, kernel, mode='same')
        smoothed_pdf = convolve2d(smoothed_pdf, kernel, mode='same') # Apply twice for stronger smoothing
        renormalized_pdf = smoothed_pdf / np.sum(smoothed_pdf * dT_norm * dN_norm * dZ_norm) # Renormalize after smoothing
        srf = ax.plot_surface(normalized_T, normalized_N, normalized_pdf, cmap='viridis')
        ax.set_title('Normalized Distribution')
        ax.set_xlabel('T [-]')
        ax.set_ylabel('N [-]')
        ax.view_init(elev=elev, azim=azim)
        ax = fig.add_subplot(122, projection='3d')
        srf = ax.plot_surface(normalized_T, normalized_N, renormalized_pdf, cmap='viridis')
        ax.set_title('Smoothed Normalized Distribution')
        ax.set_xlabel('T [-]')
        ax.set_ylabel('N [-]')  
        ax.view_init(elev=elev, azim=azim)
        plt.show()
        # print("Integral of original PDF:", np.sum(approx_pdf_unnormalized * (Tcenters[1, 2] - Tcenters[1, 1]) * (Ncenters[2, 1] - Ncenters[1, 1])))
        # Slice through T=0
        # idx = normalized_T.shape[1] // 2
        # plt.subplot(224)
        # plt.plot(normalized_N[:, idx], normalized_pdf[:, idx], label="True PDF")
        # plt.plot(normalized_N[:, idx], approx_pdf[:, idx], label="Hermite Approximation", linestyle='--')
        # plt.title('Slice at T=0 (Normalized)')
        # plt.xlabel('Normalized N Velocity')
        # plt.ylabel('Probability Density')
        # plt.legend()
        # plt.grid(True, alpha=0.3)

    # mean = (mean_T, mean_N)
    # var = (var_T, var_N)
    return fit, error

def fit_hermite_distribution_2d(data, type=1, use_quadrature=False, PRECISION = 20, mean=None, var=None, keep_positive = False,m_spec=6.36e-26, n_terms=3, plot=False, window=None):
    """Fit a PDF using a Hermite polynomial expansion on normalized data.
    
    Parameters:
    -----------
    data : dict
        Dictionary with 'bin_coords' and 'bin_normalized' keys
    type : int
        Type of data rescaling/fit to perform
            1 = Standardize using calculated means/variances
            2 = Standardize using given means and variances
    bin_width : float
        Width of histogram bins
    m_spec : float
        Mass of particle (default: argon)
    n_terms : int
        Number of Hermite terms to compute
    use_quadrature : bool
        If True, use Gaussian quadrature for integration
    plot : bool
        If True, plot both the normalized and unnormalized distribution fits
    window : tuple of (min, max) or None
        If provided, only data within [min, max] range is used for fitting
    
    Returns:
    --------
    mean, alpha, T, terms : tuple
        Mean and variance of original distribution, temperature, and Hermite coefficients
    """

    # Extract data
    Tcenters = data['Tdata']
    Ncenters = data['Ndata']
    pmf = data['joint_pmf']
    pmf /= np.sum(np.abs(pmf)) # Ensure it's normalized
    pdf = pmf / ((Tcenters[1, 2] - Tcenters[1, 1]) * (Ncenters[2, 1] - Ncenters[1, 1]))

    
    terms = np.zeros((n_terms, n_terms))
    if type == 1:
        mean_T = np.sum(Tcenters * pmf)
        mean_N = np.sum(Ncenters * pmf)
        var_T = np.sum((Tcenters - mean_T)**2 * pmf)
        var_N = np.sum((Ncenters - mean_N)**2 * pmf)
        std_T = np.sqrt(var_T)
        std_N = np.sqrt(var_N)
        cov_TN = np.sum((Tcenters - mean_T) * (Ncenters - mean_N) * pmf)

        print(f"Calculated Mean T: {mean_T:.2f}, Mean N: {mean_N:.2f}")
        print(f" CalculatedVar T: {var_T:.2e}, Var N: {var_N:.2e}, Cov(T,N): {cov_TN:.2e}")
        print(f"Calculated Std T: {std_T:.2f}, Std N: {std_N:.2f}")
    
        normalized_T = (Tcenters - mean_T) / std_T
        normalized_N = (Ncenters - mean_N) / std_N

        dT_norm = normalized_T[1, 2] - normalized_T[1, 1]
        dN_norm = normalized_N[2, 1] - normalized_N[1, 1]
        print("BBBBBBB dt_Norm:", dT_norm, "dN_norm:", dN_norm)
        normalized_pdf = pdf * std_T*std_N # Adjust for change of variables
        
        if use_quadrature:
            # Get quadrature points and weights
            xq, wq = hermegauss(PRECISION)
            xq_T, xq_N = np.meshgrid(xq, xq)
            W = np.outer(wq, wq)

            # Extract 1D coordinate axes (assumes structured grid)
            T_axis = normalized_T[0, :]   # shape (Nx,)
            N_axis = normalized_N[:, 0]   # shape (Ny,)

            # Find nearest indices for each quadrature point
            T_idx = np.abs(T_axis[:, None] - xq[None, :]).argmin(axis=0)
            N_idx = np.abs(N_axis[:, None] - xq[None, :]).argmin(axis=0)

            # Build 2D index grids
            T_idx_2d, N_idx_2d = np.meshgrid(T_idx, N_idx)

            # Evaluate PDF at quadrature points via nearest bin
            fq = normalized_pdf[N_idx_2d, T_idx_2d]   # shape (PRECISION, PRECISION)

            gq = fq / np.exp(-0.5 * (xq_T**2 + xq_N**2))  # shape (PRECISION, PRECISION)
            
            for m in range(n_terms):
                Hm = hermite_polynomial(m, xq_T, alpha=1)
                for n in range(n_terms):
                    Hn = hermite_polynomial(n, xq_N, alpha=1)
                    xi = np.sum(W * gq * Hm * Hn)
                    terms[m, n] = xi / (factorial(m) * factorial(n))
        else:
            for m in range(n_terms):
                Hm = hermite_polynomial(m, normalized_T, alpha=1)
                for n in range(n_terms):
                    Hn = hermite_polynomial(n, normalized_N, alpha=1)
                    terms[m, n] = np.sum(normalized_pdf * Hm * Hn * dT_norm * dN_norm) / (factorial(m) * factorial(n))

    elif type == 2:
        mean_T, mean_N = mean
        var_T, var_N = var
        std_T = np.sqrt(var_T)
        std_N = np.sqrt(var_N)
        print("mean_T from input:", mean_T, "mean_N from input:", mean_N)
        print("var_T from input:", var_T, "var_N from input:", var_N)
        print("std_T from input:", std_T, "std_N from input:", std_N)
        normalized_T = (Tcenters - mean_T) / std_T
        normalized_N = (Ncenters - mean_N) / std_N

        dT_norm = normalized_T[1, 2] - normalized_T[1, 1]
        dN_norm = normalized_N[2, 1] - normalized_N[1, 1]
        print("AAAAAA dt_Norm:", dT_norm, "dN_norm:", dN_norm)
        normalized_pdf = pdf * std_T*std_N # Adjust for change of variables
        
        if use_quadrature:
            # Get quadrature points and weights
            xq, wq = hermegauss(PRECISION)
            xq_T, xq_N = np.meshgrid(xq, xq)
            W = np.outer(wq, wq)

            # Extract 1D coordinate axes (assumes structured grid)
            T_axis = normalized_T[0, :]   # shape (Nx,)
            N_axis = normalized_N[:, 0]   # shape (Ny,)

            # Find nearest indices for each quadrature point
            T_idx = np.abs(T_axis[:, None] - xq[None, :]).argmin(axis=0)
            N_idx = np.abs(N_axis[:, None] - xq[None, :]).argmin(axis=0)

            # Build 2D index grids
            T_idx_2d, N_idx_2d = np.meshgrid(T_idx, N_idx)

            # Evaluate PDF at quadrature points via nearest bin
            fq = normalized_pdf[N_idx_2d, T_idx_2d]   # shape (PRECISION, PRECISION)

            gq = fq / np.exp(-0.5 * (xq_T**2 + xq_N**2))  # shape (PRECISION, PRECISION)
            
            for m in range(n_terms):
                Hm = hermite_polynomial(m, xq_T, alpha=1)
                for n in range(n_terms):
                    Hn = hermite_polynomial(n, xq_N, alpha=1)
                    xi = np.sum(W * gq * Hm * Hn)
                    terms[m, n] = xi / (factorial(m) * factorial(n))
        else: 
            for m in range(n_terms):
                Hm = hermite_polynomial(m, normalized_T, alpha=1)
                for n in range(n_terms):
                    Hn = hermite_polynomial(n, normalized_N, alpha=1)
                    terms[m, n] = np.sum(normalized_pdf * Hm * Hn * dT_norm * dN_norm) / (factorial(m) * factorial(n))

    print("Hermite coefficients:", terms)
    fit = {
        'terms' : terms,
        'mean_T': mean_T,
        'mean_N': mean_N,
        'var_T': var_T,
        'var_N': var_N,
    }
    # Calculate Error
    approx_pdf = pdf_hermite_2d(normalized_T, normalized_N, terms, alpha=1)

    if keep_positive:
        approx_pdf = np.maximum(approx_pdf, 0)
        approx_pdf /= np.sum(approx_pdf * dT_norm * dN_norm)

    # Sanity Checking
    print("Normalized Integral: ", np.sum(normalized_pdf * dT_norm * dN_norm))
    print("dt_Norm: ", dT_norm)
    print("dN_Norm: ", dN_norm)
    print("sum approx_pdf:", np.sum(approx_pdf))
    print("Approx Integral: ", np.sum(approx_pdf * dT_norm * dN_norm))

    abs_error = np.abs(normalized_pdf - approx_pdf)
    mean_L1_error = np.sum(abs_error) / np.size(abs_error)
    mean_L2_error = np.sqrt(np.sum(abs_error**2) / np.size(abs_error))
    L_inf_error = np.max(abs_error)
    error = {
        'mean_L1': mean_L1_error,
        'mean_L2': mean_L2_error,
        'L_inf': L_inf_error
    }

    if plot==True:
        elev = 0
        azim = 90
        # Normalized Distribution (Centered about zero with unit variance)
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(221, projection='3d')
        srf = ax.plot_surface(normalized_T, normalized_N, normalized_pdf, cmap='viridis')
        ax.set_title('Normalized Distribution')
        ax.set_xlabel('T [-]')
        ax.set_ylabel('N [-]')
        ax.view_init(elev=elev, azim=azim)

        # Unnormalized Distribution (Original T and N coordinates)
        ax = fig.add_subplot(222, projection='3d')
        srf = ax.plot_surface(Tcenters, Ncenters, pdf, cmap='viridis')
        ax.set_title('Original Distribution')
        ax.set_xlabel('T [m/s]')
        ax.set_ylabel('N [m/s]')
        ax.set_zlabel('Probability Density')
        ax.view_init(elev=elev, azim=azim)

        #Hermite Approximation (Back in Normalized Coordinates)
        approx_pdf = np.zeros_like(pmf)
        for m in range(n_terms):
            Hm = hermite_polynomial(m, normalized_T, alpha=1)
            for n in range(n_terms):
                Hn = hermite_polynomial(n, normalized_N, alpha=1)
                approx_pdf += terms[m, n] * Hm * Hn * np.exp(-0.5 * (normalized_T**2 + normalized_N**2)) / (2 * np.pi)  

        if keep_positive:
            approx_pdf = np.maximum(approx_pdf, 0)
            approx_pdf /= np.sum(approx_pdf * dT_norm * dN_norm)
        ax = fig.add_subplot(223, projection='3d')
        srf = ax.plot_surface(normalized_T, normalized_N, approx_pdf, cmap='viridis')
        ax.set_title('Hermite Approximation')
        ax.set_xlabel('T [-]')
        ax.set_ylabel('N [-]')
        ax.set_zlabel('Probability Density')
        ax.view_init(elev=elev, azim=azim)

        # Hermite Approximation in Original Coordinates (Back-transforming the approximation)
        ax = fig.add_subplot(224, projection='3d')
        approx_pdf_unnormalized = approx_pdf / (std_T * std_N) # Adjust for change of variables
        srf = ax.plot_surface(Tcenters, Ncenters, approx_pdf_unnormalized, cmap='viridis')
        ax.set_title('Hermite Approximation (Original Coordinates)')
        ax.set_xlabel('T [m/s]')
        ax.set_ylabel('N [m/s]')
        ax.set_zlabel('Probability Density')
        ax.view_init(elev=elev, azim=azim)
        plt.show()

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(121, projection='3d')

        # import numpy as np

        # 1. Define a smoothing kernel (3x3 average)
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)

        # 2. Apply the smoothing
        # 'same' keeps the output matrix the same size as the input
        smoothed_pdf = convolve2d(normalized_pdf, kernel, mode='same')
        smoothed_pdf = convolve2d(smoothed_pdf, kernel, mode='same') # Apply twice for stronger smoothing
        renormalized_pdf = smoothed_pdf / np.sum(smoothed_pdf * dT_norm * dN_norm) # Renormalize after smoothing
        srf = ax.plot_surface(normalized_T, normalized_N, normalized_pdf, cmap='viridis')
        ax.set_title('Normalized Distribution')
        ax.set_xlabel('T [-]')
        ax.set_ylabel('N [-]')
        ax.view_init(elev=elev, azim=azim)
        ax = fig.add_subplot(122, projection='3d')
        srf = ax.plot_surface(normalized_T, normalized_N, renormalized_pdf, cmap='viridis')
        ax.set_title('Smoothed Normalized Distribution')
        ax.set_xlabel('T [-]')
        ax.set_ylabel('N [-]')  
        ax.view_init(elev=elev, azim=azim)
        plt.show()
        # print("Integral of original PDF:", np.sum(approx_pdf_unnormalized * (Tcenters[1, 2] - Tcenters[1, 1]) * (Ncenters[2, 1] - Ncenters[1, 1])))
        # Slice through T=0
        # idx = normalized_T.shape[1] // 2
        # plt.subplot(224)
        # plt.plot(normalized_N[:, idx], normalized_pdf[:, idx], label="True PDF")
        # plt.plot(normalized_N[:, idx], approx_pdf[:, idx], label="Hermite Approximation", linestyle='--')
        # plt.title('Slice at T=0 (Normalized)')
        # plt.xlabel('Normalized N Velocity')
        # plt.ylabel('Probability Density')
        # plt.legend()
        # plt.grid(True, alpha=0.3)

    # mean = (mean_T, mean_N)
    # var = (var_T, var_N)
    return fit, error
