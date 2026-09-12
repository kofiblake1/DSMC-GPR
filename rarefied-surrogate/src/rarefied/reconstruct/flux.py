"""Moments -> dimensional surface stress/heat-flux reconstruction.

Relocated verbatim (cut/paste, no edits) from
src/rarefied/moments/proess_3D_distributions.ipynb, cell 2, as part of the
docs/CLEANUP_TODO.md Phase 3 extraction. Canonical per PIPELINE.md stage [6] --
do not "clean up" the math.

momentum_flux_from_3D_hermite_fit / momentum_flux_from_2D_hermite_fit /
kinetic_energy_flux_from_2D rebuild the PDF grid from the fit coefficients via
pdf_hermite_3d/pdf_hermite_2d, so this module depends on
rarefied.moments.hermite (not just a plain fit-coefficient dict).
"""

import numpy as np
from scipy.special import comb, factorial, factorial2

from rarefied.moments.hermite import hermite_polynomial, pdf_hermite_2d, pdf_hermite_3d

def momentum_flux_from_3D_hermite_fit(i, j, fit, n_i, m_spec = 6.36e-26):
    """Calculate the (i,j)th momentum flux moment from the 3D Hermite fit coefficients."""
    terms = fit['terms']
    mean_T = fit['mean_T']
    mean_N = fit['mean_N']
    mean_Z = fit['mean_Z']
    var_T = fit['var_T']
    var_N = fit['var_N']
    var_Z = fit['var_Z']

    # Calculate high resolution distribution
    vel_range = 5000
    res = 101
    T,N,Z = np.meshgrid(np.linspace(-vel_range, vel_range, res), np.linspace(-vel_range, vel_range, res), np.linspace(-vel_range, vel_range, res), indexing='ij')
    T_shifted = (T - mean_T) / np.sqrt(var_T)
    N_shifted = (N - mean_N) / np.sqrt(var_N)
    Z_shifted = (Z - mean_Z) / np.sqrt(var_Z)
    normalized_pdf = pdf_hermite_3d(T_shifted, N_shifted, Z_shifted, terms, alpha=1)
    original_pdf = normalized_pdf / (np.sqrt(var_T) * np.sqrt(var_N) * np.sqrt(var_Z)) # Adjust for change of variables
    
    dT_norm = T[2, 1, 1] - T[1, 1, 1]
    dN_norm = N[1, 2, 1] - N[1, 1, 1]
    dZ_norm = Z[1, 1, 2] - Z[1, 1, 1]

    if i == 0 and j == 0:
        flux = np.sum(m_spec * n_i * original_pdf * T * T * dT_norm * dN_norm * dZ_norm)
    elif i == 1 and j == 1:
        flux = np.sum(m_spec * n_i * original_pdf * N * N * dT_norm * dN_norm * dZ_norm)
    else:
        half = res // 2
        Thalf = T[:,:half, :]
        Nhalf = N[:,:half, :]
        pdfhalf = original_pdf[:,:half, :]
        flux = np.sum(m_spec * n_i * pdfhalf * Thalf * Nhalf * dT_norm * dN_norm * dZ_norm)

    return flux

def momentum_flux_from_2D_hermite_fit(i, j, fit, n_i, m_spec = 6.36e-26):
    """Calculate the (i,j)th momentum flux moment from the 2D Hermite fit coefficients."""
    terms = fit['terms']
    mean_T = fit['mean_T']
    mean_N = fit['mean_N']
    var_T = fit['var_T']
    var_N = fit['var_N']

    # Calculate high resolution distribution
    vel_range = 5000
    res = 501
    T,N = np.meshgrid(np.linspace(-vel_range, vel_range, res), np.linspace(-vel_range, vel_range, res))
    T_shifted = (T - mean_T) / np.sqrt(var_T)
    N_shifted = (N - mean_N) / np.sqrt(var_N)
    normalized_pdf = pdf_hermite_2d(T_shifted, N_shifted, terms, alpha=1)
    original_pdf = normalized_pdf / (np.sqrt(var_T) * np.sqrt(var_N)) # Adjust for change of variables
    
    dT_norm = T[1, 2] - T[1, 1]
    dN_norm = N[2, 1] - N[1, 1]

    if i == 0 and j == 0:
        flux = np.sum(m_spec * n_i * original_pdf * T * T * dT_norm * dN_norm)
    elif i == 1 and j == 1:
        flux = np.sum(m_spec * n_i * original_pdf * N * N * dT_norm * dN_norm)
    else:
        half = res // 2
        Thalf = T[:half, :]
        Nhalf = N[:half, :]
        pdfhalf = original_pdf[:half, :]
        flux = np.sum(m_spec * n_i * pdfhalf * Thalf * Nhalf * dT_norm * dN_norm)

    return flux

def momentum_flux_from_3d_distribution(i,j, data, n_i, m_spec=6.36e-26):
    """Calculate the (i,j)th momentum flux moment directly from the 3D distribution data."""
    Tcenters = data['Tdata']
    Ncenters = data['Ndata']
    Zcenters = data['Zdata']
    pmf = data['joint_pmf']
    pmf /= np.sum(np.abs(pmf)) # Ensure it's normalized
    pdf = pmf / ((Tcenters[2, 1, 1] - Tcenters[1, 1, 1]) * (Ncenters[1, 2, 1] - Ncenters[1, 1, 1]) * (Zcenters[1, 1, 2] - Zcenters[1, 1, 1]))

    dT = Tcenters[2, 1, 1] - Tcenters[1, 1, 1]
    dN = Ncenters[1, 2, 1] - Ncenters[1, 1, 1]
    dZ = Zcenters[1, 1, 2] - Zcenters[1, 1, 1]

    if i == 0 and j == 0:
        flux = np.sum(m_spec * n_i * pdf * Tcenters * Tcenters * dT * dN * dZ)
    elif i == 1 and j == 1:
        flux = np.sum(m_spec * n_i * pdf * Ncenters * Ncenters * dT * dN * dZ)
    else:
        half = pdf.shape[1] // 2
        Thalf = Tcenters[:,:half, :]
        Nhalf = Ncenters[:,:half, :]
        pdfhalf = pdf[:,:half, :]
        flux = np.sum(m_spec * n_i * pdfhalf * Thalf * Nhalf * dT * dN * dZ)

    return flux

def momentum_flux_from_2d_distribution(i,j, data, n_i, m_spec=6.36e-26):
    """Calculate the (i,j)th momentum flux moment directly from the 2D distribution data."""
    Tcenters = data['Tdata']
    Ncenters = data['Ndata']
    pmf = data['joint_pmf']
    pmf /= np.sum(np.abs(pmf)) # Ensure it's normalized
    pdf = pmf / ((Tcenters[1, 2] - Tcenters[1, 1]) * (Ncenters[2, 1] - Ncenters[1, 1]))

    dT = Tcenters[1, 2] - Tcenters[1, 1]
    dN = Ncenters[2, 1] - Ncenters[1, 1]

    if i == 0 and j == 0:
        flux = np.sum(m_spec * n_i * pdf * Tcenters * Tcenters * dT * dN)
    elif i == 1 and j == 1:
        flux = np.sum(m_spec * n_i * pdf * Ncenters * Ncenters * dT * dN)
    else:
        half = pdf.shape[0] // 2
        Thalf = Tcenters[:half, :]
        Nhalf = Ncenters[:half, :]
        pdfhalf = pdf[:half, :]
        flux = np.sum(m_spec * n_i * pdfhalf * Thalf * Nhalf * dT * dN)

    return flux

def kinetic_energy_flux_from_2D(fit, n_i, m_spec=6.36e-26):
    """Compute kinetic energy flux to a surface (SPARTA ke equivalent)."""

    terms = fit['terms']
    mean_T = fit['mean_T']
    mean_N = fit['mean_N']
    var_T = fit['var_T']
    var_N = fit['var_N']

    # --- Velocity grid ---
    vel_range = 5000
    res = 251
    T, N = np.meshgrid(
        np.linspace(-vel_range, vel_range, res),
        np.linspace(-vel_range, vel_range, res)
    )

    # --- Build PDF ---
    T_shifted = (T - mean_T) / np.sqrt(var_T)
    N_shifted = (N - mean_N) / np.sqrt(var_N)

    normalized_pdf = pdf_hermite_2d(T_shifted, N_shifted, terms, alpha=1)
    pdf = normalized_pdf / (np.sqrt(var_T) * np.sqrt(var_N))

    dT = T[0,1] - T[0,0]
    dN = N[1,0] - N[0,0]

    # --- 3D energy closure ---
    # <v3^2> = variance
    var_V3 = (var_T + var_N) / 2

    energy = 0.5 * m_spec * (T**2 + N**2 + var_V3)

    # --- Split incoming/outgoing ---
    incoming = N < 0
    outgoing = N > 0

    # --- Flux-weighted integrals ---
    E_in = np.sum(
        n_i * pdf[incoming] *
        energy[incoming] *
        np.abs(N[incoming]) *
        dT * dN
    )

    E_out = np.sum(
        n_i * pdf[outgoing] *
        energy[outgoing] *
        np.abs(N[outgoing]) *
        dT * dN
    )

    # --- SPARTA convention: energy lost by particles is positive
    E_flux = E_in - E_out

    return E_flux

def heat_flux_from_3D_hermite_fit(fit, n_i, m_spec = 6.36e-26):
    """Calculate the heat flux from the 3D Hermite fit coefficients."""
    terms = fit['terms']
    mean_T = fit['mean_T']
    mean_N = fit['mean_N']
    mean_Z = fit['mean_Z']
    var_T = fit['var_T']
    var_N = fit['var_N']
    var_Z = fit['var_Z']

    # Calculate high resolution distribution
    vel_range = 5000
    res = 101
    T,N,Z = np.meshgrid(np.linspace(-vel_range, vel_range, res), np.linspace(-vel_range, vel_range, res), np.linspace(-vel_range, vel_range, res), indexing='ij')
    T_shifted = (T - mean_T) / np.sqrt(var_T)
    N_shifted = (N - mean_N) / np.sqrt(var_N)
    Z_shifted = (Z - mean_Z) / np.sqrt(var_Z)
    
    normalized_pdf = pdf_hermite_3d(T_shifted, N_shifted, Z_shifted, terms, alpha=1)
    original_pdf = normalized_pdf / (np.sqrt(var_T) * np.sqrt(var_N) * np.sqrt(var_Z)) # Adjust for change of variables
    
    dT_norm = T[2, 1, 1] - T[1, 1, 1]
    dN_norm = N[1, 2, 1] - N[1, 1, 1]
    dZ_norm = Z[1, 1, 2] - Z[1, 1, 1]

    kinetic_energy = 0.5 * m_spec * (T**2 + N**2 + Z**2)
    heat_flux_T = np.sum(n_i * original_pdf * kinetic_energy * T * dT_norm * dN_norm * dZ_norm)
    heat_flux_N = np.sum(n_i * original_pdf * kinetic_energy * N * dT_norm * dN_norm * dZ_norm)
    heat_flux_Z = np.sum(n_i * original_pdf * kinetic_energy * Z * dT_norm * dN_norm * dZ_norm)

    return heat_flux_T, heat_flux_N, heat_flux_Z

def heat_flux_from_2D_hermite_fit(fit, n_i, m_spec = 6.36e-26):
    """Calculate the heat flux from the 2D Hermite fit coefficients."""
    terms = fit['terms']
    mean_T = fit['mean_T']
    mean_N = fit['mean_N']
    var_T = fit['var_T']
    var_N = fit['var_N']

    # Calculate high resolution distribution
    vel_range = 5000
    res = 501
    T,N = np.meshgrid(np.linspace(-vel_range, vel_range, res), np.linspace(-vel_range, vel_range, res))
    T_shifted = (T - mean_T) / np.sqrt(var_T)
    N_shifted = (N - mean_N) / np.sqrt(var_N)
    normalized_pdf = pdf_hermite_2d(T_shifted, N_shifted, terms, alpha=1)
    original_pdf = normalized_pdf / (np.sqrt(var_T) * np.sqrt(var_N)) # Adjust for change of variables
    
    dT_norm = T[1, 2] - T[1, 1]
    dN_norm = N[2, 1] - N[1, 1]

    kinetic_energy = 0.5 * m_spec * (T**2 + N**2)
    heat_flux_T_1 = np.sum(n_i * original_pdf * kinetic_energy * T * dT_norm * dN_norm)
    heat_flux_N_1 = np.sum(n_i * original_pdf * kinetic_energy * N * dT_norm * dN_norm)

    # Calculate incoming / outgoing contributions
    incoming = N < 0
    outgoing = N > 0

    hfN1_in = np.sum(n_i * original_pdf[incoming] * kinetic_energy[incoming] * N[incoming] * dT_norm * dN_norm)
    hfN1_out = np.sum(n_i * original_pdf[outgoing] * kinetic_energy[outgoing] * N[outgoing] * dT_norm * dN_norm)
    print(f"hfN1_in: {hfN1_in:.4e} W/m², hfN1_out: {hfN1_out:.4e} W/m²")
    # Non 3D Correction, 3rd dimension Maxwellian is centered at zero with variance equal to average of T and N variances

    heat_flux_T_2 = np.sum(0.5 * m_spec * n_i * T * original_pdf * dT_norm * dN_norm)
    heat_flux_N_2 = np.sum(0.5 * m_spec * n_i * N * original_pdf * dT_norm * dN_norm)

    v3 = np.linspace(-vel_range, vel_range, res)
    dV3 = v3[1] - v3[0]
    var_V3 = (var_T + var_N) / 2 # Approximate 3D variance as average of T and N variances
    pdf_v3 = 1 / (np.sqrt(2 * np.pi * var_V3)) * np.exp(-0.5 * v3**2 / var_V3)
    # print("Sanity Check:", np.sum(pdf_v3 * dV3)) # Should be close to 1
    # return
    prod = np.sum(pdf_v3 * v3**2 * dV3)

    heat_flux_T_2 *= prod
    heat_flux_N_2 *= prod

    heat_flux_T = heat_flux_T_1 + heat_flux_T_2
    heat_flux_N = heat_flux_N_1 + heat_flux_N_2

    print(f"heat_flux_T_1: {heat_flux_T_1:.4e} W/m², heat_flux_T_2: {heat_flux_T_2:.4e} W/m²")
    print(f"heat_flux_N_1: {heat_flux_N_1:.4e} W/m², heat_flux_N_2: {heat_flux_N_2:.4e} W/m²")
    return heat_flux_T, heat_flux_N

def heat_flux_from_3D_distribution(data, n_i, m_spec=6.36e-26):
    """Calculate the heat flux directly from the 3D distribution data."""
    Tcenters = data['Tdata']
    Ncenters = data['Ndata']
    Zcenters = data['Zdata']
    pmf = data['joint_pmf']
    pmf /= np.sum(np.abs(pmf)) # Ensure it's normalized
    pdf = pmf / ((Tcenters[2, 1, 1] - Tcenters[1, 1, 1]) * (Ncenters[1, 2, 1] - Ncenters[1, 1, 1]) * (Zcenters[1, 1, 2] - Zcenters[1, 1, 1]))

    dT = Tcenters[2, 1, 1] - Tcenters[1, 1, 1]
    dN = Ncenters[1, 2, 1] - Ncenters[1, 1, 1]
    dZ = Zcenters[1, 1, 2] - Zcenters[1, 1, 1]

    kinetic_energy = 0.5 * m_spec * (Tcenters**2 + Ncenters**2 + Zcenters**2)
    heat_flux_T = np.sum(n_i * pdf * kinetic_energy * Tcenters * dT * dN * dZ)
    heat_flux_N = np.sum(n_i * pdf * kinetic_energy * Ncenters * dT * dN * dZ)
    heat_flux_Z = np.sum(n_i * pdf * kinetic_energy * Zcenters * dT * dN * dZ)

    return heat_flux_T, heat_flux_N, heat_flux_Z

def heat_flux_from_2D_distribution(data, n_i, m_spec=6.36e-26):
    """Calculate the heat flux directly from the 2D distribution data."""
    Tcenters = data['Tdata']
    Ncenters = data['Ndata']
    pmf = data['joint_pmf']
    pmf /= np.sum(np.abs(pmf)) # Ensure it's normalized
    pdf = pmf / ((Tcenters[1, 2] - Tcenters[1, 1]) * (Ncenters[2, 1] - Ncenters[1, 1]))

    dT = Tcenters[1, 2] - Tcenters[1, 1]
    dN = Ncenters[2, 1] - Ncenters[1, 1]

    kinetic_energy = 0.5 * m_spec * (Tcenters**2 + Ncenters**2)
    heat_flux_T_1 = np.sum(n_i * pdf * kinetic_energy * Tcenters * dT * dN)
    heat_flux_N_1 = np.sum(n_i * pdf * kinetic_energy * Ncenters * dT * dN)

    # Calculate incoming / outgoing contributions
    # incoming = Ncenters < 0
    # outgoing = Ncenters > 0

    # hfN1_in = np.sum(n_i * pdf[incoming] * kinetic_energy[incoming] * Ncenters[incoming] * dT * dN)
    # hfN1_out = np.sum(n_i * pdf[outgoing] * kinetic_energy[outgoing] * Ncenters[outgoing] * dT * dN)
    # print(f"hfN1_in: {hfN1_in:.4e} W/m², hfN1_out: {hfN1_out:.4e} W/m²")

    heat_flux_T_2 = np.sum(0.5 * m_spec * n_i * Tcenters * pdf * dT * dN)
    heat_flux_N_2 = np.sum(0.5 * m_spec * n_i * Ncenters * pdf * dT * dN)

    v3 = np.linspace(-5000, 5000, 501)
    dV3 = v3[1] - v3[0]
    var_T = np.sum((Tcenters - np.sum(Tcenters*pdf))**2 * pmf)
    var_N = np.sum((Ncenters - np.sum(Ncenters*pdf))**2 * pmf)
    var_V3 = (var_T + var_N) / 2 # Approximate 3D variance as average of T and N variances
    pdf_v3 = 1 / (np.sqrt(2 * np.pi * var_V3)) * np.exp(-0.5 * v3**2 / var_V3)
    print("Sanity Check:", np.sum(pdf_v3 * dV3)) # Should be close to 1
    prod = np.sum(pdf_v3 * v3**2 * dV3) # <v3^2> for Maxwellian in 3rd dimension

    heat_flux_T_2 *= prod
    heat_flux_N_2 *= prod

    heat_flux_T = heat_flux_T_1 + heat_flux_T_2
    heat_flux_N = heat_flux_N_1 + heat_flux_N_2
    return heat_flux_T, heat_flux_N

def energy_flux_from_3D_hermite_fit(fit, n_i, m_spec=6.36e-26):
    """Compute energy flux vector from 3D Hermite PDF fit."""

    terms = fit['terms']
    mean_T = fit['mean_T']
    mean_N = fit['mean_N']
    mean_Z = fit['mean_Z']
    var_T = fit['var_T']
    var_N = fit['var_N']
    var_Z = fit['var_Z']

    # Velocity grid
    vel_range = 8000  # increase for better tail capture
    res = 121

    T, N, Z = np.meshgrid(
        np.linspace(-vel_range, vel_range, res),
        np.linspace(-vel_range, vel_range, res),
        np.linspace(-vel_range, vel_range, res),
        indexing='ij'
    )

    # Normalize coordinates
    T_shifted = (T - mean_T) / np.sqrt(var_T)
    N_shifted = (N - mean_N) / np.sqrt(var_N)
    Z_shifted = (Z - mean_Z) / np.sqrt(var_Z)

    # PDF in normalized space
    normalized_pdf = pdf_hermite_3d(T_shifted, N_shifted, Z_shifted, terms, alpha=1)

    # Transform back to physical PDF
    jacobian = np.sqrt(var_T * var_N * var_Z)
    pdf = normalized_pdf / jacobian

    # Volume element
    dT = T[1,0,0] - T[0,0,0]
    dN = N[0,1,0] - N[0,0,0]
    dZ = Z[0,0,1] - Z[0,0,0]
    dV = dT * dN * dZ

    # Kinetic energy
    v2 = T**2 + N**2 + Z**2
    kinetic_energy = 0.5 * m_spec * v2

    # Energy flux components
    F_T = np.sum(n_i * pdf * kinetic_energy * T * dV)
    F_N = np.sum(n_i * pdf * kinetic_energy * N * dV)
    F_Z = np.sum(n_i * pdf * kinetic_energy * Z * dV)

    return F_T, F_N, F_Z

def energy_flux_from_3D_distribution(data, n_i, m_spec=6.36e-26):
    """Compute energy flux vector directly from discrete velocity distribution."""

    T = data['Tdata']
    N = data['Ndata']
    Z = data['Zdata']
    pmf = data['joint_pmf']

    # Grid spacing
    dT = T[1,0,0] - T[0,0,0]
    dN = N[0,1,0] - N[0,0,0]
    dZ = Z[0,0,1] - Z[0,0,0]
    dV = dT * dN * dZ

    # Proper normalization: integral of PDF = 1
    normalization = np.sum(pmf) * dV
    pdf = pmf / normalization

    # Kinetic energy
    v2 = T**2 + N**2 + Z**2
    kinetic_energy = 0.5 * m_spec * v2

    # Energy flux components
    F_T = np.sum(n_i * pdf * kinetic_energy * T * dV)
    F_N = np.sum(n_i * pdf * kinetic_energy * N * dV)
    F_Z = np.sum(n_i * pdf * kinetic_energy * Z * dV)

    return F_T, F_N, F_Z

def Q_n_p(n, p, mean, std):
    ''' Evaluate the pth moment of the nth order Hermite polynomial times a normal distribution with given mean and std. 
    
    This is the integral of v^p * H_n((v - mean) / std) * (1/(sqrt(2*pi))) * exp(-0.5 * ((v - mean)/std)^2) dv from -inf to inf.
    '''

    if n > p:
        return 0.0
    else:
        output = 0.0
        for j in range(0, (p-n) + 1, 2):
            if j == 0:
                output += comb(p-n,j) * std**j * mean**(p-n-j)
            else:
                output += comb(p-n,j) * std**j * mean**(p-n-j) * factorial2(j-1, exact=True)

        output *= std**(n+1) * (-1)**(2*n) * factorial(p) / factorial(p-n)
        return output

def Q_n_p_half(n, p, mean, std):
    ''' Evaluate the pth moment of the nth order Hermite polynomial times a normal distribution with given mean and std. 
    
    This is the integral of v^p * H_n((v - mean) / std) * (1/(sqrt(2*pi))) * exp(-0.5 * ((v - mean)/std)^2) dv from -inf to 0.
    '''

    if n > p:
        return std**(p+1) * (-1)**(2*p+1)/np.sqrt(2*np.pi) * factorial(p) * hermite_polynomial(n-p-1,-mean/std, alpha=1) * np.exp(-0.5 * (mean/std)**2)
    else:
        output = 0.0
        for j in range(0, (p-n) + 1):
            if j == 0: # Special treatment for j=0
                output += 0.5*(-1)**(2*n)*std**(n+1)*factorial(p) / factorial(p-n)*comb(p-n,j) * std**j * mean**(p-n-j)
            elif j % 2 == 1: # Odd j terms
                output += -1/np.sqrt(2*np.pi) * (-1)**(2*n) * std**(n+1)* factorial(p) / factorial(p-n) *comb(p-n,j) * std**j * mean**(p-n-j) * factorial2(j-1, exact=True)
            else: # Even j terms
                output += 0.5*(-1)**(2*n)*std**(n+1)*factorial(p) / factorial(p-n)*comb(p-n,j) * std**j * mean**(p-n-j)*factorial2(j-1, exact=True)
        return output

def fast_momentum_flux_from_3D_hermite_fit(i, j, fit, n_i, m_spec = 6.36e-26):
    """Calculate the (i,j)th momentum flux moment from the 3D Hermite fit coefficients."""
    terms = fit['terms']
    mean_T = fit['mean_T']
    mean_N = fit['mean_N']
    mean_Z = fit['mean_Z']
    var_T = fit['var_T']
    std_T = np.sqrt(var_T)
    var_N = fit['var_N']
    std_N = np.sqrt(var_N)
    var_Z = fit['var_Z']
    std_Z = np.sqrt(var_Z)

    n_terms = terms.shape[0]
    output = 0.0

    if i == 0 and j == 0:
        for k in range(n_terms):
            for l in range(n_terms):
                for m in range(n_terms):
                    Q_T = Q_n_p(k, 2, mean_T, std_T)
                    Q_N = Q_n_p(l, 0, mean_N, std_N)
                    Q_Z = Q_n_p(m, 0, mean_Z, std_Z)
                    output += terms[k, l, m] * Q_T * Q_N * Q_Z
    elif i == 1 and j == 1:
        for k in range(n_terms):
            for l in range(n_terms):
                for m in range(n_terms):
                    Q_T = Q_n_p(k, 0, mean_T, std_T)
                    Q_N = Q_n_p(l, 2, mean_N, std_N)
                    Q_Z = Q_n_p(m, 0, mean_Z, std_Z)
                    output += terms[k, l, m] * Q_T * Q_N * Q_Z
    else:
        for k in range(n_terms):
            for l in range(n_terms):
                for m in range(n_terms):
                    Q_T = Q_n_p(k, 1, mean_T, std_T)
                    Q_N = Q_n_p_half(l, 1, mean_N, std_N)
                    Q_Z = Q_n_p(m, 0, mean_Z, std_Z)
                    output += terms[k, l, m] * Q_T * Q_N * Q_Z

    return m_spec * n_i / (std_T * std_N * std_Z) * output

def fast_heat_flux_from_3D_hermite_fit(fit, n_i, m_spec = 6.36e-26):
    """Calculate the normal kinetic energy flux moment from the 3D Hermite fit coefficients."""
    terms = fit['terms']
    mean_T = fit['mean_T']
    mean_N = fit['mean_N']
    mean_Z = fit['mean_Z']
    var_T = fit['var_T']
    std_T = np.sqrt(var_T)
    var_N = fit['var_N']
    std_N = np.sqrt(var_N)
    var_Z = fit['var_Z']
    std_Z = np.sqrt(var_Z)

    n_terms = terms.shape[0]
    output = 0.0
    for k in range(n_terms):
        for l in range(n_terms):
            for m in range(n_terms):
                # Term 1
                QT_1 = Q_n_p(k, 2, mean_T, std_T)
                QN_1 = Q_n_p(l, 1, mean_N, std_N)
                QZ_1 = Q_n_p(m, 0, mean_Z, std_Z)
                output += terms[k, l, m] * QT_1*QN_1*QZ_1

                # Term 2
                QT_2 = Q_n_p(k, 0, mean_T, std_T)
                QN_2 = Q_n_p(l, 3, mean_N, std_N)
                QZ_2 = Q_n_p(m, 0, mean_Z, std_Z)
                output += terms[k, l, m] * QT_2*QN_2*QZ_2

                # Term 3
                QT_3 = Q_n_p(k, 0, mean_T, std_T)
                QN_3 = Q_n_p(l, 1, mean_N, std_N)
                QZ_3 = Q_n_p(m, 2, mean_Z, std_Z)
                output += terms[k, l, m] * QT_3*QN_3*QZ_3

    output *= 0.5 * m_spec * n_i / (std_T * std_N * std_Z)
    return output
