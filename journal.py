import numpy as np
from flask import Flask, request, jsonify
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from typing import Tuple

def compute_ke_ce(eps: float, fc: float = 1.0, Omega: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    eps = float(eps)
    if not (0.0 < abs(eps) < 1.0):
        raise ValueError("eps must satisfy 0 < |eps| < 1")

    pi = np.pi
    s = np.sqrt(1.0 - eps**2)

    denom_common = (pi**2 * (1.0 - eps**2) + 16.0 * eps**2)
    h0 = 1.0 / (denom_common ** 1.5)

    a_uu = h0 * 4.0 * (pi**2 * (2.0 - eps**2) + 16.0 * eps**2)
    a_uv = h0 * pi * (pi**2 * (1.0 - eps**2)**2 - 16.0 * eps**4) / (eps * s)
    a_vu = -h0 * pi * (
        pi**2 * (1.0 - eps**2) * (1.0 + 2.0 * eps**2)
        + 32.0 * eps**2 * (1.0 + eps**2)
    ) / (eps * s)
    a_vv = h0 * 4.0 * (
        pi**2 * (1.0 + 2.0 * eps**2)
        + 32.0 * eps**2 * (1.0 + eps**2) / (1.0 - eps**2)
    )

    b_uu = h0 * 2.0 * pi * s * \
        (pi**2 * (1.0 + 2.0 * eps**2) - 16.0 * eps**2) / eps
    b_uv = -h0 * 8.0 * (pi**2 * (1.0 + 2.0 * eps**2) - 16.0 * eps**2)
    b_vu = b_uv
    b_vv = h0 * 2.0 * pi * (pi**2 * (1.0 - eps**2) **
                            2 + 48.0 * eps**2) / (eps * s)

    Ke = fc * np.array([[a_uu, a_uv], [a_vu, a_vv]], dtype=float)
    Ce = fc / Omega * np.array([[b_uu, b_uv], [b_vu, b_vv]], dtype=float)

    return Ke, Ce


def solve_quartic(Ss: float) -> float:
    pi = np.pi
    coeffs = [
        1,
        -4,
        6 - Ss**2 * (16 - pi**2),
        -(4 + pi**2 * Ss**2),
        1
    ]
    roots = np.roots(coeffs)
    eps_squared = next(
        root.real for root in roots
        if abs(root.imag) < 1e-10 and 0 < root.real < 1
    )
    return eps_squared


def compute_eps(
    diameter: float,
    length: float,
    load: float,
    clearance: float,
    viscosity: float,
    speed_rpm: float
) -> Tuple[float, float, float]:

    omega = speed_rpm * 2 * np.pi / 60.0
    Ss = (diameter * omega * viscosity * length**3) / \
        (8.0 * load * clearance**2)
    S = Ss / np.pi * ((diameter / length) ** 2)

    eps_squared = solve_quartic(Ss)
    eps = np.sqrt(eps_squared)

    return eps, Ss, S


def solve_journal_bearing_pressure(
    viscosity: float,
    speed_rad: float,
    diameter: float,
    length: float,
    clearance: float,
    eps: float,
    Ntheta: int = 80,
    Nz: int = 40
):
    mu = viscosity
    omega = -speed_rad
    R = diameter / 2.0
    L = length
    c = clearance

    theta = np.linspace(0.0, 2.0 * np.pi, Ntheta)
    z = np.linspace(-L/2.0, L/2.0, Nz)
    dtheta = theta[1] - theta[0]
    dz = z[1] - z[0]

    Theta, Z = np.meshgrid(theta, z, indexing="ij")

    h = c * (1.0 + eps * np.cos(Theta))

    Kth = h**3
    Kzz = h**3
    dh_dtheta = -c * eps * np.sin(Theta)
    S = 6.0 * mu * R * omega * dh_dtheta

    def idx(i, j):
        return i * (Nz - 2) + (j - 1)

    N_unknowns = Ntheta * (Nz - 2)
    A = lil_matrix((N_unknowns, N_unknowns))
    b = np.zeros(N_unknowns)

    Kth_iphalf = np.zeros((Ntheta, Nz))
    for i in range(Ntheta):
        ip = (i + 1) % Ntheta
        Kth_iphalf[i, :] = 0.5 * (Kth[i, :] + Kth[ip, :])

    Kzz_jphalf = 0.5 * (Kzz[:, 1:] + Kzz[:, :-1])

    for i in range(Ntheta):
        for j in range(1, Nz - 1):
            row = idx(i, j)

            ip = (i + 1) % Ntheta
            im = (i - 1) % Ntheta

            Kth_e = Kth_iphalf[i, j]
            Kth_w = Kth_iphalf[im, j]
            Kzz_n = Kzz_jphalf[i, j]
            Kzz_s = Kzz_jphalf[i, j - 1]

            diag = (Kth_e + Kth_w) / (R**2 * dtheta**2) + \
                (Kzz_n + Kzz_s) / dz**2
            A[row, row] = diag

            A[row, idx(ip, j)] = -Kth_e / (R**2 * dtheta**2)
            A[row, idx(im, j)] = -Kth_w / (R**2 * dtheta**2)

            if j + 1 <= Nz - 2:
                A[row, idx(i, j + 1)] = -Kzz_n / dz**2
            if j - 1 >= 1:
                A[row, idx(i, j - 1)] = -Kzz_s / dz**2

            b[row] = S[i, j]

    p_inner = spsolve(A.tocsr(), b)

    p = np.zeros((Ntheta, Nz))
    for i in range(Ntheta):
        for j in range(1, Nz - 1):
            p[i, j] = p_inner[idx(i, j)]

    return p, Theta, Z, h


def FrFt_bar(Lambda_star, L_over_D=1.0):
    """
    Dimensionless force components (small-ε static solution) at Λ*.
    F̄r + i F̄t = (i Λ*)/(1 + i Λ*) * [1 - tanh(η L/D)/(η L/D)] * (π/2),
    with η = sqrt(1 + i Λ*).
    """
    i = 1j
    eta = np.sqrt(1.0 + i * Lambda_star)
    H = 1.0 - np.tanh(eta * L_over_D) / (eta * L_over_D)
    G = (i * Lambda_star) / (1.0 + i * Lambda_star) * H * (np.pi / 2.0)
    return np.real(G), np.imag(G)  # F̄r, F̄t


def damping_Cxx_Cxy(Lambda, sigma_vals, L_over_D=1.0):
    """
    Dynamic damping coefficients (dimensionless) for given Λ over σ:
    CXX = -[F̄t(Λ-σ) - F̄t(Λ+σ)]/(2σ)
    CXY = +[F̄r(Λ-σ) - F̄r(Λ+σ)]/(2σ)
    """
    Cxx = np.zeros_like(sigma_vals, dtype=float)
    Cxy = np.zeros_like(sigma_vals, dtype=float)
    for k, s in enumerate(sigma_vals):
        s_eff = s if s != 0 else 1e-12
        Fr_m, Ft_m = FrFt_bar(Lambda - s_eff, L_over_D)
        Fr_p, Ft_p = FrFt_bar(Lambda + s_eff, L_over_D)
        Cxx[k] = -(Ft_m - Ft_p) / (2.0 * s_eff)
        Cxy[k] = +(Fr_m - Fr_p) / (2.0 * s_eff)
    return Cxx, Cxy


def FrFt_bar(Lambda_star, L_over_D=1.0):
    """
    Dimensionless force components for the small-ε static solution at Λ*.
    F̄r + i F̄t = (i Λ*)/(1 + i Λ*) * [1 - tanh(η L/D)/(η L/D)] * (π/2),
    where η = sqrt(1 + i Λ*).
    Returns (F̄r, F̄t).
    """
    i = 1j
    eta = np.sqrt(1.0 + i * Lambda_star)
    H = 1.0 - np.tanh(eta * L_over_D) / (eta * L_over_D)
    G = (i * Lambda_star) / (1.0 + i * Lambda_star) * H * (np.pi / 2.0)
    return np.real(G), np.imag(G)  # F̄r, F̄t

# ---- Dynamic stiffness coefficients from section 9.3.2 (dimensionless) ----


def stiffness_Kxx_Kxy(Lambda, sigma_vals, L_over_D=1.0):
    """
    Compute dynamic stiffness coefficients Kxx and Kxy.
    """
    Kxx = np.zeros_like(sigma_vals, dtype=float)
    Kxy = np.zeros_like(sigma_vals, dtype=float)
    for k, s in enumerate(sigma_vals):
        Fr_m, Ft_m = FrFt_bar(Lambda - s, L_over_D)
        Fr_p, Ft_p = FrFt_bar(Lambda + s, L_over_D)
        Kxx[k] = 0.5 * (Fr_m + Fr_p)
        Kxy[k] = -0.5 * (Ft_m + Ft_p)
    return Kxx, Kxy


def compute_squeeze_number(mu, nu, pa, R, c):
    """
    Compute the squeeze number (σ) for hydrodynamic lubrication.

    Args:
        mu (float): Dynamic viscosity of the lubricant (Pa·s).
        nu (float): Normal velocity (m/s).
        pa (float): Ambient pressure (Pa).
        R (float): Radius of the journal (m).
        c (float): Radial clearance (m).

    Returns:
        float: The computed value of σ.
    """
    return (12 * mu * nu * (R / c)**2) / pa


def compute_lambda(mu, omega, pa, R, c):
    """
    Compute the dimensionless parameter Lambda (Λ) for hydrodynamic lubrication.

    Args:
        mu (float): Dynamic viscosity of the lubricant (Pa·s).
        omega (float): Angular velocity of the journal (rad/s).
        pa (float): Ambient pressure (Pa).
        R (float): Radius of the journal (m).
        c (float): Radial clearance (m).

    Returns:
        float: The computed value of Λ.
    """
    return (6 * mu * omega * (R / c)**2) / pa

def compute_eps(diameter: float, length: float, load: float, clearance: float,
                viscosity: float, speed_rpm: float) -> Tuple[float, float, float]:

    # Convert speed to rad/s
    omega = speed_rpm * 2 * np.pi / 60  # rad/s

    # Calculate modified Sommerfeld number (Ss)
    Ss = (diameter * omega * viscosity * length**3) / (8 * load * clearance**2)

    # Calculate Sommerfeld number (S)
    S = Ss / np.pi * ((diameter / length)**2)

    # Solve quartic equation for eccentricity squared (eps^2)
    # Quartic equation coefficients from the problem statement
    eps_squared = solve_quartic(Ss)
    eps = np.sqrt(eps_squared)
    
    return eps, Ss, S