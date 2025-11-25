import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# ---- Solver for steady compressible Reynolds equation (gas journal bearing) ----
def gas_journal_bearing(
    R=0.05,        # journal radius [m]
    L=0.05,        # bearing length [m]
    c=50e-6,       # radial clearance [m]
    mu=1.8e-5,     # gas viscosity [Pa*s]
    pa=101325.0,   # ambient pressure [Pa]
    omega=3000.0,  # rad/s
    eps=0.8,       # eccentricity ratio e/c in (0,1)
    phi=0.0,       # attitude angle [rad] (direction of displacement)
    ntheta=181,    # circumferential nodes
    nz=61,         # axial nodes
    bc_axial='dirichlet', # 'dirichlet' (Π=1 at ends) or 'neumann' (∂Π/∂Z=0)
    upwind=False,         # if True, use first-order upwind on Λ-term
    return_full=True
):
    """
    Solve the steady compressible Reynolds equation for a gas journal bearing.

    Returns:
        theta (ntheta,), Z (nz,), Pi (nz, ntheta) normalized pressure,
        p (nz, ntheta) dimensional pressure,
        results dict with load components and parameters
    """
    # Derived parameters
    U = omega * R              # surface speed
    Lam = 6.0 * mu * U * R / (pa * c**2)  # compressibility number Λ
    alpha = (R / L)**2         # axial-to-circumferential scaling

    # Grids
    theta = np.linspace(0.0, 2*np.pi, ntheta, endpoint=False)
    Z = np.linspace(-0.5, 0.5, nz)   # nondimensional axial coordinate
    dth = theta[1] - theta[0]
    dZ = Z[1] - Z[0] if nz > 1 else 1.0

    # Film thickness and its powers
    H = 1.0 + eps * np.cos(theta - phi)           # H(θ)
    H3 = H**3

    # Precompute face-averaged diffusivities in θ
    def avg(a, b): return 0.5*(a+b)
    H3_e = avg(H3, np.roll(H3, -1))  # at i+1/2
    H3_w = avg(H3, np.roll(H3, +1))  # at i-1/2

    # Build sparse matrix A and RHS b for unknowns ordered as (j,i) row-major
    N = ntheta * nz
    A = lil_matrix((N, N))
    b = np.zeros(N)

    def idx(i, j):
        # periodic in theta
        i = i % ntheta
        return j * ntheta + i

    # Assemble
    for j in range(nz):
        z_is_edge = (j == 0) or (j == nz-1)
        for i in range(ntheta):
            row = idx(i, j)

            if bc_axial == 'dirichlet' and z_is_edge:
                # Π = 1 at axial edges
                A[row, row] = 1.0
                b[row] = 1.0
                continue

            # Coefficients
            aE = H3_e[i] / (dth**2)
            aW = H3_w[i] / (dth**2)
            aN = alpha * H3[i] / (dZ**2)
            aS = alpha * H3[i] / (dZ**2)

            # Convection-like term for Λ ∂(Π H)/∂θ
            if upwind:
                # Simple first-order upwind in θ-direction
                cE = 0.0
                cW = Lam * H[i] / dth
                cP = -Lam * H[i] / dth
            else:
                # Central difference: ∂(Π H)/∂θ ≈ (Π_{i+1} H_{i+1} - Π_{i-1} H_{i-1}) / (2 dth)
                cE = - Lam * H[(i+1) % ntheta] / (2.0 * dth)
                cW = + Lam * H[(i-1) % ntheta] / (2.0 * dth)
                cP = 0.0

            aP = aE + aW + aN + aS - cP

            # Fill matrix
            A[row, row] = aP

            # θ-neighbors (periodic)
            A[row, idx(i+1, j)] += -(aE + cE)
            A[row, idx(i-1, j)] += -(aW + cW)

            # Axial neighbors
            if j+1 < nz:
                if bc_axial == 'dirichlet' and j+1 == nz-1:
                    b[row] += aN * 1.0
                else:
                    A[row, idx(i, j+1)] += -aN
            if j-1 >= 0:
                if bc_axial == 'dirichlet' and j-1 == 0:
                    b[row] += aS * 1.0
                else:
                    A[row, idx(i, j-1)] += -aS

            # Neumann axial BCs: ∂Π/∂Z = 0 at edges
            if bc_axial == 'neumann' and z_is_edge:
                if j == nz-1:
                    A[row, row] += aN
                if j == 0:
                    A[row, row] += aS

    # Solve
    A = csr_matrix(A)
    Pi_vec = spsolve(A, b)
    Pi = Pi_vec.reshape((nz, ntheta))
    p = pa * Pi

    # Load integration
    dp = p - pa
    dA = R * L * dth * dZ
    cos_th = np.cos(theta)[None, :]
    sin_th = np.sin(theta)[None, :]
    Wx = np.sum(dp * cos_th) * dA
    Wy = np.sum(dp * sin_th) * dA
    W = np.hypot(Wx, Wy)
    beta = np.arctan2(Wy, Wx)

    results = dict(R=R, L=L, c=c, mu=mu, pa=pa, omega=omega, U=U,
                   eps=eps, phi=phi, Lam=Lam, alpha=alpha,
                   Wx=Wx, Wy=Wy, W=W, beta=beta)
    if return_full:
        return theta, Z, Pi, p, results
    else:
        return Pi, results

# ---- Utilities: convert between (eps, phi) and x,y (dimensional) displacements ----
def epsphi_from_xy(x, y, c):
    # ex = x/c, ey = y/c
    ex = x / c
    ey = y / c
    eps = np.hypot(ex, ey)
    phi = np.arctan2(ey, ex) if eps > 0 else 0.0
    return eps, phi

def xy_from_epsphi(eps, phi, c):
    ex = eps * np.cos(phi)
    ey = eps * np.sin(phi)
    x = ex * c
    y = ey * c
    return x, y

# ---- Small-signal stiffness (finite differences around an operating point) ----
def stiffness_matrix(
    base_params,
    x0, y0,
    delta_disp=5e-8,   # small physical displacement [m]
    scheme='central'   # 'central' or 'forward'
):
    """
    Compute stiffness matrix K = -∂W/∂x,y [N/m] around (x0, y0).
    Returns Kxx, Kxy, Kyx, Kyy and a dict with detailed results.
    """
    c = base_params['c']

    def solve_at_xy(x, y):
        eps, phi = epsphi_from_xy(x, y, c)
        th, Z, Pi, p, res = gas_journal_bearing(
            R=base_params['R'], L=base_params['L'], c=base_params['c'],
            mu=base_params['mu'], pa=base_params['pa'], omega=base_params['omega'],
            eps=eps, phi=phi, ntheta=base_params['ntheta'], nz=base_params['nz'],
            bc_axial=base_params['bc_axial'], upwind=base_params['upwind'],
            return_full=True
        )
        return res['Wx'], res['Wy']

    if scheme == 'central':
        # x-direction derivatives
        Wx_p, Wy_p = solve_at_xy(x0 + delta_disp, y0)
        Wx_m, Wy_m = solve_at_xy(x0 - delta_disp, y0)
        dWx_dx = (Wx_p - Wx_m) / (2*delta_disp)
        dWy_dx = (Wy_p - Wy_m) / (2*delta_disp)

        # y-direction derivatives
        Wx_p2, Wy_p2 = solve_at_xy(x0, y0 + delta_disp)
        Wx_m2, Wy_m2 = solve_at_xy(x0, y0 - delta_disp)
        dWx_dy = (Wx_p2 - Wx_m2) / (2*delta_disp)
        dWy_dy = (Wy_p2 - Wy_m2) / (2*delta_disp)
    else:  # forward
        Wx0, Wy0 = solve_at_xy(x0, y0)
        Wx_p, Wy_p = solve_at_xy(x0 + delta_disp, y0)
        Wx_p2, Wy_p2 = solve_at_xy(x0, y0 + delta_disp)
        dWx_dx = (Wx_p - Wx0) / delta_disp
        dWy_dx = (Wy_p - Wy0) / delta_disp
        dWx_dy = (Wx_p2 - Wx0) / delta_disp
        dWy_dy = (Wy_p2 - Wy0) / delta_disp

    # Stiffness is bearing reaction to displacement: K = -∂W/∂q
    Kxx = -dWx_dx
    Kxy = -dWx_dy
    Kyx = -dWy_dx
    Kyy = -dWy_dy

    return Kxx, Kxy, Kyx, Kyy, dict(
        dWx_dx=dWx_dx, dWx_dy=dWx_dy, dWy_dx=dWy_dx, dWy_dy=dWy_dy
    )

# ---- Plotting utility for pressure ----
def plot_pressure(theta, Z, p, pa=101325.0, title="Gas journal bearing pressure"):
    theta_deg = theta * 180.0 / np.pi
    extent = [theta_deg.min(), theta_deg.max(), Z.min(), Z.max()]
    j_mid = np.argmin(np.abs(Z - 0.0))
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])

    # Left: colormap
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(p*1e-3, extent=extent, aspect='auto', origin='lower', cmap='viridis')
    cbar = plt.colorbar(im, ax=ax0)
    cbar.set_label('Pressure p [kPa]')
    ax0.set_title(title)
    ax0.set_xlabel('θ [deg]')
    ax0.set_ylabel('Axial Z [-]')
    ax0.axhline(0.0, color='w', ls='--', lw=0.8)

    # Right: mid-axial profile
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(theta_deg, p[j_mid, :]*1e-3, label='p(θ, Z=0)')
    ax1.hlines(pa*1e-3, theta_deg.min(), theta_deg.max(), colors='k', linestyles=':', label='ambient p_a')
    ax1.set_xlabel('θ [deg]')
    ax1.set_ylabel('Pressure [kPa]')
    ax1.set_title('Mid-axial profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Optional: nice plotting style if you have scienceplots installed
try:
    import scienceplots
    plt.style.use(['science'])
except Exception:
    pass

def sweep_speed_and_plot_stiffness(
    R=0.05, L=0.05, c=40e-6, mu=1.85e-5, pa=101325.0,
    eps0=0.7, phi0=0.0,
    ntheta=241, nz=81,
    bc_axial='dirichlet', upwind=False,
    omegas=np.linspace(500, 8000, 25),
    delta_disp=5e-8,  # m
    scheme='central'  # 'central' or 'forward'
):
    # Reference operating point in x,y (from eps,phi)
    x0, y0 = xy_from_epsphi(eps0, phi0, c)

    Kxxs, Kxys, Kyxs, Kyys = [], [], [], []
    Lambdas = []

    for omega in omegas:
        # compute Lambda for this omega (and ensure mesh is consistent)
        th, Z, Pi, p, res = gas_journal_bearing(
            R=R, L=L, c=c, mu=mu, pa=pa, omega=omega,
            eps=eps0, phi=phi0, ntheta=ntheta, nz=nz,
            bc_axial=bc_axial, upwind=upwind, return_full=True
        )
        Lam = res['Lam']
        Lambdas.append(Lam)

        base_params = dict(
            R=R, L=L, c=c, mu=mu, pa=pa, omega=omega,
            ntheta=ntheta, nz=nz, bc_axial=bc_axial, upwind=upwind
        )

        Kxx, Kxy, Kyx, Kyy, _ = stiffness_matrix(
            base_params, x0, y0, delta_disp=delta_disp, scheme=scheme
        )
        Kxxs.append(Kxx); Kxys.append(Kxy); Kyys.append(Kyy); Kyxs.append(Kyx)

        print(f"ω={omega:7.1f} rad/s | Λ={Lam:7.3f} | "
              f"Kxx={Kxx: .3e}  Kyy={Kyy: .3e}  Kxy={Kxy: .3e}  Kyx={Kyx: .3e}")

    Kxxs = np.array(Kxxs); Kxys = np.array(Kxys)
    Kyxs = np.array(Kyxs); Kyys = np.array(Kyys)
    Lambdas = np.array(Lambdas)

    # Plot: stiffness terms vs speed
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.0), sharex=False)
    plt.rcParams.update({'font.size': 16})  # Increase font size
    # Convert speed to rpm
    rpms = omegas * 60 / (2 * np.pi)

    # Direct stiffness
    axs[0].plot(rpms, Kxxs / 1e6, 'b-o', ms=4, label='Kxx')
    axs[0].plot(rpms, Kyys / 1e6, 'r-s', ms=4, label='Kyy')
    axs[0].set_ylabel('Direct stiffness [MN/m]')
    axs[0].set_xlabel('Speed [rpm]')
    # axs[0].set_title('Bearing stiffness vs rotation speed')
    axs[0].legend()

    # Cross-coupled stiffness
    axs[1].plot(rpms, Kxys / 1e6, 'g-^', ms=4, label='Kxy')
    axs[1].plot(rpms, Kyxs / 1e6, 'm-v', ms=4, label='Kyx')
    axs[1].set_xlabel('Speed [rpm]')
    axs[1].set_ylabel('Cross-coupled stiffness [MN/m]')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("gas_stiffness_rpm.png", dpi=300)
    plt.show()

# if __name__ == "__main__":
#     # Example: use your current solver settings
#     sweep_speed_and_plot_stiffness(
#         R=0.05, L=0.05, c=40e-6, mu=1.85e-5, pa=101325.0,
#         eps0=0.7, phi0=0.0, ntheta=241, nz=81,
#         bc_axial='dirichlet', upwind=False,
#         omegas=np.linspace(500, 8000, 21),
#         delta_disp=5e-8, scheme='central'
#     )

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def assemble_dynamic_operator(theta, Z, H0, Pi0, Lam, alpha, dth, dZ,
                              bc_axial='dirichlet', upwind=False, sigma=500.0):
    """
    Build complex sparse matrix A_dyn and RHS drivers for unit xdot and ydot:
        (L + i*sigma*H0) * Pi' = S,  where S = Pi0 * [(cosθ/c) xdot + (sinθ/c) ydot]
    Returns:
        A_dyn (csr, complex), Sx (vector), Sy (vector)
    """
    ntheta = theta.size
    nz = Z.size
    N = ntheta * nz

    # Precompute H^3 and face averages
    H3 = H0**3
    H3_e = 0.5 * (H3 + np.roll(H3, -1))
    H3_w = 0.5 * (H3 + np.roll(H3, +1))

    A = lil_matrix((N, N), dtype=np.complex128)
    Sx = np.zeros(N, dtype=np.complex128)
    Sy = np.zeros(N, dtype=np.complex128)

    def idx(i, j):
        i = i % ntheta
        return j * ntheta + i

    # Velocity-induced squeeze term coefficients for RHS:
    # ∂H'/∂t = (cosθ/c) xdot + (sinθ/c) ydot
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    for j in range(nz):
        z_is_edge = (j == 0) or (j == nz-1)
        for i in range(ntheta):
            row = idx(i, j)

            if bc_axial == 'dirichlet' and z_is_edge:
                # Dirichlet Π' = 0 at axial ends for perturbation
                A[row, row] = 1.0 + 0.0j
                # RHS is zero at boundaries for perturbations
                continue

            aE = H3_e[i] / (dth**2)
            aW = H3_w[i] / (dth**2)
            aN = alpha * H3[i] / (dZ**2)
            aS = alpha * H3[i] / (dZ**2)

            if upwind:
                cE = 0.0
                cW = Lam * H0[i] / dth
                cP = -Lam * H0[i] / dth
            else:
                cE = - Lam * H0[(i+1) % ntheta] / (2.0 * dth)
                cW = + Lam * H0[(i-1) % ntheta] / (2.0 * dth)
                cP = 0.0

            # Dynamic term: i*sigma * H0 * Π'
            aDyn = -1j * sigma * H0[i]

            aP = aE + aW + aN + aS - cP + aDyn

            # Assemble operator
            A[row, row] = aP
            A[row, idx(i+1, j)] += -(aE + cE)
            A[row, idx(i-1, j)] += -(aW + cW)

            if j+1 < nz:
                if bc_axial == 'dirichlet' and j+1 == nz-1:
                    # Next is boundary (Π'=0), contributes to RHS if needed (here zero)
                    pass
                else:
                    A[row, idx(i, j+1)] += -aN
            if j-1 >= 0:
                if bc_axial == 'dirichlet' and j-1 == 0:
                    pass
                else:
                    A[row, idx(i, j-1)] += -aS

            # RHS source from squeeze: S = Π0 * ∂H'/∂t
            Sx[row] = Pi0[j, i] * (cos_th[i])  # factor 1/c to be applied outside if desired
            Sy[row] = Pi0[j, i] * (sin_th[i])

    return csr_matrix(A), Sx, Sy

def damping_matrix_vs_speed(
    R=0.05, L=0.05, c=40e-6, mu=1.85e-5, pa=101325.0,
    eps0=0.7, phi0=0.0,
    ntheta=241, nz=81, bc_axial='dirichlet', upwind=False,
    omegas=np.linspace(500, 8000, 21),
    sigma_dyn=500.0  # perturbation frequency [1/s], heuristic
):
    """
    Sweep speed and compute damping matrix C via linearized dynamic Reynolds model.
    Returns arrays of Cxx, Cyy, Cxy, Cyx and Lambda values.
    """
    # Displacement reference (not directly needed here, but for consistency)
    # x0, y0 are the static eccentricity components
    x0 = eps0 * c * np.cos(phi0)
    y0 = eps0 * c * np.sin(phi0)

    Cxxs, Cyys, Cxys, Cyxs = [], [], [], []
    Lambdas = []

    for omega in omegas:
        # Steady-state solution at operating point
        theta, Z, Pi0, p0, res = gas_journal_bearing(
            R=R, L=L, c=c, mu=mu, pa=pa, omega=omega,
            eps=eps0, phi=phi0, ntheta=ntheta, nz=nz, bc_axial=bc_axial, upwind=upwind
        )
        Lam = res['Lam']; Lambdas.append(Lam)

        # Geometry and discretization
        dth = theta[1] - theta[0]
        dZ = Z[1] - Z[0] if nz > 1 else 1.0
        alpha = (R / L)**2

        # Film thickness H0(θ) using x0,y0 form: H = 1 + (x/c) cosθ + (y/c) sinθ
        H0 = 1.0 + (x0/c) * np.cos(theta) + (y0/c) * np.sin(theta)

        # Assemble dynamic operator and RHS for unit xdot and ydot
        A_dyn, Sx, Sy = assemble_dynamic_operator(
            theta, Z, H0, Pi0, Lam, alpha, dth, dZ,
            bc_axial=bc_axial, upwind=upwind, sigma=sigma_dyn
        )

        # Apply 1/c scaling for RHS (∂H'/∂t = (cosθ/c) xdot + (sinθ/c) ydot)
        Sx = (1.0/c) * Sx
        Sy = (1.0/c) * Sy

        # Solve for Π' response to unit velocities
        Pi_x = spsolve(A_dyn, Sx).reshape((nz, ntheta))  # response to xdot=1
        Pi_y = spsolve(A_dyn, Sy).reshape((nz, ntheta))  # response to ydot=1

        # Dimensional pressure perturbations
        p_x = pa * Pi_x
        p_y = pa * Pi_y

        # Integrate to get force perturbations
        dA = R * L * dth * dZ
        cos_th = np.cos(theta)[None, :]
        sin_th = np.sin(theta)[None, :]

        Wx_x = np.sum(p_x * cos_th) * dA  # force x due to xdot
        Wy_x = np.sum(p_x * sin_th) * dA  # force y due to xdot
        Wx_y = np.sum(p_y * cos_th) * dA  # force x due to ydot
        Wy_y = np.sum(p_y * sin_th) * dA  # force y due to ydot

        # Damping matrix: F' = -C v. Take real part as in-phase (damping) component.
        Cxx = np.real(Wx_x)   # per unit xdot
        Cyx = np.real(Wy_x)   # per unit xdot
        Cxy = np.real(Wx_y)   # per unit ydot
        Cyy = np.real(Wy_y)   # per unit ydot

        Cxxs.append(Cxx); Cyys.append(Cyy); Cxys.append(Cxy); Cyxs.append(Cyx)

        print(f"ω={omega:7.1f} rad/s | Λ={Lam:7.3f} | "
              f"Cxx={Cxx: .3e}  Cyy={Cyy: .3e}  Cxy={Cxy: .3e}  Cyx={Cyx: .3e}")

    return np.array(omegas), np.array(Lambdas), np.array(Cxxs), np.array(Cyys), np.array(Cxys), np.array(Cyxs)

def plot_damping_vs_speed(omegas, Cxx, Cyy, Cxy, Cyx):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.0), sharex=False)
    plt.rcParams.update({'font.size': 14})  # Set font size to 14

    # Direct damping
    axs[0].plot(omegas/2/np.pi*60, Cxx, 'b-o', ms=4, label='Cxx')
    axs[0].plot(omegas/2/np.pi*60, Cyy, 'r-s', ms=4, label='Cyy')
    axs[0].set_ylabel('Direct damping [N·s/m]')
    axs[0].set_xlabel(r'Speed [rpm]')
    # axs[0].set_title('Bearing damping vs rotation speed')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # Cross-coupled damping
    axs[1].plot(omegas/2/np.pi*60, Cxy, 'g-^', ms=4, label='Cxy')
    axs[1].plot(omegas/2/np.pi*60, Cyx, 'm-v', ms=4, label='Cyx')
    axs[1].set_xlabel(r'Speed [rpm]')
    axs[1].set_ylabel('Cross-coupled damping [N·s/m]')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("damping_gas.png",dpi=300)
    plt.show()

# if __name__ == "__main__":
#     # Base settings matching your solver
#     R=38e-3/2; L=20e-3; c=55e-6; mu=1.85e-5; pa=101325.0
#     eps0=0.98; phi0=0.0
#     ntheta=241; nz=81
#     bc_axial='dirichlet'; upwind=False

#     # Sweep speeds
#     omegas = np.linspace(500, 20e3, 21) *2*np.pi/60

#     # Choose a small perturbation frequency for the linearized dynamic term.
#     # This parameter influences the magnitude of C; you can try 250–2000 1/s.
#     sigma_dyn = 200

#     # Compute damping vs speed
#     omegas, Lambdas, Cxx, Cyy, Cxy, Cyx = damping_matrix_vs_speed(
#         R=R, L=L, c=c, mu=mu, pa=pa, eps0=eps0, phi0=phi0,
#         ntheta=ntheta, nz=nz, bc_axial=bc_axial, upwind=upwind,
#         omegas=omegas, sigma_dyn=sigma_dyn
#     )

#     # Plot
#     plot_damping_vs_speed(omegas, Cxx, Cyy, Cxy, Cyx)

# ---- Utilities: convert between (eps, phi) and x,y ----
def epsphi_from_xy(x, y, c):
    ex = x / c
    ey = y / c
    eps = np.hypot(ex, ey)
    phi = np.arctan2(ey, ex) if eps > 0 else 0.0
    return eps, phi

def xy_from_epsphi(eps, phi, c):
    ex = eps * np.cos(phi)
    ey = eps * np.sin(phi)
    x = ex * c
    y = ey * c
    return x, y

# ---- Stiffness via finite differences around operating point ----
def stiffness_matrix(
    base_params,
    x0, y0,
    delta_disp=5e-8,
    scheme='central'
):
    """
    Compute stiffness matrix K = -∂W/∂x,y [N/m] around (x0, y0).
    Returns Kxx, Kxy, Kyx, Kyy.
    """
    c = base_params['c']

    def solve_at_xy(x, y):
        eps, phi = epsphi_from_xy(x, y, c)
        th, Z, Pi, p, res = gas_journal_bearing(
            R=base_params['R'], L=base_params['L'], c=base_params['c'],
            mu=base_params['mu'], pa=base_params['pa'], omega=base_params['omega'],
            eps=eps, phi=phi, ntheta=base_params['ntheta'], nz=base_params['nz'],
            bc_axial=base_params['bc_axial'], upwind=base_params['upwind'],
            return_full=True
        )
        return res['Wx'], res['Wy']

    if scheme == 'central':
        Wx_p, Wy_p = solve_at_xy(x0 + delta_disp, y0)
        Wx_m, Wy_m = solve_at_xy(x0 - delta_disp, y0)
        dWx_dx = (Wx_p - Wx_m) / (2*delta_disp)
        dWy_dx = (Wy_p - Wy_m) / (2*delta_disp)

        Wx_p2, Wy_p2 = solve_at_xy(x0, y0 + delta_disp)
        Wx_m2, Wy_m2 = solve_at_xy(x0, y0 - delta_disp)
        dWx_dy = (Wx_p2 - Wx_m2) / (2*delta_disp)
        dWy_dy = (Wy_p2 - Wy_m2) / (2*delta_disp)
    else:
        Wx0, Wy0 = solve_at_xy(x0, y0)
        Wx_p, Wy_p = solve_at_xy(x0 + delta_disp, y0)
        Wx_p2, Wy_p2 = solve_at_xy(x0, y0 + delta_disp)
        dWx_dx = (Wx_p - Wx0) / delta_disp
        dWy_dx = (Wy_p - Wy0) / delta_disp
        dWx_dy = (Wx_p2 - Wx0) / delta_disp
        dWy_dy = (Wy_p2 - Wy0) / delta_disp

    Kxx = -dWx_dx
    Kxy = -dWx_dy
    Kyx = -dWy_dx
    Kyy = -dWy_dy
    return Kxx, Kxy, Kyx, Kyy

# ---- Dynamic operator for damping (corrected sign: -i σ H0) ----
def assemble_dynamic_operator(theta, Z, H0, Pi0, Lam, alpha, dth, dZ,
                              bc_axial='dirichlet', upwind=False, sigma=800.0):
    """
    Build complex sparse matrix A_dyn and RHS drivers for unit xdot and ydot:
        (L - i*sigma*H0) * Pi' = S,  where S = Pi0 * [(cosθ/c) xdot + (sinθ/c) ydot]
    Returns:
        A_dyn (csr, complex), Sx (vector), Sy (vector)
    """
    ntheta = theta.size
    nz = Z.size
    N = ntheta * nz

    H3 = H0**3
    H3_e = 0.5 * (H3 + np.roll(H3, -1))
    H3_w = 0.5 * (H3 + np.roll(H3, +1))

    A = lil_matrix((N, N), dtype=np.complex128)
    Sx = np.zeros(N, dtype=np.complex128)
    Sy = np.zeros(N, dtype=np.complex128)

    def idx(i, j):
        i = i % ntheta
        return j * ntheta + i

    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    for j in range(nz):
        z_is_edge = (j == 0) or (j == nz-1)
        for i in range(ntheta):
            row = idx(i, j)

            if bc_axial == 'dirichlet' and z_is_edge:
                A[row, row] = 1.0 + 0.0j  # Π' = 0 at ends
                continue

            aE = H3_e[i] / (dth**2)
            aW = H3_w[i] / (dth**2)
            aN = alpha * H3[i] / (dZ**2)
            aS = alpha * H3[i] / (dZ**2)

            if upwind:
                cE = 0.0
                cW = Lam * H0[i] / dth
                cP = -Lam * H0[i] / dth
            else:
                cE = - Lam * H0[(i+1) % ntheta] / (2.0 * dth)
                cW = + Lam * H0[(i-1) % ntheta] / (2.0 * dth)
                cP = 0.0

            aDyn = -1j * sigma * H0[i]  # corrected sign

            aP = aE + aW + aN + aS - cP + aDyn

            A[row, row] = aP
            A[row, idx(i+1, j)] += -(aE + cE)
            A[row, idx(i-1, j)] += -(aW + cW)

            if j+1 < nz:
                if bc_axial == 'dirichlet' and j+1 == nz-1:
                    pass
                else:
                    A[row, idx(i, j+1)] += -aN
            if j-1 >= 0:
                if bc_axial == 'dirichlet' and j-1 == 0:
                    pass
                else:
                    A[row, idx(i, j-1)] += -aS

            Sx[row] = Pi0[j, i] * (cos_th[i])
            Sy[row] = Pi0[j, i] * (sin_th[i])

    return csr_matrix(A), Sx, Sy

def damping_matrix_for_speed(theta, Z, Pi0, pa, Lam, R, L, c, x0, y0,
                             bc_axial, upwind, sigma_dyn):
    """
    Compute C matrix (Cxx, Cxy, Cyx, Cyy) for a given operating point.
    """
    dth = theta[1] - theta[0]
    dZ = Z[1] - Z[0] if Z.size > 1 else 1.0
    alpha = (R / L)**2
    H0 = 1.0 + (x0/c) * np.cos(theta) + (y0/c) * np.sin(theta)

    A_dyn, Sx, Sy = assemble_dynamic_operator(
        theta, Z, H0, Pi0, Lam, alpha, dth, dZ,
        bc_axial=bc_axial, upwind=upwind, sigma=sigma_dyn
    )

    Sx = (1.0/c) * Sx
    Sy = (1.0/c) * Sy

    Pi_x = spsolve(A_dyn, Sx).reshape((Z.size, theta.size))
    Pi_y = spsolve(A_dyn, Sy).reshape((Z.size, theta.size))

    p_x = pa * Pi_x
    p_y = pa * Pi_y

    dA = R * L * dth * dZ
    cos_th = np.cos(theta)[None, :]
    sin_th = np.sin(theta)[None, :]

    Wx_x = np.sum(p_x * cos_th) * dA
    Wy_x = np.sum(p_x * sin_th) * dA
    Wx_y = np.sum(p_y * cos_th) * dA
    Wy_y = np.sum(p_y * sin_th) * dA

    Cxx = -np.real(Wx_x)
    Cyx = -np.real(Wy_x)
    Cxy = -np.real(Wx_y)
    Cyy = -np.real(Wy_y)
    return Cxx, Cxy, Cyx, Cyy

# ---- Critical mass computation ----
def spectral_abscissa(K, C, m):
    """
    Return max real part of eigenvalues of the 4x4 state matrix for mass m.
    A = [[0, I], [-m^{-1}K, -m^{-1}C]]
    """
    Z2 = np.zeros((2,2))
    I2 = np.eye(2)
    A = np.block([
        [Z2, I2],
        [-K/m, -C/m]
    ])
    eigs = np.linalg.eigvals(A)
    return np.max(np.real(eigs))

def critical_mass(K, C, m_lo=1e-4, m_hi=50.0, tol=1e-3, max_iter=50):
    """
    Find the smallest m in [m_lo, m_hi] such that spectral_abscissa <= 0.
    Uses bisection on m. Returns (mcrit, info_dict).
    """
    f_lo = spectral_abscissa(K, C, m_lo)
    f_hi = spectral_abscissa(K, C, m_hi)

    if f_lo <= 0:
        return m_lo, {'status': 'stable_at_lower_bound', 'f_lo': f_lo, 'f_hi': f_hi}
    if f_hi > 0:
        return np.nan, {'status': 'unstable_up_to_upper_bound', 'f_lo': f_lo, 'f_hi': f_hi}

    a, b = m_lo, m_hi
    fa, fb = f_lo, f_hi
    for _ in range(max_iter):
        m_mid = 0.5*(a+b)
        f_mid = spectral_abscissa(K, C, m_mid)
        if abs(f_mid) < tol:
            return m_mid, {'status': 'converged', 'f_mid': f_mid}
        # We want the root where f crosses zero; f decreases with m typically
        if f_mid > 0:
            a, fa = m_mid, f_mid
        else:
            b, fb = m_mid, f_mid
    return 0.5*(a+b), {'status': 'max_iter', 'f_a': fa, 'f_b': fb}

# ---- Main: sweep speed, compute K, C, and mcrit(ω) ----
if __name__ == "__main__":
    # Base geometry and gas properties
    R = 0.05
    L = 0.05
    c = 40e-6
    mu = 1.85e-5
    pa = 101325.0

    # Operating eccentricity
    eps0 = 0.7
    phi0 = 0.0
    x0, y0 = xy_from_epsphi(eps0, phi0, c)

    # Numerics
    ntheta, nz = 241, 81
    bc_axial = 'dirichlet'
    upwind = False

    # Sweep speeds
    omegas = np.linspace(500, 8000, 21)
    sigma_dyn = 800.0  # motion frequency for damping linearization

    mcrits = []
    Lambdas = []

    for omega in omegas:
        # Steady solve (to compute K and Λ)
        theta, Z, Pi0, p0, res = gas_journal_bearing(
            R=R, L=L, c=c, mu=mu, pa=pa, omega=omega,
            eps=eps0, phi=phi0, ntheta=ntheta, nz=nz,
            bc_axial=bc_axial, upwind=upwind, return_full=True
        )
        Lam = res['Lam']
        Lambdas.append(Lam)

        base_params = dict(R=R, L=L, c=c, mu=mu, pa=pa, omega=omega,
                           ntheta=ntheta, nz=nz, bc_axial=bc_axial, upwind=upwind)

        # Stiffness
        Kxx, Kxy, Kyx, Kyy = stiffness_matrix(base_params, x0, y0, delta_disp=5e-8, scheme='central')
        K = np.array([[Kxx, Kxy],
                      [Kyx, Kyy]])

        # Damping
        Cxx, Cxy, Cyx, Cyy = damping_matrix_for_speed(
            theta, Z, Pi0, pa, Lam, R, L, c, x0, y0,
            bc_axial, upwind, sigma_dyn
        )
        C = np.array([[Cxx, Cxy],
                      [Cyx, Cyy]])

        # Critical mass
        mcrit, info = critical_mass(K, C, m_lo=1e-4, m_hi=50.0, tol=1e-3, max_iter=60)
        mcrits.append(mcrit)

        print(f"ω={omega:7.1f} rad/s | Λ={Lam:7.3f} | "
              f"Kxx={Kxx: .3e} Kyy={Kyy: .3e} Kxy={Kxy: .3e} Kyx={Kyx: .3e} | "
              f"Cxx={Cxx: .3e} Cyy={Cyy: .3e} Cxy={Cxy: .3e} Cyx={Cyx: .3e} | "
              f"m_crit={mcrit: .3e} kg ({info['status']})")

    mcrits = np.array(mcrits)
    Lambdas = np.array(Lambdas)

    # Plot critical mass vs speed (and optional vs Λ)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    axs[0].plot(omegas/2/np.pi*60, mcrits, 'b-o', ms=4)
    axs[0].set_xlabel('Speed [rpm]')
    axs[0].set_ylabel('Critical mass m_crit [kg]')
    axs[0].set_title('Critical mass vs rotation speed')
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(Lambdas, mcrits, 'r-s', ms=4)
    axs[1].set_xlabel(r'Compressibility number $\Lambda$')
    axs[1].set_ylabel(r'Critical mass m$_crit$ [kg]')
    # axs[1].set_title(r'Critical mass vs $\Lambda')
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()