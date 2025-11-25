import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

import scienceplots
plt.style.use(['science'])
def gas_journal_bearing(
    R=0.05,        # journal radius [m]
    L=0.05,        # bearing length [m]
    c=50e-6,       # radial clearance [m]
    mu=1.8e-5,     # gas viscosity [Pa*s]
    pa=101325.0,   # ambient pressure [Pa]
    omega=3000.0,  # rad/s
    eps=0.8,       # eccentricity ratio e/c in (0,1)
    phi=0.0,       # attitude angle [rad]
    ntheta=181,    # circumferential nodes
    nz=61,         # axial nodes
    bc_axial='dirichlet', # 'dirichlet' (Π=1 at ends) or 'neumann' (∂Π/∂Z=0)
    upwind=False,        # if True, use first-order upwind on Λ-term
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

    # Axial "diffusivity" uses H3 at center; H varies only with θ
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
                # First-order upwind in θ-direction (wind from +θ if Λ<0)
                # Here Λ >= 0 typically. Implement as flux splitting.
                # F = Λ*H at faces -> use nodal H for simplicity.
                # Approximate ∂(Π H)/∂θ ≈ ( (ΠH)_i - (ΠH)_{i-1} ) / dth
                # This gives cW = +Λ*H_i/dth on Π_{i-1}, cP = -Λ*H_i/dth on Π_i
                # But to keep matrix symmetry style, we place contributions on neighbors:
                # Use hybrid: upstream neighbor gets -Λ H / dth, center gets +Λ H / dth
                # Implement as: add to coefficients in A involving i and i-1.
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
                    # Next is boundary Dirichlet Π=1
                    b[row] += aN * 1.0
                else:
                    A[row, idx(i, j+1)] += -aN
            else:
                # top boundary (should not happen due to earlier Dirichlet)
                pass

            if j-1 >= 0:
                if bc_axial == 'dirichlet' and j-1 == 0:
                    b[row] += aS * 1.0
                else:
                    A[row, idx(i, j-1)] += -aS
            else:
                # bottom boundary
                pass

            # For Neumann axial BCs: ∂Π/∂Z = 0 at edges
            if bc_axial == 'neumann' and z_is_edge:
                # mirror approach: replace missing neighbor with itself
                # which effectively enforces zero gradient.
                # That means add aN or aS to diagonal, and no RHS change
                if j == nz-1:
                    A[row, row] += aN
                if j == 0:
                    A[row, row] += aS

    # Solve
    A = csr_matrix(A)
    Pi_vec = spsolve(A, b)
    Pi = Pi_vec.reshape((nz, ntheta))

    # Dimensional pressure
    p = pa * Pi

    # Load integration (per unit axial length -> integrate across θ and z; here we integrate full area)
    # Pressure over ambient: Δp = p - p_a
    dp = p - pa
    # dA on the journal surface per nodal cell: R dθ * dZ * L (since Z is nondimensional across length L)
    # But Z is nondimensional in [-1/2, 1/2], so actual dz = L dZ. Area element = R * L * dθ * dZ
    dA = R * L * dth * dZ

    # Force components (integrate pressure over area projected onto x,y via unit normals cosθ,sinθ)
    # Using standard convention: θ measured from x-axis
    cos_th = np.cos(theta)[None, :]
    sin_th = np.sin(theta)[None, :]

    Wx = np.sum(dp * cos_th) * dA
    Wy = np.sum(dp * sin_th) * dA
    W = np.hypot(Wx, Wy)
    # Attitude angle of resultant load (bearing reaction), measured from x-axis
    beta = np.arctan2(Wy, Wx)

    results = dict(
        R=R, L=L, c=c, mu=mu, pa=pa, omega=omega, U=U,
        eps=eps, phi=phi, Lam=Lam, alpha=alpha,
        Wx=Wx, Wy=Wy, W=W, beta=beta
    )

    if return_full:
        return theta, Z, Pi, p, results
    else:
        return Pi, results
def plot_pressure(theta, Z, p, pa=101325.0, title="Gas journal bearing pressure"):
    """
    Plot dimensional pressure p( Z, theta ) and axial midline profile.
    Inputs:
        theta: (ntheta,) array in [0, 2π)
        Z:     (nz,) array in [-0.5, 0.5]
        p:     (nz, ntheta) dimensional pressure [Pa]
        pa:    ambient pressure [Pa] for reference
    """
    # Convert to degrees for nicer ticks
    theta_deg = theta * 180.0 / np.pi
    # For imshow extent: [θ_min, θ_max, Z_min, Z_max]
    extent = [theta_deg.min(), theta_deg.max(), Z.min(), Z.max()]

    # Mid-axial index (closest to Z=0)
    j_mid = np.argmin(np.abs(Z - 0.0))

    fig = plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 14})
    # fig.set_size_inches(12, 6)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])

    # Left: colormap of pressure
    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(p*1e-3, extent=extent, aspect='auto', origin='lower', cmap='jet')
    cbar = plt.colorbar(im, ax=ax0)
    cbar.set_label('Pressure $p$ [kPa]')
    # ax0.set_title(title)
    ax0.set_xlabel(r'$\theta$ [deg]')
    ax0.set_ylabel('Axial coordinate Z [-]')
    # Reference line at Z=0
    ax0.axhline(0.0, color='w', ls='--', lw=0.8)

    # Right: mid-axial pressure vs θ
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(theta_deg, p[j_mid, :]*1e-3, label=r'$p(\theta, Z=0)$')
    ax1.hlines(pa*1e-3, theta_deg.min(), theta_deg.max(), colors='k', linestyles=':', label='ambient $p_a$')
    ax1.set_xlabel(r'$\theta$ [deg]')
    ax1.set_ylabel('Pressure [Pa]')
    # ax1.set_title('Mid-axial pressure profile')
    # ax1.grid(False, alpha=0.3)
    ax1.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("compressible_final.png",dpi=300)
    plt.show()
    
if __name__ == "__main__":
    from scipy.sparse.linalg import spsolve  # ensure SciPy is available
    
    theta, Z, Pi, p, res = gas_journal_bearing(
        R=0.05, L=0.05, c=40e-6, mu=1.85e-5, pa=101325, omega=2000,
        eps=0.7, phi=0.0, ntheta=241, nz=81, bc_axial='dirichlet', upwind=False
    )
    print("Λ =", res["Lam"], " Load W [N] =", res["W"])
    plot_pressure(theta, Z, p, pa=res["pa"])
