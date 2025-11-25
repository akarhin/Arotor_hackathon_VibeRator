import numpy as np
from typing import Tuple
import io
import base64
from flask import Flask, request, jsonify
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from flask_cors import CORS
import matplotlib.pyplot as plt
import ross as rs

app = Flask(__name__)

# Allow any origin for /api/* (local dev)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)

# =========================================================
# Analysis functions BEARING
# =========================================================


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


def add_bearing_matrices_user(C, K, Kb, Cb, node, Nn):
    """
    Insert user-provided bearing matrices into global K and C.

    Kb, Cb: small matrices (e.g. 2x2 for x,y translations)
    node: bearing node index
    """
    C_new = C.copy()
    K_new = K.copy()

    Kb = np.asarray(Kb, dtype=float)
    Cb = np.asarray(Cb, dtype=float)

    # only translational DOFs: x and y
    dof_x = 2 * node
    dof_y = 2 * Nn + 2 * node

    # Map 2x2 user matrix into global matrix
    dofs = [dof_x, dof_y]

    for i in range(2):
        for j in range(2):
            K_new[dofs[i], dofs[j]] += Kb[i, j]
            C_new[dofs[i], dofs[j]] += Cb[i, j]

    return C_new, K_new


# =========================================================
# Analysis functions CRITICAL SPEED:
# =========================================================
def eigenfrequencies(M, C, K, G=None, imag_tol=1e-6, uniq_tol=1e-3):
    """
    Damped natural frequencies (Hz) at Ω = 0 from

        M q¨ + (C + G) q˙ + K q = 0
    """
    M = np.asarray(M, dtype=np.complex128)
    C = np.asarray(C, dtype=np.complex128)
    K = np.asarray(K, dtype=np.complex128)

    if G is None:
        G = np.zeros_like(M, dtype=np.complex128)
    else:
        G = np.asarray(G, dtype=np.complex128)

    nd = M.shape[0]
    Z = np.zeros((nd, nd), dtype=np.complex128)
    I = np.eye(nd, dtype=np.complex128)
    Minv = np.linalg.inv(M)

    A = np.block([
        [Z,                  I],
        [-Minv @ K, -Minv @ (C + G)],
    ])

    s, _ = np.linalg.eig(A)

    w = np.abs(np.imag(s))
    w = w[w > imag_tol]

    if w.size == 0:
        return np.array([])

    w = np.sort(w)
    w_unique = []
    for wi in w:
        if not w_unique or abs(wi - w_unique[-1]) > uniq_tol:
            w_unique.append(wi)
    w = np.array(w_unique)

    f_hz = w / (2.0 * np.pi)
    return f_hz


def eigenfrequencies_speed(M, C, K, G_base, Omega,
                           imag_tol=1e-6, uniq_tol=1e-3):
    """
    Damped natural frequencies (Hz) at a given spin speed Ω [rad/s] from

        M q¨ + (C + Ω G_base) q˙ + K q = 0
    """
    M = np.asarray(M, dtype=np.complex128)
    C = np.asarray(C, dtype=np.complex128)
    K = np.asarray(K, dtype=np.complex128)
    G = np.asarray(G_base, dtype=np.complex128)

    nd = M.shape[0]
    Z = np.zeros((nd, nd), dtype=np.complex128)
    I = np.eye(nd, dtype=np.complex128)
    Minv = np.linalg.inv(M)

    C_eff = C + Omega * G

    A = np.block([
        [Z,                    I],
        [-Minv @ K, -Minv @ C_eff],
    ])

    s, _ = np.linalg.eig(A)

    w = np.abs(np.imag(s))
    w = w[w > imag_tol]

    if w.size == 0:
        return np.array([])

    w = np.sort(w)
    w_unique = []
    for wi in w:
        if not w_unique or abs(wi - w_unique[-1]) > uniq_tol:
            w_unique.append(wi)
    w = np.array(w_unique)

    f_hz = w / (2.0 * np.pi)
    return f_hz


# ------------------------------------------------------------
# 2) FE assembly: Euler–Bernoulli shaft in two planes
# ------------------------------------------------------------

def euler_bernoulli_element_mk(E, I, rhoA, L):
    L2 = L*L
    L3 = L2*L

    k = E * I / L3
    Ke = k * np.array([
        [12,    6*L,  -12,    6*L],
        [6*L,  4*L2,  -6*L,  2*L2],
        [-12,   -6*L,   12,   -6*L],
        [6*L,  2*L2,  -6*L,  4*L2]
    ], dtype=float)

    m = rhoA * L / 420.0
    Me = m * np.array([
        [156,    22*L,   54,   -13*L],
        [22*L,  4*L2,  13*L,  -3*L2],
        [54,    13*L,  156,   -22*L],
        [-13*L, -3*L2, -22*L,   4*L2]
    ], dtype=float)

    return Me, Ke


def assemble_shaft_mk(num_elements, L_total, E, I, rhoA):
    Nn = num_elements + 1
    dof_per_node_plane = 2
    ndof_plane = dof_per_node_plane * Nn

    Mx = np.zeros((ndof_plane, ndof_plane))
    Kx = np.zeros((ndof_plane, ndof_plane))

    Le = L_total / num_elements
    Me, Ke = euler_bernoulli_element_mk(E, I, rhoA, Le)

    for e in range(num_elements):
        n1 = e
        n2 = e + 1
        idx = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1])
        for i in range(4):
            for j in range(4):
                Mx[idx[i], idx[j]] += Me[i, j]
                Kx[idx[i], idx[j]] += Ke[i, j]

    My = Mx.copy()
    Ky = Kx.copy()

    M = np.block([
        [Mx, np.zeros_like(Mx)],
        [np.zeros_like(My), My]
    ])
    K = np.block([
        [Kx, np.zeros_like(Kx)],
        [np.zeros_like(Ky), Ky]
    ])

    return M, K, Nn


def add_disc(M, Nn, node_disc, m_disc, Id):
    M_new = M.copy()

    dof_x = 2*node_disc                # x translation
    dof_y = 2*Nn + 2*node_disc         # y translation
    M_new[dof_x, dof_x] += m_disc
    M_new[dof_y, dof_y] += m_disc

    dof_tx = 2*node_disc + 1           # rotation (x-plane)
    dof_ty = 2*Nn + 2*node_disc + 1    # rotation (y-plane)
    M_new[dof_tx, dof_tx] += Id
    M_new[dof_ty, dof_ty] += Id

    return M_new


def add_disk_gyroscopic(G, Nn, node_disc, Ip):
    """
    Disk gyroscopic matrix at node_disc.
    Couples bending rotations θx and θy with polar inertia Ip.
    """
    G_new = G.copy()

    dof_tx = 2*node_disc + 1           # rotation in x-plane
    dof_ty = 2*Nn + 2*node_disc + 1    # rotation in y-plane

    G_new[dof_tx, dof_ty] += -Ip
    G_new[dof_ty, dof_tx] += Ip

    return G_new


def add_bearing_matrices(C, K, bearing_left, bearing_right, node_left, node_right, Nn):
    C_new = C.copy()
    K_new = K.copy()

    def add_case(case, node):
        kxx = case["kxx"] * 1e6
        kyy = case["kyy"] * 1e6
        cxx = case["cxx"] * 1e3
        cyy = case["cyy"] * 1e3

        dof_x = 2*node
        dof_y = 2*Nn + 2*node

        K_new[dof_x, dof_x] += kxx
        K_new[dof_y, dof_y] += kyy
        C_new[dof_x, dof_x] += cxx
        C_new[dof_y, dof_y] += cyy

    add_case(bearing_left,  node_left)
    add_case(bearing_right, node_right)

    return C_new, K_new


def eigs_at_speed(M, C, K, G_base, Omega, imag_tol=1e-6):
    """
    Eigenvalues s at given spin speed Ω [rad/s] for

        M q¨ + (C + Ω G_base) q˙ + K q = 0

    Returns eigenvalues s (complex) with non-negligible imag part.
    """
    M = np.asarray(M,  dtype=np.complex128)
    C0 = np.asarray(C,  dtype=np.complex128)
    K = np.asarray(K,  dtype=np.complex128)
    G = np.asarray(G_base, dtype=np.complex128)

    nd = M.shape[0]
    Z = np.zeros((nd, nd), dtype=np.complex128)
    I = np.eye(nd, dtype=np.complex128)
    Minv = np.linalg.inv(M)

    C_eff = C0 + Omega * G

    A = np.block([
        [Z,                   I],
        [-Minv @ K, -Minv @ C_eff],
    ])

    s, _ = np.linalg.eig(A)

    return s[np.abs(np.imag(s)) > imag_tol]


def iterative_critical_speeds(M, C, K, G_base,
                              Omega0_list,
                              n=1,
                              max_iter=20,
                              rel_tol=1e-6):
    """
    Iterative critical-speed search for each initial guess Ω_r^(0) in Omega0_list.

    For mode r (0-based):
      Ω_r^(k+1) = Im(s_r(Ω_r^(k))) / n,
    where s_r is the r-th eigenvalue sorted by Im(s) > 0.
    """
    M = np.asarray(M,  dtype=np.complex128)
    C0 = np.asarray(C,  dtype=np.complex128)
    K = np.asarray(K,  dtype=np.complex128)
    G = np.asarray(G_base, dtype=np.complex128)

    n_modes = len(Omega0_list)
    Omega_hist = [[] for _ in range(n_modes)]
    Omega_conv = np.zeros(n_modes, dtype=float)

    for r, Omega0 in enumerate(Omega0_list):
        Omega = float(Omega0)
        for it in range(max_iter):
            s = eigs_at_speed(M, C0, K, G, Omega)

            s_pos = s[np.imag(s) > 0.0]
            if s_pos.size == 0:
                raise RuntimeError(
                    f"No positive-imag eigenvalues at Ω={Omega}")

            s_pos = s_pos[np.argsort(np.imag(s_pos))]
            idx = min(r, len(s_pos) - 1)
            s_r = s_pos[idx]

            Omega_new = np.imag(s_r) / float(n)
            Omega_hist[r].append(Omega_new)

            if it > 0:
                if abs(Omega_new - Omega) / max(Omega, 1e-12) < rel_tol:
                    Omega = Omega_new
                    break

            Omega = Omega_new

        Omega_conv[r] = Omega

    Omega_rpm = Omega_conv * 60.0 / (2.0 * np.pi)
    return Omega_conv, Omega_rpm, Omega_hist


def modes_at_speed(M, C, K, G_base, Omega, imag_tol=1e-6):
    """
    Eigenvalues s and eigenvectors V at spin speed Ω [rad/s] for

        M q¨ + (C + Ω G_base) q˙ + K q = 0  →  x˙ = A x
    """
    M = np.asarray(M,  dtype=np.complex128)
    C0 = np.asarray(C,  dtype=np.complex128)
    K = np.asarray(K,  dtype=np.complex128)
    G = np.asarray(G_base, dtype=np.complex128)

    nd = M.shape[0]
    Z = np.zeros((nd, nd), dtype=np.complex128)
    I = np.eye(nd, dtype=np.complex128)
    Minv = np.linalg.inv(M)

    C_eff = C0 + Omega * G

    A = np.block([
        [Z,                   I],
        [-Minv @ K, -Minv @ C_eff],
    ])

    s, V = np.linalg.eig(A)
    mask = np.abs(np.imag(s)) > imag_tol
    return s[mask], V[:, mask]


def mode_shape_data(M, C, K, G_base, Nn, L_total, mode_index, rpm):
    Omega = 2.0 * np.pi * rpm / 60.0
    s, V = modes_at_speed(M, C, K, G_base, Omega)

    idx = np.where(np.imag(s) > 0)[0]
    s = s[idx]
    V = V[:, idx]
    order = np.argsort(np.imag(s))
    s = s[order]
    V = V[:, order]

    mode_idx = min(mode_index, len(s) - 1)
    v = V[:, mode_idx]
    q = v[:M.shape[0]]

    dof_x = np.arange(0, 2 * Nn, 2)
    dof_y = 2 * Nn + np.arange(0, 2 * Nn, 2)

    wx = np.real(q[dof_x])
    wy = np.real(q[dof_y])

    x_nodes = np.linspace(0.0, L_total, Nn)

    amp = max(np.max(np.abs(wx)), np.max(np.abs(wy)), 1e-12)
    wx_n = (wx / amp).tolist()
    wy_n = (wy / amp).tolist()

    return {
        "x": x_nodes.tolist(),
        "wx": wx_n,
        "wy": wy_n,
        "rpm": float(rpm),
        "mode": int(mode_index) + 1
    }


def fig_to_base64(fig):
    buf = io.BytesIO()
    buf.write(fig.to_image(format="png", scale=2))
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# =========================================================
# API route
# =========================================================


@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def api_analyze():
    if request.method == "OPTIONS":
        # Preflight – Flask-CORS will attach the proper headers
        return "", 200

    data = request.get_json(force=True)
    try:
        bearing_type = data.get("bearingType", "journal")
        fluid_type = data.get("fluidType", "oil")

        diameter = float(data["diameter"])
        length = float(data["length"])
        clearance = float(data["clearance"])
        load = float(data["load"])
        viscosity = float(data["viscosity"])
        speed_rpm = float(data["speed_rpm"])

        eps, Ss, S = compute_eps(
            diameter=diameter,
            length=length,
            load=load,
            clearance=clearance,
            viscosity=viscosity,
            speed_rpm=speed_rpm,
        )

        speed_rad = speed_rpm * 2.0 * np.pi / 60.0
        fc = load if load != 0.0 else 1.0
        Ke, Ce = compute_ke_ce(eps, fc=fc, Omega=speed_rad)

        p, Theta, Z, h = solve_journal_bearing_pressure(
            viscosity=viscosity,
            speed_rad=speed_rad,
            diameter=diameter,
            length=length,
            clearance=clearance,
            eps=eps,
            Ntheta=80,
            Nz=40,
        )

        theta_vec = Theta[:, 0]
        z_vec = Z[0, :]

        result = {
            "eps": float(eps),
            "K": Ke.tolist(),
            "C": Ce.tolist(),
            "theta": theta_vec.tolist(),
            "z": z_vec.tolist(),
            "p": p.tolist(),
        }
        return jsonify(result)

    except KeyError as e:
        return jsonify({"error": f"missing field: {e}"}), 400
    except Exception as e:
        print("Error in /api/analyze:", e)
        return jsonify({"error": str(e)}), 500


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


# =========================================================
# Correct critical-speed endpoint
# =========================================================
@app.route("/api/critical-speeds", methods=["POST"])
def critical():
    d = request.get_json(force=True)

    # ---- Inputs ----
    L = float(d["shaftLength"])
    dshaft = float(d["shaftDiameter"])
    ne = int(d["numElements"])
    rho = float(d["density"])
    E = float(d["youngsModulus"])

    m = float(d["diskMass"])
    Id = float(d["diskId"])
    Ip = float(d["diskIp"])

    Kb = np.array(d["Kbearing"], dtype=float)
    Cb = np.array(d["Cbearing"], dtype=float)
    rpm_max = float(d["omegaMaxRpm"])

    bearing = {
        "kxx": Kb[0][0],
        "kyy": Kb[1][1],
        "cxx": Cb[0][0],
        "cyy": Cb[1][1],
    }

    # ---- Build model (matches your working reference) ----
    A = np.pi * dshaft**2 / 4
    I = np.pi * dshaft**4 / 64
    rhoA = rho * A

    M_shaft, K_shaft, Nn = assemble_shaft_mk(ne, L, E, I, rhoA)

    C_struct = np.zeros_like(M_shaft)
    G_shaft = np.zeros_like(M_shaft)

    node_disc = Nn - 1
    M_model = add_disc(M_shaft, Nn, node_disc, m, Id)
    G_model = add_disk_gyroscopic(G_shaft, Nn, node_disc, Ip)

    node_left = 0
    node_right = ne

    C_model, K_model = add_bearing_matrices(
        C_struct, K_shaft,
        bearing, bearing,
        node_left, node_right, Nn
    )

    # ---- Campbell data ----
    n_points = 120
    n_modes = 6

    rpm = np.linspace(0, rpm_max, n_points)
    freqs = np.full((n_points, n_modes), np.nan)

    for i, r in enumerate(rpm):
        Omega = 2.0 * np.pi * r / 60.0
        f = eigenfrequencies_speed(M_model, C_model, K_model, G_model, Omega)
        mcount = min(n_modes, len(f))
        freqs[i, :mcount] = f[:mcount]

    # ---- Critical speeds (correct method) ----
    rpm_ref = 1000.0
    Omega_ref = 2.0 * np.pi * rpm_ref / 60.0
    f_ref = eigenfrequencies_speed(
        M_model, C_model, K_model, G_model, Omega_ref
    )[:5]

    Omega0 = 2.0 * np.pi * f_ref

    _, Omega_crit_rpm, _ = iterative_critical_speeds(
        M_model, C_model, K_model, G_model,
        Omega0,
        n=1,
        max_iter=20,
        rel_tol=1e-6
    )

    # ---- MODE SHAPES FOR EACH CRITICAL SPEED ----
    mode_shapes = []

    for i, rpm_c in enumerate(Omega_crit_rpm):
        Omega_c = 2.0 * np.pi * rpm_c / 60.0

        s, V = modes_at_speed(M_model, C_model, K_model, G_model, Omega_c)

        idx = np.where(np.imag(s) > 0)[0]
        s_pos = s[idx]
        V_pos = V[:, idx]
        order = np.argsort(np.imag(s_pos))
        s_sort = s_pos[order]
        V_sort = V_pos[:, order]

        mode_idx = min(i, len(s_sort) - 1)
        v = V_sort[:, mode_idx]
        q = v[:M_model.shape[0]]

        dof_x = np.arange(0, 2 * Nn, 2)
        dof_y = 2 * Nn + np.arange(0, 2 * Nn, 2)

        wx = np.real(q[dof_x])
        wy = np.real(q[dof_y])

        x_nodes = np.linspace(0.0, L, Nn)

        amp = max(np.max(np.abs(wx)), np.max(np.abs(wy)), 1e-12)
        wx_n = (wx / amp).tolist()
        wy_n = (wy / amp).tolist()

        mode_shapes.append({
            "mode": int(i + 1),
            "rpm": float(rpm_c),
            "x": x_nodes.tolist(),
            "wx": wx_n,
            "wy": wy_n
        })

    # ---- ROSS rotor image (only this extra) ----
    steel = rs.Material(name="Steel", rho=rho, E=E, G_s=E / 2.6)

    Le = L / ne
    shaft_elements = [
        rs.ShaftElement(
            L=Le,
            idl=0.0,
            odl=dshaft,
            material=steel,
            shear_effects=True,
            rotary_inertia=True,
            gyroscopic=True,
        )
        for _ in range(ne)
    ]

    disk = rs.DiskElement(
        n=node_disc,
        m=m,
        Ip=Ip,
        Id=Id
    )

    bearingL = rs.BearingElement(
        n=node_left,
        kxx=bearing["kxx"],
        kyy=bearing["kyy"],
        cxx=bearing["cxx"],
        cyy=bearing["cyy"],
    )
    bearingR = rs.BearingElement(
        n=node_right,
        kxx=bearing["kxx"],
        kyy=bearing["kyy"],
        cxx=bearing["cxx"],
        cyy=bearing["cyy"],
    )

    rotor = rs.Rotor(
        shaft_elements=shaft_elements,
        disk_elements=[disk],
        bearing_elements=[bearingL, bearingR],
    )

    fig_rotor = rotor.plot_rotor()
    # rotor_img = fig_to_base64(fig_rotor)

    return jsonify({
        "rpm": rpm.tolist(),
        "freqs": freqs.tolist(),
        "critical_speeds_rpm": Omega_crit_rpm.tolist(),
        "mode_shapes": mode_shapes,
        "rotor_figure": fig_rotor.to_json()
    })


if __name__ == "__main__":
    app.run(debug=True)
