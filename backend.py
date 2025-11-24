import numpy as np
from typing import Tuple

from flask import Flask, request, jsonify
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from flask_cors import CORS

app = Flask(__name__)

# Allow any origin for /api/* (local dev)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)

# =========================================================
# Analysis functions (your code)
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


if __name__ == "__main__":
    app.run(debug=True)
