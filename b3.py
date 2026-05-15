
import numpy as np
from scipy.special import lpmv
from math import factorial, sqrt, pi
import os

INPUT  = "b2_potentials"
OUTDIR = "b3_sha"
os.makedirs(OUTDIR, exist_ok=True)

N_MAX = 13   # Buchete used n = 0 … 13

files = sorted(f for f in os.listdir(INPUT) if f.endswith("_U.npy"))
if not files:
    print("[!] No _U.npy files found in b2_potentials/ — run b2.py first.")
    raise SystemExit


def build_legendre_table(n_max, cos_theta_arr):
    
    table = {}
    for n in range(n_max + 1):
        for m in range(n + 1):
            table[(n, m)] = lpmv(m, n, cos_theta_arr)   # shape (n_theta,)
    return table


for fname in files:
    print(f"[i] SHA: {fname}")

    U = np.load(os.path.join(INPUT, fname))   # shape (n_theta, n_phi)
    n_theta, n_phi = U.shape

    # ── Grid (cell-centred, matching b1.py) ───────────────────────────────────
    dtheta = np.pi  / n_theta
    dphi   = 2*np.pi / n_phi

    theta_vals = np.arange(n_theta) * dtheta + dtheta / 2.0   # centres in (0, π)
    phi_vals   = np.arange(n_phi)   * dphi   + dphi   / 2.0   # centres in (0, 2π)

    sin_theta  = np.sin(theta_vals)   # shape (n_theta,)
    cos_theta  = np.cos(theta_vals)   # shape (n_theta,)

    # ── Integration weights: sin(θ) Δθ Δφ ────────────────────────────────────
    # shape (n_theta, n_phi) — same weight for each phi at a given theta
    weights = (sin_theta * dtheta)[:, None] * dphi   # broadcast over phi

    # Weighted potential: U * sin(θ) Δθ Δφ — shape (n_theta, n_phi)
    Uw = U * weights

    # ── Pre-compute cos/sin of phi for all m up to N_MAX ─────────────────────
    # cos_mphi[m, j] = cos(m * phi_j),  shape (N_MAX+1, n_phi)
    m_arr = np.arange(N_MAX + 1)[:, None]          # (N_MAX+1, 1)
    phi_row = phi_vals[None, :]                    # (1, n_phi)
    cos_mphi = np.cos(m_arr * phi_row)             # (N_MAX+1, n_phi)
    sin_mphi = np.sin(m_arr * phi_row)             # (N_MAX+1, n_phi)

    # ── Pre-compute Legendre table ────────────────────────────────────────────
    Ptable = build_legendre_table(N_MAX, cos_theta)

    # ── Compute coefficients ──────────────────────────────────────────────────
    coeffs = []

    for n in range(N_MAX + 1):
        for m in range(n + 1):

            # Normalisation factor (Buchete eq. 6)
            alpha_nm = sqrt((2*n + 1) / (2*pi) * factorial(n - m) / factorial(n + m))

            Pnm = Ptable[(n, m)]   # shape (n_theta,)

            # Weighted Legendre: shape (n_theta, 1) * (n_theta, n_phi) → (n_theta, n_phi)
            PUw = Pnm[:, None] * Uw   # P_nm(cos θ) × U × sin θ Δθ Δφ

            # a_nm = alpha_nm × ΣΣ PUw × cos(m φ)
            a_nm = alpha_nm * np.sum(PUw * cos_mphi[m])
            b_nm = alpha_nm * np.sum(PUw * sin_mphi[m])

            coeffs.append((n, m, a_nm, b_nm))

    # Save as structured array
    dtype  = np.dtype([("n", np.int32), ("m", np.int32),
                       ("a_nm", np.float64), ("b_nm", np.float64)])
    arr    = np.array(coeffs, dtype=dtype)

    outname = fname.replace("_U.npy", "_coeffs.npy")
    np.save(os.path.join(OUTDIR, outname), arr)
    print(f"    → {outname}  ({len(coeffs)} coefficients)")

print(f"\n[✓] B3 complete — SHA coefficients saved to {OUTDIR}/")
print(f"    Up to n={N_MAX}, giving (n+1)²={(N_MAX+1)**2} coefficients per file")