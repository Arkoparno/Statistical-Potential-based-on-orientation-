
import re, os
import numpy as np
from scipy.special import lpmv
from math import factorial, sqrt, pi as _pi
from collections import defaultdict
from matplotlib.patches import Patch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── I/O ────────────────────────────────────────────────────────────────────────
SHA_DIR  = "b3_sha"
PMF_DIR  = "b2_potentials"
HIST_DIR = "b1_histograms"
OUTDIR   = "results"
os.makedirs(OUTDIR, exist_ok=True)

# ── Grid / resolution ──────────────────────────────────────────────────────────
GRID       = "12x24"
DIST_BINS  = ["short", "medium", "long"]
DIST_RANGE = {"short": "2–8 Å", "medium": "8–15 Å", "long": "15–25 Å"}
FRAMES     = ["nuc", "aa"]
THETA_FINE = 96
PHI_FINE   = 192
# CRITICAL: cell-centred, NOT linspace(0,π) — the poles θ=0 and θ=π cause
# lpmv(m,n,±1) to blow up for high orders (m≥8), making the SHS diverge to 10¹²
# and masking the raw data in the colour scale.
_dth_fine  = np.pi   / THETA_FINE
_dphi_fine = 2*np.pi / PHI_FINE
theta_fine = np.arange(THETA_FINE) * _dth_fine  + _dth_fine  / 2  # (0°,180°) exclusive
phi_fine   = np.arange(PHI_FINE)   * _dphi_fine + _dphi_fine / 2  # (0°,360°) exclusive

CMAP_PMF  = "RdBu_r"   # blue = attractive (U<0), red = repulsive (U>0)
CMAP_PROB = "YlOrRd"
EPS       = 1e-12

def sin_weights(n_theta, n_phi):
    """Solid-angle weights matching b2.py."""
    dth   = np.pi / n_theta
    theta = np.arange(n_theta) * dth + dth / 2.0
    w     = np.sin(theta)
    return np.maximum(w[:, None] * np.ones(n_phi), EPS)


# ══════════════════════════════════════════════════════════════════════════════
# MATHS
# ══════════════════════════════════════════════════════════════════════════════

def grid_angles(n_theta, n_phi):
    dth = np.pi   / n_theta
    dph = 2*np.pi / n_phi
    return np.arange(n_theta)*dth + dth/2, np.arange(n_phi)*dph + dph/2


def reconstruct(coeffs_arr, theta_vals, phi_vals):
   
    U  = np.zeros((len(theta_vals), len(phi_vals)))
    ct = np.cos(theta_vals)
    for row in coeffs_arr:
        n, m = int(row["n"]), int(row["m"])
        a, b = float(row["a_nm"]), float(row["b_nm"])
        # normalization factor — same formula as b3.py
        alpha_nm = sqrt((2*n + 1) / (2*_pi) * factorial(n - m) / factorial(n + m))
        Pnm  = lpmv(m, n, ct)          # unnormalized, can be large for high m
        nPnm = alpha_nm * Pnm          # normalized: bounded, max ~O(1)
        ang  = nPnm[:, None] * (a*np.cos(m*phi_vals) + b*np.sin(m*phi_vals))[None,:]
        if m == 0: ang *= 0.5
        U += ang
    return U


def sym_clim(U, margin=1.05):
    """Full-range symmetric colour limit."""
    return max(max(abs(float(U.min())), abs(float(U.max())))*margin, 1e-3)

def sym_clim_p95(U):
    """FIX 4: 95th-percentile symmetric limit for smoother visual appearance."""
    return max(float(np.percentile(np.abs(U), 95)), 1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def _moll(ax, U, tv, pv, vmin, vmax, cmap, title=""):
    lon = pv - np.pi; lat = np.pi/2 - tv
    dlat = (lat[1]-lat[0]) if len(lat)>1 else 0.01
    dlon = (lon[1]-lon[0]) if len(lon)>1 else 0.01
    le = np.concatenate([[lat[0]-dlat/2], lat+dlat/2])
    lo = np.concatenate([[lon[0]-dlon/2], lon+dlon/2])
    LO, LA = np.meshgrid(lo, le)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    ax.pcolormesh(LO, LA, U, cmap=cmap, norm=norm, shading="flat", rasterized=True)
    if title: ax.set_title(title, fontsize=7, pad=2)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(True, alpha=0.12, lw=0.3)
    return cm.ScalarMappable(norm=norm, cmap=cmap)


def moll_pmf(ax, U, tv, pv, vmax, title=""):
    return _moll(ax, U, tv, pv, -vmax, vmax, CMAP_PMF, title)

def moll_prob(ax, P, tv, pv, title=""):
    return _moll(ax, P, tv, pv, 0, max(float(P.max()),1e-6), CMAP_PROB, title)


def sphere3d(ax, U, tv, pv, vmax, title=""):
    """FIX 5: Buchete-style sphere with shade=True, alpha=0.95."""
    tm, pm = np.meshgrid(tv, pv, indexing="ij")
    X=np.sin(tm)*np.cos(pm); Y=np.sin(tm)*np.sin(pm); Z=np.cos(tm)
    norm = mcolors.Normalize(-vmax, vmax)
    fc   = matplotlib.colormaps[CMAP_PMF](norm(U))
    ax.plot_surface(X,Y,Z,facecolors=fc,rstride=1,cstride=1,linewidth=0,
                    antialiased=True,shade=True,alpha=0.95)
    ax.set_title(title, fontsize=7, pad=2); ax.set_axis_off()
    try: ax.set_box_aspect([1,1,1])
    except AttributeError: pass


def polar3d(ax, U, tv, pv, title=""):
    """FIX 5: Polar surface with shade=True, alpha=0.95."""
    tm, pm = np.meshgrid(tv, pv, indexing="ij")
    R = np.abs(U)/(np.abs(U).max()+1e-12)
    X=R*np.sin(tm)*np.cos(pm); Y=R*np.sin(tm)*np.sin(pm); Z=R*np.cos(tm)
    fc = np.where(U[...,None]>0,[0.85,0.18,0.18,1.0],[0.18,0.35,0.85,1.0])
    ax.plot_surface(X,Y,Z,facecolors=fc,rstride=1,cstride=1,linewidth=0,
                    antialiased=True,shade=True,alpha=0.95)
    ax.set_title(title, fontsize=7, pad=2); ax.set_axis_off()
    try: ax.set_box_aspect([1,1,1])
    except AttributeError: pass


# ══════════════════════════════════════════════════════════════════════════════
# FILE PARSING
# ══════════════════════════════════════════════════════════════════════════════

STEM_RE = re.compile(
    r'^(12x24|36x36)_(short|medium|long)_([A-Z]+)_(D[ATGC])_(BB-[BSP]|SC-[BSP])_(nuc|aa)$'
)

def parse_stem(stem):
    m = STEM_RE.match(stem)
    if not m: return None
    return dict(grid=m.group(1), dist=m.group(2), aa=m.group(3),
                nuc=m.group(4), itype=m.group(5), frame=m.group(6))

def safe_load(path):
    try:    return np.load(path, allow_pickle=True)
    except: return None

def save_fig(fig, path, dpi=180):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL PLOTS (per range, per frame + total)
# ══════════════════════════════════════════════════════════════════════════════

def make_per_range_plots(pdir, frame_tag, U_fine, theta_d, phi_d, hist, label):
    """Generate P_map, sphere, combined hemispheres, polar3D for one (range, frame/total)."""
    # FIX 4: use 95th-percentile vmax for smoother visual (avoids extreme outliers dominating scale)
    vmax = sym_clim_p95(U_fine)

    # P_map (probability histogram)
    if hist is not None and frame_tag != "total":
        nt, np_ = hist.shape
        W = sin_weights(nt, np_)
        P = hist.astype(float) / W
        P /= (P.sum() + EPS)
        fig, ax = plt.subplots(1, 1, figsize=(6, 3.2),
                               subplot_kw={"projection": "mollweide"})
        sm = moll_prob(ax, P, theta_d, phi_d, f"P(θ,φ)  {label}")
        fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.05, label="Probability")
        fig.tight_layout()
        fig.subplots_adjust(top=0.90)
        save_fig(fig, os.path.join(pdir, f"P_map_{frame_tag}.pdf"), 160)

    # sphere  (FIX 5: shade=True already applied inside sphere3d)
    fig = plt.figure(figsize=(5, 5))
    ax  = fig.add_subplot(111, projection="3d")
    sphere3d(ax, U_fine, theta_fine, phi_fine, vmax, f"3D  {label}")
    sm = cm.ScalarMappable(cmap=CMAP_PMF, norm=mcolors.Normalize(-vmax, vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.0, label="U (kBT)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    save_fig(fig, os.path.join(pdir, f"sphere_{frame_tag}.pdf"), 160)

    # FIX 1: combined front + back in one side-by-side figure
    half = THETA_FINE // 2
    U_front, th_front = U_fine[:half, :], theta_fine[:half]
    U_back,  th_back  = U_fine[half:, :], theta_fine[half:]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.5),
                            subplot_kw={"projection": "mollweide"})
    sm_f = moll_pmf(axs[0], U_front, th_front, phi_fine, vmax,
                    title="Front  θ = 0→90°")
    sm_b = moll_pmf(axs[1], U_back,  th_back,  phi_fine, vmax,
                    title="Back   θ = 90→180°")
    for a in axs:
        a.set_xticklabels([])
        a.set_yticklabels([])
    # shared colourbar on the right
    fig.subplots_adjust(right=0.88, wspace=0.06, top=0.88)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.65])
    fig.colorbar(cm.ScalarMappable(cmap=CMAP_PMF,
                                   norm=mcolors.Normalize(-vmax, vmax)),
                 cax=cax, label="U (kBT)")
    # FIX 2 + FIX 3: clean title, no formula, no overlap
    fig.suptitle(label, fontsize=9, fontweight="bold", y=0.97)
    save_fig(fig, os.path.join(pdir, f"hemispheres_{frame_tag}.pdf"), 160)

    # polar3D  (FIX 5: shade=True already applied inside polar3d)
    fig = plt.figure(figsize=(5, 5))
    ax  = fig.add_subplot(111, projection="3d")
    polar3d(ax, U_fine, theta_fine, phi_fine, f"Polar |U|  {label}")
    ax.legend(handles=[Patch(facecolor="#D92B2B", label="Repulsive U>0"),
                       Patch(facecolor="#2B59D9", label="Attractive U<0")],
              loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    save_fig(fig, os.path.join(pdir, f"polar3D_{frame_tag}.pdf"), 160)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    stems = []
    for f in sorted(os.listdir(SHA_DIR)):
        if not f.endswith("_coeffs.npy"): continue
        stem = f[:-len("_coeffs.npy")]
        info = parse_stem(stem)
        if info and info["grid"] == GRID:
            stems.append((stem, info))

    if not stems:
        print(f"[!] No {GRID} coefficient files in {SHA_DIR}/ — run b3.py first.")
        return

    print(f"[i] {len(stems)} coefficient files (grid={GRID})\n")

    groups = defaultdict(dict)
    for stem, info in stems:
        groups[(info["aa"], info["nuc"], info["itype"])][(info["dist"], info["frame"])] = stem

    total_g = len(groups)

    for g_idx, ((aa, nuc, itype), dfmap) in enumerate(sorted(groups.items())):
        label     = f"{aa}–{nuc}  {itype}"
        itype_dir = os.path.join(OUTDIR, aa, nuc, itype)
        os.makedirs(itype_dir, exist_ok=True)
        print(f"[{g_idx+1}/{total_g}] {aa}/{nuc}/{itype}")

        # ── Load all (dist, frame) data ─────────────────────────────────────
        # per_range[dist] = {nuc: {...}, aa: {...}} with arrays
        per_range = {d: {} for d in DIST_BINS}

        for dist in DIST_BINS:
            for frame in FRAMES:
                stem = dfmap.get((dist, frame))
                if not stem: continue
                coeffs = safe_load(os.path.join(SHA_DIR,  f"{stem}_coeffs.npy"))
                U_raw  = safe_load(os.path.join(PMF_DIR,  f"{stem}_U.npy"))
                hist   = safe_load(os.path.join(HIST_DIR, f"{stem}_counts.npy"))
                if coeffs is None or U_raw is None: continue

                nt, np_ = U_raw.shape
                td, pd  = grid_angles(nt, np_)
                U_sd    = reconstruct(coeffs, td, pd)
                U_sf    = reconstruct(coeffs, theta_fine, phi_fine)

                per_range[dist][frame] = dict(
                    U_raw=U_raw, U_sd=U_sd, U_sf=U_sf,
                    td=td, pd=pd, coeffs=coeffs, hist=hist)

                # save data files
                ddir = os.path.join(itype_dir, dist, "data")
                os.makedirs(ddir, exist_ok=True)
                np.save(os.path.join(ddir, f"U_{frame}.npy"),      U_raw)
                np.save(os.path.join(ddir, f"coeffs_{frame}.npy"), coeffs)
                np.save(os.path.join(ddir, f"recon_{frame}.npy"),  U_sf)
                if hist is not None:
                    np.save(os.path.join(ddir, f"hist_{frame}.npy"), hist)

        # ── Build U_total = U_nuc + U_aa for each dist range ────────────────
        # (Buchete Eq. 4: sum over the same angular grid)
        for dist in DIST_BINS:
            d_nuc = per_range[dist].get("nuc")
            d_aa  = per_range[dist].get("aa")
            if d_nuc is None and d_aa is None:
                continue

            # Use whichever frames exist; if both, add them
            if d_nuc is not None and d_aa is not None:
                U_tot_raw = d_nuc["U_raw"] + d_aa["U_raw"]
                U_tot_sd  = d_nuc["U_sd"]  + d_aa["U_sd"]
                U_tot_sf  = d_nuc["U_sf"]  + d_aa["U_sf"]
                td, pd    = d_nuc["td"],    d_nuc["pd"]
            elif d_nuc is not None:
                U_tot_raw, U_tot_sd, U_tot_sf = d_nuc["U_raw"], d_nuc["U_sd"], d_nuc["U_sf"]
                td, pd = d_nuc["td"], d_nuc["pd"]
            else:
                U_tot_raw, U_tot_sd, U_tot_sf = d_aa["U_raw"], d_aa["U_sd"], d_aa["U_sf"]
                td, pd = d_aa["td"], d_aa["pd"]

            per_range[dist]["total"] = dict(
                U_raw=U_tot_raw, U_sd=U_tot_sd, U_sf=U_tot_sf, td=td, pd=pd)

            # save total
            ddir = os.path.join(itype_dir, dist, "data")
            os.makedirs(ddir, exist_ok=True)
            np.save(os.path.join(ddir, "U_total.npy"),     U_tot_raw)
            np.save(os.path.join(ddir, "recon_total.npy"), U_tot_sf)

        # ── Per-range individual plots ───────────────────────────────────────
        for dist in DIST_BINS:
            pdir = os.path.join(itype_dir, dist, "plots")
            os.makedirs(pdir, exist_ok=True)
            rng_label = f"{label}  {dist} ({DIST_RANGE[dist]})"

            for frame in FRAMES:
                d = per_range[dist].get(frame)
                if d is None: continue
                make_per_range_plots(pdir, frame, d["U_sf"],
                                     d["td"], d["pd"], d.get("hist"), rng_label)

            # also generate total plots
            dt = per_range[dist].get("total")
            if dt:
                make_per_range_plots(pdir, "total", dt["U_sf"],
                                     dt["td"], dt["pd"], None,
                                     rng_label + "  [U_total = U_nuc+U_aa]")

        # ── 3×3 Buchete combined figure using U_total ────────────────────────
        # rows = short / medium / long
        # cols = raw U_total | SHS 12×24 | SHS 96×192
        has_any = any(per_range[d].get("total") for d in DIST_BINS)
        if not has_any:
            continue

        # FIX 2 + FIX 3: clean suptitle, no Buchete formula text, no overlap
        fig = plt.figure(figsize=(13, 10))
        fig.suptitle(f"{aa} – {nuc}  ({itype})", fontsize=10,
                     fontweight="bold", y=0.995)

        # Column headers (kept compact, fontsize=7 to avoid overlap)
        for c, hdr in enumerate(["Raw  U_total",
                                   "SHS  12×24",
                                   "SHS  96×192"]):
            fig.text(0.19 + c*0.27, 0.975, hdr, ha="center",
                     fontsize=7, fontweight="bold")

        for r_idx, dist in enumerate(DIST_BINS):
            dt = per_range[dist].get("total")
            if dt is None:
                # blank row
                for c in range(3):
                    ax = fig.add_subplot(3, 3, r_idx*3 + c + 1, projection="mollweide")
                    ax.set_axis_off()
                continue

            # FIX 4 + Correction 3: Global row scale using 95th-percentile for smoothness
            vmax = max(sym_clim_p95(dt["U_raw"]),
                       sym_clim_p95(dt["U_sd"]),
                       sym_clim_p95(dt["U_sf"]))

            for c_idx in range(3):
                ax = fig.add_subplot(3, 3, r_idx*3+c_idx+1,
                                     projection="mollweide")
                
                if c_idx == 0:   Up,tv,pv = dt["U_raw"], dt["td"], dt["pd"]
                elif c_idx == 1: Up,tv,pv = dt["U_sd"],  dt["td"], dt["pd"]
                else:            Up,tv,pv = dt["U_sf"],  theta_fine, phi_fine

                if c_idx == 0:
                    ax.set_ylabel(f"{dist}  ({DIST_RANGE[dist]})",
                                  fontsize=8, labelpad=4)
                sm = moll_pmf(ax, Up, tv, pv, vmax)

            # per-row colorbar
            v = round(vmax, 3)
            y0   = 0.675 - r_idx*0.315
            cax  = fig.add_axes([0.925, y0, 0.013, 0.21])
            cbar = fig.colorbar(
                cm.ScalarMappable(cmap=CMAP_PMF,
                                  norm=mcolors.Normalize(-vmax,vmax)),
                cax=cax)
            cbar.set_ticks([-v, 0, v])
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label("U (kBT)", fontsize=7)

        # FIX 2: prevent text overlap – tighter layout with top margin
        fig.subplots_adjust(wspace=0.04, hspace=0.12,
                            left=0.10, right=0.91, top=0.93, bottom=0.02)
        out = os.path.join(itype_dir, "raw_vs_shs.pdf")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    ✓ raw_vs_shs.pdf  (3×3, U_total = U_nuc + U_aa)")
        print(f"    ✓ per-range plots in short/ medium/ long/")

    print(f"\n[✓] B4 complete")
    print(f"    results/<AA>/<NUC>/<ITYPE>/")
    print(f"      ├── raw_vs_shs.pdf  (3×3 Buchete Fig4 style, Eq.4 U_total)")
    print(f"      └── <range>/data/  +  <range>/plots/")


if __name__ == "__main__":
    main()