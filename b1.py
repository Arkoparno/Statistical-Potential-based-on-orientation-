

import pandas as pd
import numpy as np
import os
from itertools import product

INPUT  = "interaction_orientations.csv"
OUTDIR = "b1_histograms"
os.makedirs(OUTDIR, exist_ok=True)

# ── Grid definitions ──────────────────────────────────────────────────────────
GRIDS = {
    "12x24": (12, 24),   # exact Buchete grid
    "36x36": (36, 36),   # higher-resolution option
}

# ── Distance ranges (Å) — adapted for coarse-grained protein–DNA ────────────
# CG contacts: SC–B ~4-12 Å, SC–P ~5-15 Å, BB–P ~6-18 Å at COM level.
DIST_BINS = {
    "short":  (2.0,  8.0),   # direct contact / van der Waals / stacking
    "medium": (8.0, 15.0),   # H-bond / close electrostatic
    "long":   (15.0, 25.0),  # long-range electrostatic (Lys/Arg–phosphate)
}

# ── Residue / nucleotide types ────────────────────────────────────────────────
AA_TYPES = [
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
]
NUC_TYPES  = ["DA", "DT", "DG", "DC"]          # DNA
# RNA would be: ["A", "U", "G", "C"]            # extend if needed

INTERACTION_TYPES = ["BB-B", "BB-S", "BB-P", "SC-B", "SC-S", "SC-P"]

FRAMES = {
    "nuc": ("theta_nuc", "phi_nuc"),
    "aa":  ("theta_aa",  "phi_aa"),
}

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"[i] Reading {INPUT} …")
df = pd.read_csv(INPUT)
print(f"[i] {len(df):,} total pairs loaded")

# Drop rows where any angle is NaN
df = df.dropna(subset=["theta_nuc", "phi_nuc", "theta_aa", "phi_aa"])
print(f"[i] {len(df):,} pairs after dropping NaN angles\n")

# Fast lookup: pre-index the dataframe
df["aa_type"]  = df["aa_type"].str.strip().str.upper()
df["nuc_type"] = df["nuc_type"].str.strip().str.upper()


# ── Vectorised histogram builder ──────────────────────────────────────────────
def build_counts(theta_deg, phi_deg, n_theta, n_phi):
    """
    2D count histogram on an equiangular (theta, phi) grid.
    theta_deg : 1D array in [0, 180] degrees
    phi_deg   : 1D array in [0, 360) degrees
    Returns integer count array of shape (n_theta, n_phi).
    """
    dtheta = 180.0 / n_theta
    dphi   = 360.0 / n_phi

    ti = np.clip((theta_deg / dtheta).astype(int), 0, n_theta - 1)
    pi = np.clip((phi_deg   / dphi  ).astype(int), 0, n_phi   - 1)

    flat_idx    = ti * n_phi + pi
    counts_flat = np.bincount(flat_idx, minlength=n_theta * n_phi)
    return counts_flat.reshape(n_theta, n_phi)


# ── Main loop ─────────────────────────────────────────────────────────────────
total_files = 0
skipped     = 0

for grid_name, (n_theta, n_phi) in GRIDS.items():
    for dist_name, (rmin, rmax) in DIST_BINS.items():

        # Distance filter — do once per (grid, dist) combination
        mask_r = (df["distance"] >= rmin) & (df["distance"] < rmax)
        df_r   = df[mask_r]

        if len(df_r) == 0:
            print(f"[!] No data: {grid_name} | {dist_name}")
            continue

        for aa, nuc, itype in product(AA_TYPES, NUC_TYPES, INTERACTION_TYPES):

            # Filter to this specific (aa, nuc, itype) combination
            mask = (
                (df_r["aa_type"]         == aa)   &
                (df_r["nuc_type"]        == nuc)  &
                (df_r["interaction_type"]== itype)
            )
            df_i = df_r[mask]

            if len(df_i) == 0:
                skipped += 1
                continue

            tag = f"{grid_name}_{dist_name}_{aa}_{nuc}_{itype}"

            for frame_name, (theta_col, phi_col) in FRAMES.items():
                theta = df_i[theta_col].values
                phi   = df_i[phi_col].values

                counts = build_counts(theta, phi, n_theta, n_phi)
                N      = int(counts.sum())

                np.save(f"{OUTDIR}/{tag}_{frame_name}_counts.npy", counts)
                np.save(f"{OUTDIR}/{tag}_{frame_name}_N.npy", np.array([N]))
                total_files += 1

        print(f"[OK] {grid_name} | {dist_name}  done")

print(f"\n[✓] B1 complete")
print(f"    Histograms saved : {total_files} files  → {OUTDIR}/")
print(f"    Empty bins skipped (no data): {skipped} combinations")
print(f"    File naming: <grid>_<dist>_<aa>_<nuc>_<itype>_<frame>_counts.npy")