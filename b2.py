

import numpy as np
import os

INPUT  = "b1_histograms"
OUTDIR = "b2_potentials"
os.makedirs(OUTDIR, exist_ok=True)

kBT   = 1.0
SIGMA = 1.0 / 50.0
EPS   = 1e-12       # Correction 2: safer numerical floor

def sin_weights(n_theta, n_phi):
    """Solid-angle weights w[i,j] = sin(theta_i).
    Correction 1: equiangular bins near poles have smaller area
    on the sphere → divide counts by sin(theta) before normalising.
    """
    dth   = np.pi / n_theta
    theta = np.arange(n_theta) * dth + dth / 2.0   # cell-centred
    w     = np.sin(theta)                            # (n_theta,)
    return np.maximum(w[:, None] * np.ones(n_phi), EPS)  # (n_theta, n_phi)

AA_TYPES = [
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
]
NUC_TYPES         = ["DA", "DT", "DG", "DC"]
INTERACTION_TYPES = ["BB-B", "BB-S", "BB-P", "SC-B", "SC-S", "SC-P"]
GRIDS             = ["12x24", "36x36"]
DIST_BINS         = ["short", "medium", "long"]   # 2-8, 8-15, 15-25 Å
FRAMES            = ["nuc", "aa"]

total_saved = 0
total_skip  = 0

for grid in GRIDS:
    for dist in DIST_BINS:
        for nuc in NUC_TYPES:
            for itype in INTERACTION_TYPES:
                for frame in FRAMES:

                    # ── Build reference: sum counts over ALL amino acid types ──
                    counts_ref  = None
                    N_ref_total = 0.0
                    aa_data     = {}   # aa → (counts, N)

                    for aa in AA_TYPES:
                        tag      = f"{grid}_{dist}_{aa}_{nuc}_{itype}_{frame}"
                        cnt_path = os.path.join(INPUT, f"{tag}_counts.npy")
                        n_path   = os.path.join(INPUT, f"{tag}_N.npy")

                        if not os.path.exists(cnt_path):
                            continue

                        counts = np.load(cnt_path).astype(float)
                        N      = float(np.load(n_path)[0])

                        aa_data[aa] = (counts, N)

                        if counts_ref is None:
                            counts_ref = np.zeros_like(counts)

                        counts_ref  += counts
                        N_ref_total += N

                    if counts_ref is None or N_ref_total == 0:
                        total_skip += 1
                        continue

                    # ── Correction 1: area-corrected reference probability ──
                    nt, np_ = counts_ref.shape
                    W       = sin_weights(nt, np_)      # (nt, np_)

                    P_ref_w = counts_ref / W            # area-corrected counts
                    P_ref   = P_ref_w / (P_ref_w.sum() + EPS)  # normalise → sums to 1

                    # ── Boltzmann inversion per amino acid ────────────────────
                    for aa, (counts, N) in aa_data.items():
                        if N == 0:
                            continue

                        # Correction 1: area-corrected observed probability
                        P_obs_w = counts / W
                        P_obs   = P_obs_w / (P_obs_w.sum() + EPS)

                        # Sippl sparse-data correction
                        # N stays as raw count (statistical weight, not area)
                        P_corr = (N * P_obs + SIGMA * N_ref_total * P_ref) / \
                                 (N         + SIGMA * N_ref_total)

                        # Correction 2: clip AFTER Sippl combination, not before
                        P_corr = np.clip(P_corr, EPS, None)
                        P_ref_safe = np.clip(P_ref, EPS, None)

                        # Boltzmann inversion
                        U = -kBT * np.log(P_corr / P_ref_safe)

                        tag = f"{grid}_{dist}_{aa}_{nuc}_{itype}_{frame}"
                        np.save(os.path.join(OUTDIR, f"{tag}_U.npy"), U)
                        total_saved += 1

        print(f"[OK] {grid} | {dist} done")

print(f"\n[✓] B2 complete")
print(f"    PMFs saved : {total_saved}  → {OUTDIR}/")
print(f"    Skipped (no data): {total_skip} combinations")
print(f"    Units: kBT  |  Reference: averaged over all 20 amino acid types")