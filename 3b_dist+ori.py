import os
import csv
import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
MAX_DIST = 100   # Å — Buchete long-range cutoff; pairs beyond this are skipped

# Maps interaction_type → (aa COM cols, nuc COM cols)
INTERACTION_MAP = {
    "BB-B": (("bb_x", "bb_y", "bb_z"), ("b_x", "b_y", "b_z")),
    "BB-S": (("bb_x", "bb_y", "bb_z"), ("s_x", "s_y", "s_z")),
    "BB-P": (("bb_x", "bb_y", "bb_z"), ("p_x", "p_y", "p_z")),
    "SC-B": (("sc_x", "sc_y", "sc_z"), ("b_x", "b_y", "b_z")),
    "SC-S": (("sc_x", "sc_y", "sc_z"), ("s_x", "s_y", "s_z")),
    "SC-P": (("sc_x", "sc_y", "sc_z"), ("p_x", "p_y", "p_z")),
}

# ── Geometry helpers ─────────────────────────────────────────────────────────

def to_local_frame(vec, x_hat, y_hat, z_hat):
    """Project vec into the local frame defined by (x_hat, y_hat, z_hat)."""
    R_mat = np.vstack([x_hat, y_hat, z_hat]).T   # columns = basis vectors
    return R_mat.T @ vec                           # = R_mat^{-1} @ vec


def spherical_angles(v):
    """
    Return (theta_deg, phi_deg) of v in the local frame.
    theta in [0, 180], phi in [0, 360).
    Returns (None, None) for degenerate vectors.
    """
    r = np.linalg.norm(v)
    if r < 1e-6:
        return None, None
    theta = np.degrees(np.arccos(np.clip(v[2] / r, -1.0, 1.0)))
    phi   = np.degrees(np.arctan2(v[1], v[0])) % 360.0
    return theta, phi


def compute_alpha_beta(z_aa, z_nuc, R_hat):
    """
    Mukherjee angles (degrees):
      alpha = angle between aa principal axis and R
      beta  = angle between nuc principal axis and -R (i.e., facing toward aa)
    """
    alpha = np.degrees(np.arccos(np.clip(np.dot(z_aa,   R_hat), -1.0, 1.0)))
    beta  = np.degrees(np.arccos(np.clip(np.dot(z_nuc, -R_hat), -1.0, 1.0)))
    return alpha, beta


def compute_chi(z_aa, z_nuc, R_hat):
    """
    Signed torsion angle (degrees) between the two principal axes
    projected onto the plane perpendicular to R.
    Range (-180, +180].
    """
    z1 = z_aa  - np.dot(z_aa,  R_hat) * R_hat
    z2 = z_nuc - np.dot(z_nuc, R_hat) * R_hat
    if np.linalg.norm(z1) < 1e-6 or np.linalg.norm(z2) < 1e-6:
        return np.nan
    z1 /= np.linalg.norm(z1)
    z2 /= np.linalg.norm(z2)
    cos_chi = np.clip(np.dot(z1, z2), -1.0, 1.0)
    chi = np.arccos(cos_chi)
    sign = np.sign(np.dot(np.cross(z1, z2), R_hat))
    return np.degrees(chi * sign)


# ── Core pair processor ──────────────────────────────────────────────────────

def process_pair(pdb_id, aa_row, nuc_row, interaction_type, nuc_class,
                 aa_com, nuc_com):
    """
    Compute distance + Buchete angles (nuc frame, aa frame) +
    Mukherjee angles (alpha, beta, chi) for one AA–nucleotide COM pair.

    aa_com / nuc_com are passed explicitly so all 6 interaction types
    can share this function without any hardcoded column reads.
    """
    # ── LRF vectors ──────────────────────────────────────────────────────────
    x_aa  = np.array([aa_row["lrf_x_x"],  aa_row["lrf_x_y"],  aa_row["lrf_x_z"]])
    y_aa  = np.array([aa_row["lrf_y_x"],  aa_row["lrf_y_y"],  aa_row["lrf_y_z"]])
    z_aa  = np.array([aa_row["lrf_z_x"],  aa_row["lrf_z_y"],  aa_row["lrf_z_z"]])

    x_nuc = np.array([nuc_row["lrf_x_x"], nuc_row["lrf_x_y"], nuc_row["lrf_x_z"]])
    y_nuc = np.array([nuc_row["lrf_y_x"], nuc_row["lrf_y_y"], nuc_row["lrf_y_z"]])
    z_nuc = np.array([nuc_row["lrf_z_x"], nuc_row["lrf_z_y"], nuc_row["lrf_z_z"]])

    # Skip residues whose LRF could not be defined (NaN from 03a)
    if not (np.all(np.isfinite(x_aa)) and np.all(np.isfinite(x_nuc))):
        return None

    # ── Distance ─────────────────────────────────────────────────────────────
    R = nuc_com - aa_com      # vector from aa COM → nuc COM
    r = np.linalg.norm(R)
    if r < 1e-6:
        return None
    R_hat = R / r

    # ── Buchete: nucleotide frame ─────────────────────────────────────────────
    # Project R (aa→nuc) into the nuc LRF → "where is aa as seen by nuc?"
    v_nuc = to_local_frame(R, x_nuc, y_nuc, z_nuc)
    theta_nuc, phi_nuc = spherical_angles(v_nuc)

    # ── Buchete: amino acid frame ─────────────────────────────────────────────
    # Project -R (nuc→aa direction) into aa LRF → "where is nuc as seen by aa?"
    v_aa = to_local_frame(-R, x_aa, y_aa, z_aa)
    theta_aa, phi_aa = spherical_angles(v_aa)

    # ── Mukherjee ────────────────────────────────────────────────────────────
    alpha, beta = compute_alpha_beta(z_aa, z_nuc, R_hat)
    chi = compute_chi(z_aa, z_nuc, R_hat)

    return [
        pdb_id,
        aa_row["res_name"],  aa_row["res_id"],  aa_row["chain_id"],
        nuc_row["res_name"], nuc_row["res_id"], nuc_row["chain_id"],
        nuc_class, interaction_type,
        round(r, 4),
        round(theta_nuc, 4) if theta_nuc is not None else np.nan,
        round(phi_nuc,   4) if phi_nuc   is not None else np.nan,
        round(theta_aa,  4) if theta_aa  is not None else np.nan,
        round(phi_aa,    4) if phi_aa    is not None else np.nan,
        round(alpha, 4), round(beta, 4),
        round(chi, 4) if not np.isnan(chi) else np.nan,
    ]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    input_dir   = "pdb_separate_COMs"
    output_file = "interaction_orientations.csv"

    header = [
        "pdb_id",
        "aa_type",  "aa_resid",  "aa_chain",
        "nuc_type", "nuc_resid", "nuc_chain",
        "nuc_class", "interaction_type",
        "distance",
        "theta_nuc", "phi_nuc",
        "theta_aa",  "phi_aa",
        "alpha", "beta", "chi",
    ]

    # Only pick files that are actual COM CSVs (avoids .DS_Store etc.)
    all_files = [f for f in os.listdir(input_dir) if f.endswith("_COM.csv")]
    pdb_ids   = sorted(set(f.split("_")[0] for f in all_files))
    print(f"[i] {len(pdb_ids)} PDB structures found in {input_dir}/")

    total_rows = 0

    with open(output_file, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)

        for pdb_id in pdb_ids:
            prot_path = os.path.join(input_dir, f"{pdb_id}_protein_COM.csv")
            dna_path  = os.path.join(input_dir, f"{pdb_id}_dna_COM.csv")
            rna_path  = os.path.join(input_dir, f"{pdb_id}_rna_COM.csv")

            if not os.path.exists(prot_path):
                continue

            prot_df = pd.read_csv(prot_path)

            # Load whichever nucleic acid is present (DNA preferred; RNA fallback)
            if os.path.exists(dna_path):
                nuc_df, nuc_class = pd.read_csv(dna_path), "DNA"
            elif os.path.exists(rna_path):
                nuc_df, nuc_class = pd.read_csv(rna_path), "RNA"
            else:
                continue

            pdb_rows = 0

            for _, aa_row in prot_df.iterrows():
                for _, nuc_row in nuc_df.iterrows():

                    for itype, (aa_cols, nuc_cols) in INTERACTION_MAP.items():

                        # Extract COMs for this interaction type
                        aa_com  = np.array([aa_row[c]  for c in aa_cols],  dtype=float)
                        nuc_com = np.array([nuc_row[c] for c in nuc_cols], dtype=float)

                        # Skip if COM is NaN (e.g. GLY has no SC)
                        if not (np.all(np.isfinite(aa_com)) and
                                np.all(np.isfinite(nuc_com))):
                            continue

                        # ── Distance cutoff (applied before heavy geometry) ──
                        r_quick = np.linalg.norm(nuc_com - aa_com)
                        if r_quick > MAX_DIST:
                            continue

                        result = process_pair(
                            pdb_id, aa_row, nuc_row,
                            itype, nuc_class,
                            aa_com, nuc_com,
                        )

                        if result is not None:
                            writer.writerow(result)
                            pdb_rows += 1

            total_rows += pdb_rows
            print(f"[OK] {pdb_id.upper():8s}  {pdb_rows:>7,} interactions written")

    print(f"\n[✓] Done — {total_rows:,} rows saved → {output_file}")
    print(f"    All angles in DEGREES  |  phi in [0, 360)  |  r ≤ {MAX_DIST} Å")


if __name__ == "__main__":
    main()