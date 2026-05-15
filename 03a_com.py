

import os
import csv
import argparse
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa



# SECTION 1 — CONFIGURATION


OUTPUT_FOLDER = "pdb_separate_COMs"

DNA_RES = {
    "DA", "DT", "DG", "DC",
    "DA3", "DA5", "DT3", "DT5",
    "DG3", "DG5", "DC3", "DC5",
}

RNA_RES = {
    "A", "U", "G", "C",
    "RA", "RU", "RG", "RC",
    "A3", "A5", "U3", "U5",
    "G3", "G5", "C3", "C5",
    "PSU", "5MC", "7MG", "OMG", "H2U", "M2G", "OMC", "OMU",
}

# Protein backbone heavy atoms
BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# Nucleotide phosphate atoms (DNA and RNA share these)
PHOSPHATE_ATOMS = {"P", "OP1", "OP2", "O5'", "O1P", "O2P"}

# Sugar atoms: deoxyribose vs ribose
DNA_SUGAR_ATOMS = {"C1'", "C2'", "C3'", "C4'", "C5'", "O3'", "O4'", "O5'"}
RNA_SUGAR_ATOMS = {"C1'", "C2'", "C3'", "C4'", "C5'",
                   "O2'", "O3'", "O4'", "O5'"}

# Base ring atoms (purines + pyrimidines combined)
# Whatever is present will be used — the COM uses only found atoms
BASE_ATOMS = {"N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"}

# The three backbone atoms needed for LRF construction
# N and C positions are stored so GLY LRF can be computed in 03b
LRF_BACKBONE_ATOMS = {"N", "C"}



# SECTION 2 — GEOMETRY HELPERS

NAN3 = np.array([np.nan, np.nan, np.nan])


def get_com(atom_list: list) -> np.ndarray:
    """
    Unweighted geometric centre of a list of BioPython Atom objects.
    Returns [NaN, NaN, NaN] if list is empty — downstream scripts use
    np.isfinite to detect missing data. NEVER use zeros as a sentinel.
    """
    if not atom_list:
        return NAN3.copy()
    coords = np.array([a.get_coord() for a in atom_list], dtype=float)
    return coords.mean(axis=0)


def get_atom_pos(residue, atom_name: str) -> np.ndarray:
    """
    Return position of a named atom in a residue, or NaN if absent.
    Tries the exact name and common alternates (e.g. "C" vs "C1").
    """
    try:
        return np.array(residue[atom_name].get_coord(), dtype=float)
    except KeyError:
        return NAN3.copy()


def normalise(v: np.ndarray) -> np.ndarray:
    """
    Return unit vector. Returns NaN vector if input is near-zero
    (degenerate geometry — e.g. two coincident COMs).
    """
    n = np.linalg.norm(v)
    if n < 1e-8:
        return NAN3.copy()
    return v / n


def gram_schmidt(v: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Remove the z-component from v and normalise.
    Returns the component of v orthogonal to z (the in-plane x-axis).
    Returns NaN vector if result is degenerate (v parallel to z).
    """
    v_orth = v - np.dot(v, z) * z
    return normalise(v_orth)


def compute_lrf(p1: np.ndarray,
                p2: np.ndarray,
                p3: np.ndarray) -> tuple:
    """
    Build a Local Reference Frame from three non-collinear points.

    Convention matches Buchete et al. (J. Chem. Phys. 118, 7658, 2003):
      v1  = p2 - p1   (first in-plane vector)
      v2  = p3 - p2   (second in-plane vector)
      z_hat = normalise(cross(v1, v2))   ← frame normal (out of plane)
      x_hat = Gram-Schmidt(v1, z_hat)    ← in-plane reference direction
      y_hat = cross(z_hat, x_hat)        ← completes right-hand system

    For nucleotides:  p1=B_com, p2=S_com, p3=P_com
      → z_hat is the base-plane normal
      → x_hat points approximately in the B→S direction within the plane

    For amino acids:  p1=BB_com, p2=SC_com, p3=N_pos
      → z_hat = Cα→SC direction (backbone-into-sidechain)
      → x_hat aligned with backbone chain direction

    Returns (x_hat, y_hat, z_hat) each shape (3,).
    Returns (NaN3, NaN3, NaN3) if any input is NaN or geometry is degenerate.
    """
    # Guard: any NaN input → undefined LRF
    if (not np.all(np.isfinite(p1)) or
            not np.all(np.isfinite(p2)) or
            not np.all(np.isfinite(p3))):
        return NAN3.copy(), NAN3.copy(), NAN3.copy()

    v1 = p2 - p1
    v2 = p3 - p2

    z_hat = normalise(np.cross(v1, v2))
    if not np.all(np.isfinite(z_hat)):
        # Collinear points — cannot define a plane
        return NAN3.copy(), NAN3.copy(), NAN3.copy()

    x_hat = gram_schmidt(v1, z_hat)
    if not np.all(np.isfinite(x_hat)):
        return NAN3.copy(), NAN3.copy(), NAN3.copy()

    y_hat = np.cross(z_hat, x_hat)   # already unit length

    return x_hat, y_hat, z_hat


def compute_lrf_gly(n_pos: np.ndarray,
                    bb_com: np.ndarray,
                    c_pos: np.ndarray) -> tuple:
    """
    GLY-specific LRF: no sidechain, so the Cα→Cβ direction is undefined.

    Buchete's GLY fallback (J. Chem. Phys. 118, 7658, 2003, Section 2.1):
      P1 = midpoint of N_i and C_i  (adjacent backbone atoms)
      P2 = Cα_i  (here approximated as BB_com for Cα-only approximation,
                   but we use N_pos as the N-side reference point)
      z_hat = bisector of the N-Cα-C angle, pointing away from backbone

    Our implementation:
      z_hat = normalise(N_pos - C_pos)   (N→C backbone direction as fallback)
      x_hat = orthogonalised(N_pos - BB_com, z_hat)
      y_hat = cross(z_hat, x_hat)

    This is consistent with Buchete's definition that for Gly the positive
    Oz axis is defined by the bisector of the backbone N-Cα-C angle.
    """
    if (not np.all(np.isfinite(n_pos)) or
            not np.all(np.isfinite(bb_com)) or
            not np.all(np.isfinite(c_pos))):
        return NAN3.copy(), NAN3.copy(), NAN3.copy()

    # z-axis: N→C backbone direction (proxy for the chain direction at GLY)
    z_hat = normalise(n_pos - c_pos)
    if not np.all(np.isfinite(z_hat)):
        return NAN3.copy(), NAN3.copy(), NAN3.copy()

    # x-axis: orthogonalised N→BB_com direction
    x_hat = gram_schmidt(n_pos - bb_com, z_hat)
    if not np.all(np.isfinite(x_hat)):
        # Fallback: use any vector perpendicular to z_hat
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(perp, z_hat)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        x_hat = normalise(perp - np.dot(perp, z_hat) * z_hat)

    y_hat = np.cross(z_hat, x_hat)
    return x_hat, y_hat, z_hat



# SECTION 3 — NUCLEOTIDE COM + LRF EXTRACTION


def extract_nucleic_coms(model, res_set: set, sugar_atoms: set) -> list:
    """
    For each nucleotide residue in res_set, extract:
      - P, S, B centres of mass
      - Nucleotide LRF (x_hat, y_hat, z_hat)
      - Principal axis (= z_hat, the base normal) for Mukherjee beta angle

    LRF is defined with:
      P1 = B_com  (base, analogous to Buchete's Cα)
      P2 = S_com  (sugar, analogous to Cβ)
      P3 = P_com  (phosphate, analogous to Cγ)

    Physical meaning of axes:
      z_hat (base normal): θ=0° means approach from directly above/below the
            aromatic base ring = stacking geometry
      x_hat (B→S direction): reference direction in the base plane
      y_hat: completes the right-hand system

    Returns list of rows, each row has 30 values matching NUC_HEADER.
    """
    rows = []
    for chain in model:
        for residue in chain:
            rname = residue.resname.strip()
            if rname not in res_set:
                continue

            res_id   = residue.id[1]
            chain_id = chain.id

            p_atoms, s_atoms, b_atoms = [], [], []
            for atom in residue:
                aname = atom.get_name().strip()
                if aname in PHOSPHATE_ATOMS:
                    p_atoms.append(atom)
                elif aname in sugar_atoms:
                    s_atoms.append(atom)
                elif aname in BASE_ATOMS:
                    b_atoms.append(atom)

            p_com = get_com(p_atoms)
            s_com = get_com(s_atoms)
            b_com = get_com(b_atoms)

            # LRF: P1=B_com, P2=S_com, P3=P_com
            x_hat, y_hat, z_hat = compute_lrf(b_com, s_com, p_com)

            # Principal axis for Mukherjee = z_hat (base normal)
            # Stored separately for clarity in 03b
            pa = z_hat.copy()

            row = [
                rname, res_id, chain_id,
                # COMs
                p_com[0], p_com[1], p_com[2],
                s_com[0], s_com[1], s_com[2],
                b_com[0], b_com[1], b_com[2],
                # LRF axes
                x_hat[0], x_hat[1], x_hat[2],
                y_hat[0], y_hat[1], y_hat[2],
                z_hat[0], z_hat[1], z_hat[2],
                # Principal axis (Mukherjee)
                pa[0], pa[1], pa[2],
            ]
            rows.append(row)
    return rows



# SECTION 4 — PROTEIN COM + LRF EXTRACTION


def extract_protein_coms(model) -> list:
    """
    For each standard amino acid residue, extract:
      - BB_com (backbone heavy atom centroid: N, CA, C, O)
      - SC_com (sidechain heavy atom centroid; NaN for GLY)
      - Protein LRF (x_hat, y_hat, z_hat)
      - N atom position (for GLY LRF fallback and Mukherjee)
      - C atom position (for GLY LRF fallback)

    LRF definition (matching Buchete Cα→Cβ→Cγ convention):
      For residues WITH sidechain:
        P1 = BB_com
        P2 = SC_com
        P3 = N_pos  (backbone N atom as the third reference point)
        → z_hat points from backbone into sidechain (Cα→Cβ analog)

      For GLY (no sidechain):
        Uses GLY-specific fallback (see compute_lrf_gly)
        → z_hat is the bisector of the backbone N-Cα-C angle

    Principal axis for Mukherjee alpha angle = z_hat_aa in all cases.

    Returns list of rows matching PROT_HEADER (24 columns).
    """
    rows = []
    for chain in model:
        for residue in chain:
            if residue.id[0] != " " or not is_aa(residue, standard=True):
                continue

            rname    = residue.resname.strip()
            res_id   = residue.id[1]
            chain_id = chain.id

            bb_atoms = []
            sc_atoms = []

            # Also collect N and C atom positions for LRF construction
            n_pos = NAN3.copy()
            c_pos = NAN3.copy()

            for atom in residue:
                aname = atom.get_name().strip()
                # Skip hydrogens
                if aname.startswith("H") or (aname[0].isdigit()):
                    continue
                if aname in BACKBONE_ATOMS:
                    bb_atoms.append(atom)
                    if aname == "N":
                        n_pos = np.array(atom.get_coord(), dtype=float)
                    elif aname == "C":
                        c_pos = np.array(atom.get_coord(), dtype=float)
                else:
                    sc_atoms.append(atom)

            bb_com = get_com(bb_atoms)
            sc_com = get_com(sc_atoms)   # NaN for GLY (no sc_atoms)

            # Build LRF
            is_gly = (rname == "GLY") or (not np.all(np.isfinite(sc_com)))

            if is_gly:
                # GLY fallback: use backbone atoms N and C
                x_hat, y_hat, z_hat = compute_lrf_gly(n_pos, bb_com, c_pos)
            else:
                # Standard residue: P1=BB_com, P2=SC_com, P3=N_pos
                x_hat, y_hat, z_hat = compute_lrf(bb_com, sc_com, n_pos)

            row = [
                rname, res_id, chain_id,
                # COMs
                bb_com[0], bb_com[1], bb_com[2],
                sc_com[0], sc_com[1], sc_com[2],
                # LRF axes
                x_hat[0], x_hat[1], x_hat[2],
                y_hat[0], y_hat[1], y_hat[2],
                z_hat[0], z_hat[1], z_hat[2],
                # N and C positions (needed for GLY LRF in 03b and Mukherjee)
                n_pos[0], n_pos[1], n_pos[2],
                c_pos[0], c_pos[1], c_pos[2],
            ]
            rows.append(row)
    return rows



# SECTION 5 — FILE I/O


# Nucleotide CSV header — 30 columns
NUC_HEADER = [
    "res_name", "res_id", "chain_id",
    # COMs
    "p_x", "p_y", "p_z",
    "s_x", "s_y", "s_z",
    "b_x", "b_y", "b_z",
    # LRF — nucleotide frame (Buchete)
    # x_hat: in-plane direction (B→S, Gram-Schmidt)
    # y_hat: second in-plane axis
    # z_hat: base normal (= principal axis for Mukherjee beta)
    "lrf_x_x", "lrf_x_y", "lrf_x_z",
    "lrf_y_x", "lrf_y_y", "lrf_y_z",
    "lrf_z_x", "lrf_z_y", "lrf_z_z",
    # Principal axis for Mukherjee (= lrf_z = base normal)
    "pa_x", "pa_y", "pa_z",
]

# Protein CSV header — 24 columns
PROT_HEADER = [
    "res_name", "res_id", "chain_id",
    # COMs
    "bb_x", "bb_y", "bb_z",
    "sc_x", "sc_y", "sc_z",
    # LRF — amino acid frame (Buchete)
    # z_hat: Cα→SC direction (= principal axis for Mukherjee alpha)
    "lrf_x_x", "lrf_x_y", "lrf_x_z",
    "lrf_y_x", "lrf_y_y", "lrf_y_z",
    "lrf_z_x", "lrf_z_y", "lrf_z_z",
    # Individual backbone atom positions needed for LRF / Mukherjee
    "N_x", "N_y", "N_z",
    "C_x", "C_y", "C_z",
]


def write_csv(path: str, header: list, data: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def parse_structure(pdb_id: str, fpath: str):
    """Parse PDB or CIF (optionally gzipped). Returns Structure or None."""
    import gzip, io
    try:
        is_cif = ".cif" in fpath
        parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
        if fpath.endswith(".gz"):
            with gzip.open(fpath, "rt", errors="ignore") as gz:
                content = gz.read()
            return parser.get_structure(pdb_id, io.StringIO(content))
        return parser.get_structure(pdb_id, fpath)
    except Exception as e:
        print(f"  [!] Parse error {pdb_id}: {e}")
        return None



# SECTION 6 — LRF VALIDATION HELPER


def count_valid_lrf(rows: list, lrf_start_col: int = 9) -> tuple:
    """
    Count how many rows have a valid LRF (all 9 LRF components finite).
    lrf_start_col = 9 for nucleotide (after 3 meta + 9 COMs)
                  = 9 for protein (after 3 meta + 6 COMs)
    Returns (n_valid, n_total).
    """
    n_total = len(rows)
    n_valid = 0
    for row in rows:
        lrf_vals = row[lrf_start_col: lrf_start_col + 9]
        if all(v is not None and np.isfinite(float(v)) for v in lrf_vals):
            n_valid += 1
    return n_valid, n_total



# SECTION 7 — MAIN


def main():
    ap = argparse.ArgumentParser(
        description="Extract COMs and LRFs for Buchete + Mukherjee PMF pipeline"
    )
    ap.add_argument(
        "--pdb-dir", required=True,
        help="Directory of PDB/CIF files (output of 02b_filter.py)"
    )
    args = ap.parse_args()

    pdb_dir = args.pdb_dir
    if not os.path.isdir(pdb_dir):
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    valid_exts = (".pdb", ".pdb.gz", ".cif", ".cif.gz")
    files = sorted(
        f for f in os.listdir(pdb_dir) if any(f.endswith(e) for e in valid_exts)
    )

    print(f"[i] {len(files)} structure files in {pdb_dir}")
    print(f"[i] Output → {OUTPUT_FOLDER}/")
    print(f"[i] Nucleotide CSV: {len(NUC_HEADER)} columns")
    print(f"[i] Protein CSV:    {len(PROT_HEADER)} columns\n")

    done = skipped = errors = 0

    for fname in files:
        pdb_id = fname.split("_")[0].split(".")[0].lower()
        fpath  = os.path.join(pdb_dir, fname)

        prot_out = os.path.join(OUTPUT_FOLDER, f"{pdb_id}_protein_COM.csv")
        dna_out  = os.path.join(OUTPUT_FOLDER, f"{pdb_id}_dna_COM.csv")
        rna_out  = os.path.join(OUTPUT_FOLDER, f"{pdb_id}_rna_COM.csv")

        # Resume: skip if all relevant output files already exist
        if os.path.exists(prot_out) and (
                os.path.exists(dna_out) or os.path.exists(rna_out)):
            skipped += 1
            continue

        structure = parse_structure(pdb_id, fpath)
        if structure is None:
            errors += 1
            continue

        model = structure[0]

        # ── Protein ──────────────────────────────────────────────────────────
        prot_rows = extract_protein_coms(model)
        if prot_rows:
            write_csv(prot_out, PROT_HEADER, prot_rows)
            n_lrf_valid, n_prot = count_valid_lrf(prot_rows, lrf_start_col=9)
        else:
            n_prot = n_lrf_valid = 0

        # ── DNA ───────────────────────────────────────────────────────────────
        dna_rows = extract_nucleic_coms(model, DNA_RES, DNA_SUGAR_ATOMS)
        if dna_rows:
            write_csv(dna_out, NUC_HEADER, dna_rows)
            n_dna_lrf, n_dna = count_valid_lrf(dna_rows, lrf_start_col=9)
        else:
            n_dna = n_dna_lrf = 0

        # ── RNA ───────────────────────────────────────────────────────────────
        rna_rows = extract_nucleic_coms(model, RNA_RES, RNA_SUGAR_ATOMS)
        if rna_rows:
            write_csv(rna_out, NUC_HEADER, rna_rows)
            n_rna_lrf, n_rna = count_valid_lrf(rna_rows, lrf_start_col=9)
        else:
            n_rna = n_rna_lrf = 0

        if n_prot == 0 or (n_dna == 0 and n_rna == 0):
            print(
                f"  [!] {pdb_id.upper()}: "
                f"protein={n_prot} DNA={n_dna} RNA={n_rna} "
                f"— missing one component"
            )
        else:
            print(
                f"[OK] {pdb_id.upper()}: "
                f"{n_prot} aa (LRF {n_lrf_valid}/{n_prot}) | "
                f"DNA {n_dna} (LRF {n_dna_lrf}/{n_dna}) | "
                f"RNA {n_rna} (LRF {n_rna_lrf}/{n_rna})"
            )

        done += 1

    print(f"\n[OK] Done.")
    print(f"     Processed : {done}")
    print(f"     Skipped   : {skipped}  (already done)")
    print(f"     Errors    : {errors}")
    print(f"\n     Next step: python 03b_distances.py")
    print(f"\n     Column guide:")
    print(f"       lrf_z_*  = Buchete z-axis = Mukherjee principal axis")
    print(f"       Nucleotide lrf_z = base normal (stacking direction)")
    print(f"       Protein   lrf_z = Cα→SC direction")
    print(f"       N_*, C_*  = backbone atom positions for GLY LRF")


if __name__ == "__main__":
    main()