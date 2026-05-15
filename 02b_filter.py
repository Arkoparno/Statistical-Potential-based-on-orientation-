#python 02b_filter.py --identity 30

import os
import re
import shutil
import argparse
import csv
import pandas as pd
import matplotlib
matplotlib.use("Agg")              # non-interactive backend — safe for pipeline
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────────────────────────────────────────────
PDB_DIR         = "pdb_files"
MMSEQS_BASE     = "mmseqs2_results"
CHAIN_INVENTORY = "chain_inventory.csv"
COUNTS_CSV      = "protein_dna_counts.csv"
GROUPS_CSV      = "protein_dna_groups.csv"
PLOT_PATH       = "plots/redundancy_plot.png"

# Amino acid group membership
AA_GROUPS = {
    "Aliphatic (%)":       {"G","A","V","L","I","P","M"},
    "Aromatic (%)":        {"F","Y","W"},
    "Polar Uncharged (%)": {"S","T","C","N","Q"},
    "Acidic (%)":          {"D","E"},
    "Basic (%)":           {"R","K"},
    "Histidine (%)":       {"H"},
}
# ──────────────────────────────────────────────────────────────────────────────


def get_pdb_candidates(pdb_id: str) -> list:
    """
    Return all possible filenames for a given pdb_id,
    matching the naming convention used in 00_download.py.
    """
    pid = pdb_id.lower()
    return [
        f"{pid}_assembly1.pdb.gz",
        f"{pid}_assembly1.cif.gz",
        f"{pid}.pdb",
        f"{pid}.cif",
        f"{pid}.pdb.gz",
        f"{pid}.cif.gz",
    ]


def extract_pdb_ids_from_fasta(fasta_path: str) -> set:
    """
    Extract unique PDB IDs from FASTA headers.
    Handles both 4-char (1abc_A) and 8-char (abcd1234_A) IDs.
    Header format produced by 01_fasta_extract.py: >pdbid_chainid
    """
    pdb_ids = set()
    # Match 4 or 8 alphanumeric characters before underscore
    pattern = re.compile(r'^>([a-zA-Z0-9]{4,8})_')

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith(">"):
                continue
            m = pattern.match(line)
            if m:
                pdb_ids.add(m.group(1).lower())
            else:
                # Fallback: take everything before first underscore
                header = line[1:]
                part   = header.split("_")[0].lower()
                if 4 <= len(part) <= 8:
                    pdb_ids.add(part)

    return pdb_ids


def copy_pdb_files(pdb_ids: set, pdb_dir: str, output_dir: str) -> tuple:
    """
    Copy matching PDB files to output_dir.
    Returns (copied_ids, missing_ids).
    """
    os.makedirs(output_dir, exist_ok=True)
    copied  = set()
    missing = set()

    for pdb_id in sorted(pdb_ids):
        found = False
        for fname in get_pdb_candidates(pdb_id):
            src = os.path.join(pdb_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.exists(src):
                if not os.path.exists(dst):      # resume-safe: skip if already copied
                    shutil.copy2(src, dst)
                copied.add(pdb_id)
                found = True
                break
        if not found:
            missing.add(pdb_id)

    return copied, missing


def compute_aa_composition(sequences: list) -> dict:
    """
    Given a list of one-letter protein sequences,
    compute the percentage of residues falling in each AA group.
    Returns dict {group_name: percentage}.
    """
    if not sequences:
        return {g: 0.0 for g in AA_GROUPS}

    full_seq  = "".join(sequences)
    total     = len(full_seq)
    if total == 0:
        return {g: 0.0 for g in AA_GROUPS}

    result = {}
    for group, members in AA_GROUPS.items():
        count = sum(1 for aa in full_seq if aa in members)
        result[group] = round(100.0 * count / total, 3)

    return result


def build_stats_across_all_identities(
    mmseqs_base: str,
    chain_inventory: str
) -> tuple:
    """
    For each identity 1–100%:
      - Read the non-redundant FASTA
      - Count representative sequences
      - Look up sequences in chain_inventory
      - Compute AA group percentages

    Returns two DataFrames:
      counts_df : Identity_Percentage, Fasta_Count
      groups_df : Identity(%), Aliphatic(%), Aromatic(%), ...
    """
    # Load chain inventory — we need sequences keyed by pdb_id+chain_id
    inv_df = pd.read_csv(chain_inventory, dtype=str)
    # Build lookup: (pdb_id, chain_id) → sequence
    seq_lookup = {}
    for _, row in inv_df.iterrows():
        if row["chain_type"] in ("protein", "mixed") and pd.notna(row["sequence"]):
            key = (row["pdb_id"].lower(), row["chain_id"])
            seq_lookup[key] = row["sequence"]

    counts_rows = []
    groups_rows = []

    print("[i] Building stats across all 100 identity levels...")

    for i in range(1, 101):
        fasta_path = os.path.join(mmseqs_base, f"{i}pc",
                                  f"non_redundant_{i}pc.fasta")

        if not os.path.exists(fasta_path):
            # MMseqs2 run not completed for this level — fill with NaN
            counts_rows.append({
                "Identity_Percentage": i,
                "Fasta_Count":         None
            })
            groups_rows.append({"Identity(%)": i,
                                 **{g: None for g in AA_GROUPS}})
            continue

        # Parse FASTA headers — format is >pdbid_chainid
        sequences  = []
        fasta_count = 0
        header_pattern = re.compile(r'^>([a-zA-Z0-9]{4,8})_([A-Za-z0-9]+)')

        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    fasta_count += 1
                    m = header_pattern.match(line)
                    if m:
                        pid  = m.group(1).lower()
                        cid  = m.group(2)
                        seq  = seq_lookup.get((pid, cid), "")
                        if seq:
                            sequences.append(seq)

        aa_comp = compute_aa_composition(sequences)

        counts_rows.append({
            "Identity_Percentage": i,
            "Fasta_Count":         fasta_count
        })
        groups_rows.append({
            "Identity(%)": i,
            **aa_comp
        })

        if i % 10 == 0:
            print(f"  [i] {i}%: {fasta_count} representatives")

    counts_df = pd.DataFrame(counts_rows)
    groups_df = pd.DataFrame(groups_rows)
    return counts_df, groups_df


def make_plot(counts_df: pd.DataFrame, groups_df: pd.DataFrame,
              plot_path: str) -> None:
    """
    Reproduce the original combined redundancy plot.
    Left axis: FASTA count. Right axis: AA group percentages.
    """
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    merged = pd.merge(
        counts_df, groups_df,
        left_on="Identity_Percentage", right_on="Identity(%)"
    ).dropna(subset=["Fasta_Count"])

    colors = {
        "Aliphatic (%)":       "purple",
        "Aromatic (%)":        "orange",
        "Polar Uncharged (%)": "green",
        "Acidic (%)":          "red",
        "Basic (%)":           "blue",
        "Histidine (%)":       "brown",
    }

    fig, ax1 = plt.subplots(figsize=(13, 6))
    plt.rcParams["font.family"] = "DejaVu Serif"

    # Left axis — raw FASTA count
    ax1.plot(merged["Identity_Percentage"], merged["Fasta_Count"],
             color="black", lw=2.5, label="FASTA Count (left axis)")
    ax1.set_xlabel("Sequence Identity (%)", fontsize=12)
    ax1.set_ylabel("Representative Sequence Count", fontsize=11, color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Right axis — AA percentages
    ax2 = ax1.twinx()
    for group, color in colors.items():
        if group in merged.columns:
            ax2.plot(merged["Identity_Percentage"], merged[group],
                     color=color, linestyle="--", lw=1.5, label=group)
    ax2.set_ylabel("Amino Acid Group (%)", fontsize=11)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=9, framealpha=0.85)

    plt.title("Representative Sequences and AA Composition vs Sequence Identity",
              fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[✓] Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter PDB files to non-redundant set at chosen identity.")
    parser.add_argument("--identity", type=int, required=True,
                        help="Sequence identity threshold (1–100)")
    args = parser.parse_args()

    i = args.identity
    assert 1 <= i <= 100, "Identity must be between 1 and 100"

    fasta_path = os.path.join(MMSEQS_BASE, f"{i}pc",
                               f"non_redundant_{i}pc.fasta")
    output_dir = f"filtered_pdbs_{i}pc"

    if not os.path.exists(fasta_path):
        raise FileNotFoundError(
            f"FASTA not found: {fasta_path}\n"
            f"Run 02a_mmseqs2.sh first."
        )

    print(f"[i] Filtering at {i}% identity")
    print(f"[i] FASTA : {fasta_path}")
    print(f"[i] Output: {output_dir}")

    # ── Step 1: Extract PDB IDs from FASTA ───────────────────────────────────
    pdb_ids = extract_pdb_ids_from_fasta(fasta_path)
    print(f"[i] {len(pdb_ids)} unique PDB IDs in FASTA")

    # ── Step 2: Copy matching PDB files ──────────────────────────────────────
    copied, missing = copy_pdb_files(pdb_ids, PDB_DIR, output_dir)
    print(f"[✓] Copied  : {len(copied)} PDB files to {output_dir}")
    print(f"[!] Missing : {len(missing)} PDB IDs not found on disk")

    if missing:
        missing_log = f"missing_pdbs_{i}pc.txt"
        with open(missing_log, "w") as f:
            for pid in sorted(missing):
                f.write(pid + "\n")
        print(f"    Missing IDs written to {missing_log}")

    # ── Step 3: Build stats across all 100 identity levels ───────────────────
    if not os.path.exists(CHAIN_INVENTORY):
        print(f"[!] {CHAIN_INVENTORY} not found — skipping CSV/plot generation")
        return

    counts_df, groups_df = build_stats_across_all_identities(
        MMSEQS_BASE, CHAIN_INVENTORY
    )

    # ── Step 4: Write CSVs ────────────────────────────────────────────────────
    counts_df.to_csv(COUNTS_CSV, index=False)
    groups_df.to_csv(GROUPS_CSV, index=False)
    print(f"[✓] {COUNTS_CSV} written")
    print(f"[✓] {GROUPS_CSV} written")

    # ── Step 5: Plot ──────────────────────────────────────────────────────────
    make_plot(counts_df, groups_df, PLOT_PATH)

    print(f"\n[✓] Done. Filtered PDBs are in: {output_dir}/")
    print(f"    Pass this directory to 03_geometry.py")


if __name__ == "__main__":
    main()