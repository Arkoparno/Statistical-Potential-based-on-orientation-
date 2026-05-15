import os
import io
import gzip
import csv
from Bio.PDB import PDBParser, MMCIFParser, PPBuilder

# ── CONFIG ─────────────────────────────────────────────────────────────────────
INPUT_DIR       = "pdb_files"
OUTPUT_FASTA    = "all_proteins.fasta"
CHAIN_INVENTORY = "chain_inventory.csv"
# ──────────────────────────────────────────────────────────────────────────────

# Residue classification sets
PROTEIN_RES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
    "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
    "TYR","VAL",
    "MSE","SEC","PYL","HYP","SEP","TPO","PTR"   # modified AA
}
DNA_PREFIXES = {"DA", "DT", "DG", "DC"}         # DA, DT, DG, DC, DA3, DA5 etc.
RNA_RES      = {"A", "U", "G", "C",             # standard RNA
                "RA", "RU", "RG", "RC",         # alternate naming in some PDBs
                "PSU", "5MC", "7MG", "OMG",     # common modified RNA bases
                "H2U", "M2G", "OMC", "OMU"}

# Modified AA → nearest standard AA (for sequence output)
MOD_TO_STD = {
    "MSE": "MET", "SEC": "CYS", "PYL": "LYS",
    "HYP": "PRO", "SEP": "SER", "TPO": "THR", "PTR": "TYR"
}

# Hardcoded 3→1 letter map — no BioPython version dependency
AA3TO1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
}


def is_dna_res(resname: str) -> bool:
    return any(resname.startswith(p) for p in DNA_PREFIXES)


def is_rna_res(resname: str) -> bool:
    return resname in RNA_RES


def parse_structure(pdb_id: str, fpath: str):
    5
    try:
        is_cif = ".cif" in fpath
        parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)

        if fpath.endswith(".gz"):
            with gzip.open(fpath, "rt", errors="ignore") as gz:
                content = gz.read()
            handle = io.StringIO(content)
            return parser.get_structure(pdb_id, handle)
        else:
            return parser.get_structure(pdb_id, fpath)

    except Exception as e:
        print(f"  [!] Parse error {pdb_id}: {e}")
        return None


def build_protein_sequence(chain, ppb: PPBuilder) -> str:
    """
    Build one-letter protein sequence from a chain.
    Tries PPBuilder first (respects peptide bonds).
    Falls back to residue-by-residue scan if PPBuilder returns nothing
    (handles broken bond records common in low-resolution structures).
    """
    sequence = ""

    # Pass 1: PPBuilder
    for pp in ppb.build_peptides(chain, aa_only=False):
        for res in pp:
            rname = res.resname.strip()
            std   = MOD_TO_STD.get(rname, rname)
            sequence += AA3TO1.get(std, "X")

    # Pass 2: residue scan fallback
    if not sequence:
        for res in chain.get_residues():
            if res.id[0] != " ":
                continue                        # skip HETATM (waters, ligands)
            rname = res.resname.strip()
            std   = MOD_TO_STD.get(rname, rname)
            if std in PROTEIN_RES:
                sequence += AA3TO1.get(std, "X")

    return sequence


def classify_chain(chain) -> tuple:
    """
    Returns (chain_type: str, n_residues: int).

    Classification logic:
      - Count ATOM residues only (id[0] == ' '), skip HETATM
      - Check for protein, DNA, RNA membership
      - hybrid   = DNA + RNA present, no protein
      - mixed    = protein + nucleic acid in same chain (flagged, not skipped)
      - protein/dna/rna = exclusive
      - other    = none of the above
    """
    residues = [r for r in chain.get_residues() if r.id[0] == " "]
    n = len(residues)
    if n == 0:
        return "other", 0

    has_protein = False
    has_dna     = False
    has_rna     = False

    for res in residues:
        rname = res.resname.strip()
        if rname in PROTEIN_RES:
            has_protein = True
        elif is_dna_res(rname):
            has_dna = True
        elif is_rna_res(rname):
            has_rna = True

    if has_protein and not has_dna and not has_rna:
        return "protein", n
    if has_dna and has_rna and not has_protein:
        return "hybrid", n      # DNA:RNA duplex
    if has_dna and not has_rna and not has_protein:
        return "dna", n
    if has_rna and not has_dna and not has_protein:
        return "rna", n
    if has_protein and (has_dna or has_rna):
        return "mixed", n       # unusual — keep but flag
    return "other", n


def extract_chain_info(structure, pdb_id: str) -> list:
    """
    Process model 0 only. Returns list of dicts, one per chain.
    """
    ppb     = PPBuilder()
    model   = structure[0]
    records = []

    for chain in model:
        chain_type, n_res = classify_chain(chain)
        sequence = ""

        if chain_type in ("protein", "mixed"):
            sequence = build_protein_sequence(chain, ppb)

        records.append({
            "pdb_id":     pdb_id,
            "chain_id":   chain.id,
            "chain_type": chain_type,
            "n_residues": n_res,
            "sequence":   sequence,
        })

    return records


def main():
    # ── Resume: load already-processed PDB IDs ────────────────────────────────
    done_pdbs = set()
    if os.path.exists(CHAIN_INVENTORY):
        with open(CHAIN_INVENTORY, newline="") as f:
            for row in csv.DictReader(f):
                done_pdbs.add(row["pdb_id"])
        print(f"[~] Resume: {len(done_pdbs)} PDBs already processed")

    # ── Collect files ─────────────────────────────────────────────────────────
    valid_exts = (".pdb", ".pdb.gz", ".cif", ".cif.gz")
    to_process = []

    for fname in sorted(os.listdir(INPUT_DIR)):
        if not any(fname.endswith(e) for e in valid_exts):
            continue
        # pdb_id = everything before first underscore or dot
        # works for: 1abc.pdb, 1abc_assembly1.pdb.gz, abcd1234.cif.gz
        pdb_id = fname.split("_")[0].split(".")[0].lower()
        if pdb_id not in done_pdbs:
            to_process.append((pdb_id, os.path.join(INPUT_DIR, fname)))

    print(f"[i] {len(to_process)} new files to process")

    # ── Open outputs (append if resuming) ────────────────────────────────────
    fasta_mode = "a" if os.path.exists(OUTPUT_FASTA)    else "w"
    inv_mode   = "a" if os.path.exists(CHAIN_INVENTORY) else "w"

    FIELDNAMES = ["pdb_id", "chain_id", "chain_type", "n_residues", "sequence"]

    with open(OUTPUT_FASTA, fasta_mode) as fasta_fh, \
         open(CHAIN_INVENTORY, inv_mode, newline="") as inv_fh:

        writer = csv.DictWriter(inv_fh, fieldnames=FIELDNAMES)
        if inv_mode == "w":
            writer.writeheader()

        for pdb_id, fpath in to_process:
            structure = parse_structure(pdb_id, fpath)
            if structure is None:
                continue

            chain_records = extract_chain_info(structure, pdb_id)

            # Must have at least one protein chain AND one nucleic acid chain
            has_protein = any(r["chain_type"] in ("protein", "mixed")
                              for r in chain_records)
            has_nucleic = any(r["chain_type"] in ("dna", "rna", "hybrid", "mixed")
                              for r in chain_records)

            if not (has_protein and has_nucleic):
                print(f"  [!] {pdb_id.upper()}: no protein+nucleic pair — skipping")
                continue

            # Summarise what was found
            type_counts = {}
            for r in chain_records:
                type_counts[r["chain_type"]] = type_counts.get(r["chain_type"], 0) + 1

            summary = ", ".join(f"{v} {k}" for k, v in sorted(type_counts.items()))
            print(f"[✓] {pdb_id.upper()}: {summary}")

            for rec in chain_records:
                writer.writerow(rec)

                # Write protein sequences to FASTA for MMseqs2
                if rec["chain_type"] in ("protein", "mixed") and rec["sequence"]:
                    fasta_id = f"{rec['pdb_id']}_{rec['chain_id']}"
                    fasta_fh.write(f">{fasta_id}\n{rec['sequence']}\n")

    print(f"\n[✓] FASTA  → {OUTPUT_FASTA}")
    print(f"[✓] Inventory → {CHAIN_INVENTORY}")


if __name__ == "__main__":
    main()