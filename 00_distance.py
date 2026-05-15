import os
import time     
import gzip
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed 

# ── CONFIG ─────────────────────────────────────────────────────────────────────
PDB_IDS_FILE = "/Users/arko/Desktop/Nucleosome/pid.txt"
SAVE_DIR     = "pdb_files"
FAILED_LOG   = "failed_downloads.txt"   
MAX_WORKERS  = 8    # number of parallel download threads (adjust based on your network and CPU)
MAX_RETRIES  = 3    # total attempts per URL (initial try + retries)
RETRY_DELAY  = 2      # seconds, only between genuine retries
TIMEOUT      = 30     # seconds per HTTP request
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

# All residue names that count as protein
PROTEIN_RES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
    "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP",
    "TYR","VAL","MSE","SEC","PYL","HYP","SEP","TPO","PTR"
}
# DNA residue name prefixes
DNA_PREFIXES = {"DA", "DT", "DG", "DC"}
# RNA residue names (single-letter names in PDB format)
RNA_RES = {"A", "U", "G", "C", "RA", "RU", "RG", "RC"}


def get_url_candidates(pdb_id: str) -> list:

    pid = pdb_id.upper() 
    return [
        (f"https://files.rcsb.org/download/{pid}-assembly1.pdb.gz",
         f"{pdb_id}_assembly1.pdb.gz"),
        (f"https://files.rcsb.org/download/{pid}-assembly1.cif.gz",
         f"{pdb_id}_assembly1.cif.gz"),
        (f"https://files.rcsb.org/download/{pid}.pdb",
         f"{pdb_id}.pdb"),
        (f"https://files.rcsb.org/download/{pid}.cif",
         f"{pdb_id}.cif"),
    ]




def has_protein_and_nucleic(filepath: str) -> bool:
   
    opener = gzip.open if filepath.endswith(".gz") else open
    found_protein = False
    found_nucleic = False

    try:
        with opener(filepath, "rt", errors="ignore") as fh:
            for line in fh:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                res = line[17:20].strip()

                if not found_protein and res in PROTEIN_RES:
                    found_protein = True

                if not found_nucleic:
                    # DNA: starts with D prefix (DA, DT, DG, DC, DA3 etc.)
                    if any(res.startswith(p) for p in DNA_PREFIXES):
                        found_nucleic = True
                    # RNA: exact match on single-letter names
                    elif res in RNA_RES:
                        found_nucleic = True

                if found_protein and found_nucleic:
                    return True
    except Exception:
        return False

    return False


def download_one(pdb_id: str) -> tuple:
    """
    Try each URL candidate in order.
    Returns (pdb_id, status) where status is:
      'ok'               — downloaded and validated
      'skipped'          — already on disk and valid
      'invalid_id'       — not 4 or 8 chars
      'failed_all_sources' — all URLs failed or content invalid
    """
    pdb_id = pdb_id.strip().lower()

    if len(pdb_id) not in (4, 8):
        return pdb_id, "invalid_id"

    # Resume: if any candidate file already exists and is non-empty, accept it
    for _, fname in get_url_candidates(pdb_id):
        fpath = os.path.join(SAVE_DIR, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            return pdb_id, "skipped"

    # Try each URL
    for url, fname in get_url_candidates(pdb_id):
        fpath = os.path.join(SAVE_DIR, fname)
        downloaded = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(url, timeout=TIMEOUT)

                if resp.status_code == 200:
                    with open(fpath, "wb") as f:
                        f.write(resp.content)
                    downloaded = True
                    break                          # exit retry loop — got the file

                elif resp.status_code == 404:
                    break                          # this URL doesn't exist, try next
                                                   # — no sleep needed

                # Any other HTTP error: retry after delay
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

            except requests.exceptions.RequestException:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)        # sleep only between real retries

        if downloaded:
            if has_protein_and_nucleic(fpath):
                return pdb_id, "ok"
            else:
                os.remove(fpath)                   # invalid content, try next URL
                # fall through to next candidate

    return pdb_id, "failed_all_sources"


def main():
    with open(PDB_IDS_FILE) as f:
        raw = [line.strip().lower() for line in f if line.strip()]
    pdb_ids = list(dict.fromkeys(raw))             # deduplicate, preserve order

    print(f"[i] {len(pdb_ids)} unique IDs — {MAX_WORKERS} parallel workers")

    results = {"ok": [], "skipped": [], "failed": {}}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, pid): pid for pid in pdb_ids}
        for future in as_completed(futures):
            pid, status = future.result()
            if status == "ok":
                results["ok"].append(pid)
                print(f"[✓] {pid.upper()}")
            elif status == "skipped":
                results["skipped"].append(pid)
                print(f"[~] {pid.upper()} (already exists)")
            else:
                results["failed"][pid] = status
                print(f"[!] {pid.upper()} — {status}")

    if results["failed"]:
        with open(FAILED_LOG, "w") as f:
            for pid, reason in results["failed"].items():
                f.write(f"{pid}\t{reason}\n")
        print(f"[i] Failed IDs written to {FAILED_LOG}")

    print(f"\n[✓] Done.")
    print(f"    Downloaded : {len(results['ok'])}")
    print(f"    Skipped    : {len(results['skipped'])}")
    print(f"    Failed     : {len(results['failed'])}")

def get_organism_from_api(pdb_id: str):
    try:
        # Step 1: get entity IDs
        entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        r = requests.get(entry_url, timeout=10)
        if r.status_code != 200:
            return "unknown"

        data = r.json()
        entity_ids = data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])

        organisms = []

        # Step 2: query each entity
        for eid in entity_ids:
            entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{eid}"
            r2 = requests.get(entity_url, timeout=10)

            if r2.status_code == 200:
                edata = r2.json()

                src = edata.get("rcsb_entity_source_organism", [])
                for org in src:
                    name = org.get("ncbi_scientific_name")
                    if name:
                        organisms.append(name)

        return list(set(organisms)) if organisms else "unknown"

    except Exception:
        return "unknown"

    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
 
