#!/bin/bash

#   Run with: bash 02a_mmseqs2.sh

set -euo pipefail

# ── CONFIG — edit these if needed ─────────────────────────────────────────────
INPUT_FASTA="all_proteins.fasta"
OUTPUT_BASE="mmseqs2_results"
COVERAGE="0.8"
COV_MODE="1"
MEMORY_LIMIT="8G"
# To run only selected values replace with e.g.: IDENTITIES=(30 40 50 70 90)
IDENTITIES=($(seq 1 100))
THREADS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
# ─────────────────────────────────────────────────────────────────────────────

# ── Preflight checks ──────────────────────────────────────────────────────────
if ! command -v mmseqs &> /dev/null; then
    echo "[ERROR] MMseqs2 not found."
    echo "        Install: conda install -c bioconda mmseqs2"
    echo "        Or see:  https://github.com/soedinglab/MMseqs2"
    exit 1
fi

if [ ! -f "$INPUT_FASTA" ]; then
    echo "[ERROR] Input FASTA not found: $INPUT_FASTA"
    echo "        Run 01_fasta_extract.py first."
    exit 1
fi

FASTA_COUNT=$(grep -c "^>" "$INPUT_FASTA" || true)
if [ "$FASTA_COUNT" -eq 0 ]; then
    echo "[ERROR] $INPUT_FASTA contains no sequences."
    exit 1
fi

# macOS bash 3.2 safe: get first and last identity without [-1]
ID_FIRST="${IDENTITIES[0]}"
ID_LAST="${IDENTITIES[${#IDENTITIES[@]}-1]}"   # ${#array[@]}-1 works in bash 3.2

echo "============================================================"
echo " 02a_mmseqs2.sh — MMseqs2 redundancy clustering"
echo "============================================================"
echo " Input FASTA   : $INPUT_FASTA ($FASTA_COUNT sequences)"
echo " Output dir    : $OUTPUT_BASE"
echo " Identities    : ${ID_FIRST}% – ${ID_LAST}%  (${#IDENTITIES[@]} levels)"
echo " Coverage      : $COVERAGE (mode $COV_MODE — over shorter sequence)"
echo " Memory limit  : $MEMORY_LIMIT"
echo " Threads       : $THREADS"
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_BASE"

# ── Build MMseqs2 database once ───────────────────────────────────────────────
DB_PATH="$OUTPUT_BASE/proteins_db"

if [ ! -f "${DB_PATH}.index" ]; then
    echo "[i] Creating MMseqs2 sequence database..."
    mmseqs createdb "$INPUT_FASTA" "$DB_PATH"
    echo "[✓] Database created: $DB_PATH"
else
    echo "[~] Database already exists, skipping createdb"
fi
echo ""

# ── Main clustering loop ──────────────────────────────────────────────────────
DONE=0
SKIPPED=0
FAILED=0

for i in "${IDENTITIES[@]}"; do

    IDENTITY=$(awk "BEGIN {printf \"%.2f\", $i/100}")
    RUN_DIR="$OUTPUT_BASE/${i}pc"
    OUTPUT_FASTA_PATH="$RUN_DIR/non_redundant_${i}pc.fasta"
    LOG_FILE="$RUN_DIR/mmseqs2_${i}pc.log"

    # Resume: skip if output FASTA already exists and is non-empty
    if [ -f "$OUTPUT_FASTA_PATH" ] && [ -s "$OUTPUT_FASTA_PATH" ]; then
        echo "[~] ${i}%: already done, skipping"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    mkdir -p "$RUN_DIR"

    CLUSTER_DB="$RUN_DIR/clustered_db"
    REP_DB="$RUN_DIR/rep_db"
    TMP_DIR="$RUN_DIR/tmp"

    mkdir -p "$TMP_DIR"

    echo "[>] Clustering at ${i}% identity..."

    # Step 1: Cluster
    if ! mmseqs cluster \
            "$DB_PATH" \
            "$CLUSTER_DB" \
            "$TMP_DIR" \
            --min-seq-id "$IDENTITY" \
            -c "$COVERAGE" \
            --cov-mode "$COV_MODE" \
            --threads "$THREADS" \
            --split-memory-limit "$MEMORY_LIMIT" \
            >> "$LOG_FILE" 2>&1; then
        echo "[ERROR] ${i}%: mmseqs cluster failed — see $LOG_FILE"
        rm -rf "$TMP_DIR"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Step 2: Extract one representative per cluster
    # createsubdb = correct for non-redundancy (one seq per cluster)
    # Do NOT use createseqfiledb — that extracts all cluster members
    if ! mmseqs createsubdb \
            "$CLUSTER_DB" \
            "$DB_PATH" \
            "$REP_DB" \
            >> "$LOG_FILE" 2>&1; then
        echo "[ERROR] ${i}%: createsubdb failed — see $LOG_FILE"
        rm -rf "$TMP_DIR"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Step 3: Convert representative DB to FASTA
    if ! mmseqs convert2fasta \
            "$REP_DB" \
            "$OUTPUT_FASTA_PATH" \
            >> "$LOG_FILE" 2>&1; then
        echo "[ERROR] ${i}%: convert2fasta failed — see $LOG_FILE"
        rm -rf "$TMP_DIR"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Step 4: Validate output
    REP_COUNT=$(grep -c "^>" "$OUTPUT_FASTA_PATH" 2>/dev/null || echo 0)

    if [ "$REP_COUNT" -eq 0 ]; then
        echo "[ERROR] ${i}%: output FASTA is empty — see $LOG_FILE"
        rm -rf "$TMP_DIR"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Step 5: Clean up tmp
    rm -rf "$TMP_DIR"

    echo "[✓] ${i}%: $REP_COUNT representatives → $OUTPUT_FASTA_PATH"
    DONE=$((DONE + 1))

done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo " MMseqs2 clustering complete"
echo " Completed : $DONE"
echo " Skipped   : $SKIPPED  (already done)"
echo " Failed    : $FAILED   (check logs in $OUTPUT_BASE/*/)"
echo " Results   : $OUTPUT_BASE/"
echo ""
