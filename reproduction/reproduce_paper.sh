#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Reproduce all paper tables and figures from frozen CSV data.
#
# Run from anywhere:
#
#   bash reproduction/reproduce_paper.sh
#
# Optional:
#
#   PYTHON=/path/to/python bash reproduction/reproduce_paper.sh
# ============================================================


# ------------------------------------------------------------
# Locate repository root
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON="${PYTHON:-python3}"


# ------------------------------------------------------------
# Input directories
# ------------------------------------------------------------

DATA_DIR="${ROOT_DIR}/data/paper_results"

RQ1_DATA="${DATA_DIR}/rq1/global_stats.csv"
RQ2_DATA="${DATA_DIR}/rq2/global_stats.csv"

RQ3_ZEPHYR_DIR="${DATA_DIR}/rq3/zephyr"
RQ3_PEGASUS_DIR="${DATA_DIR}/rq3/pegasus"

RQ3_ZEPHYR_ATTEMPTS="${RQ3_ZEPHYR_DIR}/embedding_attempts.csv"
RQ3_ZEPHYR_STATS="${RQ3_ZEPHYR_DIR}/embedding_stats.csv"

RQ3_PEGASUS_ATTEMPTS="${RQ3_PEGASUS_DIR}/embedding_attempts.csv"
RQ3_PEGASUS_STATS="${RQ3_PEGASUS_DIR}/embedding_stats.csv"

HITMAN_DATA="${DATA_DIR}/rq3/hitman_minimum_runtime.csv"

PAPER_AGGREGATED_DIR="${DATA_DIR}/paper_aggregated"

# ------------------------------------------------------------
# Analysis / plotting scripts
# ------------------------------------------------------------

SUMMARY_SCRIPT="${ROOT_DIR}/plotting/create_summary_tables.py"
RQ1_PLOT_SCRIPT="${ROOT_DIR}/plotting/main_plots_rq1.py"
RQ2_PLOT_SCRIPT="${ROOT_DIR}/plotting/main_plots_rq2.py"
RQ3_PLOT_SCRIPT="${ROOT_DIR}/plotting/main_plots_rq3.py"

# ------------------------------------------------------------
# Output directories
# ------------------------------------------------------------

REPRO_DIR="${ROOT_DIR}/reproduction"
CHECK_AGGREGATED_SCRIPT="${REPRO_DIR}/check_csvs.py"

TABLE_DIR="${REPRO_DIR}/tables"

FIGURE_DIR="${REPRO_DIR}/figures"
RQ1_FIG_DIR="${FIGURE_DIR}/rq1"
RQ2_FIG_DIR="${FIGURE_DIR}/rq2"
RQ3_ZEPHYR_FIG_DIR="${FIGURE_DIR}/rq3_zephyr"
RQ3_PEGASUS_FIG_DIR="${FIGURE_DIR}/rq3_pegasus"


mkdir -p \
    "${TABLE_DIR}" \
    "${RQ1_FIG_DIR}" \
    "${RQ2_FIG_DIR}" \
    "${RQ3_ZEPHYR_FIG_DIR}" \
    "${RQ3_PEGASUS_FIG_DIR}"


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

check_file() {
    if [[ ! -f "$1" ]]; then
        echo "ERROR: required file not found:"
        echo "  $1"
        exit 1
    fi
}

run_step() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}


# ------------------------------------------------------------
# Validate inputs
# ------------------------------------------------------------

run_step "Checking required files"

check_file "${RQ1_DATA}"
check_file "${RQ2_DATA}"

check_file "${RQ3_ZEPHYR_ATTEMPTS}"
check_file "${RQ3_ZEPHYR_STATS}"

check_file "${RQ3_PEGASUS_ATTEMPTS}"
check_file "${RQ3_PEGASUS_STATS}"

check_file "${HITMAN_DATA}"

check_file "${SUMMARY_SCRIPT}"
check_file "${RQ1_PLOT_SCRIPT}"
check_file "${RQ2_PLOT_SCRIPT}"
check_file "${RQ3_PLOT_SCRIPT}"

check_file "${CHECK_AGGREGATED_SCRIPT}"

check_file "${PAPER_AGGREGATED_DIR}/tables/table_rq1_summary.csv"
check_file "${PAPER_AGGREGATED_DIR}/tables/table_rq2_summary.csv"
check_file "${PAPER_AGGREGATED_DIR}/tables/table_rq3_summary.csv"

check_file "${PAPER_AGGREGATED_DIR}/rq1_aggregated_heatmap_values_all_pa.csv"
check_file "${PAPER_AGGREGATED_DIR}/rq2_aggregated_heatmap_values_all_pa.csv"

check_file "${PAPER_AGGREGATED_DIR}/zephyr/embedding_submissions_averaged_over_attempts.csv"
check_file "${PAPER_AGGREGATED_DIR}/zephyr/embedding_algorithm_instances.csv"
check_file "${PAPER_AGGREGATED_DIR}/zephyr/embedding_heatmap_values_all_pa.csv"
check_file "${PAPER_AGGREGATED_DIR}/zephyr/embedding_hitman_ratio_instances.csv"
check_file "${PAPER_AGGREGATED_DIR}/zephyr/embedding_hitman_ratio_heatmap_values_all_pa.csv"

check_file "${PAPER_AGGREGATED_DIR}/pegasus/embedding_submissions_averaged_over_attempts.csv"
check_file "${PAPER_AGGREGATED_DIR}/pegasus/embedding_algorithm_instances.csv"
check_file "${PAPER_AGGREGATED_DIR}/pegasus/embedding_heatmap_values_all_pa.csv"
check_file "${PAPER_AGGREGATED_DIR}/pegasus/embedding_hitman_ratio_instances.csv"
check_file "${PAPER_AGGREGATED_DIR}/pegasus/embedding_hitman_ratio_heatmap_values_all_pa.csv"

echo "All required files found."


# ============================================================
# TABLES
# ============================================================

run_step "Generating paper summary tables"

"${PYTHON}" "${SUMMARY_SCRIPT}" \
    --rq1-global "${RQ1_DATA}" \
    --rq2-global "${RQ2_DATA}" \
    --rq3-attempts "${RQ3_ZEPHYR_ATTEMPTS}" \
    --rq3-stats "${RQ3_ZEPHYR_STATS}" \
    --hitman "${HITMAN_DATA}" \
    --rq3-topology zephyr \
    --output-dir "${TABLE_DIR}"


# ============================================================
# RQ1 FIGURE
# ============================================================

run_step "Generating RQ1 figure"

"${PYTHON}" "${RQ1_PLOT_SCRIPT}" \
    --input "${RQ1_DATA}" \
    --output-dir "${RQ1_FIG_DIR}" \
    --mode smooth \
    --format pdf


# ============================================================
# RQ2 FIGURE
# ============================================================

run_step "Generating RQ2 figure"

"${PYTHON}" "${RQ2_PLOT_SCRIPT}" \
    --input "${RQ2_DATA}" \
    --output-dir "${RQ2_FIG_DIR}" \
    --mode smooth \
    --format pdf


# ============================================================
# RQ3 — ZEPHYR
# ============================================================

run_step "Generating RQ3 Zephyr figures"

"${PYTHON}" "${RQ3_PLOT_SCRIPT}" \
    --attempts "${RQ3_ZEPHYR_ATTEMPTS}" \
    --stats "${RQ3_ZEPHYR_STATS}" \
    --classical-comparison "${HITMAN_DATA}" \
    --output-dir "${RQ3_ZEPHYR_FIG_DIR}" \
    --mode smooth \
    --format pdf


# ============================================================
# APPENDIX B — PEGASUS
# ============================================================

run_step "Generating Appendix B Pegasus figures"

"${PYTHON}" "${RQ3_PLOT_SCRIPT}" \
    --attempts "${RQ3_PEGASUS_ATTEMPTS}" \
    --stats "${RQ3_PEGASUS_STATS}" \
    --classical-comparison "${HITMAN_DATA}" \
    --output-dir "${RQ3_PEGASUS_FIG_DIR}" \
    --mode smooth \
    --format pdf


# ============================================================
# VERIFY REGENERATED AGGREGATED DATA
# ============================================================

run_step "Verifying regenerated aggregated CSVs"

"${PYTHON}" "${CHECK_AGGREGATED_SCRIPT}" \
    --reference-dir "${PAPER_AGGREGATED_DIR}" \
    --reproduction-dir "${REPRO_DIR}"


# ============================================================
# DONE
# ============================================================

run_step "Reproduction completed successfully"

echo
echo "Generated tables:"
echo "  ${TABLE_DIR}"
echo
echo "Generated figures:"
echo "  RQ1:           ${RQ1_FIG_DIR}"
echo "  RQ2:           ${RQ2_FIG_DIR}"
echo "  RQ3 Zephyr:    ${RQ3_ZEPHYR_FIG_DIR}"
echo "  Appendix B:    ${RQ3_PEGASUS_FIG_DIR}"
echo
echo "Done."