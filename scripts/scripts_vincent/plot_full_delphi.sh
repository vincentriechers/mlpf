#!/bin/bash
#SBATCH --job-name=mlpf_delphi_plot_full
#SBATCH --partition=private-dpnc-cpu
#SBATCH --output=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.out
#SBATCH --error=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Full-pipeline evaluation plots (EC + PID) for DELPHI via
# src.evaluation.full_evaluation_delphi — the ARC/05 comparison plotter
# adapted to a single DELPHI model: same style, confusion grid, event and
# resolution plots, but the overlaid curves distinguish target cuts
# (no cut / CH must have track / >=1,3,5 calo hits) instead of
# geometry/method, no ratio panels, no Pandora slots, data-driven event
# windows.  Per-target hit counts come from the evaluated parquets
# (VAL_DIR) via the sorted-energy event fingerprint.
#
# Env overrides: MLPF_1, VAL_DIR, EVAL_TAG, OUTPUT_DIR

set -euo pipefail

SCRATCH=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf
SIF=${SCRATCH}/containers/gatr_v9.sif
REPO=/home/users/r/riechers/mlpf

EVAL_TAG="${EVAL_TAG:-firstlook}"
MLPF_1="${MLPF_1:-${SCRATCH}/models/delphi_props_smoketest/showers_df_evaluation/eval_full_delphi_${EVAL_TAG}.pkl*.pt}"
VAL_DIR="${VAL_DIR:-${SCRATCH}/validation_filtered}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/users/r/riechers/delphi_converter/analysis/plots/full_eval_${EVAL_TAG}}"
mkdir -p "${OUTPUT_DIR}"

export APPTAINERENV_PYTHONPATH=${SCRATCH}/containers/pyextra
export APPTAINERENV_MLPF_PLOT_USETEX=0
cd "${REPO}"

echo "MLPF   : ${MLPF_1}"
echo "VAL    : ${VAL_DIR}"
echo "Output : ${OUTPUT_DIR}"

apptainer exec -B /srv/beegfs/scratch -B /home \
    "${SIF}" \
    python -m src.evaluation.full_evaluation_delphi \
    --mlpf "${MLPF_1}" \
    --val-dir "${VAL_DIR}" \
    --output-dir "${OUTPUT_DIR}"
