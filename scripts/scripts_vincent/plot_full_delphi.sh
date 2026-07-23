#!/bin/bash
#SBATCH --job-name=mlpf_delphi_plot_full
#SBATCH --partition=private-dpnc-cpu
#SBATCH --output=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.out
#SBATCH --error=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00

# Full-pipeline evaluation plots (EC + PID) from the eval_full dataframe,
# DELPHI port of mlpf/scripts/scripts_vincent/plot_full_comparison_arc_05.sh.
# src.evaluation.full_evaluation requires TWO mlpf inputs (--arc-mlpf and
# --o5-mlpf); with a single DELPHI model we pass the same dataframe to both
# slots — plot labels then read "ARC"/"05" for the same curves (cosmetic).
# Once a second props model exists, point MLPF_2 at it for a real comparison.
# No Pandora baseline for DELPHI; usetex auto-disables without LaTeX.
#
# Env overrides: MLPF_1, MLPF_2, EVAL_TAG, OUTPUT_DIR

set -euo pipefail

SCRATCH=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf
SIF=${SCRATCH}/containers/gatr_v9.sif
REPO=/home/users/r/riechers/mlpf

EVAL_TAG="${EVAL_TAG:-firstlook}"
MLPF_1="${MLPF_1:-${SCRATCH}/models/delphi_props_smoketest/showers_df_evaluation/eval_full_delphi_${EVAL_TAG}.pkl*.pt}"
MLPF_2="${MLPF_2:-${MLPF_1}}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/users/r/riechers/delphi_converter/analysis/plots/full_eval_${EVAL_TAG}}"
mkdir -p "${OUTPUT_DIR}"

export APPTAINERENV_PYTHONPATH=${SCRATCH}/containers/pyextra
export APPTAINERENV_MLPF_PLOT_USETEX=0
cd "${REPO}"

echo "MLPF 1 : ${MLPF_1}"
echo "MLPF 2 : ${MLPF_2}"
echo "Output : ${OUTPUT_DIR}"

apptainer exec -B /srv/beegfs/scratch -B /home \
    "${SIF}" \
    python -m src.evaluation.full_evaluation \
    --arc-mlpf "${MLPF_1}" \
    --o5-mlpf "${MLPF_2}" \
    --datatype "hitpf" \
    --output-dir "${OUTPUT_DIR}"
