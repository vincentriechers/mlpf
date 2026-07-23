#!/bin/bash
#SBATCH --job-name=mlpf_delphi_plot_full
#SBATCH --partition=private-dpnc-cpu
#SBATCH --output=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.out
#SBATCH --error=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# DELPHI-native full-pipeline evaluation plots (clustering + EC + PID) from
# the eval_full dataframe, via training/plot_full_eval_delphi.py.
#
# This replaces the earlier port of plot_full_comparison_arc_05.sh
# (src.evaluation.full_evaluation): that plotter is a CLD o3-vs-o2 detector
# comparison — duplicate model slots, ratio panels, Pandora baselines and
# CLD-tuned event-energy windows, none of which apply to a single DELPHI
# model.  Instead the curves here distinguish target categories (no cut /
# CH must have track / >=1,3,5 calo hits), with axis windows from the data.
# Hit counts come from the evaluated parquets (awkward), so this runs in the
# delphi_converter conda env, not the gatr container.
#
# Env overrides: DF, VAL_DIR, OUTPUT_DIR, EVAL_TAG

set -euo pipefail

SCRATCH=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf
PY=/home/users/r/riechers/.conda/envs/delphi_converter/bin/python
# NOT $(dirname BASH_SOURCE): sbatch copies the script to /var/spool/slurmd
TRAIN_DIR=${TRAIN_DIR:-/home/users/r/riechers/delphi_converter/training}

EVAL_TAG="${EVAL_TAG:-firstlook}"
DF="${DF:-${SCRATCH}/models/delphi_props_smoketest/showers_df_evaluation/eval_full_delphi_${EVAL_TAG}.pkl0_0_None.pt}"
VAL_DIR="${VAL_DIR:-${SCRATCH}/validation_filtered}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/users/r/riechers/delphi_converter/analysis/plots/full_eval_${EVAL_TAG}_delphi}"

echo "DF     : ${DF}"
echo "VAL    : ${VAL_DIR}"
echo "Output : ${OUTPUT_DIR}"

"${PY}" "${TRAIN_DIR}/plot_full_eval_delphi.py" \
    --df "${DF}" \
    --val-dir "${VAL_DIR}" \
    --out "${OUTPUT_DIR}"
