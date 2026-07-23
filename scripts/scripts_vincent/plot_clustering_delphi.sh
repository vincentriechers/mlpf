#!/bin/bash
#SBATCH --job-name=mlpf_delphi_plot_clust
#SBATCH --partition=private-dpnc-cpu
#SBATCH --output=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.out
#SBATCH --error=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Clustering performance plots from the matched-showers dataframes written by
# training/eval_clustering_delphi.sh.  DELPHI analogue of
# mlpf/scripts/scripts_vincent/plot_clustering_compare_arc_05.sh
# (src.evaluation.clustering; no Pandora baseline exists for DELPHI).
#
# Single model:
#   sbatch training/plot_clustering_delphi.sh
# Compare two runs (e.g. baseline vs s_B-tuned retrain):
#   MLPF_2="${SCRATCH}/models/delphi_500k_sB02/showers_df_evaluation/eval_clustering_delphi_val50k.pkl*.pt" \
#   LABEL_2="s_B=0.2" sbatch training/plot_clustering_delphi.sh
#
# Env overrides: MLPF_1, LABEL_1, MLPF_2, LABEL_2, OUTPUT_DIR, EVAL_TAG

set -euo pipefail
shopt -s nullglob

SCRATCH=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf
SIF=${SCRATCH}/containers/gatr_v9.sif
REPO=/home/users/r/riechers/mlpf

EVAL_TAG="${EVAL_TAG:-val50k}"
MLPF_1="${MLPF_1:-${SCRATCH}/models/delphi_500k_clustering/showers_df_evaluation/eval_clustering_delphi_${EVAL_TAG}.pkl*.pt}"
LABEL_1="${LABEL_1:-DELPHI 500k baseline}"
# plots land in the repo's (gitignored) plots area in HOME, not on scratch
OUTPUT_DIR="${OUTPUT_DIR:-/home/users/r/riechers/delphi_converter/analysis/plots/clustering_${EVAL_TAG}}"
mkdir -p "${OUTPUT_DIR}"

MLPF_ARGS=("${MLPF_1}")
LABEL_ARGS=("${LABEL_1}")
if [[ -n "${MLPF_2:-}" ]]; then
    MLPF_ARGS+=("${MLPF_2}")
    LABEL_ARGS+=("${LABEL_2:-model 2}")
fi

export APPTAINERENV_PYTHONPATH=${SCRATCH}/containers/pyextra
cd "${REPO}"

echo "Inputs : ${MLPF_ARGS[*]}"
echo "Labels : ${LABEL_ARGS[*]}"
echo "Output : ${OUTPUT_DIR}"

apptainer exec -B /srv/beegfs/scratch -B /home \
    "${SIF}" \
    python -m src.evaluation.clustering \
    --mlpf "${MLPF_ARGS[@]}" \
    --labels "${LABEL_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}"
