#!/bin/bash
#SBATCH --job-name=mlpf_delphi_eval_clust
#SBATCH --partition=private-dpnc-gpu
#SBATCH --output=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.out
#SBATCH --error=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00

# Clustering evaluation on the DEDICATED DELPHI validation set (50k events),
# DELPHI analogue of mlpf/scripts/scripts_vincent/eval_clustering_500k_05.sh.
# Unlike mlpf/scripts/eval_delphi.sh this does NOT reuse the tail of the
# training folder — point VAL_DIR at the held-out sample.
#
# Produces the matched-showers dataframe at
#   ${MODEL_DIR}/showers_df_evaluation/${OUTPUT_NAME}*
# which training/plot_clustering_delphi.sh turns into performance plots.
#
# Env overrides:
#   VAL_DIR       validation parquet folder            (default: ${SCRATCH}/validation)
#   MODEL_DIR     model folder                          (default: delphi_500k_clustering)
#   EVAL_CKPT     checkpoint path                       (default: newest ckpt in MODEL_DIR)
#   EVAL_TAG      output name suffix                    (default: val50k)
#   N_EVAL_FILES  number of validation files, "all" ok  (default: all)
#
# Usage:  [VAL_DIR=...] sbatch training/eval_clustering_delphi.sh

set -euo pipefail
shopt -s nullglob

SCRATCH=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf
SIF=${SCRATCH}/containers/gatr_v9.sif
REPO=/home/users/r/riechers/mlpf
MODEL_DIR=${MODEL_DIR:-${SCRATCH}/models/delphi_500k_clustering}

VAL_DIR=${VAL_DIR:-${SCRATCH}/validation}
if [[ ! -d "${VAL_DIR}" ]]; then
    # convenience: look for an obvious validation folder before giving up
    for cand in "${SCRATCH}"/validation* "${SCRATCH}"/val_* "${SCRATCH}"/digitized_val*; do
        [[ -d "${cand}" ]] && VAL_DIR="${cand}" && break
    done
fi
if [[ ! -d "${VAL_DIR}" ]]; then
    echo "ERROR: validation folder not found (tried VAL_DIR=${VAL_DIR:-unset} and" >&2
    echo "       ${SCRATCH}/validation*, val_*, digitized_val*)." >&2
    echo "       Run with VAL_DIR=/path/to/validation sbatch $0" >&2
    exit 2
fi

# subshell + `|| true` so head's early close (SIGPIPE on ls with ~500 ckpts)
# doesn't kill the script under pipefail
CKPT="${EVAL_CKPT:-$( (ls -1t "${MODEL_DIR}"/*.ckpt 2>/dev/null || true) | head -n1)}"
EVAL_TAG="${EVAL_TAG:-val50k}"
OUTPUT_NAME="eval_clustering_delphi_${EVAL_TAG}.pkl"

N_EVAL_FILES="${N_EVAL_FILES:-all}"
ALL_FILES=("${VAL_DIR}"/pf_tree_*.parquet)
if [[ "${N_EVAL_FILES}" == "all" ]]; then
    DATA_FILES=("${ALL_FILES[@]}")
else
    # bash-native slice: `ls | head` dies with SIGPIPE under pipefail
    DATA_FILES=("${ALL_FILES[@]:0:${N_EVAL_FILES}}")
fi
if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
    echo "ERROR: no pf_tree_*.parquet in ${VAL_DIR}" >&2
    exit 2
fi

echo "Validation dir : ${VAL_DIR}  (${#DATA_FILES[@]} files)"
echo "Checkpoint     : ${CKPT}"
echo "Output         : ${MODEL_DIR}/showers_df_evaluation/${OUTPUT_NAME}*"

export WANDB_MODE=online
export WANDB_DIR=${SCRATCH}/wandb/eval_${SLURM_JOB_ID}
mkdir -p "${WANDB_DIR}"
export APPTAINERENV_PYTHONPATH=${SCRATCH}/containers/pyextra

nvidia-smi || true
cd "${REPO}"

apptainer exec --nv -B /srv/beegfs/scratch -B /home \
    --env WANDB_MODE=online --env WANDB_DIR="${WANDB_DIR}" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${SIF}" \
    python -m src.train_lightning1 \
    --delphi \
    --predict \
    --data-test "${DATA_FILES[@]}" \
    --name-output "${OUTPUT_NAME}" \
    --data-config config_files/config_hits_track_delphi.yaml \
    -clust \
    -clust_dim 3 \
    --network-config src/models/wrapper/example_mode_gatr_noise.py \
    --model-prefix "${MODEL_DIR}" \
    --load-model-weights "${CKPT}" \
    --wandb-displayname "delphi_eval_clust_${EVAL_TAG}" \
    --num-workers 4 \
    --gpus 0 \
    --batch-size 1 \
    --start-lr 2.5e-4 \
    --num-epochs 10 \
    --optimizer ranger \
    --fetch-step 0.1 \
    --condensation \
    --log-wandb \
    --wandb-projectname mlpf-delphi \
    --wandb-entity optimal-design \
    --frac_cluster_loss 0 \
    --qmin 3 \
    --use-average-cc-pos 0.98 \
    --tracks
