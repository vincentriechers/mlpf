#!/bin/bash
#SBATCH --job-name=mlpf_delphi_eval_full
#SBATCH --partition=private-dpnc-gpu
#SBATCH --output=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.out
#SBATCH --error=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# FULL-pipeline evaluation for DELPHI (clustering + EC + PID), port of
# mlpf/scripts/scripts_vincent/eval_full_500k_05.sh: loads the clustering
# checkpoint AND the properties checkpoint, predicts on REAL clusters
# (no --use-gt-clusters), writes the showers dataframe with
# calibrated_E / pred_pid_matched / pred_pos_matched columns to
#   ${PROPS_MODEL_DIR}/showers_df_evaluation/${OUTPUT_NAME}*
#
# Env overrides:
#   VAL_DIR            validation parquets   (default ${SCRATCH}/validation_filtered)
#   N_EVAL_FILES       "all" or a number      (default 50 — first-look subset)
#   CLUSTER_MODEL_DIR  clustering model dir   (default delphi_500k_sB02_trk005)
#   PROPS_MODEL_DIR    properties model dir   (default delphi_props_smoketest)
#   CLUSTER_CKPT / PROPS_CKPT  explicit checkpoints (default: newest in each dir)
#   EVAL_TAG           output name suffix     (default firstlook)

set -euo pipefail
shopt -s nullglob

SCRATCH=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf
SIF=${SCRATCH}/containers/gatr_v9.sif
REPO=/home/users/r/riechers/mlpf

VAL_DIR=${VAL_DIR:-${SCRATCH}/validation_filtered}
CLUSTER_MODEL_DIR=${CLUSTER_MODEL_DIR:-${SCRATCH}/models/delphi_500k_sB02_trk005}
PROPS_MODEL_DIR=${PROPS_MODEL_DIR:-${SCRATCH}/models/delphi_props_smoketest}
EVAL_TAG=${EVAL_TAG:-firstlook}
OUTPUT_NAME="eval_full_delphi_${EVAL_TAG}.pkl"

CLUSTER_CKPT="${CLUSTER_CKPT:-$( (ls -1t "${CLUSTER_MODEL_DIR}"/*.ckpt 2>/dev/null || true) | head -n1)}"
PROPS_CKPT="${PROPS_CKPT:-$( (ls -1t "${PROPS_MODEL_DIR}"/*.ckpt 2>/dev/null || true) | head -n1)}"
[[ -n "${CLUSTER_CKPT}" ]] || { echo "no clustering ckpt in ${CLUSTER_MODEL_DIR}" >&2; exit 2; }
[[ -n "${PROPS_CKPT}" ]] || { echo "no props ckpt in ${PROPS_MODEL_DIR}" >&2; exit 2; }

N_EVAL_FILES="${N_EVAL_FILES:-50}"
ALL_FILES=("${VAL_DIR}"/pf_tree_*.parquet)
if [[ "${N_EVAL_FILES}" == "all" ]]; then
    DATA_FILES=("${ALL_FILES[@]}")
else
    # bash-native slice: `ls | head` dies with SIGPIPE under pipefail
    DATA_FILES=("${ALL_FILES[@]:0:${N_EVAL_FILES}}")
fi
[[ ${#DATA_FILES[@]} -gt 0 ]] || { echo "no parquets in ${VAL_DIR}" >&2; exit 2; }

echo "Clustering ckpt : ${CLUSTER_CKPT}"
echo "Properties ckpt : ${PROPS_CKPT}"
echo "Files           : ${#DATA_FILES[@]} from ${VAL_DIR}"
echo "Output          : ${PROPS_MODEL_DIR}/showers_df_evaluation/${OUTPUT_NAME}*"

export WANDB_MODE=online
export WANDB_DIR=${SCRATCH}/wandb/eval_full_${SLURM_JOB_ID}
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
    --model-prefix "${PROPS_MODEL_DIR}" \
    --load-model-weights-clustering "${CLUSTER_CKPT}" \
    --load-model-weights "${PROPS_CKPT}" \
    --wandb-displayname "delphi_eval_full_${EVAL_TAG}" \
    --num-workers 4 \
    --gpus 0 \
    --batch-size 1 \
    --start-lr 1e-3 \
    --num-epochs 100 \
    --optimizer ranger \
    --fetch-step 1 \
    --condensation \
    --log-wandb \
    --wandb-projectname mlpf-delphi \
    --wandb-entity optimal-design \
    --frac_cluster_loss 0 \
    --qmin 1 \
    --use-average-cc-pos 0.99 \
    --lr-scheduler reduceplateau \
    --tracks \
    --correction \
    --ec-model gatr-neutrals \
    --regress-pos \
    --add-track-chis \
    --freeze-clustering \
    --regress-unit-p \
    --n-layers-PID-head 3 \
    --separate-PID-GATr \
    --fetch-by-files \
    --restrict_PID_charge \
    --PID-4-class
