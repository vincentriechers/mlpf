#!/bin/bash
#SBATCH --job-name=mlpf_delphi_eval
#SBATCH --partition=private-dpnc-gpu
#SBATCH --output=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.out
#SBATCH --error=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# Clustering evaluation on DELPHI (mirrors scripts/eval_clustering_500k_05.sh).
# Produces the matched-showers dataframe at
#   ${MODEL_DIR}/showers_df_evaluation/${OUTPUT_NAME}
# Env overrides: EVAL_CKPT (default: newest ckpt in MODEL_DIR),
#                EVAL_TAG (suffix for the output name), N_EVAL_FILES (default 20)

set -euo pipefail
shopt -s nullglob

SCRATCH=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf
SIF=${SCRATCH}/containers/gatr_v9.sif
REPO=/home/users/r/riechers/mlpf
MODEL_DIR=${SCRATCH}/models/delphi_500k_clustering

CKPT="${EVAL_CKPT:-$(ls -1t "${MODEL_DIR}"/*.ckpt | head -n1)}"
EVAL_TAG="${EVAL_TAG:-latest}"
N_EVAL_FILES="${N_EVAL_FILES:-20}"
OUTPUT_NAME="eval_clustering_delphi_${EVAL_TAG}.pkl"
echo "Using checkpoint: ${CKPT}"

# last N files in seed order as the eval slice
DATA_FILES=($(ls ${SCRATCH}/digitized_filtered/pf_tree_*.parquet | tail -n "${N_EVAL_FILES}"))
echo "Evaluating on ${#DATA_FILES[@]} files"

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
    --wandb-displayname "delphi_eval_${EVAL_TAG}" \
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
