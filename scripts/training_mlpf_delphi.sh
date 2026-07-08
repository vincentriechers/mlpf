#!/bin/bash
#SBATCH --job-name=mlpf_delphi_500k
#SBATCH --partition=private-dpnc-gpu
#SBATCH --output=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.out
#SBATCH --error=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00

# Full DELPHI clustering training: 500k Z->qqbar events (filtered pf_trees).
# Mirrors scripts/training_mlpf_cld_arc_05.sh (CLD 500k baseline), adapted to
# Baobab (private-dpnc-gpu, gatr:v9 apptainer) and the --delphi dataset flag.
# Resumable: checkpoints land in ${MODEL_PREFIX} every 500 steps; to resume,
# resubmit with RESUME_CKPT=/path/to/last.ckpt sbatch scripts/training_mlpf_delphi.sh

set -euo pipefail

SCRATCH=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf
SIF=${SCRATCH}/containers/gatr_v9.sif
REPO=/home/users/r/riechers/mlpf
DATA_DIR=${SCRATCH}/digitized_filtered/

RUN_NAME=${RUN_NAME:-delphi_500k_clustering}
MODEL_PREFIX=${SCRATCH}/models/${RUN_NAME}/
mkdir -p "${MODEL_PREFIX}"

# online W&B (entity as in the fasernu repo; auth via ~/.netrc, visible in the container)
export WANDB_MODE=online
export WANDB_DIR=${SCRATCH}/wandb/${RUN_NAME}
mkdir -p "${WANDB_DIR}"

# gatr:v9 ships with a few broken/missing python deps; pyextra fills them in
export APPTAINERENV_PYTHONPATH=${SCRATCH}/containers/pyextra

RESUME_ARGS=()
if [[ -n "${RESUME_CKPT:-}" ]]; then
    RESUME_ARGS=(--resume-ckpt "${RESUME_CKPT}")
fi

nvidia-smi || true
cd "${REPO}"

# 4991 files x ~100 events; batch 20 x 1 GPU x 24000 steps ~ 480k events/epoch
apptainer exec --nv -B /srv/beegfs/scratch -B /home \
    --env WANDB_MODE=online --env WANDB_DIR="${WANDB_DIR}" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env SLURM_CPU_BIND=none \
    "${SIF}" \
    python -m src.train_lightning1 \
    --delphi \
    --data-train "${DATA_DIR}" \
    --data-config config_files/config_hits_track_delphi.yaml \
    --network-config src/models/wrapper/example_mode_gatr_noise.py \
    --model-prefix "${MODEL_PREFIX}" \
    --num-workers 6 \
    --gpus 0 \
    --batch-size 20 \
    --start-lr 1e-3 \
    --num-epochs 10 \
    --optimizer ranger \
    --fetch-by-files \
    --fetch-step 4 \
    --condensation \
    --log-wandb \
    --wandb-displayname "${RUN_NAME}" \
    --wandb-projectname mlpf-delphi \
    --wandb-entity optimal-design \
    --frac_cluster_loss 0 \
    --qmin 3 \
    --use-average-cc-pos 0.98 \
    --tracks \
    --train-val-split 0.98 \
    --train-batches 24000 \
    "${RESUME_ARGS[@]}"
