#!/bin/bash
#SBATCH --job-name=mlpf_delphi_500k
#SBATCH --partition=private-dpnc-gpu
#SBATCH --output=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.out
#SBATCH --error=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=6-00:00:00

# Full DELPHI clustering training (copy of mlpf/scripts/training_mlpf_delphi.sh
# so the DELPHI training recipe lives alongside the converter; keep the two in
# sync).  500k Z->qqbar events (filtered pf_trees), Baobab private-dpnc-gpu,
# gatr:v9 apptainer, --delphi dataset flag.
#
# Env overrides:
#   RUN_NAME            model/wandb name        (default delphi_500k_clustering)
#   BETA_NOISE_WEIGHT   s_B noise term weight   (default 1.0)
#   START_LR            learning rate           (default 2.5e-4)
#   RESUME_CKPT         checkpoint to resume from
#   TRAIN_BATCHES       steps per epoch         (default 24000)
#   NUM_EPOCHS          epochs                  (default 10)
# Short s_B-scan run (collapse shows by step ~1000, so ~4k steps suffice):
#   RUN_NAME=delphi_scan_sB020 BETA_NOISE_WEIGHT=0.2 TRAIN_BATCHES=4000 \
#     NUM_EPOCHS=1 sbatch -t 04:00:00 training/training_mlpf_delphi.sh
#
# Example (s_B-tuned retrain):
#   RUN_NAME=delphi_500k_sB02 BETA_NOISE_WEIGHT=0.2 sbatch training/training_mlpf_delphi.sh

set -euo pipefail

SCRATCH=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf
SIF=${SCRATCH}/containers/gatr_v9.sif
REPO=/home/users/r/riechers/mlpf
DATA_DIR=${SCRATCH}/digitized_filtered/

RUN_NAME=${RUN_NAME:-delphi_500k_clustering}
MODEL_PREFIX=${SCRATCH}/models/${RUN_NAME}/
mkdir -p "${MODEL_PREFIX}"

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

# 4991 files x ~100 events; batch 20 x 1 GPU x 24000 steps ~ 480k events/epoch.
# start-lr 2.5e-4: the CLD 1e-3 recipe (effective batch 80) beta-collapsed +
# NaN'd at single-GPU batch 20 (job 10170558) — scaled linearly with batch.
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
    --start-lr "${START_LR:-2.5e-4}" \
    --num-epochs "${NUM_EPOCHS:-10}" \
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
    --beta-noise-weight "${BETA_NOISE_WEIGHT:-1.0}" \
    --use-average-cc-pos 0.98 \
    --tracks \
    --train-val-split 0.98 \
    --train-batches "${TRAIN_BATCHES:-24000}" \
    "${RESUME_ARGS[@]}"
