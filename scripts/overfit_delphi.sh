#!/bin/bash
#SBATCH --job-name=mlpf_delphi_overfit
#SBATCH --partition=private-dpnc-gpu
#SBATCH --output=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.out
#SBATCH --error=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

# Overfit sanity check: train the clustering on ~100 DELPHI events (1 file)
# until the loss clearly memorizes. Validates the full input pipeline + loss
# before real training. Mirrors scripts/training_mlpf_cld_arc_05.sh.

set -euo pipefail

SCRATCH=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf
SIF=${SCRATCH}/containers/gatr_v9.sif
REPO=/home/users/r/riechers/mlpf

# 1-file dataset for overfitting
OVERFIT_DIR=${SCRATCH}/overfit_data
mkdir -p "${OVERFIT_DIR}"
cp -n "${SCRATCH}/digitized_filtered/pf_tree_100000.parquet" "${OVERFIT_DIR}/" || true

MODEL_PREFIX=${SCRATCH}/models/overfit_${SLURM_JOB_ID}/
mkdir -p "${MODEL_PREFIX}"

export WANDB_MODE=offline
export WANDB_DIR=${SCRATCH}/wandb/${SLURM_JOB_ID}
mkdir -p "${WANDB_DIR}"

# gatr:v9 ships with a few broken/missing python deps; pyextra fills them in
export APPTAINERENV_PYTHONPATH=${SCRATCH}/containers/pyextra

nvidia-smi || true
cd "${REPO}"

apptainer exec --nv -B /srv/beegfs/scratch -B /home \
    --env WANDB_MODE=offline --env WANDB_DIR="${WANDB_DIR}" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${SIF}" \
    python -m src.train_lightning1 \
    --delphi \
    --data-train "${OVERFIT_DIR}/" \
    --data-config config_files/config_hits_track_delphi.yaml \
    --network-config src/models/wrapper/example_mode_gatr_noise.py \
    --model-prefix "${MODEL_PREFIX}" \
    --num-workers 1 \
    --gpus 0 \
    --batch-size 10 \
    --start-lr 1e-3 \
    --num-epochs 100 \
    --optimizer ranger \
    --fetch-by-files \
    --fetch-step 1 \
    --condensation \
    --log-wandb \
    --wandb-displayname delphi_overfit \
    --wandb-projectname mlpf_delphi \
    --wandb-entity ml4hep \
    --frac_cluster_loss 0 \
    --qmin 3 \
    --use-average-cc-pos 0.98 \
    --tracks \
    --train-val-split 0.9 \
    --train-batches 9
