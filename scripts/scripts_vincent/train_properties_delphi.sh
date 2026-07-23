#!/bin/bash
#SBATCH --job-name=mlpf_delphi_props
#SBATCH --partition=private-dpnc-gpu
#SBATCH --output=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.out
#SBATCH --error=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00

# PID + properties ("correction") training for DELPHI — port of
# mlpf/scripts/scripts_vincent/train_properties_full_arc.sh to Baobab
# (gatr:v9 apptainer, --delphi).  Trains energy-correction + PID heads on
# GROUND-TRUTH clusters (--use-gt-clusters, frozen clustering backbone),
# exactly like the ARC recipe.
#
# Deviations from the ARC script, on purpose:
#   - single GPU, batch 20 default: calibration (jobs 10997427/8) showed
#     per-EVENT cost GROWS with batch (3.8 ev/s @20, 2.0 @40, 2.0 @80),
#     consistent with unmasked attention over the batched graph in the EC
#     submodels — bigger batches make it slower, not faster
#   - START_LR auto-scales linearly with BATCH_SIZE from the ARC anchor
#     (1e-3 at effective batch 160); unscaled LR NaN'd the clustering stage
#
# Defaults = 2-EPOCH SMOKE TEST (~2k batches/epoch); for the full run
# (full epoch at bs20 = 500k/20 = 24500 batches):
#   RUN_NAME=delphi_props_500k NUM_EPOCHS=100 TRAIN_BATCHES=24500 \
#     sbatch -t 3-00:00:00 training/train_properties_delphi.sh
#
# Env overrides: RUN_NAME, NUM_EPOCHS, TRAIN_BATCHES, BATCH_SIZE, START_LR

set -euo pipefail

SCRATCH=/srv/beegfs/scratch/users/r/riechers/delphi_mlpf
SIF=${SCRATCH}/containers/gatr_v9.sif
REPO=/home/users/r/riechers/mlpf
DATA_DIR=${SCRATCH}/digitized_filtered/

RUN_NAME=${RUN_NAME:-delphi_props_smoketest}
MODEL_PREFIX=${SCRATCH}/models/${RUN_NAME}/
mkdir -p "${MODEL_PREFIX}"

export WANDB_MODE=online
export WANDB_DIR=${SCRATCH}/wandb/${RUN_NAME}
mkdir -p "${WANDB_DIR}"
export APPTAINERENV_PYTHONPATH=${SCRATCH}/containers/pyextra

NUM_EPOCHS=${NUM_EPOCHS:-2}
TRAIN_BATCHES=${TRAIN_BATCHES:-2000}
BATCH_SIZE=${BATCH_SIZE:-20}
# linear LR scaling from the ARC anchor: 1e-3 at effective batch 160
START_LR=${START_LR:-$(awk "BEGIN{printf \"%.3e\", 1.0e-3*${BATCH_SIZE}/160}")}

nvidia-smi || true
cd "${REPO}"

apptainer exec --nv -B /srv/beegfs/scratch -B /home \
    --env WANDB_MODE=online --env WANDB_DIR="${WANDB_DIR}" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env SLURM_CPU_BIND=none \
    "${SIF}" \
    python -m src.train_lightning1 \
    --delphi \
    --data-train "${DATA_DIR}" \
    --data-config config_files/config_hits_track_delphi.yaml \
    -clust \
    -clust_dim 3 \
    --network-config src/models/wrapper/example_mode_gatr_noise.py \
    --model-prefix "${MODEL_PREFIX}" \
    --num-workers 6 \
    --wandb-displayname "${RUN_NAME}" \
    --gpus 0 \
    --batch-size "${BATCH_SIZE}" \
    --start-lr "${START_LR}" \
    --num-epochs "${NUM_EPOCHS}" \
    --optimizer ranger \
    --fetch-step 4 \
    --condensation \
    --log-wandb \
    --wandb-projectname mlpf-delphi \
    --wandb-entity optimal-design \
    --frac_cluster_loss 0 \
    --qmin 1 \
    --use-average-cc-pos 0.99 \
    --lr-scheduler reduceplateau \
    --tracks \
    --add-track-chis \
    --correction \
    --ec-model gatr-neutrals \
    --regress-pos \
    --freeze-clustering \
    --regress-unit-p \
    --separate-PID-GATr \
    --n-layers-PID-head 3 \
    --fetch-by-files \
    --train-val-split 0.98 \
    --restrict_PID_charge \
    --PID-4-class \
    --balance-pid-classes \
    --train-batches "${TRAIN_BATCHES}" \
    --use-gt-clusters
