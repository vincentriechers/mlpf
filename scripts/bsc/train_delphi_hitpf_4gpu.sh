#!/bin/bash
# =============================================================================
# DELPHI HitPF (object-condensation clustering) — BSC MareNostrum 5, 4 GPUs.
#
# Same recipe as the proven 1-GPU smoke_delphi_hitpf.sh; the only deltas are
# the task topology and the batch size. Topology chosen deliberately — the repo
# carries three mutually inconsistent variants for this cluster:
#
#   slurm/submit_job.sh                #SBATCH --ntasks-per-node=4, but then
#                                      `srun --ntasks-per-node=1` + torchrun
#                                      --nproc_per_node=4. The sbatch directive
#                                      is dead; it is really the torchrun form.
#   slurm/launch_clustering_training.sh --ntasks-per-node=1, bare python,
#                                      Lightning spawns its own per-GPU workers.
#   scripts/bsc/smoke_delphi_hitpf.sh  --ntasks-per-node=1 + torchrun (PROVEN
#                                      at 1 GPU on this account).
#
# We keep the proven torchrun form and change only --nproc_per_node 1 -> 4 and
# --gpus 0 -> 0,1,2,3, so the delta from a known-good run is minimal.
#
# >>> CHECK THE `[shard]` LINES IN THE LOG. <<<
# train_utils.to_filelist prints one per rank:
#     [shard] global_rank <r> world_size <W> files <n>
# A correct 4-GPU run shows global_rank 0..3, world_size 4, and ~1250 files
# each (5000 / 4). If every rank prints `global_rank 0 world_size 1 files 5000`
# the ranks are NOT reading disjoint shards — see the rank-detection note in
# ../../delphi_study/bsc_training_howto.md.
#
# >>> THROUGHPUT IS DATA-BOUND, NOT COMPUTE-BOUND. <<<
# Measured on job 45068768 (4 GPU, --batch-size 20, --num-workers 4,
# --prefetch-factor 1): ~80 steps run at 5 steps/s, then EVERYTHING stalls for
# ~16 s, repeating. Net 3.06 steps/s = 245 events/s, with the GPUs idle roughly
# half the wall clock. The period is exact: 4 workers x --fetch-step 4 files x
# 100 events = 1600 events = 80 steps at --batch-size 20, i.e. all four workers
# exhaust their buffers at the same moment and re-read 16 parquets in lockstep.
# `--prefetch-factor` defaults to **1**, so nothing is read ahead to cover it.
# NUM_WORKERS=12 / PREFETCH_FACTOR=8 (now the defaults here) remove the stall
# outright: job 45069069 holds 4.84 steps/s = 387 events/s end to end, 1.77x the
# baseline, for ~11 s more startup. There are 20 cores per rank (80 per node / 4
# ranks), so 12 workers is well within budget.
#
# Scale to production with:
#   sbatch --qos=acc_ehpc --time=48:00:00 scripts/bsc/train_delphi_hitpf_4gpu.sh
# =============================================================================
#SBATCH --job-name=delphi_hitpf_4gpu
#SBATCH --output=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --qos=acc_debug
#SBATCH --account=ehpc1013

set -uo pipefail

PROJ=/gpfs/projects/ehpc1013/vriecher
SCR=/gpfs/scratch/ehpc1013/vriecher
PY=$PROJ/envs/mlpf-overlay/bin/python        # BSC mlpf env + cu118 xformers
REPO=$PROJ/delphi_study/mlpf

export WANDB_MODE=offline
export WANDB_DIR=$SCR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none

# 80 cores / 4 ranks. Keep BLAS from oversubscribing (each rank would otherwise
# think it owns all 80) — same reasoning as slurm/launch_clustering_training.sh.
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
# numexpr caps itself at 64 threads but autodetects the node's 80 cores, so it
# prints `Error. nthreads cannot be larger than environment variable
# NUMEXPR_MAX_THREADS (64)` once per rank at import. Harmless, but it says
# "Error" and shows up in the stderr scan; pin it. MUST be set before any import
# that pulls numexpr in (pandas -> numexpr), hence here rather than in Python.
export NUMEXPR_MAX_THREADS=20
export NUMEXPR_NUM_THREADS=20

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
GPU_LIST=${GPU_LIST:-0,1,2,3}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

RUN_NAME=${RUN_NAME:-delphi_hitpf_4gpu}
mkdir -p $SCR/wandb $SCR/trained-models $SCR/slurm-logs

cd "$REPO" || exit 1
echo "[4gpu] host=$(hostname) gpus=$GPUS_PER_NODE list=$GPU_LIST python=$PY"
echo "[4gpu] SLURM_NTASKS=${SLURM_NTASKS:-unset} SLURM_PROCID=${SLURM_PROCID:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  $PY -m torch.distributed.run \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    --node_rank="$SLURM_PROCID" \
    -m src.train_lightning1 \
      --delphi \
      --data-train "$SCR/data/500k_delana/digitized/" \
      --data-config config_files/config_hits_track_delphi.yaml \
      --network-config src/models/wrapper/example_mode_gatr_noise.py \
      --model-prefix "$SCR/trained-models/${RUN_NAME}" \
      --num-workers "${NUM_WORKERS:-12}" \
      --prefetch-factor "${PREFETCH_FACTOR:-8}" \
      --gpus "$GPU_LIST" \
      --batch-size "${BATCH_SIZE:-20}" \
      --start-lr "${START_LR:-2.5e-4}" \
      --num-epochs "${NUM_EPOCHS:-1}" \
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
      --beta-noise-weight "${BETA_NOISE_WEIGHT:-1.0}" \
      --beta-noise-weight-track "${BETA_NOISE_WEIGHT_TRACK:-0.05}" \
      --train-batches "${TRAIN_BATCHES:-100}"

echo "[4gpu] exit=$?"
