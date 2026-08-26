#!/bin/bash
# =============================================================================
# DELPHI Mask3D / Attn-IPA — BSC MareNostrum 5 SMOKE RUN.
#
# Sibling of smoke_delphi_hitpf.sh, which proves the SAME chain for the
# object-condensation (HitPF) model. Everything up to the model is shared:
# env -> parquet loader -> DELPHI dgl graph. Mask3D then replaces the OC head
# with query-based Hungarian matching + mask BCE/dice, so the loss-side flags
# are completely different. The differences that MATTER, all read off the code:
#
#   --optimizer adamW      REQUIRED. attn_ipa_model.py:358 configure_optimizers
#                          raises ValueError on anything that is not adam/adamw.
#                          (The HitPF script uses `ranger`; copying it crashes.)
#   --gradient-clip-val    REQUIRED in practice. train_lightning1.py:117 records
#                          NaN/Inf crashes inside scipy.linear_sum_assignment
#                          ~36k steps into a no-clip H100/bf16 run. The parser
#                          default is 0.0 (= disabled), so it must be passed.
#   --train-batches        Does double duty: Trainer.limit_train_batches AND
#                          OneCycleLR total_steps (= num_epochs * train_batches,
#                          attn_ipa_model.py:371). Never leave it implicit.
#   -o track_loss_weight   The Mask3D analogue of --beta-noise-weight-track.
#                          loss.py:411-423 turns it into a per-hit weight of
#                          `track_loss_weight` on hits with hit_type ==
#                          track_hit_type (1 = tracks for DELPHI) and 1.0
#                          elsewhere. DELPHI is ~29 tracks vs ~500 calo hits
#                          (5.5% of nodes), so the 1.0 default in
#                          attn_ipa_model.py:65 leaves tracks drowned; 3.0 is
#                          the value mask3d_model.py:216 defaults to.
#   no --condensation      OC-only flag, and dead in this entrypoint anyway
#                          (only src/deprecated/* reads args.condensation).
#   no --qmin / --frac_cluster_loss / --use-average-cc-pos
#                          OC clustering knobs; Mask3D never reads them.
#   -o window_size None    REQUIRED HERE, and correct on the physics too.
#                          encoder.py picks its attention kernel off this knob:
#                          None -> BlockDiagonalSelfAttention (xformers, which
#                          the mlpf-overlay provides), any int -> phi-windowed
#                          FlashVarlenSelfAttention, which imports `flash_attn`
#                          — NOT in the BSC env, so the 1024 default dies with
#                          ModuleNotFoundError in the first forward (job
#                          45068276). Installing flash-attn would buy nothing:
#                          a DELPHI event is ~502 calo hits + ~29 tracks ~= 531
#                          tokens, so a +/-512 sliding window already spans the
#                          whole event. The windowing exists for CLD-scale
#                          events (~10k hits); at DELPHI scale full-event
#                          attention is the same computation without the
#                          dependency.
#
# Runs on acc_debug (2 h cap, priority 10000) so it jumps the queue.
# =============================================================================
#SBATCH --job-name=delphi_attn_ipa_smoke
#SBATCH --output=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --qos=acc_debug
#SBATCH --account=ehpc1013

set -uo pipefail

PROJ=/gpfs/projects/ehpc1013/vriecher
SCR=/gpfs/scratch/ehpc1013/vriecher
PY=$PROJ/envs/mlpf-overlay/bin/python        # BSC mlpf env + cu118 xformers
REPO=$PROJ/delphi_study/mlpf

# Compute nodes have NO outbound internet -> W&B must be offline, synced later
# from a transfer node with:  wandb sync --include-offline <dir>
export WANDB_MODE=offline
export WANDB_DIR=$SCR          # wandb creates $SCR/wandb/ itself — don't nest
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none
# numexpr caps at 64 threads but autodetects the ACC node's 80 cores and prints
# a spurious `Error. nthreads cannot be larger than ...` per rank. Must be set
# before the first import that pulls numexpr in (pandas).
export NUMEXPR_MAX_THREADS=${NUMEXPR_MAX_THREADS:-20}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-20}

# Multi-GPU: override BOTH of these together with the sbatch --gres. NOTE that
# sbatch's --export parses COMMAS as item separators, so GPU_LIST=0,1,2,3 cannot
# be passed inside --export=ALL,... — export it in the calling shell instead and
# let --export=ALL carry it through:
#   GPUS_PER_NODE=4 GPU_LIST=0,1,2,3 sbatch --export=ALL \
#       --gres=gpu:4 --cpus-per-task=80 scripts/bsc/smoke_delphi_attn_ipa.sh
# GPU_LIST must list every GPU: set_gpus splits it into Trainer(devices=...).
# Then CHECK the [shard] lines — ranks must be distinct and world_size must
# equal the GPU count (see the sharding note in bsc_training_howto.md).
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
GPU_LIST=${GPU_LIST:-0}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

RUN_NAME=${RUN_NAME:-delphi_attn_ipa_smoke}
TRACK_LOSS_WEIGHT=${TRACK_LOSS_WEIGHT:-3.0}
WINDOW_SIZE=${WINDOW_SIZE:-None}     # None => xformers full-event attention
# Extra `-o key value` pairs, e.g. NETWORK_OPTS="-o num_queries 128 -o dim 128"
NETWORK_OPTS=${NETWORK_OPTS:-}

mkdir -p $SCR/wandb $SCR/trained-models $SCR/slurm-logs

cd "$REPO" || exit 1
echo "[smoke] host=$(hostname) gpus=$GPUS_PER_NODE list=$GPU_LIST python=$PY"
echo "[smoke] run=$RUN_NAME track_loss_weight=$TRACK_LOSS_WEIGHT window_size=$WINDOW_SIZE opts='$NETWORK_OPTS'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

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
      --network-config src/models/wrapper/example_mode_attn_ipa.py \
      --model-prefix "$SCR/trained-models/${RUN_NAME}" \
      --num-workers "${NUM_WORKERS:-12}" \
      --prefetch-factor "${PREFETCH_FACTOR:-8}" \
      --gpus "$GPU_LIST" \
      --batch-size "${BATCH_SIZE:-8}" \
      --start-lr "${START_LR:-1e-4}" \
      --num-epochs 1 \
      --optimizer adamW \
      --fetch-by-files \
      --fetch-step 4 \
      --use-amp \
      --gradient-clip-val "${GRAD_CLIP:-0.1}" \
      --log-wandb \
      --wandb-displayname "${RUN_NAME}" \
      --wandb-projectname mlpf-delphi \
      --wandb-entity optimal-design \
      --tracks \
      --train-val-split 0.98 \
      --train-batches "${TRAIN_BATCHES:-50}" \
      -o track_loss_weight "$TRACK_LOSS_WEIGHT" \
      -o window_size "$WINDOW_SIZE" \
      $NETWORK_OPTS

echo "[smoke] exit=$?"
