#!/bin/bash
# =============================================================================
# DELPHI STAGE 2 — energy correction + PID — BSC MareNostrum 5.
# Port of scripts/scripts_vincent/train_properties_delphi.sh (Baobab,
# apptainer) with the same physics flags.
#
# >>> STAGE 2 DOES NOT DEPEND ON STAGE 1. <<< It trains the EC/PID heads on
# ground-truth clusters (--use-gt-clusters) with the clustering backbone frozen
# (--freeze-clustering) and loads no checkpoint, so the two trainings are
# independent and can run in parallel. The stage-1 checkpoint is needed only at
# full evaluation, where eval_delphi_full.sh grafts the two together.
#
#   sbatch scripts/bsc/train_delphi_stage2.sh
#   PID_WEIGHTING_NEUTRAL=sqrt_inv sbatch ...        # neutral head only
#   PID_WEIGHTING=sqrt_inv SOFT_MUON_CUT=1.5 sbatch ...   # everything at once
#
# TWO FLAGS ARE NOT WHAT THEY LOOK LIKE: --add-track-chis is REQUIRED (asserted
# at energy_correction_NN_v1.py:358), and --balance-pid-classes is a NO-OP on
# this network-config — the knob that works is --pid-class-weighting. Class
# weighting is OFF by default. Validate any reweighting against visible-energy
# and mass resolution, NOT against the PID confusion matrix.
# =============================================================================
#SBATCH --job-name=delphi_stage2
#SBATCH --output=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/%x-%j.out
#SBATCH --error=/gpfs/scratch/ehpc1013/vriecher/slurm-logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --qos=acc_ehpc
#SBATCH --account=ehpc1013

set -uo pipefail

PROJ=/gpfs/projects/ehpc1013/vriecher
SCR=/gpfs/scratch/ehpc1013/vriecher
PY=$PROJ/envs/mlpf-overlay/bin/python
REPO=$PROJ/delphi_study/mlpf

export WANDB_MODE=offline
export WANDB_DIR=$SCR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLURM_CPU_BIND=none
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export NUMEXPR_MAX_THREADS=20
export NUMEXPR_NUM_THREADS=20

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
GPU_LIST=${GPU_LIST:-0,1,2,3}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}

RUN_NAME=${RUN_NAME:-delphi_stage2}
BATCH_SIZE=${BATCH_SIZE:-20}
# Linear LR scaling from the ARC anchor (1e-3 at effective batch 160). The
# The reference calibration found per-EVENT cost GROWS with per-rank batch here
# (3.8 ev/s @20, 2.0 @40, 2.0 @80) — unmasked attention over the batched graph
# in the EC submodels — which matches what we measured for HitPF stage 1. So
# scale with MORE RANKS, not a bigger BATCH_SIZE.
EFF_BATCH=$(( BATCH_SIZE * GPUS_PER_NODE ))
START_LR=${START_LR:-$(awk "BEGIN{printf \"%.3e\", 1.0e-3*${EFF_BATCH}/160}")}
NUM_EPOCHS=${NUM_EPOCHS:-2}
TRAIN_BATCHES=${TRAIN_BATCHES:-2000}
PID_WEIGHTING=${PID_WEIGHTING:-none}
SOFT_MUON_CUT=${SOFT_MUON_CUT:-0.0}
# Which model gets the EC/PID heads. Default = HitPF (GATr). For Mask3D pass
#   NETWORK_CONFIG=src/models/wrapper/example_mode_mask3d.py \
#   NETWORK_OPTS="-o use_ipa_decoder False -o num_queries 320 -o window_size None"
# The comparison needs ONE EC PER MODEL: the heads sit inside the model and are
# trained on its own backbone features, then grafted onto its clustering at full
# evaluation. A HitPF EC is not reusable for Mask3D.
NETWORK_CONFIG=${NETWORK_CONFIG:-src/models/wrapper/example_mode_gatr_noise.py}
NETWORK_OPTS=${NETWORK_OPTS:-}
# HitPF/GATr IGNORES --optimizer entirely — Gatr_pf_e_noise.configure_optimizers
# hardcodes torch.optim.Adam — so `ranger` there is a no-op kept for fidelity
# with the reference recipe. mask3d_model DOES read it, and raises because
# `pytorch_optimizer` is not installed in the BSC env. So Mask3D stage-2 needs
# OPTIMIZER=adamW.
OPTIMIZER=${OPTIMIZER:-ranger}

mkdir -p "$SCR/wandb" "$SCR/slurm-logs" "$SCR/trained-models/${RUN_NAME}"
cd "$REPO" || exit 1
echo "[data] train=${DATA_TRAIN:-$SCR/data/500k_delana/digitized/}"
echo "[stage2] host=$(hostname) gpus=$GPUS_PER_NODE list=$GPU_LIST"
echo "[stage2] network-config=$NETWORK_CONFIG optimizer=$OPTIMIZER opts='$NETWORK_OPTS'"
echo "[stage2] batch=$BATCH_SIZE x $GPUS_PER_NODE = $EFF_BATCH   start_lr=$START_LR"
echo "[stage2] EC_NEUTRAL_INDIST=${EC_NEUTRAL_INDIST:-0.6 (default)} seed=${SEED:-none}"
echo "[stage2] pid_class_weighting=$PID_WEIGHTING charged=${PID_WEIGHTING_CHARGED:-<global>} neutral=${PID_WEIGHTING_NEUTRAL:-<global>}  soft_muon_cut=$SOFT_MUON_CUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

GPU_CSV=$SCR/slurm-logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv
if [[ "${GPU_MONITOR:-1}" == "1" ]]; then
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
               --format=csv,noheader,nounits -l 1 > "$GPU_CSV" 2>/dev/null &
    GPU_MON_PID=$!
    trap 'kill $GPU_MON_PID 2>/dev/null' EXIT
    echo "[gpu] sampling to $GPU_CSV"
fi

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
      --data-train "${DATA_TRAIN:-$SCR/data/500k_delana/digitized/}" \
      --data-config config_files/config_hits_track_delphi.yaml \
      --network-config "$NETWORK_CONFIG" \
      --model-prefix "$SCR/trained-models/${RUN_NAME}" \
      -clust \
      -clust_dim 3 \
      --num-workers "${NUM_WORKERS:-12}" \
      --prefetch-factor "${PREFETCH_FACTOR:-8}" \
      --gpus "$GPU_LIST" \
      --batch-size "$BATCH_SIZE" \
      --start-lr "$START_LR" \
      --num-epochs "$NUM_EPOCHS" \
      --optimizer "$OPTIMIZER" \
      --fetch-step 4 \
      --fetch-by-files \
      --condensation \
      --log-wandb \
      --wandb-displayname "${RUN_NAME}" \
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
      --train-val-split 0.98 \
      --restrict_PID_charge \
      --PID-4-class \
      --balance-pid-classes \
      ${SEED:+--seed "$SEED"} \
      --pid-class-weighting "$PID_WEIGHTING" \
      ${PID_WEIGHTING_CHARGED:+--pid-class-weighting-charged "$PID_WEIGHTING_CHARGED"} \
      ${PID_WEIGHTING_NEUTRAL:+--pid-class-weighting-neutral "$PID_WEIGHTING_NEUTRAL"} \
      --pid-soft-muon-cut "$SOFT_MUON_CUT" \
      --train-batches "$TRAIN_BATCHES" \
      --use-gt-clusters \
      $NETWORK_OPTS

RC=$?
# Capture BEFORE the echo, and exit with it. Without the explicit
# `exit $RC` the script ends on a successful `echo`, so SLURM records
# the job as COMPLETED even when training died — job 45080197 failed
# on its first batch and still showed State=COMPLETED, ExitCode=0:0.
echo "[stage2] exit=$RC"
exit $RC
