#!/usr/bin/env bash
# =============================================================================
# Mask3D tuning scan — submit every stage-1 variant in ONE go, in PARALLEL.
# The variants are independent and the allocation is nowhere near binding, so
# the constraint is wall clock, not node-hours. Every variant changes exactly
# ONE thing against the base and keeps SEED=42. STAGE 1 ONLY — stage 2 is
# independent of clustering quality and is grafted at evaluation time.
#
#   bash scripts/bsc/launch_mask3d_scan.sh            # submit
#   bash scripts/bsc/launch_mask3d_scan.sh --dry-run  # print the commands
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
DRY=${1:-}

# name | extra `-o` options | epochs
VARIANTS=(
  "m3d_tcw1   |-o track_cost_weight 1.0 |16"
  "m3d_tcw2   |-o track_cost_weight 2.0 |16"
  "m3d_q192   |-o num_queries 192       |16"
  "m3d_obj05  |-o obj_bce_weight 0.5    |16"
  "m3d_tlw10  |-o track_loss_weight 10  |16"
  "m3d_tlw20  |-o track_loss_weight 20  |16"
  "m3d_long32 |                         |32"
)

for v in "${VARIANTS[@]}"; do
    IFS='|' read -r NAME OPTS EP <<< "$v"
    NAME=$(echo "$NAME" | xargs); OPTS=$(echo "$OPTS" | xargs); EP=$(echo "$EP" | xargs)
    # `num_queries` and `track_loss_weight` are set by the script's own env knobs,
    # not by NETWORK_OPTS, so a `-o` duplicate would be passed twice. Route them.
    NQ=320; TLW=3.0; EXTRA="$OPTS"
    case "$OPTS" in
      "-o num_queries "*)      NQ=${OPTS##* }; EXTRA="" ;;
      "-o track_loss_weight "*) TLW=${OPTS##* }; EXTRA="" ;;
    esac
    # OneCycleLR total_steps = num_epochs * train_batches, so the schedule
    # follows the length automatically as long as TRAIN_BATCHES is the real
    # steps-per-epoch: 490 000 events / (64 * 4 GPUs) = 1914.
    CMD=(env RUN_NAME="$NAME" NUM_EPOCHS="$EP" TRAIN_BATCHES=1914 BATCH_SIZE=64
         SEED=42 NUM_QUERIES="$NQ" TRACK_LOSS_WEIGHT="$TLW"
         NETWORK_OPTS="$EXTRA" GPUS_PER_NODE=4 GPU_LIST=0,1,2,3
         sbatch --export=ALL --qos=acc_ehpc
         --time=$([ "$EP" -gt 16 ] && echo 16:00:00 || echo 10:00:00)
         --job-name="$NAME" scripts/bsc/train_delphi_mask3d.sh)
    if [[ "$DRY" == "--dry-run" ]]; then
        printf '%q ' "${CMD[@]}"; echo
    else
        echo "== $NAME  (epochs $EP, num_queries $NQ, tlw $TLW, opts '$EXTRA')"
        "${CMD[@]}"
    fi
done
