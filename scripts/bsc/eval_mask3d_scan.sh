#!/usr/bin/env bash
# =============================================================================
# Evaluate every Mask3D scan variant on the SAME events, stage-2 grafted.
# Stage 1 is retrained per variant; stage 2 is NOT, and is held fixed across
# variants — swapping between two otherwise-equivalent EC checkpoints moves
# sigma68 by 0.0025, which is the size of the effects being looked for.
#
#   bash scripts/bsc/eval_mask3d_scan.sh                 # the 150-file tune set
#   bash scripts/bsc/eval_mask3d_scan.sh --test m3d_tcw1 # the held-back 308
#   bash scripts/bsc/eval_mask3d_scan.sh --dry-run
#
# TUNE / TEST SPLIT: the scan runs on the first 150 files of digitized_novoid;
# the remaining 308 are held back for the one configuration that wins.
# NUM_QUERIES MUST BE PASSED AT EVAL TIME — query_bias is shape-bearing, so a
# 192-query checkpoint read through a 320-query model silently drops it.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
S=/gpfs/scratch/ehpc1013/vriecher
EC=$S/trained-models/mask3d_stage2_bal/_epoch=1_step=12000.ckpt
OUT=$S/trained-models/scan_m3d_tuning
OUTDIR=$OUT/showers_df_evaluation
VAL=$S/data/50k_validation_delana/digitized_novoid

MODE_SEL=tune; ONLY=""; DRY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)    MODE_SEL=test; ONLY="${2:-}"; shift 2 ;;
        --only)    ONLY="${2:-}"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "usage: $0 [--only <variant>] [--test <variant>] [--dry-run]" >&2; exit 2 ;;
    esac
done

# name | run dir | last checkpoint | num_queries
VARIANTS=(
  "base    |mask3d_fixdice_tlw3|_epoch=15_step=30500.ckpt|320"
  "tcw1    |m3d_tcw1           |_epoch=15_step=30500.ckpt|320"
  "tcw2    |m3d_tcw2           |_epoch=15_step=30500.ckpt|320"
  "q192    |m3d_q192           |_epoch=15_step=30500.ckpt|192"
  "obj05   |m3d_obj05          |_epoch=15_step=30500.ckpt|320"
  "tlw10   |m3d_tlw10           |_epoch=15_step=30500.ckpt|320"
  "tlw20   |m3d_tlw20          |_epoch=15_step=30500.ckpt|320"
  "long32  |m3d_long32         |_epoch=31_step=61000.ckpt|320"
  "tcw1q192|m3d_tcw1_q192      |_epoch=15_step=30500.ckpt|192"
  "tcw1long|m3d_tcw1_long      |_epoch=31_step=61000.ckpt|320"
  "ecw1    |m3d_ecw1           |_epoch=15_step=30500.ckpt|320"
  "recall01|m3d_recall01       |_epoch=15_step=30500.ckpt|320"
  "ecw1rec |m3d_ecw1_rec       |_epoch=15_step=30500.ckpt|320"
)

for v in "${VARIANTS[@]}"; do
    IFS='|' read -r TAG RUN CK NQ <<< "$v"
    TAG=$(echo "$TAG" | xargs); RUN=$(echo "$RUN" | xargs)
    CK=$(echo "$CK" | xargs); NQ=$(echo "$NQ" | xargs)
    [[ -n "$ONLY" && "$RUN" != "$ONLY" && "$TAG" != "$ONLY" ]] && continue
    CKPT="$S/trained-models/$RUN/$CK"
    if [[ ! -f "$CKPT" ]]; then
        echo "skip $TAG — no checkpoint at $CKPT"; continue
    fi
    if [[ -f "$OUT/showers_df_evaluation/eval_${TAG}_${TAG}.pkl0_0_None.pt" && "$MODE_SEL" != "test" ]]; then
        echo "skip $TAG — already evaluated"; continue
    fi
    if [[ "$MODE_SEL" == "test" ]]; then
        NF=308; SKIP=150; LBL="${TAG}_test"
    else
        NF=150; SKIP=0;  LBL="$TAG"
    fi
    # Already evaluated? Re-running costs ~1.8 h and overwrites an identical
    # dataframe. Pass --force to redo one deliberately.
    if [[ -z "${FORCE:-}" && -f "$OUTDIR/eval_${LBL}_${LBL}.pkl0_0_None.pt" ]]; then
        echo "skip $LBL — already evaluated (FORCE=1 to redo)"; continue
    fi
    CMD=(env MODE=full MODEL=mask3d TAG="$LBL" N_EVAL_FILES="$NF" SKIP_FILES="$SKIP"
         NUM_QUERIES="$NQ" VAL_DIR="$VAL" CKPT="$CKPT" EC_CKPT="$EC"
         MODEL_DIR="$OUT" DEBUG_RHO_FIRST=0
         POINTS="${LBL}:EVAL_CLS_THRESHOLD=0.5"
         sbatch --qos=acc_ehpc --time=08:00:00 --job-name="ev_$LBL"
         scripts/bsc/scan_dpc_grid.sh)
    if [[ -n "$DRY" ]]; then printf '%q ' "${CMD[@]}"; echo
    else echo "== eval $LBL  ($RUN, num_queries $NQ)"; "${CMD[@]}"; fi
done
