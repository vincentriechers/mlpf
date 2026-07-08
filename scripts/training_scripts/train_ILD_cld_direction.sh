#!/bin/bash
# =============================================================================
# ILD clustering training — "new model" (CLD-symmetry GATr + track direction)
#
#   Model   : src/models/wrapper/example_mode_gatr_noise_cld_direction.py
#             (Gatr_pf_e_noise_cld_direction: beam-axis z-hat reference node
#              -> SO(2)_phi azimuthal symmetry, + track direction at calo,
#              + logE/logp inputs)
#   Stage   : clustering (object condensation), frac_cluster_loss=0, qmin=3
#   Data    : ILD_l5_o1_v02 e+e- @ 91 GeV, staged locally on /data
#             uds = Zdd + Zuu + Zss   (592 files)
#             Zcc (200) + Zbb (200)   -> ~193 GB, 992 files
#             NOTE: Ztautau is empty in the source (not produced yet);
#                   ee / mumu / bhabha / gammagamma are not in this dataset.
#   Output  : EOS  (model checkpoints + logs)
#
#   ENVIRONMENT (Blackwell sm_120):
#     The gatr:v9 container ships torch cu118 (sm_90 max) and CANNOT run on the
#     RTX PRO 6000 Blackwell GPUs of this node. Instead we use a Python 3.11
#     interpreter (python:3.11 sandbox) + a venv on /data carrying
#     torch 2.7.0+cu128 (Blackwell-capable) + dgl 2.4 (graph lives on GPU via
#     memcpy; no dgl compute kernel is ever launched) + xformers/pyg/gatr/
#     lightning. Validated by a full forward/loss/backward smoke test.
#     Runs on all 4 GPUs.
# =============================================================================
set -euo pipefail

# ---- paths -------------------------------------------------------------------
REPO=/afs/cern.ch/work/m/mgarciam/private/mlpf
SANDBOX=/data/mgarciam/py311                 # python 3.11 interpreter (container)
VENV_PY=/data/mgarciam/venv_bw/bin/python    # Blackwell env (torch 2.7/cu128 + deps)
DATA_ROOT=/data/mgarciam/ILD_train
CFG_DATA=config_files/config_hits_track_v4.yaml
CFG_NET=src/models/wrapper/example_mode_gatr_noise_cld_direction.py
MODEL_PREFIX=/eos/user/m/mgarciam/datasets_mlpf/models_trained_ILD/ILD_cld_direction_clustering_20260629_p2p

# Full CLD-style composition: uds + cc + bb (~200k each, 1000 ev/file) +
# tau/mu/ee + gammagamma/bhabha (50k each; tau 1000 ev/file, the rest 100 ev/file).
# Each entry is a folder; the trainer globs "<folder>*.parquet", so trailing slash required.
DATA_TRAIN=(
    "${DATA_ROOT}/p8_ee_Zdd_ecm91/"
    "${DATA_ROOT}/p8_ee_Zuu_ecm91/"
    "${DATA_ROOT}/p8_ee_Zss_ecm91/"
    "${DATA_ROOT}/p8_ee_Zcc_ecm91/"
    "${DATA_ROOT}/p8_ee_Zbb_ecm91/"
    "${DATA_ROOT}/p8_ee_Ztautau_ecm91/"
    "${DATA_ROOT}/p8_ee_Zmumu_ecm91/"
    "${DATA_ROOT}/p8_ee_Zee_ecm91/"
    "${DATA_ROOT}/p8_ee_gammagamma_ecm91/"
    "${DATA_ROOT}/p8_ee_bhabha_ecm91/"
)

mkdir -p "${MODEL_PREFIX}"
# Log to LOCAL disk, NOT EOS: tee'ing the log to EOS makes every rank block on EOS
# write latency (xrootd/kerberos), which stalls a rank and triggers an NCCL collective
# timeout. Local XFS never blocks. Copy to EOS afterwards if needed.
LOG_DIR=/data/mgarciam/ild_train_logs
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# ---- kerberos (for EOS writes during the long run) ---------------------------
# Keep the ticket alive in a separate tmux with:
#   /usr/bin/python3 ~/kinit/renewticket.py -s 3600
: "${KRB5CCNAME:=$(klist 2>/dev/null | awk '/Ticket cache/{print $3}')}"
export KRB5CCNAME
CC_PATH="${KRB5CCNAME#FILE:}"
echo "Using kerberos cache: ${KRB5CCNAME}"

# ---- show config -------------------------------------------------------------
echo "=================================================================="
echo " Interp    : ${SANDBOX}  (python 3.11)"
echo " Env       : ${VENV_PY}  (torch 2.7/cu128)"
echo " Repo      : ${REPO}"
echo " Net cfg   : ${CFG_NET}"
echo " Data cfg  : ${CFG_DATA}"
echo " Data dirs :"
printf '   %s\n' "${DATA_TRAIN[@]}"
echo " Output    : ${MODEL_PREFIX}"
echo " Log       : ${LOG_FILE}"
echo " GPUs      : 0,1,2,3"
echo "=================================================================="
nvidia-smi -L || true

# ---- run ---------------------------------------------------------------------
# PYTHONNOUSERSITE=1 keeps the AFS ~/.local out of the venv; HOME is left at the
# user's AFS home so wandb finds ~/.netrc for online logging.
singularity exec \
    -B /eos/user \
    -B /eos/home-m \
    -B /eos \
    -B /afs \
    -B /data/mgarciam \
    -B "${CC_PATH}:${CC_PATH}" \
    --env KRB5CCNAME="${KRB5CCNAME}" \
    --env PYTHONPATH="${REPO}" \
    --env PYTHONNOUSERSITE=1 \
    --env WANDB_MODE=offline \
    --env WANDB_DIR=/data/mgarciam/wandb \
    --env WANDB_CACHE_DIR=/data/mgarciam/wandb/cache \
    --env WANDB_CONFIG_DIR=/data/mgarciam/wandb/config \
    --env NCCL_P2P_DISABLE=0 \
    --env NCCL_IB_DISABLE=1 \
    --env NCCL_CUMEM_ENABLE=0 \
    --env NCCL_SOCKET_IFNAME=lo \
    --env NCCL_DEBUG=WARN \
    --env TORCH_NCCL_ENABLE_MONITORING=0 \
    --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
    --nv \
    "${SANDBOX}" \
    bash -c "cd ${REPO} && ${VENV_PY} -m src.train_lightning1 \
        --data-train ${DATA_TRAIN[*]} \
        --data-config ${CFG_DATA} \
        -clust -clust_dim 3 \
        --network-config ${CFG_NET} \
        --model-prefix ${MODEL_PREFIX}/ \
        --num-workers 4 \
        --gpus 0,1,2,3 \
        --batch-size 40 \
        --start-lr 1e-3 \
        --num-epochs 10 \
        --optimizer ranger \
        --fetch-step 1 \
        --condensation \
        --log-wandb \
        --wandb-displayname ILD_first_training \
        --wandb-projectname mlpf_debug \
        --wandb-entity ml4hep \
        --frac_cluster_loss 0 \
        --qmin 3 \
        --use-average-cc-pos 0.98 \
        --tracks \
        --train-val-split 0.98 \
        --fetch-by-files \
        --ILD \
        --min-objects 2 \
        --train-batches 5400" \
    2>&1 | tee "${LOG_FILE}"
