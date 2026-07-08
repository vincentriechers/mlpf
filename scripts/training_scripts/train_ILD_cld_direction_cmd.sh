#!/bin/bash
# Bare launch command for the running ILD clustering training (PID 29340, started 2026-06-29 15:18).
# Branch: ild-blackwell-dgl-cpu   Env: /data/mgarciam/venv_bw (torch 2.7/cu128, Blackwell sm_120)
# For the full reproducible launch (singularity wrapper, kerberos, NCCL env, logging),
# use scripts/training_scripts/train_ILD_cld_direction.sh instead.
set -euo pipefail
cd /afs/cern.ch/work/m/mgarciam/private/mlpf

/data/mgarciam/venv_bw/bin/python -m src.train_lightning1 \
    --data-train \
        /data/mgarciam/ILD_train/p8_ee_Zdd_ecm91/ \
        /data/mgarciam/ILD_train/p8_ee_Zuu_ecm91/ \
        /data/mgarciam/ILD_train/p8_ee_Zss_ecm91/ \
        /data/mgarciam/ILD_train/p8_ee_Zcc_ecm91/ \
        /data/mgarciam/ILD_train/p8_ee_Zbb_ecm91/ \
        /data/mgarciam/ILD_train/p8_ee_Ztautau_ecm91/ \
        /data/mgarciam/ILD_train/p8_ee_Zmumu_ecm91/ \
        /data/mgarciam/ILD_train/p8_ee_Zee_ecm91/ \
        /data/mgarciam/ILD_train/p8_ee_gammagamma_ecm91/ \
        /data/mgarciam/ILD_train/p8_ee_bhabha_ecm91/ \
    --data-config config_files/config_hits_track_v4.yaml \
    -clust -clust_dim 3 \
    --network-config src/models/wrapper/example_mode_gatr_noise_cld_direction.py \
    --model-prefix /eos/user/m/mgarciam/datasets_mlpf/models_trained_ILD/ILD_cld_direction_clustering_20260629_p2p/ \
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
    --train-batches 5400
