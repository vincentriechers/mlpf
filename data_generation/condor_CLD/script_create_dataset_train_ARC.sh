#!/bin/bash

python3 submit_jobs_train_ARC.py \
  --sample gun \
  --cldgeo CLD_o2_v05 \
  --config config_spread_031224_fair.gun \
  --outdir /eos/experiment/fcc/ee/simulation/key4hep_2024_10_03/91GeV/CLD_ARC/500k_training \
  --condordir /eos/experiment/fcc/ee/simulation/key4hep_2024_10_03/91GeV/CLD_ARC/500k_training/condor \
  --njobs 5000 \
  --nev  100 \
  --queue tomorrow \
  --cldconfig /eos/user/v/vriecher/mlpf_arc/CLDConfig_ARC/CLDConfig \
  --arc \
  --gentracking