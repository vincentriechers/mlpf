#!/bin/bash



# python submit_jobs_train.py --sample Zcard --cldgeo  CLD_o2_v07 --config p8_ee_ZHuds_ecm240 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_ZH_240_train/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_ZH_240_train/  --njobs  10000 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_240226/CLDConfig/ 


python submit_jobs_train.py --sample Zcard --cldgeo  CLD_o2_v07 --config p8_ee_Zuds_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD07_Z_91_train/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD07_Z_91_train/  --njobs  10000 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_240226/CLDConfig/ 


# CLD_02_v07 uds /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_ZH_240_train/
# CLD_02_v07 uds /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD07_Z_91_train/
