#!/bin/bash


## train pythia 240
# python submit_jobs_train.py --sample Zcard --cldgeo  CLD_o2_v07 --config p8_ee_ZHuds_ecm240 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_ZH_240_test/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_ZH_240_eval/  --njobs  10500 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_240226/CLDConfig/ 

## train gun 240
#python submit_jobs_train.py --sample gun --cldgeo  CLD_o2_v07 --config ecor.gun --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_gun_240_train/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_gun_240_train/  --njobs  2000 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_240226/CLDConfig/ 

## eval pythia 240
# python submit_jobs_train.py --sample Zcard --cldgeo  CLD_o2_v07 --config p8_ee_ZHuds_ecm240 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_ZH_240_test/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_ZH_240_eval/  --njobs  10500 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_240226/CLDConfig/ 


# python submit_jobs_train.py --sample Zcard --cldgeo  CLD_o2_v07 --config p8_ee_Zuds_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD07_Z_91_eval/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD07_Z_91_eval/  --njobs  10500 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_240226/CLDConfig/ 


# CLD_02_v07 uds /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_ZH_240_train/
# CLD_02_v07 uds /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD07_Z_91_train/

## eval pythia 91 bb 
# python submit_jobs_train.py --sample Zcard --cldgeo  CLD_o2_v05 --config p8_ee_Zcc_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_o2_v05_Zcc_91_eval/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_o2_v05_Zcc_91_eval/  --njobs  100 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_240226/CLDConfig/ 

# python submit_jobs_train_ARC.py --sample Zcard --cldgeo  CLD_o2_v05 --config p8_ee_Zbb_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_o2_v05_Zbb_91_eval/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_o2_v05_Zbb_91_eval/  --njobs  3000 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_ARC/CLDConfig/ 

# ## eval pythia 91 cc
# python submit_jobs_train_ARC.py --sample Zcard --cldgeo  CLD_o2_v05 --config p8_ee_Zcc_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_o2_v05_Zcc_91_train/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_o2_v05_Zcc_91_train/  --njobs  3000 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_ARC/CLDConfig/
## eval pythia 91 cc 
# python submit_jobs_train_ARC.py --sample Zcard --cldgeo  CLD_o2_v05 --config p8_ee_Zcc_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_o2_v05_Zcc_91_eval  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_o2_v05_Zcc_91_eval/  --njobs  3000 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_ARC/CLDConfig/ 

# ## eval pythia 91 tautau 

# # ##  lepton train 
# python submit_jobs_train_ARC.py --sample Zcard --cldgeo  CLD_o2_v05 --config p8_ee_Ztautau_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_o2_v05_Ztautau_91_eval/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_o2_v05_Ztautau_91_eval/  --njobs  3000 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_ARC/CLDConfig/

#python submit_jobs_train_ARC.py --sample Zcard --cldgeo  CLD_o2_v05 --config p8_ee_Zmumu_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_o2_v05_Zmumu_91_eval/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_o2_v05_Zmumu_91_eval/  --njobs  3000 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_ARC/CLDConfig/


# python submit_jobs_train_ARC.py --sample Zcard --cldgeo  CLD_o2_v05 --config p8_ee_Zee_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_o2_v05_Zee_91_eval/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_o2_v05_Zee_91_eval/  --njobs  3000 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_ARC/CLDConfig/

 
# python submit_jobs_train_ARC.py --sample Zcard --cldgeo  CLD_o2_v05 --config p8_ee_bhabha_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_o2_v05_bhabha_91_eval/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_o2_v05_bhabha_91_eval/  --njobs  3000 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_ARC/CLDConfig/


# python submit_jobs_train_ARC.py --sample Zcard --cldgeo  CLD_o2_v05 --config p8_ee_gammagamma_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_o2_v05_gammagamma_91_eval/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_o2_v05_gammagamma_91_eval/  --njobs  3000 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_ARC/CLDConfig/

python submit_jobs_train_ARC.py --sample Zcard --cldgeo  CLD_o2_v05 --config p8_ee_Zuds_ecm91 --outdir /eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/CLD_o2_v05_Zuds_91_eval/  --condordir /eos/experiment/fcc/users/m/mgarciam/mlpf/condor/CLD_o2_v05_Zuds_91_eval/  --njobs  2500 --nev 100 --queue tomorrow --cldconfig /afs/cern.ch/work/m/mgarciam/private/CLD_Config_versions/CLDConfig_ARC/CLDConfig/ 
