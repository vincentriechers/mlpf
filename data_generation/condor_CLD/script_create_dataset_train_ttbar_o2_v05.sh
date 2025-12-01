# 1) Umgebung laden, damit python etc. da sind
source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh

REPO=/eos/user/v/vriecher/mlpf/data_generation/condor_CLD

# 3) Lokales Arbeitsverzeichnis für Condor-Submit (NICHT auf EOS)
WORKDIR=/tmp/$USER/mlpf_submit_o2_v05
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Executable lokal verfügbar machen (als symlink)
ln -sf $REPO/run_sequence_CLD_train_02_v05.sh run_sequence_CLD_train_02_v05.sh

# Logs müssen auf /tmp liegen — niemals EOS!
mkdir -p /tmp/$USER/condor_logs/

python $REPO/submit_jobs_train_vriecher.py \
  --sample Zcard \
  --cldgeo CLD_o2_v05 \
  --config p8_ee_tt_ecm365 \
  --outdir /eos/user/v/vriecher/mlpf_events_new/CLD_o2_v05/ \
  --condordir /tmp/$USER/condor_logs/ \
  --njobs 2 \
  --nev 1 \
  --queue longlunch \
  --cldconfig /eos/user/v/vriecher/CLDConfig/CLDConfig
