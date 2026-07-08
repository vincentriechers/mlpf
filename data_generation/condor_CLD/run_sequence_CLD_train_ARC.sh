#!/usr/bin/env bash
########################################
# Defaults
########################################

CLDGEO="CLD_o2_v06"
GENTRACKING=false
ARC=false
DATASET=false

########################################
# Argument parsing
########################################

while [[ $# -gt 0 ]]; do
    case "$1" in
        --homedir)        HOMEDIR="$2"; shift 2 ;;
        --guncard)        GUNCARD="$2"; shift 2 ;;
        --nev)            NEV="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --outputdir)      OUTPUTDIR="$2"; shift 2 ;;
        --dir)            DIR="$2"; shift 2 ;;
        --sample)         SAMPLE="$2"; shift 2 ;;
        --cldgeo)         CLDGEO="$2"; shift 2 ;;
        --pathcldconfig)  PATHCLDCONFIG="$2"; shift 2 ;;
        --gentracking)    GENTRACKING=true; shift ;;
        --arc)            ARC=true; shift ;;
        --dataset)        DATASET=true; shift ;;
        *) echo "Unknown option $1"; exit 1 ;;
    esac
done

########################################
# Required arguments check
########################################

: "${HOMEDIR:?Missing --homedir}"
: "${NEV:?Missing --nev}"
: "${SEED:?Missing --seed}"
: "${OUTPUTDIR:?Missing --outputdir}"
: "${DIR:?Missing --dir}"
: "${SAMPLE:?Missing --sample}"
: "${PATHCLDCONFIG:?Missing --pathcldconfig}"

echo "Using CLD geometry: $CLDGEO"

########################################
# Working directory
########################################

mkdir -p "${DIR}/${SEED}"
cd "${DIR}/${SEED}"

########################################
# Load Key4HEP
########################################

source /cvmfs/sw-nightlies.hsf.org/key4hep/setup.sh

########################################
# Event generation
########################################

if [[ "$SAMPLE" == "gun" ]]; then
    cp -r "${HOMEDIR}/data_generation/guns/gun_log_dr/"{gun.cpp,CMakeLists.txt} .
    PATH_GUN_CONFIG="${HOMEDIR}/data_generation/guns/gun_log_dr/config_files/${GUNCARD}"

    mkdir build install
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install
    make install -j 8
    cd ..
    ./build/gun "${PATH_GUN_CONFIG}"
fi

if [[ "$SAMPLE" == "Zcard" ]]; then
    xrdcp "${HOMEDIR}/data_generation/pythia/${GUNCARD}.cmd" card.cmd
    echo "Random:seed=${SEED}" >> card.cmd
    k4run "${HOMEDIR}/data_generation/pythia/pythia.py" \
        -n "$NEV" \
        --Dumper.Filename out.hepmc \
        --Pythia8.PythiaInterface.pythiacard card.cmd
    cp out.hepmc events.hepmc
fi

########################################
# Simulation
########################################

xrdcp -r "${PATHCLDCONFIG}"/* .

ddsim \
    --compactFile "$K4GEO/FCCee/CLD/compact/$CLDGEO/$CLDGEO.xml" \
    --outputFile out_sim_edm4hep.root \
    --steeringFile "${PATHCLDCONFIG}/cld_steer.py" \
    --inputFiles events.hepmc \
    --numberOfEvents "$NEV" \
    --random.seed "$SEED"


RECO_CMD=(k4run CLDReconstruction.py
          -n "$NEV"
          --inputFiles out_sim_edm4hep.root
          --outputBasename out_reco_edm4hep)

if $GENTRACKING; then
    RECO_CMD+=(--genTracking)
fi

"${RECO_CMD[@]}"

if $ARC; then
    ddsim \
        --compactFile "$K4GEO/FCCee/CLD/compact/CLD_o3_v01/CLD_o3_v01.xml" \
        --outputFile out_sim_edm4hep_ARC.root \
        --steeringFile "${PATHCLDCONFIG}/cld_arc_steer.py" \
        --inputFiles events.hepmc \
        --numberOfEvents "$NEV" \
        --random.seed "$SEED"

    ARC_RECO_CMD=(k4run CLDReconstruction.py
                  -n "$NEV"
                  --inputFiles out_sim_edm4hep_ARC.root
                  --outputBasename out_reco_edm4hep_ARC)

    if $GENTRACKING; then
        ARC_RECO_CMD+=(--genTracking)
    fi

    "${ARC_RECO_CMD[@]}"
fi



########################################
# Switch environment for preprocessing
########################################

source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc15-opt/setup.sh

########################################
# Dataset creation (dynamic flags)
########################################

if [ ! -f "out_reco_edm4hep_REC.parquet" ]; then

    cp -r "${HOMEDIR}/data_generation/preprocessing/" .

    DATA_CMD=(python -m preprocessing.dataset_creation
              --input out_reco_edm4hep_REC.edm4hep.root
              --outpath .)

    # truth automatically follows gentracking
    if $GENTRACKING; then
        DATA_CMD+=(--truth)
    fi

    if $DATASET; then
        DATA_CMD+=(--dataset)
    fi

    "${DATA_CMD[@]}"
fi

########################################
# Copy outputs
########################################

mkdir -p "${OUTPUTDIR}/05"
#mkdir -p "${OUTPUTDIR}/root_files"

python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py \
    out_reco_edm4hep_REC.parquet \
    "${OUTPUTDIR}/05/pf_tree_${SEED}.parquet"

#python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py \
#    out_reco_edm4hep_REC.edm4hep.root \
#    "${OUTPUTDIR}/root_files/pf_tree_${SEED}.edm4hep.root"



if $ARC; then

    
    ARC_DATA_CMD=(python -m preprocessing.dataset_creation
                  --input out_reco_edm4hep_ARC_REC.edm4hep.root
                  --outpath .
                  --ARC)

    if $GENTRACKING; then
        ARC_DATA_CMD+=(--truth)
    fi

    if $DATASET; then
        ARC_DATA_CMD+=(--dataset)
    fi

    "${ARC_DATA_CMD[@]}"

    mkdir -p "${OUTPUTDIR}/arc"

    python /afs/cern.ch/work/f/fccsw/public/FCCutils/eoscopy.py \
        out_reco_edm4hep_ARC_REC.parquet \
        "${OUTPUTDIR}/arc/pf_tree_${SEED}_arc.parquet"
fi

########################################
# Cleanup
########################################

rm -rf "${DIR}/${SEED}" 
