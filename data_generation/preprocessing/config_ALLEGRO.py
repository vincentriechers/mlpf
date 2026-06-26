
import os
import numpy as np
import awkward
import uproot
import vector
import tqdm
from scipy.sparse import coo_matrix
from preprocessing.utils import Geometry, Names_Collections

track_coll = "SiTracks_Refitted"
mc_coll = "MCParticles"

def create_name_coll(truth_tracking):
    NAMES_COL = Names_Collections()

    # Configure all Collection names
    # NOTE: Should be in the configuration file
    NAMES_COL.MC_PARTICLE_COL = "MCParticles"
    NAMES_COL.PANDORA_PFO_COL = "PandoraPFOs"
    NAMES_COL.TRACKS_COL = "TracksFromGenParticles"
    NAMES_COL.CLUSTERS_COL = "PandoraClusters"
    NAMES_COL.CALOHIT_TO_MC_LINK_COL = "CaloHitMCParticleLinks"
    NAMES_COL.TRACK_TO_MC_LINK_COL = "TracksFromGenParticlesAssociation"
    NAMES_COL.CALO_HIT_COLS = [
    "ECalBarrelModuleThetaMergedPositioned",
    "ECalEndcapTurbinePositioned",
    "HCalBarrelReadoutPositioned",
    "HCalEndcapReadoutPositioned",
    "MuonTaggerBarrelPhiThetaPositioned",
    "MuonTaggerEndcapPhiThetaPositioned",

]
    return NAMES_COL


#! needs to be updated! 
geometry = Geometry(BarrelRadius=2150, NBarrelSides=12, EndCapZ=2307, B=2)




















