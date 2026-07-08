import numpy as np
import awkward
import vector
from scipy.sparse import coo_matrix
from typing import Any, List, Optional
# Bit positions from edm4hep
BIT_CREATED_IN_SIMULATION = 30
BIT_BACKSCATTER = 29
BIT_VERTEX_NOT_ENDPOINT = 28
BIT_DECAYED_IN_TRACKER = 27
BIT_DECAYED_IN_CALO = 26
BIT_LEFT_DETECTOR = 25
BIT_STOPPED = 24

PandoraPFO_feature_order = [
    "PDG", 
    "momentum.x", 
    "momentum.y",
    "momentum.z",
    "referencePoint.x", 
    "referencePoint.y",
    "referencePoint.z",
    "energy",
    "p"
]

particle_feature_order = [
"PDG", 
"generatorStatus",
"charge",
"pt",
"eta",
"phi",
"sin_phi",
"cos_phi",
"energy",
"simulatorStatus",
"mass", 
"p", 
"momentum.x", 
"momentum.y",
"momentum.z",
"vertex.x",
"vertex.y",
"vertex.z",
"endpoint.x",
"endpoint.y",
"endpoint.z",
]

track_feature_order = [
    "elemtype", #0
    "pt", #1
    "eta",#2
    "sin_phi", #3
    "cos_phi", #4
    "p", # at vertex
    "px", # at vertex
    "py",#7
    "pz",#8
    "referencePoint.x", # store the reference at vertex
    "referencePoint.y", #10
    "referencePoint.z",#11
    "referencePoint_calo.x",
    "referencePoint_calo.y",
    "referencePoint_calo.z", #store the reference at calo
    "chi2",
    "ndf",
    "tanLambda",
    "D0",
    "omega",
    "Z0",
    "time",
    "px_calo", # at vertex
    "py_calo",#7
    "pz_calo",#8
    "collectionID", 
    "index"
]
hit_feature_order = [
    "elemtype",
    "et",
    "eta",
    "sin_phi",
    "cos_phi",
    "energy",
    "position.x",
    "position.y",
    "position.z",
    "time",
    "subdetector",
    "type",
    "collectionID", 
    "index"
]
def isProducedInCalo(vertices, BarrelRadius, NBarrelSides, EndCapZ):
    x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]

    maskProducedInCalo = np.zeros_like(x, dtype=bool)

    # Check if the vertex is outside the barrel polygon
    angles = np.linspace(0, 2*np.pi, NBarrelSides, endpoint=False)
    for a in angles:
        nx = np.cos(a)
        ny = np.sin(a)
        maskProducedInCalo |= (x * nx + y * ny > BarrelRadius)

    # Check if the vertex is outside the endcap planes
    maskProducedInCalo |= (np.abs(z) > EndCapZ)

    return maskProducedInCalo



def check_bit(value, bit):
    """Return True if bit is set in value."""
    return (value >> bit) & 1 == 1

def decode_simulator_status(sim_status):
    """Return dictionary of decoded simulator status flags."""
    return {
        "CreatedInSimulation": check_bit(sim_status, BIT_CREATED_IN_SIMULATION),
        "Backscatter": check_bit(sim_status, BIT_BACKSCATTER),
        "VertexIsNotEndpointOfParent": check_bit(sim_status, BIT_VERTEX_NOT_ENDPOINT),
        "DecayedInTracker": check_bit(sim_status, BIT_DECAYED_IN_TRACKER),
        "DecayedInCalorimeter": check_bit(sim_status, BIT_DECAYED_IN_CALO),
        "LeftDetector": check_bit(sim_status, BIT_LEFT_DETECTOR),
        "Stopped": check_bit(sim_status, BIT_STOPPED),
    }


def backscattered_and_not_decayed_in_tracker(simStatus):
    decoded_sim = decode_simulator_status(simStatus)
    DecayedInTracker = decoded_sim["DecayedInTracker"]
    Backscatter = decoded_sim["Backscatter"]
    isBackScattered_NotDecayedinTracker = Backscatter*(~DecayedInTracker)
    return isBackScattered_NotDecayedinTracker

def correct_link(MCParticles_p4, gen_arr, parents, geometry):
    index = []
    isproducedincalo = True
    for particle_idx in range(0,len(MCParticles_p4.pt)):
        particle_idx_search = particle_idx
        while isproducedincalo:
            vertex = np.array([gen_arr["vertex.x"][particle_idx_search],gen_arr["vertex.y"][particle_idx_search],gen_arr["vertex.z"][particle_idx_search]]).reshape(1,3)
            isproducedincalo = isProducedInCalo(vertex, geometry.BarrelRadius, geometry.NBarrelSides, geometry.EndCapZ) 
            isBackScattered_NotDecayedinTracker = backscattered_and_not_decayed_in_tracker(gen_arr["simulatorStatus"][particle_idx_search])
            # print("i: ", particle_idx, "producedcalo ", isproducedincalo, isBackScattered_NotDecayedinTracker)
            if isproducedincalo and (not isBackScattered_NotDecayedinTracker):
                parents_begin = gen_arr["parents_begin"][particle_idx_search]
                parents_end = gen_arr["parents_end"][particle_idx_search]
                particle_idx_search = parents[parents_begin:parents_end][0]
        index.append(particle_idx_search)
        isproducedincalo = True
    return index

class Geometry:
    def __init__(self, BarrelRadius, NBarrelSides, EndCapZ, B):
        self.BarrelRadius = BarrelRadius
        self.NBarrelSides = NBarrelSides
        self.EndCapZ = EndCapZ
        self.B = B


def get_feature_matrix(feature_dict, features):
    feats = []
    for feat in features:
        feat_arr = awkward.to_numpy(feature_dict[feat])
        feats.append(feat_arr)
    feats = np.array(feats)
    return feats.T



def sanitize(arr):
    arr[np.isnan(arr)] = 0.0
    arr[np.isinf(arr)] = 0.0


class Names_Collections:
    def __init__(self):
        self.MC_PARTICLE_COL: Optional[Any] = None
        self.PANDORA_PFO_COL: Optional[Any] = None
        self.TRACKS_COL: Optional[Any] = None
        self.CLUSTERS_COL: Optional[Any] = None
        self.CALOHIT_TO_MC_LINK_COL: Optional[Any] = None
        self.TRACK_TO_MC_LINK_COL: Optional[Any] = None
        self.CALO_HIT_COLS: Optional[Any] = None

def get_reco_properties(prop_data, iev, NAMES_COL):
    reco_arr = prop_data[NAMES_COL.PANDORA_PFO_COL][iev]
    reco_arr = {k.replace(NAMES_COL.PANDORA_PFO_COL + ".", ""): reco_arr[k] for k in reco_arr.fields}

    reco_p4 = vector.awk(
        awkward.zip({"mass": reco_arr["mass"], "x": reco_arr["momentum.x"], "y": reco_arr["momentum.y"], "z": reco_arr["momentum.z"]})
    )
    reco_arr["pt"] = reco_p4.pt
    reco_arr["eta"] = reco_p4.eta
    reco_arr["phi"] = reco_p4.phi
    reco_arr["energy"] = reco_p4.energy

    msk = reco_arr["PDG"] != 0
    reco_arr = awkward.Record({k: reco_arr[k][msk] for k in reco_arr.keys()})
    return reco_arr

def build_dummy_array(num, dtype=np.int64):
    return awkward.Array(
        awkward.contents.ListOffsetArray(
            awkward.index.Index64(np.zeros(num + 1, dtype=np.int64)),
            awkward.from_numpy(np.array([], dtype=dtype), highlevel=False),
        )
    )

def pandora_to_features(prop_data, iev, NAMES_COL):
    pandora_arr = prop_data[NAMES_COL.PANDORA_PFO_COL][iev]
    pandora_arr = {k.replace(NAMES_COL.PANDORA_PFO_COL + ".", ""): pandora_arr[k] for k in pandora_arr.fields}
    pandora_arr["p"] = np.sqrt(pandora_arr["momentum.x"]**2 + pandora_arr["momentum.x"]**2 + pandora_arr["momentum.z"]**2)
    ret = {
        "energy": pandora_arr["energy"],
        "PDG": pandora_arr["PDG"],
        "referencePoint.x": pandora_arr["referencePoint.x"],
        "referencePoint.y": pandora_arr["referencePoint.y"],
        "referencePoint.z": pandora_arr["referencePoint.z"],
        "momentum.x": pandora_arr["momentum.x"],
        "momentum.y": pandora_arr["momentum.y"],
        "momentum.z": pandora_arr["momentum.z"],
        "p": pandora_arr["p"]
    }
    return ret 


def gen_to_features(prop_data, iev, NAMES_COL, geometry):

    
    gen_arr = prop_data[NAMES_COL.MC_PARTICLE_COL][iev]

    gen_arr = {k.replace(NAMES_COL.MC_PARTICLE_COL + ".", ""): gen_arr[k] for k in gen_arr.fields}

    MCParticles_p4 = vector.awk(
        awkward.zip({"mass": gen_arr["mass"], "x": gen_arr["momentum.x"], "y": gen_arr["momentum.y"], "z": gen_arr["momentum.z"]})
    )

    parents = prop_data[f"_{NAMES_COL.MC_PARTICLE_COL}_parents/_{NAMES_COL.MC_PARTICLE_COL}_parents.index"][iev]
    gen_arr["pt"] = MCParticles_p4.pt
    gen_arr["p"] = np.sqrt(gen_arr["momentum.x"]**2 + gen_arr["momentum.y"]**2 + gen_arr["momentum.z"]**2)
    gen_arr["eta"] = MCParticles_p4.eta
    gen_arr["phi"] = MCParticles_p4.phi
    gen_arr["energy"] = MCParticles_p4.energy
    gen_arr["sin_phi"] = np.sin(gen_arr["phi"])
    gen_arr["cos_phi"] = np.cos(gen_arr["phi"])

    index = correct_link(MCParticles_p4, gen_arr, parents, geometry)
       
       

    # placeholder flag
    gen_arr["ispu"] = np.zeros_like(gen_arr["phi"])

    ret = {
        "PDG": gen_arr["PDG"],
        "generatorStatus": gen_arr["generatorStatus"],
        "charge": gen_arr["charge"],
        "pt": gen_arr["pt"],
        "p": gen_arr["p"],
        "eta": gen_arr["eta"],
        "phi": gen_arr["phi"],
        "sin_phi": gen_arr["sin_phi"],
        "cos_phi": gen_arr["cos_phi"],
        "energy": gen_arr["energy"],
        "ispu": gen_arr["ispu"],
        "mass": gen_arr["mass"], 
        "simulatorStatus": gen_arr["simulatorStatus"],
        "gp_to_track": np.zeros(len(gen_arr["PDG"]), dtype=np.float64),
        "gp_to_cluster": np.zeros(len(gen_arr["PDG"]), dtype=np.float64),
        "jet_idx": np.zeros(len(gen_arr["PDG"]), dtype=np.int64),
        "daughters_begin": gen_arr["daughters_begin"],
        "daughters_end": gen_arr["daughters_end"],
        "index_calomother": np.array(index), 
        "momentum.x"    : gen_arr["momentum.x"],
        "momentum.y"    : gen_arr["momentum.y"],
        "momentum.z"    : gen_arr["momentum.z"],
        "vertex.x"      : gen_arr["vertex.x"],
        "vertex.y"      : gen_arr["vertex.y"],
        "vertex.z"      : gen_arr["vertex.z"],
        "endpoint.x"    : gen_arr["endpoint.x"],
        "endpoint.y"    : gen_arr["endpoint.y"],
        "endpoint.z"    : gen_arr["endpoint.z"],
    }


    ret["index"] = prop_data[f"_{NAMES_COL.MC_PARTICLE_COL}_daughters/_{NAMES_COL.MC_PARTICLE_COL}_daughters.index"][iev]
    
    return ret

def get_calohit_matrix_and_genadj(hit_data, calohit_links, iev, collectionIDs, NAMES_COL, args):
    feats = ["type", "cellID", "energy", "energyError", "time", "position.x", "position.y", "position.z"]

    hit_idx_global = 0
    hit_idx_global_to_local = {}
    hit_feature_matrix = []
    for col in sorted(hit_data.keys()):
        icol = collectionIDs[col]
        hit_features = hits_to_features(hit_data[col], iev, col, feats, args, icol)
        nhits = len(hit_features["type"])
        hit_feature_matrix.append(hit_features)
        for ihit in range(nhits):
            hit_idx_global_to_local[hit_idx_global] = (icol, ihit)
            hit_idx_global += 1

    hit_idx_local_to_global = {v: k for k, v in hit_idx_global_to_local.items()}
    hit_feature_matrix = awkward.Record(
        {k: awkward.concatenate([hit_feature_matrix[i][k] for i in range(len(hit_feature_matrix))]) for k in hit_feature_matrix[0].fields}
    )

    # add all edges from genparticle to calohit
    calohit_to_gen_weight = calohit_links[f"{NAMES_COL.CALOHIT_TO_MC_LINK_COL}.weight"][iev]
    
    calohit_to_gen_calo_colid = calohit_links[f"_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_from/_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_from.collectionID"][iev]
    calohit_to_gen_gen_colid = calohit_links[f"_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_to/_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_to.collectionID"][iev]
    calohit_to_gen_calo_idx = calohit_links[f"_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_from/_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_from.index"][iev]
    calohit_to_gen_gen_idx = calohit_links[f"_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_to/_{NAMES_COL.CALOHIT_TO_MC_LINK_COL}_to.index"][iev]

    genparticle_to_hit_matrix_coo0 = []
    genparticle_to_hit_matrix_coo1 = []
    genparticle_to_hit_matrix_w = []
    for calo_colid, calo_idx, gen_colid, gen_idx, w in zip(
        calohit_to_gen_calo_colid,
        calohit_to_gen_calo_idx,
        calohit_to_gen_gen_colid,
        calohit_to_gen_gen_idx,
        calohit_to_gen_weight,
    ):
        genparticle_to_hit_matrix_coo0.append(gen_idx)
        genparticle_to_hit_matrix_coo1.append(hit_idx_local_to_global[(calo_colid, calo_idx)])
        genparticle_to_hit_matrix_w.append(w)
   
    return (
        hit_feature_matrix,
        (
            np.array(genparticle_to_hit_matrix_coo0),
            np.array(genparticle_to_hit_matrix_coo1),
            np.array(genparticle_to_hit_matrix_w),
        ),
        hit_idx_local_to_global,
    )

def track_to_features(prop_data, iev, NAMES_COL, geometry, track_col_id=-1):
    track_arr = prop_data[NAMES_COL.TRACKS_COL][iev]
    feats_from_track = ["type", "chi2", "ndf"]
    ret = {feat: track_arr[NAMES_COL.TRACKS_COL + "." + feat] for feat in feats_from_track}
    n_tr = len(ret["type"])

    # get the index of the first track state
    trackstate_idx = prop_data[NAMES_COL.TRACKS_COL][NAMES_COL.TRACKS_COL + ".trackStates_begin"][iev]
    # get the properties of the track at the first track state (at the origin)
    for k in ["tanLambda", "D0", "phi", "omega", "Z0", "time", "referencePoint.x", "referencePoint.y", "referencePoint.z"]:
        ret[k] = awkward.to_numpy(prop_data[f"_{NAMES_COL.TRACKS_COL}_trackStates"][f"_{NAMES_COL.TRACKS_COL}_trackStates." + k][iev][trackstate_idx])
    
    ret["referencePoint_calo.x"] = awkward.to_numpy(prop_data[f"_{NAMES_COL.TRACKS_COL}_trackStates"][f"_{NAMES_COL.TRACKS_COL}_trackStates.referencePoint.x"][iev][trackstate_idx+3])
    ret["referencePoint_calo.y"] = awkward.to_numpy(prop_data[f"_{NAMES_COL.TRACKS_COL}_trackStates"][f"_{NAMES_COL.TRACKS_COL}_trackStates.referencePoint.y"][iev][trackstate_idx+3])
    ret["referencePoint_calo.z"] = awkward.to_numpy(prop_data[f"_{NAMES_COL.TRACKS_COL}_trackStates"][f"_{NAMES_COL.TRACKS_COL}_trackStates.referencePoint.z"][iev][trackstate_idx+3])
    ret["phi_calo"] = awkward.to_numpy(prop_data[f"_{NAMES_COL.TRACKS_COL}_trackStates"][f"_{NAMES_COL.TRACKS_COL}_trackStates.phi"][iev][trackstate_idx+3])
    ret["tanLambda_calo"] = awkward.to_numpy(prop_data[f"_{NAMES_COL.TRACKS_COL}_trackStates"][f"_{NAMES_COL.TRACKS_COL}_trackStates.tanLambda"][iev][trackstate_idx+3])
    ret["omega_calo"] = awkward.to_numpy(prop_data[f"_{NAMES_COL.TRACKS_COL}_trackStates"][f"_{NAMES_COL.TRACKS_COL}_trackStates.omega"][iev][trackstate_idx+3])

    ret["pt"] = awkward.to_numpy(track_pt(ret["omega"], geometry))
    # from the track state at IP (location 1)
    ret["px"] = awkward.to_numpy(np.cos(ret["phi"])) * ret["pt"] 
    ret["py"] = awkward.to_numpy(np.sin(ret["phi"])) * ret["pt"]
    ret["pz"] = awkward.to_numpy(ret["tanLambda"]) * ret["pt"]

    ret["pt_calo"] = awkward.to_numpy(track_pt(ret["omega_calo"], geometry))
    ret["px_calo"] = awkward.to_numpy(np.cos(ret["phi_calo"])) * ret["pt_calo"] 
    ret["py_calo"] = awkward.to_numpy(np.sin(ret["phi_calo"])) * ret["pt_calo"]
    ret["pz_calo"] = awkward.to_numpy(ret["tanLambda_calo"]) * ret["pt_calo"]

    ret["p"] = np.sqrt(ret["px"] ** 2 + ret["py"] ** 2 + ret["pz"] ** 2)
    cos_theta = np.divide(ret["pz"], ret["p"], where=ret["p"] > 0)
    theta = np.arccos(cos_theta)
    tt = np.tan(theta / 2.0)
    eta = awkward.to_numpy(-np.log(tt, where=tt > 0))
    eta[tt <= 0] = 0.0
    ret["eta"] = eta

    ret["sin_phi"] = np.sin(ret["phi"])
    ret["cos_phi"] = np.cos(ret["phi"])

    # track is always type 1
    ret["elemtype"] = 1 * np.ones(n_tr, dtype=np.float32)

    ret["collectionID"] = np.full(n_tr, track_col_id, dtype=np.int64)
    ret["index"] = np.arange(n_tr, dtype=np.int64)

    return awkward.Record(ret)

def track_pt(omega, geometry):
    a = 2.99792e-4
    b = geometry.B  # B-field in tesla, for CLD
    return a * np.abs(b / omega)


def add_daughters_to_status1(gen_features, gp_interacted_with_detector_2 ):
    mask_status1 = gen_features["generatorStatus"] == 1
    dau_beg = gen_features["daughters_begin"]
    dau_end = gen_features["daughters_end"]
    dau_ind = gen_features["index"]
    genparticle_to_hit_additional_gp = []
    genparticle_to_hit_additional_hit = []
    genparticle_to_hit_additional_w = []
    genparticle_to_trk_additional_gp = []
    genparticle_to_trk_additional_trk = []
    genparticle_to_trk_additional_w = []
    for idx_st1 in np.where(mask_status1)[0]:
        pdg = abs(gen_features["PDG"][idx_st1])
        if pdg not in [12, 14, 16]:
            db = dau_beg[idx_st1]
            de = dau_end[idx_st1]
            daus = dau_ind[db:de]
            for dau in daus:
                if gp_interacted_with_detector_2[dau]:
                    gp_interacted_with_detector_2[idx_st1] =True
    
    return gp_interacted_with_detector_2



def track_pfo_adj(prop_data, hit_idx_local_to_global, iev, NAMES_COL):
    tracks_begin = prop_data[NAMES_COL.PANDORA_PFO_COL][f"{NAMES_COL.PANDORA_PFO_COL}.tracks_begin"][iev]
    tracks_end = prop_data[NAMES_COL.PANDORA_PFO_COL][f"{NAMES_COL.PANDORA_PFO_COL}.tracks_end"][iev]
    idx_arr_track = prop_data[f"_{NAMES_COL.PANDORA_PFO_COL}_tracks/_{NAMES_COL.PANDORA_PFO_COL}_tracks.index"][iev]
   
    # index in the array of all hits
    track_to_pfo_matrix_coo0 = []
    # index in the track array
    track_to_pfo_matrix_coo1 = []
    # weight
    track_to_pfo_matrix_w = []

    # loop over all pfos 
    for ipfo in range(len(tracks_begin)):
        track_begin = tracks_begin[ipfo]
        track_end = tracks_end[ipfo]
        idx_range = idx_arr_track[track_begin:track_end]
        for index_track, itrack in enumerate(idx_range):
            track_to_pfo_matrix_coo0.append(itrack)
            track_to_pfo_matrix_coo1.append(ipfo)
            track_to_pfo_matrix_w.append(1.0)
    return track_to_pfo_matrix_coo0, track_to_pfo_matrix_coo1, track_to_pfo_matrix_w



def hit_pfo_adj(prop_data, hit_idx_local_to_global, iev, NAMES_COL):


    clusters_begin = prop_data[f"{NAMES_COL.PANDORA_PFO_COL}"][f"{NAMES_COL.PANDORA_PFO_COL}.clusters_begin"][iev]
    clusters_end = prop_data[f"{NAMES_COL.PANDORA_PFO_COL}"][f"{NAMES_COL.PANDORA_PFO_COL}.clusters_end"][iev]
    idx_arr_cluster = prop_data[f"_{NAMES_COL.PANDORA_PFO_COL}_clusters/_{NAMES_COL.PANDORA_PFO_COL}_clusters.index"][iev]
    coll_arr = prop_data[f"_{NAMES_COL.CLUSTERS_COL}_hits/_{NAMES_COL.CLUSTERS_COL}_hits.collectionID"][iev]
    idx_arr = prop_data[f"_{NAMES_COL.CLUSTERS_COL}_hits/_{NAMES_COL.CLUSTERS_COL}_hits.index"][iev]
    hits_begin = prop_data[f"{NAMES_COL.CLUSTERS_COL}"][f"{NAMES_COL.CLUSTERS_COL}.hits_begin"][iev]
    hits_end = prop_data[f"{NAMES_COL.CLUSTERS_COL}"][f"{NAMES_COL.CLUSTERS_COL}.hits_end"][iev]
    # index in the array of all hits
    hit_to_cluster_matrix_coo0 = []
    # index in the cluster array
    hit_to_cluster_matrix_coo1 = []

    # weight
    hit_to_cluster_matrix_w = []

    # loop over all pfos 
    for ipfo in range(len(clusters_begin)):
        cluster_begin = clusters_begin[ipfo]
        cluster_end = clusters_end[ipfo]
        idx_range = idx_arr_cluster[cluster_begin:cluster_end]
        for index_cluster, icluster in enumerate(idx_range):
            # get the slice in the hit array corresponding to this cluster
            hbeg = hits_begin[icluster]
            hend = hits_end[icluster]
            idx_range = idx_arr[hbeg:hend]
            coll_range = coll_arr[hbeg:hend]

            # add edges from hit to cluster
            for icol, idx in zip(coll_range, idx_range):
                try:
                    hit_to_cluster_matrix_coo0.append(hit_idx_local_to_global[(icol, idx)])
                    hit_to_cluster_matrix_coo1.append(ipfo)
                    hit_to_cluster_matrix_w.append(1.0)
                except KeyError:
                    continue
    return hit_to_cluster_matrix_coo0, hit_to_cluster_matrix_coo1, hit_to_cluster_matrix_w



class EventData:
    def __init__(
        self,
        gen_features_target,
        # gen_features_true,
        hit_features,
        track_features,
        hit_to_gp,
        track_to_gp,
        pandora_features=None,
        pfo_to_calohit = None, 
        pfo_to_track = None, 
        gp_to_calohit_beforecalomother = None, 
        gp_to_calohit_idx = None
    ):
        self.gen_features_target = gen_features_target  # feature matrix of the genparticles
        # self.gen_features_true = gen_features_true
        self.hit_features = hit_features  # feature matrix of the calo hits
        self.track_features = track_features  # feature matrix of the tracks
        self.hit_to_gp = hit_to_gp  # array linking hit to gen MC
        self.track_to_gp = track_to_gp  # array linking track to gen MC
        self.pandora_features = pandora_features  # feature matrix of the PandoraPFOs
        self.pfo_to_calohit = pfo_to_calohit # array linking pfo to calohit
        self. pfo_to_track = pfo_to_track # array linking pfo to track
        self.gp_to_calohit_beforecalomother = gp_to_calohit_beforecalomother
        self.gp_to_calohit_idx = gp_to_calohit_idx

def filter_adj(adj, all_to_filtered):
    i0s_new = []
    i1s_new = []
    ws_new = []
    for i0, i1, w in zip(*adj):
        if i0 in all_to_filtered:
            i0_new = all_to_filtered[i0]
            i0s_new.append(i0_new)
            i1s_new.append(i1)
            ws_new.append(w)
    return np.array(i0s_new), np.array(i1s_new), np.array(ws_new)

def index_to_range(arr, mapping):
    map_func = np.vectorize(lambda x: mapping.get(x, -1))
    mapped_arr = map_func(arr)
    return mapped_arr

def genparticle_track_adj(sitrack_links, iev, NAMES_COL):
    trk_to_gen_trkidx = sitrack_links[f"_{NAMES_COL.TRACK_TO_MC_LINK_COL}_from/_{NAMES_COL.TRACK_TO_MC_LINK_COL}_from.index"][iev]
    trk_to_gen_genidx = sitrack_links[f"_{NAMES_COL.TRACK_TO_MC_LINK_COL}_to/_{NAMES_COL.TRACK_TO_MC_LINK_COL}_to.index"][iev]
    trk_to_gen_w = sitrack_links[f"{NAMES_COL.TRACK_TO_MC_LINK_COL}.weight"][iev]

    genparticle_to_track_matrix_coo0 = awkward.to_numpy(trk_to_gen_genidx)
    genparticle_to_track_matrix_coo1 = awkward.to_numpy(trk_to_gen_trkidx)
    genparticle_to_track_matrix_w = awkward.to_numpy(trk_to_gen_w)
    return genparticle_to_track_matrix_coo0, genparticle_to_track_matrix_coo1, genparticle_to_track_matrix_w


def hits_to_features(hit_data, iev, coll, feats, args, icol=-1):
    feat_arr = {f: hit_data[coll + "." + f][iev] for f in feats}

    nhits = len(feat_arr["type"]
    )
    feat_arr["collectionID"] = np.full(nhits, icol, dtype=np.int64)
    feat_arr["index"] = np.arange(nhits, dtype=np.int64)

    # set the subdetector type
    sdcoll = "subdetector"
    feat_arr[sdcoll] = np.zeros(nhits, dtype=np.int32)
    if args.ILD:
        if coll.startswith("Ecal"):
            feat_arr[sdcoll][:] = 1
        elif coll.startswith("Hcal"):
            feat_arr[sdcoll][:] = 2
        elif coll.startswith("MUON"):
            feat_arr[sdcoll][:] = 3
        else:
            feat_arr[sdcoll][:] = 4
    else:
        if coll.startswith("ECAL"):
            feat_arr[sdcoll][:] = 1
        elif coll.startswith("HCAL"):
            feat_arr[sdcoll][:] = 2
        elif coll.startswith("MUON"):
            feat_arr[sdcoll][:] = 3
        else:
            feat_arr[sdcoll][:] = 4

    # hit elemtype is always 2
    feat_arr["elemtype"] = 2 * np.ones(len(feat_arr["type"]), dtype=np.int32)

    # precompute some approximate et, eta, phi
    pos_mag = np.sqrt(feat_arr["position.x"] ** 2 + feat_arr["position.y"] ** 2 + feat_arr["position.z"] ** 2)
    px = (feat_arr["position.x"] / pos_mag) * feat_arr["energy"]
    py = (feat_arr["position.y"] / pos_mag) * feat_arr["energy"]
    pz = (feat_arr["position.z"] / pos_mag) * feat_arr["energy"]
    feat_arr["et"] = np.sqrt(px**2 + py**2)
    feat_arr["eta"] = 0.5 * np.log((feat_arr["energy"] + pz) / (feat_arr["energy"] - pz))
    feat_arr["sin_phi"] = py / feat_arr["energy"]
    feat_arr["cos_phi"] = px / feat_arr["energy"]
    if args.ILD:

        feat_arr["time_10ps"] = feat_arr["time"] + np.random.normal(0, 0.01, size=len(feat_arr["time"]))
        feat_arr["time_50ps"] = feat_arr["time"] + np.random.normal(0, 0.05, size=len(feat_arr["time"]))
        feat_arr["time_100ps"] = feat_arr["time"] + np.random.normal(0, 0.1, size=len(feat_arr["time"]))
        feat_arr["time_1000ps"] = feat_arr["time"] + np.random.normal(0, 1.0, size=len(feat_arr["time"]))
    return awkward.Record(feat_arr)



def get_genparticles_and_adjacencies( prop_data, hit_data, pandora_data, calohit_links, sitrack_links, iev, collectionIDs, NAMES_COL, geometry, args):
    gen_features = gen_to_features(prop_data, iev, NAMES_COL, geometry)
    
    hit_features, genparticle_to_hit, hit_idx_local_to_global = get_calohit_matrix_and_genadj(hit_data, calohit_links, iev, collectionIDs, NAMES_COL, args)
    track_features = track_to_features(prop_data, iev, NAMES_COL, geometry, collectionIDs[NAMES_COL.TRACKS_COL])
    genparticle_to_trk = genparticle_track_adj( sitrack_links, iev, NAMES_COL)

    n_gp = awkward.count(gen_features["PDG"])
    n_track = awkward.count(track_features["type"])
    n_hit = awkward.count(hit_features["type"])
    if args.dataset:
        pandora_features = pandora_to_features(pandora_data, iev, NAMES_COL)
        hit_to_pfo = hit_pfo_adj(pandora_data, hit_idx_local_to_global, iev, NAMES_COL)
        n_pfo = awkward.count(pandora_features["PDG"])
        pfo_to_calohit_matrix = coo_matrix((hit_to_pfo[2], (hit_to_pfo[1], hit_to_pfo[0])), shape=(n_pfo, n_hit))
        pfo_to_calohit = pfo_to_calohit_matrix.toarray().argmax(axis=0)
        pfo_to_calohit_nolink_mask  = (pfo_to_calohit_matrix.sum(axis=0).reshape(-1))==0
        pfo_to_calohit_nolink_mask = np.array(pfo_to_calohit_nolink_mask).reshape(-1)
        pfo_to_calohit[pfo_to_calohit_nolink_mask] = -1 #if no link set to -1

        pfo_to_track = track_pfo_adj(pandora_data, hit_idx_local_to_global, iev, NAMES_COL)
        pfo_to_track_matrix = coo_matrix((pfo_to_track[2], (pfo_to_track[1], pfo_to_track[0])), shape=(n_pfo, n_track))
        pfo_to_track= pfo_to_track_matrix.toarray().argmax(axis=0).reshape(-1)
        pfo_to_track_nolink_mask  = (pfo_to_track_matrix.sum(axis=0))==0
        pfo_to_track_nolink_mask = np.array(pfo_to_track_nolink_mask).reshape(-1)
        pfo_to_track[pfo_to_track_nolink_mask] = -1 #if no link set to -1
    else:
        pandora_features = None
        pfo_to_calohit = None
        pfo_to_track = None
        pfo_to_track = None
    # hit_to_cluster = hit_cluster_adj(dataset, prop_data, hit_idx_local_to_global, iev)
    # cluster_features = cluster_to_features(prop_data, hit_features, hit_to_cluster, iev)
    

    # # collect hits of st=1 daughters to the st=1 particles
    # mask_status1 = gen_features["generatorStatus"] == 1

    # if gen_features["index"] is not None:  # if there are even daughters
    #     genparticle_to_hit, genparticle_to_trk = add_daughters_to_status1(gen_features, genparticle_to_hit, genparticle_to_trk)
    # n_cluster = awkward.count(cluster_features["type"])

    if len(genparticle_to_trk[0]) > 0:
        gp_to_track_matrix = coo_matrix((genparticle_to_trk[2], (genparticle_to_trk[0], genparticle_to_trk[1])), shape=(n_gp, n_track))
        gp_to_track = gp_to_track_matrix.max(axis=1).todense()
        gp_to_track_index = gp_to_track_matrix.toarray().argmax(axis=0).reshape(-1)
        # print("gp_to_track_index", gp_to_track_index)
    else:
        gp_to_track = np.zeros((n_gp, 1))
    # one hit has contribution from different MCs
    gp_to_calohit = coo_matrix((genparticle_to_hit[2], (genparticle_to_hit[0], genparticle_to_hit[1])), shape=(n_gp, n_hit))
    # count hits per MC can't count enegy because there are more links than hits (one hit has contribution from different MCs)
    gp_to_calohit_hitcount = coo_matrix((np.ones_like(genparticle_to_hit[2]), (genparticle_to_hit[0], genparticle_to_hit[1])), shape=(n_gp, n_hit))
    gp_hitcount = gp_to_calohit_hitcount.toarray().sum(axis=1) #hit count of particles
    gp_to_calohit = gp_to_calohit.toarray().argmax(axis=0).reshape(-1) #hit to MC link 
    gp_to_recoE = coo_matrix((hit_features["energy"], (gp_to_calohit, np.arange(n_hit))), shape=(n_gp, n_hit)).toarray().sum(axis=1)
    gp_to_calohit_beforecalomother = gp_to_calohit
    gp_to_calohit = np.array(gen_features["index_calomother"])[gp_to_calohit] #assign to the MC parent that was produced before calo (index of calomother)
    gp_to_calohit_idx = gp_to_calohit
    # print(gp_to_calohit_beforecalomother[gp_to_calohit_beforecalomother!=gp_to_calohit])
    # print(gp_to_calohit_idx[gp_to_calohit_beforecalomother!=gp_to_calohit])
    # gp_to_calohit_beforecalomother = gp_to_calohit_beforecalomother!=gp_to_calohit
    gp_to_recoE = coo_matrix((hit_features["energy"], (gp_to_calohit, np.arange(n_hit))), shape=(n_gp, n_hit)).toarray().sum(axis=1)
    
    #! deprecated (bases the definition of reconstructable in cluster E)
    # calohit_to_cluster = coo_matrix((hit_to_cluster[2], (hit_to_cluster[0], hit_to_cluster[1])), shape=(n_hit, n_cluster))
    # gp_to_cluster = (gp_to_calohit * calohit_to_cluster).sum(axis=1)
    # 60% of the hits of a track must come from the genparticle
    # gp_in_tracker = np.array(gp_to_track >= 0.6)[:, 0]
    # at least 10% of the energy of the genparticle should be matched to a calorimeter cluster
    # gp_in_calo = (np.array(gp_to_cluster)[:, 0] / gen_features["energy"]) > 0.1
    # did the particle leave hits or track? (=interacted with detector) 
    # gp_interacted_with_detector = gp_in_tracker | gp_in_calo
    # mask_visible = awkward.to_numpy(mask_status1 & gp_interacted_with_detector)

    # particle has more than 10 MeV enegy in the calo
    gp_in_calo = np.array(gp_to_recoE>0.01) 
    gp_in_tracker = gp_in_calo*0 #np.array(gp_to_track >= 0.1)[:, 0]
    gp_in_tracker[gp_to_track_index] = 1
    gp_in_tracker = gp_in_tracker==1
    gp_interacted_with_detector = gp_in_tracker*gp_in_calo+gp_in_calo
    #store particles that left only track, track+calo, calo and generator status 1 (reconstructable particles)
    gp_interacted_with_detector_2 = (gp_in_tracker+gp_in_calo)
    # gp_interacted_with_detector_with_daughters = add_daughters_to_status1(gen_features,gp_interacted_with_detector_2 )
    # gp_interacted_with_detector_status1 = gp_interacted_with_detector_with_daughters*((np.abs(gen_features["generatorStatus"])==1)+(np.abs(gen_features["generatorStatus"])==2))
    gp_interacted_with_tracker_no_calo = gp_in_tracker*(~gp_in_calo)
    # mask_visible_true = awkward.to_numpy( gp_interacted_with_detector_status1)
    # gp_interacted_with_tracker_no_calo = gp_interacted_with_tracker_no_calo[mask_visible_true]
    # idx_all_masked_true = np.where(mask_visible_true)[0]
    mask_visible = awkward.to_numpy(gp_interacted_with_detector)
    idx_all_masked = np.where(mask_visible)[0]

    genpart_idx_all_to_filtered = {idx_all: idx_filtered for idx_filtered, idx_all in enumerate(idx_all_masked)}
    # print(genpart_idx_all_to_filtered)
    # genpart_idx_all_to_filtered_true = {idx_all: idx_filtered for idx_filtered, idx_all in enumerate(idx_all_masked_true)}
    if np.array(mask_visible).sum() == 0:
        print("event does not have even one 'visible' particle. will skip event")
        return None
    

    if len(np.array(mask_visible)) == 1:
        # event has only one particle (then index will be empty because no daughters)
        gen_features_rec = awkward.Record({feat: (gen_features[feat][mask_visible] if feat != "index" else None) for feat in gen_features.keys()})
    else:
        gen_features_rec = awkward.Record({feat: gen_features[feat][mask_visible] for feat in gen_features.keys()})
    # if len(np.array(mask_visible_true)) == 1:
    #     # event has only one particle (then index will be empty because no daughters)
    #     gen_features_true = awkward.Record({feat: (gen_features[feat][mask_visible_true] if feat != "index" else None) for feat in gen_features.keys()})
    # else:
    #     gen_features_true = awkward.Record({feat: gen_features[feat][mask_visible_true] for feat in gen_features.keys()})


    # get the track/cluster -> genparticle map
    # assign 0,..N indices to adjacency, -1 if genparticle not in filtered list
    hit_to_gp = index_to_range(gp_to_calohit, genpart_idx_all_to_filtered)
    if len(genparticle_to_trk[0]) > 0:
        track_to_gp = index_to_range(gp_to_track_index, genpart_idx_all_to_filtered)
        # print("track_to_gp", track_to_gp)
    else:
        track_to_gp = []

    return EventData(
        gen_features_rec,
        # gen_features_true,
        hit_features,
        track_features,
        hit_to_gp,
        track_to_gp,
        pandora_features, 
        pfo_to_calohit, 
        pfo_to_track, 
        gp_to_calohit_beforecalomother, 
        gp_to_calohit_idx
    ) 



