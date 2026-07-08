from dataclasses import dataclass
from typing import Any, List, Optional
import torch 
from src.dataset.functions_data import modify_index_link_for_gamma_e
import numpy as np 
from src.dataset.utils_hits import CachedIndexList
import awkward as ak


def _ak_to_tensor(a):
    """Fast awkward Array -> torch tensor: one bulk ak.to_numpy() conversion
    instead of torch.tensor() iterating element-by-element over an awkward Array
    (which pays awkward's per-op dispatch/error-context tax per row)."""
    return torch.from_numpy(np.ascontiguousarray(ak.to_numpy(a)))


@dataclass
class PandoraFeatures:
    # Features associated to the hits 
    pandora_cluster: Optional[Any] = None
    pandora_cluster_energy: Optional[Any] = None
    pfo_energy: Optional[Any] = None
    pandora_mom: Optional[Any] = None
    pandora_ref_point: Optional[Any] = None
    pandora_pid: Optional[Any] = None
    pandora_pfo_link: Optional[Any] = None
    pandora_mom_components: Optional[Any] = None


@dataclass
class Hits:
    pos_xyz_hits: Any
    pos_pxpypz: Any
    pos_pxpypz_calo: Any
    p_hits: Any
    e_hits: Any
    hit_particle_link: Any
    pandora_features: Any # type PandoraFeatures
    hit_type_feature: Any
    chi_squared_tracks: Any
    hit_type_one_hot: Any
    time_v:Any
    # index: Any
    # collectionID: Any
    # hit_particle_link_calomother: Any

    
    @classmethod
    def from_data(cls, output, number_hits, args, number_part):
        hit_particle_link_hits = _ak_to_tensor(output["ygen_hit"])
        if len(output["ygen_track"])>0:
            hit_particle_link_tracks= _ak_to_tensor(output["ygen_track"])
            hit_particle_link = torch.cat((hit_particle_link_hits, hit_particle_link_tracks), dim=0)
        else:
            hit_particle_link = hit_particle_link_hits
        # hit_particle_link_calomother = torch.cat((hit_particle_link_hits_calomother, hit_particle_link_tracks), dim=0)
        if args.pandora:
            pandora_features = PandoraFeatures()
            X_pandora = _ak_to_tensor(output["X_pandora"])
            pfo_link_hits = _ak_to_tensor(output["pfo_calohit"])
            if len(output["pfo_track"])>0:
                pfo_link_tracks = _ak_to_tensor(output["pfo_track"])
                pfo_link = torch.cat((pfo_link_hits, pfo_link_tracks), dim=0)
            else:
                pfo_link = pfo_link_hits
            pandora_features.pandora_pfo_link = pfo_link
            pfo_link_temp = pfo_link.clone()
            pfo_link_temp[pfo_link_temp==-1]=0
            
            pandora_features.pandora_mom = X_pandora[pfo_link_temp, 8]
            pandora_features.pandora_ref_point = X_pandora[pfo_link_temp, 4:7]
            pandora_features.pandora_mom_components = X_pandora[pfo_link_temp, 1:4]
            pandora_features.pandora_pid = X_pandora[pfo_link_temp, 0]
            pandora_features.pfo_energy = X_pandora[pfo_link_temp, 7]
            pandora_features.pandora_mom[pfo_link==-1]=0
            pandora_features.pandora_mom_components[pfo_link==-1]=0
            pandora_features.pandora_ref_point[pfo_link==-1]=0
            pandora_features.pandora_pid[pfo_link==-1]=0
            pandora_features.pfo_energy[pfo_link==-1]=0

        else:
            pandora_features = None
        # DELPHI files store positions in cm; the rest of the pipeline assumes mm
        # (e.g. calculate_distance_to_boundary / make_bad_tracks_noise_tracks constants)
        delphi = getattr(args, "delphi", False)
        X_hit = _ak_to_tensor(output["X_hit"])
        if delphi:
            X_hit[:, 6:9] = X_hit[:, 6:9] * 10.0
        if len(output["X_track"])>0:
            X_track = _ak_to_tensor(output["X_track"])
            if delphi:
                X_track[:, 12:15] = X_track[:, 12:15] * 10.0  # referencePoint_calo
        # obtain hit type
        if args.ILD:
            hit_type_feature_hit = X_hit[:,14]+1 #tyep (1,2,3,4 hits)
            # tracks carry no calo timing -> append one zero per track; empty if no tracks
            # (e.g. gammagamma events have zero tracks, so X_track is unassigned here).
            track_time = X_track[:,5]*0 if len(output["X_track"])>0 else X_hit[:0,9]
            time = torch.cat((X_hit[:,9], track_time),dim=0).view(-1,1)
            time_10ps = torch.cat((X_hit[:,10], track_time),dim=0).view(-1,1)
            time_50ps = torch.cat((X_hit[:,11], track_time),dim=0).view(-1,1)
            time_100ps = torch.cat((X_hit[:,12], track_time),dim=0).view(-1,1)
            time_1000ps =torch.cat(( X_hit[:,13], track_time),dim=0).view(-1,1)
            time_v = [time, time_10ps, time_50ps, time_100ps, time_1000ps]
        else:
            hit_type_feature_hit = X_hit[:,10]+1 #tyep (1,2,3,4 hits)
            time_v = None
        if len(output["X_track"])>0:
            hit_type_feature_track = X_track[:,0] #elemtype (1 for tracks)
            hit_type_feature = torch.cat((hit_type_feature_hit, hit_type_feature_track), dim=0).to(torch.int64)
        else:
            hit_type_feature = hit_type_feature_hit.to(torch.int64)
        # obtain the position of the hits and the energies and p
        pos_xyz_hits_hits = X_hit[:,6:9]
        # index_h = X_hit[:,-1]
        # collectionID_h = X_hit[:,-2]
        e_hits = X_hit[:,5]
        p_hits = X_hit[:,5]*0

        if len(output["X_track"])>0:
            pos_xyz_hits_tracks = X_track[:,12:15] #(referencePoint_calo.i)
            pos_xyz_hits = torch.cat((pos_xyz_hits_hits, pos_xyz_hits_tracks), dim=0)
            e_tracks =X_track[:,5]*0
            e = torch.cat((e_hits, e_tracks), dim=0).view(-1,1)
            p_tracks =X_track[:,5]
            pos_pxpypz_hits_tracks = X_track[:,6:9]
            pos_pxpypz = torch.cat((pos_xyz_hits_hits*0, pos_pxpypz_hits_tracks), dim=0)
            pos_pxpypz_hits_tracks = X_track[:,22:25]
            pos_pxpypz_calo = torch.cat((pos_xyz_hits_hits*0, pos_pxpypz_hits_tracks), dim=0)
            p = torch.cat((p_hits, p_tracks), dim=0).view(-1,1)

            # index_t = X_track[:,-1] #(index.i)
            # index_ht = torch.cat((index_h, index_t), dim=0)
            # collectionID_t = X_track[:,-2] #(collectionID.i)
            # collectionID_ht = torch.cat((collectionID_h, collectionID_t), dim=0)

        else:
            pos_xyz_hits = pos_xyz_hits_hits
            e = e_hits.view(-1,1)
            pos_pxpypz = pos_xyz_hits_hits*0
            pos_pxpypz_calo = pos_pxpypz
            p = p_hits.view(-1,1)
    
   
        if len(output["X_track"])>0:
            chi_tracks = X_track[:,15]/ X_track[:,16]
            chi_squared_tracks = torch.cat((p_hits, chi_tracks), dim=0)
        else:
            chi_squared_tracks = p_hits
        
        if args.ILD or args.allegro:
            hit_type_one_hot = torch.nn.functional.one_hot(
                hit_type_feature, num_classes=6
            )
        else:
            hit_type_one_hot = torch.nn.functional.one_hot(
                hit_type_feature, num_classes=5
            )
        return cls(
            pos_xyz_hits=pos_xyz_hits,
            pos_pxpypz=pos_pxpypz,
            pos_pxpypz_calo = pos_pxpypz_calo,
            p_hits=p,
            e_hits=e,
            hit_particle_link=hit_particle_link,
            pandora_features= pandora_features, 
            hit_type_feature=hit_type_feature,
            chi_squared_tracks=chi_squared_tracks,
            hit_type_one_hot = hit_type_one_hot, 
            time_v = time_v, 
            # collectionID = collectionID_ht, 
            # index = index_ht
            # hit_particle_link_calomother = hit_particle_link_calomother
        )


