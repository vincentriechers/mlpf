import numpy as np
import torch
import dgl
from src.dataset.functions_data import (
    calculate_distance_to_boundary,
)
import time
from src.dataset.functions_particles import concatenate_Particles_GT, Particles_GT
from src.dataset.utils_hits import create_noise_label
from src.dataset.dataclasses import Hits
from src.dataset.uot_matching import uot_track_hit_labels

try:
    import sys as _sys
    _sys.path.insert(0, "/afs/cern.ch/work/m/mgarciam/private/DiffusionGeometry")
    import diffusion_geometry as _dg_lib
    _DIFFUSION_GEOMETRY_AVAILABLE = True
except ImportError:
    _DIFFUSION_GEOMETRY_AVAILABLE = False


def create_inputs_from_table(
    output, prediction=False, args=None
):
    number_hits = np.int32(len(output["X_track"])+len(output["X_hit"]))
    number_part = np.int32(len(output["X_gen"]))

    hits = Hits.from_data(
    output,
    number_hits,
    args,
    number_part
    )

    y_data_graph = Particles_GT()
    y_data_graph.fill( output, prediction,args)

    result = [
        y_data_graph,  
        hits
    ]
    return result
 



def compute_tangents(pos, weights=None, k=10):
    """Estimate tangent direction at each node via energy-weighted KNN + PCA,
    oriented outward from the origin.

    Args:
        pos     (Tensor): [N, 3] node positions.
        weights (Tensor): [N] per-node energy weights (None = uniform).
        k       (int):   number of nearest neighbours.

    Returns:
        Tensor: [N, 3] tangent vectors directed outward from the origin.
    """
    N = pos.shape[0]
    k_actual = min(k, N - 1)

    knn_g = dgl.knn_graph(pos, k_actual + 1)
    knn_g = dgl.remove_self_loop(knn_g)
    src, dst = knn_g.edges()                    # src=center, dst=neighbour

    rel_pos = pos[dst] - pos[src]               # [E, 3]

    # Energy weight at each neighbour (uniform if not provided)
    if weights is not None:
        w = weights[dst].view(-1, 1, 1)         # [E, 1, 1]
    else:
        w = torch.ones(rel_pos.shape[0], 1, 1, device=pos.device, dtype=pos.dtype)

    outer = (w * rel_pos.unsqueeze(-1) * rel_pos.unsqueeze(-2)).reshape(-1, 9)  # [E, 9]

    cov = torch.zeros(N, 9, device=pos.device, dtype=pos.dtype)
    cov.scatter_add_(0, src.unsqueeze(1).expand(-1, 9), outer)
    cov = cov.view(N, 3, 3)

    _, vecs = torch.linalg.eigh(cov)            # ascending eigenvalues
    tangents = vecs[:, :, -1].clone()           # [N, 3] principal direction

    # Orient outward: flip if dot product with radial direction is negative
    radial = pos / (torch.norm(pos, dim=1, keepdim=True) + 1e-8)
    flip = (tangents * radial).sum(dim=1) < 0
    tangents[flip] *= -1

    return tangents


def _build_diffusion_geometry(pos, k_neighbors, n_basis):
    """Build a DiffusionGeometry object from xyz positions (numpy)."""
    return _dg_lib.DiffusionGeometry.from_point_cloud(
        pos,
        n_function_basis=n_basis,
        knn_kernel=k_neighbors,
    )


def compute_tangents_diffusion(pos, weights=None, k=20, n_basis=12):
    """Estimate tangent directions using diffusion geometry.

    Uses the gradient of the Fiedler vector (first non-trivial Laplacian
    eigenfunction), which captures the principal direction of variation of
    the point cloud manifold.  More robust than PCA for curved / overlapping
    showers.  Falls back to ``compute_tangents`` when the library is not
    installed or the point cloud is too small.

    Args:
        pos     (Tensor): [N, 3] node positions.
        weights (Tensor): unused (kept for API compatibility with compute_tangents).
        k       (int):   number of neighbours for the diffusion kernel.
        n_basis (int):   number of Laplacian eigenfunctions to compute.

    Returns:
        Tensor: [N, 3] tangent vectors directed outward from the origin.
    """
    N = pos.shape[0]
    print("_DIFFUSION_GEOMETRY_AVAILABLE", _DIFFUSION_GEOMETRY_AVAILABLE)
    if not _DIFFUSION_GEOMETRY_AVAILABLE or N < k + 2:
        return compute_tangents(pos, weights, k=min(k, max(N - 1, 1)))

    pos_np = pos.detach().cpu().float().numpy()
    k_actual = min(k, N - 1)
    n_basis_actual = min(n_basis, N - 1)

    try:
        print("computing diff")
        dg = _build_diffusion_geometry(pos_np, k_actual, n_basis_actual)
        # Fiedler vector: first non-trivial eigenfunction (index 1)
        phi1 = dg.triple.function_basis[:, 1]          # (N,)
        f = dg.function(phi1)
        grad_np = f.grad().to_ambient()                # (N, 3)
        tangents = torch.tensor(grad_np, dtype=pos.dtype, device=pos.device)
        norm = torch.norm(tangents, dim=1, keepdim=True).clamp(min=1e-8)
        tangents = tangents / norm
    except Exception:
        print("in exception")
        return compute_tangents(pos, weights, k=k_actual)

    # Orient outward
    radial = pos / (torch.norm(pos, dim=1, keepdim=True) + 1e-8)
    flip = (tangents * radial).sum(dim=1) < 0
    tangents[flip] *= -1

    return tangents


def compute_diffusion_features(pos, k=20, n_basis=12, n_coords=4):
    """Compute diffusion-geometry-based per-hit features for shower separation.

    Returns a dict with the following tensors, all shaped [N, *]:

    ``diffusion_coords``  [N, n_coords]
        First ``n_coords`` non-trivial Laplacian eigenvectors.  Hits from the
        same shower cluster together in this space; nearby but distinct showers
        are well separated.

    ``metric_anisotropy`` [N, 3]
        Linearity, planarity and sphericity computed from the eigenvalues
        (λ₁ ≥ λ₂ ≥ λ₃) of the pointwise metric tensor Γ(x,x):
          linearity  = (λ₁ − λ₂) / (λ₁ + ε)  — high inside a linear shower
          planarity  = (λ₂ − λ₃) / (λ₁ + ε)  — high at the plane of overlap
          sphericity = λ₃         / (λ₁ + ε)  — high at isotropic / boundary regions

    ``fiedler_gradient`` [N, 3]
        Gradient of the Fiedler vector in ambient 3D space.  Its direction
        separates the two dominant clusters of hits; its magnitude is largest
        near the boundary between two nearby showers.

    Args:
        pos     (Tensor): [N, 3] node positions.
        k       (int):   number of neighbours for the diffusion kernel.
        n_basis (int):   number of Laplacian eigenfunctions to compute.
        n_coords (int):  how many diffusion coordinates to return (≤ n_basis−1).

    Returns:
        dict[str, Tensor] or None if DiffusionGeometry is not available / N too small.
    """
    N = pos.shape[0]
    if not _DIFFUSION_GEOMETRY_AVAILABLE or N < k + 2:
        return None

    pos_np = pos.detach().cpu().float().numpy()
    k_actual = min(k, N - 1)
    n_basis_actual = min(n_basis, N - 1)
    n_coords = min(n_coords, n_basis_actual - 1)

    try:
        dg = _build_diffusion_geometry(pos_np, k_actual, n_basis_actual)
        
        # --- 1. Diffusion coordinates: eigenvectors 1..n_coords ---------------
        # Skip index 0 (constant eigenfunction, eigenvalue ≈ 0)
        diff_coords_np = dg.triple.function_basis[:, 1: 1 + n_coords]   # (N, n_coords)
        diffusion_coords = torch.tensor(diff_coords_np.copy(), dtype=pos.dtype, device=pos.device)

        # --- 2. Metric anisotropy from Γ(x, x) --------------------------------
        gamma = dg.cache.gamma_coords                    # (N, 3, 3)
        gamma_t = torch.tensor(gamma, dtype=pos.dtype, device=pos.device)
        eigvals = torch.linalg.eigvalsh(gamma_t)        # (N, 3), ascending
        lam1 = eigvals[:, 2]                            # largest
        lam2 = eigvals[:, 1]
        lam3 = eigvals[:, 0]                            # smallest
        eps = 1e-8
        linearity  = (lam1 - lam2) / (lam1 + eps)
        planarity  = (lam2 - lam3) / (lam1 + eps)
        sphericity =  lam3         / (lam1 + eps)
        metric_anisotropy = torch.stack([linearity, planarity, sphericity], dim=1)  # (N, 3)

        # --- 3. Fiedler vector gradient in ambient 3D -------------------------
        phi1 = dg.triple.function_basis[:, 1]
        f = dg.function(phi1)
        fiedler_grad_np = f.grad().to_ambient()          # (N, 3)
        fiedler_gradient = torch.tensor(fiedler_grad_np, dtype=pos.dtype, device=pos.device)
        print("computed fiedler_gradient")
    except Exception:
        return None

    return {
        "diffusion_coords":   diffusion_coords,    # [N, n_coords]
        "metric_anisotropy":  metric_anisotropy,   # [N, 3]
        "fiedler_gradient":   fiedler_gradient,    # [N, 3]
    }


def create_graph(
    output,
    for_training =True, args=None
):
    prediction = not for_training
    graph_empty = False
   
    result = create_inputs_from_table(
        output,
        prediction=prediction, 
        args=args
    )

   
    if len(result) == 1:
        graph_empty = True
        return [0, 0], graph_empty
    else:
        (y_data_graph,hits) = result

        g = dgl.graph(([], []))
        g.add_nodes(hits.pos_xyz_hits.shape[0])
        g.ndata["h"] = torch.cat(
                (hits.pos_xyz_hits, hits.hit_type_one_hot, hits.e_hits, hits.p_hits), dim=1
            ).float()  
        g.ndata["p_hits"] = hits.p_hits.float() 
        g.ndata["pos_hits_xyz"] = hits.pos_xyz_hits.float()
        if getattr(args, "diffusion_features", False):
            g.ndata["tangents"] = compute_tangents_diffusion(hits.pos_xyz_hits.float())
            # _diff = compute_diffusion_features(hits.pos_xyz_hits.float())
            # if _diff is not None:
            #     g.ndata["diffusion_coords"]  = _diff["diffusion_coords"].float()   # [N, 4]
            #     g.ndata["metric_anisotropy"] = _diff["metric_anisotropy"].float()  # [N, 3]
            #     g.ndata["fiedler_gradient"]  = _diff["fiedler_gradient"].float()   # [N, 3]
        g.ndata["pos_pxpypz_at_vertex"] = hits.pos_pxpypz.float()
        g.ndata["pos_pxpypz"] = hits.pos_pxpypz  #TrackState::AtIP
        g.ndata["pos_pxpypz_at_calo"] = hits.pos_pxpypz_calo  #TrackState::AtCalorimeter
        g = calculate_distance_to_boundary(g)
        g.ndata["hit_type"] = hits.hit_type_feature.float()
        g.ndata["e_hits"] = hits.e_hits.float()  
        # g.ndata["index"] = hits.index.float() 
        # g.ndata["collectionID"] = hits.collectionID.float()  
        g.ndata["chi_squared_tracks"] = hits.chi_squared_tracks.float()
        g.ndata["particle_number"] = hits.hit_particle_link.float()+1 #(noise idx is 0 and particle MC 0 starts at 1)
        # g.ndata["particle_number_calomother"] = hits.hit_particle_link_calomother.float()+1 #(noise idx is 0 and particle MC 0 starts at 1)
        if args.ILD:
            g.ndata["time"]=hits.time_v[0].float()
            g.ndata["time_10ps"]=hits.time_v[1].float()
            g.ndata["time_50ps"]=hits.time_v[2].float()
            g.ndata["time_100ps"]=hits.time_v[3].float()
            g.ndata["time_1000ps"]=hits.time_v[4].float()
        if prediction and (args.pandora):
            # g.ndata["pandora_cluster"] = hits.pandora_features.pandora_cluster
            g.ndata["pandora_pfo"] = hits.pandora_features.pandora_pfo_link.float()
            # g.ndata["pandora_cluster_energy"] = hits.pandora_features.pandora_cluster_energy
            g.ndata["pandora_pfo_energy"] = hits.pandora_features.pfo_energy.float()
      
            g.ndata["pandora_momentum"] = hits.pandora_features.pandora_mom_components.float()
            g.ndata["pandora_reference_point"] = hits.pandora_features.pandora_ref_point.float()
            # g.ndata["daughters"] = hits.daughters
            g.ndata["pandora_pid"] = hits.pandora_features.pandora_pid.float()
        # y_data_graph.calculate_corrected_E(g, hits.connection_list)
        graph_empty = False
        if torch.unique(hits.hit_particle_link).shape[0]==1 and torch.unique(hits.hit_particle_link)[0]==-1:
            graph_empty = True
        if hits.pos_xyz_hits.shape[0] < 10:
            graph_empty = True

        if getattr(args, "uot_labels", False):
            uot_lbl = uot_track_hit_labels(g)
            g.ndata["uot_labels"] = uot_lbl

            # For each node store the calo entry point of its assigned track.
            # Neutral / unassigned nodes get (0, 0, 0).
            # Track ref_xyz is already in pos_hits_xyz for track nodes.
            track_mask = g.ndata["p_hits"].squeeze(1) > 0          # [N] bool
            track_ref_xyz = g.ndata["pos_hits_xyz"][track_mask]    # [N_trk, 3]
            uot_ref = torch.zeros(g.num_nodes(), 3)
            charged = uot_lbl > 0
            uot_ref[charged] = track_ref_xyz[uot_lbl[charged] - 1]
            g.ndata["uot_ref_xyz"] = uot_ref
    
    if ( not args.truth_tracking) and (not args.predict):
        g = make_bad_tracks_noise_tracks(g, y_data_graph)
    if args.truth_tracking and (not args.predict):
        g = remove_hits_outside_cone(g,y_data_graph, args.allegro)
        g = remove_tracks_in_noise_collection(g)
        

    return [g, y_data_graph], graph_empty

def remove_hits_outside_cone(g,y, allegro=False):
    if allegro:
        cut = 2000
    else:
        cut = 3700
    for mask_particle in range(1,len(y)):
        allHitX = g.ndata["pos_hits_xyz"][:,0][g.ndata["particle_number"]==mask_particle]
        allHitY = g.ndata["pos_hits_xyz"][:,1][g.ndata["particle_number"]==mask_particle]
        allHitZ = g.ndata["pos_hits_xyz"][:,2][g.ndata["particle_number"]==mask_particle]
        allHitTX = y.endpoint[mask_particle-1,0]
        allHitTY = y.endpoint[mask_particle-1,1]
        allHitTZ = y.endpoint[mask_particle-1,2]
        allHitE = y.E[mask_particle-1]
        allHistDist = ((allHitX-allHitTX)**2 + (allHitY-allHitTY)**2 + (allHitZ-allHitTZ)**2 )**.5
        if np.abs(y.pid[mask_particle-1])!=13:
            mask_hits = allHistDist>cut
            mask_p = g.ndata["particle_number"] == mask_particle        # full-size mask for the particle
            full_mask_hits = torch.zeros_like(mask_p, dtype=torch.bool)
            full_mask_hits[mask_p] = mask_hits
            index_modify = np.where(full_mask_hits)[0]
            number_of_hits_in_particle = g.ndata["particle_number"]==mask_particle
            if len(index_modify)<torch.sum(number_of_hits_in_particle):
                g.ndata['particle_number'][index_modify]=0
        
    return g 

def remove_tracks_in_noise_collection(g):
    # remove all hits in the noise collection 
    # tracks (for evaluation we should put back in these tracks as pandora has to deal with them)
    # hits (better to remove for eval as well)
    # currently not removing double tracks 
    mask = (g.ndata["particle_number"]==0)
    g = dgl.remove_nodes(
        g, torch.where(mask)[0]
    )

    return g 

def connect_mask():
    def func(edges):
        hit_type_src = edges.src["hit_type"]
        hit_type_dst = edges.dst["hit_type"]
        pos_src = edges.src["pos_hits_xyz"]
        pos_dst = edges.dst["pos_hits_xyz"]
        ecal_src = hit_type_src == 2
        ecal_dst = hit_type_dst == 2
        track_src = hit_type_src == 1
        track_dst = hit_type_dst == 1
        hcal_src = hit_type_src == 3
        hcal_dst = hit_type_dst == 3
        muon_src = hit_type_src == 4
        muon_dst = hit_type_dst == 4
        distance = torch.norm(pos_src-pos_dst, dim=-1)
        angle = torch.sum(pos_src*pos_dst, dim=-1)
        angle = angle/(torch.norm(pos_src, dim=-1)*torch.norm(pos_dst, dim=-1))

        ecal_mask = ecal_src*ecal_dst*(angle>0.999)*(distance<50)
        hcal_mask = hcal_src*hcal_dst*(angle>0.999)*(distance<150)
        ecal_hcal_mask = ecal_src*hcal_dst*(angle>0.999)*(distance<250)
        hcal_muon_mask = hcal_src*muon_dst*(angle>0.999)*(distance<1200)
        muon_muon = muon_src*muon_dst*(angle>0.999)*(distance<300)
        track_ecal = track_src*ecal_dst*(angle>0.999)*(distance<15)
        mask_total = ecal_mask+hcal_mask+ecal_hcal_mask+hcal_muon_mask+muon_muon+track_ecal
        connect_mask = 1*(mask_total>0)
        return {"connect": connect_mask }

    return func

def make_graph_with_edges(g):
    number_p = g.number_of_nodes()-1
    i, j = torch.tril_indices(number_p, number_p)  # , offset=-1)
    g.add_edges(i, j) # create fully connected graph
    g = dgl.to_simple(g)  # remove repated edges
    g = dgl.to_bidirected(g, copy_ndata=True)
    g = dgl.remove_self_loop(g)
    g.apply_edges(connect_mask())
    g.remove_edges(torch.where(g.edata["connect"]==0)[0].long())
    return g 



def graph_batch_func(list_graphs):
    """collator function for graph dataloader

    Args:
        list_graphs (list): list of graphs from the iterable dataset

    Returns:
        batch dgl: dgl batch of graphs
    """
    list_graphs_g = [el[0] for el in list_graphs]
    # list_y = add_batch_number(list_graphs)
    # ys = torch.cat(list_y, dim=0)
    # ys = torch.reshape(ys, [-1, list_y[0].shape[1]])
    ys = concatenate_Particles_GT(list_graphs)
    bg = dgl.batch(list_graphs_g)
    # reindex particle number
    return bg, ys

def make_bad_tracks_noise_tracks(g, y ):
    # is_chardged =scatter_add((g.ndata["hit_type"]==1).view(-1), g.ndata["particle_number"].long())[1:]
    mask_hit_type_t1 = g.ndata["hit_type"]==2
    mask_hit_type_t2 = g.ndata["hit_type"]==1
    mask_all = mask_hit_type_t1
    # the other error could come from no hits in the ECAL for a cluster
    # mean_pos_cluster = scatter_mean(g.ndata["pos_hits_xyz"][mask_all], g.ndata["particle_number"][mask_all].long().view(-1), dim=0)
    mean_pos_cluster_all = []
    mean_pos_cluster_ecal = []
    E_cluster = []
    p_tracks = []
    pos_track = g.ndata["pos_hits_xyz"][mask_hit_type_t2]
    p_tracks = g.ndata["p_hits"][mask_hit_type_t2]
    particle_track = g.ndata["particle_number"][mask_hit_type_t2]
    if len(particle_track)>0:
        for index, i in enumerate(particle_track):
            if i ==0:
                mean_pos_cluster_all.append(torch.zeros((1,3)).view(-1,3))
                mean_pos_cluster_ecal.append(torch.zeros((1,3)).view(-1,3))
                E_cluster.append(torch.zeros((1)).view(-1))
            else:
                mask_labels_i = g.ndata["particle_number"] ==i
                mean_pos_cluster = torch.mean(g.ndata["pos_hits_xyz"][mask_labels_i*mask_hit_type_t1], dim=0)
                mean_pos_cluster_all.append(mean_pos_cluster.view(-1,3))
                if len(g.ndata["radial_distance"][mask_labels_i*mask_hit_type_t1])>50:
                    index_search_ecal = 50
                else:
                    index_search_ecal = len(g.ndata["radial_distance"][mask_labels_i*mask_hit_type_t1])
    
                index_sort = torch.argsort(g.ndata["radial_distance"][mask_labels_i*mask_hit_type_t1])[0:index_search_ecal]
                distance_from_track = (torch.norm(g.ndata["pos_hits_xyz"][mask_labels_i*mask_hit_type_t1][index_sort]-pos_track[index], dim=1)/1000) < 0.1
                if torch.sum(distance_from_track)==0:
                    mean_pos_cluster_ecal.append(torch.zeros((1,3)).view(-1,3))
                else:
                    mean_cl = torch.sum(g.ndata["pos_hits_xyz"][mask_labels_i*mask_hit_type_t1][index_sort][distance_from_track]*g.ndata["e_hits"][mask_labels_i*mask_hit_type_t1][index_sort][distance_from_track], dim=0)/torch.sum(g.ndata["e_hits"][mask_labels_i*mask_hit_type_t1][index_sort][distance_from_track])
                    mean_pos_cluster_ecal.append(mean_cl.view(-1,3))
                E_cluster.append(torch.sum(g.ndata["e_hits"][mask_labels_i]).view(-1))
           
        mean_pos_cluster_all = torch.cat(mean_pos_cluster_all, dim=0)
        mean_pos_cluster_ecal = torch.cat(mean_pos_cluster_ecal, dim=0)
        E_cluster =  torch.cat(E_cluster, dim=0)
        diffs = torch.abs(E_cluster-p_tracks.view(-1))/p_tracks.view(-1)
   
        angles = torch.sum(mean_pos_cluster_ecal*pos_track,dim=1)/(torch.norm(mean_pos_cluster_ecal, dim=1)*torch.norm(pos_track, dim=1))
        angles[torch.isnan(angles)]=0
    
        distance_track_cluster = torch.norm(mean_pos_cluster_all-pos_track,dim=1)/1000
        pid = y.pid[particle_track.long()-1]
        pid[particle_track.long()==0]=0
        pid = torch.abs(pid)
        bad_tracks = ((distance_track_cluster>0.25))*(pid.view(-1)!=13)*(angles<0.9)
        bad_tracks = bad_tracks+((distance_track_cluster>0.5))*(pid.view(-1)==13)*(angles<0.9)
        index_bad_tracks = mask_hit_type_t2.nonzero().view(-1)[bad_tracks]
        

        # Remove tracks with the highest energy difference if there are multiple tracks for the same particle
        for id in torch.unique(particle_track):
            track_indices = (particle_track == id).nonzero().view(-1)
            if len(track_indices) > 1:
                # Calculate energy differences for multiple tracks of the same particle
                track_diffs = diffs[track_indices]
                max_diff_index = track_diffs.argmax()  # Get the index of the track with the max energy diff
                # Mark the track with the max diff as bad
                index_bad_tracks_double = mask_hit_type_t2.nonzero().view(-1)[track_indices[max_diff_index].view(1)]
                
                index_bad_tracks = torch.cat((index_bad_tracks, index_bad_tracks_double))

        # Set the particle_number of the bad tracks to 0
        g.ndata["particle_number"][index_bad_tracks]=0
    return g


