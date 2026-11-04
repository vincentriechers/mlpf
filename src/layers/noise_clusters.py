"""Noise-only clusters for stage 2 trained on truth clusters (EC_NOISE_CLUSTERS, off by default).

With --use-gt-clusters the energy and PID heads only ever see truth particles; cells without a particle (label 0: read-out
noise, or recorded noise overlaid on the simulation) never form a cluster, so at inference every trackless cluster becomes a
photon or a neutral hadron. noise_pseudo_clusters groups those cells into connected clusters that utils_training appends as
pseudo-particles (true energy 0, PID NOISE_PID); EnergyCorrection maps them to a neutral class 5 ("noise").
"""
import numpy as np
import torch

NOISE_PID = 0  # pseudo-particle code of a noise cluster (true energy 0); EnergyCorrection maps it to class 5


def noise_pseudo_clusters(graph, labels, link_cm, emin=0.0):
    """Calorimeter cells without a particle (label 0) linked within link_cm -> one boolean node mask per connected
    group above emin GeV. Lets stage 2, trained on truth clusters, also see noise-only clusters (EC_NOISE_CLUSTERS)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    calo = (labels == 0) & (graph.ndata["hit_type"].view(-1) != 1)
    idx = torch.where(calo)[0]
    if idx.numel() == 0:
        return []
    xyz = graph.ndata["pos_hits_xyz"][idx].detach().cpu().numpy()
    e = graph.ndata["e_hits"].view(-1)[idx].detach().cpu().numpy()
    pr = cKDTree(xyz).query_pairs(link_cm, output_type="ndarray")
    n = len(idx)
    A = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)) if len(pr) else coo_matrix((n, n))
    nc, lab = connected_components(A, directed=False)
    masks = []
    for c in range(nc):
        m = lab == c
        if e[m].sum() > emin:
            full = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
            full[idx[torch.from_numpy(np.flatnonzero(m)).to(idx.device)]] = True
            masks.append(full)
    return masks
