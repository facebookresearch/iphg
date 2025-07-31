# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import scipy.sparse as sp

# This is a custom version of largesteps, for a collection of 1D strands
#
# A combinatorial laplacian is built, assuming the a topology of the form:
# 1. Each strand is a sequence of n vertices
# 2. All vertices are connected within the strand (intra-strand edges)
# 3. Each i-th vertex in a strand is connected to the i-th vertex in the neighboring strands (inter-strand edges)
#
# Intra-strand edges are weighted by intra_weight, and inter-strand edges are weighted by inter_weight
# to enable different amounts of smoothing within and between strands


def strand_laplacian(
    n_points, intra_edges, inter_edges, intra_weight=1.0, inter_weight=1.0
):
    is_torch = isinstance(inter_edges, torch.Tensor)
    cat = torch.cat if is_torch else np.concatenate
    stack = torch.stack if is_torch else np.stack
    unique = lambda x, axis: (
        torch.unique(x, dim=axis) if is_torch else np.unique(x, axis=axis)
    )
    ones = lambda shape: (
        torch.ones(shape, device=inter_edges.device) if is_torch else np.ones(shape)
    )

    # Neighbor indices
    def compute_adj(edges, weight):
        ii = edges[:, [0, 1]].flatten()
        jj = edges[:, [1, 0]].flatten()
        adj = unique(stack([cat([ii, jj]), cat([jj, ii])], 0), 1)
        adj_values = weight * ones(adj.shape[1])
        return adj, adj_values

    adj_inter, adj_values_inter = compute_adj(inter_edges, inter_weight)

    if intra_edges is None:
        adj = adj_inter
        adj_values = adj_values_inter
    elif inter_edges is None:
        adj_intra, adj_values_intra = compute_adj(intra_edges, intra_weight)
        adj = adj_intra
        adj_values = adj_values_intra
    else:
        adj_intra, adj_values_intra = compute_adj(intra_edges, intra_weight)
        adj = cat((adj_intra, adj_inter), 1)
        adj_values = cat((adj_values_intra, adj_values_inter))

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = cat((adj, stack((diag_idx, diag_idx), 0)), 1)
    values = cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    if is_torch:
        return torch.sparse_coo_tensor(idx, values, (n_points, n_points)).coalesce()
    return sp.coo_matrix((values, idx), shape=(n_points, n_points)).tocsr()


def compute_strand_matrix(
    n_points, intra_edges, inter_edges, intra_weight=1.0, inter_weight=1.0
):
    L = strand_laplacian(n_points, intra_edges, inter_edges, intra_weight, inter_weight)

    idx = torch.arange(n_points, dtype=torch.long, device="cuda")
    eye = torch.sparse_coo_tensor(
        torch.stack((idx, idx), dim=0),
        torch.ones(n_points, dtype=torch.float, device="cuda"),
        (n_points, n_points),
    )
    M = torch.add(eye, L)  # M = I + L
    return M.coalesce()


def repeat_edge_structure(edges, n_inc_size, n_repeat):
    return (
        edges.repeat(n_repeat, 1)
        + torch.arange(n_repeat, device="cuda")
        .repeat_interleave(len(edges))
        .unsqueeze(1)
        .repeat((1, 2))
        * n_inc_size
    )
