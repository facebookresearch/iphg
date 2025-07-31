# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import faiss
from sklearn_extra.cluster import KMedoids
from scene import Scene


def get_visible_strand_idx(dataset_path: str, scene: Scene):
    train_cameras = scene.train_cameras
    groom = scene.target_groom
    renderer = scene.renderer

    visible_strand_idx_path = os.path.join(dataset_path, "visible_strands_idx.npy")
    if os.path.exists(visible_strand_idx_path):
        visible_strand_idx = np.load(visible_strand_idx_path)
    else:
        groom = groom.clone().requires_grad_(True)

        for cam in train_cameras:
            renderer.render(cam, groom).sum().backward()

        visible_strand_idx = (
            (groom.grad.abs().sum(dim=(1, 2)) > 0).nonzero().squeeze().cpu().numpy()
        )

        np.save(
            os.path.join(dataset_path, "visible_strands_idx.npy"), visible_strand_idx
        )

    return visible_strand_idx


def smooth_strands(x, lambda_=1.0, max_it=80, plot_energy=True):
    def laplacian_matrix(n):
        L = torch.zeros(n, n, device=x.device)
        for i in range(n):
            L[i, i] = 1 if i == 0 or i == n - 1 else 2
            if i > 0:
                L[i, i - 1] = -1
            if i < n - 1:
                L[i, i + 1] = -1
        return L

    def dirichlet_energy(x):
        root_to_tip = x[:, -1] - x[:, 0]
        root_to_tip_dist = torch.norm(root_to_tip, dim=-1) + 1e-12

        e = x[:, 1:] - x[:, :-1]
        return e.norm(dim=-1).sum(-1) - root_to_tip_dist

    x = x.clone()
    n = x.shape[1]
    start_idx = int(0.05 * n)
    M = torch.eye(n, device=x.device) + lambda_ * laplacian_matrix(n)
    S = list(range(start_idx + 1)) + [-1]
    MII = M[start_idx + 1 : -1, start_idx + 1 : -1]
    MIIinv = MII.inverse()
    MIS = M[start_idx + 1 : -1, S]
    xs = x[:, S]
    mxs = torch.einsum("jk,nkl->njl", MIS, xs)

    ss = []
    for _ in range(max_it):
        s = dirichlet_energy(x[:, start_idx + 1 : -1])
        ss.append(s.max().item())
        xi = x[:, start_idx + 1 : -1]
        b = xi - mxs
        x[:, start_idx + 1 : -1] = torch.einsum("jk,nkl->njl", MIIinv, b)

    if plot_energy and len(ss) > 0:
        plt.figure(figsize=(7, 5))
        plt.plot(ss)
        plt.xlabel("Iteration")
        plt.ylabel("Dirichlet Energy")
        plt.title("Dirichlet Energy of Guides over Iterations")

    return x


def initialize_guides(
    scene: Scene,
    n_guides: int,
    smooth_steps: int,
    cluster_operator_guides: bool = False,
):
    grooming_chain = scene.grooming_chain
    head = scene.head
    target_groom = scene.target_groom
    roots = scene.roots
    render = scene.render_fn

    kmeans_feat = target_groom.view(target_groom.shape[0], -1).cpu().numpy()
    kmeans = faiss.Kmeans(
        kmeans_feat.shape[1],
        n_guides,
        niter=25,
        min_points_per_centroid=1,
        gpu=True,
        verbose=False,
        nredo=10,
        seed=0,
    )
    kmeans.train(kmeans_feat)

    cluster_guides = torch.from_numpy(
        kmeans.centroids.reshape(-1, target_groom.shape[1], 3)
    ).cuda()
    cluster_guides = head.project_strands_onto_scalp(cluster_guides).detach()

    initial_guides = smooth_strands(cluster_guides, max_it=smooth_steps)

    if cluster_operator_guides:
        print("Choosing operator guides based on k-medoids")

        inst = grooming_chain.get_instance()
        n_operator_guides = int(inst.operator_guide_ratio * n_guides)

        initial_guide_lengths = torch.norm(
            initial_guides[:, 1:] - initial_guides[:, :-1], dim=-1
        ).sum(dim=-1)

        n_length_strata = 5
        op_guide_idxs = []

        # Stratify guides based on length so we choose guides from all length ranges
        for i in range(n_length_strata):
            thres_low = torch.quantile(
                initial_guide_lengths, max(i / n_length_strata, 0.0)
            )
            thres_high = torch.quantile(
                initial_guide_lengths, (i + 1) / n_length_strata
            )
            if i == n_length_strata - 1:
                thres_high += 1e-6
            op_guide_mapping = torch.arange(
                initial_guides.shape[0], device=initial_guides.device
            )[
                (initial_guide_lengths >= thres_low)
                & (initial_guide_lengths < thres_high)
            ]
            op_guides = initial_guides[op_guide_mapping]

            k = min(n_operator_guides // n_length_strata, op_guides.shape[0])
            kmediods = KMedoids(
                k,
                random_state=0,
                method="pam",
            )
            kmediods.fit(op_guides.view(op_guides.shape[0], -1).cpu().numpy())
            op_guide_idxs.append(op_guide_mapping[kmediods.medoid_indices_])
        inst.operator_guide_idxs = torch.cat(op_guide_idxs)

        render(
            {
                "strands": target_groom,
                "title": "Target Groom",
            },
            {
                "strands": grooming_chain(
                    roots,
                    initial_guides,
                    head.scalp_get_normal(initial_guides[:, 0]),
                ),
                "title": "Initial Groom",
                # "cluster_idx": inst.compute_operator_guides(roots, initial_guides)[1],
            },
            {
                "strands": initial_guides[inst.operator_guide_idxs],
                "title": "Operator Guides",
                "cluster_idx": inst.operator_guide_idxs,
            },
        )

    render(
        {
            "strands": cluster_guides,
            "title": "Clustered Strands",
        },
        {
            "strands": initial_guides,
            "title": "Initial Guides",
        },
        save_name="stage1_initialization",
    )

    return initial_guides, cluster_guides
