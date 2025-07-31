# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from sklearn.neighbors import KDTree
from scene import Scene


@torch.no_grad()
def reassign_operator_guides(
    scene: Scene, k_closest_op_guides: int = 16, head_dist_threshold: float = 0.2
):
    grooming_chain = scene.grooming_chain
    head = scene.head
    roots = scene.roots
    guide_roots = scene.guide_roots
    guide_strands = scene.guide_strands

    inst = grooming_chain.get_instance()
    guides_opt = guide_strands.compute_strands(guide_roots)
    op_guide_roots = (
        head.scalp_uv_mapping(guides_opt[inst.operator_guide_idxs, 0]).cpu().numpy()
    )
    op_guide_tree = KDTree(op_guide_roots)
    assignments = op_guide_tree.query(
        head.scalp_uv_mapping(roots).cpu().numpy(), k=k_closest_op_guides
    )[1]

    best_assignment = torch.from_numpy(assignments[:, 0]).cuda()
    needs_update = torch.full(roots.shape[:1], True, device=roots.device)
    for i in range(k_closest_op_guides):
        inst.operator_guide_assignment = torch.from_numpy(assignments[:, i]).cuda()
        groom_opt = grooming_chain(
            roots,
            guides_opt,
            head.scalp_get_normal(guides_opt[:, 0]),
        )
        head_dist_inside = (
            head.compute_distance_to_head(groom_opt.reshape(-1, 3))
            .reshape((groom_opt.shape[0], groom_opt.shape[1]))
            .clamp(max=0)
            .abs()
            .amax(dim=-1)
        )
        in_head = head_dist_inside > head_dist_threshold
        update = needs_update & ~in_head
        best_assignment[update] = inst.operator_guide_assignment[update]
        needs_update &= in_head

    inst.operator_guide_assignment = best_assignment

    print(
        f"{needs_update.sum()} / {needs_update.shape[0]} strands could not be reassigned."
    )
