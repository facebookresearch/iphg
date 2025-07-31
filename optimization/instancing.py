# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tqdm import tqdm
from scene import Scene
from optimization.loss import hair_head_penetration_reg


def optimize_instancing(
    scene: Scene,
    lr: float = 1e-2,
    n_iterations: int = 2000,
):
    grooming_chain = scene.grooming_chain
    head = scene.head
    target_groom = scene.target_groom
    roots = scene.roots
    opt_strand_idx = scene.opt_strand_idx
    cluster_guides = scene.cluster_guides
    guide_roots = scene.guide_roots
    render = scene.render_fn

    inst = grooming_chain.get_instance()
    inst.roi_weights.requires_grad = True

    opt = torch.optim.Adam(
        params=[
            {
                "params": [inst.roi_weights],
                "lr": lr,
            }
        ],
    )

    losses = []

    progress_bar = tqdm(range(n_iterations))

    for it in range(1, n_iterations + 1):
        opt.zero_grad()

        groom_opt = grooming_chain(
            roots, cluster_guides, head.scalp_get_normal(guide_roots), mean=True
        )
        loss = (groom_opt[opt_strand_idx] - target_groom).square().mean()
        # loss += hair_head_penetration_reg(head, groom_opt[opt_strand_idx])

        loss.backward()

        opt.step()

        with torch.no_grad():
            inst.clamp_parameters()

        losses.append(loss.item())

        if it % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.4g}"})
            progress_bar.update(10)

    progress_bar.close()

    inst.roi_weights.requires_grad = False

    groom_opt = grooming_chain(
        roots, cluster_guides, head.scalp_get_normal(guide_roots), mean=True
    )
    render(
        {
            "strands": groom_opt,
            "title": "Instance Optimized Groom",
        },
        save_name="stage2_instancing",
    )

    return losses, groom_opt
