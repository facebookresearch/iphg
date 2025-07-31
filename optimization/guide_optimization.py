# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from scene import Scene
from tqdm import tqdm

from grooming_graph.optimizer.stadam import SpatioTemporalAdam


def optimize_guides(
    scene: Scene,
    curv_lr: float = 5e-2,
    length_lr: float = 1.5e-3,
    root_dir_lr: float = 5e-2,
    n_iterations: int = 3000,
    sigma_d=0.3,
):
    grooming_chain = scene.grooming_chain
    head = scene.head
    target_groom = scene.target_groom
    roots = scene.roots
    opt_strand_idx = scene.opt_strand_idx
    guide_roots = scene.guide_roots
    render = scene.render_fn
    guide_strands = scene.guide_strands

    opt = SpatioTemporalAdam(
        params=[
            {
                "params": [guide_strands.params["curv"]],
                "lr": curv_lr,
                "M": getattr(guide_strands.pp, "M", None),
            },
            {
                "params": [guide_strands.params["length"]],
                "lr": length_lr,
                "M": getattr(guide_strands.pp, "M", None),
            },
            {
                "params": [guide_strands.params["root_dirs"]],
                "lr": root_dir_lr,
                "M": getattr(guide_strands.root_pp, "M", None),
            },
        ],
        sigma_d=sigma_d,
    )
    losses = []
    l2_losses = []

    progress_bar = tqdm(range(n_iterations))

    for it in range(1, n_iterations + 1):
        opt.zero_grad()

        guides_opt = guide_strands.compute_strands(guide_roots)
        groom_opt = grooming_chain(
            roots, guides_opt, head.scalp_get_normal(guide_roots), mean=True
        )
        loss = (groom_opt[opt_strand_idx] - target_groom).square().mean()
        loss.backward()

        guide_opt_dir = torch.nn.functional.normalize(
            guides_opt[:, 1:2] - guides_opt[:, 0:1], dim=-1
        )
        opt.step(guide_opt_dir)

        l2_loss = (groom_opt[opt_strand_idx] - target_groom).square().mean()

        losses.append(loss.item())
        l2_losses.append(l2_loss.item())

        if it % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.4g}"})
            progress_bar.update(10)

    progress_bar.close()

    with torch.no_grad():
        guides_opt = guide_strands.compute_strands(guide_roots)
        groom_opt = grooming_chain(
            roots,
            guides_opt,
            head.scalp_get_normal(guide_roots),
            mean=True,
        )

        render(
            {
                "strands": groom_opt,
                "title": f"Guide Optimized Groom",
            },
            {
                "strands": guides_opt,
                "title": f"Guide Optimized Guides",
            },
            save_name="stage3_guides",
        )

    return losses, l2_losses, groom_opt, guides_opt
